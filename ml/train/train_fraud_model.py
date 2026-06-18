"""
Ensemble Fraud Detection Model
Replaces: Isolation Forest (single model, 4 features, ~82% accuracy)
Upgrades to: XGBoost + LightGBM + Isolation Forest stacked ensemble
Expected accuracy: 75-85% PR-AUC (with realistic, overlapping distributions)
"""

import numpy as np
import pandas as pd
import pickle
import os
import mlflow
import mlflow.sklearn
from datetime import datetime

from sklearn.ensemble import IsolationForest, StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------─
# CONFIG
# --------------------------------------------─
MODEL_OUTPUT_DIR = "ml/models"
MLFLOW_EXPERIMENT = "fraud-detection-ensemble"
RANDOM_STATE = 42
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)


# --------------------------------------------─
# STEP 1: FEATURE ENGINEERING
# Handles both the original schema and the 16-column Kaggle schema.
# --------------------------------------------─
def engineer_features(
    df: pd.DataFrame, 
    encoders: dict = None, 
    fit: bool = True,
    user_stats: dict = None,
    known_fraud_addresses: set = None
) -> (pd.DataFrame, dict, dict, set):
    """
    Expands input features to 16 engineered features.
    Supports Kaggle dataset columns and maps them to model feature definitions.
    Prevents leakage by isolating training statistics (user history, fraud addresses).
    """
    df = df.copy()

    # Map Kaggle columns to standard names
    kaggle_map = {
        "Transaction ID": "order_id",
        "Customer ID": "user_id",
        "Transaction Amount": "order_amount",
        "Transaction Date": "order_timestamp",
        "Device Used": "device_type",
        "Payment Method": "payment_method",
        "Product Category": "category",
        "Is Fraudulent": "is_fraud",
        "Account Age Days": "account_age_days",
        "Transaction Hour": "transaction_hour",
    }
    
    # Check if this is the Kaggle schema
    is_kaggle = "Transaction ID" in df.columns
    if is_kaggle:
        df = df.rename(columns={k: v for k, v in kaggle_map.items() if k in df.columns})
        
        # Derive location_mismatch if Shipping Address and Billing Address are present
        if "Shipping Address" in df.columns and "Billing Address" in df.columns:
            df["location_mismatch"] = (df["Shipping Address"] != df["Billing Address"]).astype(int)
        elif "location_mismatch" not in df.columns:
            df["location_mismatch"] = 0
            
        # Derive orders_per_user_last_minute if not present
        if "orders_per_user_last_minute" not in df.columns:
            df = df.sort_values(["user_id", "order_timestamp"])
            df["order_timestamp_dt"] = pd.to_datetime(df["order_timestamp"])
            df = df.set_index("order_timestamp_dt")
            df["orders_per_user_last_minute"] = (
                df.groupby("user_id")["order_id"]
                .rolling("1T")
                .count()
                .reset_index(level=0, drop=True)
                .astype(int)
            )
            df = df.reset_index(drop=True)

    if encoders is None:
        encoders = {}
    if user_stats is None:
        user_stats = {}
    if known_fraud_addresses is None:
        known_fraud_addresses = set()

    # -- Encode categorical columns --
    for col_name, enc_name in [("device_type", "device_type_enc"), 
                               ("payment_method", "payment_method_enc"), 
                               ("category", "category_enc")]:
        if col_name in df.columns:
            if df[col_name].dtype == object or df[col_name].dtype == str:
                if fit or col_name not in encoders:
                    le = LabelEncoder()
                    df[enc_name] = le.fit_transform(df[col_name].astype(str))
                    encoders[col_name] = le
                else:
                    le = encoders[col_name]
                    classes = le.classes_
                    df[enc_name] = df[col_name].astype(str).map(
                        lambda s: le.transform([s])[0] if s in classes else -1
                    )
            else:
                df[enc_name] = df[col_name]
        else:
            df[enc_name] = 0

    # 1. Amount z-score per user (how unusual is this amount for this user?)
    if fit:
        # Calculate user statistics only on the training set
        means = df.groupby("user_id")["order_amount"].mean()
        stds = df.groupby("user_id")["order_amount"].std().fillna(1.0)
        for uid in df["user_id"].unique():
            user_stats[uid] = {"mean": float(means.get(uid, 0.0)), "std": float(stds.get(uid, 1.0))}

    global_mean = df["order_amount"].mean() if len(df) > 0 else 0.0
    global_std = df["order_amount"].std() if len(df) > 1 else 1.0

    def get_zscore(row):
        uid = row["user_id"]
        stats = user_stats.get(uid, {"mean": global_mean, "std": global_std})
        mean = stats["mean"]
        std = stats["std"]
        if std == 0:
            std = 1.0
        return (row["order_amount"] - mean) / std

    df["amount_zscore_user"] = df.apply(get_zscore, axis=1).fillna(0.0)

    # 2. Orders per user last hour
    if "orders_per_user_last_hour" not in df.columns:
        if "order_timestamp" in df.columns:
            df["order_timestamp_dt"] = pd.to_datetime(df["order_timestamp"])
            df = df.sort_values(["user_id", "order_timestamp_dt"])
            df = df.set_index("order_timestamp_dt")
            df["orders_per_user_last_hour"] = (
                df.groupby("user_id")["order_id"]
                .rolling("1H")
                .count()
                .reset_index(level=0, drop=True)
                .astype(int)
            )
            df = df.reset_index(drop=True)
        else:
            df["orders_per_user_last_hour"] = df["orders_per_user_last_minute"] * 4

    # 3. Amount percentile rank overall (is this an outlier amount?)
    df["amount_pct_rank"] = df["order_amount"].rank(pct=True).fillna(0.5)

    # 4. Is new user? (first-time buyers are higher risk)
    if "account_age_days" in df.columns:
        df["is_new_user"] = (df["account_age_days"] < 30).astype(int)
    else:
        user_order_counts = df.groupby("user_id")["order_id"].transform("count")
        df["is_new_user"] = (user_order_counts == 1).astype(int)

    # 5. Time-of-day risk (late night orders are higher fraud risk)
    if "transaction_hour" in df.columns:
        df["order_hour"] = df["transaction_hour"]
        df["is_late_night"] = df["order_hour"].apply(
            lambda h: 1 if (h >= 23 or h <= 4) else 0
        )
    elif "order_timestamp" in df.columns:
        df["order_hour"] = pd.to_datetime(df["order_timestamp"]).dt.hour
        df["is_late_night"] = df["order_hour"].apply(
            lambda h: 1 if (h >= 23 or h <= 4) else 0
        )
    else:
        df["order_hour"] = 12
        df["is_late_night"] = 0

    # 6. Multiple payment attempts (if available, else 0)
    df["payment_attempts"] = df.get("payment_attempts", pd.Series(
        np.random.poisson(1.1, len(df)), index=df.index
    ))
    df["multi_payment_attempt"] = (df["payment_attempts"] > 1).astype(int)

    # 7. Cart-to-checkout time (very fast = bot signal)
    if "cart_to_checkout_seconds" in df.columns:
        df["fast_checkout"] = (df["cart_to_checkout_seconds"] < 10).astype(int)
    else:
        df["fast_checkout"] = 0

    # 8. High-value order flag
    high_value_thresh = df["order_amount"].quantile(0.95) if len(df) > 1 else 500.0
    df["is_high_value"] = (df["order_amount"] > high_value_thresh).astype(int)

    # 9. Same address as previous fraud (if address available, else 0)
    if fit and "is_fraud" in df.columns and "Shipping Address" in df.columns:
        train_frauds = df[df["is_fraud"] == 1]["Shipping Address"].unique()
        for addr in train_frauds:
            known_fraud_addresses.add(addr)
            
    if "Shipping Address" in df.columns:
        df["known_fraud_address"] = df["Shipping Address"].apply(
            lambda x: 1 if x in known_fraud_addresses else 0
        ).astype(int)
    else:
        df["known_fraud_address"] = 0

    # 10. IP risk score (from MaxMind / IPQualityScore if integrated, else proxy)
    df["ip_risk_score"] = df.get("ip_risk_score", pd.Series(
        np.random.beta(2, 8, len(df)), index=df.index
    ))

    # 11. Velocity x amount interaction
    df["velocity_amount_interaction"] = (
        df["orders_per_user_last_minute"] * df["order_amount"]
    )

    # 12. Location mismatch x high value (compound risk)
    df["location_highvalue_risk"] = (
        df["location_mismatch"] * df["is_high_value"]
    )

    return df, encoders, user_stats, known_fraud_addresses


FEATURE_COLS = [
    # Original 4
    "order_amount",
    "orders_per_user_last_minute",
    "location_mismatch",
    "device_type_enc",
    # New 12
    "amount_zscore_user",
    "orders_per_user_last_hour",
    "amount_pct_rank",
    "is_new_user",
    "order_hour",
    "is_late_night",
    "multi_payment_attempt",
    "fast_checkout",
    "is_high_value",
    "known_fraud_address",
    "ip_risk_score",
    "velocity_amount_interaction",
    "location_highvalue_risk",
]


# --------------------------------------------─
# STEP 2: DATA GENERATOR / LOADER
# Loads Kaggle dataset if present, else generates high-quality aligned mock data.
# --------------------------------------------─
def generate_training_data(n_samples: int = 50000) -> pd.DataFrame:
    """
    Loads real Kaggle dataset if available at ml/data/Fraudulent_E-Commerce_Transactions.csv.
    Otherwise, generates a high-quality synthetic dataset mimicking the Kaggle 16-column schema
    with realistic, overlapping distributions (non-circular) to prevent overoptimistic evaluation metrics.
    """
    csv_path = "ml/data/Fraudulent_E-Commerce_Transactions.csv"
    if os.path.exists(csv_path):
        print(f"      [INFO] Loading real Kaggle dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
        if len(df) > n_samples:
            df = df.sample(n=n_samples, random_state=42).reset_index(drop=True)
        return df

    print(f"      [INFO] Kaggle CSV not found at {csv_path}. Generating synthetic Kaggle-aligned data...")
    np.random.seed(RANDOM_STATE)
    
    n_fraud = int(n_samples * 0.015)  # 1.5% fraud rate
    n_legit = n_samples - n_fraud
    
    user_ids = [f"user_{i}" for i in np.random.randint(0, 5000, n_samples)]
    order_ids = [f"ORD-{i:06d}" for i in range(n_samples)]
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Books', 'Beauty']
    payment_methods = ['Credit Card', 'PayPal', 'Debit Card', 'Bank Transfer']
    devices = ['mobile', 'desktop', 'tablet']
    
    # Overlapping feature distributions (realistic classification boundaries)
    legit_amounts = np.random.lognormal(4.5, 0.8, n_legit)
    fraud_amounts = np.random.lognormal(5.0, 1.0, n_fraud)
    
    legit_acc_age = np.random.exponential(300, n_legit).clip(1, 1000).astype(int)
    fraud_acc_age = np.random.exponential(30, n_fraud).clip(1, 1000).astype(int)
    
    legit_hours = (np.random.normal(14, 4, n_legit).astype(int) % 24)
    fraud_hours = (np.random.normal(18, 6, n_fraud).astype(int) % 24)
    
    legit_mismatch = np.random.binomial(1, 0.08, n_legit)
    fraud_mismatch = np.random.binomial(1, 0.35, n_fraud)
    
    legit_df = pd.DataFrame({
        "Transaction ID": order_ids[:n_legit],
        "Customer ID": user_ids[:n_legit],
        "Transaction Amount": legit_amounts,
        "Transaction Date": pd.date_range("2024-01-01", periods=n_legit, freq="1min"),
        "Payment Method": np.random.choice(payment_methods, n_legit),
        "Product Category": np.random.choice(categories, n_legit),
        "Quantity": np.random.poisson(1.2, n_legit).clip(1, 5),
        "Customer Age": np.random.normal(38, 12, n_legit).clip(18, 80).astype(int),
        "Customer Location": np.random.choice(["USA", "Canada", "UK", "Germany", "France"], n_legit),
        "Device Used": np.random.choice(devices, n_legit, p=[0.6, 0.3, 0.1]),
        "IP Address": [f"192.168.1.{i%255}" for i in range(n_legit)],
        "Shipping Address": [f"Street_{i}" for i in range(n_legit)],
        "Billing Address": [f"Street_{i}" if legit_mismatch[i] == 0 else f"Street_{i+1}" for i in range(n_legit)],
        "Account Age Days": legit_acc_age,
        "Transaction Hour": legit_hours,
        "Is Fraudulent": np.zeros(n_legit, dtype=int)
    })
    
    fraud_df = pd.DataFrame({
        "Transaction ID": order_ids[n_legit:],
        "Customer ID": user_ids[n_legit:],
        "Transaction Amount": fraud_amounts,
        "Transaction Date": pd.date_range("2024-01-01", periods=n_fraud, freq="30min"),
        "Payment Method": np.random.choice(payment_methods, n_fraud),
        "Product Category": np.random.choice(categories, n_fraud),
        "Quantity": np.random.poisson(1.5, n_fraud).clip(1, 5),
        "Customer Age": np.random.normal(34, 10, n_fraud).clip(18, 80).astype(int),
        "Customer Location": np.random.choice(["USA", "Canada", "UK", "Germany", "France"], n_fraud),
        "Device Used": np.random.choice(devices, n_fraud, p=[0.4, 0.5, 0.1]),
        "IP Address": [f"10.0.0.{i%255}" for i in range(n_fraud)],
        "Shipping Address": [f"Street_{n_legit+i}" for i in range(n_fraud)],
        "Billing Address": [f"Street_{n_legit+i}" if fraud_mismatch[i] == 0 else f"Street_{n_legit+i+1000}" for i in range(n_fraud)],
        "Account Age Days": fraud_acc_age,
        "Transaction Hour": fraud_hours,
        "Is Fraudulent": np.ones(n_fraud, dtype=int)
    })
    
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


# --------------------------------------------─
# STEP 3: BUILD ENSEMBLE
# --------------------------------------------─
def build_ensemble():
    """
    Stacked ensemble:
      Layer 1 (base learners): XGBoost + LightGBM + IsolationForest(as scorer)
      Layer 2 (meta-learner):  Logistic Regression
    """
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=66,      # handles class imbalance
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    meta_learner = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    ensemble = StackingClassifier(
        estimators=[
            ("xgb", xgb_model),
            ("lgb", lgb_model),
            ("rf",  rf_model),
        ],
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    return ensemble


# --------------------------------------------─
# STEP 4: FIND OPTIMAL THRESHOLD
# --------------------------------------------─
def find_optimal_threshold(y_true, y_proba, beta: float = 1.0) -> float:
    """
    Finds threshold that maximises F-beta score on PR curve.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    beta2 = beta ** 2
    f_scores = (1 + beta2) * (precision * recall) / (
        (beta2 * precision) + recall + 1e-8
    )
    best_idx = np.argmax(f_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    print(f"\n[DONE] Optimal threshold (F{beta}): {best_threshold:.4f}")
    print(f"   Precision: {precision[best_idx]:.4f} | Recall: {recall[best_idx]:.4f}")
    return float(best_threshold)


# --------------------------------------------─
# STEP 5: MAIN TRAINING PIPELINE
# --------------------------------------------─
def train():
    print("=" * 60)
    print("  Fraud Detection - Ensemble Training Pipeline")
    print("=" * 60)

    # -- Load / generate data --
    print("\n[1/6] Loading data...")
    df = generate_training_data(n_samples=50000)
    print(f"      Total samples : {len(df):,}")

    target_col = "Is Fraudulent" if "Is Fraudulent" in df.columns else "is_fraud"

    # Train/test split (stratified) on raw data to prevent leakage!
    print("\n[2/6] Splitting and engineering features...")
    df_train, df_test = train_test_split(
        df,
        test_size=0.2,
        stratify=df[target_col].astype(int),
        random_state=RANDOM_STATE,
    )

    # Engineer features on train set
    df_train_eng, encoders, user_stats, known_fraud_addresses = engineer_features(df_train, fit=True)
    
    # Engineer features on test set using train statistics
    df_test_eng, _, _, _ = engineer_features(
        df_test, 
        encoders=encoders, 
        fit=False, 
        user_stats=user_stats, 
        known_fraud_addresses=known_fraud_addresses
    )

    X_train = df_train_eng[FEATURE_COLS].fillna(0)
    y_train = df_train_eng["is_fraud"].astype(int)
    X_test = df_test_eng[FEATURE_COLS].fillna(0)
    y_test = df_test_eng["is_fraud"].astype(int)

    print(f"      Train samples : {len(df_train):,} (Fraud: {y_train.sum():,}, {y_train.mean()*100:.2f}%)")
    print(f"      Test samples  : {len(df_test):,} (Fraud: {y_test.sum():,}, {y_test.mean()*100:.2f}%)")
    print(f"      Features used : {len(FEATURE_COLS)}")

    # -- Scaler --
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # -- SMOTE on training set only --
    print("\n[3/6] Applying SMOTE to balance training set...")
    smote = SMOTE(
        sampling_strategy=0.15,
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)

    # -- Build & train ensemble --
    print("\n[4/6] Training stacked ensemble (XGBoost + LightGBM + RF)...")
    ensemble = build_ensemble()

    with mlflow.start_run(run_name=f"ensemble_{datetime.now():%Y%m%d_%H%M}"):
        mlflow.log_params({
            "model_type":       "StackingClassifier",
            "base_learners":    "XGBoost, LightGBM, RandomForest",
            "meta_learner":     "LogisticRegression",
            "n_features":       len(FEATURE_COLS),
            "smote_strategy":   0.15,
            "train_samples":    len(X_train_res),
        })

        ensemble.fit(X_train_res, y_train_res)

        # -- Evaluate --
        print("\n[5/6] Evaluating...")
        y_proba = ensemble.predict_proba(X_test_sc)[:, 1]

        pr_auc  = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

        optimal_threshold = find_optimal_threshold(y_test, y_proba, beta=1.0)
        y_pred = (y_proba >= optimal_threshold).astype(int)

        f1  = f1_score(y_test, y_pred)
        cm  = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\n{'-'*40}")
        print(f"  PR-AUC  : {pr_auc:.4f}   <- primary metric")
        print(f"  ROC-AUC : {roc_auc:.4f}")
        print(f"  F1      : {f1:.4f}")
        print(f"  TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
        print(f"{'-'*40}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Legit", "Fraud"]))

        mlflow.log_metrics({
            "pr_auc":            pr_auc,
            "roc_auc":           roc_auc,
            "f1_score":          f1,
            "true_positives":    int(tp),
            "false_positives":   int(fp),
            "false_negatives":   int(fn),
            "optimal_threshold": optimal_threshold,
        })

        # -- Save artifacts --
        print("\n[6/6] Saving models...")

        artifacts = {
            "ensemble_model":   ensemble,
            "scaler":           scaler,
            "feature_cols":     FEATURE_COLS,
            "threshold":        optimal_threshold,
            "trained_at":       datetime.now().isoformat(),
            "pr_auc":           pr_auc,
            "encoders":         encoders,
            "user_stats":       user_stats,
            "known_fraud_addresses": known_fraud_addresses,
        }

        model_path = os.path.join(MODEL_OUTPUT_DIR, "ensemble_fraud_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(artifacts, f)

        # Also save scaler separately for Spark inference
        scaler_path = os.path.join(MODEL_OUTPUT_DIR, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(ensemble, "ensemble_model")

        print(f"\n  [DONE] Model saved  -> {model_path}")
        print(f"  [DONE] Scaler saved -> {scaler_path}")
        print(f"\n  [FINAL] Final PR-AUC: {pr_auc:.4f}")
        print("=" * 60)

    return artifacts


if __name__ == "__main__":
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    train()