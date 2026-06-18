"""
MLOps Drift Detection (Layer 5)
Uses 'Evidently' to monitor data and prediction drift.

This script compares training data (reference) vs production data (current)
to detect if the fraud patterns have changed.
"""

import pandas as pd
import numpy as np
import os
import json
from evidently import Report
from evidently.presets import DataDriftPreset, DataSummaryPreset

# CONFIG
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
REF_DATA_PATH = "ml/models/reference_data.csv"
TRAIN_STATS_PATH = "ml/models/training_distribution_stats.json"
PROD_PRED_PATH = "ml/models/production_predictions.csv"

def detect_drift():
    print("--- MLOps: Fraud Detection Drift Monitor ---")
    
    # 1. LOAD TRAINING REFERENCE DATA & STATS
    if os.path.exists(REF_DATA_PATH):
        print(f"      [INFO] Loading training reference dataset from {REF_DATA_PATH}...")
        reference_data = pd.read_csv(REF_DATA_PATH)
    else:
        print(f"      [WARNING] Reference dataset not found at {REF_DATA_PATH}. Generating baseline mock...")
        reference_data = pd.DataFrame({
            "order_amount": np.random.lognormal(4.5, 0.8, 1000),
            "orders_per_user_last_minute": np.random.poisson(0.3, 1000),
            "location_mismatch": np.random.binomial(1, 0.05, 1000),
            "is_fraud": np.random.binomial(1, 0.02, 1000)
        })

    # 2. LOAD PRODUCTION DATA
    if os.path.exists(PROD_PRED_PATH):
        print(f"      [INFO] Loading production predictions from {PROD_PRED_PATH}...")
        current_data = pd.read_csv(PROD_PRED_PATH)
        
        # Ensure current_data has enough samples for statistical significance
        if len(current_data) < 20:
            print(f"      [INFO] Production log file too small ({len(current_data)} rows). Simulating drift batch on top of production...")
            # Simulate a drifted current dataset based on reference
            drifted = reference_data.copy()
            drifted["order_amount"] = drifted["order_amount"] * np.random.uniform(1.2, 1.8, len(drifted))
            drifted["orders_per_user_last_minute"] = drifted["orders_per_user_last_minute"] + np.random.poisson(1.5, len(drifted))
            drifted["is_fraud"] = np.random.binomial(1, 0.09, len(drifted))
            current_data = pd.concat([current_data, drifted]).reset_index(drop=True)
    else:
        print(f"      [INFO] Production logs not found at {PROD_PRED_PATH}. Simulating drifted production batch for demo/testing...")
        current_data = reference_data.copy()
        current_data["order_amount"] = current_data["order_amount"] * np.random.uniform(1.3, 2.0, len(current_data))
        current_data["orders_per_user_last_minute"] = current_data["orders_per_user_last_minute"] + np.random.poisson(2.0, len(current_data))
        current_data["is_fraud"] = np.random.binomial(1, 0.12, len(current_data))

    # Align columns between reference and current
    common_cols = list(set(reference_data.columns) & set(current_data.columns))
    reference_data = reference_data[common_cols]
    current_data = current_data[common_cols]

    print("Analyzing drift between training and production data...")

    # 3. STATISTICAL COMPARISON PRINT (Interview Value)
    if os.path.exists(TRAIN_STATS_PATH):
        try:
            with open(TRAIN_STATS_PATH, "r") as f:
                train_stats = json.load(f)
            
            print("\n" + "="*50)
            print("  DRIFT STATISTICS COMPARISON (TRAIN VS CURRENT)")
            print("="*50)
            for col in sorted(common_cols):
                if col in train_stats:
                    ref_mean = train_stats[col].get("mean", 0.0)
                    ref_std = train_stats[col].get("std", 0.0)
                    curr_mean = current_data[col].mean()
                    curr_std = current_data[col].std()
                    pct_diff = abs(curr_mean - ref_mean) / (ref_mean + 1e-8) * 100
                    print(f"Column: {col:<30}")
                    print(f"   Train   -> Mean: {ref_mean:8.4f} | Std: {ref_std:8.4f}")
                    print(f"   Current -> Mean: {curr_mean:8.4f} | Std: {curr_std:8.4f}")
                    print(f"   Difference in Mean: {pct_diff:.2f}%")
            print("="*50 + "\n")
        except Exception as e:
            print(f"      [WARNING] Could not format summary statistics: {e}")

    # 4. CREATE EVIDENTLY REPORT
    drift_report = Report(metrics=[
        DataDriftPreset(),
        DataSummaryPreset()
    ])

    eval_report = drift_report.run(reference_data=reference_data, current_data=current_data)

    # 5. SAVE REPORT
    report_path = os.path.join(REPORTS_DIR, "drift_report.html")
    eval_report.save_html(report_path)
    
    print(f"\n[DONE] Drift Analysis complete.")
    print(f"Report saved to: {os.path.abspath(report_path)}")
    print("In production, this would trigger a model retraining pipeline if drift > threshold.")

if __name__ == "__main__":
    detect_drift()
