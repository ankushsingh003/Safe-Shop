"""
Upgraded FastAPI Inference Server
Replaces: ml/serve/app.py (Isolation Forest)
Upgrades:  Stacked Ensemble with 16 features + SHAP explanations

Drop this file into: ml/serve/app.py
"""

import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import shap
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fraud Detection API — Ensemble Model",
    description="XGBoost + LightGBM + RF stacked ensemble with SHAP explanations",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────
MODEL_PATH = "../models/ensemble_fraud_model.pkl"

model_artifacts: Dict[str, Any] = {}

@app.on_event("startup")
async def load_model():
    global model_artifacts
    try:
        with open(MODEL_PATH, "rb") as f:
            model_artifacts = pickle.load(f)
        logger.info(f"✅ Ensemble model loaded | "
                    f"PR-AUC: {model_artifacts.get('pr_auc', 'N/A'):.4f} | "
                    f"Features: {len(model_artifacts['feature_cols'])}")
    except FileNotFoundError:
        logger.error(f"❌ Model not found at {MODEL_PATH}. Run train_ensemble_fraud_model.py first.")


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────
class OrderRequest(BaseModel):
    order_id: str
    user_id: str
    order_amount: float = Field(..., gt=0)
    orders_per_user_last_minute: int = Field(default=0, ge=0)
    location_mismatch: int = Field(default=0, ge=0, le=1)
    device_type: str = Field(default="mobile")

    # New rich features (optional — defaults to safe values if not provided)
    orders_per_user_last_hour: Optional[int]  = Field(default=0)
    payment_attempts: Optional[int]           = Field(default=1)
    cart_to_checkout_seconds: Optional[float] = Field(default=60.0)
    ip_risk_score: Optional[float]            = Field(default=0.1, ge=0, le=1)
    order_hour: Optional[int]                 = Field(default=12, ge=0, le=23)
    known_fraud_address: Optional[int]        = Field(default=0, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "order_id": "order_12345",
                "user_id": "user_789",
                "order_amount": 4999.0,
                "orders_per_user_last_minute": 4,
                "location_mismatch": 1,
                "device_type": "desktop",
                "orders_per_user_last_hour": 8,
                "payment_attempts": 3,
                "cart_to_checkout_seconds": 5.2,
                "ip_risk_score": 0.85,
                "order_hour": 2,
                "known_fraud_address": 0,
            }
        }


class FraudResponse(BaseModel):
    order_id: str
    user_id: str
    is_fraud: bool
    fraud_score: float          # 0.0 – 1.0 probability
    risk_level: str             # LOW / MEDIUM / HIGH / CRITICAL
    confidence: str             # model confidence tier
    top_fraud_reasons: list     # SHAP-based top 3 reasons
    recommended_action: str     # AUTO_APPROVE / REVIEW / BLOCK
    model_version: str


# ─────────────────────────────────────────────
# FEATURE BUILDER (mirrors train script)
# ─────────────────────────────────────────────
DEVICE_MAP = {"mobile": 0, "desktop": 1, "tablet": 2}

def build_feature_vector(req: OrderRequest) -> pd.DataFrame:
    device_enc = DEVICE_MAP.get(req.device_type.lower(), 0)
    is_late_night = 1 if (req.order_hour >= 23 or req.order_hour <= 4) else 0
    fast_checkout = 1 if req.cart_to_checkout_seconds < 10 else 0
    multi_payment = 1 if req.payment_attempts > 1 else 0

    # Approximate z-score (at inference we don't have user history)
    # In production: fetch from Redis feature store
    amount_zscore_user = (req.order_amount - 200) / 300
    amount_pct_rank    = min(req.order_amount / 10000, 1.0)
    is_new_user        = 1 if req.orders_per_user_last_hour == 0 else 0
    is_high_value      = 1 if req.order_amount > 5000 else 0

    velocity_amount_interaction = req.orders_per_user_last_minute * req.order_amount
    location_highvalue_risk     = req.location_mismatch * is_high_value

    features = {
        "order_amount":                  req.order_amount,
        "orders_per_user_last_minute":   req.orders_per_user_last_minute,
        "location_mismatch":             req.location_mismatch,
        "device_type_enc":               device_enc,
        "amount_zscore_user":            amount_zscore_user,
        "orders_per_user_last_hour":     req.orders_per_user_last_hour,
        "amount_pct_rank":               amount_pct_rank,
        "is_new_user":                   is_new_user,
        "order_hour":                    req.order_hour,
        "is_late_night":                 is_late_night,
        "multi_payment_attempt":         multi_payment,
        "fast_checkout":                 fast_checkout,
        "is_high_value":                 is_high_value,
        "known_fraud_address":           req.known_fraud_address,
        "ip_risk_score":                 req.ip_risk_score,
        "velocity_amount_interaction":   velocity_amount_interaction,
        "location_highvalue_risk":       location_highvalue_risk,
    }

    feature_cols = model_artifacts["feature_cols"]
    return pd.DataFrame([features])[feature_cols]


# ─────────────────────────────────────────────
# SHAP EXPLAINER
# ─────────────────────────────────────────────
def get_shap_reasons(feature_vector: pd.DataFrame, top_n: int = 3) -> list:
    """Returns top N fraud reasons as human-readable strings."""
    try:
        ensemble  = model_artifacts["ensemble"]
        xgb_model = ensemble.estimators_[0]   # XGBoost base learner

        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(feature_vector)

        # Map feature index → magnitude
        feature_cols  = model_artifacts["feature_cols"]
        shap_row      = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        top_indices   = np.argsort(np.abs(shap_row))[::-1][:top_n]

        reason_map = {
            "order_amount":                "Unusually high order amount",
            "orders_per_user_last_minute": "Multiple orders in rapid succession",
            "location_mismatch":           "Billing/shipping location mismatch",
            "device_type_enc":             "Suspicious device type",
            "amount_zscore_user":          "Amount far above user's normal range",
            "orders_per_user_last_hour":   "High order velocity this hour",
            "is_new_user":                 "First-time buyer with high-risk pattern",
            "is_late_night":               "Order placed in late-night window",
            "multi_payment_attempt":       "Multiple payment attempts detected",
            "fast_checkout":               "Checkout completed suspiciously fast",
            "is_high_value":               "High-value order above 95th percentile",
            "known_fraud_address":         "Address linked to previous fraud",
            "ip_risk_score":               "High-risk IP address or proxy detected",
            "velocity_amount_interaction": "High velocity combined with high amount",
            "location_highvalue_risk":     "Location mismatch on a high-value order",
        }

        reasons = []
        for i in top_indices:
            feat_name = feature_cols[i]
            reasons.append(reason_map.get(feat_name, feat_name))

        return reasons

    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return ["High fraud score from ensemble model"]


# ─────────────────────────────────────────────
# RISK LEVEL + ACTION LOGIC
# ─────────────────────────────────────────────
def score_to_risk(score: float) -> tuple:
    """Returns (risk_level, confidence, recommended_action)"""
    if score < 0.2:
        return "LOW",      "HIGH",   "AUTO_APPROVE"
    elif score < 0.4:
        return "MEDIUM",   "MEDIUM", "AUTO_APPROVE"
    elif score < 0.6:
        return "HIGH",     "MEDIUM", "REVIEW"        # → human-in-the-loop
    elif score < 0.8:
        return "HIGH",     "HIGH",   "REVIEW"
    else:
        return "CRITICAL", "HIGH",   "BLOCK"


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────
@app.post("/predict", response_model=FraudResponse)
async def predict_fraud(order: OrderRequest):
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Build features
        feature_vector = build_feature_vector(order)

        # Scale
        scaler         = model_artifacts["scaler"]
        X_scaled       = scaler.transform(feature_vector)

        # Predict
        ensemble       = model_artifacts["ensemble_model"]
        threshold      = model_artifacts["threshold"]
        fraud_score    = float(ensemble.predict_proba(X_scaled)[0][1])
        is_fraud       = fraud_score >= threshold

        # Risk assessment
        risk_level, confidence, action = score_to_risk(fraud_score)

        # SHAP explanations (only for flagged orders to save latency)
        reasons = get_shap_reasons(
            pd.DataFrame(X_scaled, columns=model_artifacts["feature_cols"])
        ) if is_fraud else []

        return FraudResponse(
            order_id=order.order_id,
            user_id=order.user_id,
            is_fraud=is_fraud,
            fraud_score=round(fraud_score, 4),
            risk_level=risk_level,
            confidence=confidence,
            top_fraud_reasons=reasons,
            recommended_action=action,
            model_version="ensemble-v2.0",
        )

    except Exception as e:
        logger.error(f"Prediction error for {order.order_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  bool(model_artifacts),
        "model_version": "ensemble-v2.0",
        "pr_auc":        model_artifacts.get("pr_auc", "N/A"),
        "n_features":    len(model_artifacts.get("feature_cols", [])),
        "threshold":     model_artifacts.get("threshold", "N/A"),
    }


@app.get("/features")
async def get_features():
    """Lists all 16 features the model expects."""
    return {
        "feature_cols": model_artifacts.get("feature_cols", []),
        "total":        len(model_artifacts.get("feature_cols", [])),
    }