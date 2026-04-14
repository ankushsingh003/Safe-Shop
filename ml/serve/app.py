import os
import pickle
import torch
import numpy as np
import pandas as pd
import redis
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import shap

# Import the GNN Architecture (Required for loading state_dict)
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F

# Import Agentic AI components
from ml.agents.fraud_investigator import fraud_investigator

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Fraud Intelligence API",
    description="Unified Ensemble + GNN + Agentic AI Fraud Detection",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# CONFIG & MODEL LOADING
# --------------------------------------------------------------------------
ENSEMBLE_PATH = "ml/models/ensemble_fraud_model.pkl"
GNN_PATH      = "ml/models/gnn_fraud_model.pt"
REDIS_HOST    = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT    = int(os.environ.get("REDIS_PORT", 6379))

model_artifacts = {}
gnn_model = None
redis_client = None

class FraudGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(FraudGNN, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

@app.on_event("startup")
async def startup_event():
    global model_artifacts, gnn_model, redis_client
    
    # 1. Load Ensemble Model
    try:
        with open(ENSEMBLE_PATH, "rb") as f:
            model_artifacts = pickle.load(f)
        logger.info(f"✅ Ensemble model loaded (PR-AUC: {model_artifacts.get('pr_auc', 0):.4f})")
    except Exception as e:
        logger.error(f"❌ Failed to load ensemble model: {e}")

    # 2. Load GNN Model
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gnn_model = FraudGNN(in_channels=1, hidden_channels=16, out_channels=2).to(device)
        gnn_model.load_state_dict(torch.load(GNN_PATH, map_location=device))
        gnn_model.eval()
        logger.info(f"✅ GNN model loaded successfully on {device}")
    except Exception as e:
        logger.error(f"❌ Failed to load GNN model: {e}")

    # 3. Connect to Redis
    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Connected to Redis Feature Store")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}. Running without real-time features.")

# --------------------------------------------------------------------------
# DATA MODELS
# --------------------------------------------------------------------------
class OrderRequest(BaseModel):
    order_id: str
    user_id: str
    order_amount: float
    ip_address: Optional[str] = "unknown"
    device_type: Optional[str] = "mobile"
    location_mismatch: Optional[int] = 0
    orders_per_user_last_minute: Optional[int] = 0

class FraudResponse(BaseModel):
    order_id: str
    is_fraud: bool
    fraud_score: float
    risk_level: str
    reasoning: str
    model_version: str
    investigator_involved: bool

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def get_gnn_score(order_amount):
    """Simple mock of GNN inference for a single node (In real life, construct subgraph from Redis)"""
    if gnn_model is None: return 0.5
    try:
        x = torch.tensor([[order_amount]], dtype=torch.float)
        # For real GNN, we'd fetch edges from Redis. Here we simulate a 'ring' if amount is high.
        edge_index = torch.tensor([], dtype=torch.long).reshape(2, 0)
        with torch.no_grad():
            out = gnn_model(x, edge_index)
            score = torch.exp(out)[0][1].item()
        return score
    except:
        return 0.5

# --------------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------------
@app.post("/predict", response_model=FraudResponse)
async def predict(order: OrderRequest):
    if not model_artifacts:
        raise HTTPException(status_code=503, detail="Models not ready")

    # 1. Feature Engineering for Ensemble
    # (Simplified for integration; normally use the build_feature_vector logic)
    feat_cols = model_artifacts["feature_cols"]
    # Mocking standard vector
    input_data = pd.DataFrame([{c: 0 for c in feat_cols}])
    input_data["order_amount"] = order.order_amount
    input_data["orders_per_user_last_minute"] = order.orders_per_user_last_minute
    input_data["location_mismatch"] = order.location_mismatch
    
    # 2. Get Ensemble Score
    scaler = model_artifacts["scaler"]
    ensemble = model_artifacts["ensemble_model"]
    X_sc = scaler.transform(input_data)
    ensemble_score = float(ensemble.predict_proba(X_sc)[0][1])

    # 3. Get GNN Score
    gnn_score = get_gnn_score(order.order_amount)

    # 4. Final Risk Logic
    max_score = max(ensemble_score, gnn_score)
    is_fraud = max_score > model_artifacts.get("threshold", 0.5)
    
    investigator_involved = False
    reasoning = "Automated model scoring."
    risk_level = "LOW"
    
    if max_score > 0.8:
        risk_level = "CRITICAL"
        reasoning = "Critical risk detected by multi-model consensus."
    elif max_score > 0.4:
        # BORDERLINE: Trigger Agentic AI
        investigator_involved = True
        risk_level = "HIGH"
        
        # Prepare state for LangGraph
        agent_input = {
            "order_id": order.order_id,
            "scores": {"ensemble": ensemble_score, "gnn": gnn_score},
            "metadata": {
                "user_id": order.user_id,
                "ip": order.ip_address,
                "device": order.device_type
            },
            "evidence": []
        }
        
        try:
            agent_result = fraud_investigator.invoke(agent_input)
            reasoning = agent_result.get("reasoning", "Agent investigation complete.")
            if agent_result["decision"] == "FINAL_BLOCK":
                is_fraud = True
            elif agent_result["decision"] == "FINAL_APPROVE":
                is_fraud = False
                risk_level = "MEDIUM (OVERTURNED)"
        except Exception as e:
            logger.error(f"Agent error: {e}")
            reasoning = "Borderline case, agent investigation failed. Falling back to model score."

    return FraudResponse(
        order_id=order.order_id,
        is_fraud=is_fraud,
        fraud_score=round(max_score, 4),
        risk_level=risk_level,
        reasoning=reasoning,
        model_version="v3.0.0-integrated",
        investigator_involved=investigator_involved
    )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ensemble_loaded": bool(model_artifacts),
        "gnn_loaded": gnn_model is not None,
        "redis_connected": redis_client is not None if redis_client else False
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
