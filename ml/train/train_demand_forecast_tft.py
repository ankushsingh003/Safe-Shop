"""
Demand Forecasting — Temporal Fusion Transformer (TFT)
Layer 4 of Upgrade Roadmap: Order Prediction Upgrade

This script upgrades demand forecasting from simple Prophet/Regression 
to a State-of-the-Art Deep Learning Transformer (TFT).
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer, QuantileLoss, GroupNormalizer

# CONFIG
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# --------------------------------------------------------------------------
# 1. SYNTHETIC TIME-SERIES GENERATOR
# --------------------------------------------------------------------------
def generate_forecast_data(n_days=730, n_groups=5):
    """
    Generates daily order data for multiple product groups.
    """
    np.random.seed(RANDOM_SEED)
    data_list = []
    
    for group_id in range(n_groups):
        # Base trend and seasonality
        time = np.arange(n_days)
        base = 100 + (group_id * 20)
        trend = 0.05 * time
        seasonality = 15 * np.sin(2 * np.pi * time / 7) # Weekly
        noise = np.random.normal(0, 5, n_days)
        
        volume = base + trend + seasonality + noise
        volume = np.maximum(volume, 0) # No negative orders
        
        group_df = pd.DataFrame({
            "day": time,
            "date": pd.date_range("2022-01-01", periods=n_days),
            "group": f"category_{group_id}",
            "volume": volume,
            "is_weekend": ((pd.date_range("2022-01-01", periods=n_days).dayofweek) >= 5).astype(int)
        })
        data_list.append(group_df)
        
    df = pd.concat(data_list, ignore_index=True)
    df["time_idx"] = df["day"] # Required for TFT
    df["group"] = df["group"].astype("category")
    return df

# --------------------------------------------------------------------------
# 2. MAIN TRAINING PIPELINE
# --------------------------------------------------------------------------
def train():
    print("Generating Time-Series Data (2 years, 5 categories)...")
    data = generate_forecast_data()
    
    # --------------------------------------------------------------------------
    # 3. CONSTRUCT TIMESERIES DATASET
    # --------------------------------------------------------------------------
    max_prediction_length = 30  # Predict next 30 days
    max_encoder_length = 90     # Look back at last 90 days
    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="volume",
        group_ids=["group"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["group"],
        time_varying_known_reals=["time_idx", "is_weekend"],
        time_varying_unknown_reals=["volume"],
        target_normalizer=GroupNormalizer(groups=["group"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Dataloaders
    batch_size = 64
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

    # --------------------------------------------------------------------------
    # 4. INITIALIZE TFT MODEL
    # --------------------------------------------------------------------------
    # Fast training for demo purposes: reduced hidden size and heads
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(), # Provides confidence intervals
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

    # --------------------------------------------------------------------------
    # 5. TRAIN USING PYTORCH LIGHTNING
    # --------------------------------------------------------------------------
    trainer = pl.Trainer(
        max_epochs=10, # Reduced for demo speed
        accelerator="cpu",
        enable_model_summary=True,
        callbacks=[LearningRateMonitor()],
    )

    print("\nStarting TFT Training (Deep Learning Forecast)...")
    trainer.fit(tft, train_dataloaders=train_dataloader)

    # --------------------------------------------------------------------------
    # 6. SAVE MODEL
    # --------------------------------------------------------------------------
    model_path = os.path.join(MODEL_DIR, "demand_forecast_tft.pt")
    torch.save(tft.state_dict(), model_path)
    print(f"\n[DONE] TFT Forecasting Model saved to {model_path}")

if __name__ == "__main__":
    train()
