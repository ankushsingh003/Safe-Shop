import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_model():
    # Create directory if not exists
    os.makedirs('ml/models', exist_ok=True)
    
    print("Generating synthetic training data...")
    # Features: order_amount, orders_per_user_1m
    # In a real scenario, we would use historical data from PostgreSQL
    
    # 95% Normal data
    normal_data = np.random.normal(loc=[100, 1], scale=[50, 0.5], size=(1000, 2))
    
    # 5% Anomalous data (high amount or high frequency)
    anomalous_data = np.random.uniform(low=[500, 5], high=[2000, 15], size=(50, 2))
    
    X = np.vstack([normal_data, anomalous_data])
    
    print("Training Isolation Forest model...")
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(X)
    
    model_path = 'ml/models/fraud_model.pkl'
    print(f"Saving model to {model_path}...")
    joblib.dump(model, model_path)
    print("Training complete.")

if __name__ == "__main__":
    train_model()
