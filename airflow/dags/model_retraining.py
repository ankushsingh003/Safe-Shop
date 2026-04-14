from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import subprocess

default_args = {
    'owner': 'data_engineer',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def check_drift():
    """Triggers the drift detection script"""
    script_path = "ml/mlops/drift_detector.py"
    if os.path.exists(script_path):
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Warning: Drift detection failed: {result.stderr}")
    else:
        print(f"Drift detector not found at {script_path}")

def train_gnn():
    """Triggers the GNN fraud ring model training"""
    script_path = "ml/train/train_gnn_fraud_model.py"
    if os.path.exists(script_path):
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise Exception(f"GNN Training failed: {result.stderr}")
    else:
        raise FileNotFoundError(f"GNN Training script not found at {script_path}")

def train_model():
    """Triggers the model training script"""
    script_path = "ml/train/train_fraud_model.py"
    if os.path.exists(script_path):
        result = subprocess.run(["python", script_path], capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")
    else:
        raise FileNotFoundError(f"Training script not found at {script_path}")

with DAG(
    'fraud_model_lifecycle',
    default_args=default_args,
    description='Monitors drift and retrains the ensemble fraud model',
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:

    drift_task = PythonOperator(
        task_id='monitor_drift',
        python_callable=check_drift,
    )

    retrain_task = PythonOperator(
        task_id='retrain_ensemble_model',
        python_callable=train_model,
    )

    gnn_task = PythonOperator(
        task_id='train_gnn_fraud_rings',
        python_callable=train_gnn,
    )

    drift_task >> [retrain_task, gnn_task]
