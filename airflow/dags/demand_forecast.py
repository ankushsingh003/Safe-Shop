from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import requests
import psycopg2

default_args = {
    'owner': 'data_engineer',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

def generate_forecast():
    """Layer 4: Triggers TFT Forecast from the Intelligence API"""
    api_url = os.environ.get("ML_API_URL", "http://localhost:8000/forecast")
    api_key = os.environ.get("ML_API_KEY", "dev-secret-key")
    
    payload = {"category": "All-Product-Aggregated", "horizon_hours": 24}
    headers = {"X-API-KEY": api_key}
    
    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"API Forecast failed: {response.text}")
            
        forecast_data = response.json()["predictions"]
        
        # Write to DB
        conn = psycopg2.connect(
            dbname=os.environ.get("POSTGRES_DB", "ecommerce_orders"),
            user=os.environ.get("POSTGRES_USER", "postgres"),
            password=os.environ.get("POSTGRES_PASSWORD", "postgres"),
            host=os.environ.get("POSTGRES_HOST", "localhost")
        )
        cur = conn.cursor()
        
        # Table creation code remains same...
        cur.execute("""
            CREATE TABLE IF NOT EXISTS demand_forecast (
                forecast_time TIMESTAMP PRIMARY KEY,
                predicted_orders FLOAT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Insert predictions
        now = datetime.now()
        for p in forecast_data:
            forecast_time = now + timedelta(hours=p["hour_offset"])
            cur.execute("""
                INSERT INTO demand_forecast (forecast_time, predicted_orders)
                VALUES (%s, %s)
                ON CONFLICT (forecast_time) DO UPDATE SET predicted_orders = EXCLUDED.predicted_orders;
            """, (forecast_time, p["predicted_volume"]))
            
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"FAILED to generate forecast: {e}")
    print("Forecast generated and stored.")

with DAG(
    'hourly_demand_forecast',
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False,
) as dag:

    forecast_task = PythonOperator(
        task_id='generate_demand_forecast',
        python_callable=generate_forecast
    )
