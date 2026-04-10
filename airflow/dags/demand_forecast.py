from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import joblib
import pandas as pd
import psycopg2

default_args = {
    'owner': 'data_engineer',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

def generate_forecast():
    db_params = {
        'dbname': 'ecommerce_orders',
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432'
    }
    
    model_path = 'ml/models/forecast_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError("Forecast model not found. Run training first.")
        
    model = joblib.load(model_path)
    
    # Generate features for next 6 hours
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    future_hours = [now + timedelta(hours=i) for i in range(1, 7)]
    
    forecast_df = pd.DataFrame({'ds': future_hours})
    forecast_df['hour'] = forecast_df['ds'].dt.hour
    forecast_df['is_weekend'] = (forecast_df['ds'].dt.dayofweek >= 5).astype(int)
    
    predictions = model.predict(forecast_df[['hour', 'is_weekend']])
    forecast_df['predicted_volume'] = predictions
    
    # Write to DB
    conn = psycopg2.connect(**db_params)
    cur = conn.cursor()
    
    # Create forecast table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS demand_forecast (
            forecast_time TIMESTAMP PRIMARY KEY,
            predicted_orders FLOAT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    for _, row in forecast_df.iterrows():
        cur.execute("""
            INSERT INTO demand_forecast (forecast_time, predicted_orders)
            VALUES (%s, %s)
            ON CONFLICT (forecast_time) DO UPDATE 
            SET predicted_orders = EXCLUDED.predicted_orders,
                generated_at = EXCLUDED.generated_at;
        """, (row['ds'], row['predicted_volume']))
        
    conn.commit()
    cur.close()
    conn.close()
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
