import psycopg2
import os
import time

def init_db():
    db_name = os.environ.get("POSTGRES_DB", "ecommerce_orders")
    db_user = os.environ.get("POSTGRES_USER", "postgres")
    db_password = os.environ.get("POSTGRES_PASSWORD", "postgres")
    db_host = os.environ.get("POSTGRES_HOST", "localhost")
    db_port = os.environ.get("POSTGRES_PORT", "5432")

    print(f"Connecting to database {db_name} at {db_host}:{db_port}...")
    
    conn = None
    retry_count = 0
    while not conn and retry_count < 5:
        try:
            conn = psycopg2.connect(
                dbname=db_name,
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port
            )
        except Exception as e:
            print(f"Connection failed ({e}). Retrying in 5 seconds...")
            time.sleep(5)
            retry_count += 1

    if not conn:
        print("Could not connect to PostgreSQL. Exiting.")
        return

    cur = conn.cursor()

    # Create table for processed orders
    create_table_query = """
    CREATE TABLE IF NOT EXISTS processed_orders (
        order_id UUID PRIMARY KEY,
        user_id VARCHAR(100),
        product_id VARCHAR(100),
        product_name VARCHAR(255),
        category VARCHAR(100),
        amount DECIMAL(10, 2),
        quantity INTEGER,
        payment_method VARCHAR(50),
        ip_address VARCHAR(45),
        timestamp TIMESTAMP,
        orders_per_user_1m INTEGER,
        fraud_score FLOAT,
        is_fraud BOOLEAN,
        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    try:
        print("Creating table 'processed_orders'...")
        cur.execute(create_table_query)
        conn.commit()
        print("Table created successfully.")
    except Exception as e:
        print(f"Error creating table: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    init_db()
