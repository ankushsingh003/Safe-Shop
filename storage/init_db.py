"""
storage/init_db.py
───────────────────
UPGRADED: Adds full Star Schema warehouse design alongside
the existing processed_orders table.

Star Schema Tables:
  fact_orders     — central fact table (every order, all measures)
  dim_users       — user dimension (profile, reputation, value tier)
  dim_products    — product dimension (stats, fraud rate, risk tier)
  dim_time        — time dimension (hour, day, week, is_weekend)

Why Star Schema?
  The processed_orders table Spark writes to is a flat operational table —
  good for writing fast, bad for analytics queries. The star schema organizes
  the same data into facts (what happened) and dimensions (who, what, when)
  so analysts can answer business questions in one clean JOIN instead of
  complex nested subqueries.

Interview answer:
  "I designed a star schema with one fact table and three dimension tables.
   fact_orders is the grain — one row per order. Dimensions are dim_users,
   dim_products, and dim_time. dbt refreshes the dimensions hourly from
   processed_orders. This gives the Grafana dashboard sub-second query
   performance even on millions of rows."
"""

import psycopg2
import os
import time

def get_connection(retries=5):
    conn = None
    for attempt in range(retries):
        try:
            conn = psycopg2.connect(
                dbname=os.environ.get("POSTGRES_DB",       "safeshop_orders"),
                user=os.environ.get("POSTGRES_USER",       "postgres"),
                password=os.environ.get("POSTGRES_PASSWORD","postgres"),
                host=os.environ.get("POSTGRES_HOST",       "localhost"),
                port=os.environ.get("POSTGRES_PORT",       "5432"),
            )
            print("✅ Connected to PostgreSQL")
            return conn
        except Exception as e:
            print(f"⏳ Attempt {attempt+1}/{retries} failed: {e}. Retrying in 5s...")
            time.sleep(5)
    raise RuntimeError("❌ Could not connect to PostgreSQL after retries.")


def init_db():
    conn = get_connection()
    cur  = conn.cursor()

    # ──────────────────────────────────────────────────────────────────────
    # TABLE 1: processed_orders (EXISTING — Spark writes here in real-time)
    # Keep this exactly as before. This is the raw operational table.
    # ──────────────────────────────────────────────────────────────────────
    print("Creating table: processed_orders (raw operational)...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS processed_orders (
            order_id         VARCHAR(100) PRIMARY KEY,
            user_id          VARCHAR(100),
            product_id       VARCHAR(100),
            product_name     VARCHAR(255),
            category         VARCHAR(100),
            amount           DECIMAL(12, 2),
            order_amount     DECIMAL(12, 2),
            quantity         INTEGER,
            payment_method   VARCHAR(50),
            ip_address       VARCHAR(100),
            device_type      VARCHAR(50),
            location_mismatch BOOLEAN DEFAULT FALSE,
            timestamp        TIMESTAMP,
            orders_per_user_1m INTEGER DEFAULT 0,
            fraud_score      FLOAT,
            is_fraud         BOOLEAN DEFAULT FALSE,
            risk_level       VARCHAR(20) DEFAULT 'LOW',
            reasoning        TEXT,
            processed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Index for Grafana dashboard queries (time-based filtering)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_processed_orders_timestamp
        ON processed_orders(timestamp DESC);
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_processed_orders_is_fraud
        ON processed_orders(is_fraud) WHERE is_fraud = TRUE;
    """)

    # ──────────────────────────────────────────────────────────────────────
    # TABLE 2: demand_forecast (EXISTING — Airflow writes here hourly)
    # ──────────────────────────────────────────────────────────────────────
    print("Creating table: demand_forecast...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS demand_forecast (
            forecast_id        SERIAL PRIMARY KEY,
            forecast_time      TIMESTAMP NOT NULL,
            category           VARCHAR(100) DEFAULT 'All',
            predicted_orders   FLOAT NOT NULL,
            confidence_upper   FLOAT,
            confidence_lower   FLOAT,
            model_version      VARCHAR(50) DEFAULT 'TFT-v1',
            generated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (forecast_time, category)
        );
    """)

    # ══════════════════════════════════════════════════════════════════════
    # STAR SCHEMA — DIMENSION TABLES
    # These are populated by dbt from processed_orders.
    # ══════════════════════════════════════════════════════════════════════

    # ──────────────────────────────────────────────────────────────────────
    # DIM 1: dim_time
    # Pre-computed time attributes for every hour in the data.
    # Avoids expensive EXTRACT() calls on every Grafana query.
    # ──────────────────────────────────────────────────────────────────────
    print("Creating dimension table: dim_time...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dim_time (
            time_id      SERIAL PRIMARY KEY,
            full_timestamp TIMESTAMP UNIQUE NOT NULL,
            hour         INTEGER NOT NULL,         -- 0-23
            day_of_week  INTEGER NOT NULL,         -- 0=Sun, 6=Sat
            day_name     VARCHAR(10) NOT NULL,     -- Monday, Tuesday...
            day_of_month INTEGER NOT NULL,         -- 1-31
            week_of_year INTEGER NOT NULL,         -- 1-52
            month        INTEGER NOT NULL,         -- 1-12
            month_name   VARCHAR(10) NOT NULL,     -- January...
            quarter      INTEGER NOT NULL,         -- 1-4
            year         INTEGER NOT NULL,
            is_weekend   BOOLEAN NOT NULL,         -- Sat or Sun
            is_late_night BOOLEAN NOT NULL,        -- 00:00-05:59
            is_business_hours BOOLEAN NOT NULL     -- 09:00-18:00 weekday
        );
    """)

    # ──────────────────────────────────────────────────────────────────────
    # DIM 2: dim_products
    # One row per product. Aggregated stats refreshed by dbt hourly.
    # ──────────────────────────────────────────────────────────────────────
    print("Creating dimension table: dim_products...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dim_products (
            product_id          VARCHAR(100) PRIMARY KEY,
            product_name        VARCHAR(255) NOT NULL,
            category            VARCHAR(100) NOT NULL,

            -- Volume metrics
            total_orders        INTEGER DEFAULT 0,
            unique_buyers       INTEGER DEFAULT 0,

            -- Revenue metrics
            total_revenue_usd   DECIMAL(14, 2) DEFAULT 0,
            avg_order_value_usd DECIMAL(10, 2) DEFAULT 0,
            min_order_value_usd DECIMAL(10, 2) DEFAULT 0,
            max_order_value_usd DECIMAL(10, 2) DEFAULT 0,

            -- Fraud metrics
            fraud_order_count   INTEGER DEFAULT 0,
            fraud_rate_pct      DECIMAL(6, 3) DEFAULT 0,
            total_fraud_exposure_usd DECIMAL(14, 2) DEFAULT 0,

            -- Risk classification
            product_risk_tier   VARCHAR(30) DEFAULT 'LOW_RISK_PRODUCT',
            rank_in_category    INTEGER,
            revenue_rank_overall INTEGER,

            -- Audit
            last_refreshed_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # ──────────────────────────────────────────────────────────────────────
    # DIM 3: dim_users
    # One row per user. Reputation score feeds back to Redis feature store.
    # ──────────────────────────────────────────────────────────────────────
    print("Creating dimension table: dim_users...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS dim_users (
            user_id              VARCHAR(100) PRIMARY KEY,

            -- Order history
            total_orders         INTEGER DEFAULT 0,
            first_order_at       TIMESTAMP,
            last_order_at        TIMESTAMP,
            days_as_customer     INTEGER DEFAULT 0,

            -- Spending profile
            total_spend_usd      DECIMAL(14, 2) DEFAULT 0,
            avg_order_value_usd  DECIMAL(10, 2) DEFAULT 0,
            max_single_order_usd DECIMAL(10, 2) DEFAULT 0,

            -- Fraud history
            fraud_order_count    INTEGER DEFAULT 0,
            fraud_rate_pct       DECIMAL(6, 3) DEFAULT 0,
            avg_fraud_score      DECIMAL(6, 4) DEFAULT 0,
            max_fraud_score      DECIMAL(6, 4) DEFAULT 0,

            -- Behavioral signals
            location_mismatch_count INTEGER DEFAULT 0,
            late_night_order_count  INTEGER DEFAULT 0,
            high_value_order_count  INTEGER DEFAULT 0,
            peak_velocity           INTEGER DEFAULT 0,

            -- Computed scores (refreshed by dbt + pushed to Redis)
            reputation_score     INTEGER DEFAULT 100, -- 0-100, higher = safer
            user_segment         VARCHAR(20) DEFAULT 'STANDARD',
            -- STANDARD | TRUSTED | MEDIUM_RISK | HIGH_RISK | BANNED
            customer_value_tier  VARCHAR(10) DEFAULT 'BRONZE',
            -- BRONZE | SILVER | GOLD | PLATINUM

            -- Audit
            last_refreshed_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # ──────────────────────────────────────────────────────────────────────
    # FACT TABLE: fact_orders
    # Central fact table. One row per order. References all 3 dimensions.
    # This is the star schema core — surrogate keys link to dimensions.
    # ──────────────────────────────────────────────────────────────────────
    print("Creating fact table: fact_orders...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fact_orders (
            -- Surrogate key
            fact_id         SERIAL PRIMARY KEY,

            -- Natural key
            order_id        VARCHAR(100) UNIQUE NOT NULL,

            -- Foreign keys to dimensions (the "star" connections)
            user_id         VARCHAR(100) REFERENCES dim_users(user_id),
            product_id      VARCHAR(100) REFERENCES dim_products(product_id),
            time_id         INTEGER REFERENCES dim_time(time_id),

            -- Measures (the facts — numeric values we aggregate)
            order_amount    DECIMAL(12, 2) NOT NULL,
            quantity        INTEGER DEFAULT 1,

            -- Fraud measures
            fraud_score     DECIMAL(6, 4) DEFAULT 0,
            is_fraud        BOOLEAN DEFAULT FALSE,
            risk_level      VARCHAR(20) DEFAULT 'LOW',
            fraud_pattern   VARCHAR(30),

            -- Degenerate dimensions (low-cardinality attributes kept in fact)
            payment_method  VARCHAR(50),
            device_type     VARCHAR(50),
            category        VARCHAR(100),
            location_mismatch BOOLEAN DEFAULT FALSE,

            -- Audit
            loaded_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Indexes for common Grafana query patterns
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_orders_user ON fact_orders(user_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_orders_product ON fact_orders(product_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_orders_fraud ON fact_orders(is_fraud) WHERE is_fraud = TRUE;")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_fact_orders_risk ON fact_orders(risk_level);")

    # ──────────────────────────────────────────────────────────────────────
    # MLOps TABLE: model_performance_log
    # Tracks PR-AUC and drift scores per model version over time.
    # Used by Airflow DAG to decide whether to promote challenger to champion.
    # ──────────────────────────────────────────────────────────────────────
    print("Creating table: model_performance_log...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_performance_log (
            log_id          SERIAL PRIMARY KEY,
            model_name      VARCHAR(100) NOT NULL,  -- 'ensemble_champion', 'gnn_challenger'
            model_version   VARCHAR(50) NOT NULL,
            pr_auc          DECIMAL(6, 4),
            f1_score        DECIMAL(6, 4),
            precision_score DECIMAL(6, 4),
            recall_score    DECIMAL(6, 4),
            drift_score     DECIMAL(6, 4),          -- from Evidently
            drift_detected  BOOLEAN DEFAULT FALSE,
            retrain_triggered BOOLEAN DEFAULT FALSE,
            evaluated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ SafeShop Database Initialized Successfully

Tables created:
  Operational:  processed_orders, demand_forecast
  Star Schema:
    Fact:       fact_orders
    Dimensions: dim_time, dim_products, dim_users
  MLOps:        model_performance_log

Next steps:
  1. Run Spark pipeline → populates processed_orders
  2. Run dbt → populates star schema tables
  3. Run Airflow DAGs → populates demand_forecast
  4. Open Grafana → query from fact_orders + dims
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """)


if __name__ == "__main__":
    init_db()