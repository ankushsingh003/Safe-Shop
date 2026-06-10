
# SafeShop dbt Project

Transformation layer for the Safe-Shop Real-Time E-Commerce Pipeline.
Sits between the Spark-written PostgreSQL tables and the Grafana dashboard.

```
Kafka → Spark → PostgreSQL (raw) → dbt → Warehouse Tables → Grafana
                                    ↑
                              YOU ARE HERE
```

---

## Project Structure

```
safeshop/
├── dbt_project.yml              ← project config
├── profiles.yml                 ← DB connection (dev + prod)
│
├── models/
│   ├── staging/
│   │   ├── sources.yml          ← declares raw PostgreSQL source tables
│   │   ├── stg_orders.sql       ← cleans + deduplicates processed_orders
│   │   └── stg_orders.yml       ← column tests for staging layer
│   │
│   └── marts/
│       ├── schema.yml           ← all mart model tests
│       ├── core/
│       │   ├── dim_products.sql         ← one row per product, aggregated stats
│       │   ├── dim_users.sql            ← one row per user, reputation score
│       │   └── fct_forecast_vs_actuals.sql ← TFT forecast accuracy
│       └── fraud/
│           ├── fct_fraud_orders.sql         ← all confirmed fraud orders
│           └── fct_fraud_daily_summary.sql  ← daily fraud KPIs + trends
│
├── tests/
│   ├── assert_fraud_score_between_0_and_1.sql
│   ├── assert_no_future_orders.sql
│   └── assert_fraud_label_consistent_with_score.sql
│
└── macros/
    └── date_spine.sql           ← reusable date series generator
```

---

## Setup (One Time)

```bash
# 1. Install dbt with PostgreSQL adapter
pip install dbt-postgres

# 2. Copy profiles to dbt's expected location
mkdir -p ~/.dbt
cp profiles.yml ~/.dbt/profiles.yml

# 3. Verify connection to PostgreSQL
dbt debug

# 4. Install dbt packages (if any)
dbt deps
```

---

## Running dbt

```bash
# Run all models (staging → marts)
dbt run

# Run only staging layer
dbt run --select staging

# Run only fraud mart
dbt run --select marts.fraud

# Run a single model
dbt run --select fct_fraud_orders

# Run all tests
dbt test

# Run tests for one model
dbt test --select stg_orders

# Generate + view data lineage docs
dbt docs generate
dbt docs serve          # opens browser at localhost:8080
```

---

## Data Lineage (What Feeds What)

```
processed_orders (raw PostgreSQL)
        │
        ▼
  stg_orders (staging view)
        │
        ├──────────────────────┬──────────────────────┬────────────────────────
        ▼                      ▼                      ▼                       ▼
fct_fraud_orders        dim_products            dim_users        fct_fraud_daily_summary
(fraud fact table)    (product dimension)   (user dimension)     (daily KPI rollup)
        │
        └── fct_forecast_vs_actuals
            (joins with demand_forecast raw table)
```

---

## dbt Tests Summary

| Test | What it catches |
|------|----------------|
| `unique` + `not_null` on order_id | Duplicate orders from Spark retries |
| `accepted_values` on risk_level | Invalid risk tiers from FastAPI bugs |
| `assert_fraud_score_between_0_and_1` | ML scaler returning out-of-range scores |
| `assert_no_future_orders` | Producer clock misconfiguration or replay attacks |
| `assert_fraud_label_consistent_with_score` | LangGraph agent mislabeling orders |

---

## Connecting to Grafana

After `dbt run`, update your Grafana panels to query the mart tables:

| Panel | Old query table | New dbt table |
|-------|----------------|---------------|
| Fraud alerts feed | `processed_orders WHERE is_fraud=TRUE` | `marts.fct_fraud_orders` |
| Revenue by category | `processed_orders` | `marts.dim_products` |
| Daily fraud trend | manual SQL | `marts.fct_fraud_daily_summary` |
| Forecast vs actuals | manual join | `marts.fct_forecast_vs_actuals` |

---

## Running in Production (with Airflow)

Add to your Airflow DAG to run dbt after every Spark micro-batch completes:

```python
from airflow.operators.bash import BashOperator

dbt_run = BashOperator(
    task_id='dbt_run',
    bash_command='cd /opt/safeshop/dbt && dbt run --profiles-dir ~/.dbt',
    dag=dag,
)

dbt_test = BashOperator(
    task_id='dbt_test',
    bash_command='cd /opt/safeshop/dbt && dbt test --profiles-dir ~/.dbt',
    dag=dag,
)

spark_job >> dbt_run >> dbt_test
```