
/*
  stg_orders.sql
  ──────────────
  STAGING LAYER — Layer 1 of the dbt transformation pipeline.

  Purpose:
    Clean, rename, cast, and filter the raw `processed_orders` table
    that Spark writes to PostgreSQL. This is the single source of truth
    for all downstream models.

  What it does:
    - Renames raw columns to consistent snake_case names
    - Casts all columns to correct data types
    - Filters out any null or invalid order_ids (DLQ leftovers)
    - Adds derived helper columns used by multiple downstream models
    - Deduplicates on order_id (idempotency guarantee)

  Materialization: VIEW (always fresh, no storage cost)
*/

WITH raw_orders AS (
    SELECT * FROM {{ source('safeshop_raw', 'processed_orders') }}
),

deduplicated AS (
    -- Remove any duplicate order_ids (Spark retries can create dupes)
    SELECT *,
        ROW_NUMBER() OVER (
            PARTITION BY order_id
            ORDER BY timestamp DESC
        ) AS row_num
    FROM raw_orders
    WHERE order_id IS NOT NULL
),

cleaned AS (
    SELECT
        -- ── IDs ─────────────────────────────────────────────────────────
        order_id::VARCHAR(100)          AS order_id,
        user_id::VARCHAR(100)           AS user_id,

        -- ── ORDER DETAILS ────────────────────────────────────────────────
        order_amount::NUMERIC(12, 2)    AS order_amount,
        product_name::VARCHAR(200)      AS product_name,
        category::VARCHAR(100)          AS category,
        device_type::VARCHAR(50)        AS device_type,

        -- ── TIMESTAMPS ──────────────────────────────────────────────────
        timestamp::TIMESTAMP            AS order_timestamp,
        DATE(timestamp)                 AS order_date,
        EXTRACT(HOUR FROM timestamp)    AS order_hour,
        EXTRACT(DOW FROM timestamp)     AS day_of_week,   -- 0=Sun, 6=Sat
        CASE
            WHEN EXTRACT(DOW FROM timestamp) IN (0, 6) THEN TRUE
            ELSE FALSE
        END                             AS is_weekend,

        -- ── FRAUD SIGNALS ────────────────────────────────────────────────
        is_fraud::BOOLEAN               AS is_fraud,
        fraud_score::NUMERIC(6, 4)      AS fraud_score,
        risk_level::VARCHAR(20)         AS risk_level,
        location_mismatch::BOOLEAN      AS location_mismatch,
        orders_per_user_1m::INT         AS orders_per_user_1m,
        reasoning::TEXT                 AS fraud_reasoning,

        -- ── DERIVED FLAGS ────────────────────────────────────────────────
        CASE
            WHEN order_amount > {{ var('high_value_threshold') }} THEN TRUE
            ELSE FALSE
        END                             AS is_high_value,

        CASE
            WHEN EXTRACT(HOUR FROM timestamp) BETWEEN 0 AND 5 THEN TRUE
            ELSE FALSE
        END                             AS is_late_night_order

    FROM deduplicated
    WHERE row_num = 1                  -- keep only the latest record per order
      AND order_amount > 0             -- filter invalid amounts
      AND order_amount < 100000        -- filter obviously corrupt records
)

SELECT * FROM cleaned