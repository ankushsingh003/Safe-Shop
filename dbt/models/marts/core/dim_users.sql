
/*
  dim_users.sql
  ─────────────
  MART LAYER — User Dimension Table

  Purpose:
    One row per user with their full behavioral profile —
    order history, fraud history, spending patterns, and risk score.
    Powers the user reputation system and feeds back into the
    Redis feature store for real-time fraud scoring.

  Who uses this:
    - Risk team: identify repeat fraudsters
    - Marketing team: segment high-value legitimate customers
    - ML team: generate user-level features for model retraining
    - Airflow DAG: weekly refresh of Redis user reputation scores

  Materialization: TABLE
*/

WITH user_orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

user_stats AS (
    SELECT
        user_id,

        -- ── ORDER HISTORY ────────────────────────────────────────────────
        COUNT(*)                                        AS total_orders,
        MIN(order_timestamp)                            AS first_order_at,
        MAX(order_timestamp)                            AS last_order_at,
        EXTRACT(
            DAY FROM (MAX(order_timestamp) - MIN(order_timestamp))
        )                                               AS days_as_customer,

        -- ── SPENDING PROFILE ─────────────────────────────────────────────
        SUM(order_amount)                               AS total_spend_usd,
        AVG(order_amount)                               AS avg_order_value_usd,
        MAX(order_amount)                               AS max_single_order_usd,
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY order_amount
        )                                               AS median_order_value_usd,

        -- ── FRAUD HISTORY ────────────────────────────────────────────────
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)      AS fraud_order_count,
        ROUND(
            100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        )                                               AS fraud_rate_pct,
        AVG(fraud_score)                                AS avg_fraud_score,
        MAX(fraud_score)                                AS max_fraud_score,

        -- ── BEHAVIORAL SIGNALS ───────────────────────────────────────────
        SUM(CASE WHEN location_mismatch THEN 1 ELSE 0 END)
                                                        AS location_mismatch_count,
        SUM(CASE WHEN is_late_night_order THEN 1 ELSE 0 END)
                                                        AS late_night_order_count,
        SUM(CASE WHEN is_high_value THEN 1 ELSE 0 END) AS high_value_order_count,
        AVG(orders_per_user_1m)                         AS avg_velocity,
        MAX(orders_per_user_1m)                         AS peak_velocity,

        -- ── DEVICE & CATEGORY PREFERENCES ───────────────────────────────
        MODE() WITHIN GROUP (ORDER BY device_type)      AS preferred_device,
        MODE() WITHIN GROUP (ORDER BY category)         AS preferred_category,

        -- ── RECENT ACTIVITY (last 30 days) ───────────────────────────────
        SUM(
            CASE WHEN order_timestamp > NOW() - INTERVAL '{{ var("lookback_days") }} days'
            THEN 1 ELSE 0 END
        )                                               AS orders_last_30d,
        SUM(
            CASE WHEN order_timestamp > NOW() - INTERVAL '{{ var("lookback_days") }} days'
            THEN order_amount ELSE 0 END
        )                                               AS spend_last_30d_usd

    FROM user_orders
    GROUP BY user_id
),

with_reputation AS (
    SELECT
        *,

        -- ── USER REPUTATION SCORE (0-100, feeds back into Redis) ──────────
        -- Higher is safer. Used by FastAPI feature store.
        GREATEST(0, LEAST(100,
            100
            - (fraud_order_count * 20)          -- -20 per confirmed fraud order
            - (location_mismatch_count * 5)      -- -5 per location mismatch
            - (CASE WHEN peak_velocity > 10 THEN 15 ELSE 0 END)  -- -15 if burst detected
            + (CASE WHEN days_as_customer > 365 THEN 10 ELSE 0 END) -- +10 for old accounts
            + (CASE WHEN total_orders > 20 THEN 5 ELSE 0 END)    -- +5 for loyal customers
        ))                                              AS reputation_score,

        -- ── USER RISK SEGMENT ─────────────────────────────────────────────
        CASE
            WHEN fraud_order_count > 2              THEN 'BANNED'
            WHEN fraud_order_count > 0
              OR avg_fraud_score > 0.6              THEN 'HIGH_RISK'
            WHEN avg_fraud_score > 0.3
              OR location_mismatch_count > 2        THEN 'MEDIUM_RISK'
            WHEN total_orders > 20
              AND fraud_order_count = 0             THEN 'TRUSTED'
            ELSE                                         'STANDARD'
        END                                             AS user_segment,

        -- ── CUSTOMER VALUE TIER ───────────────────────────────────────────
        CASE
            WHEN total_spend_usd > 10000            THEN 'PLATINUM'
            WHEN total_spend_usd > 5000             THEN 'GOLD'
            WHEN total_spend_usd > 1000             THEN 'SILVER'
            ELSE                                         'BRONZE'
        END                                             AS customer_value_tier,

        -- ── AUDIT ─────────────────────────────────────────────────────────
        CURRENT_TIMESTAMP AS dbt_updated_at

    FROM user_stats
)

SELECT * FROM with_reputation
ORDER BY total_spend_usd DESC