
/*
  fct_fraud_orders.sql
  ────────────────────
  MART LAYER — Fraud Fact Table

  Purpose:
    The primary analytics table for the fraud operations team.
    Contains only confirmed fraud orders enriched with all
    investigation details, risk tiers, and attribution signals.

  Who uses this:
    - Grafana fraud alert dashboard panels
    - Fraud analyst reporting queries
    - Weekly fraud trend reports
    - Model performance evaluation (precision/recall tracking)

  Materialization: TABLE (analysts query this frequently — needs to be fast)
*/

WITH fraud_orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
    WHERE is_fraud = TRUE
),

enriched AS (
    SELECT
        -- ── IDENTIFIERS ──────────────────────────────────────────────────
        order_id,
        user_id,

        -- ── ORDER INFO ───────────────────────────────────────────────────
        order_amount,
        category,
        product_name,
        device_type,

        -- ── TIME DIMENSIONS ──────────────────────────────────────────────
        order_timestamp,
        order_date,
        order_hour,
        day_of_week,
        is_weekend,
        is_late_night_order,

        -- ── FRAUD INTELLIGENCE ───────────────────────────────────────────
        fraud_score,
        risk_level,
        fraud_reasoning,
        location_mismatch,
        orders_per_user_1m,
        is_high_value,

        -- ── RISK TIER LABEL (human readable) ─────────────────────────────
        CASE risk_level
            WHEN 'CRITICAL' THEN '🔴 CRITICAL'
            WHEN 'HIGH'     THEN '🟠 HIGH'
            ELSE                 '🟡 FLAGGED'
        END AS risk_display,

        -- ── FRAUD PATTERN CLASSIFICATION ─────────────────────────────────
        CASE
            WHEN location_mismatch AND orders_per_user_1m > 5
                THEN 'COORDINATED_ATTACK'
            WHEN location_mismatch AND is_high_value
                THEN 'STOLEN_CARD'
            WHEN orders_per_user_1m > 10
                THEN 'BOT_ACTIVITY'
            WHEN is_high_value AND is_late_night_order
                THEN 'ACCOUNT_TAKEOVER'
            WHEN location_mismatch
                THEN 'LOCATION_FRAUD'
            ELSE
                'ANOMALY'
        END AS fraud_pattern,

        -- ── FINANCIAL IMPACT ─────────────────────────────────────────────
        order_amount AS potential_loss_usd,

        CASE risk_level
            WHEN 'CRITICAL' THEN order_amount * 0.95   -- 95% of CRITICAL orders are actual fraud
            WHEN 'HIGH'     THEN order_amount * 0.70   -- 70% of HIGH orders are actual fraud
            ELSE                 order_amount * 0.30
        END AS estimated_loss_usd,

        -- ── AUDIT ────────────────────────────────────────────────────────
        CURRENT_TIMESTAMP AS dbt_updated_at

    FROM fraud_orders
)

SELECT * FROM enriched
ORDER BY order_timestamp DESC