
/*
  fct_fraud_daily_summary.sql
  ───────────────────────────
  MART LAYER — Daily Fraud Aggregation

  Purpose:
    One row per day with full fraud KPIs. Used by leadership dashboards
    and weekly business reviews. Also powers the model performance
    tracking — PR-AUC decay is visible when fraud_rate_pct spikes
    without a corresponding rise in model detections.

  Who uses this:
    - Weekly business review reports
    - Model performance monitoring (drift signal)
    - Finance team: daily chargeback risk exposure
    - Grafana: 30-day fraud trend chart

  Materialization: TABLE
*/

WITH daily_base AS (
    SELECT
        order_date,

        -- ── VOLUME ───────────────────────────────────────────────────────
        COUNT(*)                                        AS total_orders,
        COUNT(DISTINCT user_id)                         AS unique_users,
        SUM(order_amount)                               AS total_revenue_usd,

        -- ── FRAUD COUNTS ─────────────────────────────────────────────────
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)       AS fraud_orders,
        SUM(CASE WHEN risk_level = 'CRITICAL' THEN 1 ELSE 0 END)
                                                        AS critical_orders,
        SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END)
                                                        AS high_risk_orders,

        -- ── FINANCIAL EXPOSURE ────────────────────────────────────────────
        SUM(CASE WHEN is_fraud THEN order_amount ELSE 0 END)
                                                        AS fraud_revenue_at_risk_usd,

        -- ── FRAUD RATES ──────────────────────────────────────────────────
        ROUND(
            100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 3
        )                                               AS fraud_rate_pct,

        -- ── MODEL PERFORMANCE PROXY ───────────────────────────────────────
        ROUND(AVG(fraud_score), 4)                      AS avg_fraud_score,
        ROUND(AVG(CASE WHEN is_fraud THEN fraud_score END), 4)
                                                        AS avg_fraud_score_on_positives,

        -- ── BEHAVIORAL PATTERNS ───────────────────────────────────────────
        SUM(CASE WHEN location_mismatch THEN 1 ELSE 0 END)
                                                        AS location_mismatch_count,
        SUM(CASE WHEN is_late_night_order AND is_fraud THEN 1 ELSE 0 END)
                                                        AS late_night_fraud_count,
        SUM(CASE WHEN is_high_value AND is_fraud THEN 1 ELSE 0 END)
                                                        AS high_value_fraud_count,

        -- ── TOP FRAUD CATEGORY ────────────────────────────────────────────
        MODE() WITHIN GROUP (
            ORDER BY CASE WHEN is_fraud THEN category END
        )                                               AS top_fraud_category

    FROM {{ ref('stg_orders') }}
    GROUP BY order_date
),

with_trends AS (
    SELECT
        *,

        -- ── 7-DAY ROLLING FRAUD RATE ──────────────────────────────────────
        ROUND(AVG(fraud_rate_pct) OVER (
            ORDER BY order_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ), 3)                                           AS fraud_rate_7d_avg,

        -- ── DAY OVER DAY CHANGE ───────────────────────────────────────────
        fraud_orders - LAG(fraud_orders, 1) OVER (
            ORDER BY order_date
        )                                               AS fraud_orders_dod_change,

        total_orders - LAG(total_orders, 1) OVER (
            ORDER BY order_date
        )                                               AS total_orders_dod_change,

        -- ── WEEK OVER WEEK CHANGE ─────────────────────────────────────────
        ROUND(fraud_rate_pct - LAG(fraud_rate_pct, 7) OVER (
            ORDER BY order_date
        ), 3)                                           AS fraud_rate_wow_change,

        -- ── AUDIT ─────────────────────────────────────────────────────────
        CURRENT_TIMESTAMP AS dbt_updated_at

    FROM daily_base
)

SELECT * FROM with_trends
ORDER BY order_date DESC