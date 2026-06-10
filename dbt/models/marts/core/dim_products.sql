
/*
  dim_products.sql
  ────────────────
  MART LAYER — Product Dimension Table

  Purpose:
    One row per product with aggregated order stats, fraud rates,
    and demand signals. Powers the "Top Products" dashboard panel
    and the demand forecast vs actuals comparison.

  Who uses this:
    - Business analysts tracking product performance
    - Category managers monitoring fraud by product type
    - Demand forecasting team validating TFT predictions
    - Grafana: top products by volume panel

  Materialization: TABLE
*/

WITH order_stats AS (
    SELECT
        product_name,
        category,

        -- ── VOLUME METRICS ───────────────────────────────────────────────
        COUNT(*)                                    AS total_orders,
        COUNT(DISTINCT user_id)                     AS unique_buyers,

        -- ── REVENUE METRICS ──────────────────────────────────────────────
        SUM(order_amount)                           AS total_revenue_usd,
        AVG(order_amount)                           AS avg_order_value_usd,
        MIN(order_amount)                           AS min_order_value_usd,
        MAX(order_amount)                           AS max_order_value_usd,
        PERCENTILE_CONT(0.5) WITHIN GROUP (
            ORDER BY order_amount
        )                                           AS median_order_value_usd,

        -- ── FRAUD METRICS ────────────────────────────────────────────────
        SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)  AS fraud_order_count,
        ROUND(
            100.0 * SUM(CASE WHEN is_fraud THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        )                                           AS fraud_rate_pct,
        SUM(CASE WHEN is_fraud THEN order_amount ELSE 0 END)
                                                    AS total_fraud_exposure_usd,

        -- ── TIME METRICS ─────────────────────────────────────────────────
        MIN(order_timestamp)                        AS first_order_at,
        MAX(order_timestamp)                        AS last_order_at,

        -- ── BEHAVIORAL METRICS ───────────────────────────────────────────
        ROUND(AVG(orders_per_user_1m), 2)           AS avg_velocity_per_user,
        SUM(CASE WHEN is_late_night_order THEN 1 ELSE 0 END)
                                                    AS late_night_order_count,
        SUM(CASE WHEN is_high_value THEN 1 ELSE 0 END)
                                                    AS high_value_order_count

    FROM {{ ref('stg_orders') }}
    GROUP BY product_name, category
),

with_risk_tier AS (
    SELECT
        *,

        -- ── PRODUCT RISK CLASSIFICATION ───────────────────────────────────
        CASE
            WHEN fraud_rate_pct > 20 THEN 'HIGH_RISK_PRODUCT'
            WHEN fraud_rate_pct > 10 THEN 'MEDIUM_RISK_PRODUCT'
            ELSE 'LOW_RISK_PRODUCT'
        END AS product_risk_tier,

        -- ── POPULARITY RANK WITHIN CATEGORY ──────────────────────────────
        RANK() OVER (
            PARTITION BY category
            ORDER BY total_orders DESC
        ) AS rank_in_category,

        -- ── REVENUE RANK OVERALL ──────────────────────────────────────────
        RANK() OVER (
            ORDER BY total_revenue_usd DESC
        ) AS revenue_rank_overall,

        -- ── AUDIT ─────────────────────────────────────────────────────────
        CURRENT_TIMESTAMP AS dbt_updated_at

    FROM order_stats
)

SELECT * FROM with_risk_tier
ORDER BY total_orders DESC