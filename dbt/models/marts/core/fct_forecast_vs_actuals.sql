
/*
  fct_forecast_vs_actuals.sql
  ───────────────────────────
  MART LAYER — TFT Demand Forecast vs Actual Orders

  Purpose:
    Joins TFT model predictions with actual order volumes per hour.
    Used to evaluate forecast accuracy (MAPE, MAE) and visualize
    predicted vs actual demand in Grafana.

  Who uses this:
    - Grafana: forecast vs actuals chart (Panel 5)
    - ML team: TFT model performance monitoring
    - Warehouse/ops team: was the stock pre-positioned correctly?

  Materialization: TABLE
*/

WITH actuals AS (
    SELECT
        DATE_TRUNC('hour', order_timestamp)     AS hour_bucket,
        category,
        COUNT(*)                                AS actual_orders,
        SUM(order_amount)                       AS actual_revenue_usd
    FROM {{ ref('stg_orders') }}
    GROUP BY 1, 2
),

forecasts AS (
    SELECT
        forecast_time::TIMESTAMP                AS hour_bucket,
        category,
        predicted_orders,
        confidence_upper,
        confidence_lower
    FROM {{ source('safeshop_raw', 'demand_forecast') }}
),

joined AS (
    SELECT
        COALESCE(a.hour_bucket, f.hour_bucket)  AS hour_bucket,
        COALESCE(a.category, f.category)        AS category,

        -- ── ACTUALS ──────────────────────────────────────────────────────
        COALESCE(a.actual_orders, 0)            AS actual_orders,
        COALESCE(a.actual_revenue_usd, 0)       AS actual_revenue_usd,

        -- ── FORECAST ─────────────────────────────────────────────────────
        f.predicted_orders,
        f.confidence_upper,
        f.confidence_lower,

        -- ── ACCURACY METRICS ─────────────────────────────────────────────
        -- Absolute Error
        ABS(COALESCE(a.actual_orders, 0) - COALESCE(f.predicted_orders, 0))
                                                AS absolute_error,

        -- MAPE component (Mean Absolute Percentage Error)
        CASE
            WHEN COALESCE(a.actual_orders, 0) > 0
            THEN ABS(
                (COALESCE(a.actual_orders, 0) - COALESCE(f.predicted_orders, 0))
                / COALESCE(a.actual_orders, 0)
            ) * 100
            ELSE NULL
        END                                     AS absolute_pct_error,

        -- Was actual within confidence interval?
        CASE
            WHEN a.actual_orders BETWEEN f.confidence_lower AND f.confidence_upper
            THEN TRUE ELSE FALSE
        END                                     AS within_confidence_interval,

        -- ── AUDIT ─────────────────────────────────────────────────────────
        CURRENT_TIMESTAMP AS dbt_updated_at

    FROM actuals a
    FULL OUTER JOIN forecasts f
        ON  a.hour_bucket = f.hour_bucket
        AND a.category    = f.category
)

SELECT * FROM joined
ORDER BY hour_bucket DESC, category