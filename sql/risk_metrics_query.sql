-- Risk-adjusted performance ranking across the 20-stock watchlist.
-- Run after sql/schema.sql and after daily_returns / computed_metrics are populated.

SELECT
    dr.symbol,
    ROUND((AVG(dr.daily_return) * 252 * 100)::numeric, 2) AS annualized_return_pct,
    ROUND((STDDEV_SAMP(dr.daily_return) * SQRT(252) * 100)::numeric, 2) AS annualized_vol_pct,
    ROUND(
        ((AVG(dr.daily_return) * 252 - 0.04) / (STDDEV_SAMP(dr.daily_return) * SQRT(252)))::numeric, 2
    ) AS sharpe_ratio,
    ROUND(
        ((AVG(dr.daily_return) * 252 - 0.04) /
        (STDDEV_SAMP(dr.daily_return) FILTER (WHERE dr.daily_return < 0) * SQRT(252)))::numeric, 2
    ) AS sortino_ratio,
    ROUND((SUM(CASE WHEN cm.is_anomalous THEN 1 ELSE 0 END) * 100.0 / COUNT(*))::numeric, 1) AS pct_anomalous_days
FROM daily_returns dr
JOIN computed_metrics cm USING (symbol, trade_date)
GROUP BY dr.symbol
ORDER BY sharpe_ratio DESC;