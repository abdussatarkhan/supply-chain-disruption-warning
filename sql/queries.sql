-- ============================================================================
-- Supply Chain Disruption Early Warning System
-- PostgreSQL Production Schema DDL & Analytical Query Suite
-- ============================================================================

-- 1. SCHEMAS & EXTENSIONS
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gist";

-- ============================================================================
-- 2. DDL: TABLE DEFINITIONS
-- ============================================================================

-- A. Bilateral Trade Flow Records (Partitioned by Year)
CREATE TABLE IF NOT EXISTS bilateral_trade_records (
    record_id UUID DEFAULT uuid_generate_v4(),
    period VARCHAR(6) NOT NULL, -- Format: YYYYMM
    trade_year INT NOT NULL,
    trade_month INT NOT NULL,
    reporter_iso CHAR(3) NOT NULL,
    partner_iso CHAR(3) NOT NULL,
    cmd_code VARCHAR(10) NOT NULL, -- Harmonized System (HS 84/85)
    flow_code INT NOT NULL,        -- 1: Import, 2: Export
    trade_value_usd NUMERIC(16, 2) NOT NULL,
    net_weight_kg NUMERIC(16, 2),
    mirror_partner_value_usd NUMERIC(16, 2),
    harmonized_trade_usd NUMERIC(16, 2),
    mirror_discrepancy_ratio NUMERIC(6, 4),
    data_quality_flag VARCHAR(20) DEFAULT 'VALID',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (record_id, trade_year)
) PARTITION BY RANGE (trade_year);

-- Partitions for historical and current horizons
CREATE TABLE IF NOT EXISTS bilateral_trade_records_2018 PARTITION OF bilateral_trade_records
    FOR VALUES FROM (2018) TO (2019);
CREATE TABLE IF NOT EXISTS bilateral_trade_records_2019 PARTITION OF bilateral_trade_records
    FOR VALUES FROM (2019) TO (2020);
CREATE TABLE IF NOT EXISTS bilateral_trade_records_2020 PARTITION OF bilateral_trade_records
    FOR VALUES FROM (2020) TO (2021);
CREATE TABLE IF NOT EXISTS bilateral_trade_records_2021 PARTITION OF bilateral_trade_records
    FOR VALUES FROM (2021) TO (2022);
CREATE TABLE IF NOT EXISTS bilateral_trade_records_2022 PARTITION OF bilateral_trade_records
    FOR VALUES FROM (2022) TO (2023);
CREATE TABLE IF NOT EXISTS bilateral_trade_records_2023 PARTITION OF bilateral_trade_records
    FOR VALUES FROM (2023) TO (2024);

-- B. GDELT Disruption Event Records
CREATE TABLE IF NOT EXISTS gdelt_disruption_events (
    event_id VARCHAR(32) PRIMARY KEY,
    event_date DATE NOT NULL,
    cameo_code VARCHAR(10) NOT NULL,
    goldstein_scale NUMERIC(5, 2),
    avg_tone NUMERIC(6, 2),
    num_mentions INT DEFAULT 1,
    action_geo_country CHAR(3),
    action_geo_port VARCHAR(100),
    source_title TEXT NOT NULL,
    cleaned_title TEXT,
    ner_facilities TEXT,
    ner_commodities TEXT,
    ner_carriers TEXT,
    negative_sentiment_score NUMERIC(5, 4),
    disruption_theme VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- C. World Bank Logistics Performance Index Benchmarks
CREATE TABLE IF NOT EXISTS lpi_benchmarks (
    country_iso CHAR(3) PRIMARY KEY,
    country_name VARCHAR(100) NOT NULL,
    lpi_overall NUMERIC(4, 2) NOT NULL,
    lpi_customs NUMERIC(4, 2),
    lpi_infra NUMERIC(4, 2),
    lpi_shipments NUMERIC(4, 2),
    lpi_quality NUMERIC(4, 2),
    lpi_tracking NUMERIC(4, 2),
    lpi_timeliness NUMERIC(4, 2),
    vulnerability_score NUMERIC(5, 4),
    updated_year INT DEFAULT 2023
);

-- D. Composite Risk Scores & Corridor Health Table
CREATE TABLE IF NOT EXISTS composite_risk_evaluations (
    evaluation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    period VARCHAR(6) NOT NULL,
    reporter_iso CHAR(3) NOT NULL,
    partner_iso CHAR(3) NOT NULL,
    cmd_code VARCHAR(10) NOT NULL,
    harmonized_trade_usd NUMERIC(16, 2),
    trade_anomaly_score NUMERIC(5, 4),
    news_sentiment_score NUMERIC(5, 4),
    topic_intensity_score NUMERIC(5, 4),
    lpi_vulnerability_score NUMERIC(5, 4),
    composite_risk_score NUMERIC(5, 4) NOT NULL,
    alert_tier VARCHAR(20) NOT NULL,
    recommended_action TEXT,
    evaluated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for ultra-fast query and dashboard performance
CREATE INDEX IF NOT EXISTS idx_trade_corridor ON bilateral_trade_records (reporter_iso, partner_iso, cmd_code, period);
CREATE INDEX IF NOT EXISTS idx_gdelt_date_country ON gdelt_disruption_events (event_date, action_geo_country);
CREATE INDEX IF NOT EXISTS idx_risk_tier_period ON composite_risk_evaluations (alert_tier, period, composite_risk_score DESC);


-- ============================================================================
-- 3. PRODUCTION ANALYTICAL QUERIES & WINDOW FUNCTIONS
-- ============================================================================

-- Query 1: Rolling 90-Day Trade Volatility & Mirror Reporting Discrepancy by Corridor
WITH corridor_monthly AS (
    SELECT
        period,
        TO_DATE(period, 'YYYYMM') AS trade_date,
        reporter_iso,
        partner_iso,
        cmd_code,
        harmonized_trade_usd,
        mirror_discrepancy_ratio,
        AVG(harmonized_trade_usd) OVER (
            PARTITION BY reporter_iso, partner_iso, cmd_code
            ORDER BY TO_DATE(period, 'YYYYMM')
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_3m,
        STDDEV(harmonized_trade_usd) OVER (
            PARTITION BY reporter_iso, partner_iso, cmd_code
            ORDER BY TO_DATE(period, 'YYYYMM')
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ) AS rolling_std_3m,
        LAG(harmonized_trade_usd, 12) OVER (
            PARTITION BY reporter_iso, partner_iso, cmd_code
            ORDER BY TO_DATE(period, 'YYYYMM')
        ) AS trade_val_lag_12m
    FROM bilateral_trade_records
)
SELECT
    period,
    reporter_iso,
    partner_iso,
    cmd_code,
    harmonized_trade_usd,
    ROUND(rolling_avg_3m, 2) AS rolling_mean_90d,
    ROUND(rolling_std_3m, 2) AS rolling_std_90d,
    CASE 
        WHEN rolling_std_3m IS NOT NULL AND rolling_std_3m > 0
        THEN ROUND((harmonized_trade_usd - rolling_avg_3m) / rolling_std_3m, 3)
        ELSE 0.0
    END AS zscore_shock_90d,
    CASE 
        WHEN trade_val_lag_12m IS NOT NULL AND trade_val_lag_12m > 0
        THEN ROUND(((harmonized_trade_usd - trade_val_lag_12m) / trade_val_lag_12m) * 100.0, 2)
        ELSE NULL
    END AS yoy_growth_pct,
    ROUND(mirror_discrepancy_ratio, 4) AS mirror_discrepancy
FROM corridor_monthly
ORDER BY reporter_iso, partner_iso, cmd_code, period;


-- Query 2: Correlated Disruption News Spikes Preceding Trade Volume Plunges (Lagged Lead Indicator)
WITH news_monthly_agg AS (
    SELECT
        TO_CHAR(event_date, 'YYYYMM') AS period,
        action_geo_country,
        COUNT(*) AS article_count,
        ROUND(AVG(negative_sentiment_score), 4) AS avg_negative_sentiment,
        COUNT(CASE WHEN disruption_theme IN ('PORT_CONGESTION', 'SEMICONDUCTOR_SHORTAGE', 'LABOR_STRIKE') THEN 1 END) AS critical_topic_mentions
    FROM gdelt_disruption_events
    GROUP BY TO_CHAR(event_date, 'YYYYMM'), action_geo_country
),
trade_shocks AS (
    SELECT
        period,
        reporter_iso,
        partner_iso,
        cmd_code,
        harmonized_trade_usd,
        LAG(harmonized_trade_usd, 1) OVER (
            PARTITION BY reporter_iso, partner_iso, cmd_code
            ORDER BY period
        ) AS prev_month_trade
    FROM bilateral_trade_records
)
SELECT
    t.period AS trade_drop_period,
    TO_CHAR(TO_DATE(t.period, 'YYYYMM') - INTERVAL '1 month', 'YYYYMM') AS lead_news_period,
    t.reporter_iso,
    t.partner_iso,
    t.cmd_code,
    t.harmonized_trade_usd,
    ROUND(((t.harmonized_trade_usd - t.prev_month_trade) / NULLIF(t.prev_month_trade, 0)) * 100.0, 2) AS mom_trade_drop_pct,
    n.article_count AS lead_month_articles,
    n.avg_negative_sentiment AS lead_month_sentiment,
    n.critical_topic_mentions AS lead_month_critical_mentions
FROM trade_shocks t
INNER JOIN news_monthly_agg n
    ON n.period = TO_CHAR(TO_DATE(t.period, 'YYYYMM') - INTERVAL '1 month', 'YYYYMM')
    AND n.action_geo_country = t.reporter_iso
WHERE ((t.harmonized_trade_usd - t.prev_month_trade) / NULLIF(t.prev_month_trade, 0)) <= -0.20 -- Plunge > 20%
  AND n.avg_negative_sentiment >= 0.55                            -- High leading negative narrative
ORDER BY mom_trade_drop_pct ASC;


-- Query 3: High-Risk Country-Commodity Corridor Leaderboard (Executive Warning Radar)
SELECT
    r.period,
    r.reporter_iso,
    r.partner_iso,
    r.cmd_code,
    r.composite_risk_score,
    r.alert_tier,
    ROUND(r.harmonized_trade_usd / 1000000.0, 2) AS monthly_exposure_usd_millions,
    l.lpi_overall AS partner_lpi_score,
    r.trade_anomaly_score,
    r.news_sentiment_score,
    r.recommended_action
FROM composite_risk_evaluations r
LEFT JOIN lpi_benchmarks l ON r.partner_iso = l.country_iso
WHERE r.alert_tier IN ('CRITICAL', 'HIGH')
ORDER BY r.period DESC, r.composite_risk_score DESC
LIMIT 50;


-- Query 4: Geospatial Port Chokepoint Disruption Density & Sentiment
SELECT
    action_geo_port,
    action_geo_country,
    COUNT(*) AS total_disruption_events,
    ROUND(AVG(negative_sentiment_score), 4) AS mean_disruption_severity,
    disruption_theme,
    MIN(event_date) AS earliest_report,
    MAX(event_date) AS latest_report
FROM gdelt_disruption_events
WHERE action_geo_port IS NOT NULL AND action_geo_port != 'None'
GROUP BY action_geo_port, action_geo_country, disruption_theme
HAVING COUNT(*) >= 5
ORDER BY total_disruption_events DESC, mean_disruption_severity DESC;
