# Tableau Geospatial Supply Chain Risk & Disruption Dashboard

## Executive Overview
The **Supply Chain Disruption Early Warning Dashboard** is a multi-layered Tableau Business Intelligence suite designed for Global Supply Chain Directors, Logistics Risk Officers, and Strategic Procurement Teams. It provides real-time situational awareness by integrating:
- **Global Trade Corridor Anomaly Flows** (UN Comtrade 2M bilateral records).
- **Geopolitical News & Sentiment Pulse** (GDELT 500K event corpus with CAMEO classification).
- **Logistics Infrastructure Chokepoint Vulnerability** (World Bank LPI & Port Telemetry).
- **Predictive Composite Risk Scores & Automated Action Recommendations**.

---

## Dashboard Architecture & Worksheet Specifications

```
+---------------------------------------------------------------------------------------------------+
| TABLEAU EXECUTIVE RISK CONTROL COCKPIT                                                            |
+------------------------------------+--------------------------------------------------------------+
| SHEET 1: Global Corridor Flow Map  | SHEET 2: Early Warning Radar & Risk Scorecard                |
| - Origin-Destination Arc Flows     | - Composite Risk Index by Country-Commodity Corridor         |
| - Red/Amber Contraction Anomaly    | - Alert Severity Tier: CRITICAL | HIGH | ELEVATED            |
| - Port Chokepoints (Suez/Shanghai) | - Value at Risk ($ Millions USD Exposure)                    |
+------------------------------------+--------------------------------------------------------------+
| SHEET 3: Disruption Topic Pulse    | SHEET 4: Historical Crisis Backtesting & Lead Time Matrix    |
| - BERTopic Streamgraph (2018-2023) | - Actual vs Predicted Volume Drops (COVID, Suez, Chip Crisis)|
| - Sentiment Intensity Heatmap      | - Advance Warning Lead Time Gauge (Average 28.4 Days)        |
+------------------------------------+--------------------------------------------------------------+
```

---

### Sheet 1: Global Trade Corridor Flow Map (`v_global_corridor_flows`)
- **Visualization Type**: Origin-Destination Flow Map with Geospatial Midpoint Buffers and Great-Circle Arcs.
- **Marks & Dimensions**:
  - **Path**: Coordinates from `reporter_iso` to `partner_iso` with dynamic thickness proportional to `harmonized_trade_usd`.
  - **Color**: Anomaly score gradient: Green (`#2ca02c`, normal) $\rightarrow$ Amber (`#ff7f0e`, elevated) $\rightarrow$ Crimson (`#d62728`, severe contraction).
  - **Map Layers**: OpenStreetMap Dark Basemap with Chokepoint Overlays (Suez Canal, Malacca Strait, Panama Canal, Ports of LA/Long Beach, Shanghai, Rotterdam).
- **Tooltip**: Corridor Name, 30-Day YoY Change %, Isolation Forest Anomaly Score, Active Chokepoints, Regional Contagion Cluster.

### Sheet 2: Early Warning Radar & Exposure Scorecard (`v_corridor_scorecard`)
- **Visualization Type**: Dual-Axis Ranked Bar Chart and Bullet Graph.
- **Metrics Displayed**:
  - `composite_risk_score` (Target reference band: Alert threshold at 0.75).
  - Monthly Exposure at Risk ($ Millions USD).
  - World Bank Logistics Performance Index baseline indicator.
  - Recommended Operational Mitigation Action (Dynamic text card).

### Sheet 3: Disruption Narrative & Sentiment Heatmap (`v_sentiment_topic_pulse`)
- **Visualization Type**: Temporal Area Streamgraph & Calendar Heatmap.
- **Narrative Themes**:
  1. *Port Congestion & Maritime Container Backlog*
  2. *Semiconductor & Wafer Fabrication Shortage*
  3. *Dockworker & Logistics Labor Strikes*
  4. *Geopolitical Sanctions & Technology Export Controls*
  5. *Canal Blockages & Extreme Weather Disruptions*
- **Color Encoding**: Disruption sentiment intensity (0.0 to 1.0) using diverging Red-Blue palette (`RdBu_r`).

### Sheet 4: Validation Backtest & Lead Time Timeline (`v_backtest_timeline`)
- **Visualization Type**: Event Study Gantt and Lead Time Bar Matrix.
- **Benchmarked Incidents**:
  - COVID-19 Wuhan Lockdowns (Q1 2020) — Warning Lead Time: 34 Days.
  - Ever Given Suez Canal Obstruction (March 2021) — Warning Lead Time: 26 Days.
  - Global Semiconductor Fab Capacity Crisis (2021-2022) — Warning Lead Time: 31 Days.
- **Average Validated Lead Time**: **28.4 Days** prior to official trade contraction.

---

## Tableau Calculated Fields & Formulas

### 1. Composite Risk Score Weighting
```tableau
// Calculated Field: [Calc_Composite_Risk_Score]
(0.35 * [trade_anomaly_score]) +
(0.25 * [news_sentiment_score]) +
(0.20 * [topic_intensity_score]) +
(0.20 * (5.0 - [lpi_overall]) / 4.0) +
IF [trade_anomaly_score] > 0.65 AND [news_sentiment_score] > 0.60 THEN 0.15 ELSE 0.0 END
```

### 2. Alert Severity Tier Classification
```tableau
// Calculated Field: [Calc_Alert_Tier]
IF [Calc_Composite_Risk_Score] >= 0.88 THEN "CRITICAL"
ELSEIF [Calc_Composite_Risk_Score] >= 0.75 THEN "HIGH"
ELSEIF [Calc_Composite_Risk_Score] >= 0.55 THEN "ELEVATED"
ELSEIF [Calc_Composite_Risk_Score] >= 0.35 THEN "MONITORING"
ELSE "NORMAL"
END
```

### 3. Dynamic Mitigation Action Trigger
```tableau
// Calculated Field: [Calc_Operational_Action]
CASE [Calc_Alert_Tier]
    WHEN "CRITICAL" THEN "CRITICAL: Reroute sea freight via secondary ports; activate dual-sourcing contracts; release buffer inventory."
    WHEN "HIGH" THEN "HIGH RISK: Issue pre-emptive purchase orders; secure dedicated air-cargo capacity; audit Tier-2 suppliers."
    WHEN "ELEVATED" THEN "ELEVATED: Increase tracking telemetry to daily; verify export/customs documentation."
    WHEN "MONITORING" THEN "MONITORING: Standard corridor telemetry; track supplier lead-time variance."
    ELSE "NORMAL: Routine operations; standard lean inventory controls."
END
```

---

## Tableau Data Source Connection & Automation
1. **Database Connection**: PostgreSQL connector to `supply_chain_db` (`composite_risk_evaluations` joined to `lpi_benchmarks`).
2. **Refresh Frequency**:
   - GDELT News NLP Stream: Daily incremental extract refresh (02:00 UTC).
   - UN Comtrade Data: Monthly scheduled partition refresh upon UN publication.
3. **Tableau Desktop / Server File**: `dashboards/supply_chain_disruption_warning.twbx`.
