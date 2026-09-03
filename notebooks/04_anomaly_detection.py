# %% [markdown]
# # 04 - Unsupervised Anomaly Detection & Geospatial Contagion
# **Supply Chain Disruption Early Warning System**
#
# This notebook explores:
# 1. Feature engineering of multi-scale volume shocks (log diff, YoY drop, rolling z-score, volatility ratio).
# 2. Isolation Forest model training, contamination tuning, and anomaly scoring.
# 3. DBSCAN geospatial clustering of concurrent corridor contractions (identifying regional contagion).
# 4. Diagnostic inspection of false positives vs confirmed disruptions.

# %%
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))

from scripts.utils import load_config
from scripts.anomaly_detection import TradeFeatureEngineer, IsolationForestDetector, GeospatialContagionClusterer

config = load_config("config/config.yaml")

# %% [markdown]
# ## 1. Feature Engineering: Multi-Scale Shock Indicators

# %%
anom_path = "data/processed/trade_anomalies_detected.parquet"
if not os.path.exists(anom_path):
    from scripts.anomaly_detection import run_anomaly_detection
    run_anomaly_detection()

df_anom = pd.read_parquet(anom_path)
print(f"Loaded {len(df_anom):,} anomaly records with features.")
display(df_anom[["period", "reporter_iso", "partner_iso", "trade_value_usd_log_diff", "trade_value_usd_zscore_90d", "trade_anomaly_score", "is_trade_anomaly"]].head()) if "display" in dir() else print(df_anom.head())

# %% [markdown]
# ## 2. Feature Correlation with Anomaly Score

# %%
feature_cols = [
    "trade_value_usd_log_diff",
    "trade_value_usd_zscore_90d",
    "trade_value_usd_yoy_pct_drop",
    "volatility_ratio_30_90",
    "trade_anomaly_score"
]

corr_matrix = df_anom[feature_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".3f", ax=ax, cbar_kws={"label": "Pearson Correlation"})
ax.set_title("Correlation Matrix: Shock Features vs. Isolation Forest Anomaly Score", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Anomaly Score Distribution and Contamination Threshold

# %%
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df_anom["trade_anomaly_score"], bins=50, kde=True, ax=ax, color="#7570b3")
ax.axvline(0.65, color="red", linestyle="--", linewidth=2, label="Anomaly Cutoff Score (0.65)")
ax.set_title("Distribution of Isolation Forest Normalized Anomaly Scores", fontsize=13, fontweight="bold")
ax.set_xlabel("Anomaly Score [0.0 = Normal, 1.0 = Extreme Deviation]")
ax.set_ylabel("Observation Count")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Spatial Contagion Clustering (DBSCAN)
# Identifying trade corridors experiencing simultaneous regional volume contractions.

# %%
contagion_events = df_anom[df_anom["spatial_contagion_cluster"] >= 0]
print(f"Total Corridors Involved in Spatial Contagion Clusters: {len(contagion_events)}")

# Group by period and cluster
contagion_summary = contagion_events.groupby(["period", "spatial_contagion_cluster"]).agg(
    corridor_count=("reporter_iso", "count"),
    mean_drop=("trade_value_usd_yoy_pct_drop", "mean"),
    max_anomaly_score=("trade_anomaly_score", "max")
).reset_index()

print("\nSample Geospatial Contagion Incidents Detected by DBSCAN:")
print(contagion_summary.head(10))

# %% [markdown]
# ## 5. Visual Corridor Anomaly Inspection
# Overlaying detected anomaly flags on US-China electronics trade series.

# %%
us_cn = df_anom[(df_anom["reporter_iso"] == "USA") & (df_anom["partner_iso"] == "CHN") & (df_anom["cmd_code"] == "8542")].copy()
us_cn["date"] = pd.to_datetime(us_cn["period"].astype(str), format="%Y%m")
us_cn = us_cn.sort_values("date")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(us_cn["date"], us_cn["harmonized_trade_usd"] / 1e6, label="Harmonized Trade Value ($M)", color="#1b9e77", linewidth=2)

# Highlight anomalies
anom_pts = us_cn[us_cn["is_trade_anomaly"] == 1]
ax.scatter(anom_pts["date"], anom_pts["harmonized_trade_usd"] / 1e6, color="red", s=80, zorder=5, label="Isolation Forest Anomaly Detected")

ax.set_title("US-China Semiconductor Trade (HS 8542) with Detected Anomalies", fontsize=13, fontweight="bold")
ax.set_ylabel("Trade Value ($ Millions USD)", fontsize=11)
ax.legend()
plt.tight_layout()
plt.show()
