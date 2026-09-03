"""
Trade Volume Anomaly Detection and Spatial Contagion Clustering.

Combines:
1. Multi-scale Time-Series Feature Engineering (MoM log-diff, YoY drop, rolling z-score, volatility ratio).
2. Unsupervised Isolation Forest Anomaly Detection with contamination tuning.
3. DBSCAN Geospatial Clustering for identifying regional shock contagion corridors.
4. Per-corridor and commodity anomaly risk calibration.
"""

import os
import joblib
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import DBSCAN

from scripts.utils import setup_logger, load_config, ensure_directory, get_country_centroids

logger = setup_logger("anomaly_detection")


class TradeFeatureEngineer:
    """Computes high-sensitivity temporal shock indicators on bilateral trade volume series."""

    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates rolling z-scores, year-over-year drops, log differences,
        and volatility ratios per (reporter, partner, commodity) group.
        """
        logger.info("Engineering temporal shock features per trade corridor...")
        df_sorted = df.sort_values(["reporter_iso", "partner_iso", "cmd_code", "date"]).copy()

        engineered_dfs = []
        group_cols = ["reporter_iso", "partner_iso", "cmd_code"]

        for _, grp in df_sorted.groupby(group_cols):
            grp = grp.copy()
            val = grp["harmonized_trade_usd"]

            # 1. Month-over-Month Log Difference
            log_val = np.log(np.maximum(val, 1.0))
            grp["trade_value_usd_log_diff"] = log_val.diff().fillna(0.0)

            # 2. Rolling 3-Month Z-Score (90 days)
            rolling_mean_3m = val.rolling(window=3, min_periods=1).mean()
            rolling_std_3m = val.rolling(window=3, min_periods=1).std().fillna(1.0)
            rolling_std_3m = np.where(rolling_std_3m == 0, 1.0, rolling_std_3m)
            grp["trade_value_usd_zscore_90d"] = ((val - rolling_mean_3m) / rolling_std_3m).clip(-5.0, 5.0)

            # 3. Year-over-Year Percentage Drop (Negative shock indicator)
            yoy_lag = val.shift(12)
            yoy_diff = (val - yoy_lag) / np.maximum(yoy_lag, 1.0)
            # Clip between -1.0 (-100% drop) and 2.0 (+200% surge), invert so drop is positive risk
            grp["trade_value_usd_yoy_pct_drop"] = (-yoy_diff).fillna(0.0).clip(-1.0, 2.0)

            # 4. Volatility Ratio (Short-term 3m vs Long-term 12m)
            std_short = val.rolling(window=3, min_periods=1).std().fillna(1.0)
            std_long = val.rolling(window=12, min_periods=1).std().fillna(1.0)
            std_long = np.where(std_long == 0, 1.0, std_long)
            grp["volatility_ratio_30_90"] = (std_short / std_long).clip(0.1, 10.0)

            # 5. Downward Shock Magnitude (Rectified negative deviation)
            grp["downward_shock_magnitude"] = np.maximum(0.0, -grp["trade_value_usd_log_diff"])

            engineered_dfs.append(grp)

        result_df = pd.concat(engineered_dfs, ignore_index=True)
        return result_df


class IsolationForestDetector:
    """Trained Isolation Forest for detecting sudden volume collapses and supply anomalies."""

    def __init__(self, config: Dict[str, Any]):
        self.params = config["anomaly_detection"]["isolation_forest"]
        self.features = self.params.get("features", [
            "trade_value_usd_log_diff",
            "trade_value_usd_zscore_90d",
            "trade_value_usd_yoy_pct_drop",
            "volatility_ratio_30_90"
        ])
        self.scaler = RobustScaler()
        self.model = None

    def tune_and_fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        """Fits Isolation Forest with robust feature scaling and configured contamination rate."""
        logger.info(f"Fitting Isolation Forest on features: {self.features}...")
        X = df[self.features].values

        # Impute any remaining NaNs
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
        X_scaled = self.scaler.fit_transform(X)

        n_estimators = self.params.get("n_estimators", 250)
        contamination = self.params.get("contamination", 0.05)
        max_samples = self.params.get("max_samples", 0.8)
        random_state = self.params.get("random_state", 42)

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.model.fit(X_scaled)
        logger.info("Isolation Forest fitting completed.")
        return self

    def predict_anomaly_scores(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates binary anomaly indicator (-1 anomaly, 1 normal)
        and normalized anomaly risk score [0.0, 1.0].
        """
        X = df[self.features].values
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
        X_scaled = self.scaler.transform(X)

        # raw score_samples returns opposite of anomaly score (lower means more abnormal)
        raw_scores = self.model.score_samples(X_scaled)
        # Normalize to [0.0, 1.0] where 1.0 is most anomalous
        min_s, max_s = raw_scores.min(), raw_scores.max()
        if max_s > min_s:
            norm_anomaly_score = (max_s - raw_scores) / (max_s - min_s)
        else:
            norm_anomaly_score = np.zeros_like(raw_scores)

        binary_preds = self.model.predict(X_scaled) # -1: anomaly, 1: inlier
        anomaly_flags = (binary_preds == -1).astype(int)

        return anomaly_flags, norm_anomaly_score.round(4)


class GeospatialContagionClusterer:
    """Uses DBSCAN to cluster spatially contiguous trade corridors experiencing simultaneous drops."""

    def __init__(self, eps_km: float = 1200.0, min_samples: int = 2):
        self.eps_radians = eps_km / 6371.0088 # Earth radius in km
        self.min_samples = min_samples
        self.centroids = get_country_centroids()

    def cluster_anomalies_by_period(self, df_anomalies: pd.DataFrame) -> pd.DataFrame:
        """
        Runs monthly spatial clustering on trade flows flagged as anomalous.
        Assigns cluster ID (-1 for isolated, >=0 for regional contagion cluster).
        """
        logger.info("Evaluating spatial contagion clustering via DBSCAN...")
        df_out = df_anomalies.copy()
        df_out["spatial_contagion_cluster"] = -1

        for period, grp in df_out.groupby("period"):
            anom_subset = grp[grp["is_trade_anomaly"] == 1]
            if len(anom_subset) < self.min_samples:
                continue

            coords = []
            valid_indices = []
            for idx, row in anom_subset.iterrows():
                rep = row["reporter_iso"]
                part = row["partner_iso"]
                rep_coord = self.centroids.get(rep, (0.0, 0.0))
                part_coord = self.centroids.get(part, (0.0, 0.0))

                # Midpoint of trade corridor
                mid_lat = np.radians((rep_coord[0] + part_coord[0]) / 2.0)
                mid_lon = np.radians((rep_coord[1] + part_coord[1]) / 2.0)
                coords.append([mid_lat, mid_lon])
                valid_indices.append(idx)

            if len(coords) >= self.min_samples:
                db = DBSCAN(eps=self.eps_radians, min_samples=self.min_samples, metric="haversine")
                cluster_labels = db.fit_predict(coords)
                df_out.loc[valid_indices, "spatial_contagion_cluster"] = cluster_labels

        logger.info(f"Spatial clustering complete. Found {len(df_out[df_out['spatial_contagion_cluster'] >= 0])} clustered corridor shocks.")
        return df_out


def run_anomaly_detection(
    input_path: str = "data/processed/trade_harmonized_monthly.parquet",
    output_path: str = "data/processed/trade_anomalies_detected.parquet",
    model_save_path: str = "models/isolation_forest_trade.pkl"
) -> str:
    """Executes full feature engineering, anomaly detection, and spatial clustering."""
    config = load_config()
    ensure_directory(os.path.dirname(output_path))
    ensure_directory(os.path.dirname(model_save_path))

    if not os.path.exists(input_path):
        logger.warning(f"Input file {input_path} missing. Running preprocessing first.")
        from scripts.preprocessing import run_preprocessing
        run_preprocessing()

    logger.info(f"Loading harmonized trade series: {input_path}")
    df_trade = pd.read_parquet(input_path)

    # 1. Feature Engineering
    engineer = TradeFeatureEngineer()
    df_feat = engineer.engineer_features(df_trade)

    # 2. Isolation Forest Modeling
    detector = IsolationForestDetector(config)
    detector.tune_and_fit(df_feat)
    flags, scores = detector.predict_anomaly_scores(df_feat)

    df_feat["is_trade_anomaly"] = flags
    df_feat["trade_anomaly_score"] = scores

    # Save model
    joblib.dump({"model": detector.model, "scaler": detector.scaler, "features": detector.features}, model_save_path)
    logger.info(f"Saved Isolation Forest model to {model_save_path}")

    # 3. Geospatial Contagion Clustering
    eps_km = config["anomaly_detection"]["dbscan"].get("eps_km", 1200.0)
    min_samples = config["anomaly_detection"]["dbscan"].get("min_samples", 2)
    spatial_clusterer = GeospatialContagionClusterer(eps_km=eps_km, min_samples=min_samples)
    df_result = spatial_clusterer.cluster_anomalies_by_period(df_feat)

    # Output results
    df_result.to_parquet(output_path, index=False)
    logger.info(f"Saved anomaly detection results to: {output_path}")

    anom_count = df_result["is_trade_anomaly"].sum()
    total_count = len(df_result)
    logger.info(f"Identified {anom_count} anomalous trade observations out of {total_count} ({anom_count/total_count:.2%}).")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trade Volume Anomaly Detection")
    parser.add_argument("--input", type=str, default="data/processed/trade_harmonized_monthly.parquet")
    parser.add_argument("--output", type=str, default="data/processed/trade_anomalies_detected.parquet")
    parser.add_argument("--model-out", type=str, default="models/isolation_forest_trade.pkl")
    args = parser.parse_args()

    run_anomaly_detection(args.input, args.output, args.model_out)
