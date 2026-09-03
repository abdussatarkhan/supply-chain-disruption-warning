"""
Signal Fusion and Composite Disruption Risk Scoring Engine.

Integrates:
1. Trade Volume Anomaly Score (Isolation Forest + Shock Features).
2. NLP Sentiment Intensity Score (Transformers DistilBERT).
3. Disruption Topic Relevance & Frequency (BERTopic Theme Signals).
4. World Bank Logistics Performance Index (LPI Structural Vulnerability).

Outputs multi-tier early warning alerts and operational mitigation guidance.
"""

import os
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from scripts.utils import setup_logger, load_config, ensure_directory

logger = setup_logger("signal_fusion")


class CompositeRiskEngine:
    """Combines multi-modal signals into calibrated early warning disruption scores."""

    def __init__(self, config: Dict[str, Any]):
        self.weights = config["signal_fusion"]["weights"]
        self.thresholds = config["signal_fusion"]["alert_thresholds"]

        self.w_trade = self.weights.get("trade_anomaly", 0.35)
        self.w_sentiment = self.weights.get("news_sentiment", 0.25)
        self.w_topic = self.weights.get("topic_relevance", 0.20)
        self.w_lpi = self.weights.get("lpi_vulnerability", 0.20)

    def calculate_lpi_vulnerability(self, df_lpi: pd.DataFrame) -> pd.DataFrame:
        """
        Converts 1-5 LPI scores into a 0.0 to 1.0 structural vulnerability index.
        Lower LPI implies higher baseline supply chain friction.
        Formula: Vulnerability = (5.0 - LPI_overall) / 4.0
        """
        df_out = df_lpi.copy()
        if "lpi_overall" in df_out.columns:
            vuln = (5.0 - df_out["lpi_overall"]) / 4.0
            df_out["lpi_vulnerability_score"] = vuln.clip(0.0, 1.0).round(4)
        else:
            df_out["lpi_vulnerability_score"] = 0.35
        return df_out

    def aggregate_news_signals_by_month(self, df_gdelt: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregates GDELT daily NLP events into monthly country-level sentiment and topic signals.
        """
        logger.info("Aggregating news sentiment and topic signals into monthly dimensions...")
        df = df_gdelt.copy()
        df["period"] = pd.to_datetime(df["event_date"]).dt.strftime("%Y%m")

        # Flag high impact topics
        disruption_themes = ["PORT_CONGESTION", "SEMICONDUCTOR_SHORTAGE", "LABOR_STRIKE", "GEOPOLITICAL_SANCTION", "INFRASTRUCTURE_ACCIDENT"]
        df["is_severe_topic"] = df["disruption_theme"].isin(disruption_themes).astype(int)

        agg = df.groupby(["period", "action_geo_country"]).agg(
            mean_sentiment=("negative_sentiment_score", "mean"),
            max_sentiment=("negative_sentiment_score", "max"),
            topic_intensity=("is_severe_topic", "mean"),
            news_volume=("event_id", "count")
        ).reset_index()

        # Scale news volume to [0.0, 1.0]
        max_vol = agg["news_volume"].max()
        agg["news_volume_score"] = (agg["news_volume"] / (max_vol if max_vol > 0 else 1.0)).clip(0.0, 1.0)

        # Composite news score
        agg["news_risk_component"] = (
            0.60 * agg["mean_sentiment"] + 0.40 * agg["topic_intensity"]
        ).clip(0.0, 1.0).round(4)

        return agg

    def fuse_signals(
        self,
        df_trade: pd.DataFrame,
        df_news_agg: pd.DataFrame,
        df_lpi: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fuses trade anomalies, monthly news signals, and LPI vulnerability into composite risk score.
        """
        logger.info("Executing multi-modal signal fusion algorithm...")
        merged = df_trade.copy()

        # Merge News signals on reporter country and period
        merged = merged.merge(
            df_news_agg,
            how="left",
            left_on=["period", "reporter_iso"],
            right_on=["period", "action_geo_country"]
        )

        # Fill missing news with neutral baseline
        merged["mean_sentiment"] = merged["mean_sentiment"].fillna(0.20)
        merged["topic_intensity"] = merged["topic_intensity"].fillna(0.15)
        merged["news_risk_component"] = merged["news_risk_component"].fillna(0.20)

        # Merge LPI on partner country
        lpi_vuln = self.calculate_lpi_vulnerability(df_lpi)
        merged = merged.merge(
            lpi_vuln[["country_iso", "lpi_vulnerability_score"]],
            how="left",
            left_on="partner_iso",
            right_on="country_iso"
        )
        merged["lpi_vulnerability_score"] = merged["lpi_vulnerability_score"].fillna(0.35)

        # 1. Base Linear Weighted Score
        base_score = (
            self.w_trade * merged["trade_anomaly_score"] +
            self.w_sentiment * merged["mean_sentiment"] +
            self.w_topic * merged["topic_intensity"] +
            self.w_lpi * merged["lpi_vulnerability_score"]
        )

        # 2. Compound Multiplier (Non-linear surge when news narrative confirms physical drop)
        compound_boost = np.where(
            (merged["trade_anomaly_score"] > 0.65) & (merged["mean_sentiment"] > 0.60),
            0.15,
            0.0
        )

        final_risk_score = (base_score + compound_boost).clip(0.0, 1.0).round(4)
        merged["composite_risk_score"] = final_risk_score

        # 3. Alert Tier Classification
        merged["alert_tier"] = merged["composite_risk_score"].apply(self._classify_alert_tier)

        # 4. Recommended Mitigation Actions
        merged["recommended_action"] = merged["alert_tier"].apply(self._get_mitigation_action)

        return merged

    def _classify_alert_tier(self, score: float) -> str:
        if score >= self.thresholds.get("critical", 0.88):
            return "CRITICAL"
        elif score >= self.thresholds.get("high", 0.75):
            return "HIGH"
        elif score >= self.thresholds.get("medium", 0.55):
            return "ELEVATED"
        elif score >= self.thresholds.get("low", 0.35):
            return "MONITORING"
        else:
            return "NORMAL"

    @staticmethod
    def _get_mitigation_action(tier: str) -> str:
        actions = {
            "CRITICAL": "Activate secondary qualified suppliers; reroute sea-freight via alternate ports; release regional safety buffer stocks.",
            "HIGH": "Issue pre-emptive purchase orders; audit Tier-2 semiconductor suppliers; secure dedicated air-cargo block space.",
            "ELEVATED": "Increase shipment tracking telemetry frequency to daily; verify customs clearance documentation at destination.",
            "MONITORING": "Maintain routine corridor telemetry; inspect supplier lead-time variance reports weekly.",
            "NORMAL": "Standard lean inventory operations; regular periodic review schedule."
        }
        return actions.get(tier, "Maintain standard operating procedures.")


def run_signal_fusion(
    trade_path: str = "data/processed/trade_anomalies_detected.parquet",
    gdelt_path: str = "data/processed/gdelt_topics_assigned.parquet",
    lpi_path: str = "data/processed/lpi_harmonized_benchmarks.parquet",
    output_path: str = "data/processed/composite_risk_scores.parquet",
    alerts_csv_path: str = "data/processed/active_alerts.csv"
) -> str:
    """Executes multi-modal fusion across trade, news, and logistics datasets."""
    config = load_config()
    ensure_directory(os.path.dirname(output_path))

    # Verify input availability
    if not os.path.exists(trade_path):
        logger.warning("Trade anomaly file missing. Running anomaly detection first.")
        from scripts.anomaly_detection import run_anomaly_detection
        run_anomaly_detection()

    if not os.path.exists(gdelt_path):
        logger.warning("GDELT topic file missing. Running topic modeling first.")
        from scripts.topic_modeling import run_topic_modeling
        run_topic_modeling()

    logger.info("Loading inputs for signal fusion...")
    df_trade = pd.read_parquet(trade_path)
    df_gdelt = pd.read_parquet(gdelt_path)

    if os.path.exists(lpi_path):
        df_lpi = pd.read_parquet(lpi_path)
    else:
        from scripts.data_collection import WorldBankLPIFetcher
        df_lpi = WorldBankLPIFetcher(config).generate_synthetic_lpi_benchmarks()

    engine = CompositeRiskEngine(config)
    df_news_agg = engine.aggregate_news_signals_by_month(df_gdelt)
    df_fused = engine.fuse_signals(df_trade, df_news_agg, df_lpi)

    # Save full parquet
    df_fused.to_parquet(output_path, index=False)
    logger.info(f"Composite risk scores generated for {len(df_fused)} corridor-periods. Saved to {output_path}")

    # Extract and save active critical/high alerts
    high_alerts = df_fused[df_fused["alert_tier"].isin(["CRITICAL", "HIGH"])].sort_values(
        "composite_risk_score", ascending=False
    )
    high_alerts.to_csv(alerts_csv_path, index=False)
    logger.info(f"Exported {len(high_alerts)} high-severity alerts to {alerts_csv_path}")

    # Log tier distribution
    tier_counts = df_fused["alert_tier"].value_counts()
    logger.info("Alert Tier Distribution across portfolio:\n" + str(tier_counts))

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal Signal Fusion Pipeline")
    parser.add_argument("--trade", type=str, default="data/processed/trade_anomalies_detected.parquet")
    parser.add_argument("--gdelt", type=str, default="data/processed/gdelt_topics_assigned.parquet")
    parser.add_argument("--lpi", type=str, default="data/processed/lpi_harmonized_benchmarks.parquet")
    parser.add_argument("--output", type=str, default="data/processed/composite_risk_scores.parquet")
    args = parser.parse_args()

    run_signal_fusion(args.trade, args.gdelt, args.lpi, args.output)
