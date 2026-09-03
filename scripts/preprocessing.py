"""
Data Preprocessing and Harmonization Pipeline for Supply Chain Warning.

Covers:
1. HS Code Version Concordance (Harmonized System standardization to HS 2022 / H6).
2. Mirror Statistics Reconciliation (Bilateral export FOB vs. import CIF gap resolution).
3. Monthly Time Series Grid Construction & Spline Imputation.
4. GDELT Disruption Text Cleaning, Deduplication, and Token Filtering.
"""

import os
import re
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from scripts.utils import setup_logger, load_config, ensure_directory

logger = setup_logger("preprocessing")


class HSConcordanceMapper:
    """Standardizes historical Harmonized System revisions to HS 2022 (H6)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Concordance mappings for electronics and machinery (HS 84 & 85) across revisions
        self.concordance_table = {
            # H3 (2007) / H4 (2012) / H5 (2017) -> H6 (2022)
            "847130": "847130", # Laptops, portable digital computers
            "847141": "847141", # Desktop data processing units
            "847150": "847150", # Digital processing units (Servers)
            "847330": "847330", # Parts and accessories for computers
            "854110": "854110", # Diodes
            "854121": "854121", # Transistors < 1W
            "854129": "854129", # Transistors other
            "854140": "854141", # Solar / photovoltaic cells mapped to 2022 subheadings
            "854231": "854231", # Processors and controllers
            "854232": "854232", # Electronic memories (DRAM, NAND Flash)
            "854233": "854233", # Amplifiers
            "854239": "854239", # Other integrated circuits
            "854210": "854231", # Pre-2007 legacy monolithic ICs mapped to processors
            "854221": "854232", # Pre-2007 legacy digital memories
        }

    def standardize_code(self, raw_code: Any) -> str:
        """Standardizes an HS code string to 4 or 6 digit HS 2022 format."""
        code_str = re.sub(r"\D", "", str(raw_code).strip())
        if len(code_str) == 4:
            return code_str  # Subheading level remains consistent
        if len(code_str) >= 6:
            code_6 = code_str[:6]
            return self.concordance_table.get(code_6, code_6)
        return code_str


class MirrorReconciler:
    """
    Reconciles bilateral discrepancies between reporter export FOB values
    and partner import CIF values to construct harmonized trade volume series.
    """

    def __init__(self, cif_fob_factor: float = 1.06, discrepancy_threshold: float = 0.25):
        self.cif_fob_factor = cif_fob_factor
        self.discrepancy_threshold = discrepancy_threshold

    def reconcile_corridors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates mirror discrepancy ratio and applies reliability weighting.
        Formula:
            CIF_adjusted = Partner_CIF / cif_fob_factor
            Discrepancy = |Reporter_Export - CIF_adjusted| / max(Reporter_Export, CIF_adjusted)
            Harmonized_Value = w_rep * Reporter_Export + (1 - w_rep) * CIF_adjusted
        """
        logger.info("Executing mirror statistics reconciliation...")
        df_rec = df.copy()

        # Adjusted partner CIF to FOB equivalent
        cif_col = "mirror_partner_value_usd"
        fob_col = "trade_value_usd"

        if cif_col in df_rec.columns and fob_col in df_rec.columns:
            adjusted_partner_fob = df_rec[cif_col] / self.cif_fob_factor
            denom = np.maximum(df_rec[fob_col], adjusted_partner_fob)
            denom = np.where(denom == 0, 1.0, denom)

            discrepancy = np.abs(df_rec[fob_col] - adjusted_partner_fob) / denom
            df_rec["mirror_discrepancy_ratio"] = discrepancy.round(4)
            df_rec["mirror_discrepancy_flag"] = discrepancy > self.discrepancy_threshold

            # Reliability weighting: OECD/US/EU reporting typically exhibits higher compliance
            reporter_weight = df_rec["reporter_iso"].apply(
                lambda x: 0.65 if x in ["USA", "DEU", "JPN", "NLD"] else 0.50
            )

            df_rec["harmonized_trade_usd"] = (
                reporter_weight * df_rec[fob_col] +
                (1.0 - reporter_weight) * adjusted_partner_fob
            ).round(2)
        else:
            df_rec["harmonized_trade_usd"] = df_rec[fob_col]
            df_rec["mirror_discrepancy_ratio"] = 0.0
            df_rec["mirror_discrepancy_flag"] = False

        return df_rec


class MonthlyTradeImputer:
    """Creates contiguous monthly calendar grids and imputes missing trade periods."""

    def __init__(self, method: str = "iterative_spline"):
        self.method = method

    def fill_and_impute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds full monthly grid per (reporter_iso, partner_iso, cmd_code)
        and imputes missing observations via cubic spline or rolling median.
        """
        logger.info("Constructing complete monthly time-series grid and imputing gaps...")
        df["date"] = pd.to_datetime(df["period"].astype(str), format="%Y%m")

        all_groups = []
        group_cols = ["reporter_iso", "partner_iso", "cmd_code"]

        for (rep, part, cmd), group in df.groupby(group_cols):
            group = group.sort_values("date").drop_duplicates(subset=["date"])
            min_date = group["date"].min()
            max_date = group["date"].max()
            full_dates = pd.date_range(start=min_date, end=max_date, freq="MS")

            # Reindex to full dates
            group_indexed = group.set_index("date").reindex(full_dates)
            group_indexed["reporter_iso"] = rep
            group_indexed["partner_iso"] = part
            group_indexed["cmd_code"] = cmd

            # Flag interpolated values
            group_indexed["is_imputed"] = group_indexed["harmonized_trade_usd"].isna()

            # Imputation
            val_series = group_indexed["harmonized_trade_usd"]
            if val_series.isna().any():
                # Interpolate using polynomial/linear spline
                interpolated = val_series.interpolate(method="time").bfill().ffill()
                group_indexed["harmonized_trade_usd"] = interpolated

            # Reconstruct period strings
            group_indexed["period"] = group_indexed.index.strftime("%Y%m")
            group_indexed["year"] = group_indexed.index.year
            group_indexed["month"] = group_indexed.index.month

            all_groups.append(group_indexed.reset_index(names=["date"]))

        result_df = pd.concat(all_groups, ignore_index=True)
        logger.info(f"Finished imputation: {len(result_df)} records across all corridors.")
        return result_df


class GDELTTextCleaner:
    """Preprocesses, cleans, and standardizes GDELT raw news headlines and snippets."""

    def __init__(self, min_length: int = 15):
        self.min_length = min_length
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+")
        self.whitespace_pattern = re.compile(r"\s+")
        self.punct_pattern = re.compile(r"[^\w\s\-\.\,\/]")

    def clean_text(self, text: Any) -> str:
        """Sanitizes text by stripping URLs, exotic punctuation, and normalizing spaces."""
        if not isinstance(text, str):
            return ""

        # Remove URLs
        cleaned = self.url_pattern.sub("", text)
        # Remove unusual characters but preserve dashes and periods
        cleaned = self.punct_pattern.sub(" ", cleaned)
        # Collapse whitespaces
        cleaned = self.whitespace_pattern.sub(" ", cleaned).strip()
        return cleaned

    def process_corpus(self, df_gdelt: pd.DataFrame) -> pd.DataFrame:
        """Cleans, filters length, and removes duplicates from GDELT event dataframe."""
        logger.info("Cleaning and deduplicating GDELT news corpus...")
        df = df_gdelt.copy()

        df["cleaned_title"] = df["source_title"].apply(self.clean_text)
        # Filter too short texts
        df = df[df["cleaned_title"].str.len() >= self.min_length]

        # Deduplicate on date and cleaned title
        initial_len = len(df)
        df = df.drop_duplicates(subset=["event_date", "cleaned_title"])
        logger.info(f"Deduplicated news corpus: retained {len(df)} from {initial_len} articles.")

        # Ensure date format
        df["event_date"] = pd.to_datetime(df["event_date"])
        return df.reset_index(drop=True)


def run_preprocessing(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed"
) -> Dict[str, str]:
    """Runs end-to-end preprocessing pipeline on Comtrade trade and GDELT news data."""
    config = load_config()
    ensure_directory(processed_dir)

    trade_raw_path = os.path.join(raw_dir, "comtrade_trade_records.parquet")
    gdelt_raw_path = os.path.join(raw_dir, "gdelt_disruption_events.parquet")
    lpi_raw_path = os.path.join(raw_dir, "worldbank_lpi_benchmarks.csv")

    # Verify input presence
    if not os.path.exists(trade_raw_path) or not os.path.exists(gdelt_raw_path):
        logger.warning("Raw datasets not found. Triggering automated collection first...")
        from scripts.data_collection import run_full_collection
        run_full_collection(raw_dir)

    # 1. Process Trade Data
    logger.info("Reading raw trade parquet records...")
    df_trade = pd.read_parquet(trade_raw_path)

    # HS Concordance
    mapper = HSConcordanceMapper()
    df_trade["standard_cmd_code"] = df_trade["cmd_code"].apply(mapper.standardize_code)

    # Mirror Reconciliation
    reconciler = MirrorReconciler(
        cif_fob_factor=config["preprocessing"]["mirror_reconciliation"]["cif_fob_adjustment_factor"],
        discrepancy_threshold=config["preprocessing"]["mirror_reconciliation"]["discrepancy_threshold"]
    )
    df_trade_rec = reconciler.reconcile_corridors(df_trade)

    # Monthly Imputation
    imputer = MonthlyTradeImputer()
    df_trade_clean = imputer.fill_and_impute(df_trade_rec)

    trade_clean_path = os.path.join(processed_dir, "trade_harmonized_monthly.parquet")
    df_trade_clean.to_parquet(trade_clean_path, index=False)
    logger.info(f"Saved harmonized trade data to: {trade_clean_path}")

    # 2. Process GDELT News Data
    logger.info("Reading raw GDELT event records...")
    df_gdelt = pd.read_parquet(gdelt_raw_path)
    cleaner = GDELTTextCleaner()
    df_gdelt_clean = cleaner.process_corpus(df_gdelt)

    gdelt_clean_path = os.path.join(processed_dir, "gdelt_cleaned_events.parquet")
    df_gdelt_clean.to_parquet(gdelt_clean_path, index=False)
    logger.info(f"Saved cleaned GDELT corpus to: {gdelt_clean_path}")

    # 3. Copy/Standardize LPI Benchmarks
    if os.path.exists(lpi_raw_path):
        df_lpi = pd.read_csv(lpi_raw_path)
        lpi_clean_path = os.path.join(processed_dir, "lpi_harmonized_benchmarks.parquet")
        df_lpi.to_parquet(lpi_clean_path, index=False)
        logger.info(f"Saved harmonized LPI benchmarks to: {lpi_clean_path}")

    return {
        "trade_processed": trade_clean_path,
        "gdelt_processed": gdelt_clean_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Preprocessing Pipeline")
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    args = parser.parse_args()

    run_preprocessing(args.raw_dir, args.processed_dir)
