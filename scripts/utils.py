"""
Utility functions, logging configuration, geospatial coordinate lookups,
and metric calculation tools for the Supply Chain Disruption Warning System.
"""

import os
import sys
import math
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import yaml
import numpy as np
import pandas as pd


def setup_logger(
    name: str = "supply_chain_warning",
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configures and returns a structured logger with console and optional file handlers.

    Parameters:
        name (str): Name of the logger instance.
        log_file (Optional[str]): Path to output log file.
        level (int): Logging level (default INFO).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] (%(name)s) %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration parameters from disk.

    Parameters:
        config_path (str): Relative or absolute path to config.yaml.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    path = Path(config_path)
    if not path.exists():
        # Search relative to current script directory
        alt_path = Path(__file__).resolve().parent.parent / config_path
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def ensure_directory(dir_path: str) -> Path:
    """
    Ensures a directory exists, creating all parent directories if necessary.

    Parameters:
        dir_path (str): Path to directory.

    Returns:
        Path: Path object for created directory.
    """
    p = Path(dir_path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates great-circle distance between two points on the Earth in kilometers.

    Parameters:
        lat1, lon1: Coordinates of point 1 in decimal degrees.
        lat2, lon2: Coordinates of point 2 in decimal degrees.

    Returns:
        float: Distance in kilometers.
    """
    r_earth_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth_km * c


def get_country_centroids() -> Dict[str, Tuple[float, float]]:
    """
    Returns mapping of ISO3 country codes to geographical centroids (latitude, longitude).
    Used for geospatial anomaly projection and DBSCAN distance calculations.

    Returns:
        Dict[str, Tuple[float, float]]: ISO3 to (lat, lon) dictionary.
    """
    return {
        "USA": (37.0902, -95.7129),
        "CHN": (35.8617, 104.1954),
        "DEU": (51.1657, 10.4515),
        "JPN": (36.2048, 138.2529),
        "KOR": (35.9078, 127.7669),
        "TWN": (23.6978, 120.9605),
        "NLD": (52.1326, 5.2913),
        "SGP": (1.3521, 103.8198),
        "MYS": (4.2105, 101.9758),
        "VNM": (14.0583, 108.2772),
        "MEX": (23.6345, -102.5528),
        "CAN": (56.1304, -106.3468),
        "GBR": (55.3781, -3.4360),
        "FRA": (46.2276, 2.2137),
        "ITA": (41.8719, 12.5674),
        "IND": (20.5937, 78.9629),
        "EGY": (26.8206, 30.8025), # Suez Canal corridor
        "PAN": (8.5379, -80.7821),  # Panama Canal corridor
    }


def get_port_chokepoints() -> Dict[str, Dict[str, Any]]:
    """
    Returns catalog of strategic global maritime and multimodal logistics choke points.

    Returns:
        Dict[str, Dict[str, Any]]: Choke point metadata.
    """
    return {
        "SUEZ_CANAL": {
            "name": "Suez Canal Transit",
            "lat": 30.7050,
            "lon": 32.3442,
            "country": "EGY",
            "annual_teu_capacity_millions": 26.0,
            "vulnerability_weight": 0.95
        },
        "PANAMA_CANAL": {
            "name": "Panama Canal Locks",
            "lat": 9.0800,
            "lon": -79.6800,
            "country": "PAN",
            "annual_teu_capacity_millions": 14.0,
            "vulnerability_weight": 0.85
        },
        "PORT_SHANGHAI": {
            "name": "Port of Shanghai (Yangshan)",
            "lat": 30.6272,
            "lon": 122.0645,
            "country": "CHN",
            "annual_teu_capacity_millions": 47.3,
            "vulnerability_weight": 0.90
        },
        "PORT_LA_LONG_BEACH": {
            "name": "Ports of Los Angeles & Long Beach",
            "lat": 33.7432,
            "lon": -118.2673,
            "country": "USA",
            "annual_teu_capacity_millions": 19.5,
            "vulnerability_weight": 0.88
        },
        "PORT_ROTTERDAM": {
            "name": "Port of Rotterdam",
            "lat": 51.9567,
            "lon": 4.1486,
            "country": "NLD",
            "annual_teu_capacity_millions": 14.5,
            "vulnerability_weight": 0.82
        },
        "STRAIT_OF_MALACCA": {
            "name": "Strait of Malacca",
            "lat": 1.4300,
            "lon": 102.8900,
            "country": "SGP",
            "annual_teu_capacity_millions": 85.0,
            "vulnerability_weight": 0.92
        }
    }


def compute_early_warning_metrics(
    actual_disruptions: pd.DataFrame,
    detected_alerts: pd.DataFrame,
    tolerance_window_days: int = 45
) -> Dict[str, float]:
    """
    Evaluates early warning alerting accuracy against ground truth historical disruptions.

    Parameters:
        actual_disruptions (pd.DataFrame): DataFrame containing ['event_name', 'start_date', 'corridor'].
        detected_alerts (pd.DataFrame): DataFrame containing ['alert_date', 'corridor', 'risk_score'].
        tolerance_window_days (int): Maximum lookback days before event to count as true warning.

    Returns:
        Dict[str, float]: Precision, Recall, F1-Score, and Mean Lead Time in days.
    """
    if detected_alerts.empty or actual_disruptions.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_lead_time_days": 0.0}

    true_positives = 0
    lead_times: List[float] = []

    for _, event in actual_disruptions.iterrows():
        ev_date = pd.to_datetime(event["start_date"])
        corridor = event.get("corridor")

        # Find candidate alerts matching corridor within [ev_date - tolerance, ev_date]
        window_start = ev_date - pd.Timedelta(days=tolerance_window_days)
        alerts_subset = detected_alerts[
            (detected_alerts["corridor"] == corridor) &
            (pd.to_datetime(detected_alerts["alert_date"]) >= window_start) &
            (pd.to_datetime(detected_alerts["alert_date"]) <= ev_date)
        ]

        if not alerts_subset.empty:
            true_positives += 1
            earliest_alert = pd.to_datetime(alerts_subset["alert_date"]).min()
            lead_time = (ev_date - earliest_alert).days
            lead_times.append(float(lead_time))

    total_events = len(actual_disruptions)
    total_alerts = len(detected_alerts)

    recall = true_positives / total_events if total_events > 0 else 0.0
    precision = true_positives / total_alerts if total_alerts > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_lead_time = float(np.mean(lead_times)) if lead_times else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "mean_lead_time_days": round(mean_lead_time, 2),
        "true_positives": true_positives,
        "total_actual_events": total_events,
        "total_generated_alerts": total_alerts
    }
