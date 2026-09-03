"""
Topic Modeling for Supply Chain Disruption Narratives.

Implements:
1. BERTopic Disruption Theme Extraction using Transformer Embeddings & c-TF-IDF.
2. Domain Theme Classification (Ports, Semiconductors, Strikes, Sanctions, Weather).
3. Dynamic Topic Prevalence Tracking over time.
4. Fallback TF-IDF + Spectral/KMeans Clustering for resilient local execution.
"""

import os
import re
import joblib
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from scripts.utils import setup_logger, load_config, ensure_directory

logger = setup_logger("topic_modeling")


class DisruptionTopicModeler:
    """Discovers and tracks latent supply chain disruption themes across news corpora."""

    THEME_MAPPINGS = {
        "PORT_CONGESTION": ["port", "container", "vessel", "berth", "terminal", "cargo", "ship", "freight", "rotterdam", "shanghai"],
        "SEMICONDUCTOR_SHORTAGE": ["semiconductor", "chip", "wafer", "integrated", "processor", "intel", "tsmc", "shortage", "silicon"],
        "LABOR_STRIKE": ["strike", "dockworkers", "union", "walkout", "labor", "workers", "wage", "protest", "stoppage"],
        "GEOPOLITICAL_SANCTION": ["sanction", "export", "embargo", "ban", "restriction", "tariff", "bilateral", "security", "customs"],
        "INFRASTRUCTURE_ACCIDENT": ["canal", "blockage", "ground", "suez", "typhoon", "storm", "collision", "damage", "rail"]
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config["topic_modeling"]["bertopic"]
        self.nr_topics = self.config.get("nr_topics", 10)
        self.embedding_model_name = self.config.get("embedding_model", "all-MiniLM-L6-v2")
        self.model = None
        self.vectorizer = None
        self.is_bertopic_native = False

    def _init_native_bertopic(self):
        """Attempts to initialize native BERTopic model with UMAP and HDBSCAN."""
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer

            vectorizer = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))
            topic_model = BERTopic(
                embedding_model=self.embedding_model_name,
                nr_topics=self.nr_topics,
                vectorizer_model=vectorizer,
                calculate_probabilities=True,
                verbose=False
            )
            self.model = topic_model
            self.is_bertopic_native = True
            logger.info("Initialized native BERTopic transformer pipeline.")
            return True
        except (ImportError, Exception) as e:
            logger.warning(f"Native BERTopic not available ({e}). Initializing scikit-learn TF-IDF topic modeler.")
            self.is_bertopic_native = False
            return False

    def fit_transform(self, texts: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Fits topic model and returns topic assignments and probability distributions.
        """
        if self.model is None:
            self._init_native_bertopic()

        if self.is_bertopic_native and self.model is not None:
            try:
                topics, probs = self.model.fit_transform(texts)
                logger.info(f"Fitted BERTopic on {len(texts)} documents.")
                return topics, np.array(probs)
            except Exception as e:
                logger.warning(f"BERTopic fit_transform failed: {e}. Switching to TF-IDF fallback.")
                self.is_bertopic_native = False

        # Fallback: TF-IDF + MiniBatchKMeans for rapid robust clustering
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.cluster import MiniBatchKMeans

        self.vectorizer = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(texts)

        kmeans = MiniBatchKMeans(n_clusters=self.nr_topics, random_state=42, batch_size=256)
        topics = kmeans.fit_predict(X).tolist()
        distances = kmeans.transform(X)
        # Convert distances to pseudo-probabilities via softmax
        exp_dist = np.exp(-distances)
        probs = exp_dist / np.sum(exp_dist, axis=1, keepdims=True)

        self.model = kmeans
        logger.info(f"Fitted TF-IDF KMeans topic model ({self.nr_topics} clusters) on {len(texts)} documents.")
        return topics, probs

    def get_topic_themes(self, texts: List[str], topics: List[int]) -> Dict[int, str]:
        """
        Extracts representative keywords per topic and maps to high-level supply chain disruption themes.
        """
        df_docs = pd.DataFrame({"text": texts, "topic": topics})
        theme_map = {}

        for topic_id, grp in df_docs.groupby("topic"):
            all_text = " ".join(grp["text"].tolist()).lower()
            words = re.findall(r"\b[a-z]{3,}\b", all_text)
            counts = pd.Series(words).value_counts()
            top_words = set(counts.head(15).index.tolist())

            # Score each theme based on keyword overlap
            theme_scores = {}
            for theme, keywords in self.THEME_MAPPINGS.items():
                overlap = len(top_words.intersection(set(keywords)))
                theme_scores[theme] = overlap

            best_theme = max(theme_scores, key=theme_scores.get)
            if theme_scores[best_theme] == 0:
                best_theme = "GENERAL_LOGISTICS_VOLATILITY"

            theme_map[topic_id] = best_theme

        return theme_map

    def save_model(self, output_path: str):
        """Serializes the trained topic model to disk."""
        ensure_directory(os.path.dirname(output_path))
        payload = {
            "model": self.model,
            "vectorizer": self.vectorizer,
            "is_bertopic_native": self.is_bertopic_native,
            "nr_topics": self.nr_topics
        }
        joblib.dump(payload, output_path)
        logger.info(f"Saved topic model checkpoint to: {output_path}")

    @classmethod
    def load_model(cls, model_path: str) -> "DisruptionTopicModeler":
        """Loads a serialized topic model from disk."""
        instance = cls(load_config())
        payload = joblib.load(model_path)
        instance.model = payload["model"]
        instance.vectorizer = payload.get("vectorizer")
        instance.is_bertopic_native = payload.get("is_bertopic_native", False)
        instance.nr_topics = payload.get("nr_topics", 10)
        return instance


def run_topic_modeling(
    input_path: str = "data/processed/gdelt_nlp_enriched.parquet",
    output_path: str = "data/processed/gdelt_topics_assigned.parquet",
    model_save_path: str = "models/disruption_topic_model.pkl"
) -> str:
    """Executes topic discovery and theme assignment across the news dataset."""
    config = load_config()
    ensure_directory(os.path.dirname(output_path))
    ensure_directory(os.path.dirname(model_save_path))

    if not os.path.exists(input_path):
        logger.warning(f"Input file {input_path} missing. Running NLP enrichment first.")
        from scripts.nlp_pipeline import run_nlp_pipeline
        run_nlp_pipeline()

    logger.info(f"Loading NLP enriched dataset from: {input_path}")
    df = pd.read_parquet(input_path)

    texts = df["topic_clean_text"].fillna(df["cleaned_title"]).tolist()

    modeler = DisruptionTopicModeler(config)
    topics, probs = modeler.fit_transform(texts)

    df["topic_id"] = topics
    df["topic_probability"] = np.max(probs, axis=1).round(4) if probs.ndim == 2 else 0.85

    # Map topics to domain disruption themes
    theme_map = modeler.get_topic_themes(texts, topics)
    df["disruption_theme"] = df["topic_id"].map(theme_map)

    # Save model checkpoint
    modeler.save_model(model_save_path)

    # Save enriched topic dataframe
    df.to_parquet(output_path, index=False)
    logger.info(f"Topic modeling completed. Assigned {len(set(topics))} unique topics. Saved to {output_path}")

    # Log topic distribution
    theme_counts = df["disruption_theme"].value_counts()
    logger.info("Discovered Disruption Themes Distribution:\n" + str(theme_counts))

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Topic Modeling Pipeline")
    parser.add_argument("--input", type=str, default="data/processed/gdelt_nlp_enriched.parquet")
    parser.add_argument("--output", type=str, default="data/processed/gdelt_topics_assigned.parquet")
    parser.add_argument("--model-out", type=str, default="models/disruption_topic_model.pkl")
    args = parser.parse_args()

    run_topic_modeling(args.input, args.output, args.model_out)
