"""
Natural Language Processing Pipeline for Supply Chain Intelligence.

Performs:
1. Named Entity Recognition (NER) with spaCy: Country (GPE), Port/Chokepoint (LOC), Logistics Carriers (ORG), Commodities.
2. Sentiment Intensity Scoring via Hugging Face Transformers.
3. Lemmatized and Filtered Text Tokenization for downstream Topic Modeling.
"""

import os
import re
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

from scripts.utils import setup_logger, load_config, ensure_directory

logger = setup_logger("nlp_pipeline")


class SpacyEntityExtractor:
    """Extracts geopolitical, facility, corporate, and commodity entities using spaCy and regex patterns."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = self._load_spacy(spacy_model)
        self._init_custom_patterns()

    def _load_spacy(self, model_name: str):
        try:
            import spacy
            try:
                nlp = spacy.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
                return nlp
            except OSError:
                logger.warning(f"spaCy model {model_name} not found. Loading blank English pipeline.")
                return spacy.blank("en")
        except ImportError:
            logger.warning("spaCy library not installed. Falling back to rule-based pattern extractor.")
            return None

    def _init_custom_patterns(self):
        """Initializes regex gazetteers for supply chain domain entities."""
        self.commodity_patterns = [
            re.compile(r"\b(semiconductor|microchip|silicon wafer|integrated circuit|chip|cpu|gpu|dram|flash memory)\b", re.IGNORECASE),
            re.compile(r"\b(auto parts|machinery|boiler|lithography|optical equipment|printed circuit board|pcb)\b", re.IGNORECASE)
        ]
        self.facility_patterns = [
            re.compile(r"\b(port of [a-z\s]+|suez canal|panama canal|strait of malacca|strait of hormuz|terminal|berth)\b", re.IGNORECASE)
        ]
        self.carrier_patterns = [
            re.compile(r"\b(maersk|msc|cma cgm|cosco|hapag-lloyd|evergreen|one|yang ming|tsmc|foxconn|intel|samsung)\b", re.IGNORECASE)
        ]
        self.iso_country_lookup = {
            "china": "CHN", "chinese": "CHN", "beijing": "CHN", "shanghai": "CHN",
            "united states": "USA", "us": "USA", "american": "USA",
            "germany": "DEU", "german": "DEU",
            "japan": "JPN", "japanese": "JPN", "tokyo": "JPN",
            "south korea": "KOR", "korea": "KOR", "korean": "KOR", "seoul": "KOR",
            "taiwan": "TWN", "taiwanese": "TWN", "taipei": "TWN",
            "netherlands": "NLD", "dutch": "NLD", "rotterdam": "NLD",
            "singapore": "SGP", "malaysia": "MYS", "vietnam": "VNM",
            "india": "IND", "egypt": "EGY", "panama": "PAN"
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts entities from text combining spaCy NER tokens and domain gazetteers.
        """
        countries = set()
        ports_facilities = set()
        carriers = set()
        commodities = set()

        text_lower = text.lower()

        # Regex Gazetteer matches
        for pat in self.commodity_patterns:
            matches = pat.findall(text)
            for m in matches:
                commodities.add(m.lower())

        for pat in self.facility_patterns:
            matches = pat.findall(text)
            for m in matches:
                ports_facilities.add(m.title())

        for pat in self.carrier_patterns:
            matches = pat.findall(text)
            for m in matches:
                carriers.add(m.upper())

        for term, iso in self.iso_country_lookup.items():
            if re.search(rf"\b{re.escape(term)}\b", text_lower):
                countries.add(iso)

        # spaCy model tokens if available
        if self.nlp is not None and hasattr(self.nlp, "pipe_names") and "ner" in self.nlp.pipe_names:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["GPE", "LOC"]:
                    iso = self.iso_country_lookup.get(ent.text.lower())
                    if iso:
                        countries.add(iso)
                    elif "port" in ent.text.lower() or "canal" in ent.text.lower():
                        ports_facilities.add(ent.text.title())
                elif ent.label_ == "ORG":
                    carriers.add(ent.text)

        return {
            "countries": list(countries),
            "facilities": list(ports_facilities),
            "carriers": list(carriers),
            "commodities": list(commodities)
        }


class HuggingFaceSentimentAnalyzer:
    """Quantifies negative sentiment intensity and supply disruption risk tone."""

    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.pipeline = self._init_pipeline()

    def _init_pipeline(self):
        try:
            from transformers import pipeline
            pipe = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1, # CPU inference
                truncation=True,
                max_length=256
            )
            logger.info(f"Loaded Hugging Face sentiment pipeline: {self.model_name}")
            return pipe
        except Exception as e:
            logger.warning(f"Transformers sentiment model could not be initialized: {e}. Using calibrated heuristic.")
            return None

    def score_negative_sentiment(self, texts: List[str]) -> np.ndarray:
        """
        Returns negative sentiment risk intensity between 0.0 (neutral/positive) and 1.0 (extreme disruption).
        """
        if self.pipeline is not None:
            try:
                results = self.pipeline(texts, batch_size=32)
                scores = []
                for res in results:
                    label = res.get("label", "POSITIVE")
                    score = res.get("score", 0.5)
                    # If NEGATIVE, score is between 0.5 and 1.0 -> map to high disruption
                    if label == "NEGATIVE":
                        intensity = 0.5 + 0.5 * score
                    else:
                        intensity = 0.5 * (1.0 - score)
                    scores.append(intensity)
                return np.array(scores, dtype=np.float32)
            except Exception as e:
                logger.warning(f"Batch transformers inference failed: {e}. Falling back to lexicon.")

        # Heuristic disruption sentiment calculation based on high-risk keywords
        disruption_lexicon = {
            "halt": 0.85, "strike": 0.90, "disruption": 0.80, "blockage": 0.95,
            "shortage": 0.85, "bottleneck": 0.75, "delay": 0.65, "congestion": 0.70,
            "embargo": 0.90, "sanction": 0.80, "severe": 0.75, "typhoon": 0.80,
            "backlog": 0.70, "crisis": 0.85, "stranded": 0.75, "surge": 0.60
        }
        scores = []
        for text in texts:
            words = text.lower().split()
            matched_weights = [disruption_lexicon[w] for w in words if w in disruption_lexicon]
            if matched_weights:
                score = min(1.0, float(np.max(matched_weights) + 0.05 * len(matched_weights)))
            else:
                score = 0.20 # Neutral baseline
            scores.append(score)

        return np.array(scores, dtype=np.float32)


class TextTopicPreprocessor:
    """Prepares sanitized, tokenized, and lemmatized texts for BERTopic clustering."""

    def __init__(self):
        self.stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "with",
            "about", "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "from", "up", "down", "of", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all",
            "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
            "don", "should", "now", "said", "says", "reported", "according"
        }

    def preprocess_for_topics(self, text: str) -> str:
        """Strips noise words and normalizes whitespace."""
        tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
        filtered = [t for t in tokens if t not in self.stop_words]
        return " ".join(filtered)


def run_nlp_pipeline(
    input_path: str = "data/processed/gdelt_cleaned_events.parquet",
    output_path: str = "data/processed/gdelt_nlp_enriched.parquet"
) -> str:
    """Runs complete NLP feature engineering on the cleaned GDELT news events."""
    config = load_config()
    ensure_directory(os.path.dirname(output_path))

    if not os.path.exists(input_path):
        logger.warning(f"Input file {input_path} missing. Running preprocessing first.")
        from scripts.preprocessing import run_preprocessing
        run_preprocessing()

    logger.info(f"Loading cleaned news data from: {input_path}")
    df = pd.read_parquet(input_path)

    # 1. Named Entity Extraction
    logger.info("Executing NER extraction for countries, ports, carriers, and commodities...")
    extractor = SpacyEntityExtractor(config["nlp"].get("spacy_model", "en_core_web_sm"))

    all_countries = []
    all_facilities = []
    all_commodities = []
    all_carriers = []

    for title in df["cleaned_title"]:
        ents = extractor.extract_entities(title)
        all_countries.append(",".join(ents["countries"]))
        all_facilities.append(",".join(ents["facilities"]))
        all_commodities.append(",".join(ents["commodities"]))
        all_carriers.append(",".join(ents["carriers"]))

    df["ner_countries"] = all_countries
    df["ner_facilities"] = all_facilities
    df["ner_commodities"] = all_commodities
    df["ner_carriers"] = all_carriers

    # 2. Sentiment Intensity Scoring
    logger.info("Scoring negative disruption sentiment intensity...")
    sentiment_analyzer = HuggingFaceSentimentAnalyzer(config["nlp"]["sentiment"]["model_name"])
    sentiment_scores = sentiment_analyzer.score_negative_sentiment(df["cleaned_title"].tolist())
    df["negative_sentiment_score"] = sentiment_scores.round(4)

    # 3. Topic Prep
    logger.info("Preprocessing text representations for BERTopic...")
    topic_preprocessor = TextTopicPreprocessor()
    df["topic_clean_text"] = df["cleaned_title"].apply(topic_preprocessor.preprocess_for_topics)

    df.to_parquet(output_path, index=False)
    logger.info(f"Enriched {len(df)} articles with NLP features. Saved to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Feature Extraction Pipeline")
    parser.add_argument("--input", type=str, default="data/processed/gdelt_cleaned_events.parquet")
    parser.add_argument("--output", type=str, default="data/processed/gdelt_nlp_enriched.parquet")
    args = parser.parse_args()

    run_nlp_pipeline(args.input, args.output)
