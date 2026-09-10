# Multimodal Supply Chain Disruption Early-Warning System

[![Python](https://img.shields.io/badge/Python-Logistics_AI-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/) [![Machine Learning](https://img.shields.io/badge/ML-Signal_Fusion-00C896?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **An early-warning forecasting model fusing maritime port telemetry, shipping bill of ladings, and global news NLP risk sentiment to forecast logistics bottlenecks up to 30 days in advance.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    AIS[Port Telemetry & AIS Maritime Feeds] --> Fusion[Multimodal Signal Fusion Layer]
    News[Global Shipping News & NLP Sentiment] --> Fusion
    Fusion --> Anomaly[Isolation Forest & Gradient Boosted Classifier]
    Anomaly --> Alert[Early-Warning Bottleneck Alert Matrix]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `NLP` `Scikit-Learn` `Pandas` `Anomaly Detection`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![supply-chain-disruption-warning preview](images/multimodal_fusion_roc.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/supply-chain-disruption-warning.git
cd supply-chain-disruption-warning
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.
