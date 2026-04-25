<div align="center">

# 🏥 MLOps Production Pipeline
### *Hospital Readmission Risk Prediction — End-to-End MLOps*

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.15-0194E2?style=for-the-badge&logo=mlflow)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)](https://docker.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-006400?style=for-the-badge)](https://xgboost.readthedocs.io)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/muzzamilmustafa0-cyber/mlops-production-pipeline/ci-cd.yml?style=for-the-badge&label=CI%2FCD)](https://github.com/muzzamilmustafa0-cyber/mlops-production-pipeline/actions)

**A production-grade MLOps pipeline predicting 30-day hospital readmission risk — with experiment tracking, model registry, REST API, and automated CI/CD.**

[🚀 Quick Start](#-quick-start) • [📊 Results](#-results) • [🏗️ Architecture](#-architecture) • [🔌 API Docs](#-api-reference)

</div>

---

## 🎯 Problem Statement

Hospital readmission within 30 days is a major quality indicator and cost driver in healthcare. This project builds a **production ML system** that:
- Predicts readmission risk from patient admission features
- Benchmarks 4 models with full MLflow tracking
- Serves predictions via a REST API with <5ms inference latency
- Ships with Docker and automated CI/CD

---

## 📊 Results

| Model | ROC-AUC | F1 (weighted) | Accuracy |
|---|---|---|---|
| **XGBoost** ⭐ | **0.6821** | **0.6543** | **0.6410** |
| GradientBoosting | 0.6754 | 0.6488 | 0.6378 |
| RandomForest | 0.6612 | 0.6321 | 0.6201 |
| LogisticRegression | 0.6287 | 0.5987 | 0.5904 |

*Dataset: UCI Diabetes 130-US hospitals (101,766 patients)*

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                          │
│                                                              │
│  CSV Data → Feature Engineering → Train/Test Split          │
│      ↓                                                       │
│  4 Models (LR, RF, GB, XGB) → MLflow Experiment Tracking    │
│      ↓                                                       │
│  Best Model → MLflow Model Registry → artifacts/            │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│                    SERVING PIPELINE                           │
│                                                              │
│  MLflow Registry / Local .pkl                                │
│         ↓                                                    │
│  FastAPI  →  /predict (single)  →  Risk Label + Probability  │
│          →  /predict/batch      →  1000 patients <50ms       │
│          →  /health + /model/info                           │
└──────────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────────┐
│                      CI/CD (GitHub Actions)                   │
│  Push → pytest → black/flake8 → docker build → Docker Hub   │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/muzzamilmustafa0-cyber/mlops-production-pipeline.git
cd mlops-production-pipeline
pip install -r requirements.txt

# Train all models (downloads UCI dataset automatically)
python src/train.py --data data/diabetic_data.csv

# View MLflow dashboard
mlflow ui --port 5000   # Open http://localhost:5000

# Start prediction API
uvicorn api.app:app --reload --port 8000

# Run tests
pytest tests/ -v
```

---

## 🔌 API Reference

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "time_in_hospital": 5,
    "num_lab_procedures": 44,
    "num_medications": 12,
    "number_diagnoses": 9,
    "age_encoded": 65.0
  }'

# Response:
# {
#   "readmission_risk": "MEDIUM",
#   "probability_readmitted": 0.4821,
#   "recommendation": "⚡ Moderate risk. Schedule follow-up within 2 weeks.",
#   "inference_time_ms": 1.24
# }
```

---

## 🐳 Docker

```bash
docker build -t readmission-api .
docker run -p 8000:8000 readmission-api

# Or with docker-compose
docker compose up
```

---

## 📁 Structure

```
mlops-production-pipeline/
├── src/
│   ├── preprocess.py    # Feature engineering + train/test split
│   └── train.py         # MLflow training pipeline (4 models)
├── api/
│   └── app.py           # FastAPI inference service
├── tests/
│   └── test_api.py      # API unit tests (pytest + httpx)
├── .github/workflows/
│   └── ci-cd.yml        # Test → lint → Docker Hub push
├── Dockerfile
└── requirements.txt
```

---

## 👨‍💻 Author

**Muzzamil Mustafa** — AI Researcher & ML Engineer
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github)](https://github.com/muzzamilmustafa0-cyber)

</div>
