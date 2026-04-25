"""
FastAPI inference service for the Hospital Readmission Prediction model.

Endpoints:
  POST /predict        — Predict 30-day readmission risk
  POST /predict/batch  — Batch predictions
  GET  /model/info     — Current model metadata
  GET  /health         — Health + model status
"""
import os
import pickle
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global model state ────────────────────────────────────────────────────────
MODEL = None
MODEL_INFO = {}

FEATURE_ORDER = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    "age_encoded", "admission_type_encoded", "discharge_encoded",
    "total_visits", "medication_ratio", "age_visit_interaction",
]


def load_model():
    """Load model from MLflow registry or local pickle."""
    global MODEL, MODEL_INFO
    model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")

    if mlflow_uri:
        try:
            model_name = os.getenv("MLFLOW_MODEL_NAME", "readmission-randomforest")
            model_version = os.getenv("MLFLOW_MODEL_VERSION", "latest")
            mlflow.set_tracking_uri(mlflow_uri)
            MODEL = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
            MODEL_INFO = {"source": "mlflow", "name": model_name, "version": model_version}
            logger.info(f"Loaded model from MLflow: {model_name}@{model_version}")
            return
        except Exception as e:
            logger.warning(f"MLflow load failed ({e}), falling back to local model")

    if Path(model_path).exists():
        with open(model_path, "rb") as f:
            MODEL = pickle.load(f)
        MODEL_INFO = {"source": "local", "path": model_path}
        logger.info(f"Loaded model from: {model_path}")
    else:
        logger.warning("No model found — using dummy model for demo")
        from sklearn.dummy import DummyClassifier
        MODEL = DummyClassifier(strategy="stratified", random_state=42)
        MODEL.classes_ = np.array([0, 1])
        MODEL._sklearn_version = "demo"
        MODEL_INFO = {"source": "demo", "note": "Train a model first with src/train.py"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Hospital Readmission Prediction API",
    description="🏥 ML-powered 30-day hospital readmission risk scoring with MLflow model registry.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ───────────────────────────────────────────────────────────────────
class PatientFeatures(BaseModel):
    time_in_hospital: int = Field(..., ge=1, le=30, example=5)
    num_lab_procedures: int = Field(..., ge=0, le=200, example=44)
    num_procedures: int = Field(..., ge=0, le=10, example=1)
    num_medications: int = Field(..., ge=0, le=100, example=12)
    number_outpatient: int = Field(0, ge=0, example=0)
    number_emergency: int = Field(0, ge=0, example=1)
    number_inpatient: int = Field(0, ge=0, example=0)
    number_diagnoses: int = Field(..., ge=1, le=20, example=9)
    age_encoded: float = Field(65.0, ge=0, le=100, example=65.0,
                               description="Age midpoint: 65=age 60-70")
    admission_type_encoded: int = Field(0, ge=0, example=1)
    discharge_encoded: int = Field(0, ge=0, example=1)

    @validator("*", pre=True)
    def none_to_zero(cls, v):
        return v if v is not None else 0


class PredictionResponse(BaseModel):
    readmission_risk: str           # "LOW" | "MEDIUM" | "HIGH"
    probability_readmitted: float
    probability_not_readmitted: float
    recommendation: str
    inference_time_ms: float


class BatchRequest(BaseModel):
    patients: List[PatientFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_patients: int
    high_risk_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────
def _engineer_features(p: PatientFeatures) -> np.ndarray:
    """Add derived features matching training pipeline."""
    total_visits = p.number_outpatient + p.number_emergency + p.number_inpatient
    medication_ratio = p.num_medications / (p.num_lab_procedures + 1e-5)
    age_visit_interaction = p.age_encoded * total_visits

    row = [
        p.time_in_hospital, p.num_lab_procedures, p.num_procedures,
        p.num_medications, p.number_outpatient, p.number_emergency,
        p.number_inpatient, p.number_diagnoses,
        p.age_encoded, p.admission_type_encoded, p.discharge_encoded,
        total_visits, medication_ratio, age_visit_interaction,
    ]
    return np.array(row).reshape(1, -1)


def _risk_label(prob: float) -> tuple:
    if prob >= 0.65:
        return "HIGH", "⚠️ High readmission risk. Consider extended monitoring or follow-up care."
    elif prob >= 0.35:
        return "MEDIUM", "⚡ Moderate risk. Schedule follow-up within 2 weeks."
    else:
        return "LOW", "✅ Low readmission risk. Standard discharge protocol."


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model_info": MODEL_INFO,
    }


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Return current model metadata."""
    return {
        "model_info": MODEL_INFO,
        "features": FEATURE_ORDER,
        "output": "Binary classification: readmitted within 30 days (1) or not (0)",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientFeatures):
    """Predict 30-day readmission risk for a single patient."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")

    t0 = time.perf_counter()
    X = _engineer_features(patient)

    try:
        proba = MODEL.predict_proba(X)[0]
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    p_readmit = float(proba[1])
    risk, recommendation = _risk_label(p_readmit)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        readmission_risk=risk,
        probability_readmitted=round(p_readmit, 4),
        probability_not_readmitted=round(float(proba[0]), 4),
        recommendation=recommendation,
        inference_time_ms=round(elapsed_ms, 2),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(batch: BatchRequest):
    """Batch predict for multiple patients."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded")
    if len(batch.patients) > 1000:
        raise HTTPException(400, "Batch size limit: 1000 patients")

    X = np.vstack([_engineer_features(p) for p in batch.patients])
    probas = MODEL.predict_proba(X)

    predictions = []
    for prob in probas:
        p_readmit = float(prob[1])
        risk, recommendation = _risk_label(p_readmit)
        predictions.append(PredictionResponse(
            readmission_risk=risk,
            probability_readmitted=round(p_readmit, 4),
            probability_not_readmitted=round(float(prob[0]), 4),
            recommendation=recommendation,
            inference_time_ms=0.0,
        ))

    high_risk = sum(1 for p in predictions if p.readmission_risk == "HIGH")
    return BatchResponse(predictions=predictions, total_patients=len(predictions), high_risk_count=high_risk)
