"""
Model training with MLflow experiment tracking.

Trains 4 models, logs all metrics/artifacts to MLflow,
and registers the best model to the MLflow Model Registry.

Usage:
    python src/train.py --data data/diabetic_data.csv --experiment hospital-readmission
"""
import argparse
import logging
import os
import pickle
import json
from typing import Dict, Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocess import load_data, preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed — skipping XGB model")


def get_models() -> Dict[str, Any]:
    """Define model configurations to benchmark."""
    models = {
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            random_state=42
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=5, random_state=42, n_jobs=-1,
        )
    return models


def plot_confusion_matrix(y_true, y_pred, model_name: str, save_path: str) -> None:
    """Save a styled confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Readmitted", "Readmitted"],
                yticklabels=["Not Readmitted", "Readmitted"])
    plt.title(f"Confusion Matrix — {model_name}", fontsize=13)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, model_name: str, save_path: str) -> None:
    """Plot feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    importance = pd.Series(model.feature_importances_, index=feature_names)
    importance = importance.sort_values(ascending=True).tail(15)

    plt.figure(figsize=(8, 6))
    importance.plot(kind="barh", color="#667eea")
    plt.title(f"Top Feature Importances — {model_name}", fontsize=13)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_and_log(
    model_name: str,
    model,
    data: Dict[str, Any],
    experiment_name: str,
) -> Dict[str, float]:
    """Train one model and log everything to MLflow."""

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    with mlflow.start_run(run_name=model_name):
        # ── Train ──────────────────────────────────────────────────────────
        logger.info(f"Training {model_name}…")
        model.fit(X_train, y_train)

        # ── Evaluate ───────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred, average="weighted"),
            "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_test, y_pred, average="weighted"),
        }

        # ── Log to MLflow ──────────────────────────────────────────────────
        mlflow.log_params(model.get_params() if hasattr(model, "get_params") else {})
        mlflow.log_metrics(metrics)

        # Classification report as artifact
        report = classification_report(y_test, y_pred, target_names=["Not Readmitted", "Readmitted"])
        mlflow.log_text(report, "classification_report.txt")

        # Confusion matrix plot
        cm_path = f"/tmp/{model_name}_cm.png"
        plot_confusion_matrix(y_test, y_pred, model_name, cm_path)
        mlflow.log_artifact(cm_path)

        # Feature importance plot
        fi_path = f"/tmp/{model_name}_fi.png"
        plot_feature_importance(model, feature_names, model_name, fi_path)
        if os.path.exists(fi_path):
            mlflow.log_artifact(fi_path)

        # Log model
        mlflow.sklearn.log_model(
            model, artifact_path="model",
            registered_model_name=f"readmission-{model_name.lower()}",
        )

        logger.info(f"{model_name} — ROC-AUC: {metrics['roc_auc']:.4f} | F1: {metrics['f1']:.4f}")
        return metrics


def train_all(data_path: str, experiment_name: str = "hospital-readmission") -> None:
    """Run full training pipeline for all models."""
    mlflow.set_experiment(experiment_name)

    # Load & preprocess
    df = load_data(data_path)
    data = preprocess(df)

    # Train all models
    results = {}
    for model_name, model in get_models().items():
        metrics = train_and_log(model_name, model, data, experiment_name)
        results[model_name] = metrics

    # Print leaderboard
    print("\n" + "=" * 60)
    print(f"{'Model':<25} {'ROC-AUC':>10} {'F1':>8} {'Accuracy':>10}")
    print("=" * 60)
    for name, m in sorted(results.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
        print(f"{name:<25} {m['roc_auc']:>10.4f} {m['f1']:>8.4f} {m['accuracy']:>10.4f}")
    print("=" * 60)

    best = max(results, key=lambda k: results[k]["roc_auc"])
    print(f"\n✅ Best model: {best} (ROC-AUC={results[best]['roc_auc']:.4f})")
    print(f"🔗 MLflow UI: mlflow ui --port 5000")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hospital readmission models")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--experiment", default="hospital-readmission")
    args = parser.parse_args()
    train_all(args.data, args.experiment)
