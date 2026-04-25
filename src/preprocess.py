"""
Data preprocessing pipeline with MLflow artifact logging.
Handles feature engineering, encoding, and train/test splitting.
"""
import logging
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


FEATURE_COLS = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
    "age_encoded", "admission_type_encoded", "discharge_encoded",
    "total_visits", "medication_ratio", "age_visit_interaction",
]
TARGET_COL = "readmitted_30days"


def load_data(path: str) -> pd.DataFrame:
    """Load CSV dataset."""
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-informed features for hospital readmission prediction.
    """
    df = df.copy()

    # Target: 1 if readmitted within 30 days, 0 otherwise
    if "readmitted" in df.columns:
        df[TARGET_COL] = (df["readmitted"] == "<30").astype(int)

    # Aggregate visit count
    df["total_visits"] = (
        df.get("number_outpatient", 0)
        + df.get("number_emergency", 0)
        + df.get("number_inpatient", 0)
    )

    # Medication burden ratio
    df["medication_ratio"] = df.get("num_medications", 0) / (
        df.get("num_lab_procedures", 1) + 1e-5
    )

    # Age-visit interaction
    if "age" in df.columns:
        age_map = {
            "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
            "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
            "[80-90)": 85, "[90-100)": 95,
        }
        df["age_encoded"] = df["age"].map(age_map).fillna(50)
        df["age_visit_interaction"] = df["age_encoded"] * df["total_visits"]
    else:
        df["age_encoded"] = 50
        df["age_visit_interaction"] = 50 * df["total_visits"]

    # Encode categoricals
    for col, new_col in [
        ("admission_type_id", "admission_type_encoded"),
        ("discharge_disposition_id", "discharge_encoded"),
    ]:
        if col in df.columns:
            df[new_col] = LabelEncoder().fit_transform(df[col].fillna("Unknown").astype(str))
        else:
            df[new_col] = 0

    return df


def preprocess(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Full preprocessing pipeline:
    1. Feature engineering
    2. Missing value imputation
    3. Feature scaling
    4. Train/test split

    Returns a dict with X_train, X_test, y_train, y_test, scaler, feature_names.
    """
    df = engineer_features(df)

    # Ensure all feature columns exist
    available_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available_cols].copy()
    y = df[TARGET_COL].values if TARGET_COL in df.columns else None

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        logger.info(f"Class balance — Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "scaler": scaler, "imputer": imputer,
            "feature_names": available_cols,
        }

    return {
        "X": X_scaled, "scaler": scaler,
        "imputer": imputer, "feature_names": available_cols,
    }
