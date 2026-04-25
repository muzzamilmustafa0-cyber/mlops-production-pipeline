"""
Tests for the prediction API.
Run: pytest tests/ -v
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np


@pytest.fixture(scope="module")
def client():
    """Create test client with a mocked model."""
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

    with patch("api.app.load_model"), \
         patch("api.app.MODEL", mock_model), \
         patch("api.app.MODEL_INFO", {"source": "test"}):
        from api.app import app
        with TestClient(app) as c:
            yield c


SAMPLE_PATIENT = {
    "time_in_hospital": 5,
    "num_lab_procedures": 44,
    "num_procedures": 1,
    "num_medications": 12,
    "number_outpatient": 0,
    "number_emergency": 1,
    "number_inpatient": 0,
    "number_diagnoses": 9,
    "age_encoded": 65.0,
    "admission_type_encoded": 1,
    "discharge_encoded": 1,
}


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_has_status_field(self, client):
        data = client.get("/health").json()
        assert "status" in data


class TestPredictEndpoint:

    def test_predict_returns_200(self, client):
        with patch("api.app.MODEL") as mock:
            mock.predict_proba.return_value = np.array([[0.7, 0.3]])
            resp = client.post("/predict", json=SAMPLE_PATIENT)
            assert resp.status_code == 200

    def test_predict_response_structure(self, client):
        with patch("api.app.MODEL") as mock:
            mock.predict_proba.return_value = np.array([[0.65, 0.35]])
            data = client.post("/predict", json=SAMPLE_PATIENT).json()
            assert "readmission_risk" in data
            assert "probability_readmitted" in data
            assert "recommendation" in data
            assert "inference_time_ms" in data

    def test_predict_risk_labels(self, client):
        for prob, expected_risk in [(0.8, "HIGH"), (0.5, "MEDIUM"), (0.1, "LOW")]:
            with patch("api.app.MODEL") as mock:
                mock.predict_proba.return_value = np.array([[1 - prob, prob]])
                data = client.post("/predict", json=SAMPLE_PATIENT).json()
                assert data["readmission_risk"] == expected_risk

    def test_predict_probabilities_sum_to_one(self, client):
        with patch("api.app.MODEL") as mock:
            mock.predict_proba.return_value = np.array([[0.6, 0.4]])
            data = client.post("/predict", json=SAMPLE_PATIENT).json()
            total = data["probability_readmitted"] + data["probability_not_readmitted"]
            assert abs(total - 1.0) < 1e-3

    def test_predict_invalid_time_in_hospital(self, client):
        bad = {**SAMPLE_PATIENT, "time_in_hospital": -1}
        resp = client.post("/predict", json=bad)
        assert resp.status_code == 422   # Validation error


class TestBatchEndpoint:

    def test_batch_predict(self, client):
        with patch("api.app.MODEL") as mock:
            mock.predict_proba.return_value = np.array([[0.7, 0.3], [0.4, 0.6]])
            resp = client.post("/predict/batch", json={"patients": [SAMPLE_PATIENT, SAMPLE_PATIENT]})
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_patients"] == 2
            assert "high_risk_count" in data
