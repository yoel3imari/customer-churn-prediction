"""Tests for observability, metrics, structured logging, and data drift detection."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.monitoring.drift import DataDriftDetector
from src.monitoring.logging import configure_logging
from src.monitoring.metrics import (
    CHURN_PREDICTIONS_COUNTER,
    CHURN_PROBABILITY_HISTOGRAM,
)


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


def test_configure_logging():
    """Test that structured logging configures without error."""
    configure_logging(log_level="DEBUG", json_format=True)
    configure_logging(log_level="INFO", json_format=False)


def test_metrics_counter_and_histogram():
    """Test Prometheus counter and histogram metrics recording."""
    CHURN_PREDICTIONS_COUNTER.labels(prediction="1").inc()
    CHURN_PROBABILITY_HISTOGRAM.observe(0.85)


def test_prometheus_metrics_endpoint(client):
    """Test that /metrics endpoint returns Prometheus formatted metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text or "churn_predictions_total" in response.text


def test_data_drift_detector_no_drift():
    """Test data drift detector when reference and current data are identical."""
    rng = np.random.RandomState(42)
    df_ref = pd.DataFrame({
        "revenue": rng.normal(100, 20, 100),
        "mou": rng.normal(500, 50, 100),
        "refurb": rng.choice([0, 1], 100),
    })

    detector = DataDriftDetector(reference_data=df_ref)
    result = detector.detect_drift(df_ref)

    assert isinstance(result, dict)
    assert result["dataset_drift_detected"] is False
    assert result["number_of_drifted_features"] == 0


def test_data_drift_detector_with_drift():
    """Test data drift detector when current data has shifted distribution."""
    rng = np.random.RandomState(42)
    df_ref = pd.DataFrame({
        "revenue": rng.normal(100, 20, 200),
        "mou": rng.normal(500, 50, 200),
    })

    df_curr = pd.DataFrame({
        "revenue": rng.normal(500, 20, 200),  # Significant mean shift
        "mou": rng.normal(500, 50, 200),
    })

    detector = DataDriftDetector(reference_data=df_ref)
    result = detector.detect_drift(df_curr)

    assert result["number_of_drifted_features"] >= 1
    assert result["features"]["revenue"]["drift_detected"] is True


def test_monitoring_drift_api_endpoint(client):
    """Test the POST /api/v1/monitoring/drift API endpoint."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "revenue": rng.normal(100, 20, 50),
        "mou": rng.normal(500, 50, 50),
        "months": rng.randint(1, 48, 50),
    })

    csv_data = df.to_csv(index=False).encode("utf-8")
    files = {"file": ("test_batch.csv", csv_data, "text/csv")}

    response = client.post("/api/v1/monitoring/drift", files=files)
    assert response.status_code == 200

    json_resp = response.json()
    assert "dataset_drift_detected" in json_resp
    assert "drift_share" in json_resp
    assert "features" in json_resp
