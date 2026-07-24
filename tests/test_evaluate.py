import numpy as np
import pytest

from src.models.evaluate import compute_metrics, find_optimal_threshold


def test_find_optimal_threshold_f1():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    res = find_optimal_threshold(y_true, y_proba, metric="f1")
    assert "optimal_threshold" in res
    assert "best_score" in res
    assert 0.4 <= res["optimal_threshold"] <= 0.6
    assert res["best_score"] == pytest.approx(1.0)


def test_find_optimal_threshold_cost():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    res = find_optimal_threshold(y_true, y_proba, metric="cost", fn_cost=10.0, fp_cost=1.0)
    assert "optimal_threshold" in res
    assert "best_score" in res
    assert res["best_score"] == pytest.approx(0.0)


def test_compute_metrics_includes_optimal_thresholds():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert "optimal_threshold_f1" in metrics
    assert "optimal_f1" in metrics
    assert "optimal_threshold_cost" in metrics
    assert "optimal_cost" in metrics
    assert metrics["optimal_f1"] == pytest.approx(1.0)
