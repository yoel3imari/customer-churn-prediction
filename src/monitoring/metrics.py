"""Prometheus metrics definitions for inference and model monitoring."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Metric for single inference predictions (labeled by prediction class "0" or "1")
CHURN_PREDICTIONS_COUNTER = Counter(
    "churn_predictions_total",
    "Total count of churn predictions generated",
    labelnames=["prediction"],
)

# Metric for batch predictions count
CHURN_BATCH_PREDICTIONS_COUNTER = Counter(
    "churn_batch_predictions_total",
    "Total count of customer rows processed in batch prediction requests",
)

# Metric for prediction probability distribution
CHURN_PROBABILITY_HISTOGRAM = Histogram(
    "churn_prediction_probability",
    "Histogram of predicted churn probabilities",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# Metric for feature data drift score (p-value or Wasserstein distance per feature)
CHURN_DRIFT_SCORE_GAUGE = Gauge(
    "churn_feature_drift_score",
    "Statistical drift score per feature comparing current batch vs reference data",
    labelnames=["feature_name"],
)
