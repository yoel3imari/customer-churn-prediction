"""API router for churn prediction endpoints."""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_REGISTRY_NAME
from src.models.predict import ChurnPredictor
from src.monitoring import (
    CHURN_BATCH_PREDICTIONS_COUNTER,
    CHURN_DRIFT_SCORE_GAUGE,
    CHURN_PREDICTIONS_COUNTER,
    CHURN_PROBABILITY_HISTOGRAM,
    DataDriftDetector,
)

from .schemas import (
    BatchPredictResponse,
    DriftReportResponse,
    HealthResponse,
    ModelInfo,
    PredictRequest,
    PredictResponse,
)

logger = logging.getLogger(__name__)

# Module-level sentinel for required file uploads (avoids ruff B008)
_FILE_SENTINEL = File(...)

router = APIRouter()


# ── Model & Detector loading (lazy, cached) ─────────────────────────────


@lru_cache(maxsize=1)
def _get_predictor() -> ChurnPredictor:
    """Load and cache the Production-stage model from the MLflow registry."""
    logger.info("Loading ChurnPredictor (stage=Production) from MLflow…")
    return ChurnPredictor(stage="Production")


@lru_cache(maxsize=1)
def _get_drift_detector() -> DataDriftDetector:
    """Load and cache the DataDriftDetector."""
    logger.info("Initializing DataDriftDetector…")
    return DataDriftDetector()


def _try_get_model_version() -> str | None:
    """Best-effort lookup of the current Production model version."""
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(
            MLFLOW_REGISTRY_NAME, stages=["Production"]
        )
        if versions:
            return versions[0].version
    except Exception:
        logger.debug("Could not resolve model version", exc_info=True)
    return None


# ── Endpoints ──────────────────────────────────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Return service health status."""
    return HealthResponse(status="ok", timestamp=datetime.now(UTC))


@router.get("/model/info", response_model=ModelInfo)
async def model_info() -> ModelInfo:
    """Return metadata about the currently loaded model."""
    predictor = _get_predictor()
    meta: dict[str, Any] = predictor.get_metadata()

    return ModelInfo(
        model_name=meta.get("model_name", MLFLOW_REGISTRY_NAME),
        stage=meta.get("stage", "Production"),
        version=_try_get_model_version(),
        metrics=meta.get("metrics", {}),
        feature_names=meta.get("feature_names", []),
        shap_available=meta.get("shap_available", False),
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict churn for a single customer record.

    Optionally include SHAP feature attribution by setting ``explain=true``.
    """
    if not request.features:
        raise HTTPException(
            status_code=422, detail="'features' dict must not be empty"
        )

    predictor = _get_predictor()
    df = pd.DataFrame([request.features])

    if request.explain:
        try:
            result = predictor.predict_with_shap(df)
            pred = int(result["prediction"][0])
            prob = float(result["churn_probability"][0])
            shap_raw = result.get("shap_values")
            shap_values: list[float] | None = (
                shap_raw[0] if shap_raw and len(shap_raw) > 0 else None
            )
        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("SHAP calculation failed (%s); falling back to standard inference", e)
            pred_arr = predictor.predict(df)
            proba = predictor.predict_proba(df)
            pred = int(pred_arr[0])
            prob = float(proba[0, 1])
            shap_values = None

        CHURN_PREDICTIONS_COUNTER.labels(prediction=str(pred)).inc()
        CHURN_PROBABILITY_HISTOGRAM.observe(prob)

        return PredictResponse(
            prediction=pred,
            churn_probability=prob,
            shap_values=shap_values,
        )


    pred_arr = predictor.predict(df)
    proba = predictor.predict_proba(df)

    pred = int(pred_arr[0])
    prob = float(proba[0, 1])

    CHURN_PREDICTIONS_COUNTER.labels(prediction=str(pred)).inc()
    CHURN_PROBABILITY_HISTOGRAM.observe(prob)

    return PredictResponse(
        prediction=pred,
        churn_probability=prob,
        shap_values=None,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(file: UploadFile = _FILE_SENTINEL) -> BatchPredictResponse:
    """Upload a CSV of customer records and return predictions for every row.

    The CSV must contain all features expected by the model (feature order
    does not matter — the preprocessing pipeline handles alignment).
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted (upload a .csv file)",
        )

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except (pd.errors.ParserError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse CSV: {exc}"
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    predictor = _get_predictor()

    try:
        result_df = predictor.predict_batch(df)
    except (ValueError, TypeError, RuntimeError) as exc:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {exc}"
        )

    predictions: list[PredictResponse] = []
    for _, row in result_df.iterrows():
        shap_val = row.get("shap_values")
        if isinstance(shap_val, np.ndarray):
            shap_val = shap_val.tolist()

        pred = int(row["prediction"])
        prob = float(row["churn_probability"])

        CHURN_PREDICTIONS_COUNTER.labels(prediction=str(pred)).inc()
        CHURN_PROBABILITY_HISTOGRAM.observe(prob)

        predictions.append(
            PredictResponse(
                prediction=pred,
                churn_probability=prob,
                shap_values=shap_val if shap_val is not None else None,
            )
        )

    total = len(predictions)
    CHURN_BATCH_PREDICTIONS_COUNTER.inc(total)

    churn_count = sum(1 for p in predictions if p.prediction == 1)
    churn_rate = round(churn_count / total, 4) if total > 0 else 0.0

    return BatchPredictResponse(
        predictions=predictions,
        total_count=total,
        churn_count=churn_count,
        churn_rate=churn_rate,
    )


@router.post("/monitoring/drift", response_model=DriftReportResponse)
async def detect_drift(file: UploadFile = _FILE_SENTINEL) -> DriftReportResponse:
    """Upload a CSV dataset to analyze feature data drift against baseline reference data."""
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are accepted (upload a .csv file)",
        )

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except (pd.errors.ParserError, ValueError, OSError) as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to parse CSV: {exc}"
        )

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty")

    detector = _get_drift_detector()
    report = detector.detect_drift(df)

    for feat, detail in report.get("features", {}).items():
        CHURN_DRIFT_SCORE_GAUGE.labels(feature_name=feat).set(detail["p_value"])

    return DriftReportResponse(**report)

