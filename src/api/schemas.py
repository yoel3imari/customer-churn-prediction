"""Pydantic v2 models for the Churn Prediction API."""

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for a single prediction."""

    features: dict[str, float] = Field(
        ..., description="Feature dictionary for a single customer record"
    )
    explain: bool = Field(
        False,
        description="If true, include SHAP feature attribution values",
    )


class PredictResponse(BaseModel):
    """Response for a single prediction."""

    prediction: int = Field(
        ..., description="Predicted class (0 = no churn, 1 = churn)"
    )
    churn_probability: float = Field(
        ..., description="Probability of churn in [0.0, 1.0]"
    )
    shap_values: list[float] | None = Field(
        None,
        description=(
            "Per-feature SHAP values for this prediction, "
            "ordered by feature_names; null when explain=False or SHAP unavailable"
        ),
    )


class BatchPredictResponse(BaseModel):
    """Response for a batch prediction CSV upload."""

    predictions: list[PredictResponse] = Field(
        ..., description="Individual predictions for each input row"
    )
    total_count: int = Field(..., description="Total number of predictions")
    churn_count: int = Field(
        ..., description="Number of rows predicted to churn"
    )
    churn_rate: float = Field(
        ..., description="Churn rate across the batch [0.0, 1.0]"
    )


class ModelInfo(BaseModel):
    """Metadata about the currently loaded model."""

    model_name: str = Field(..., description="MLflow registered model name")
    stage: str = Field(
        ..., description="Model stage (e.g. Production, Staging)"
    )
    version: str | None = Field(
        None, description="Concrete model version when available"
    )
    metrics: dict[str, float] = Field(
        ..., description="Training metrics logged in MLflow"
    )
    feature_names: list[str] = Field(
        ..., description="Ordered feature names the model expects"
    )
    shap_available: bool = Field(
        ...,
        description="Whether SHAP explainability is available in this environment",
    )


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = Field("ok", description="Service status")
    timestamp: datetime = Field(
        ..., description="UTC timestamp of the health check"
    )
