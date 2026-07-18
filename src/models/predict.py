"""Inference wrapper for the churn prediction model.

Loads a trained model from the MLflow Model Registry (or a local path) and
exposes a clean prediction interface with optional SHAP explainability.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_REGISTRY_NAME, MLFLOW_TRACKING_URI, Columns

try:
    import shap

    _HAS_SHAP = True
except ImportError:  # pragma: no cover - optional dependency
    shap = None
    _HAS_SHAP = False

_PIPELINE_ARTIFACT = "preprocessing_pipeline.joblib"


class ChurnPredictor:
    def __init__(self, stage: str = "Production", model_uri: str | None = None) -> None:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        self.stage = stage
        self.model_uri = model_uri
        self._source: str = "unknown"
        self._run_id: str | None = None
        self._training_date: str | None = None
        self._metrics: dict[str, float] = {}

        if model_uri and str(model_uri).endswith(".joblib"):
            self._load_from_joblib(model_uri)
        else:
            self._load_from_mlflow(model_uri, stage)

        self.feature_names: list[str] = self._resolve_feature_names()

    def _load_from_joblib(self, model_uri: str) -> None:
        model_path = Path(model_uri)
        self.model = joblib.load(model_path)
        self._source = str(model_path)

        pipeline_path = model_path.parent / _PIPELINE_ARTIFACT
        if pipeline_path.exists():
            self.pipeline = joblib.load(pipeline_path)
        else:
            self.pipeline = None

    def _load_from_mlflow(self, model_uri: str | None, stage: str) -> None:
        resolved = model_uri if model_uri else f"models:/{MLFLOW_REGISTRY_NAME}/{stage}"

        try:
            self.model = mlflow.sklearn.load_model(resolved)
        except Exception:
            self.model = mlflow.pyfunc.load_model(resolved)

        self._source = resolved
        self.pipeline = self._load_pipeline_artifact(resolved, stage)

    def _load_pipeline_artifact(self, resolved: str, stage: str) -> Any | None:
        client = MlflowClient()

        run_id: str | None = None
        if resolved.startswith("models:/"):
            name = resolved.split("/")[1]
            versions = client.get_latest_versions(name, stages=[stage])
            if versions:
                run_id = versions[0].run_id
        elif resolved.startswith("runs:/"):
            run_id = resolved.split("/")[1]

        if run_id is None:
            return None

        self._run_id = run_id
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=_PIPELINE_ARTIFACT
            )
            return joblib.load(local)
        except Exception:
            return None

    def _resolve_feature_names(self) -> list[str]:
        model = self.model
        if hasattr(model, "feature_names_in_") and model.feature_names_in_ is not None:
            return list(model.feature_names_in_)

        if self.pipeline is not None:
            try:
                pre = self.pipeline.named_steps["preprocessor"]
                return list(pre.get_feature_names_out())
            except Exception:
                pass

        return list(Columns.ALL_FEATURES)

    def _transform(self, features: pd.DataFrame) -> np.ndarray:
        if self.pipeline is not None:
            return np.asarray(self.pipeline.transform(features))
        return features.to_numpy()

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        X = self._transform(features)
        return np.asarray(self.model.predict(X)).astype(int)

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        X = self._transform(features)
        return np.asarray(self.model.predict_proba(X))

    def predict_with_shap(self, features: pd.DataFrame) -> dict:
        if not _HAS_SHAP:
            raise RuntimeError(
                "SHAP is not installed; install 'shap' to use explainability."
            )

        X = self._transform(features)
        proba = np.asarray(self.model.predict_proba(X))
        preds = np.asarray(self.model.predict(X)).astype(int)
        churn_prob = proba[:, 1]

        explainer = shap.Explainer(self.model, X)  # type: ignore[union-attr]
        explanation = explainer(X)
        values = np.asarray(explanation.values)

        if values.ndim == 3:
            shap_matrix = values[:, :, 1]
        else:
            shap_matrix = values

        shap_lists = [row.tolist() for row in shap_matrix]

        return {
            "prediction": preds.tolist(),
            "churn_probability": churn_prob.tolist(),
            "shap_values": shap_lists,
            "feature_names": self.feature_names,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        proba = self.predict_proba(df)
        preds = self.predict(df)
        churn_prob = proba[:, 1]

        out = pd.DataFrame(
            {
                "churn_probability": churn_prob,
                "prediction": preds,
            },
            index=df.index,
        )

        if _HAS_SHAP:
            try:
                shap_result = self.predict_with_shap(df)
                out["shap_values"] = shap_result["shap_values"]
            except Exception:
                out["shap_values"] = None
        else:
            out["shap_values"] = None

        return out

    def get_metadata(self) -> dict:
        if self._run_id is not None:
            try:
                client = MlflowClient()
                run = client.get_run(self._run_id)
                self._training_date = (
                    pd.to_datetime(run.info.start_time, unit="ms").isoformat()
                    if run.info.start_time
                    else None
                )
                self._metrics = dict(run.data.metrics)
            except Exception:
                pass

        return {
            "model_name": MLFLOW_REGISTRY_NAME,
            "stage": self.stage,
            "source": self._source,
            "run_id": self._run_id,
            "training_date": self._training_date,
            "metrics": self._metrics,
            "feature_names": self.feature_names,
            "shap_available": _HAS_SHAP,
        }
