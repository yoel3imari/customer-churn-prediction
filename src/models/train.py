"""Model training with MLflow tracking and Model Registry registration."""

from __future__ import annotations

import importlib
import os
import tempfile
import time
from typing import Any, Optional

import joblib
import mlflow
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    ALGORITHMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTRY_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
)
from src.features.pipeline import (
    create_preprocessing_pipeline,
    get_feature_names,
)


def resolve_class(algorithm_name: str) -> tuple:
    for algo in ALGORITHMS:
        if algo["name"] == algorithm_name:
            module_path, class_name = algo["class"].rsplit(".", 1)
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            return (model_class, algo["is_sklearn"])
    raise ValueError(f"Unknown algorithm: {algorithm_name}")


def _get_algorithm_default_params(algorithm_name: str, strategy: str, y_train: Optional[Any] = None) -> dict:
    defaults: dict = {}
    if algorithm_name == "LogisticRegression":
        defaults["max_iter"] = 1000
        if strategy == "class_weight":
            defaults["class_weight"] = "balanced"
    elif algorithm_name == "RandomForest":
        if strategy == "class_weight":
            defaults["class_weight"] = "balanced"
    elif algorithm_name == "XGBoost":
        if strategy == "class_weight":
            if y_train is None:
                defaults["scale_pos_weight"] = 1.0
            else:
                counts = np.bincount(np.asarray(y_train).astype(int))
                neg, pos = counts[0], counts[1]
                weight = (neg / pos) if pos > 0 else 1.0
                defaults["scale_pos_weight"] = float(weight)
    elif algorithm_name == "LightGBM":
        if strategy == "class_weight":
            defaults["is_unbalance"] = True
    elif algorithm_name == "CatBoost":
        if strategy == "class_weight":
            defaults["auto_class_weights"] = "Balanced"
    return defaults


def _evaluate(model: Any, X_val: Any, y_val: Any) -> dict:
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    return {
        "roc_auc": float(roc_auc_score(y_val, y_proba)),
        "pr_auc": float(average_precision_score(y_val, y_proba)),
        "f1": float(f1_score(y_val, y_pred)),
        "precision": float(precision_score(y_val, y_pred)),
        "recall": float(recall_score(y_val, y_pred)),
    }


def _enable_autolog(algorithm: str, is_sklearn: bool) -> None:
    if is_sklearn:
        mlflow.sklearn.autolog(log_models=False, log_datasets=False)
    elif algorithm == "XGBoost":
        mlflow.xgboost.autolog(log_models=False, log_datasets=False)
    elif algorithm == "LightGBM":
        mlflow.lightgbm.autolog(log_models=False, log_datasets=False)


def _register_model(algorithm: str, strategy: str, model: Any) -> None:
    if strategy == "smote" or algorithm in ("LogisticRegression", "RandomForest"):
        kwargs = {}
        if strategy == "smote":
            kwargs["serialization_format"] = "cloudpickle"
        mlflow.sklearn.log_model(
            model, "model", registered_model_name=MLFLOW_REGISTRY_NAME, **kwargs
        )
    elif algorithm == "XGBoost":
        mlflow.xgboost.log_model(
            model, "model", registered_model_name=MLFLOW_REGISTRY_NAME
        )
    elif algorithm == "LightGBM":
        mlflow.lightgbm.log_model(
            model, "model", registered_model_name=MLFLOW_REGISTRY_NAME
        )
    elif algorithm == "CatBoost":
        mlflow.catboost.log_model(
            model, "model", registered_model_name=MLFLOW_REGISTRY_NAME
        )


def train_model(
    X_train: Any,
    y_train: Any,
    X_val: Any = None,
    y_val: Any = None,
    algorithm: str = "LogisticRegression",
    strategy: str = "class_weight",
    params: Optional[dict] = None,
    run_name: Optional[str] = None,
) -> dict:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    model_class, is_sklearn = resolve_class(algorithm)
    _enable_autolog(algorithm, is_sklearn)

    model_params: dict = _get_algorithm_default_params(algorithm, strategy, y_train)
    if params:
        model_params.update(params)

    if algorithm == "CatBoost":
        model_params.setdefault("random_seed", RANDOM_STATE)
        model_params.setdefault("verbose", 0)
    else:
        model_params.setdefault("random_state", RANDOM_STATE)

    # ── 1. Preprocess: clean → features → scale/OHE ─────────────────────────
    preprocessor = create_preprocessing_pipeline()
    X_train_processed = preprocessor.fit_transform(X_train)
    feature_names = get_feature_names(preprocessor)
    X_val_processed = preprocessor.transform(X_val) if X_val is not None else None

    fit_start = time.time()
    print(f"    Model: {algorithm} | Strategy: {strategy} | "
          f"Samples: {len(X_train)} | Features: {len(feature_names)} | "
          f"Params: {len(model_params)}")

    # ── 2. Train on processed (NaN-free, engineered, scaled) data ────────────
    with mlflow.start_run(run_name=run_name, nested=True) as run:
        run_id = run.info.run_id

        if strategy == "class_weight":
            model = model_class(**model_params)
            model.fit(X_train_processed, y_train)
        elif strategy == "smote":
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline

            model = ImbPipeline(
                steps=[
                    ("smote", SMOTE(random_state=RANDOM_STATE)),
                    ("classifier", model_class(**model_params)),
                ]
            )
            model.fit(X_train_processed, y_train)
        else:
            raise ValueError(f"Unknown imbalance strategy: {strategy}")

        mlflow.log_param("model_algorithm", algorithm)
        mlflow.log_param("imbalance_strategy", strategy)
        mlflow.log_params(model_params)
        mlflow.log_param("dataset_shape", X_train_processed.shape)
        mlflow.log_param("n_features", len(feature_names))

        # ── 3. Evaluate on processed validation data ─────────────────────────
        metrics: dict = {}
        if X_val_processed is not None and y_val is not None:
            metrics = _evaluate(model, X_val_processed, y_val)
            mlflow.log_metrics(metrics)

        # ── 4. Log preprocessing artifact ────────────────────────────────────
        with tempfile.TemporaryDirectory() as tmp:
            preproc_path = os.path.join(tmp, "preprocessor.joblib")
            joblib.dump(preprocessor, preproc_path)
            mlflow.log_artifact(preproc_path)

            feat_path = os.path.join(tmp, "feature_names.txt")
            with open(feat_path, "w") as fh:
                fh.write("\n".join(feature_names))
            mlflow.log_artifact(feat_path)

        # ── 5. Register model with MLflow ────────────────────────────────────
        _register_model(algorithm, strategy, model)

        fit_elapsed = time.time() - fit_start
        metric_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        print(f"    ✓ {algorithm} | {strategy} | {fit_elapsed:6.1f}s | {metric_str}")

        return {
            "run_id": run_id,
            "model": model,
            "metrics": metrics,
            "preprocessor": preprocessor,
        }
