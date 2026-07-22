"""Optuna hyperparameter tuning for churn prediction models."""

import time
from functools import partial
from tempfile import NamedTemporaryFile
from typing import Any, Callable

import mlflow
import optuna
from optuna.integration import MLflowCallback

from src.config import (
    ALGORITHMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
)
from src.models.train import train_model


def get_param_space(algorithm: str, trial: optuna.Trial) -> dict:
    if algorithm == "LogisticRegression":
        params: dict[str, Any] = {
            "C": trial.suggest_float("C", 1e-4, 1e4, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
        }
        penalty = params["penalty"]
        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
            params["solver"] = "saga"
        elif penalty == "l1":
            params["solver"] = "saga"
        else:
            params["solver"] = trial.suggest_categorical(
                "solver", ["lbfgs", "liblinear", "saga", "newton-cg"]
            )
        return params

    if algorithm == "RandomForest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

    if algorithm == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }

    if algorithm == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    if algorithm == "CatBoost":
        return {
            "iterations": trial.suggest_int("iterations", 50, 500),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 1, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
        }

    raise ValueError(f"Unknown algorithm: {algorithm}")


def _objective(
    trial: optuna.Trial,
    algorithm: str,
    strategy: str,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
) -> float:
    params = get_param_space(algorithm, trial)
    result = train_model(
        X_train, y_train,
        X_val=X_val, y_val=y_val,
        algorithm=algorithm,
        strategy=strategy,
        params=params,
    )
    # train_model preprocesses internally and returns validation metrics
    return float(result["metrics"]["pr_auc"])


def tune_model(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    algorithm: str = "LogisticRegression",
    strategy: str = "class_weight",
    n_trials: int = 100,
    study_name: str | None = None,
) -> dict:
    valid = {a["name"] for a in ALGORITHMS}
    if algorithm not in valid:
        raise ValueError(f"Unknown algorithm: {algorithm}. Valid: {sorted(valid)}")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
        study_name=study_name or f"tune-{algorithm}-{strategy}",
    )

    objective: Callable[[optuna.Trial], float] = partial(
        _objective,
        algorithm=algorithm,
        strategy=strategy,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    print(f"    Tuning {algorithm} with {n_trials} Optuna trials...")
    tune_start = time.time()

    with mlflow.start_run(run_name=study.study_name):
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("best_pr_auc", float(study.best_value))

        with NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=f"{study.study_name}_"
        ) as tmp:
            tmp.write(study.trials_dataframe().to_json(orient="records"))
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="study")

    tune_elapsed = time.time() - tune_start
    print(f"    ✓ Tuning complete: {tune_elapsed:.1f}s, best PR-AUC: {study.best_value:.4f}")

    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "study": study,
        "trials_dataframe": study.trials_dataframe(),
    }
