import numpy as np
import pandas as pd
import pytest

from src.models.train import train_model
from src.models.tune import get_param_space, tune_model
from src.models.evaluate import find_optimal_threshold, compute_metrics


def test_train_model_xgboost_with_eval_set(sample_df):
    df = sample_df.copy()
    df["mourec"] = [300.0, 400.0, 350.0, 250.0] * 5
    df["blckvce"] = [2, 3, 1, 4] * 5
    df["unansvce"] = [5, 10, 7, 3] * 5
    df["dropblk"] = [1, 2, 0, 3] * 5
    df["income"] = [3, 5, 2, 4] * 5
    df["setprc"] = [40.0, 50.0, 35.0, 60.0] * 5

    X_train = df.drop(columns=["churn"])
    y_train = df["churn"]
    X_val = X_train.copy()
    y_val = y_train.copy()

    res = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        algorithm="XGBoost",
        strategy="class_weight",
        params={"n_estimators": 10, "max_depth": 2},
    )

    assert "model" in res
    assert "metrics" in res
    assert "pr_auc" in res["metrics"]
    assert "optimal_threshold_f1" in res["metrics"]
    assert "optimal_threshold_cost" in res["metrics"]


def test_get_param_space_xgboost():
    class DummyTrial:
        def suggest_int(self, name, low, high, **kw):
            return low
        def suggest_float(self, name, low, high, **kw):
            return low

    space = get_param_space("XGBoost", DummyTrial())
    assert "scale_pos_weight" in space
    assert "reg_alpha" in space
    assert "reg_lambda" in space
    assert space["early_stopping_rounds"] == 30
