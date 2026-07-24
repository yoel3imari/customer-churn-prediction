"""Model evaluation: metrics, plots, and SHAP explanations logged to MLflow."""

import os
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

from src.config import Paths, RANDOM_STATE

sns.set_style("whitegrid")

_TREE_MODEL_HINTS: tuple[str, ...] = (
    "randomforest",
    "random forest",
    "xgb",
    "xgboost",
    "lightgbm",
    "lgbm",
    "catboost",
    "decisiontree",
    "gradientboosting",
    "extratrees",
    "histgradientboosting",
)


def _is_tree_model(model: Any) -> bool:
    name = type(model).__name__.lower()
    module = type(model).__module__.lower()
    haystack = f"{name} {module}"
    return any(hint in haystack for hint in _TREE_MODEL_HINTS)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = "f1",
    fn_cost: float = 10.0,
    fp_cost: float = 1.0,
) -> dict[str, float]:
    """Find decision threshold maximizing F1 or minimizing cost ($C_{FN} * FN + C_{FP} * FP$)."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    thresholds = np.linspace(0.05, 0.95, 91)
    best_thresh = 0.5
    best_val = -1.0 if metric == "f1" else float("inf")
    best_metrics: dict[str, float] = {}

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        p = float(precision_score(y_true, preds, zero_division=0))
        r = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))

        if metric == "f1":
            if f1 > best_val:
                best_val = f1
                best_thresh = float(t)
                best_metrics = {"f1": f1, "precision": p, "recall": r}
        elif metric == "cost":
            cm = confusion_matrix(y_true, preds)
            fn = int(cm[1, 0]) if cm.shape == (2, 2) else 0
            fp = int(cm[0, 1]) if cm.shape == (2, 2) else 0
            cost = float(fn * fn_cost + fp * fp_cost)
            if cost < best_val:
                best_val = cost
                best_thresh = float(t)
                best_metrics = {"cost": cost, "f1": f1, "precision": p, "recall": r}

    return {
        "optimal_threshold": best_thresh,
        "best_score": best_val,
        **best_metrics,
    }


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    classes = np.unique(y_true)
    single_class = classes.shape[0] < 2

    metrics: dict[str, Any] = {
        "roc_auc": None,
        "pr_auc": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "optimal_threshold_f1": 0.5,
        "optimal_f1": None,
        "optimal_threshold_cost": 0.5,
        "optimal_cost": None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if not single_class:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

        if y_proba is not None:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except ValueError:
                metrics["roc_auc"] = None
            try:
                metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
            except ValueError:
                metrics["pr_auc"] = None

            opt_f1 = find_optimal_threshold(y_true, y_proba, metric="f1")
            metrics["optimal_threshold_f1"] = opt_f1["optimal_threshold"]
            metrics["optimal_f1"] = opt_f1["best_score"]

            opt_cost = find_optimal_threshold(y_true, y_proba, metric="cost")
            metrics["optimal_threshold_cost"] = opt_cost["optimal_threshold"]
            metrics["optimal_cost"] = opt_cost["best_score"]
    else:
        metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        metrics["precision"] = float(
            precision_score(y_true, y_pred, zero_division=0)
        )
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))

    return metrics


def log_metrics_to_mlflow(run_id: str, metrics: dict) -> None:
    import mlflow

    for key, value in metrics.items():
        if key == "confusion_matrix":
            continue
        if value is None:
            continue
        mlflow.log_metric(key, float(value), run_id=run_id)


def _resolve_save_path(save_path: str | None, default_name: str) -> str:
    if save_path is not None:
        path = save_path
    else:
        path = str(Paths.REPORTS_FIGURES / default_name)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | None = None,
) -> str:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    path = _resolve_save_path(save_path, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | None = None,
) -> str:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()

    path = _resolve_save_path(save_path, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_path: str | None = None,
) -> str:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except ValueError:
        pr_auc = float("nan")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(
        recall, precision, color="#2ca02c", lw=2, label=f"PR (AUC = {pr_auc:.3f})"
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()

    path = _resolve_save_path(save_path, "pr_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _get_feature_names(X_explain: Any, feature_names: list[str] | None) -> list[str]:
    if feature_names is not None:
        return list(feature_names)
    if hasattr(X_explain, "columns"):
        return list(X_explain.columns)
    n = X_explain.shape[1]
    return [f"feature_{i}" for i in range(n)]


def shap_explanation(
    model: Any,
    X_explain: Any,
    run_id: str | None = None,
    save_path: str | None = None,
) -> dict:
    try:
        import shap
    except ImportError:
        return {}

    try:
        if _is_tree_model(model):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_explain)

        raw_values = explainer.shap_values(X_explain)

        if isinstance(raw_values, list):
            shap_values = np.asarray(raw_values[1])
            base_value = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        elif isinstance(raw_values, np.ndarray) and raw_values.ndim == 3:
            shap_values = np.asarray(raw_values[:, :, 1])
            base_value = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
        else:
            shap_values = np.asarray(raw_values)
            base_value = explainer.expected_value
    except Exception:
        return {}

    feature_names = _get_feature_names(X_explain, None)

    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    feature_importance = [
        {"feature": feature_names[i], "importance": float(mean_abs[i])}
        for i in order
    ]

    base_dir = save_path if save_path is not None else str(Paths.REPORTS_FIGURES)
    os.makedirs(base_dir, exist_ok=True)

    summary_path = os.path.join(base_dir, "shap_summary.png")
    bar_path = os.path.join(base_dir, "shap_bar.png")
    waterfall_path = os.path.join(base_dir, "shap_waterfall.png")

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.summary_plot(
            shap_values, X_explain, feature_names=feature_names, show=False, ax=ax
        )
        fig.tight_layout()
        fig.savefig(summary_path, dpi=150)
        plt.close(fig)
    except Exception:
        summary_path = ""

    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.bar_plot(
            mean_abs[order],
            feature_names=np.array(feature_names)[order],
            show=False,
            ax=ax,
        )
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
    except Exception:
        bar_path = ""

    try:
        first = shap_values[0]
        exp = shap.Explanation(
            values=first,
            base_values=base_value,
            data=(X_explain.iloc[0] if hasattr(X_explain, "iloc") else X_explain[0]),
            feature_names=feature_names,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.waterfall(exp, show=False)
        fig.tight_layout()
        fig.savefig(waterfall_path, dpi=150)
        plt.close(fig)
    except Exception:
        waterfall_path = ""

    result: dict[str, Any] = {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "base_value": base_value,
        "summary_plot": summary_path,
        "bar_plot": bar_path,
        "waterfall_plot": waterfall_path,
    }

    if run_id is not None:
        import mlflow

        for p in (summary_path, bar_path, waterfall_path):
            if p:
                try:
                    mlflow.log_artifact(p, run_id=run_id)
                except Exception:
                    pass

    return result


def evaluate_model(
    model: Any,
    X_test: Any,
    y_test: np.ndarray,
    feature_names: list[str] | None = None,
    run_id: str | None = None,
) -> dict:
    y_test = np.asarray(y_test)

    y_pred = model.predict(X_test)

    y_proba: np.ndarray | None = None
    try:
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba = proba[:, 1]
        else:
            y_proba = proba.ravel()
    except (AttributeError, Exception):
        y_proba = None

    metrics = compute_metrics(y_test, y_pred, y_proba)

    cm_path = plot_confusion_matrix(y_test, y_pred)
    plot_paths: dict[str, str] = {"confusion_matrix": cm_path}

    if y_proba is not None:
        plot_paths["roc_curve"] = plot_roc_curve(y_test, y_proba)
        plot_paths["pr_curve"] = plot_pr_curve(y_test, y_proba)

    shap_result = shap_explanation(model, X_test, run_id=run_id)

    if run_id is not None:
        log_metrics_to_mlflow(run_id, metrics)
        import mlflow

        for p in plot_paths.values():
            try:
                mlflow.log_artifact(p, run_id=run_id)
            except Exception:
                pass

    return {
        "metrics": metrics,
        "plots": plot_paths,
        "shap": shap_result,
    }
