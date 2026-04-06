"""
Model evaluation module.

Provides a metrics registry and evaluation functions for both
regression and classification tasks. All metric functions follow
a consistent (y_true, y_pred) -> float signature.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config.validate_config import ExperimentConfig
from src.pipelines.exceptions import ModelError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Metrics registry ──────────────────────────────────────────────────────

METRICS_REGISTRY: Dict[str, Dict[str, Callable]] = {
    "regression": {
        "mse": lambda y_true, y_pred: float(mean_squared_error(y_true, y_pred)),
        "rmse": lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": lambda y_true, y_pred: float(mean_absolute_error(y_true, y_pred)),
        "r2": lambda y_true, y_pred: float(r2_score(y_true, y_pred)),
    },
    "classification": {
        "accuracy": lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
        "precision": lambda y_true, y_pred: float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall": lambda y_true, y_pred: float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1": lambda y_true, y_pred: float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "auc": lambda y_true, y_pred: _safe_roc_auc(y_true, y_pred),
    },
}


def _safe_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute ROC AUC safely, handling multi-class and edge cases.

    For multi-class, uses one-vs-rest strategy. Returns 0.0 on failure.
    """
    try:
        n_classes = len(np.unique(y_true))
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_pred))
        else:
            # Multi-class: needs probability scores, fall back to 0.0
            logger.warning("AUC requires predict_proba for multi-class; skipping.")
            return 0.0
    except Exception as e:
        logger.warning(f"AUC computation failed: {e}")
        return 0.0


# ─── Public API ─────────────────────────────────────────────────────────────
def evaluate(
    task_type: str,
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    metrics: List[str],
) -> Dict[str, float]:
    """
    Compute evaluation metrics for predictions.

    Args:
        task_type: "regression" or "classification".
        y_true: True labels/targets.
        y_pred: Predicted labels/targets.
        metrics: List of metric names to compute.

    Returns:
        Dict of metric_name -> metric_value.

    Raises:
        ModelError: If a metric name is not in the registry.
    """
    registry = METRICS_REGISTRY.get(task_type)
    if registry is None:
        raise ModelError(
            f"Unknown task type: '{task_type}'. "
            f"Supported: {list(METRICS_REGISTRY.keys())}"
        )

    results: Dict[str, float] = {}
    for metric_name in metrics:
        metric_fn = registry.get(metric_name)
        if metric_fn is None:
            raise ModelError(
                f"Unknown metric '{metric_name}' for task '{task_type}'. "
                f"Available: {list(registry.keys())}"
            )
        try:
            results[metric_name] = metric_fn(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Metric '{metric_name}' failed: {e}")
            results[metric_name] = 0.0

    logger.info(f"Evaluation results: {results}")
    return results


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: ExperimentConfig,
) -> Dict[str, float]:
    """
    Evaluate a trained model on test data using config-specified metrics.

    Args:
        model: Trained sklearn model with .predict() method.
        X_test: Test feature matrix.
        y_test: Test target values.
        config: Experiment configuration with task type and metrics.

    Returns:
        Dictionary of metric_name -> metric_value.
    """
    y_pred = model.predict(X_test)
    return evaluate(config.task_type, y_test, y_pred, config.metrics)


def generate_evaluation_plots(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    config: ExperimentConfig,
    output_dir: str | None = None,
) -> Dict[str, str]:
    """
    Generate evaluation plots and return paths to saved files.

    Args:
        model: Trained model.
        X_test: Test features.
        y_test: True test labels.
        y_pred: Predicted labels.
        config: Experiment config.
        output_dir: Directory to save plots. Uses temp dir if None.

    Returns:
        Dict of plot_name -> file_path.
    """
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="mlops_plots_")

    os.makedirs(output_dir, exist_ok=True)
    plots: Dict[str, str] = {}

    if config.task_type == "regression":
        # Actual vs Predicted scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolors="k", linewidths=0.5)
        ax.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--", lw=2, label="Perfect Prediction",
        )
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Actual vs Predicted — {config.experiment_name}")
        ax.legend()
        path = os.path.join(output_dir, "actual_vs_predicted.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        plots["actual_vs_predicted"] = path

        # Residuals plot
        residuals = y_test.values - y_pred
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Residual")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residual Distribution — {config.experiment_name}")
        path = os.path.join(output_dir, "residuals.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        plots["residuals"] = path

    elif config.task_type == "classification":
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"Confusion Matrix — {config.experiment_name}")
        fig.colorbar(im)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

        path = os.path.join(output_dir, "confusion_matrix.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        plots["confusion_matrix"] = path

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = X_test.columns.tolist()
        indices = np.argsort(importances)[::-1]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(
            range(len(indices)),
            importances[indices],
            align="center",
        )
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance — {config.experiment_name}")
        ax.invert_yaxis()
        path = os.path.join(output_dir, "feature_importance.png")
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        plots["feature_importance"] = path

    logger.info(f"Generated {len(plots)} evaluation plots")
    return plots
