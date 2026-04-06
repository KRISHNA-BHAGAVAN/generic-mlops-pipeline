"""
Model training module.

Provides task-specific training functions that use the model factory
to instantiate models and return trained models with metadata.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.config.validate_config import ExperimentConfig
from src.models.factory import create_model
from src.pipelines.exceptions import ModelError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ExperimentConfig,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a model based on the experiment config.

    Routes to the appropriate task-specific training function.

    Args:
        X_train: Training feature matrix.
        y_train: Training target vector.
        config: Experiment configuration.

    Returns:
        Tuple of (trained_model, metadata_dict).

    Raises:
        ModelError: If training fails.
    """
    if config.task_type == "regression":
        return train_regression(X_train, y_train, config)
    elif config.task_type == "classification":
        return train_classification(X_train, y_train, config)
    else:
        raise ModelError(f"Unsupported task type: {config.task_type}")


def train_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ExperimentConfig,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a regression model.

    Args:
        X_train: Feature matrix (rows=samples, cols=features).
        y_train: Target vector (numeric).
        config: Experiment configuration.

    Returns:
        Tuple of:
        - trained_model: sklearn-compatible estimator with .predict(X).
        - metadata: dict with training info.

    Raises:
        ModelError: If training fails.
    """
    logger.info(
        f"Training regression model: {config.model_type} "
        f"({X_train.shape[0]} samples, {X_train.shape[1]} features)"
    )

    model = create_model(config.model_type, "regression", config.model_params)

    start_time = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise ModelError(f"Regression training failed: {e}") from e
    training_duration = time.time() - start_time

    # Attach metadata to model
    model._metadata = {
        "task_type": "regression",
        "target_column": config.target_column,
        "training_samples": len(X_train),
    }

    metadata = {
        "feature_names": X_train.columns.tolist(),
        "target_name": config.target_column,
        "training_samples": len(X_train),
        "training_duration_seconds": round(training_duration, 3),
        "model_type": config.model_type,
        "task_type": "regression",
        # Store small samples for MLflow signature inference
        "X_sample": X_train.head(5),
        "y_pred_sample": model.predict(X_train.head(5)),
    }

    logger.info(f"Training completed in {training_duration:.2f}s")

    return model, metadata


def train_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ExperimentConfig,
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a classification model.

    Args:
        X_train: Feature matrix.
        y_train: Target vector (categorical/binary).
        config: Experiment configuration.

    Returns:
        Tuple of (trained_model, metadata).

    Raises:
        ModelError: If training fails.
    """
    logger.info(
        f"Training classification model: {config.model_type} "
        f"({X_train.shape[0]} samples, {X_train.shape[1]} features)"
    )

    model = create_model(config.model_type, "classification", config.model_params)

    start_time = time.time()
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        raise ModelError(f"Classification training failed: {e}") from e
    training_duration = time.time() - start_time

    # Attach metadata
    model._metadata = {
        "task_type": "classification",
        "target_column": config.target_column,
        "training_samples": len(X_train),
    }

    # Get class labels
    classes = model.classes_.tolist() if hasattr(model, "classes_") else []

    metadata = {
        "feature_names": X_train.columns.tolist(),
        "target_name": config.target_column,
        "training_samples": len(X_train),
        "training_duration_seconds": round(training_duration, 3),
        "model_type": config.model_type,
        "task_type": "classification",
        "classes": classes,
        "n_classes": len(classes),
        # Store small samples for MLflow signature inference
        "X_sample": X_train.head(5),
        "y_pred_sample": model.predict(X_train.head(5)),
    }

    logger.info(
        f"Training completed in {training_duration:.2f}s. "
        f"Classes: {classes}"
    )

    return model, metadata
