"""
Model factory module.

Provides a registry of supported model classes and a factory function
to instantiate models by task type and model type strings from config.
"""

from typing import Type

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.pipelines.exceptions import ModelError


# ─── Model registry ────────────────────────────────────────────────────────

MODEL_REGISTRY: dict[tuple[str, str], Type[BaseEstimator]] = {
    # Regression models
    ("regression", "linear_regression"): LinearRegression,
    ("regression", "random_forest_regression"): RandomForestRegressor,
    # Classification models
    ("classification", "logistic_regression"): LogisticRegression,
    ("classification", "random_forest_classification"): RandomForestClassifier,
}


def get_model_class(model_type: str, task_type: str) -> Type[BaseEstimator]:
    """
    Return the appropriate model class based on task and model type.

    Args:
        model_type: Model type string from config (e.g. "random_forest_regression").
        task_type: Task type string ("regression" or "classification").

    Returns:
        Uninstantiated sklearn model class.

    Raises:
        ModelError: If the model type is not supported for the task type.
    """
    key = (task_type, model_type)
    model_cls = MODEL_REGISTRY.get(key)

    if model_cls is None:
        available = [
            f"{t}:{m}" for (t, m) in MODEL_REGISTRY.keys()
            if t == task_type
        ]
        raise ModelError(
            f"Model '{model_type}' not supported for task '{task_type}'. "
            f"Available for {task_type}: {available}"
        )

    return model_cls


def create_model(
    model_type: str,
    task_type: str,
    model_params: dict | None = None,
) -> BaseEstimator:
    """
    Instantiate a model with the given parameters.

    Args:
        model_type: Model type string from config.
        task_type: Task type string.
        model_params: Dictionary of model hyperparameters.

    Returns:
        Instantiated sklearn model.
    """
    model_cls = get_model_class(model_type, task_type)
    params = model_params or {}

    # Filter params to only those accepted by the model class
    valid_params = model_cls().get_params().keys()
    filtered_params = {k: v for k, v in params.items() if k in valid_params}

    ignored = set(params.keys()) - set(filtered_params.keys())
    if ignored:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.warning(
            f"Ignoring unsupported params for {model_type}: {ignored}"
        )

    return model_cls(**filtered_params)


def list_supported_models(task_type: str | None = None) -> list[str]:
    """
    List all supported model types, optionally filtered by task type.

    Args:
        task_type: Optional task type filter.

    Returns:
        List of "task_type:model_type" strings.
    """
    models = []
    for (t, m) in MODEL_REGISTRY.keys():
        if task_type is None or t == task_type:
            models.append(f"{t}:{m}")
    return models
