"""
Experiment configuration schema and validation rules.

Uses Pydantic v2 for robust config validation. Every experiment
is fully defined by a YAML config file parsed into an ExperimentConfig.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from src.pipelines.exceptions import ConfigError


# ─── Enums ──────────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    """Supported ML task types."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class SplitStrategy(str, Enum):
    """Supported data-splitting strategies."""
    RANDOM = "random"
    TEMPORAL = "temporal"
    STRATIFIED = "stratified"

class ModelSerializationFormat(str, Enum):
    """Supported MLflow sklearn serialization formats."""
    PICKLE = "pickle"
    CLOUDPICKLE = "cloudpickle"
    SKOPS = "skops"


# ─── Model-type ↔ task-type mapping ─────────────────────────────────────────

ALLOWED_MODELS: dict[str, list[str]] = {
    "regression": [
        "linear_regression",
        "random_forest_regression",
    ],
    "classification": [
        "logistic_regression",
        "random_forest_classification",
    ],
}

ALLOWED_METRICS: dict[str, list[str]] = {
    "regression": ["mse", "rmse", "mae", "r2"],
    "classification": ["accuracy", "precision", "recall", "f1", "auc"],
}


# ─── Sub-models ─────────────────────────────────────────────────────────────

class PreprocessingStep(BaseModel):
    """A single preprocessing step to apply to the data."""
    type: str  # "normalize", "standardize", "one_hot_encode", "label_encode"
    columns: Optional[List[str]] = None
    params: Optional[Dict[str, Any]] = None


# ─── Main config model ──────────────────────────────────────────────────────

class ExperimentConfig(BaseModel):
    """
    Complete experiment configuration.

    Every field needed by the pipeline is declared here so that
    experiments change through YAML configs, never through code edits.
    """

    # Metadata
    experiment_name: str = Field(..., min_length=1, max_length=100)
    user: str = Field(..., min_length=1, max_length=50)

    # Task definition
    task_type: TaskType
    target_column: str
    feature_columns: List[str] = Field(..., min_length=1)

    # Dataset
    dataset_source: str  # File path or DVC reference
    dvc_version: Optional[str] = None

    # Model
    model_type: str
    model_params: Dict[str, Any] = Field(default_factory=dict)

    # Preprocessing
    preprocessing: Optional[List[PreprocessingStep]] = None

    # Splitting
    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    test_size: float = Field(0.2, ge=0.01, le=0.5)
    val_size: float = Field(0.0, ge=0.0, le=0.5)
    random_state: int = 42
    date_column: Optional[str] = None  # For temporal split

    # Metrics
    metrics: List[str] = Field(..., min_length=1)

    # MLflow
    mlflow_tags: Dict[str, str] = Field(default_factory=dict)
    experiment_description: Optional[str] = None
    experiment_tags: Dict[str, str] = Field(default_factory=dict)
    run_name: Optional[str] = None
    run_description: Optional[str] = None
    dataset_name: Optional[str] = None
    dataset_source_uri: Optional[str] = None
    dataset_context: str = "training"
    dataset_tags: Dict[str, str] = Field(default_factory=dict)

    # Registry
    registry_name: Optional[str] = None
    serialization_format: ModelSerializationFormat = ModelSerializationFormat.CLOUDPICKLE

    model_config = {"use_enum_values": True}

    # ── Validators ──────────────────────────────────────────────────────

    @field_validator("model_type")
    @classmethod
    def validate_model_for_task(cls, v: str, info) -> str:
        """Model type must be valid for the chosen task type."""
        task = info.data.get("task_type")
        if task is None:
            return v

        task_str = task if isinstance(task, str) else task.value
        allowed = ALLOWED_MODELS.get(task_str, [])
        if v not in allowed:
            raise ValueError(
                f"Model '{v}' not supported for task '{task_str}'. "
                f"Allowed: {allowed}"
            )
        return v

    @field_validator("metrics")
    @classmethod
    def validate_metrics_for_task(cls, v: list[str], info) -> list[str]:
        """All requested metrics must be valid for the task type."""
        task = info.data.get("task_type")
        if task is None:
            return v

        task_str = task if isinstance(task, str) else task.value
        allowed = ALLOWED_METRICS.get(task_str, [])
        invalid = [m for m in v if m not in allowed]
        if invalid:
            raise ValueError(
                f"Metrics {invalid} not allowed for task '{task_str}'. "
                f"Allowed: {allowed}"
            )
        return v

    @model_validator(mode="after")
    def cross_field_validation(self) -> "ExperimentConfig":
        """Validate cross-field constraints after all fields are set."""
        # Target column must not be in feature columns
        if self.target_column in self.feature_columns:
            raise ValueError(
                f"target_column '{self.target_column}' cannot be in feature_columns"
            )
        # Temporal split requires date_column
        strategy_str = self.split_strategy if isinstance(self.split_strategy, str) else self.split_strategy.value
        if strategy_str == "temporal" and not self.date_column:
            raise ValueError(
                "date_column is required when split_strategy is 'temporal'"
            )
        return self


# ─── Config-against-data validation ─────────────────────────────────────────

def validate_config_against_data(
    config: ExperimentConfig,
    df: pd.DataFrame,
) -> None:
    """
    Validate an experiment config against the actual dataset.

    Checks that target and feature columns exist, and that the
    target column's dtype is compatible with the task type.

    Args:
        config: Validated ExperimentConfig.
        df: The loaded dataset as a DataFrame.

    Raises:
        ConfigError: If any validation check fails.
    """
    # Check target column exists
    if config.target_column not in df.columns:
        raise ConfigError(
            f"Target column '{config.target_column}' not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Check feature columns exist
    missing = set(config.feature_columns) - set(df.columns)
    if missing:
        raise ConfigError(
            f"Feature columns {missing} not found in dataset. "
            f"Available columns: {list(df.columns)}"
        )

    # Check target dtype matches task type
    target_dtype = df[config.target_column].dtype

    if config.task_type == "regression":
        if not pd.api.types.is_numeric_dtype(target_dtype):
            raise ConfigError(
                f"Regression requires numeric target, "
                f"got '{target_dtype}' for column '{config.target_column}'"
            )
    elif config.task_type == "classification":
        # Classification targets can be object/categorical or numeric with few unique values
        n_unique = df[config.target_column].nunique()
        if pd.api.types.is_numeric_dtype(target_dtype) and n_unique > 50:
            raise ConfigError(
                f"Classification target '{config.target_column}' has "
                f"{n_unique} unique values, which looks like a regression target. "
                f"Use task_type='regression' or reduce cardinality."
            )

    # Check minimum rows
    if len(df) < 10:
        raise ConfigError(
            f"Dataset has only {len(df)} rows. Minimum 10 required."
        )
