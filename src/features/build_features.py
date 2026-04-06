"""
Feature engineering / preprocessing module.

Applies config-driven preprocessing steps (normalization, encoding, etc.)
and splits data into train/test sets. Dataset-agnostic — all column
references come from config.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)

from src.config.validate_config import ExperimentConfig
from src.pipelines.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Preprocessing registry ────────────────────────────────────────────────

def _normalize(df: pd.DataFrame, columns: list[str], **kwargs) -> Tuple[pd.DataFrame, Any]:
    """Min-Max normalize specified columns to [0, 1]."""
    scaler = MinMaxScaler()
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def _standardize(df: pd.DataFrame, columns: list[str], **kwargs) -> Tuple[pd.DataFrame, Any]:
    """Standardize specified columns to zero mean and unit variance."""
    scaler = StandardScaler()
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df, scaler


def _one_hot_encode(df: pd.DataFrame, columns: list[str], **kwargs) -> Tuple[pd.DataFrame, Any]:
    """One-hot encode categorical columns, dropping the originals."""
    df = df.copy()
    df = pd.get_dummies(df, columns=columns, drop_first=False, dtype=float)
    return df, {"encoded_columns": columns}


def _label_encode(df: pd.DataFrame, columns: list[str], **kwargs) -> Tuple[pd.DataFrame, dict]:
    """Label encode categorical columns to integers."""
    df = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders


PREPROCESSING_REGISTRY: Dict[str, callable] = {
    "normalize": _normalize,
    "standardize": _standardize,
    "one_hot_encode": _one_hot_encode,
    "label_encode": _label_encode,
}

def _cast_integer_features_to_float64(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cast integer-typed feature columns to float64.

    Rationale: integer columns cannot represent missing values (NaN) in
    NumPy/Pandas without upcasting to float at runtime. MLflow signature
    inference will lock the schema based on training samples; if inference-time
    data contains missing values, pandas will upcast int->float and MLflow
    schema enforcement can fail. Casting to float64 during training avoids this.
    """
    X = X.copy()
    for col in X.columns:
        # Exclude boolean; cast any (nullable) integer dtype to float64.
        if pd.api.types.is_integer_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            X[col] = X[col].astype("float64")
    return X


# ─── Public API ─────────────────────────────────────────────────────────────

def prepare_features(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Extract features and target, apply preprocessing steps.

    Args:
        df: Raw DataFrame loaded from dataset.
        config: Experiment configuration specifying feature columns,
                target column, and preprocessing steps.

    Returns:
        Tuple of:
        - X: Preprocessed feature DataFrame.
        - y: Target Series.
        - preprocessing_artifacts: Dict of fitted preprocessors for reproducibility.

    Raises:
        DatasetError: If preprocessing fails.
    """
    logger.info("Preparing features...")

    # Extract features and target
    X = df[config.feature_columns].copy()
    y = df[config.target_column].copy()

    preprocessing_artifacts: Dict[str, Any] = {}

    # Apply preprocessing steps from config
    if config.preprocessing:
        for i, step in enumerate(config.preprocessing):
            step_fn = PREPROCESSING_REGISTRY.get(step.type)
            if step_fn is None:
                raise DatasetError(
                    f"Unknown preprocessing type: '{step.type}'. "
                    f"Supported: {list(PREPROCESSING_REGISTRY.keys())}"
                )

            # Determine which columns to transform
            columns = step.columns
            if columns is None:
                # Default: apply to all numeric columns for scaler ops,
                # all object columns for encoding ops
                if step.type in ("normalize", "standardize"):
                    columns = X.select_dtypes(include=[np.number]).columns.tolist()
                elif step.type in ("one_hot_encode", "label_encode"):
                    columns = X.select_dtypes(include=["object", "category"]).columns.tolist()

            if not columns:
                logger.warning(f"Preprocessing step '{step.type}' has no applicable columns — skipping.")
                continue

            # Filter to columns that actually exist in X
            existing = [c for c in columns if c in X.columns]
            if not existing:
                logger.warning(
                    f"Preprocessing step '{step.type}' columns {columns} "
                    f"not found in features — skipping."
                )
                continue

            try:
                params = step.params or {}
                X, artifact = step_fn(X, existing, **params)
                preprocessing_artifacts[f"step_{i}_{step.type}"] = artifact
                logger.info(f"Applied '{step.type}' to columns: {existing}")
            except Exception as e:
                raise DatasetError(
                    f"Preprocessing step '{step.type}' failed: {e}"
                ) from e
    else:
        # Auto-preprocessing: encode any remaining object columns
        obj_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        if obj_cols:
            logger.info(f"Auto-encoding categorical columns: {obj_cols}")
            X, encoders = _label_encode(X, obj_cols)
            preprocessing_artifacts["auto_label_encode"] = encoders

    # Encode classification target if it's string-based
    if config.task_type == "classification" and y.dtype == "object":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=y.name, index=y.index)
        preprocessing_artifacts["target_label_encoder"] = le
        logger.info(f"Encoded target classes: {list(le.classes_)}")

    # Handle any remaining NaN values with median/mode imputation
    if X.isnull().any().any():
        logger.warning("Filling NaN values with median (numeric) / mode (categorical)")
        for col in X.columns:
            if X[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown")

    # Ensure stable MLflow schema when integer columns may be missing at inference
    X = _cast_integer_features_to_float64(X)

    logger.info(f"Features prepared: {X.shape[1]} features, {len(y)} samples")

    return X, y, preprocessing_artifacts


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets based on config strategy.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        config: Experiment configuration with split settings.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    stratify = None
    if config.split_strategy == "stratified" and config.task_type == "classification":
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    logger.info(
        f"Data split: train={len(X_train)}, test={len(X_test)} "
        f"(strategy={config.split_strategy}, test_size={config.test_size})"
    )

    return X_train, X_test, y_train, y_test
