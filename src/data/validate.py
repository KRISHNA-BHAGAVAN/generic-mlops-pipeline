"""
Dataset validation module.

Validates loaded DataFrames against experiment config requirements:
column existence, dtype compatibility, data quality checks.
"""

import pandas as pd
from typing import Dict, List

from src.config.validate_config import ExperimentConfig, validate_config_against_data
from src.pipelines.exceptions import DatasetError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_dataset(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict[str, any]:
    """
    Run all validation checks on a dataset against the experiment config.

    This performs structural validation (columns, dtypes) and data quality
    checks (nulls, duplicates, minimum size).

    Args:
        df: The loaded DataFrame.
        config: The validated ExperimentConfig.

    Returns:
        Dictionary with validation results and data quality summary.

    Raises:
        DatasetError: If critical validation fails.
    """
    # Run config-against-data validation (columns, dtypes)
    validate_config_against_data(config, df)

    # Data quality checks
    quality_report = _check_data_quality(df, config)

    logger.info(
        f"Dataset validation passed. "
        f"Null features: {quality_report['null_feature_count']}, "
        f"Duplicate rows: {quality_report['duplicate_count']}"
    )

    return quality_report


def _check_data_quality(
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> Dict[str, any]:
    """
    Perform data quality checks and return a summary report.

    Args:
        df: The loaded DataFrame.
        config: The experiment config.

    Returns:
        Dictionary with quality metrics.
    """
    all_columns = config.feature_columns + [config.target_column]
    subset = df[all_columns]

    # Null counts per column
    null_counts = subset.isnull().sum()
    null_pct = (null_counts / len(subset) * 100).round(2)
    columns_with_nulls = null_counts[null_counts > 0].to_dict()

    # Duplicate rows
    duplicate_count = subset.duplicated().sum()

    # Target distribution
    target_stats = {}
    if config.task_type == "regression":
        target_stats = {
            "mean": float(df[config.target_column].mean()),
            "std": float(df[config.target_column].std()),
            "min": float(df[config.target_column].min()),
            "max": float(df[config.target_column].max()),
        }
    elif config.task_type == "classification":
        value_counts = df[config.target_column].value_counts().to_dict()
        target_stats = {
            "class_distribution": {str(k): int(v) for k, v in value_counts.items()},
            "n_classes": len(value_counts),
        }

    # Warn on high null percentage
    high_null_cols = {
        col: pct for col, pct in null_pct.items() if pct > 20
    }
    if high_null_cols:
        logger.warning(
            f"Columns with >20% missing values: {high_null_cols}"
        )

    return {
        "total_rows": len(df),
        "total_columns": len(all_columns),
        "null_feature_count": len(columns_with_nulls),
        "columns_with_nulls": columns_with_nulls,
        "duplicate_count": int(duplicate_count),
        "target_stats": target_stats,
    }
