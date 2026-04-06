"""
Dataset loading module.

Loads datasets from file paths (CSV, Parquet) with support for
DVC-tracked data paths. Dataset-agnostic by design.
"""

from pathlib import Path

import pandas as pd

from src.pipelines.exceptions import DatasetError
from src.utils.helpers import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Supported file extensions and their pandas readers
_READERS = {
    ".csv": pd.read_csv,
    ".parquet": pd.read_parquet,
    ".json": pd.read_json,
}


def load_dataset(dataset_source: str) -> pd.DataFrame:
    """
    Load a dataset from a file path.

    Supports CSV, Parquet, and JSON formats. Paths are resolved
    relative to the project root if not absolute.

    Args:
        dataset_source: File path string (absolute or relative to project root).

    Returns:
        Loaded pandas DataFrame.

    Raises:
        DatasetError: If the file doesn't exist, has an unsupported format,
                      or cannot be read.
    """
    path = resolve_path(dataset_source)

    if not path.exists():
        raise DatasetError(
            f"Dataset not found: {path}\n"
            f"If using DVC, ensure data is pulled: `dvc pull`"
        )

    suffix = path.suffix.lower()
    reader = _READERS.get(suffix)

    if reader is None:
        raise DatasetError(
            f"Unsupported file format: '{suffix}'. "
            f"Supported: {list(_READERS.keys())}"
        )

    try:
        df = reader(path)
    except Exception as e:
        raise DatasetError(f"Failed to read dataset '{path}': {e}") from e

    logger.info(
        f"Dataset loaded: {path.name} "
        f"({df.shape[0]} rows × {df.shape[1]} columns)"
    )
    return df
