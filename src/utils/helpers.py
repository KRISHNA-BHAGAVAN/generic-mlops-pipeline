"""
Utility helpers for the MLOps pipeline.

Provides common functions used across modules: path resolution,
timestamp formatting, environment loading.
"""

import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv


# Project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_env() -> None:
    """
    Load environment variables from .env file at project root.

    Looks for .env in the project root directory. Does not override
    existing environment variables.
    """
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def resolve_path(path_str: str) -> Path:
    """
    Resolve a path relative to the project root.

    If the path is absolute, return it as-is.
    If relative, resolve it against PROJECT_ROOT.

    Args:
        path_str: Path string (absolute or relative).

    Returns:
        Resolved Path object.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def utc_now_timestamp() -> float:
    """Return current UTC timestamp as float."""
    return datetime.now(timezone.utc).timestamp()


def safe_dict_value(value: Any) -> str | int | float | bool | None:
    """
    Convert a value to an MLflow-safe type for logging.

    MLflow parameters only accept str, int, float, or bool.

    Args:
        value: Any Python value.

    Returns:
        Converted value safe for MLflow logging.
    """
    if isinstance(value, (str, int, float, bool)):
        return value
    if value is None:
        return "None"
    return str(value)


def get_mlflow_tracking_uri() -> str:
    """
    Get the MLflow tracking URI from environment.

    Returns:
        MLflow tracking URI string.

    Raises:
        EnvironmentError: If MLFLOW_TRACKING_URI is not set.
    """
    load_env()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI is not set. "
            "Copy .env.example to .env and fill in your DagsHub credentials."
        )
    return uri


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path
