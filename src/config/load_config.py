"""
Configuration loader for YAML experiment configs.

Parses YAML files into validated ExperimentConfig objects.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

from src.config.validate_config import ExperimentConfig
from src.pipelines.exceptions import ConfigError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_yaml(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of parsed YAML content.

    Raises:
        ConfigError: If the file cannot be found or parsed.
    """
    path = Path(config_path)

    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    if not path.suffix in (".yaml", ".yml"):
        raise ConfigError(
            f"Config file must be .yaml or .yml, got: {path.suffix}"
        )

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Failed to parse YAML config: {e}") from e

    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file must contain a YAML mapping, got: {type(data).__name__}"
        )

    return data


def load_config(config_path: str | Path) -> ExperimentConfig:
    """
    Load and validate an experiment config from a YAML file.

    This is the primary entry point for config loading. It reads the
    YAML file, parses it into a Pydantic model, and runs all validators.

    Args:
        config_path: Path to the experiment YAML config file.

    Returns:
        Validated ExperimentConfig object.

    Raises:
        ConfigError: If the file is missing, malformed, or fails validation.
    """
    logger.info(f"Loading config from: {config_path}")
    raw = load_yaml(config_path)

    try:
        config = ExperimentConfig(**raw)
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}") from e

    logger.info(
        f"Config loaded: experiment='{config.experiment_name}', "
        f"task={config.task_type}, model={config.model_type}"
    )
    return config


def load_config_from_dict(data: Dict[str, Any]) -> ExperimentConfig:
    """
    Load and validate an experiment config from a dictionary.

    Useful for programmatic config creation (e.g. in tests).

    Args:
        data: Dictionary with config fields.

    Returns:
        Validated ExperimentConfig object.

    Raises:
        ConfigError: If validation fails.
    """
    try:
        return ExperimentConfig(**data)
    except Exception as e:
        raise ConfigError(f"Config validation failed: {e}") from e
