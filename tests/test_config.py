"""
Tests for the config system (loading, validation).
"""

import pytest
from pathlib import Path

from src.config.load_config import load_config, load_config_from_dict
from src.config.validate_config import ExperimentConfig, validate_config_against_data
from src.pipelines.exceptions import ConfigError

# Path to test configs
REGRESSION_CONFIG = "configs/regression/construction_duration_v1.yaml"
CLASSIFICATION_CONFIG = "configs/classification/construction_risk_v1.yaml"


class TestConfigLoading:
    """Test config file loading."""

    def test_load_regression_config(self):
        config = load_config(REGRESSION_CONFIG)
        assert config.task_type == "regression"
        assert config.model_type == "random_forest_regression"
        assert config.target_column == "Task_Duration_Days"
        assert len(config.feature_columns) == 7

    def test_load_classification_config(self):
        config = load_config(CLASSIFICATION_CONFIG)
        assert config.task_type == "classification"
        assert config.model_type == "random_forest_classification"
        assert config.target_column == "Risk_Level"
        assert len(config.feature_columns) == 8

    def test_missing_file_raises_error(self):
        with pytest.raises(ConfigError):
            load_config("nonexistent.yaml")


class TestConfigValidation:
    """Test config validation rules."""

    _base_config = {
        "experiment_name": "test_exp",
        "user": "tester",
        "task_type": "regression",
        "target_column": "target",
        "feature_columns": ["f1", "f2"],
        "dataset_source": "test.csv",
        "model_type": "linear_regression",
        "metrics": ["mse", "r2"],
    }

    def test_valid_regression_config(self):
        config = load_config_from_dict(self._base_config)
        assert config.experiment_name == "test_exp"
        assert config.test_size == 0.2  # default

    def test_target_in_features_raises_error(self):
        bad = {**self._base_config, "feature_columns": ["target", "f2"]}
        with pytest.raises(ConfigError):
            load_config_from_dict(bad)

    def test_invalid_model_for_task(self):
        bad = {**self._base_config, "model_type": "logistic_regression"}
        with pytest.raises(ConfigError):
            load_config_from_dict(bad)

    def test_invalid_metrics_for_task(self):
        bad = {**self._base_config, "metrics": ["accuracy"]}
        with pytest.raises(ConfigError):
            load_config_from_dict(bad)

    def test_classification_config(self):
        cls_config = {
            **self._base_config,
            "task_type": "classification",
            "model_type": "random_forest_classification",
            "metrics": ["accuracy", "f1"],
        }
        config = load_config_from_dict(cls_config)
        assert config.task_type == "classification"

    def test_registry_optional_metadata_parses(self):
        cfg = {
            **self._base_config,
            "registry_name": "example_model",
            "registry_description": "Sample model release candidate",
            "registry_tags": {"team": "ml", "release": "candidate"},
            "registry_alias": "staging",
            "registry_created_by": "tester",
        }
        config = load_config_from_dict(cfg)
        assert config.registry_name == "example_model"
        assert config.registry_description == "Sample model release candidate"
        assert config.registry_tags["release"] == "candidate"
        assert config.registry_alias == "staging"
        assert config.registry_created_by == "tester"

    def test_temporal_split_requires_date_column(self):
        bad = {**self._base_config, "split_strategy": "temporal"}
        with pytest.raises(ConfigError):
            load_config_from_dict(bad)


class TestConfigAgainstData:
    """Test config validation against actual data."""

    def test_validate_against_real_data(self):
        import pandas as pd
        config = load_config(REGRESSION_CONFIG)
        df = pd.read_csv("data/raw/construction_dataset.csv")
        # Should not raise
        validate_config_against_data(config, df)

    def test_missing_column_raises_error(self):
        import pandas as pd
        config = load_config(REGRESSION_CONFIG)
        df = pd.DataFrame({"col_a": [1, 2, 3]})
        with pytest.raises(ConfigError):
            validate_config_against_data(config, df)
