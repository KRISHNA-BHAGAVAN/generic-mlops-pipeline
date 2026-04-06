"""
Tests for model factory, training, and evaluation.
"""

import pytest
import numpy as np
import pandas as pd

from src.config.load_config import load_config
from src.data.load_data import load_dataset
from src.features.build_features import prepare_features, split_data
from src.models.factory import create_model, get_model_class, list_supported_models
from src.models.train import train_model
from src.models.evaluate import evaluate, evaluate_model
from src.pipelines.exceptions import ModelError


REGRESSION_CONFIG = "configs/regression/construction_duration_v1.yaml"
CLASSIFICATION_CONFIG = "configs/classification/construction_risk_v1.yaml"


class TestModelFactory:
    """Test model factory."""

    def test_list_supported_models(self):
        models = list_supported_models()
        assert len(models) >= 4

    def test_list_regression_models(self):
        models = list_supported_models("regression")
        assert "regression:linear_regression" in models
        assert "regression:random_forest_regression" in models

    def test_create_regression_model(self):
        model = create_model("random_forest_regression", "regression", {"n_estimators": 10})
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_create_classification_model(self):
        model = create_model("random_forest_classification", "classification")
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")

    def test_invalid_model_raises_error(self):
        with pytest.raises(ModelError):
            get_model_class("xgboost", "regression")


class TestTraining:
    """Test model training."""

    def test_train_regression(self):
        config = load_config(REGRESSION_CONFIG)
        df = load_dataset(config.dataset_source)
        X, y, _ = prepare_features(df, config)
        X_train, X_test, y_train, y_test = split_data(X, y, config)

        model, metadata = train_model(X_train, y_train, config)

        assert hasattr(model, "predict")
        assert metadata["task_type"] == "regression"
        assert metadata["training_duration_seconds"] > 0
        assert metadata["training_samples"] == len(X_train)

    def test_train_classification(self):
        config = load_config(CLASSIFICATION_CONFIG)
        df = load_dataset(config.dataset_source)
        X, y, _ = prepare_features(df, config)
        X_train, X_test, y_train, y_test = split_data(X, y, config)

        model, metadata = train_model(X_train, y_train, config)

        assert hasattr(model, "predict")
        assert metadata["task_type"] == "classification"
        assert "classes" in metadata
        assert len(metadata["classes"]) == 3  # Low, Medium, High


class TestEvaluation:
    """Test model evaluation."""

    def test_evaluate_regression_metrics(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 5.1])

        results = evaluate("regression", y_true, y_pred, ["mse", "rmse", "mae", "r2"])

        assert "mse" in results
        assert "rmse" in results
        assert "mae" in results
        assert "r2" in results
        assert results["r2"] > 0.9

    def test_evaluate_classification_metrics(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])

        results = evaluate(
            "classification", y_true, y_pred, ["accuracy", "precision", "recall", "f1"]
        )

        assert "accuracy" in results
        assert results["accuracy"] > 0.5

    def test_full_regression_evaluation(self):
        config = load_config(REGRESSION_CONFIG)
        df = load_dataset(config.dataset_source)
        X, y, _ = prepare_features(df, config)
        X_train, X_test, y_train, y_test = split_data(X, y, config)

        model, _ = train_model(X_train, y_train, config)
        metrics = evaluate_model(model, X_test, y_test, config)

        assert "r2" in metrics
        assert metrics["r2"] > -1  # Sanity: model runs and produces result

    def test_invalid_task_type_raises_error(self):
        with pytest.raises(ModelError):
            evaluate("unknown", np.array([1]), np.array([1]), ["mse"])
