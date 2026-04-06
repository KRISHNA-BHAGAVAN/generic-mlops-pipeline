"""
End-to-end pipeline tests.

Tests the full pipeline flow without MLflow (data loading → preprocessing
→ training → evaluation) for both regression and classification.
"""

import pytest
import pandas as pd

from src.config.load_config import load_config
from src.data.load_data import load_dataset
from src.data.validate import validate_dataset
from src.features.build_features import prepare_features, split_data
from src.models.train import train_model
from src.models.evaluate import evaluate_model, generate_evaluation_plots


REGRESSION_CONFIG = "configs/regression/construction_duration_v1.yaml"
CLASSIFICATION_CONFIG = "configs/classification/construction_risk_v1.yaml"


class TestRegressionPipeline:
    """End-to-end regression pipeline test."""

    def test_full_regression_pipeline(self):
        # Step 1: Load config
        config = load_config(REGRESSION_CONFIG)
        assert config.task_type == "regression"

        # Step 2: Load data
        df = load_dataset(config.dataset_source)
        assert len(df) > 0

        # Step 3: Validate
        quality = validate_dataset(df, config)
        assert quality["total_rows"] > 0

        # Step 4: Prepare features
        X, y, artifacts = prepare_features(df, config)
        assert X.shape[1] >= len(config.feature_columns)

        # Step 5: Split
        X_train, X_test, y_train, y_test = split_data(X, y, config)
        assert len(X_train) > len(X_test)

        # Step 6: Train
        model, metadata = train_model(X_train, y_train, config)
        assert metadata["task_type"] == "regression"

        # Step 7: Evaluate
        metrics = evaluate_model(model, X_test, y_test, config)
        assert "r2" in metrics
        assert metrics["r2"] > -1  # Sanity check

        # Step 8: Generate plots
        y_pred = model.predict(X_test)
        plots = generate_evaluation_plots(model, X_test, y_test, y_pred, config)
        assert len(plots) > 0


class TestClassificationPipeline:
    """End-to-end classification pipeline test."""

    def test_full_classification_pipeline(self):
        # Step 1: Load config
        config = load_config(CLASSIFICATION_CONFIG)
        assert config.task_type == "classification"

        # Step 2: Load data
        df = load_dataset(config.dataset_source)
        assert len(df) > 0

        # Step 3: Validate
        quality = validate_dataset(df, config)
        assert quality["target_stats"]["n_classes"] == 3

        # Step 4: Prepare features
        X, y, artifacts = prepare_features(df, config)
        assert "target_label_encoder" in artifacts

        # Step 5: Split
        X_train, X_test, y_train, y_test = split_data(X, y, config)

        # Step 6: Train
        model, metadata = train_model(X_train, y_train, config)
        assert metadata["task_type"] == "classification"
        assert metadata["n_classes"] == 3

        # Step 7: Evaluate
        metrics = evaluate_model(model, X_test, y_test, config)
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert metrics["accuracy"] > 0.3  # Should beat random

        # Step 8: Generate plots
        y_pred = model.predict(X_test)
        plots = generate_evaluation_plots(model, X_test, y_test, y_pred, config)
        assert "confusion_matrix" in plots
        assert "feature_importance" in plots


class TestDataLoading:
    """Test data loading module."""

    def test_load_csv(self):
        df = load_dataset("data/raw/construction_dataset.csv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1300
        assert "Task_Duration_Days" in df.columns
        assert "Risk_Level" in df.columns


class TestFeatureDtypes:
    def test_prepare_features_casts_integers_to_float64(self):
        config = load_config(REGRESSION_CONFIG)
        df = load_dataset(config.dataset_source)
        X, _, _ = prepare_features(df, config)

        int_cols = X.select_dtypes(include=["int", "uint", "Int64"]).columns.tolist()
        assert int_cols == []
