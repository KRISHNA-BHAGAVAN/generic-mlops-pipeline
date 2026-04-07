"""
Tests for monitoring components: PredictionLogger, DriftDetector, BatchMonitor.

PredictionLogger and BatchMonitor tests require a running PostgreSQL instance.
Start one with:
    docker compose -f deployment/docker-compose.yml up -d postgres

The tests use a dedicated ``mlops_test`` database that is created/destroyed
per test session to guarantee isolation.
"""

import os
import json
import pandas as pd
import pytest
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from src.monitoring.drift_detector import DriftDetector
from src.monitoring.prediction_logger import PredictionLogger
from src.monitoring.batch_monitor import BatchMonitor
from src.monitoring.database import get_db_url, metadata


# ── Postgres test fixtures ──────────────────────────────────────────────────

def _test_db_url() -> str:
    """
    Build a test database URL.

    Uses PREDICTION_DB_URL from env but targets the ``mlops_test`` database.
    Falls back to the default local Docker Postgres.
    """
    base_url = os.getenv(
        "TEST_DB_URL",
        "postgresql+psycopg://mlops:mlops@localhost:5432/mlops_test",
    )
    return base_url


def _ensure_test_database():
    """
    Create the ``mlops_test`` database if it doesn't exist.

    Connects to the default ``mlops`` database to issue CREATE DATABASE.
    """
    admin_url = os.getenv(
        "PREDICTION_DB_URL",
        "postgresql+psycopg://mlops:mlops@localhost:5432/mlops",
    )
    engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = 'mlops_test'")
            )
            if not result.fetchone():
                conn.execute(text("CREATE DATABASE mlops_test"))
    finally:
        engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """Session-scoped fixture: create test DB once, clean up after all tests."""
    try:
        _ensure_test_database()
    except Exception as e:
        pytest.skip(f"PostgreSQL not available for tests: {e}")


@pytest.fixture
def test_db_url():
    """Per-test fixture providing the test database URL."""
    url = _test_db_url()
    # Create tables
    engine = create_engine(url)
    try:
        metadata.create_all(engine)
        yield url
    finally:
        # Clean up all rows after each test
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM predictions"))
        engine.dispose()


# ── PredictionLogger Tests ──────────────────────────────────────────────────

class TestPredictionLogger:
    def test_init_db(self, test_db_url):
        logger = PredictionLogger(db_url=test_db_url)
        # Table should exist — verify by counting (should be 0)
        count = logger.get_prediction_count()
        assert count == 0

    def test_log_and_get_predictions(self, test_db_url):
        logger = PredictionLogger(db_url=test_db_url)

        logger.log_prediction(
            model_name="test_model",
            model_version="1",
            features={"feat1": 1.0, "feat2": 2.0},
            prediction=0.8,
            latency_seconds=0.05,
        )

        preds = logger.get_recent_predictions("test_model", hours=2)
        assert len(preds) == 1
        assert preds[0]["model_name"] == "test_model"
        assert preds[0]["features"] == {"feat1": 1.0, "feat2": 2.0}
        assert preds[0]["prediction"] == 0.8

    def test_prediction_count(self, test_db_url):
        logger = PredictionLogger(db_url=test_db_url)

        # Log two predictions
        for i in range(2):
            logger.log_prediction(
                model_name="count_model",
                model_version="1",
                features={"x": float(i)},
                prediction=float(i),
            )

        assert logger.get_prediction_count("count_model") == 2
        assert logger.get_prediction_count("nonexistent") == 0


# ── DriftDetector Tests ──────────────────────────────────────────────────────

class TestDriftDetector:
    def test_basic_drift_check(self):
        # Create Reference Data
        ref_df = pd.DataFrame(
            {"feat1": [1, 2, 3, 4, 5], "feat2": [10, 20, 30, 40, 50]}
        )

        # Create Current Data (Drifted)
        cur_df = pd.DataFrame(
            {
                "feat1": [100, 200, 300, 400, 500],  # drastically drifted
                "feat2": [12, 22, 32, 42, 52],  # slightly shifted, not drifted
            }
        )

        detector = DriftDetector(reference_data=ref_df, task_type="regression")
        results = detector._basic_drift_check(cur_df)

        assert results["dataset_drifted"] is True
        assert "feat1" in results["drifted_features"]
        assert "feat2" not in results["drifted_features"]
        assert results["number_of_drifted_columns"] >= 1
        assert "feat1" in results["drift_scores"]


# ── BatchMonitor Tests ───────────────────────────────────────────────────────

class TestBatchMonitor:
    def test_batch_monitor_init(self, test_db_url):
        monitor = BatchMonitor(
            model_name="test_model",
            task_type="regression",
            reference_data_path="dummy.csv",
            feature_columns=["feat1"],
            prediction_db_url=test_db_url,
        )
        assert monitor.model_name == "test_model"
        assert monitor.prediction_db_url == test_db_url
