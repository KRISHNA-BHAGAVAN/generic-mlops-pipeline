"""
Prediction logger for monitoring.

Logs all predictions to a SQLite database for later analysis
by the batch monitoring job and EvidentlyAI drift detection.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils.helpers import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionLogger:
    """
    Log predictions to SQLite for monitoring and drift analysis.

    Stores model name, version, input features, prediction, and timestamp.
    """

    def __init__(self, db_path: str | None = None):
        """
        Initialize prediction logger.

        Args:
            db_path: Path to SQLite database file.
                     Defaults to PREDICTION_DB_PATH env var or predictions.db in project root.
        """
        if db_path is None:
            db_path = os.getenv("PREDICTION_DB_PATH", str(PROJECT_ROOT / "predictions.db"))

        self.db_path = db_path
        self._init_db()
        logger.info(f"Prediction logger initialized: {self.db_path}")

    def _init_db(self) -> None:
        """Create the predictions table if it doesn't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    prediction_proba TEXT,
                    timestamp TEXT NOT NULL,
                    latency_seconds REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_predictions_model
                ON predictions(model_name, timestamp)
            """)
            conn.commit()

    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        features: Dict[str, Any],
        prediction: Union[float, int, str],
        prediction_proba: Optional[Dict[str, float]] = None,
        latency_seconds: Optional[float] = None,
    ) -> None:
        """
        Log a single prediction.

        Args:
            model_name: Registered model name.
            model_version: Model version string.
            features: Input feature dictionary.
            prediction: Model prediction.
            prediction_proba: Optional class probabilities.
            latency_seconds: Optional inference latency.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO predictions
                    (model_name, model_version, features_json, prediction,
                     prediction_proba, timestamp, latency_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        model_name,
                        model_version,
                        json.dumps(features),
                        json.dumps(prediction),
                        json.dumps(prediction_proba) if prediction_proba else None,
                        datetime.now(timezone.utc).isoformat(),
                        latency_seconds,
                    ),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log prediction: {e}")

    def get_recent_predictions(
        self,
        model_name: str,
        hours: int = 24,
        limit: int = 10000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recent predictions for a model.

        Args:
            model_name: Model name to filter by.
            hours: Number of hours to look back.
            limit: Maximum number of records to return.

        Returns:
            List of prediction dictionaries.
        """
        cutoff = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM predictions
                WHERE model_name = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (model_name, limit),
            )
            rows = cursor.fetchall()

        predictions = []
        for row in rows:
            pred = dict(row)
            pred["features"] = json.loads(pred.pop("features_json"))
            pred["prediction"] = json.loads(pred["prediction"])
            if pred["prediction_proba"]:
                pred["prediction_proba"] = json.loads(pred["prediction_proba"])
            predictions.append(pred)

        return predictions

    def get_prediction_count(self, model_name: str | None = None) -> int:
        """Get total prediction count, optionally filtered by model."""
        with sqlite3.connect(self.db_path) as conn:
            if model_name:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM predictions WHERE model_name = ?",
                    (model_name,),
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM predictions")
            return cursor.fetchone()[0]
