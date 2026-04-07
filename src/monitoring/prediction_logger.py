"""
Prediction logger for monitoring.

Logs all predictions to PostgreSQL for later analysis
by the batch monitoring job and EvidentlyAI drift detection.

Uses SQLAlchemy Core for database interaction via the shared
``database`` module.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import func, select

from src.monitoring.database import (
    ensure_tables,
    get_db_url,
    get_engine,
    predictions_table,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PredictionLogger:
    """
    Log predictions to PostgreSQL for monitoring and drift analysis.

    Stores model name, version, input features, prediction, and timestamp.
    """

    def __init__(self, db_url: str | None = None):
        """
        Initialize prediction logger.

        Args:
            db_url: SQLAlchemy database URL.
                    Defaults to PREDICTION_DB_URL env var or derived from DB_* vars.
        """
        self._db_url = db_url or get_db_url()
        self._engine = get_engine(self._db_url)
        self._init_db()
        logger.info(
            f"Prediction logger initialized: "
            f"{self._engine.url.render_as_string(hide_password=True)}"
        )

    def _init_db(self) -> None:
        """Create the predictions table if it doesn't exist."""
        ensure_tables(self._engine)

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
            with self._engine.begin() as conn:
                conn.execute(
                    predictions_table.insert().values(
                        model_name=model_name,
                        model_version=model_version,
                        features_json=json.dumps(features),
                        prediction=json.dumps(prediction),
                        prediction_proba=(
                            json.dumps(prediction_proba) if prediction_proba else None
                        ),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        latency_seconds=latency_seconds,
                    )
                )
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
        stmt = (
            select(predictions_table)
            .where(predictions_table.c.model_name == model_name)
            .order_by(predictions_table.c.timestamp.desc())
            .limit(limit)
        )

        with self._engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()

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
        if model_name:
            stmt = select(func.count()).select_from(predictions_table).where(
                predictions_table.c.model_name == model_name
            )
        else:
            stmt = select(func.count()).select_from(predictions_table)

        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar() or 0
