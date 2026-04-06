"""
Batch monitoring job for drift detection.

Runs periodically to compare recent production predictions against
the training reference data, detects drift, pushes metrics to
Prometheus, and logs reports to MLflow.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pandas as pd

from src.monitoring.drift_detector import DriftDetector, push_drift_metrics_to_prometheus
from src.monitoring.prediction_logger import PredictionLogger
from src.utils.helpers import load_env, resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BatchMonitor:
    """
    Batch monitoring job that runs periodically.

    Loads recent predictions, compares them against reference training
    data, and produces drift detection reports.
    """

    def __init__(
        self,
        model_name: str,
        task_type: str,
        reference_data_path: str,
        feature_columns: list[str],
        prediction_db_path: str | None = None,
        pushgateway_url: str | None = None,
    ):
        """
        Initialize batch monitor.

        Args:
            model_name: Registered model name to monitor.
            task_type: "regression" or "classification".
            reference_data_path: Path to reference (training) data CSV.
            feature_columns: List of feature column names.
            prediction_db_path: Path to prediction SQLite database.
            pushgateway_url: Prometheus PushGateway URL.
        """
        self.model_name = model_name
        self.task_type = task_type
        self.reference_data_path = reference_data_path
        self.feature_columns = feature_columns
        self.prediction_db_path = prediction_db_path or os.getenv(
            "PREDICTION_DB_PATH", "predictions.db"
        )
        self.pushgateway_url = pushgateway_url or os.getenv(
            "PROMETHEUS_PUSHGATEWAY_URL", "http://localhost:9091"
        )

    def get_reference_data(self) -> pd.DataFrame:
        """
        Load the reference (training) dataset.

        Returns:
            Reference data as DataFrame with only feature columns.
        """
        path = resolve_path(self.reference_data_path)

        if not path.exists():
            raise FileNotFoundError(f"Reference data not found: {path}")

        df = pd.read_csv(path)

        # Filter to feature columns that exist
        available_features = [c for c in self.feature_columns if c in df.columns]
        return df[available_features]

    def get_recent_predictions(self, hours: int = 24) -> pd.DataFrame:
        """
        Load recent predictions and extract their features.

        Args:
            hours: Number of hours to look back.

        Returns:
            DataFrame of recent prediction features.
        """
        pred_logger = PredictionLogger(self.prediction_db_path)
        predictions = pred_logger.get_recent_predictions(
            model_name=self.model_name,
            hours=hours,
        )

        if not predictions:
            return pd.DataFrame()

        # Extract features from prediction records
        features_list = [p["features"] for p in predictions]
        features_df = pd.DataFrame(features_list)

        # Filter to expected feature columns
        available = [c for c in self.feature_columns if c in features_df.columns]
        return features_df[available]

    def run(self, hours: int = 24) -> Dict[str, Any]:
        """
        Execute the batch monitoring job.

        Args:
            hours: Hours of prediction data to analyze.

        Returns:
            Dictionary with monitoring results.
        """
        logger.info(f"Starting batch monitoring for model: {self.model_name}")

        try:
            # Load reference data
            reference_data = self.get_reference_data()
            logger.info(f"Reference data: {reference_data.shape}")

            # Load recent predictions
            recent_data = self.get_recent_predictions(hours=hours)

            if recent_data.empty:
                logger.warning("No recent predictions found — skipping drift detection")
                return {"status": "skipped", "reason": "no_recent_data"}

            logger.info(f"Recent predictions: {recent_data.shape}")

            # Ensure columns match
            common_cols = list(set(reference_data.columns) & set(recent_data.columns))
            if not common_cols:
                logger.warning("No overlapping columns between reference and recent data")
                return {"status": "skipped", "reason": "no_common_columns"}

            reference_data = reference_data[common_cols]
            recent_data = recent_data[common_cols]

            # Run drift detection
            detector = DriftDetector(
                reference_data=reference_data,
                task_type=self.task_type,
            )

            drift_results = detector.detect_data_drift(recent_data)
            logger.info(f"Drift results: {drift_results}")

            # Push metrics to Prometheus
            push_drift_metrics_to_prometheus(drift_results, self.pushgateway_url)

            # Log to MLflow
            self._log_to_mlflow(drift_results, len(recent_data))

            # Alert if drift detected
            if drift_results.get("dataset_drifted", False):
                self._trigger_alert(drift_results)

            return {
                "status": "completed",
                "drift_results": drift_results,
                "recent_data_count": len(recent_data),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Batch monitoring failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _log_to_mlflow(self, drift_results: Dict[str, Any], data_count: int) -> None:
        """Log monitoring results to MLflow."""
        try:
            import mlflow
            from src.models.registry import setup_mlflow

            setup_mlflow(experiment_name=f"monitoring-{self.model_name}")

            with mlflow.start_run(
                run_name=f"monitoring-{self.model_name}-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}",
            ):
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("task_type", self.task_type)
                mlflow.log_param("drift_detected", drift_results.get("dataset_drifted", False))

                mlflow.log_metric("drifted_columns_count", drift_results.get("number_of_drifted_columns", 0))
                mlflow.log_metric("recent_data_count", data_count)

                if drift_results.get("drifted_features"):
                    for i, feat in enumerate(drift_results["drifted_features"]):
                        mlflow.set_tag(f"drifted_feature_{i}", feat)

            logger.info("Monitoring results logged to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log monitoring to MLflow: {e}")

    def _trigger_alert(self, drift_results: Dict[str, Any]) -> None:
        """
        Trigger an alert when drift is detected.

        Currently logs to console. Designed for future Slack webhook integration
        via SLACK_WEBHOOK_URL environment variable.
        """
        drifted = drift_results.get("drifted_features", [])
        count = drift_results.get("number_of_drifted_columns", 0)

        alert_msg = (
            f"⚠️  DRIFT ALERT: Model '{self.model_name}'\n"
            f"   Drifted columns: {count}\n"
            f"   Features: {drifted}"
        )
        logger.warning(alert_msg)

        # Future: Send to Slack if webhook URL is configured
        slack_url = os.getenv("SLACK_WEBHOOK_URL")
        if slack_url:
            try:
                import httpx
                httpx.post(slack_url, json={
                    "text": alert_msg,
                    "channel": "#ml-alerts",
                })
            except Exception as e:
                logger.warning(f"Failed to send Slack alert: {e}")
