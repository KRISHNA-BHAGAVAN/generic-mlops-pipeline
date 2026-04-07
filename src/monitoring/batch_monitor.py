"""
Batch monitoring job for drift detection.

Runs periodically to compare recent production predictions against
the training reference data, detects feature drift and prediction drift,
pushes metrics to Prometheus PushGateway, and logs reports to MLflow.

Usage (one-shot CLI — suitable for system cron / K8s CronJob):
    python -m src.monitoring.batch_monitor \\
        --model-name customer_churn \\
        --task-type regression \\
        --reference-data-path data/reference.csv \\
        --feature-columns age,income,tenure \\
        --hours 24
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
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

    Prediction drift is computed by comparing:
      - **reference predictions**: ``model.predict(X_train)`` saved as an
        MLflow artifact at training time via
        ``log_reference_predictions_artifact()``.  This is the *true*
        prediction baseline — NOT ground-truth labels.
      - **current predictions**: values stored by the prediction logger
        from live inference traffic.

    Feature drift is computed separately against the reference CSV
    (the original training feature distribution).
    """

    def __init__(
        self,
        model_name: str,
        task_type: str,
        reference_data_path: str,
        feature_columns: list[str],
        target_column: str | None = None,
        prediction_db_url: str | None = None,
        pushgateway_url: str | None = None,
    ):
        """
        Initialize batch monitor.

        Args:
            model_name: Registered model name to monitor.
            task_type: ``"regression"`` or ``"classification"``.
            reference_data_path: Path to reference (training) data CSV.
                                 Used for feature drift baseline.
            feature_columns: List of feature column names.
            target_column: Target column name (informational, not used for drift).
            prediction_db_url: SQLAlchemy database URL for prediction logs.
            pushgateway_url: Prometheus PushGateway URL.
        """
        self.model_name = model_name
        self.task_type = task_type
        self.reference_data_path = reference_data_path
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.prediction_db_url = prediction_db_url  # None → PredictionLogger resolves from env
        self.pushgateway_url = pushgateway_url or os.getenv(
            "PROMETHEUS_PUSHGATEWAY_URL", "http://localhost:9091"
        )

    def get_reference_data(self) -> pd.DataFrame:
        """
        Load the reference (training) feature dataset from CSV.

        Used as the baseline for **feature drift** detection.

        Returns:
            DataFrame containing only the configured feature columns.
        """
        path = resolve_path(self.reference_data_path)

        if not path.exists():
            raise FileNotFoundError(f"Reference data not found: {path}")

        df = pd.read_csv(path)

        # Filter to feature columns that exist in the CSV
        available_features = [c for c in self.feature_columns if c in df.columns]
        if not available_features:
            raise ValueError(
                f"None of the configured feature columns found in reference CSV: "
                f"{self.feature_columns!r}  (CSV columns: {df.columns.tolist()!r})"
            )

        return df[available_features]

    def _load_reference_predictions_from_mlflow(self) -> pd.DataFrame | None:
        """
        Download reference predictions artifact from MLflow.

        Finds the champion model's source run and downloads the
        ``reference_data/reference_predictions.parquet`` artifact that was
        logged by ``log_reference_predictions_artifact()`` at training time.

        The returned DataFrame contains all feature columns **plus** a
        ``prediction`` column holding ``model.predict(X_train)`` results.

        Returns:
            DataFrame with feature + prediction columns, or ``None`` if:
            - No champion alias is set.
            - The artifact does not exist (model trained before this feature).
            - MLflow is unreachable.
        """
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            from src.models.registry import setup_mlflow

            setup_mlflow()
            client = MlflowClient()

            # Resolve champion alias → version → source run_id
            try:
                model_version = client.get_model_version_by_alias(
                    self.model_name, "champion"
                )
                run_id = model_version.run_id
            except Exception as alias_err:
                logger.warning(
                    f"Could not resolve 'champion' alias for '{self.model_name}': {alias_err}. "
                    "Skipping prediction drift (set champion alias first)."
                )
                return None

            # Download the artifact to a temp directory
            artifact_path = "reference_data/reference_predictions.parquet"
            tmp_dir = tempfile.mkdtemp(prefix="mlops_refpreds_dl_")
            try:
                local_path = client.download_artifacts(
                    run_id, artifact_path, tmp_dir
                )
                ref_df = pd.read_parquet(local_path)
                logger.info(
                    f"Reference predictions loaded from MLflow run {run_id}: "
                    f"{ref_df.shape[0]} rows"
                )
                return ref_df
            except Exception as dl_err:
                logger.warning(
                    f"Reference predictions artifact not found for run {run_id} "
                    f"(path='{artifact_path}'): {dl_err}. "
                    "This is expected for models trained before this feature was added. "
                    "Re-train and register the model to enable prediction drift monitoring."
                )
                return None
            finally:
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            logger.warning(f"MLflow reference predictions download failed: {e}")
            return None

    def get_recent_predictions(self, hours: int = 24) -> tuple[pd.DataFrame, pd.Series]:
        """
        Load recent predictions and extract their features.

        Args:
            hours: Number of hours to look back.

        Returns:
            Tuple of (features_df, predictions_series).
        """
        pred_logger = PredictionLogger(db_url=self.prediction_db_url)
        predictions = pred_logger.get_recent_predictions(
            model_name=self.model_name,
            hours=hours,
        )

        if not predictions:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Extract features and predictions
        features_list = [p["features"] for p in predictions]
        features_df = pd.DataFrame(features_list)
        preds_series = pd.Series([p.get("prediction") for p in predictions])

        # Filter to expected feature columns
        available = [c for c in self.feature_columns if c in features_df.columns]
        return features_df[available], preds_series

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
            # ── Load reference (training) feature data for FEATURE drift ──
            reference_data = self.get_reference_data()
            logger.info(f"Reference feature data shape: {reference_data.shape}")

            # ── Load recent production predictions ──
            recent_features, recent_predictions = self.get_recent_predictions(hours=hours)

            if recent_features.empty:
                logger.warning("No recent predictions found — skipping drift detection")
                return {"status": "skipped", "reason": "no_recent_data"}

            logger.info(f"Recent production data shape: {recent_features.shape}")

            # ── Align columns ──
            common_cols = list(set(reference_data.columns) & set(recent_features.columns))
            if not common_cols:
                logger.warning("No overlapping columns between reference and recent data")
                return {"status": "skipped", "reason": "no_common_columns"}

            reference_data = reference_data[common_cols]
            recent_features = recent_features[common_cols]

            # ── Feature drift detection (DataDriftPreset on features only) ──
            detector = DriftDetector(
                reference_data=reference_data,
                task_type=self.task_type,
            )
            drift_results = detector.detect_data_drift(recent_features)
            logger.info(f"Feature drift detected: {drift_results.get('dataset_drifted')}")

            # ── Prediction drift detection (TargetDriftPreset with ColumnMapping) ──
            # Loads TRUE reference predictions (model.predict on X_train) from MLflow.
            # Falls back gracefully if the artifact is missing (older models).
            pred_drift_results = None
            ref_predictions_df = self._load_reference_predictions_from_mlflow()

            if ref_predictions_df is not None and not recent_predictions.empty:
                # Build current DataFrame: align feature columns + prediction column
                cur_pred_df = recent_features.copy().reset_index(drop=True)
                cur_pred_df["prediction"] = recent_predictions.values

                # Reference df already has both features and prediction column
                # from log_reference_predictions_artifact(). Align feature columns.
                ref_pred_cols = [c for c in common_cols if c in ref_predictions_df.columns]
                ref_pred_df = ref_predictions_df[ref_pred_cols + ["prediction"]].reset_index(drop=True)

                pred_drift_results = detector.detect_prediction_drift(
                    reference_df=ref_pred_df,
                    current_df=cur_pred_df,
                )
                logger.info(
                    f"Prediction drift detected: {pred_drift_results.get('prediction_drifted')} "
                    f"(score={pred_drift_results.get('prediction_drift_score', 0):.4f}, "
                    f"test={pred_drift_results.get('stattest', 'unknown')})"
                )
            else:
                if ref_predictions_df is None:
                    logger.info(
                        "Skipping prediction drift: reference predictions artifact not available. "
                        "Re-train model to enable this check."
                    )
                else:
                    logger.info("Skipping prediction drift: no recent predictions.")

            # ── Push metrics to Prometheus PushGateway ──
            push_drift_metrics_to_prometheus(
                drift_results,
                pred_drift_results=pred_drift_results,
                pushgateway_url=self.pushgateway_url,
            )

            # ── Log to MLflow ──
            self._log_to_mlflow(drift_results, pred_drift_results, len(recent_features))

            # ── Alert if drift detected ──
            feature_drifted = drift_results.get("dataset_drifted", False)
            prediction_drifted = (
                pred_drift_results is not None
                and pred_drift_results.get("prediction_drifted", False)
            )
            if feature_drifted or prediction_drifted:
                self._trigger_alert(drift_results, pred_drift_results)

            return {
                "status": "completed",
                "drift_results": drift_results,
                "prediction_drift_results": pred_drift_results,
                "recent_data_count": len(recent_features),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            logger.error(f"Batch monitoring failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def _log_to_mlflow(
        self,
        drift_results: Dict[str, Any],
        pred_drift_results: Dict[str, Any] | None,
        data_count: int,
    ) -> None:
        """Log monitoring results to MLflow."""
        try:
            import mlflow
            from src.models.registry import setup_mlflow

            setup_mlflow(experiment_name=f"monitoring-{self.model_name}")

            with mlflow.start_run(
                run_name=(
                    f"monitoring-{self.model_name}-"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
                ),
            ):
                mlflow.log_param("model_name", self.model_name)
                mlflow.log_param("task_type", self.task_type)
                mlflow.log_param(
                    "data_drift_detected", drift_results.get("dataset_drifted", False)
                )
                if pred_drift_results:
                    mlflow.log_param(
                        "prediction_drift_detected",
                        pred_drift_results.get("prediction_drifted", False),
                    )

                mlflow.log_metric(
                    "drifted_columns_count",
                    drift_results.get("number_of_drifted_columns", 0),
                )
                mlflow.log_metric("recent_data_count", data_count)
                if pred_drift_results:
                    mlflow.log_metric(
                        "prediction_drift_score",
                        pred_drift_results.get("prediction_drift_score", 0.0),
                    )

                if drift_results.get("drifted_features"):
                    for i, feat in enumerate(drift_results["drifted_features"]):
                        mlflow.set_tag(f"drifted_feature_{i}", feat)

            logger.info("Monitoring results logged to MLflow")

        except Exception as e:
            logger.warning(f"Failed to log monitoring to MLflow: {e}")

    def _trigger_alert(
        self,
        drift_results: Dict[str, Any],
        pred_drift_results: Dict[str, Any] | None,
    ) -> None:
        """
        Trigger an alert when drift is detected.

        Logs to console and optionally sends a Slack webhook if
        ``SLACK_WEBHOOK_URL`` is configured.
        """
        drifted = drift_results.get("drifted_features", [])
        count = drift_results.get("number_of_drifted_columns", 0)

        pred_drift_msg = ""
        if pred_drift_results and pred_drift_results.get("prediction_drifted"):
            pred_drift_msg = (
                f"\n   Prediction Drift : YES "
                f"(score={pred_drift_results.get('prediction_drift_score', 0):.4f}, "
                f"test={pred_drift_results.get('stattest', 'unknown')})"
            )

        alert_msg = (
            f"⚠️  DRIFT ALERT: Model '{self.model_name}'\n"
            f"   Drifted features : {count}  →  {drifted}"
            f"{pred_drift_msg}"
        )
        logger.warning(alert_msg)

        # Send to Slack if webhook is configured
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


# ─────────────────────────────────────────────────────────────────────────────
# One-shot CLI entrypoint
# Suitable for: system cron, K8s CronJob, manual investigation.
# The scheduler daemon (scripts/schedule_monitoring.py) calls monitor.run()
# directly — this block is the "cron hook" for ops tooling.
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--model-name", required=True, help="Registered model name to monitor")
    @click.option(
        "--task-type", default="regression",
        type=click.Choice(["regression", "classification"]),
        help="Task type",
    )
    @click.option(
        "--reference-data-path", required=True,
        help="Path to reference (training) CSV — used for feature drift baseline",
    )
    @click.option(
        "--feature-columns", required=True,
        help="Comma-separated list of feature column names",
    )
    @click.option(
        "--target-column", default=None,
        help="Target column name (informational only, not used for drift checks)",
    )
    @click.option("--hours", default=24, help="Hours of recent predictions to analyze")
    def main(model_name, task_type, reference_data_path, feature_columns, target_column, hours):
        """
        Run batch monitoring for drift detection (one-shot).

        Examples::

            python -m src.monitoring.batch_monitor \\
                --model-name customer_churn \\
                --task-type regression \\
                --reference-data-path data/processed/reference.csv \\
                --feature-columns age,income,tenure \\
                --hours 24
        """
        load_env()
        feats = [f.strip() for f in feature_columns.split(",") if f.strip()]

        monitor = BatchMonitor(
            model_name=model_name,
            task_type=task_type,
            reference_data_path=reference_data_path,
            feature_columns=feats,
            target_column=target_column,
        )

        results = monitor.run(hours=hours)
        print(json.dumps(results, indent=2, default=str))

    main()
