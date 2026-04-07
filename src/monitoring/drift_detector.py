"""
EvidentlyAI-based drift detection module.

Detects data drift, prediction drift, and data quality issues
by comparing production data against training reference data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DriftDetector:
    """
    Detect data drift and quality issues using EvidentlyAI.

    Compares a reference dataset (training data) against current
    production data to identify statistical distribution shifts.
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        task_type: str = "regression",
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Training/baseline feature data (features only,
                            no prediction or target columns).
            task_type: ``"regression"`` or ``"classification"``.
        """
        self.reference_data = reference_data
        self.task_type = task_type

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Detect drift in feature distributions using DataDriftPreset.

        Args:
            current_data: Production data to check (same feature columns as
                          reference, no prediction/target columns).
            threshold: Drift detection significance threshold (0-1).

        Returns:
            Dictionary with drift detection results.
        """
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report([
                DataDriftPreset(stattest_threshold=threshold)
            ])

            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
            )

            drift_dict = report.as_dict()
            metrics_result = drift_dict["metrics"][0]["result"]

            drifted_features = []
            drift_scores = {}
            for col_info in metrics_result.get("drift_by_columns", []):
                col_name = col_info["column_name"]
                if col_info.get("drift_detected", False):
                    drifted_features.append(col_name)
                drift_scores[col_name] = col_info.get("drift_score", 0)

            return {
                "dataset_drifted": metrics_result.get("dataset_drift", False),
                "number_of_columns": metrics_result.get("number_of_columns", 0),
                "number_of_drifted_columns": metrics_result.get("number_of_drifted_columns", 0),
                "share_of_drifted_columns": metrics_result.get("share_of_drifted_columns", 0),
                "drifted_features": drifted_features,
                "drift_scores": drift_scores,
                "threshold": threshold,
            }

        except ImportError:
            logger.warning("EvidentlyAI not available, using basic statistical comparison")
            return self._basic_drift_check(current_data)
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            return {
                "dataset_drifted": False,
                "error": str(e),
                "number_of_columns": 0,
                "number_of_drifted_columns": 0,
                "drifted_features": [],
            }

    def detect_prediction_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Detect drift in model prediction distributions using TargetDriftPreset.

        Both DataFrames must contain a ``prediction`` column holding model
        predictions (not ground-truth labels).  Additional feature columns are
        allowed and will be ignored by the preset — only the mapping-defined
        ``prediction`` column is evaluated.

        A ``ColumnMapping`` is always provided so Evidently correctly identifies
        the prediction column regardless of the column name or dtype heuristics.

        Args:
            reference_df: Reference dataset with a ``prediction`` column
                          populated by ``model.predict(X_ref)`` at training time.
                          This is loaded from the MLflow artifact logged by
                          ``log_reference_predictions_artifact()``.
            current_df: Current production dataset with a ``prediction``
                        column populated from the prediction logger.
            threshold: Statistical threshold for drift detection (0–1).

        Returns:
            Dictionary with prediction drift results::

                {
                    "prediction_drifted": bool,
                    "prediction_drift_score": float,
                    "stattest": str,            # test used by Evidently
                    "method": "evidently",
                }
        """
        try:
            from evidently.report import Report
            from evidently.metric_preset import TargetDriftPreset
            from evidently import ColumnMapping

            if "prediction" not in reference_df.columns:
                raise ValueError(
                    "reference_df must contain a 'prediction' column. "
                    "Ensure log_reference_predictions_artifact() was called during training."
                )
            if "prediction" not in current_df.columns:
                raise ValueError(
                    "current_df must contain a 'prediction' column. "
                    "The prediction logger must persist prediction values."
                )

            # Explicitly map column roles so Evidently treats 'prediction' as a
            # model output and not as a generic feature.  Setting task type
            # controls whether Evidently uses regression (KS test) or
            # classification (chi-square / PSI) drift statistics.
            column_mapping = ColumnMapping(
                prediction="prediction",
                target=None,       # target (ground truth) is NOT available here
                task=self.task_type,
            )

            # TargetDriftPreset: designed exactly for monitoring prediction/target
            # distribution shifts between reference and current data.
            report = Report([TargetDriftPreset(stattest_threshold=threshold)])
            report.run(
                reference_data=reference_df,
                current_data=current_df,
                column_mapping=column_mapping,
            )

            drift_dict = report.as_dict()
            drift_detected = False
            drift_score = 0.0
            stattest_name = "unknown"

            for metric in drift_dict.get("metrics", []):
                # TargetDriftPreset emits a ColumnDriftMetric for the prediction col
                if (
                    metric.get("metric") == "ColumnDriftMetric"
                    and metric.get("result", {}).get("column_name") == "prediction"
                ):
                    result = metric["result"]
                    drift_detected = result.get("drift_detected", False)
                    drift_score = result.get("drift_score", 0.0)
                    stattest_name = result.get("stattest", "unknown")
                    break

            return {
                "prediction_drifted": drift_detected,
                "prediction_drift_score": drift_score,
                "stattest": stattest_name,
                "method": "evidently",
            }

        except Exception as e:
            logger.error(f"Prediction drift detection failed: {e}")
            return {
                "prediction_drifted": False,
                "prediction_drift_score": 0.0,
                "stattest": "error",
                "method": "error",
                "error": str(e),
            }

    def _basic_drift_check(
        self,
        current_data: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Basic statistical drift check using mean/std comparison.

        Fallback when EvidentlyAI is not installed.
        """
        drifted_features = []
        drift_scores = {}

        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.reference_data[col]):
                continue

            ref_mean = self.reference_data[col].mean()
            ref_std = self.reference_data[col].std()
            cur_mean = current_data[col].mean()

            # Simple z-score based drift detection
            if ref_std > 0:
                z_score = abs(cur_mean - ref_mean) / ref_std
                drift_scores[col] = float(z_score)
                if z_score > 2.0:  # More than 2 std deviations
                    drifted_features.append(col)

        return {
            "dataset_drifted": len(drifted_features) > 0,
            "number_of_columns": len(self.reference_data.columns),
            "number_of_drifted_columns": len(drifted_features),
            "drifted_features": drifted_features,
            "drift_scores": drift_scores,
            "method": "basic_z_score",
        }

    def generate_html_report(
        self,
        current_data: pd.DataFrame,
        output_path: str = "drift_report.html",
    ) -> str:
        """
        Generate an interactive HTML drift report.

        Args:
            current_data: Production feature data.
            output_path: Path to save the HTML report.

        Returns:
            Path to the saved report.
        """
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report([DataDriftPreset()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
            )
            report.save_html(output_path)
            logger.info(f"Drift report saved to: {output_path}")
            return output_path

        except ImportError:
            logger.warning("EvidentlyAI not available for HTML report generation")
            return ""


def push_drift_metrics_to_prometheus(
    drift_results: Dict[str, Any],
    pred_drift_results: Dict[str, Any] | None = None,
    pushgateway_url: str = "http://localhost:9091",
) -> None:
    """
    Push drift metrics to Prometheus PushGateway.

    Industry-grade hygiene rules applied here:

    1. **Delete-before-push**: The previous metric group for ``job='batch-monitoring'``
       is deleted before pushing fresh values.  This handles the case where
       feature names change between runs (old per-feature label values would
       otherwise persist forever since PushGateway has no TTL).
    2. **Service-level grouping**: Only ``job='batch-monitoring'`` is used as the
       grouping key.  No per-instance labels are added, avoiding stale series
       accumulation when the monitor runs on different hosts.
    3. **Staleness timestamp**: ``mlops_batch_job_last_run_timestamp`` (epoch
       seconds) is pushed so Prometheus can alert when the job hasn't run
       recently (see ``BatchMonitoringJobStale`` alert rule).

    Args:
        drift_results: Results from :meth:`DriftDetector.detect_data_drift`.
        pred_drift_results: Results from :meth:`DriftDetector.detect_prediction_drift`
                            (``None`` if prediction drift was skipped).
        pushgateway_url: Prometheus PushGateway base URL.
    """
    try:
        import time as _time
        from prometheus_client import (
            CollectorRegistry, Gauge,
            push_to_gateway, delete_from_gateway,
        )

        job_name = "batch-monitoring"

        # ── Step 1: Delete the previous group so stale series never accumulate ──
        # best-effort — if PushGateway is down we continue to the push attempt
        try:
            delete_from_gateway(pushgateway_url, job=job_name)
            logger.debug("Deleted previous batch-monitoring group from PushGateway")
        except Exception as del_err:
            logger.warning(f"Could not delete old PushGateway group (continuing): {del_err}")

        # ── Step 2: Build fresh registry ──
        registry = CollectorRegistry()

        # Staleness / heartbeat gauge — lets Prometheus alert when job stops running
        last_run_ts = Gauge(
            "mlops_batch_job_last_run_timestamp",
            "Unix epoch timestamp of the last successful batch monitoring push",
            registry=registry,
        )
        last_run_ts.set(_time.time())

        # Data drift detected gauge
        data_drift_gauge = Gauge(
            "mlops_data_drift_detected",
            "Whether data drift was detected (0=no, 1=yes)",
            registry=registry,
        )
        data_drift_gauge.set(1 if drift_results.get("dataset_drifted", False) else 0)

        # Drifted columns count
        drifted_count = Gauge(
            "mlops_drifted_columns_count",
            "Number of columns with detected drift",
            registry=registry,
        )
        drifted_count.set(drift_results.get("number_of_drifted_columns", 0))

        # Drift share
        drift_share = Gauge(
            "mlops_drift_share",
            "Share of drifted columns (0-1)",
            registry=registry,
        )
        drift_share.set(drift_results.get("share_of_drifted_columns", 0))

        # Per-feature drift score gauge
        feature_drift_score = Gauge(
            "mlops_feature_drift_score",
            "Individual drift score per feature",
            ["feature"],
            registry=registry,
        )
        for feature, score in drift_results.get("drift_scores", {}).items():
            feature_drift_score.labels(feature=feature).set(score)

        # Prediction drift gauges (only when available)
        if pred_drift_results:
            pred_drift_gauge = Gauge(
                "mlops_prediction_drift_detected",
                "Whether prediction drift was detected (0=no, 1=yes)",
                registry=registry,
            )
            pred_drift_gauge.set(
                1 if pred_drift_results.get("prediction_drifted", False) else 0
            )

            pred_drift_score_gauge = Gauge(
                "mlops_prediction_drift_score",
                "Prediction drift score metric",
                registry=registry,
            )
            pred_drift_score_gauge.set(
                pred_drift_results.get("prediction_drift_score", 0.0)
            )

        # ── Step 3: Push fresh metrics (overwrites any remaining old group data) ──
        push_to_gateway(
            pushgateway_url,
            job=job_name,
            registry=registry,
        )
        logger.info(
            f"Drift metrics pushed to Prometheus PushGateway ({pushgateway_url})"
        )

    except Exception as e:
        logger.error(f"Failed to push drift metrics to Prometheus: {e}")
