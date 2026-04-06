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
            reference_data: Training/baseline data (features only).
            task_type: "regression" or "classification".
        """
        self.reference_data = reference_data
        self.task_type = task_type

    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Detect drift in feature distributions.

        Uses EvidentlyAI's DataDriftPreset with configurable statistical
        test thresholds. Falls back to a basic statistical comparison
        if EvidentlyAI is unavailable.

        Args:
            current_data: Production data to check (same columns as reference).
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
            current_data: Production data.
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
    pushgateway_url: str = "http://localhost:9091",
) -> None:
    """
    Push drift metrics to Prometheus PushGateway.

    Args:
        drift_results: Results from detect_data_drift().
        pushgateway_url: PushGateway URL.
    """
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()

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

        push_to_gateway(
            pushgateway_url,
            job="batch-monitoring",
            registry=registry,
        )
        logger.info("Drift metrics pushed to Prometheus PushGateway")

    except Exception as e:
        logger.error(f"Failed to push drift metrics to Prometheus: {e}")
