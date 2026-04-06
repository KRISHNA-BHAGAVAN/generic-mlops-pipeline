"""
Training metrics recorder for Prometheus.

Records training-related metrics (duration, evaluation metrics)
to Prometheus gauges and histograms, alongside MLflow logging.
"""

from __future__ import annotations

from src.monitoring.metrics import training_duration, training_metric_gauge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def record_training_metrics(
    model_name: str,
    model_type: str,
    task_type: str,
    metrics: dict,
    duration: float,
) -> None:
    """
    Record training metrics to Prometheus.

    Args:
        model_name: Registered model name.
        model_type: Model type string (e.g. "random_forest_regression").
        task_type: "regression" or "classification".
        metrics: Dict of metric_name -> value.
        duration: Training duration in seconds.
    """
    # Record training duration
    training_duration.labels(
        model_type=model_type,
        task_type=task_type,
    ).observe(duration)

    # Record each evaluation metric as a gauge
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            training_metric_gauge.labels(
                model_name=model_name,
                metric_type=metric_name,
            ).set(metric_value)

    logger.info(
        f"Training metrics recorded for {model_name}: "
        f"duration={duration:.2f}s, metrics={list(metrics.keys())}"
    )
