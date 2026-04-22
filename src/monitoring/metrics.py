"""
Prometheus metrics definitions for the MLOps pipeline.

Defines all custom Prometheus metrics used across the FastAPI service
and training pipeline. Uses the default registry to allow the
/metrics endpoint to serve them natively.
"""

from __future__ import annotations

import time
from typing import Callable

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# ─── REQUEST METRICS ───────────────────────────────────────────────────────

request_count = Counter(
    "mlops_api_request_total",
    "Total number of API requests",
    ["method", "endpoint", "status"],
)

request_latency = Histogram(
    "mlops_api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

requests_in_progress = Gauge(
    "mlops_api_requests_in_progress",
    "Number of requests currently being processed",
    ["method", "endpoint"],
)

# ─── MODEL INFERENCE METRICS ──────────────────────────────────────────────
prediction_latency = Histogram(
    "mlops_prediction_duration_seconds",
    "Model inference latency in seconds",
    ["model_name", "model_version"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)

prediction_distribution = Histogram(
    "mlops_prediction_value",
    "Distribution of model predictions",
    ["model_name", "task_type"],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

predictions_total = Counter(
    "mlops_predictions_total",
    "Total number of predictions made",
    ["model_name", "model_version", "status"],
)


# ─── ERROR METRICS ─────────────────────────────────────────────────────────

prediction_errors = Counter(
    "mlops_prediction_errors_total",
    "Total prediction errors",
    ["model_name", "error_type"],
)


# ─── SYSTEM METRICS ───────────────────────────────────────────────────────

models_loaded = Gauge(
    "mlops_models_loaded_total",
    "Number of models currently loaded in memory",
    ["model_name"],
)

model_load_time = Histogram(
    "mlops_model_load_duration_seconds",
    "Time to load a model from MLflow registry",
    ["model_name"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
)


# ─── TRAINING METRICS ─────────────────────────────────────────────────────

training_duration = Histogram(
    "mlops_training_duration_seconds",
    "Model training duration",
    ["model_type", "task_type"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

training_metric_gauge = Gauge(
    "mlops_training_metric",
    "Final training evaluation metric value",
    ["model_name", "metric_type"],
)


# ─── MIDDLEWARE ────────────────────────────────────────────────────────────

def setup_metrics_middleware(app) -> None:
    """
    Add Prometheus instrumentation middleware to a FastAPI app.

    Records request count, latency, in-progress requests, and errors.

    Args:
        app: FastAPI application instance.
    """
    from starlette.requests import Request
    from starlette.responses import Response

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        method = request.method
        endpoint = request.url.path

        # Don't instrument the metrics endpoint itself
        if endpoint == "/metrics":
            return await call_next(request)

        start_time = time.time()
        requests_in_progress.labels(method=method, endpoint=endpoint).inc()

        status = 500  # Default to error if something goes wrong
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        except Exception:
            prediction_errors.labels(
                model_name="unknown",
                error_type="MiddlewareError",
            ).inc()
            raise
        finally:
            duration = time.time() - start_time
            requests_in_progress.labels(method=method, endpoint=endpoint).dec()
            request_latency.labels(method=method, endpoint=endpoint).observe(duration)
            request_count.labels(
                method=method,
                endpoint=endpoint,
                status=str(status),
            ).inc()
