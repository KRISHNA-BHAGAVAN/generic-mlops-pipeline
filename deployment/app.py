"""
FastAPI inference service for the MLOps pipeline.

Serves ML models from MLflow registry with Prometheus metrics
instrumentation, prediction logging, and health checks.

Usage:
    uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response

# Add project root to path so src modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.monitoring.metrics import (
    models_loaded,
    model_load_time,
    prediction_errors,
    prediction_latency,
    prediction_distribution,
    predictions_total,
    setup_metrics_middleware,
)
from src.monitoring.prediction_logger import PredictionLogger
from src.schemas.inference_schema import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.helpers import load_env
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ─── Global state ──────────────────────────────────────────────────────────

_model_cache: Dict[str, Any] = {}
_model_versions: Dict[str, str] = {}
_prediction_logger: PredictionLogger | None = None


# ─── Model loading ─────────────────────────────────────────────────────────

def _load_model_sync(model_name: str, alias: str = "champion") -> Any:
    """
    Load a model from MLflow registry by alias with caching and timing.

    Args:
        model_name: Registered model name.
        alias: Model alias (e.g. "champion", "candidate").

    Returns:
        Loaded MLflow pyfunc model.

    Raises:
        HTTPException: If model not found or loading fails.
    """
    cache_key = f"{model_name}@{alias}"

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    start_time = time.time()
    try:
        import mlflow.pyfunc

        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        _model_cache[cache_key] = model

        # Try to get version info
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()
            version_info = client.get_model_version_by_alias(model_name, alias)
            _model_versions[cache_key] = str(version_info.version)
        except Exception:
            _model_versions[cache_key] = "unknown"

        # Record metrics
        load_duration = time.time() - start_time
        model_load_time.labels(model_name=model_name).observe(load_duration)
        models_loaded.labels(model_name=model_name).set(1)

        logger.info(
            f"Model loaded: {cache_key} "
            f"(v{_model_versions[cache_key]}, {load_duration:.2f}s)"
        )
        return model

    except Exception as e:
        prediction_errors.labels(
            model_name=model_name,
            error_type="ModelLoadError",
        ).inc()
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model {model_name}@{alias}: {str(e)}",
        )


# ─── App lifecycle ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: setup env, prediction logger on startup."""
    global _prediction_logger

    load_env()

    # Setup MLflow tracking URI
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Initialize prediction logger
    _prediction_logger = PredictionLogger()
    logger.info("FastAPI inference service started")

    yield

    # Cleanup
    _model_cache.clear()
    logger.info("FastAPI inference service stopped")


# ─── App creation ──────────────────────────────────────────────────────────

app = FastAPI(
    title="MLOps Inference Service",
    description="Serve ML models from MLflow registry with monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# Add Prometheus metrics middleware
setup_metrics_middleware(app)


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate a prediction using a registered model.

    Loads the model by name and alias from MLflow registry,
    runs inference, logs metrics, and stores the prediction.
    """
    start_time = time.time()
    model_name = request.model_name
    alias = request.model_alias
    cache_key = f"{model_name}@{alias}"

    try:
        # Load model
        model = _load_model_sync(model_name, alias)
        model_version = _model_versions.get(cache_key, "unknown")

        # Convert features to DataFrame
        X = pd.DataFrame([request.features])

        # Run inference
        inference_start = time.time()
        raw_prediction = model.predict(X)
        inference_duration = time.time() - inference_start

        prediction_value = raw_prediction[0]

        # Handle numpy types for JSON serialization
        if hasattr(prediction_value, "item"):
            prediction_value = prediction_value.item()

        # Record Prometheus metrics
        prediction_latency.labels(
            model_name=model_name,
            model_version=model_version,
        ).observe(inference_duration)

        predictions_total.labels(
            model_name=model_name,
            model_version=model_version,
            status="success",
        ).inc()

        # Record prediction distribution for numeric predictions
        if isinstance(prediction_value, (int, float)):
            prediction_distribution.labels(
                model_name=model_name,
                task_type="regression",
            ).observe(float(prediction_value))

        # Log prediction to database
        total_latency = time.time() - start_time
        if _prediction_logger:
            _prediction_logger.log_prediction(
                model_name=model_name,
                model_version=model_version,
                features=request.features,
                prediction=prediction_value,
                latency_seconds=total_latency,
            )

        return PredictionResponse(
            prediction=prediction_value,
            model_name=model_name,
            model_version=model_version,
            model_uri=f"models:/{model_name}@{alias}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        prediction_errors.labels(
            model_name=model_name,
            error_type=type(e).__name__,
        ).inc()
        predictions_total.labels(
            model_name=model_name,
            model_version=_model_versions.get(cache_key, "unknown"),
            status="error",
        ).inc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Generate batch predictions."""
    model_name = request.model_name
    alias = request.model_alias
    cache_key = f"{model_name}@{alias}"

    try:
        model = _load_model_sync(model_name, alias)
        model_version = _model_versions.get(cache_key, "unknown")

        X = pd.DataFrame(request.instances)

        start_time = time.time()
        raw_predictions = model.predict(X)
        inference_duration = time.time() - start_time

        # Convert predictions
        predictions = []
        for pred in raw_predictions:
            val = pred.item() if hasattr(pred, "item") else pred
            predictions.append(val)

        # Record metrics
        prediction_latency.labels(
            model_name=model_name,
            model_version=model_version,
        ).observe(inference_duration)

        predictions_total.labels(
            model_name=model_name,
            model_version=model_version,
            status="success",
        ).inc()

        return BatchPredictionResponse(
            predictions=predictions,
            model_name=model_name,
            model_version=model_version,
            count=len(predictions),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        prediction_errors.labels(
            model_name=model_name,
            error_type=type(e).__name__,
        ).inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now(timezone.utc).isoformat(),
        models_loaded=len(_model_cache),
    )


@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics for scraping."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/models/reload")
async def reload_model(model_name: str, alias: str = "champion"):
    """Force reload a model from registry (clears cache)."""
    cache_key = f"{model_name}@{alias}"
    if cache_key in _model_cache:
        del _model_cache[cache_key]
        models_loaded.labels(model_name=model_name).set(0)

    _load_model_sync(model_name, alias)
    return {"status": "reloaded", "model": cache_key}


# ─── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
