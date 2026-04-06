# Monitoring & Observability Implementation Guide
## Generic MLOps Pipeline – Prometheus, Grafana, EvidentlyAI

**Version:** 1.0  
**Target Phase:** 1.1 (Prometheus/Grafana) + 1.2 (EvidentlyAI)  
**Audience:** Claude Code, Antigravity, Backend Engineers

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1.1: Core Monitoring (Prometheus + Grafana)](#phase-11-core-monitoring)
3. [Phase 1.2: Advanced ML Monitoring (EvidentlyAI)](#phase-12-advanced-ml-monitoring)
4. [Implementation Steps](#implementation-steps)
5. [Configuration Files](#configuration-files)
6. [Dashboard Specifications](#dashboard-specifications)
7. [Alert Rules](#alert-rules)
8. [Testing & Verification](#testing--verification)
9. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Three Pillars of Observability

```
┌──────────────────────────────────────────────────────────────┐
│                   Observability Pillars                       │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. METRICS (Prometheus)                                     │
│     └─ Numerical measurements: latency, throughput, errors    │
│     └─ Time-series data points                                │
│     └─ Aggregated from all services                           │
│                                                               │
│  2. LOGS (Structured, sent to Prometheus)                    │
│     └─ Prediction logs: model_id, timestamp, prediction       │
│     └─ Event logs: alerts, status changes                     │
│     └─ Stored separately, queried for context                 │
│                                                               │
│  3. TRACES (Future: OpenTelemetry Phase 2)                   │
│     └─ Request flow through system                            │
│     └─ Latency breakdown per component                        │
│     └─ Dependency relationships                               │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Service Instrumentation Points

```
Training Pipeline
├─ Training start/end events
├─ Model accuracy metrics
├─ Training time and resource usage
└─ Dataset info (rows processed, features)

FastAPI Inference Service
├─ Request count (total, by endpoint, by model)
├─ Request latency (min, max, p50, p95, p99)
├─ Error rates (by error type)
├─ Prediction distribution
├─ Inference time per model
└─ Cache hit/miss rates

Batch Monitoring Job
├─ Job start/end timestamps
├─ Data drift metrics (per feature)
├─ Data quality metrics (nulls, outliers)
├─ Prediction drift
└─ Job execution time
```

---

## Phase 1.1: Core Monitoring (Prometheus + Grafana)

### 1.1.1 Overview

**Goal:** Collect and visualize system health and basic model metrics.

**Timeline:** Weeks 1-4

**Deliverables:**
- FastAPI instrumentation with Prometheus client
- Prometheus configuration and deployment
- Grafana instance with 2 dashboards
- 3 basic alert rules
- Docker Compose for local development

---

### 1.1.2 Prometheus Metrics Library

**Install:**
```bash
pip install prometheus-client
```

**FastAPI Integration:**

```python
# src/monitoring/metrics.py

from prometheus_client import (
    Counter, Histogram, Gauge,
    start_http_server, CollectorRegistry
)
import time
from typing import Callable, Any

# Create a custom registry (optional, for isolation)
registry = CollectorRegistry()

# REQUEST METRICS
request_count = Counter(
    'mlops_api_request_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

request_latency = Histogram(
    'mlops_api_request_duration_seconds',
    'API request latency in seconds',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
    registry=registry
)

requests_in_progress = Gauge(
    'mlops_api_requests_in_progress',
    'Number of requests currently being processed',
    ['method', 'endpoint'],
    registry=registry
)

# MODEL INFERENCE METRICS
prediction_latency = Histogram(
    'mlops_prediction_duration_seconds',
    'Model inference latency in seconds',
    ['model_name', 'model_version'],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=registry
)

prediction_distribution = Histogram(
    'mlops_prediction_value',
    'Distribution of model predictions',
    ['model_name', 'task_type'],
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=registry
)

predictions_total = Counter(
    'mlops_predictions_total',
    'Total number of predictions made',
    ['model_name', 'model_version', 'status'],
    registry=registry
)

# ERROR METRICS
prediction_errors = Counter(
    'mlops_prediction_errors_total',
    'Total prediction errors',
    ['model_name', 'error_type'],
    registry=registry
)

# SYSTEM METRICS
models_loaded = Gauge(
    'mlops_models_loaded_total',
    'Number of models currently loaded in memory',
    ['model_name'],
    registry=registry
)

model_load_time = Histogram(
    'mlops_model_load_duration_seconds',
    'Time to load a model from MLflow registry',
    ['model_name'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0),
    registry=registry
)

# DATASET METRICS (from training)
training_duration = Histogram(
    'mlops_training_duration_seconds',
    'Model training duration',
    ['model_type', 'task_type'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
    registry=registry
)

training_accuracy = Gauge(
    'mlops_training_accuracy',
    'Final training accuracy/metric',
    ['model_name', 'metric_type'],
    registry=registry
)

# Middleware for FastAPI
def setup_metrics_middleware(app):
    """Add Prometheus instrumentation to FastAPI app."""
    
    @app.middleware("http")
    async def metrics_middleware(request, call_next):
        method = request.method
        endpoint = request.url.path
        
        # Start timing
        start_time = time.time()
        requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        
        try:
            # Process request
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            requests_in_progress.labels(method=method, endpoint=endpoint).dec()
            prediction_errors.labels(
                model_name="unknown",
                error_type=type(e).__name__
            ).inc()
            raise
        finally:
            # Record metrics
            duration = time.time() - start_time
            requests_in_progress.labels(method=method, endpoint=endpoint).dec()
            request_latency.labels(method=method, endpoint=endpoint).observe(duration)
            request_count.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()
        
        return response

def start_metrics_server(port: int = 8001):
    """Start Prometheus metrics server."""
    start_http_server(port, registry=registry)
    print(f"Prometheus metrics available at http://localhost:{port}/metrics")
```

**FastAPI Application Instrumentation:**

```python
# deployment/app.py

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import mlflow.pyfunc
import time
import json
from src.monitoring.metrics import (
    setup_metrics_middleware,
    start_metrics_server,
    prediction_latency,
    prediction_distribution,
    predictions_total,
    prediction_errors,
    model_load_time,
)
from src.schemas.inference_schema import PredictionRequest, PredictionResponse

app = FastAPI(
    title="MLOps Inference Service",
    description="Serve ML models from MLflow registry",
)

# Setup monitoring
setup_metrics_middleware(app)

# Start metrics server on port 8001
# (separate from API port 8000)
start_metrics_server(port=8001)

# Global model cache
_model_cache = {}

async def load_model_by_alias(model_name: str, alias: str = "champion"):
    """Load model with timing instrumentation."""
    cache_key = f"{model_name}@{alias}"
    
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    start_time = time.time()
    try:
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        _model_cache[cache_key] = model
        
        # Record load time
        load_time = time.time() - start_time
        model_load_time.labels(model_name=model_name).observe(load_time)
        
        return model
    except Exception as e:
        prediction_errors.labels(
            model_name=model_name,
            error_type="ModelLoadError"
        ).inc()
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model {model_name}@{alias}: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """Generate a prediction with instrumentation."""
    start_time = time.time()
    
    try:
        # Load model
        model = await load_model_by_alias("customer_churn", request.model_alias)
        
        # Convert request to DataFrame
        import pandas as pd
        X = pd.DataFrame([request.features])
        
        # Predict with timing
        inference_start = time.time()
        prediction = model.predict(X)[0]
        inference_duration = time.time() - inference_start
        
        # Record metrics
        prediction_latency.labels(
            model_name="customer_churn",
            model_version="3"  # Should come from model metadata
        ).observe(inference_duration)
        
        prediction_distribution.labels(
            model_name="customer_churn",
            task_type="regression"
        ).observe(float(prediction))
        
        predictions_total.labels(
            model_name="customer_churn",
            model_version="3",
            status="success"
        ).inc()
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version="3",
            model_uri="models:/customer_churn@champion",
            timestamp=pd.Timestamp.now().isoformat(),
        )
    
    except Exception as e:
        prediction_errors.labels(
            model_name="customer_churn",
            error_type=type(e).__name__
        ).inc()
        
        predictions_total.labels(
            model_name="customer_churn",
            model_version="3",
            status="error"
        ).inc()
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}

@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Training Pipeline Instrumentation:**

```python
# src/monitoring/training_metrics.py

from src.monitoring.metrics import (
    training_duration,
    training_accuracy,
)
import time
import mlflow

def record_training_metrics(
    model_name: str,
    model_type: str,
    task_type: str,
    metrics: dict,
    duration: float,
):
    """Record training metrics to Prometheus."""
    
    # Record training duration
    training_duration.labels(
        model_type=model_type,
        task_type=task_type
    ).observe(duration)
    
    # Record key evaluation metrics
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            training_accuracy.labels(
                model_name=model_name,
                metric_type=metric_name
            ).set(metric_value)
    
    # Also log to MLflow as before
    mlflow.log_metric("training_duration", duration)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name, metric_value)
```

---

### 1.1.3 Prometheus Configuration

**File: `monitoring/prometheus.yml`**

```yaml
global:
  scrape_interval: 15s           # How often to scrape targets
  evaluation_interval: 15s       # How often to evaluate rules
  external_labels:
    monitor: 'mlops-pipeline'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Load rules once and periodically evaluate them
rule_files:
  - "prometheus.rules.yml"

scrape_configs:
  # FastAPI service
  - job_name: 'fastapi-service'
    static_configs:
      - targets: ['localhost:8001']
    metrics_path: '/metrics'
    scrape_interval: 15s

  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter (system metrics) - optional
  - job_name: 'node'
    static_configs:
      - targets: ['localhost:9100']
    scrape_interval: 15s
```

---

### 1.1.4 Alert Rules

**File: `monitoring/prometheus.rules.yml`**

```yaml
groups:
  - name: mlops_alerts
    interval: 30s
    rules:
      # High Request Latency
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, rate(mlops_api_request_duration_seconds_bucket[5m])) > 1.0
        for: 5m
        labels:
          severity: warning
          component: api
        annotations:
          summary: "High API latency detected"
          description: "API p95 latency is {{ $value }}s (threshold: 1.0s)"

      # High Error Rate
      - alert: HighErrorRate
        expr: |
          (
            sum(rate(mlops_api_request_total{status=~"5.."}[5m]))
            /
            sum(rate(mlops_api_request_total[5m]))
          ) > 0.05
        for: 5m
        labels:
          severity: critical
          component: api
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 5%)"

      # High Prediction Error Count
      - alert: PredictionErrors
        expr: increase(mlops_prediction_errors_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
          component: model
        annotations:
          summary: "Prediction errors detected"
          description: "{{ $value }} prediction errors in last 5 minutes"

      # Model Load Timeout
      - alert: SlowModelLoading
        expr: histogram_quantile(0.95, rate(mlops_model_load_duration_seconds_bucket[5m])) > 5.0
        for: 5m
        labels:
          severity: warning
          component: model
        annotations:
          summary: "Slow model loading"
          description: "Model load p95 latency is {{ $value }}s (threshold: 5.0s)"

      # Service Down
      - alert: ServiceDown
        expr: up{job="fastapi-service"} == 0
        for: 1m
        labels:
          severity: critical
          component: service
        annotations:
          summary: "FastAPI service is down"
          description: "FastAPI inference service is not responding"
```

---

### 1.1.5 Docker Compose

**File: `deployment/docker-compose.yml`**

```yaml
version: '3.8'

services:
  # FastAPI inference service
  fastapi:
    build:
      context: ..
      dockerfile: deployment/Dockerfile
    ports:
      - "8000:8000"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
      MLFLOW_TRACKING_USERNAME: ${MLFLOW_USERNAME}
      MLFLOW_TRACKING_PASSWORD: ${MLFLOW_PASSWORD}
      PROMETHEUS_PORT: 8001
    volumes:
      - ./:/app
    networks:
      - mlops
    depends_on:
      - mlflow
      - prometheus

  # Prometheus time-series database
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - ./monitoring/prometheus.rules.yml:/etc/prometheus/prometheus.rules.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=15d'
    networks:
      - mlops

  # Grafana visualization
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
      GF_USERS_ALLOW_SIGN_UP: false
    volumes:
      - ./monitoring/grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - grafana_data:/var/lib/grafana
    networks:
      - mlops
    depends_on:
      - prometheus

  # Alertmanager (for alerts)
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
    networks:
      - mlops

  # MLflow (for experiment tracking and model registry)
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
    networks:
      - mlops

volumes:
  prometheus_data:
  grafana_data:
  alertmanager_data:
  mlflow_data:

networks:
  mlops:
    driver: bridge
```

**Usage:**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f fastapi
docker-compose logs -f prometheus
docker-compose logs -f grafana

# Stop all services
docker-compose down

# Access services
# - FastAPI: http://localhost:8000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
# - MLflow: http://localhost:5000
```

---

### 1.1.6 Grafana Dashboards

#### Dashboard 1: System Health

**File: `monitoring/grafana/dashboards/system_health.json`**

```json
{
  "dashboard": {
    "title": "MLOps System Health",
    "tags": ["mlops", "system"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate (req/sec)",
        "targets": [
          {
            "expr": "sum(rate(mlops_api_request_total[1m]))",
            "legendFormat": "Total Requests"
          }
        ],
        "type": "graph"
      },
      {
        "title": "API Response Time (p95, p99)",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(mlops_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p95"
          },
          {
            "expr": "histogram_quantile(0.99, rate(mlops_api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "p99"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Error Rate",
        "targets": [
          {
            "expr": "sum(rate(mlops_api_request_total{status=~\"5..\"}[5m])) / sum(rate(mlops_api_request_total[5m]))",
            "legendFormat": "Error Rate"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Active Requests",
        "targets": [
          {
            "expr": "sum(mlops_api_requests_in_progress)",
            "legendFormat": "In Progress"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "Prediction Errors (last 5m)",
        "targets": [
          {
            "expr": "increase(mlops_prediction_errors_total[5m])",
            "legendFormat": "{{ error_type }}"
          }
        ],
        "type": "graph"
      }
    ]
  }
}
```

#### Dashboard 2: Model Performance

**File: `monitoring/grafana/dashboards/model_performance.json`**

```json
{
  "dashboard": {
    "title": "MLOps Model Performance",
    "tags": ["mlops", "model"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Inference Latency (ms)",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, rate(mlops_prediction_duration_seconds_bucket[5m])) * 1000",
            "legendFormat": "{{ model_name }} (p50)"
          },
          {
            "expr": "histogram_quantile(0.95, rate(mlops_prediction_duration_seconds_bucket[5m])) * 1000",
            "legendFormat": "{{ model_name }} (p95)"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Prediction Distribution",
        "targets": [
          {
            "expr": "histogram_quantile(0.5, rate(mlops_prediction_value_bucket[5m]))",
            "legendFormat": "{{ model_name }} (median)"
          }
        ],
        "type": "heatmap"
      },
      {
        "title": "Predictions per Second",
        "targets": [
          {
            "expr": "sum(rate(mlops_predictions_total{status=\"success\"}[1m]))",
            "legendFormat": "Successful"
          },
          {
            "expr": "sum(rate(mlops_predictions_total{status=\"error\"}[1m]))",
            "legendFormat": "Failed"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Models Loaded in Memory",
        "targets": [
          {
            "expr": "mlops_models_loaded_total",
            "legendFormat": "{{ model_name }}"
          }
        ],
        "type": "gauge"
      }
    ]
  }
}
```

---

## Phase 1.2: Advanced ML Monitoring (EvidentlyAI)

### 1.2.1 Overview

**Goal:** Detect data drift, prediction drift, and data quality issues automatically.

**Timeline:** Weeks 5-8

**Deliverables:**
- EvidentlyAI integration
- Batch monitoring jobs (daily)
- Drift detection dashboard in Grafana
- Advanced alerting (Slack integration)
- Retraining decision framework

---

### 1.2.2 EvidentlyAI Integration

**Install:**

```bash
pip install evidently
```

**Drift Detection Module:**

```python
# src/monitoring/drift_detector.py

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset, ClassificationPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestDataDrift, TestPredictionDrift
import pandas as pd
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """Detect data drift, prediction drift, and performance degradation."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        column_mapping: Optional[Dict[str, str]] = None,
        task_type: str = "regression",
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Training/baseline data
            column_mapping: Mapping of column roles
            task_type: "regression" or "classification"
        """
        self.reference_data = reference_data
        self.column_mapping = column_mapping or {}
        self.task_type = task_type
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Detect drift in feature distributions.
        
        Args:
            current_data: Production data to check
            threshold: Drift detection threshold (0-1)
        
        Returns:
            Dictionary with drift results
        """
        report = Report([
            DataDriftPreset(stattest_threshold=threshold)
        ])
        
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )
        
        # Extract drift information
        drift_dict = report.as_dict()
        
        return {
            "dataset_drifted": drift_dict["metrics"][0]["result"]["dataset_drift"],
            "number_of_columns": drift_dict["metrics"][0]["result"]["number_of_columns"],
            "number_of_drifted_columns": drift_dict["metrics"][0]["result"]["number_of_drifted_columns"],
            "drifted_features": [
                col["column_name"]
                for col in drift_dict["metrics"][0]["result"]["drift_by_columns"]
                if col["drift_detected"]
            ],
        }
    
    def detect_prediction_drift(
        self,
        reference_predictions: pd.Series,
        current_predictions: pd.Series,
    ) -> Dict[str, Any]:
        """
        Detect drift in model predictions.
        
        Args:
            reference_predictions: Training/baseline predictions
            current_predictions: Production predictions
        
        Returns:
            Dictionary with prediction drift results
        """
        # Create temporary data with predictions
        ref_df = self.reference_data.copy()
        ref_df['prediction'] = reference_predictions.values
        
        curr_df = self.current_data.copy() if hasattr(self, 'current_data') else None
        if curr_df is None:
            return {"error": "current_data not set"}
        
        curr_df['prediction'] = current_predictions.values
        
        report = Report([DataDriftPreset()])
        report.run(reference_data=ref_df, current_data=curr_df)
        
        drift_dict = report.as_dict()
        prediction_drift = next(
            (col for col in drift_dict["metrics"][0]["result"]["drift_by_columns"]
             if col["column_name"] == "prediction"),
            None
        )
        
        return {
            "prediction_drifted": prediction_drift["drift_detected"] if prediction_drift else False,
            "drift_score": prediction_drift.get("drift_score", 0),
        }
    
    def generate_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.
        
        Args:
            current_data: Production data
        
        Returns:
            Complete report with drift, quality, and performance metrics
        """
        # Data drift
        data_drift = self.detect_data_drift(current_data)
        
        # Task-specific metrics
        if self.task_type == "regression":
            metrics_preset = RegressionPreset()
        elif self.task_type == "classification":
            metrics_preset = ClassificationPreset()
        else:
            metrics_preset = DataDriftPreset()
        
        report = Report([metrics_preset])
        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
        )
        
        return {
            "data_drift": data_drift,
            "metrics": report.as_dict(),
            "timestamp": pd.Timestamp.now().isoformat(),
        }

def push_drift_metrics_to_prometheus(
    drift_results: Dict[str, Any],
    prometheus_pushgateway: str = "http://localhost:9091",
):
    """
    Push drift metrics to Prometheus PushGateway.
    
    Args:
        drift_results: Results from detect_data_drift()
        prometheus_pushgateway: PushGateway URL
    """
    from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
    import requests
    
    registry = CollectorRegistry()
    
    # Data drift gauge
    data_drift_gauge = Gauge(
        'mlops_data_drift_detected',
        'Whether data drift was detected (0=no, 1=yes)',
        registry=registry
    )
    data_drift_gauge.set(1 if drift_results["dataset_drifted"] else 0)
    
    # Drifted columns count
    drifted_count = Gauge(
        'mlops_drifted_columns_count',
        'Number of columns with detected drift',
        registry=registry
    )
    drifted_count.set(drift_results["number_of_drifted_columns"])
    
    # Push to PushGateway
    try:
        push_to_gateway(
            prometheus_pushgateway,
            job='batch-monitoring',
            registry=registry,
        )
    except Exception as e:
        logger.error(f"Failed to push metrics to Prometheus: {e}")
```

**Batch Monitoring Job:**

```python
# src/monitoring/batch_monitor.py

import pandas as pd
import logging
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import mlflow
from src.monitoring.drift_detector import DriftDetector, push_drift_metrics_to_prometheus
from src.monitoring.metrics import registry
from prometheus_client import Gauge, push_to_gateway

logger = logging.getLogger(__name__)

class BatchMonitor:
    """Batch monitoring job that runs periodically."""
    
    def __init__(
        self,
        model_name: str,
        task_type: str,
        database_url: str = "sqlite:///predictions.db",
        prometheus_pushgateway: str = "http://localhost:9091",
    ):
        self.model_name = model_name
        self.task_type = task_type
        self.db_engine = create_engine(database_url)
        self.prometheus_pushgateway = prometheus_pushgateway
    
    def get_reference_data(self) -> pd.DataFrame:
        """
        Load reference (training) data from MLflow.
        
        Returns:
            Reference dataset as DataFrame
        """
        # This would load from a configured location
        # For now, assume it's stored in MLflow artifacts
        client = mlflow.tracking.MlflowClient()
        
        # Find the champion model
        model = client.get_registered_model(self.model_name)
        champion_version = next(
            (v for v in model.aliases if v.alias == "champion"),
            None
        )
        
        if not champion_version:
            raise ValueError(f"No champion version found for {self.model_name}")
        
        # Load reference data (stored in model artifacts)
        run = mlflow.get_run(champion_version.version)
        # This assumes reference data is stored in artifacts
        # In practice, you'd load from a configured data location
        
        return pd.read_parquet(f"reference_data_{self.model_name}.parquet")
    
    def get_recent_predictions(
        self,
        hours: int = 24,
    ) -> pd.DataFrame:
        """
        Load recent predictions from database.
        
        Args:
            hours: Number of hours of data to retrieve
        
        Returns:
            DataFrame with recent predictions
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        query = f"""
            SELECT *
            FROM predictions
            WHERE model_name = '{self.model_name}'
            AND timestamp > '{cutoff_time.isoformat()}'
        """
        
        return pd.read_sql(query, self.db_engine)
    
    def run(self):
        """Execute batch monitoring job."""
        logger.info(f"Starting batch monitoring for {self.model_name}")
        
        try:
            # Load data
            reference_data = self.get_reference_data()
            recent_data = self.get_recent_predictions(hours=24)
            
            if recent_data.empty:
                logger.warning("No recent predictions found")
                return
            
            # Extract features (remove predictions, timestamps, etc.)
            feature_columns = [col for col in recent_data.columns
                             if col not in ['prediction', 'timestamp', 'model_name']]
            current_features = recent_data[feature_columns]
            
            # Detect drift
            detector = DriftDetector(
                reference_data=reference_data[feature_columns],
                task_type=self.task_type,
            )
            
            drift_results = detector.detect_data_drift(current_features)
            
            # Push metrics to Prometheus
            push_drift_metrics_to_prometheus(drift_results, self.prometheus_pushgateway)
            
            # Log results
            logger.info(f"Drift detection results: {drift_results}")
            
            # Generate report and log to MLflow
            self.log_monitoring_report(drift_results, recent_data)
            
            # Trigger alerts if needed
            if drift_results["dataset_drifted"]:
                self.trigger_drift_alert(drift_results)
        
        except Exception as e:
            logger.error(f"Batch monitoring failed: {e}", exc_info=True)
            # Still track the failure
            self.log_failure(str(e))
    
    def log_monitoring_report(
        self,
        drift_results: dict,
        recent_data: pd.DataFrame,
    ):
        """Log monitoring results to MLflow."""
        with mlflow.start_run(run_name=f"monitoring-{self.model_name}"):
            # Log drift metrics
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("drift_detected", drift_results["dataset_drifted"])
            mlflow.log_metric("drifted_columns_count", drift_results["number_of_drifted_columns"])
            
            # Log drifted features as tags
            if drift_results["drifted_features"]:
                mlflow.set_tags({
                    f"drifted_feature_{i}": feat
                    for i, feat in enumerate(drift_results["drifted_features"])
                })
            
            # Log summary metrics
            mlflow.log_metric("recent_data_rows", len(recent_data))
            mlflow.log_metric("timestamp", datetime.now().timestamp())
    
    def trigger_drift_alert(self, drift_results: dict):
        """Send alert when drift is detected."""
        # This would integrate with Slack, email, PagerDuty, etc.
        logger.warning(
            f"DRIFT ALERT: Model {self.model_name} shows drift. "
            f"Drifted features: {drift_results['drifted_features']}"
        )
        # TODO: Send to Slack webhook
    
    def log_failure(self, error_message: str):
        """Log monitoring job failure."""
        with mlflow.start_run(run_name=f"monitoring-failure-{self.model_name}"):
            mlflow.log_param("error", error_message)
            mlflow.log_param("model_name", self.model_name)

# Scheduled execution (using APScheduler or cron)
def schedule_batch_monitoring():
    """Schedule batch monitoring to run periodically."""
    from apscheduler.schedulers.background import BackgroundScheduler
    import atexit
    
    scheduler = BackgroundScheduler()
    
    # Run batch monitoring daily at 2 AM
    monitor = BatchMonitor(
        model_name="customer_churn",
        task_type="regression",
    )
    scheduler.add_job(
        func=monitor.run,
        trigger="cron",
        hour=2,
        minute=0,
        id='batch_monitoring_job',
    )
    
    scheduler.start()
    
    # Shut down the scheduler when exiting the app
    atexit.register(lambda: scheduler.shutdown())
    
    return scheduler
```

---

### 1.2.3 Drift Detection Dashboard

**File: `monitoring/grafana/dashboards/data_drift.json`**

```json
{
  "dashboard": {
    "title": "MLOps Data Drift Detection",
    "tags": ["mlops", "drift"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Data Drift Status",
        "targets": [
          {
            "expr": "mlops_data_drift_detected",
            "legendFormat": "Drift Detected"
          }
        ],
        "type": "stat"
      },
      {
        "title": "Drifted Columns Count",
        "targets": [
          {
            "expr": "mlops_drifted_columns_count",
            "legendFormat": "Count"
          }
        ],
        "type": "gauge"
      },
      {
        "title": "Drift Trend (7 days)",
        "targets": [
          {
            "expr": "mlops_drifted_columns_count",
            "legendFormat": "Drifted Columns"
          }
        ],
        "type": "graph"
      },
      {
        "title": "Feature Distribution Comparison",
        "targets": [
          {
            "expr": "mlops_feature_drift_score",
            "legendFormat": "{{ feature_name }}"
          }
        ],
        "type": "heatmap"
      }
    ]
  }
}
```

---

### 1.2.4 Alert Rules (Phase 1.2)

**Add to `monitoring/prometheus.rules.yml`:**

```yaml
  - name: mlops_drift_alerts
    interval: 1h  # Check hourly (batch monitoring runs daily)
    rules:
      # Data Drift Detected
      - alert: DataDriftDetected
        expr: mlops_data_drift_detected == 1
        for: 1h
        labels:
          severity: warning
          component: data
        annotations:
          summary: "Data drift detected"
          description: "{{ $value }} columns show drift"

      # High Number of Drifted Columns
      - alert: HighDriftColumnCount
        expr: mlops_drifted_columns_count > 5
        for: 1h
        labels:
          severity: critical
          component: data
        annotations:
          summary: "Multiple columns drifted"
          description: "{{ $value }} columns have drifted"

      # Model Performance Drop
      - alert: ModelPerformanceDegradation
        expr: rate(mlops_model_accuracy[24h]) < 0.85
        for: 2h
        labels:
          severity: critical
          component: model
        annotations:
          summary: "Model accuracy dropped"
          description: "Model accuracy is {{ $value }} (threshold: 0.85)"
```

---

## Configuration Files

### Prometheus Data Source

**File: `monitoring/grafana/datasources.yml`**

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: MLflow
    type: prometheus
    access: proxy
    url: http://mlflow:5000
    editable: true
```

### Alertmanager Configuration

**File: `monitoring/alertmanager.yml`**

```yaml
global:
  resolve_timeout: 5m

route:
  receiver: 'default'
  group_by: ['alertname', 'cluster']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h

receivers:
  - name: 'default'
    # Slack integration
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#ml-alerts'
        title: '{{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

---

## Testing & Verification

### Verification Checklist

```bash
# 1. Start services
docker-compose up -d

# 2. Check Prometheus is collecting metrics
curl http://localhost:9090/api/v1/targets

# 3. Verify metrics are available
curl http://localhost:9090/api/v1/query?query=mlops_api_request_total

# 4. Access Grafana dashboards
# - Open http://localhost:3000
# - Login: admin / admin
# - Check System Health dashboard
# - Check Model Performance dashboard

# 5. Generate test traffic
python -m scripts.generate_traffic.py --num_requests 100

# 6. Verify Prometheus stores metrics
# - Open http://localhost:9090/graph
# - Query: rate(mlops_api_request_total[5m])

# 7. Test alerts (optional)
python -m scripts.trigger_test_alerts.py

# 8. Check Grafana dashboards for data
# - System Health: Should show request rate, latency
# - Model Performance: Should show prediction distribution
```

### Synthetic Testing

```python
# scripts/generate_traffic.py

import requests
import random
import time
from concurrent.futures import ThreadPoolExecutor

def generate_traffic(num_requests: int = 100):
    """Generate synthetic traffic to test monitoring."""
    
    url = "http://localhost:8000/predict"
    
    def make_request():
        payload = {
            "features": {
                "age": random.randint(18, 80),
                "income": random.randint(20000, 150000),
                "account_tenure_months": random.randint(1, 120),
            },
            "model_alias": "champion"
        }
        
        try:
            response = requests.post(url, json=payload, timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    # Generate requests in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(lambda _: make_request(), range(num_requests)))
    
    success_rate = sum(results) / len(results)
    print(f"Success rate: {success_rate * 100:.1f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_requests", type=int, default=100)
    args = parser.parse_args()
    
    generate_traffic(args.num_requests)
```

---

## Troubleshooting

### Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Prometheus not scraping** | No metrics in Prometheus UI | Check `prometheus.yml` targets; verify `/metrics` endpoint is running |
| **Grafana no data** | "No data" in dashboards | Verify Prometheus datasource; check query syntax in dashboard JSON |
| **High memory usage** | Prometheus/Grafana slow | Reduce `storage.tsdb.retention.time` in prometheus.yml |
| **Drift detector import error** | `from evidently import` fails | Install evidently: `pip install evidently` |
| **Metrics not pushed to Prometheus** | Batch job succeeds but no drift metrics | Check PushGateway URL; verify `prometheus_pushgateway` config |

### Debug Commands

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Query specific metric
curl "http://localhost:9090/api/v1/query?query=mlops_api_request_total"

# Check alert status
curl http://localhost:9090/api/v1/alerts

# View Prometheus logs
docker-compose logs prometheus

# View Grafana logs
docker-compose logs grafana

# Test drift detector
python -c "from src.monitoring.drift_detector import DriftDetector; print('OK')"
```

---

## Summary

**Phase 1.1 Delivers:**
- ✅ Prometheus metrics collection from FastAPI
- ✅ Grafana dashboards for system health and model performance
- ✅ Basic alerting on latency, errors, throughput
- ✅ Docker Compose setup for local development

**Phase 1.2 Delivers:**
- ✅ EvidentlyAI integration for drift detection
- ✅ Batch monitoring jobs
- ✅ Drift detection dashboards
- ✅ Advanced alerting with Slack integration
- ✅ Historical analysis and trend detection

**Next Steps:**
1. Implement Phase 1.1 (weeks 1-4)
2. Test with synthetic traffic
3. Deploy to staging environment
4. Implement Phase 1.2 (weeks 5-8)
5. Configure alerting and notifications

