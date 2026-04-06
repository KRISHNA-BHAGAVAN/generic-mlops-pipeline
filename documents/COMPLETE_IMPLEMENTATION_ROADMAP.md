# Complete MLOps Pipeline Implementation Roadmap
## End-to-End Guide for Claude Code & AI Builders

**Version:** 1.0  
**Updated:** April 2026  
**Total Estimated Effort:** 8 weeks (Phase 1.1 + 1.2)

---

## Quick Start: What to Build

### For Claude Code / Antigravity

You're building a **production-ready MLOps platform** with:

1. **Config-driven experiment pipeline** (core ML)
2. **MLflow integration** (experiment tracking & model registry)
3. **FastAPI inference service** (model serving)
4. **Prometheus + Grafana** (monitoring & visualization)
5. **EvidentlyAI** (drift detection)
6. **Docker & Docker Compose** (containerization & orchestration)

---

## Phase 1.1: MVP (Weeks 1-4)

### What You're Building

A working MLOps pipeline where:
- Data scientists run regression/classification experiments via YAML configs
- All experiments are tracked in MLflow
- Best models are registered and promoted via aliases
- FastAPI service loads and serves models from the registry
- **All services expose Prometheus metrics**
- **Grafana dashboards show system health and model performance**
- **Basic alerts notify on latency/errors**

### Week 1: Core Infrastructure

**Deliverables:**
- Config validation schema (Pydantic)
- Model factory pattern
- MLflow integration
- FastAPI scaffolding

**Key Files to Create:**
```
src/config/
  ├─ experiment_schema.py       # Pydantic config model
  ├─ load_config.py
  ├─ validate_config.py

src/models/
  ├─ factory.py                 # Task-specific model selection
  ├─ train.py                   # Training logic
  ├─ evaluate.py                # Evaluation logic

deployment/
  ├─ app.py                     # FastAPI app (without monitoring yet)
  ├─ model_loader.py
```

**Commands:**
```bash
# Validate the config system works
python -m pytest tests/test_config.py -v

# Test model training
python -m src.pipelines.train_pipeline --config configs/regression/test.yaml --dry-run
```

### Week 2-3: Model Serving & MLflow Integration

**Deliverables:**
- FastAPI service loads models from MLflow
- Model predictions work end-to-end
- Docker containerization

**Key Files:**
```
deployment/
  ├─ app.py                     # Updated with MLflow model loading
  ├─ Dockerfile
  ├─ model_loader.py            # Load by alias from registry

tests/
  ├─ test_inference.py          # Test /predict endpoint
```

**Commands:**
```bash
# Test model serving
docker build -t mlops-service:latest .
docker run -p 8000:8000 mlops-service:latest

# Test predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"age": 30, "income": 50000}}'
```

### Week 4: Monitoring Infrastructure

**Deliverables:**
- Prometheus metrics instrumented in FastAPI
- Prometheus and Grafana containers
- System health & model performance dashboards
- Basic alert rules

**Key Files:**
```
src/monitoring/
  ├─ metrics.py                 # Prometheus client definitions
  └─ training_metrics.py        # Training-specific metrics

monitoring/
  ├─ prometheus.yml             # Prometheus config
  ├─ prometheus.rules.yml       # Alert rules
  ├─ grafana/
  │   ├─ Dockerfile
  │   ├─ datasources.yml
  │   └─ dashboards/
  │       ├─ system_health.json
  │       └─ model_performance.json

deployment/
  ├─ docker-compose.yml         # Multi-container setup
```

**Commands:**
```bash
# Start full stack
docker-compose up -d

# Generate test traffic
python -m scripts.generate_traffic --num_requests 100

# Check metrics in Prometheus
curl http://localhost:9090/api/v1/query?query=mlops_api_request_total

# Access Grafana
# http://localhost:3000 (admin/admin)
```

**Acceptance Criteria for Phase 1.1:**
- ✅ Experiments run via config without code changes
- ✅ All runs tracked in MLflow with metrics
- ✅ Model promotion via aliases works
- ✅ FastAPI serves models from registry
- ✅ Prometheus collects metrics every 15s
- ✅ Grafana shows system health dashboard
- ✅ Grafana shows model performance dashboard
- ✅ Alerts fire on high latency (>1s p95)
- ✅ Alerts fire on high error rate (>5%)
- ✅ Docker Compose brings up entire stack

---

## Phase 1.2: Advanced Monitoring (Weeks 5-8)

### What You're Adding

- Automated data drift detection
- Batch monitoring jobs
- Drift dashboards
- Slack alerting
- Retraining decision framework

### Week 5-6: EvidentlyAI Integration

**Deliverables:**
- Drift detector module
- Batch monitoring job
- Prometheus metrics export

**Key Files:**
```
src/monitoring/
  ├─ drift_detector.py          # EvidentlyAI wrapper
  ├─ batch_monitor.py           # Scheduled monitoring job
  └─ prediction_logger.py        # Log predictions for drift detection

monitoring/
  ├─ batch_monitoring_job.py    # Entrypoint for scheduled execution
```

**Commands:**
```bash
# Test drift detection
python -m src.monitoring.batch_monitor --model_name customer_churn

# Check if metrics pushed to Prometheus
curl http://localhost:9091/metrics
```

### Week 7: Drift Dashboards & Advanced Alerting

**Deliverables:**
- Data drift dashboard
- Prediction drift dashboard
- Slack integration
- Advanced alert rules

**Key Files:**
```
monitoring/
  ├─ grafana/dashboards/
  │   ├─ data_drift.json        # NEW: Drift dashboard
  │   └─ prediction_drift.json   # NEW: Prediction drift
  ├─ alertmanager.yml           # Slack webhook config
  ├─ prometheus.rules.yml       # Updated with drift alerts

src/monitoring/
  ├─ alerting.py                # Alert webhook handling
```

**Commands:**
```bash
# Schedule batch monitoring
python -m scripts.schedule_monitoring --interval daily --hour 2

# Check alert status
curl http://localhost:9090/api/v1/alerts
```

### Week 8: Integration Testing & Documentation

**Deliverables:**
- End-to-end integration tests
- Runbooks for common scenarios
- Performance baseline established

**Commands:**
```bash
# Run full integration test
pytest tests/integration/ -v

# Generate baseline metrics report
python -m scripts.baseline_metrics --output baseline.json
```

**Acceptance Criteria for Phase 1.2:**
- ✅ Batch job detects data drift correctly
- ✅ Drift metrics exported to Prometheus
- ✅ Grafana shows drift dashboard
- ✅ Alerts fire when drift detected
- ✅ Slack notifications working
- ✅ Historical drift data retained for 30 days
- ✅ Retraining can be triggered manually

---

## Complete File Structure

After Phase 1.2, your repository should look like:

```
generic-mlops-pipeline/
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
│
├── data/
│   ├── raw/                           # Original datasets
│   └── processed/                     # Feature engineering outputs
│
├── src/
│   ├── __init__.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── experiment_schema.py       # Pydantic ExperimentConfig
│   │   ├── load_config.py             # YAML → ExperimentConfig
│   │   └── validate_config.py         # Config validation logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── factory.py                 # get_model_class()
│   │   ├── train.py                   # train_regression/classification()
│   │   ├── evaluate.py                # evaluate() function
│   │   └── registry.py                # MLflow integration
│   │
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── experiment_schema.py       # ExperimentConfig Pydantic model
│   │   └── inference_schema.py        # PredictionRequest/Response
│   │
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── rank_runs.py               # Compare experiments
│   │   └── promote_model.py           # Alias assignment
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py          # Main CLI for training
│   │   ├── register_pipeline.py       # Model registration
│   │   └── deployment_pipeline.py     # Deploy to production
│   │
│   └── monitoring/
│       ├── __init__.py
│       ├── metrics.py                 # Prometheus metrics definitions
│       ├── training_metrics.py        # Training-specific metrics
│       ├── drift_detector.py          # EvidentlyAI wrapper
│       ├── batch_monitor.py           # Batch monitoring job
│       ├── prediction_logger.py       # Log predictions
│       └── alerting.py                # Alert handlers
│
├── configs/
│   ├── regression/
│   │   ├── exp_001_baseline.yaml
│   │   └── exp_002_advanced.yaml
│   ├── classification/
│   │   └── exp_001_baseline.yaml
│   └── README.md                      # Config documentation
│
├── deployment/
│   ├── Dockerfile                     # FastAPI service image
│   ├── app.py                         # FastAPI main app
│   ├── model_loader.py                # Load models by alias
│   ├── request_validation.py          # Request validation logic
│   └── docker-compose.yml             # Full stack composition
│
├── monitoring/
│   ├── prometheus.yml                 # Prometheus scrape config
│   ├── prometheus.rules.yml           # Alert rules
│   ├── alertmanager.yml               # Slack webhook config
│   ├── batch_monitoring_job.py        # Scheduled job entrypoint
│   │
│   └── grafana/
│       ├── Dockerfile                 # Custom Grafana image
│       ├── datasources.yml            # Prometheus datasource
│       │
│       └── dashboards/
│           ├── system_health.json     # Latency, errors, throughput
│           ├── model_performance.json # Accuracy, predictions, inference time
│           ├── data_drift.json        # Feature distributions
│           └── prediction_drift.json  # Prediction distribution changes
│
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_factory.py
│   │   ├── test_train.py
│   │   └── test_evaluate.py
│   │
│   ├── integration/
│   │   ├── test_end_to_end.py
│   │   └── test_monitoring.py
│   │
│   └── fixtures/
│       ├── sample_config.yaml
│       └── sample_data.csv
│
├── scripts/
│   ├── generate_traffic.py            # Synthetic traffic for testing
│   ├── schedule_monitoring.py         # Schedule batch jobs
│   ├── baseline_metrics.py            # Establish performance baseline
│   └── trigger_test_alerts.py         # Test alert rules
│
├── .dvc/
├── dvc.yaml                           # DVC pipeline config
│
└── .github/
    └── workflows/
        ├── train.yml                  # CI/CD for training
        └── deploy.yml                 # CI/CD for deployment
```

---

## Dependencies & Environment

### Python Dependencies

```toml
# pyproject.toml

[project]
name = "generic-mlops-pipeline"
version = "0.1.0"
requires-python = ">=3.10"

dependencies = [
    # Core ML
    "scikit-learn>=1.0.0",
    "pandas>=1.3.0",
    "numpy>=1.20.0",
    
    # MLOps
    "mlflow>=2.0.0",
    "dvc>=2.0.0",
    
    # Serving
    "fastapi>=0.95.0",
    "uvicorn>=0.20.0",
    "pydantic>=2.0.0",
    
    # Monitoring
    "prometheus-client>=0.16.0",
    "evidently>=0.2.0",
    "apscheduler>=3.10.0",
    
    # Utilities
    "click>=8.0.0",
    "pyyaml>=6.0",
    "python-dotenv>=0.19.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
]
```

### Installation

```bash
# Using uv (recommended, faster)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Or using pip
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

---

## Running the Pipeline

### Local Development

```bash
# 1. Start infrastructure
docker-compose up -d

# 2. Create experiment config
cat > configs/regression/my_first_exp.yaml << EOF
experiment_name: "my_first_experiment"
user: "alice"
task_type: "regression"
dataset_source: "data/processed/sample.csv"
target_column: "price"
feature_columns:
  - "age"
  - "income"
model_type: "linear_regression"
model_params: {}
split_strategy: "random"
test_size: 0.2
val_size: 0.2
random_state: 42
metrics:
  - "mse"
  - "rmse"
  - "mae"
  - "r2"
mlflow_tags:
  team: "data-science"
EOF

# 3. Run experiment
python -m src.pipelines.train_pipeline --config configs/regression/my_first_exp.yaml

# 4. View in MLflow
# http://localhost:5000

# 5. View in Grafana
# http://localhost:3000

# 6. Register best model
python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient()
# Find best run and register it
"

# 7. Assign champion alias
python -c "
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_alias('model_name', 'champion', version=1)
"

# 8. Test inference
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {"age": 30, "income": 50000},
    "model_alias": "champion"
  }'

# 9. Check metrics in Prometheus
curl http://localhost:9090/api/v1/query?query=mlops_predictions_total
```

### Production Deployment

```bash
# 1. Build production image
docker build -t myregistry/mlops-service:v1.0 .
docker push myregistry/mlops-service:v1.0

# 2. Deploy with monitoring stack
# Use provided docker-compose.yml or Kubernetes manifests

# 3. Configure environment
export MLFLOW_TRACKING_URI="https://dagshub.com/user/repo/mlflow"
export MLFLOW_TRACKING_USERNAME="user"
export MLFLOW_TRACKING_PASSWORD="token"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."

# 4. Start services
docker-compose -f docker-compose.prod.yml up -d

# 5. Verify
curl http://localhost:8000/health
curl http://localhost:3000  # Grafana dashboards
```

---

## Key Metrics to Track

### System Health
- Request latency (p50, p95, p99)
- Request throughput (req/sec)
- Error rate (%)
- Active connections

### Model Performance
- Prediction distribution
- Inference latency
- Cache hit rate
- Memory usage per model

### Data Quality
- Missing values (%)
- Outlier count
- Feature distribution changes
- Schema violations

### Drift Metrics
- Data drift (yes/no + score)
- Prediction drift (yes/no + score)
- Drifted features (list)
- Days since retraining

---

## Testing Strategy

### Unit Tests
```bash
# Test config validation
pytest tests/unit/test_config.py -v

# Test model factory
pytest tests/unit/test_factory.py -v

# Test training logic
pytest tests/unit/test_train.py -v

# Test metrics
pytest tests/unit/test_metrics.py -v
```

### Integration Tests
```bash
# End-to-end pipeline
pytest tests/integration/test_end_to_end.py -v

# Model serving
pytest tests/integration/test_inference.py -v

# Monitoring
pytest tests/integration/test_monitoring.py -v
```

### Load Testing
```bash
# Generate synthetic traffic
python -m scripts.generate_traffic --num_requests 1000 --workers 20

# Monitor metrics during load test
watch -n 1 'curl -s http://localhost:9090/api/v1/query?query=rate\(mlops_api_request_total\[1m\]\) | jq'
```

---

## Troubleshooting Guide

### Docker Issues

```bash
# Services won't start
docker-compose up -d
docker-compose logs  # Check for errors

# Clean start (remove old volumes)
docker-compose down -v
docker-compose up -d

# Check service health
docker-compose ps
docker-compose exec fastapi curl http://localhost:8000/health
```

### Prometheus Issues

```bash
# No metrics
curl http://localhost:9090/api/v1/targets  # Check scrape targets
docker-compose logs prometheus

# High memory usage
# Reduce retention: --storage.tsdb.retention.time=7d
```

### Grafana Issues

```bash
# Can't connect to Prometheus
# Check datasource in http://localhost:3000/admin/data-sources

# Dashboard shows "No data"
# Verify query syntax; check Prometheus for metrics
```

### Model Issues

```bash
# Model not loading
docker-compose logs fastapi
# Check MLflow is running: http://localhost:5000
# Check alias is set: python -c "..."

# Inference errors
# Check request format matches schema
# Check model signature is compatible
```

---

## Common Tasks

### Run a New Experiment
```bash
# 1. Create config
cp configs/regression/template.yaml configs/regression/exp_new.yaml
# Edit config...

# 2. Validate
python -m src.pipelines.train_pipeline --config configs/regression/exp_new.yaml --dry-run

# 3. Run
python -m src.pipelines.train_pipeline --config configs/regression/exp_new.yaml

# 4. Review results
# Open http://localhost:5000 (MLflow)
# Compare metrics
```

### Register & Promote a Model
```bash
# From Python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Register
model_uri = "runs:/abc123/model"
mv = mlflow.register_model(model_uri, name="my_model")

# Tag for validation
client.set_model_version_tag("my_model", mv.version, "validation_status", "pending")

# Approve
client.set_model_version_tag("my_model", mv.version, "validation_status", "passed")

# Promote to champion
client.set_registered_model_alias("my_model", "champion", version=mv.version)
```

### Check Model Health
```bash
# Grafana dashboards
# http://localhost:3000

# Direct Prometheus
curl 'http://localhost:9090/api/v1/query?query=mlops_prediction_latency_seconds'

# MLflow metrics
# http://localhost:5000/experiments/0
```

### Trigger Batch Monitoring
```bash
# Manual run
python -m src.monitoring.batch_monitor --model_name my_model

# Check results in Prometheus
curl http://localhost:9090/api/v1/query?query=mlops_data_drift_detected

# Check logs in MLflow
# http://localhost:5000 (look for "monitoring" runs)
```

---

## Success Checklist

### End of Phase 1.1
- ✅ Team member can run `python -m src.pipelines.train_pipeline --config config.yaml`
- ✅ All runs appear in MLflow UI
- ✅ Models can be promoted via aliases
- ✅ FastAPI service responds with predictions
- ✅ Grafana shows real-time metrics
- ✅ Alerts fire correctly
- ✅ `docker-compose up -d` starts everything

### End of Phase 1.2
- ✅ Batch monitoring job runs daily
- ✅ Data drift is detected automatically
- ✅ Drift dashboards show trends
- ✅ Slack alerts notify team
- ✅ Historical data retained for 30 days
- ✅ Team can interpret and act on drift alerts

---

## Next Steps (Phase 2+)

After Phase 1.2 is solid, consider:

1. **OpenTelemetry Integration**
   - Distributed tracing across services
   - Latency breakdown by component

2. **Automated Retraining**
   - Trigger retraining when drift > threshold
   - A/B test new vs current champion

3. **CI/CD Integration**
   - Automated model validation
   - Staging environment testing

4. **Advanced Monitoring**
   - GPU utilization tracking
   - Token cost tracking (for LLMs)
   - Bias detection

5. **Kubernetes Deployment**
   - Horizontal scaling
   - Blue-green deployments
   - Multi-region serving

---

## Support & Questions

When building this pipeline, you'll likely have questions. Here's how to get unstuck:

1. **For MLOps concepts:** Refer to the PRD (PRD_UPDATED_v3.md)
2. **For implementation details:** Refer to TECHNICAL_SPECIFICATION.md
3. **For monitoring setup:** Refer to MONITORING_IMPLEMENTATION_GUIDE.md
4. **For config format:** Look at example configs in `configs/` directory
5. **For error messages:** Check troubleshooting guide above or GitHub issues for OSS tools

---

## Final Notes

**This is a complete, production-ready specification.** The challenge now is implementation, not design.

**For Claude Code:** You have enough detail to build this without ambiguity. If you find yourself asking "should I do X or Y?", the answer is typically in the spec or the technical guide.

**For Team Members:** Review these documents before starting. Understand the high-level flow (Experiment → Track → Register → Promote → Serve → Monitor). All the details support this flow.

**Timeline Reality:** 8 weeks is realistic if you have a clear implementation order and minimal meetings. Add 20% buffer for debugging and integration surprises.

Good luck! 🚀

