# Feature Completion Checklist
### Generic MLOps Pipeline — Status vs PRD v3.0

> Cross-referenced against: `documents/PRD.md`, `documents/TECHNICAL_SPECIFICATION.md`, `documents/MONITORING_IMPLEMENTATION_GUIDE.md`
>
> **Last updated:** 2026-04-06

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully implemented & verified |
| ⚠️ | Partially implemented or needs polish |
| ❌ | Not yet implemented |
| 🔮 | Deferred to future phase (Phase 2+) |

---

## Phase 1.1 — Core ML Pipeline (MVP)

### 1. Config-Driven Experiments
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 1.1 | YAML experiment configs | ✅ | `configs/regression/`, `configs/classification/` |
| 1.2 | Pydantic v2 config schema (`ExperimentConfig`) | ✅ | `src/config/validate_config.py` |
| 1.3 | Config validation (task-metric-model cross-check) | ✅ | Validates metrics, model types, target columns |
| 1.4 | Config validation against dataset | ✅ | Checks columns exist, dtypes match |
| 1.5 | `--dry-run` mode for config validation | ✅ | `train_pipeline.py --dry-run` |
| 1.6 | Preprocessing via config (`normalize`, `one_hot_encode`, `label_encode`) | ✅ | `src/features/build_features.py` |
| 1.7 | Split strategies (`random`, `temporal`, `stratified`) | ✅ | All three implemented |
| 1.8 | Integer→float64 casting for stable MLflow schemas | ✅ | Recently added |

**How to verify:** `python -m src.pipelines.train_pipeline --config configs/regression/construction_duration_v1.yaml --dry-run`

---

### 2. Model Factory & Training
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 2.1 | Registry-pattern model factory | ✅ | `src/models/factory.py` |
| 2.2 | `random_forest_regression` | ✅ | |
| 2.3 | `linear_regression` | ✅ | |
| 2.4 | `random_forest_classification` | ✅ | |
| 2.5 | `logistic_regression` | ✅ | |
| 2.6 | Task-aware training functions | ✅ | `src/models/train.py` |
| 2.7 | Model hyperparameter passthrough from config | ✅ | |

**How to verify:** `python -m pytest tests/test_models.py -v`

---

### 3. Evaluation & Metrics
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 3.1 | Metrics registry (regression: mse, rmse, mae, r2) | ✅ | `src/models/evaluate.py` |
| 3.2 | Metrics registry (classification: accuracy, precision, recall, f1, auc) | ✅ | |
| 3.3 | `confusion_matrix` metric for classification | ⚠️ | Listed in PRD but not in METRICS_REGISTRY |
| 3.4 | Evaluation plots (residuals, feature importance, confusion matrix) | ✅ | Auto-generated during pipeline run |
| 3.5 | Model signature inference (input/output schemas) | ✅ | Via `mlflow.models.infer_signature` |

**How to verify:** `python -m pytest tests/test_models.py::TestEvaluation -v`

---

### 4. MLflow Experiment Tracking
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 4.1 | DagsHub-hosted MLflow tracking | ✅ | Connects via `MLFLOW_TRACKING_URI` |
| 4.2 | Log config parameters | ✅ | task_type, model_type, dataset_source, etc. |
| 4.3 | Log model hyperparameters | ✅ | All `model_params` logged |
| 4.4 | Log evaluation metrics | ✅ | |
| 4.5 | Log model artifact with signature | ✅ | Uses MLflow 3.x `name` param |
| 4.6 | Log evaluation plots as artifacts | ✅ | |
| 4.7 | Log config YAML as artifact | ✅ | |
| 4.8 | Log DVC version tag | ✅ | When `dvc_version` is set in config |
| 4.9 | Log custom tags from config (`mlflow_tags`) | ✅ | |
| 4.10 | Log training time metric | ✅ | `training_duration_seconds` |
| 4.11 | Soft-deleted experiment auto-recovery | ✅ | Handles MLflow experiment lifecycle |

**How to verify:** `python -m src.pipelines.train_pipeline --config configs/regression/construction_duration_v1.yaml` → check DagsHub MLflow UI

---

### 5. Model Registry & Promotion
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 5.1 | `register_model()` with model_uri | ✅ | `src/models/registry.py` |
| 5.2 | `promote_model()` with alias | ✅ | champion / candidate / staging |
| 5.3 | `approve_model()` with validation tags | ✅ | |
| 5.4 | `register_pipeline.py` CLI | ✅ | `--model-uri`, `--run-id`, `--alias`, `--approve` |
| 5.5 | `resolve_model_uri_from_run()` (MLflow 3.x compat) | ✅ | Auto-discovers model URI from run ID |
| 5.6 | `--register` flag in train pipeline | ✅ | Auto-register after training |

**How to verify:** `python -m src.pipelines.register_pipeline --run-id <ID> --model-name construction_duration --alias champion --approve`

---

### 6. Model Selection & Ranking
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 6.1 | `rank_runs()` — rank runs by metric | ✅ | `src/selection/rank_runs.py` |
| 6.2 | `promote_model()` via selection module | ✅ | `src/selection/promote_model.py` |
| 6.3 | Programmatic best-run selection tooling | ⚠️ | Module exists but no CLI entry point |

---

### 7. Training Pipeline CLI
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 7.1 | Click-based CLI with `--config`, `--dry-run`, `--register` | ✅ | |
| 7.2 | `--mlflow-tracking-uri` override | ✅ | |
| 7.3 | Rich console output with step-by-step progress | ✅ | |
| 7.4 | Error handling with custom exceptions | ✅ | `src/pipelines/exceptions.py` |

---

### 8. Data & Feature Engineering
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 8.1 | CSV dataset loading | ✅ | `src/data/load_data.py` |
| 8.2 | Dataset validation (nulls, duplicates) | ✅ | `src/data/validate.py` |
| 8.3 | Feature preparation with preprocessing | ✅ | `src/features/build_features.py` |
| 8.4 | DVC data versioning | ✅ | `.dvc/`, `data.dvc` present |

---

### 9. Testing
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 9.1 | Config loading & validation tests (11 tests) | ✅ | `tests/test_config.py` |
| 9.2 | Model factory, training, evaluation tests (10 tests) | ✅ | `tests/test_models.py` |
| 9.3 | End-to-end pipeline tests (3 tests) | ✅ | `tests/test_pipeline.py` |
| 9.4 | Feature dtype casting test (1 test) | ✅ | `tests/test_pipeline.py::TestFeatureDtypes` |
| 9.5 | Monitoring/Observability tests | ❌ | No tests for metrics, drift detector, batch monitor |

**How to verify:** `python -m pytest tests/ -v` (26 tests pass)

---

### 10. Infrastructure & DevOps
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 10.1 | Dockerfile for FastAPI service | ✅ | `deployment/Dockerfile` |
| 10.2 | Docker Compose (full stack) | ✅ | `deployment/docker-compose.yml` |
| 10.3 | Makefile with common targets | ✅ | `Makefile` |
| 10.4 | `.env` / `.env.example` for secrets | ✅ | |
| 10.5 | `pyproject.toml` with hatchling build | ✅ | |
| 10.6 | `uv.lock` for reproducible deps | ✅ | |

**How to verify:** `docker compose -f deployment/docker-compose.yml up -d`

---

## Phase 1.1 — Observability (Prometheus + Grafana)

### 11. FastAPI Inference Service
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 11.1 | `/predict` endpoint | ✅ | `deployment/app.py` |
| 11.2 | `/health` endpoint | ✅ | |
| 11.3 | `/metrics` Prometheus endpoint | ✅ | |
| 11.4 | Model loading by alias from MLflow registry | ✅ | |
| 11.5 | Model caching | ✅ | In-memory cache |
| 11.6 | Prediction request/response Pydantic schemas | ✅ | `src/schemas/inference_schema.py` |
| 11.7 | Prediction logging to SQLite | ✅ | `src/monitoring/prediction_logger.py` |

**How to verify:** `uvicorn deployment.app:app --host 0.0.0.0 --port 8000` then `curl localhost:8000/health`

---

### 12. Prometheus Metrics Instrumentation
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 12.1 | `mlops_api_request_total` (counter) | ✅ | `src/monitoring/metrics.py` |
| 12.2 | `mlops_api_request_duration_seconds` (histogram) | ✅ | |
| 12.3 | `mlops_api_requests_in_progress` (gauge) | ✅ | |
| 12.4 | `mlops_prediction_duration_seconds` (histogram) | ✅ | |
| 12.5 | `mlops_prediction_value` distribution (histogram) | ✅ | |
| 12.6 | `mlops_predictions_total` (counter) | ✅ | |
| 12.7 | `mlops_prediction_errors_total` (counter) | ✅ | |
| 12.8 | `mlops_models_loaded_total` (gauge) | ✅ | |
| 12.9 | `mlops_model_load_duration_seconds` (histogram) | ✅ | |
| 12.10 | `mlops_training_duration_seconds` (histogram) | ✅ | `src/monitoring/training_metrics.py` |
| 12.11 | `mlops_training_accuracy` (gauge) | ✅ | |
| 12.12 | Metrics middleware for FastAPI | ✅ | |

---

### 13. Prometheus Infrastructure
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 13.1 | `prometheus.yml` scrape config | ✅ | `monitoring/prometheus.yml` |
| 13.2 | `prometheus.rules.yml` alert rules | ✅ | `monitoring/prometheus.rules.yml` |
| 13.3 | Prometheus container in Docker Compose | ✅ | |
| 13.4 | Pushgateway container | ✅ | For batch monitoring metrics |
| 13.5 | AlertManager container | ✅ | |

---

### 14. Grafana Dashboards
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 14.1 | Prometheus data source config | ✅ | `monitoring/grafana/datasources.yml` |
| 14.2 | Dashboard provisioning config | ✅ | `monitoring/grafana/dashboards/dashboard.yml` |
| 14.3 | System Health dashboard | ✅ | `system_health.json` |
| 14.4 | Model Performance dashboard | ✅ | `model_performance.json` |
| 14.5 | Data Drift dashboard | ✅ | `data_drift.json` |

**How to verify:** Open `http://localhost:3000` (admin/admin) after `docker compose up`

---

### 15. Alert Rules (Phase 1.1)
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 15.1 | `HighAPILatency` (p95 > 1s) | ✅ | |
| 15.2 | `HighErrorRate` (> 5%) | ✅ | |
| 15.3 | `PredictionErrors` (> 10 in 5m) | ✅ | |
| 15.4 | `SlowModelLoading` (p95 > 5s) | ✅ | |
| 15.5 | `ServiceDown` | ✅ | |
| 15.6 | AlertManager config | ✅ | `monitoring/alertmanager.yml` |

---

### 16. Testing & Traffic Generation
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 16.1 | `scripts/generate_traffic.py` | ✅ | Synthetic load testing |
| 16.2 | `scripts/trigger_test_alerts.py` | ❌ | PRD mentioned, not implemented |

---

## Phase 1.2 — Advanced ML Monitoring (EvidentlyAI)

### 17. EvidentlyAI Drift Detection
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 17.1 | `DriftDetector` class | ✅ | `src/monitoring/drift_detector.py` |
| 17.2 | `detect_data_drift()` | ✅ | Uses EvidentlyAI DataDriftPreset |
| 17.3 | `detect_prediction_drift()` | ⚠️ | Code exists but depends on `current_data` attribute not always set |
| 17.4 | `generate_report()` | ✅ | Comprehensive monitoring report |
| 17.5 | `push_drift_metrics_to_prometheus()` | ✅ | Pushes to Pushgateway |
| 17.6 | Per-feature drift score metrics | ⚠️ | Partial — aggregate drift pushed, per-feature granularity incomplete |

---

### 18. Batch Monitoring Job
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 18.1 | `BatchMonitor` class | ✅ | `src/monitoring/batch_monitor.py` |
| 18.2 | Get reference data from MLflow artifacts | ✅ | |
| 18.3 | Get recent predictions from SQLite | ✅ | |
| 18.4 | Run drift detection | ✅ | |
| 18.5 | Push metrics to Prometheus Pushgateway | ✅ | |
| 18.6 | Log monitoring report to MLflow | ✅ | |
| 18.7 | Trigger drift alerts | ✅ | Logging-based alert |
| 18.8 | Scheduled execution (APScheduler / cron) | ❌ | No `schedule_batch_monitoring()` function implemented |
| 18.9 | Batch monitoring CLI entry point | ❌ | No CLI to run batch monitoring manually |

---

### 19. Phase 1.2 Alert Rules
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 19.1 | `DataDriftDetected` alert rule | ✅ | In `prometheus.rules.yml` |
| 19.2 | `HighDriftColumnCount` alert rule | ✅ | |
| 19.3 | `ModelPerformanceDegradation` alert rule | ✅ | |

---

### 20. Slack / Notification Integration
| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 20.1 | Slack webhook in AlertManager config | ⚠️ | Template exists but commented out |
| 20.2 | Slack notification on drift alert | ❌ | Not implemented in code |
| 20.3 | Email notification | ❌ | Not implemented |

---

## Phase 2+ — Future Enhancements (Deferred)

| # | Feature | Status | Notes |
|---|---------|--------|-------|
| 🔮 | OpenTelemetry distributed tracing | ❌ | Phase 2 |
| 🔮 | GPU/hardware profiling | ❌ | Phase 2 |
| 🔮 | LLM-specific monitoring (token counts, cost) | ❌ | Phase 2 |
| 🔮 | Automated retraining triggers based on drift thresholds | ❌ | Phase 2 |
| 🔮 | A/B testing framework | ❌ | Phase 2 |
| 🔮 | CI/CD integration for registration & promotion | ❌ | Phase 2 |
| 🔮 | Shadow deployments / champion-challenger traffic splitting | ❌ | Phase 2 |
| 🔮 | Kubernetes-based deployment | ❌ | Phase 2 |
| 🔮 | Automated smoke tests on candidate models | ❌ | Phase 2 |
| 🔮 | `deployment_pipeline.py` for end-to-end deploy | ❌ | Listed in PRD structure, not implemented |
| 🔮 | Environment-specific registered model naming | ❌ | Phase 2 |
| 🔮 | Ranking CLI entry point | ❌ | Module exists but no CLI |

---

## Summary Statistics

| Phase | Total Items | ✅ Done | ⚠️ Partial | ❌ Remaining |
|-------|------------|---------|-----------|-------------|
| **Phase 1.1 — Core Pipeline** | 55 | 52 | 2 | 1 |
| **Phase 1.1 — Observability** | 27 | 26 | 0 | 1 |
| **Phase 1.2 — EvidentlyAI** | 15 | 10 | 2 | 3 |
| **Phase 2+ — Future** | 12 | 0 | 0 | 12 |
| **TOTAL** | **109** | **88** | **4** | **17** |

### Completion: **~84%** of Phase 1 (1.1 + 1.2) is complete

---

## 🔴 Remaining Items to Implement (Phase 1)

### High Priority (complete Phase 1.1)
1. **Monitoring tests** — Unit tests for `metrics.py`, `drift_detector.py`, `batch_monitor.py`, `prediction_logger.py` (item 9.5)
2. **`trigger_test_alerts.py`** script (item 16.2)

### Medium Priority (complete Phase 1.2)
3. **Batch monitoring CLI** — A CLI entry point to run batch monitoring manually: `python -m src.monitoring.batch_monitor --model-name <name>` (item 18.9)
4. **Batch monitoring scheduler** — `schedule_batch_monitoring()` using APScheduler or cron-compatible setup (item 18.8)
5. **Slack notification integration** — Uncomment & configure AlertManager Slack webhook; add Slack webhook to `.env` (items 20.1, 20.2)

### Low Priority (polish)
6. **`confusion_matrix` metric** in classification METRICS_REGISTRY (item 3.3)
7. **Selection CLI** — Add a CLI to rank runs and auto-promote from terminal (item 6.3)
8. **`predict_drift()` fix** — Ensure `current_data` attribute is properly set (item 17.3)
9. **Per-feature drift scores** — Push individual feature drift scores to Prometheus for granular dashboards (item 17.6)

---

## How to Verify Currently Implemented Features

```bash
# 1. Run all tests (26 pass)
source .venv/bin/activate
MPLBACKEND=Agg python -m pytest tests/ -v

# 2. Train pipeline (end-to-end)
python -m src.pipelines.train_pipeline \
    --config configs/regression/construction_duration_v1.yaml --register

# 3. Register & promote a model
python -m src.pipelines.register_pipeline \
    --run-id <RUN_ID> --model-name construction_duration --alias champion --approve

# 4. Start inference service
uvicorn deployment.app:app --host 0.0.0.0 --port 8000

# 5. Start monitoring stack
docker compose -f deployment/docker-compose.yml up -d

# 6. Generate traffic & check dashboards
python scripts/generate_traffic.py --requests 100
# Then open: http://localhost:3000 (Grafana) and http://localhost:9090 (Prometheus)
```
