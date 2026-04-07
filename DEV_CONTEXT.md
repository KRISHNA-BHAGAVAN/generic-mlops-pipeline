# PRD — Generic MLOps Pipeline with Integrated Observability

> For the full detailed PRD with architecture diagrams and specification, see [documents/PRD_UPDATED_v3.md](documents/PRD_UPDATED_v3.md).

## Problem Statement

ML teams need a standardized, config-driven pipeline for running diverse experiments across datasets, models, and hyperparameters — with built-in production monitoring. Most teams either use ad-hoc scripts (no reproducibility) or over-engineered platforms (too complex).

## Solution

A **production-ready, config-driven MLOps pipeline** where experiments change through YAML configuration, not code changes. Integrated observability (Prometheus, Grafana, EvidentlyAI) is a first-class citizen, not an afterthought. Prediction drift baselines are stored as MLflow artifacts so drift checks compare **model outputs**, not labels.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│                    YAML Config                         │
│  (experiment_name, model_type, features, metrics...)   │
└──────────────┬─────────────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────┐
│              Training Pipeline CLI                      │
│  Load Config → Validate → Load Data → Preprocess       │
│  → Split → Train → Evaluate → Log to MLflow            │
└──────────────┬─────────────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────┐
│              MLflow (DagsHub)                           │
│  Experiment Tracking → Model Registry → Promotion      │
└──────────────┬─────────────────────────────────────────┘
               │
┌──────────────▼─────────────────────────────────────────┐
│              FastAPI Inference Service                   │
│  /predict → /predict/batch → /health → /metrics         │
└──────────────┬────────────────┬────────────────────────┘
               │                │
┌──────────────▼───────┐  ┌─────▼────────────────────────┐
│   PostgreSQL         │  │   Monitoring Stack           │
│   (Prediction Logs   │  │   Prometheus → Grafana Cloud │
│    + Scheduler Jobs) │  │   AlertManager → EvidentlyAI │
└──────────────────────┘  │   Pushgateway                │
                          └────────────────────────────────┘
```

## Core Principles

1. **Config-Driven:** Experiments defined entirely through YAML configs
2. **Static Core:** Pipeline code never changes between experiments
3. **Dataset-Agnostic:** No hardcoded dataset logic; all configured
4. **Observable by Default:** Prometheus metrics, Grafana dashboards, drift detection
5. **Registry-Based:** MLflow Model Registry with alias-based promotion (champion/candidate)
6. **Monitoring-Ready:** Batch drift checks run on a scheduler (APScheduler + persistent job store)

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.12 |
| Package Manager | uv |
| ML Framework | scikit-learn |
| Experiment Tracking | MLflow (DagsHub) |
| Data Versioning | DVC (DagsHub Storage) |
| Config Validation | Pydantic v2 |
| Serving | FastAPI |
| Database | PostgreSQL + SQLAlchemy 2.0 |
| Monitoring | Prometheus + Grafana (local / Grafana Cloud) |
| Drift Detection | EvidentlyAI |
| Containerization | Docker + Docker Compose |
| CLI | Click |

## File Structure

```
generic-mlops-pipeline/
├── configs/                    # Experiment YAML configs
│   ├── regression/
│   └── classification/
├── data/raw/                   # Dataset files
├── deployment/                 # Serving & infrastructure
│   ├── app.py                  # FastAPI inference service
│   ├── Dockerfile
│   └── docker-compose.yml
├── monitoring/                 # Prometheus, Grafana configs
│   ├── prometheus.yml          # Local-mode Prometheus config
│   ├── prometheus.cloud.yml    # Cloud-mode (Grafana Cloud remote_write)
│   ├── prometheus.rules.yml
│   ├── alertmanager.yml
│   └── grafana/
├── .secrets/                   # Local secrets (gitignored)
│   └── grafana_cloud_metrics_token
├── src/
│   ├── config/                 # Config loading & validation
│   ├── data/                   # Data loading & validation
│   ├── features/               # Feature engineering
│   ├── models/                 # Factory, training, evaluation, registry
│   ├── monitoring/             # Prometheus metrics, drift detection, DB engine
│   ├── pipelines/              # CLI pipelines (train, register)
│   ├── schemas/                # Pydantic request/response models
│   ├── selection/              # Run ranking & model promotion
│   └── utils/                  # Logger, helpers
├── tests/                      # Unit & integration tests
├── scripts/                    # Utility scripts
├── Makefile                    # Pipeline commands
├── pyproject.toml              # Dependencies
└── .env.example                # Environment variable template
```

## Supported Tasks

- **Regression:** LinearRegression, RandomForestRegressor
- **Classification:** LogisticRegression, RandomForestClassifier

## Current Dataset

- `data/raw/construction_dataset.csv` — 1,300 construction project tasks with features like Labor_Required, Equipment_Units, Material_Cost_USD, Risk_Level, etc.

## Workflows

### Training
```bash
python -m src.pipelines.train_pipeline --config configs/regression/construction_duration_v1.yaml
```

### Model Registration & Promotion
```bash
python -m src.pipelines.register_pipeline --run-id <ID> --model-name construction_duration --alias champion
```

### Serving
```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

### Monitoring
```bash
# Local mode (default)
docker compose -f deployment/docker-compose.yml up -d

# Cloud mode (Grafana Cloud shared dashboards)
docker compose -f deployment/docker-compose.yml --profile cloud up -d
```

### Batch Monitoring (one-shot)
```bash
python -m src.monitoring.batch_monitor \
    --model-name construction_duration \
    --reference-data-path data/processed/train_features.csv \
    --feature-columns "feat1,feat2" \
    --hours 24
```

### Batch Monitoring Scheduler (daemon)
```bash
export MONITORING_MODEL_NAME=construction_duration
export MONITORING_REFERENCE_DATA_PATH=data/processed/train_features.csv
export MONITORING_FEATURE_COLUMNS="feat1,feat2"
python scripts/schedule_monitoring.py
```

## Shared Services (Team Collaboration)

| Service | Purpose | Where |
|---------|---------|-------|
| DagsHub (DVC) | Shared datasets & model artifacts | Cloud (DagsHub Storage) |
| DagsHub (MLflow) | Shared experiment tracking & model registry | Cloud |
| PostgreSQL | Prediction logs & scheduler job store | Local Docker (dev) / Supabase (prod) |
| Grafana Cloud | Shared monitoring dashboards | Cloud (free tier) |
| Prometheus | Metrics collection & forwarding | Local Docker per developer |
