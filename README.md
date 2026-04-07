# Generic MLOps Pipeline

A production-ready, config-driven MLOps pipeline with integrated observability for running machine learning experiments.

## Description

This pipeline enables ML teams to run diverse experiments across different datasets, models, and hyperparameters through YAML configuration — no code changes needed. It includes full experiment tracking (MLflow/DagsHub), model serving (FastAPI), and production monitoring (Prometheus + Grafana + EvidentlyAI).

## Features

- 🔧 **Config-driven experiments** — YAML configs define everything: dataset, model, features, metrics
- 📊 **MLflow tracking** — All experiments logged to DagsHub with parameters, metrics, artifacts
- 🏭 **Model registry** — Register, version, and promote models (champion/candidate aliases)
- 🚀 **FastAPI inference** — Production-ready serving with batch prediction support
- 📈 **Prometheus metrics** — Request latency, prediction counts, model load times, error rates
- 📉 **Grafana dashboards** — System health, model performance, data drift visualization
- 🔍 **Drift detection** — EvidentlyAI data + prediction drift (prediction baseline logged to MLflow)
- 🔔 **Alert rules** — Configurable alerts for latency, errors, drift (console + optional Slack)
- ⏱️ **Batch monitoring scheduler** — APScheduler daemon with persistent job store + cron hook
- 🐳 **Docker Compose** — Full monitoring stack in one command
- ✅ **25 unit tests** — Config, model, and pipeline coverage

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose (for monitoring stack)
- DagsHub account (for MLflow tracking)

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/generic-mlops-pipeline.git
cd generic-mlops-pipeline

# Install dependencies
uv sync

# Install dev dependencies (for testing)
uv sync --extra dev

# Setup environment variables
cp .env.example .env
# Edit .env with your DagsHub credentials
```

## Usage

### Quick Start

```bash
# 1. Validate a config (no credentials needed)
python -m src.pipelines.train_pipeline \
    --config configs/regression/construction_duration_v1.yaml --dry-run

# 2. Run a training experiment (requires .env with DagsHub creds)
python -m src.pipelines.train_pipeline \
    --config configs/regression/construction_duration_v1.yaml

# 3. Train + auto-register model
python -m src.pipelines.train_pipeline \
    --config configs/classification/construction_risk_v1.yaml --register
```

### Model Registration & Promotion

```bash
# Register from a completed run
python -m src.pipelines.register_pipeline \
    --run-id <RUN_ID> --model-name construction_duration --alias champion --approve
```

### Start Inference Service

```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload
```

### Launch Monitoring Stack

```bash
docker compose -f deployment/docker-compose.yml up -d

# Access:
# Grafana:      http://localhost:3000 (admin/admin)
# Prometheus:   http://localhost:9090
# AlertManager: http://localhost:9093
```

### Run Batch Monitoring (one-shot)

```bash
python -m src.monitoring.batch_monitor \
    --model-name construction_duration \
    --reference-data-path data/processed/train_features.csv \
    --feature-columns "feat1,feat2" \
    --hours 24
```

**Note:** Prediction drift uses the MLflow artifact `reference_data/reference_predictions.parquet`
logged during training. If the model was trained before this feature or no
`champion` alias exists, prediction drift is skipped (feature drift still runs).

### Run Batch Monitoring Scheduler (daemon)

```bash
export MONITORING_MODEL_NAME=construction_duration
export MONITORING_REFERENCE_DATA_PATH=data/processed/train_features.csv
export MONITORING_FEATURE_COLUMNS="feat1,feat2"
export MONITORING_LOOKBACK_HOURS=24
export BATCH_MONITOR_CRON_HOUR=2
export BATCH_MONITOR_CRON_MINUTE=0
python scripts/schedule_monitoring.py
```

### Run Tests

```bash
MPLBACKEND=Agg python -m pytest tests/ -v
```

### Makefile Commands

```bash
make help          # Show all available commands
make train CONFIG=configs/regression/construction_duration_v1.yaml
make train-dry CONFIG=configs/regression/construction_duration_v1.yaml
make test
make serve
make monitoring-up
make monitoring-down
```

## Requirements

See [pyproject.toml](pyproject.toml) for full dependency list. Key technologies:

| Component | Technology |
|---|---|
| ML Framework | scikit-learn |
| Experiment Tracking | MLflow (DagsHub) |
| Config Validation | Pydantic v2 |
| Serving | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Drift Detection | EvidentlyAI |
| CLI | Click |
| Containerization | Docker |

## Tech Stack

Python 3.12 · scikit-learn · MLflow · FastAPI · Prometheus · Grafana · EvidentlyAI · Docker · Pydantic v2 · Click · uv

## Contributing

1. Create experiment configs in `configs/` — never modify core pipeline code
2. Add new model types by extending `src/models/factory.py`
3. Add new preprocessing steps by extending `src/features/build_features.py`
4. Add new metrics by extending `src/models/evaluate.py`

## License

MIT
