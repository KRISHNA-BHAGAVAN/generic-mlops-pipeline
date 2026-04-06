# Product Requirements Document (PRD) – UPDATED v3.0
## Generic MLOps Pipeline with Integrated Observability

**Document Version:** 3.0 (Updated with Observability Stack)  
**Status:** Ready for Implementation  
**Date:** April 2026

---

## Executive Summary

This PRD defines a **production-ready, config-driven MLOps pipeline** with integrated observability and monitoring capabilities. The pipeline supports multiple team members running diverse experiments while maintaining complete traceability, reproducibility, and operational visibility through a modern observability stack.

**Key Addition (v3.0):** Integrated monitoring with Prometheus, Grafana, and EvidentlyAI to detect model performance degradation, data drift, and infrastructure issues in production.

---

## Overview

This PRD defines a scalable, modular, and collaborative MLOps pipeline for a team that runs many experiments across multiple machine learning goals, model families, datasets, feature sets, and hyperparameter configurations. The core design principle is that experiments must change through configuration and metadata, not through changes to core pipeline code. This keeps the platform reusable, reproducible, and easy to productionize.

The pipeline includes comprehensive observability from the start, enabling teams to:
- Track model performance degradation in production
- Detect data and prediction drift automatically
- Monitor system health (latency, throughput, errors)
- Create alerts for anomalies and threshold violations
- Maintain compliance and audit trails

---

## Purpose

The purpose of this project is to build a **production-grade MLOps foundation** that enables team members to:

- Run experiments using different models, parameters, feature sets, datasets, and goals without modifying core pipeline code.
- Track and compare experiments across users, tasks, and model families in a shared system.
- Reproduce experiments reliably across environments and collaborators.
- Select the best model candidate based on explicit validation criteria.
- Promote successful models from development to staging and production with minimal code change.
- Serve selected models in production using registry-driven deployment patterns rather than hardcoded artifacts.
- **[NEW] Monitor model performance, data drift, and infrastructure health in production.**
- **[NEW] Receive alerts and automatically trigger retraining when performance degrades.**

---

## Goals

### Primary Goals

- Standardized ML workflow: ingest → validate → transform → train → evaluate → track → register → promote → **deploy → monitor → alert**.
- Static core pipeline with config-driven experiments.
- Support for multiple experiment goals such as regression and classification in the same platform.
- Support for multiple model families and parameter combinations.
- Reproducible experiments through versioned code, data, configs, and tracked metadata.
- Centralized experiment comparison and model selection workflow.
- Easy promotion of winning models into production using MLflow Model Registry aliases and metadata.
- **[NEW] Real-time monitoring of model predictions, system health, and data quality.**
- **[NEW] Automated drift detection and performance degradation alerts.**
- **[NEW] Centralized observability dashboard for operational visibility.**

### Secondary Goals

- Support different feature sets and target columns per experiment.
- Support dynamic input and output schemas per registered model.
- Support multiple deployment targets over time.
- Allow future expansion into CI/CD, scheduled retraining, and environment-aware model governance.
- Keep local setup simple in the initial phase while preserving a path to production-grade operations.
- **[NEW] Extensible monitoring architecture supporting OpenTelemetry, custom metrics, and third-party integrations.**
- **[NEW] Historical monitoring data retention for trend analysis and root cause investigation.**

---

## Non-Goals (Initial Phase)

- Kubernetes-based deployment.
- Real-time streaming pipelines.
- Complex orchestration with Airflow, Kubeflow, or Dagster in phase 1.
- Full-featured monitoring, drift detection, or alerting in phase 1. *(Note: Basic monitoring is included in Phase 1.1; advanced drift detection is Phase 1.2)*
- Automated champion/challenger rollout in production traffic in phase 1.
- AutoML as a required feature in the first milestone.
- Custom application-level RBAC inside the pipeline codebase.
- Machine learning-specific profiling (e.g., GPU memory per model layer) in phase 1.
- Distributed tracing with service mesh integration in phase 1.

---

## Key Principles

1. **Experiments change configs, metadata, and registry state — not core code.**
2. **Observability is built-in, not bolted-on.** All services expose metrics from the start.
3. **Open standards first.** Use Prometheus, Grafana, and OpenTelemetry-compatible formats.
4. **Three pillars of monitoring:** System health (latency, errors, throughput), Model quality (accuracy, drift), Data quality (missing values, distributions).

---

## Problem Statement

Machine learning teams need a platform that can support experiments with different goals, feature sets, target columns, evaluation logic, and deployment candidates while keeping the core pipeline stable. The system must support end-to-end reproducibility, comparison, validation, and production promotion without relying on hardcoded artifacts or ad hoc workflow variations.

Additionally, once models are in production, teams need visibility into:
- Whether the model is still performing well
- Whether input data has shifted in ways that degrade performance
- Whether the system is handling traffic efficiently
- Whether predictions have become skewed or erratic

Without this visibility, models silently degrade and teams miss critical issues until user impact occurs.

---

## Users and Responsibilities

### ML Engineer / Data Scientist

- Creates experiment configs.
- Selects dataset version, features, target, model family, and hyperparameters.
- Runs experiments and compares outcomes.
- Registers promising models.
- **[NEW] Monitors model performance dashboards to detect degradation.**
- **[NEW] Investigates drift alerts and decides on retraining.**

### Reviewer / Lead

- Reviews tracked runs and evaluation results.
- Applies validation decisions.
- Approves promotion candidates.
- Selects production-ready models.
- **[NEW] Reviews monitoring dashboards for production model health.**
- **[NEW] Approves threshold changes for drift detection and alerting.**

### Platform / Deployment Owner

- Maintains MLflow, DVC, DagsHub integration, containerization, and deployment infrastructure.
- Manages environment-level registry conventions.
- Ensures serving applications load models by alias or environment-specific registry reference.
- **[NEW] Maintains Prometheus, Grafana, and EvidentlyAI infrastructure.**
- **[NEW] Configures alerts and notification channels.**
- **[NEW] Manages observability data retention and archival policies.**

### Access Control Approach

The initial phase should rely on platform-level access management rather than custom in-app authorization. DagsHub provides a hosted MLflow server with team-based access control for each repository. Grafana and Prometheus can be accessed through environment-based permissions (dev/staging/prod) managed at the infrastructure level.

---

## Functional Scope

### Supported Experiment Types

The platform must support at minimum:
- Regression experiments.
- Classification experiments.

Future-compatible extension points should allow support for:
- Ranking.
- Clustering.
- Forecasting.
- Custom Python-function models.

---

## High-Level Workflow

**Development & Experimentation:**
```
Data → Validation → Feature Preparation → Task-Specific Training → 
Task-Specific Evaluation → MLflow Logging → Model Registration → 
Candidate Review → Alias/Environment Promotion
```

**Production & Monitoring:**
```
Deployment → Request Handling → Prediction Logging → Metrics Collection → 
Prometheus Scraping → Grafana Visualization → Alert Evaluation → 
(If degradation detected) → Notification / Retraining Trigger
```

This workflow must remain consistent even if the dataset, model family, feature columns, target column, or evaluation metrics differ between experiments.

---

## Required Platforms and Runtime

### DagsHub

DagsHub is a required part of the initial implementation, not a future enhancement. The platform must use DagsHub as the hosted collaboration layer for the repository, remote MLflow tracking server, and shared model registry experience.

### MLflow

MLflow is the central experiment tracking and model registry system. All runs, metrics, parameters, and artifacts are logged to MLflow. MLflow also provides the model serving and version management capabilities required for production deployment.

### Docker

Docker is required for consistent, reproducible deployments across environments. The FastAPI inference service and all monitoring components are containerized.

### Prometheus

**[NEW in Phase 1.1]** Prometheus is the metrics collection and time-series database. All application services (FastAPI, training pipeline, batch monitoring jobs) expose Prometheus-format metrics on a `/metrics` endpoint.

**Deployment:**
- Local: Prometheus container listening on `http://localhost:9090`
- Production: Managed Prometheus service or self-hosted cluster

**Configuration:**
- Scrape intervals: 15s default
- Data retention: 15 days default (configurable)
- Storage: Time-series database on persistent volume

### Grafana

**[NEW in Phase 1.1]** Grafana is the visualization and alerting frontend for observability. Dashboards display real-time metrics from Prometheus, MLflow metrics, and EvidentlyAI reports.

**Deployment:**
- Local: Grafana container listening on `http://localhost:3000`
- Production: Managed Grafana or self-hosted instance

**Dashboards:**
- System Health (CPU, memory, request latency, error rate)
- Model Performance (accuracy, precision, recall, prediction distribution)
- Data Drift Detection (feature distributions, outlier detection)

### EvidentlyAI

**[NEW in Phase 1.2]** EvidentlyAI is the specialized library for ML monitoring, detecting data drift, prediction drift, and data quality issues.

**Integration:**
- Batch monitoring: Scheduled jobs that run hourly/daily
- Metrics export: EvidentlyAI metrics sent to Prometheus
- Reports: Interactive HTML reports generated and logged to MLflow

---

## Core Requirements

### Config-Driven Experiments

Every experiment must be fully defined by a YAML configuration file. Config files must include enough information for the pipeline to execute without code edits. At minimum, each config must define:

- experiment name,
- user,
- task type,
- dataset reference,
- dataset version or DVC path,
- feature list,
- target column,
- model type,
- model parameters,
- split strategy,
- preprocessing options,
- metrics to compute,
- MLflow tags,
- optional registry name.

### Multi-Goal Support

The system must support experiments with different goals in the same repository and tracking backend. A regression experiment and a classification experiment may run against the same dataset or different datasets. The pipeline must detect the configured `task_type` and route to the appropriate training, prediction, and evaluation logic.

### Model Signatures & Input/Output Schemas

All trained models must produce MLflow model signatures that define input and output schemas. These schemas are used by:
- The inference service to validate incoming requests
- Monitoring tools to detect schema drift
- EvidentlyAI to identify structural changes in data

### Experiment Logging

All runs must be logged to MLflow with:
- Configuration parameters
- Model hyperparameters
- Evaluation metrics
- Model artifacts
- **[NEW] Input/output signatures**
- **[NEW] Training and inference examples for reference**

### Production Model Serving

Trained models are deployed via FastAPI service that:
- Loads models from MLflow registry by alias
- Validates incoming requests against model signature
- Generates predictions
- **[NEW] Logs predictions and ground truth (when available) for monitoring**
- **[NEW] Exposes Prometheus metrics (`/metrics` endpoint)**
- **[NEW] Records latency, throughput, and error metrics**

### Monitoring & Observability

**[NEW] Phase 1.1:**
- All services expose metrics in Prometheus format
- Prometheus collects metrics at regular intervals
- Grafana dashboards visualize system health and model performance
- Basic alerts on high latency, error rate, and request volume

**[NEW] Phase 1.2:**
- Batch monitoring jobs run periodically (hourly/daily)
- EvidentlyAI detects data drift and quality issues
- Drift detection metrics logged to Prometheus
- Grafana dashboards show drift trends
- Alerts trigger when drift exceeds thresholds
- Optional automated retraining when drift detected

---

## Monitoring Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Production System                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  FastAPI Service (Model Serving)                           │
│  ├─ /predict (inference endpoint)                          │
│  ├─ /metrics (Prometheus metrics)                          │
│  └─ Logs: predictions, latency, errors                     │
│                                                              │
│  Batch Monitoring Job                                       │
│  ├─ Runs hourly/daily                                       │
│  ├─ Loads recent predictions & ground truth                │
│  ├─ Calculates drift metrics                               │
│  └─ Pushes to Prometheus PushGateway                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   Metrics Collection                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Prometheus                                                  │
│  ├─ Scrapes /metrics endpoints every 15s                   │
│  ├─ Stores time-series data                                │
│  ├─ Evaluates alert rules                                  │
│  └─ Exposes PromQL API for queries                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Visualization & Alerting                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Grafana                                                     │
│  ├─ Dashboards (System Health, Model Performance)          │
│  ├─ Alert Manager (Slack, email, webhooks)               │
│  └─ Data source: Prometheus + MLflow                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Metrics Categories

#### System Health Metrics (Phase 1.1)
- **Request Latency:** Histogram of inference request duration
- **Request Volume:** Counter of successful/failed predictions
- **Error Rate:** Percentage of requests returning errors
- **Service Availability:** Uptime and health check status
- **Resource Utilization:** CPU, memory, disk usage

#### Model Performance Metrics (Phase 1.1)
- **Prediction Distribution:** Histogram of predicted values
- **Batch Accuracy/Precision/Recall:** Periodic calculation when ground truth available
- **Inference Time per Model:** Distinguish slow vs fast models

#### Data Quality Metrics (Phase 1.2)
- **Feature Distributions:** Compare production to training data
- **Missing Values:** Percentage of missing features
- **Outlier Detection:** Count of unusual feature combinations
- **Schema Compliance:** Requests matching expected schema

#### Drift Metrics (Phase 1.2)
- **Data Drift:** Statistical tests (PSI, KL-divergence, Wasserstein) on feature distributions
- **Prediction Drift:** Statistical tests on prediction distribution changes
- **Performance Drift:** Changes in model accuracy/metrics when ground truth available

---

## Technology Stack Summary

### Core Stack (Phase 1.1 – MVP)
| Component | Purpose | Status |
|-----------|---------|--------|
| Python 3.10+ | Language | Required |
| FastAPI | Model serving | Required |
| MLflow | Experiment tracking & registry | Required |
| DVC | Data versioning | Required |
| Docker | Containerization | Required |
| Prometheus | Metrics collection | **[NEW] Required Phase 1.1** |
| Grafana | Visualization | **[NEW] Required Phase 1.1** |

### Advanced Stack (Phase 1.2+)
| Component | Purpose | Status |
|-----------|---------|--------|
| EvidentlyAI | Drift detection | Phase 1.2 |
| OpenTelemetry | Tracing (future) | Phase 2 |
| AlertManager | Alert routing (with Prometheus) | Phase 1.2 |

---

## Implementation Timeline

### Phase 1.1 (MVP – Weeks 1-4)
- Config-driven experiment pipeline (existing)
- MLflow integration (existing)
- FastAPI inference service (existing)
- **[NEW] Prometheus metrics instrumentation**
- **[NEW] Grafana dashboards (system health, request metrics)**
- **[NEW] Basic alerting on latency and error rates**
- **[NEW] Containerization of monitoring stack**

### Phase 1.2 (Enhanced – Weeks 5-8)
- **[NEW] EvidentlyAI integration**
- **[NEW] Batch monitoring jobs**
- **[NEW] Drift detection dashboards**
- **[NEW] Advanced alerting with Slack integration**
- **[NEW] Historical monitoring data analysis**
- Model ranking and selection (existing)

### Phase 2+ (Advanced)
- OpenTelemetry instrumentation
- Distributed tracing
- GPU/hardware profiling
- LLM-specific monitoring (token counts, cost)
- Automated retraining triggers

---

## Repository Structure (Updated)

```text
generic-mlops-pipeline/
├── data/
├── src/
│   ├── config/
│   │   ├── load_config.py
│   │   ├── validate_config.py
│   ├── models/
│   │   ├── factory.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── registry.py
│   ├── schemas/
│   │   ├── experiment_schema.py
│   │   ├── inference_schema.py
│   ├── selection/
│   │   ├── rank_runs.py
│   │   ├── promote_model.py
│   ├── pipelines/
│   │   ├── train_pipeline.py
│   │   ├── register_pipeline.py
│   │   ├── deployment_pipeline.py
│   ├── monitoring/
│   │   ├── metrics.py                    # [NEW] Prometheus metrics definitions
│   │   ├── drift_detector.py             # [NEW] EvidentlyAI integration
│   │   ├── batch_monitor.py              # [NEW] Batch monitoring job
│   │   ├── prediction_logger.py          # [NEW] Log predictions for monitoring
├── configs/
│   ├── regression/
│   ├── classification/
├── deployment/
│   ├── app.py
│   ├── model_loader.py
│   ├── request_validation.py
│   ├── Dockerfile
│   ├── docker-compose.yml                # [NEW] Multi-container orchestration
├── monitoring/                            # [NEW] Monitoring infrastructure
│   ├── prometheus.yml                    # [NEW] Prometheus config
│   ├── prometheus.rules.yml              # [NEW] Alert rules
│   ├── grafana/
│   │   ├── Dockerfile                    # [NEW] Custom Grafana image
│   │   ├── dashboards/
│   │   │   ├── system_health.json        # [NEW] System metrics dashboard
│   │   │   ├── model_performance.json    # [NEW] Model metrics dashboard
│   │   │   ├── data_drift.json           # [NEW] Drift detection dashboard
│   │   ├── datasources.yml               # [NEW] Grafana data source config
├── .dvc/
├── dvc.yaml
├── pyproject.toml
├── uv.lock
├── Makefile
├── README.md
```

---

## Workflow Requirements

### Experiment Workflow (Unchanged)

1. Pull latest code.
2. Pull latest DVC data.
3. Create or update experiment config.
4. Validate config.
5. Run pipeline with one command.
6. Inspect MLflow results.
7. Register promising run if successful.
8. Mark validation status.
9. Promote alias when approved.
10. Deploy or reload inference service.

### Production Monitoring Workflow (New)

1. **Daily:** Review Grafana dashboards for model performance and system health.
2. **On Drift Alert:** Investigate root cause (data change, model degradation, or both).
3. **Decision Point:** 
   - If minor drift: Continue monitoring.
   - If significant drift: Run batch retraining on recent data.
4. **Post-Retraining:** Compare new model to current champion, promote if better.
5. **Archive:** Log monitoring findings and decisions for audit trail.

---

## Success Criteria

The system is successful if:

- team members can run regression and classification experiments without changing core pipeline code,
- experiments with different feature sets and targets can share the same platform,
- all runs are traceable by config, code, data version, and model artifact,
- the team can compare runs by business goal and choose a winner confidently,
- the winning model can be promoted through the registry with minimal friction,
- production services can switch to a better model without code changes other than alias or environment updates,
- **[NEW] model performance degradation is detected within hours of occurrence,**
- **[NEW] data drift is detected before it significantly impacts predictions,**
- **[NEW] the team has visibility into inference latency, error rates, and request volume,**
- **[NEW] alerts automatically notify the team of anomalies and threshold violations.**

---

## Feasibility Assessment

### Immediately Feasible (Phase 1.1)

- Add `task_type`, `target_column`, and `feature_columns` to config.
- Add config schema validation.
- Add task-aware model factory and evaluator.
- Log signatures and richer run metadata.
- Add DagsHub-backed MLflow tracking as the default collaborative setup.
- Add registry integration for registering successful runs.
- Update FastAPI to load by registry alias.
- Add and maintain the inference Dockerfile.
- **[NEW] Instrument FastAPI service with Prometheus metrics.**
- **[NEW] Set up Prometheus and Grafana containers.**
- **[NEW] Create system health and model performance dashboards.**
- **[NEW] Configure basic alerts (latency, error rate).**

### Feasible in Short Term (Phase 1.2 / Phase 2)

- Programmatic ranking and best-run selection tooling.
- Validation status tagging and approval workflow.
- Environment-specific registered model naming.
- Promotion scripts that copy model versions across environments.
- **[NEW] Integrate EvidentlyAI for drift detection.**
- **[NEW] Batch monitoring jobs that run hourly/daily.**
- **[NEW] Drift detection dashboards and alerts.**
- **[NEW] Slack/email notification integration.**

### Later Phase Enhancements

- CI/CD integration for registration and promotion.
- Automated smoke tests on candidate models.
- Shadow deployments and champion/challenger traffic splitting.
- **[NEW] Advanced monitoring (OpenTelemetry, distributed tracing).**
- **[NEW] Automated retraining triggers based on drift thresholds.**
- **[NEW] A/B testing framework with statistical significance testing.**

---

## Constraints and Rules

### Strict Rules

- No core pipeline edits for normal experimentation.
- No manual dataset sharing.
- No production deployment from arbitrary local files.
- No promotion without metadata and traceability.
- No config execution without validation.
- **[NEW] All production models must expose metrics on `/metrics` endpoint.**
- **[NEW] All predictions must be logged (at least model ID, timestamp, prediction).**
- **[NEW] Monitoring data must be retained for at least 30 days.**

### Best Practices

- Always define experiments via YAML.
- Always log signatures when possible.
- Always log dataset version and feature list.
- Always use registry aliases or environment-specific model URIs for deployment.
- Always document validation status before production promotion.
- **[NEW] Always review drift detection dashboards before retraining decisions.**
- **[NEW] Always investigate alert anomalies to understand root cause.**
- **[NEW] Always update alert thresholds based on observed data characteristics.**

---

## Implementation Recommendation

The updated scope is highly feasible. Phase 1.1 adds observability as a first-class citizen without significantly delaying core functionality delivery. Most observability features leverage well-established, open-source tools (Prometheus, Grafana) with large communities and extensive documentation.

The largest effort in Phase 1.1 is:
1. **Metrics instrumentation** in FastAPI and training pipeline (2-3 days)
2. **Dashboard creation** in Grafana (1-2 days)
3. **Docker Compose setup** for local development (1 day)

Phase 1.2 builds on this foundation by adding specialized monitoring (EvidentlyAI) for ML-specific concerns like data drift.

---

## Observability ROI

**Why Phase 1.1 observability investment is worth it:**

1. **Early Detection:** Catch model degradation within hours, not weeks.
2. **Reduced Debug Time:** Metrics show *what* is wrong; dashboards help pinpoint *why*.
3. **Compliance & Audit:** Permanent record of model behavior and decisions.
4. **Team Confidence:** Visible proof that models are healthy builds trust in ML systems.
5. **Cost Optimization:** Identify over-provisioned resources and inefficient inference patterns.

