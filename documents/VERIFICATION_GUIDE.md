# Generic MLOps Pipeline — Verification Guide

This guide details exactly how to verify the various operational stages and components of your MLOps pipeline. Each command serves a specific part in validating the end-to-end functionality.

---

### 0. Start PostgreSQL (required for prediction logging & tests)

```bash
docker compose -f deployment/docker-compose.yml up -d postgres
```

* **What is this?** Starts the local PostgreSQL container that stores prediction logs and scheduler job state. This must be running before the inference service, batch monitor, or tests.
* **Why is it implemented?** PostgreSQL replaces the previous SQLite file-based logging to support concurrent team access and production-like environments.
* **How to verify?** Run `docker compose -f deployment/docker-compose.yml ps postgres`. It should show `running (healthy)`. You can also connect directly: `psql postgresql://mlops:mlops@localhost:5432/mlops`.

---

### 1. Run Automated Tests

```bash
source .venv/bin/activate
MPLBACKEND=Agg python -m pytest tests/ -v
```

* **What is this?** This executes the automated unit and integration tests using pytest within your Python virtual environment. `MPLBACKEND=Agg` ensures that matplotlib plots generated during evaluations don't try to open a physical window on your machine. **Note:** PostgreSQL must be running (see Step 0) for monitoring tests.
* **Why is it implemented?** To ensure that every newly added component (like model training, drift detection, and logging) works perfectly without breaking older foundational logic.
* **How to verify?** If the codebase is stable, all tests should print `PASSED` with a green message at the end. Check for `0 failures` in the output summary.

---

### 2. Train Pipeline (end-to-end)

```bash
python -m src.pipelines.train_pipeline \
    --config configs/regression/construction_duration_v1.yaml --register
```

* **What is this?** This command triggers the entire training lifecycle. It reads the configured YAML file, pulls the dataset, prepares the feature spaces, trains the model, computes evaluation metrics, and the `--register` flag signals MLflow to append the produced model specifically into your tracked model registry after saving plots and configurations.
* **Optional registry metadata:** use `--register-description`, `--register-alias`, `--register-tag key=value` (repeatable), and `--register-created-by` to attach descriptions and tags during registration. These can also be defined in the experiment config using `registry_description`, `registry_tags`, `registry_alias`, and `registry_created_by`.
* **Why is it implemented?** It abstracts all the manual notebook procedures into a configurable, production-grade CLI runner that connects models directly to your DagsHub MLflow Tracking instance.
* **How to verify?** The console will progress through training logs. Go to your DagsHub MLflow UI Tracking page and verify that a new experiment run has been logged, metrics exist, and artifacts (like `confusion_matrix.png` and `config.yaml`) are attached.

---

### 3. Register & Promote a Model

```bash
python -m src.pipelines.register_pipeline \
    --run-id <RUN_ID> --model-name construction_duration \
    --description "Production candidate" \
    --tag team=engineering --tag stage=candidate \
    --created-by krishna \
    --alias champion --approve
```

* **What is this?** This command registers a completed MLflow run into the model registry and optionally attaches metadata and aliases in one step.
* **Why is it implemented?** It gives teams a declarative way to add descriptive registry metadata and alias a version during registration, which makes model comparison and review easier in MLflow.
* **How to verify?** Under the "Models" tab in DagsHub MLflow, you should see `construction_duration` with the associated description, tags, created_by metadata, and alias such as `champion`.

---

```bash
python -m src.selection.rank_runs \
    --experiment-name generic-mlops-pipeline \
    --metric mse \
    --auto-promote \
    --model-name construction_duration
```

* **What is this?** This interrogates the MLflow tracking server to find the best model produced thus far. It ranks all listed experiments, picks the model with the lowest `mse` (Mean Squared Error), aliases it properly, and promotes it within the Model Registry entirely via CLI without UI interaction.
* **Why is it implemented?** In automated CI/CD infrastructures, you need a deterministic, programmable manner to evaluate the best candidate on record and deploy it safely.
* **How to verify?** Under the "Models" tab in DagsHub MLflow, you should see `construction_duration` tagged dynamically with an alias such as `candidate` or `champion` indicating it was successfully promoted.

---

### 4. Start Inference Service

```bash
uvicorn deployment.app:app --host 0.0.0.0 --port 8000
```

* **What is this?** This spins up the FastAPI backend locally. Behind the scenes, the backend queries MLflow to pull the exact `champion` model registered previously, allowing it to receive real network payloads via HTTP requests to make predictions.
* **Why is it implemented?** This mimics production servers. End-users and applications invoke the API server for endpoints like `/predict`, rather than directly triggering Python scripts or notebooks.
* **How to verify?** Open a new terminal and run `curl http://localhost:8000/health`. You should immediately receive a JSON response saying `{"status": "ok"}` indicating the FastApi app and metrics middleware are up.

#### 4.1 Test All API Endpoints with Postman

Use a Postman environment variable:

- `base_url = http://localhost:8000`

Then test each endpoint in this order:

1. **Health Check**

```http
GET {{base_url}}/health
```

Expected result:
- `200 OK`
- JSON contains `status: "ok"`, `timestamp`, and `models_loaded`.

2. **Single Prediction**

```http
POST {{base_url}}/predict
Content-Type: application/json
```

```json
{
    "features": {
        "Labor_Required": 14,
        "Equipment_Units": 6,
        "Material_Cost_USD": 16789.73,
        "Start_Constraint": 0,
        "Resource_Constraint_Score": 0.41,
        "Site_Constraint_Score": 0.59,
        "Dependency_Count": 4
    },
    "model_name": "construction_duration",
    "model_alias": "champion"
}
```

Expected result:
- `200 OK`
- JSON contains `prediction`, `model_name`, `model_version`, `model_uri`, `timestamp`.

3. **Batch Prediction**

```http
POST {{base_url}}/predict/batch
Content-Type: application/json
```

```json
{
    "instances": [
        {
            "Labor_Required": 14,
            "Equipment_Units": 6,
            "Material_Cost_USD": 16789.73,
            "Start_Constraint": 0,
            "Resource_Constraint_Score": 0.41,
            "Site_Constraint_Score": 0.59,
            "Dependency_Count": 4
        },
        {
            "Labor_Required": 2,
            "Equipment_Units": 2,
            "Material_Cost_USD": 16885.80,
            "Start_Constraint": 5,
            "Resource_Constraint_Score": 0.75,
            "Site_Constraint_Score": 0.17,
            "Dependency_Count": 3
        }
    ],
    "model_name": "construction_duration",
    "model_alias": "champion"
}
```

Expected result:
- `200 OK`
- JSON contains `predictions` (array), `count`, `model_name`, `model_version`, `timestamp`.

4. **Model Reload**

```http
POST {{base_url}}/models/reload?model_name=construction_duration&alias=champion
```

Expected result:
- `200 OK`
- JSON contains `status: "reloaded"` and model identifier.

5. **Prometheus Metrics Endpoint**

```http
GET {{base_url}}/metrics
```

Expected result:
- `200 OK`
- Plain text metrics output including counters/histograms such as:
    - `mlops_api_request_total`
    - `mlops_api_request_duration_seconds`
    - `mlops_predictions_total`
    - `mlops_prediction_duration_seconds`

Helpful negative tests (optional):
- Send `/predict` without `features` and confirm `422` validation error.
- Send invalid `model_alias` and confirm model-load failure (typically `503`).
- Send malformed features to observe error metrics and alerts.

---

### 5. Start Monitoring Stack

```bash
docker compose -f deployment/docker-compose.yml up -d
```

* **What is this?** Docker-compose spins up multiple independent telemetry services in the background: Prometheus (scraping endpoint metrics), Pushgateway (receiver for batch jobs), Grafana (visualization boards), and AlertManager (distributor for alerts to Slack). The `-d` parameter detaches the terminal, letting them run silently.
* **Why is it implemented?** An MLOps application must be fully observable 24/7. These core infrastructure components allow real-time analysis of the API and structural drift without halting system operations.
* **How to verify?** Run `docker ps` to verify that `prometheus`, `grafana`, `alertmanager`, etc. are listed. Then, open `http://localhost:3000` via your browser to view Grafana (default login: admin / admin).

#### 5.1 What to Look for in Prometheus and Grafana

Do you need to set up dashboards manually?
- **No (local default):** Grafana datasource + dashboards are provisioned automatically by Docker mounts and provisioning files.
- You only need manual setup if you create custom dashboards or switch to a different datasource/provider.

Prometheus checks (open `http://localhost:9090` -> **Graph**):

1. **Service Up**
```promql
up{job="fastapi-service"}
```
Should be `1`.

2. **Traffic Rate**
```promql
sum(rate(mlops_api_request_total[1m]))
```
Should increase when Postman or traffic scripts run.

3. **API Latency (p95)**
```promql
histogram_quantile(0.95, rate(mlops_api_request_duration_seconds_bucket[5m]))
```
Use this to monitor response-time regressions.

4. **Error Rate**
```promql
sum(rate(mlops_api_request_total{status=~"5.."}[5m])) / sum(rate(mlops_api_request_total[5m]))
```
Should stay near `0` under healthy traffic.

5. **Prediction Error Events**
```promql
increase(mlops_prediction_errors_total[5m])
```
Should spike only when intentionally testing failure paths.

Grafana checks (open `http://localhost:3000` -> folder **MLOps**):

1. **MLOps System Health** dashboard
- Request Rate should rise during testing.
- API p50/p95/p99 should remain stable.
- Error Rate should be near zero except during fault injection.
- Active Requests should move with concurrent load.

2. **MLOps Model Performance** dashboard
- Inference latency (p50/p95) should track request volume.
- Predictions per second should rise when calling `/predict` and `/predict/batch`.
- Models Loaded should become non-zero after first successful inference.
- Model Load Time may spike during first load or `/models/reload`.

3. **MLOps Data Drift Detection** dashboard
- `Data Drift Status` and `Drifted Columns Count` update after running batch monitor or drift-trigger script.
- Use this to validate drift monitoring + alerting pipeline end-to-end.

Alert behavior to validate:
- Run `python scripts/trigger_test_alerts.py --type all` and confirm corresponding Prometheus alert rules transition to `Firing` after their `for` duration.

---

### 6. Generate Traffic & Trigger Alerts

```bash
python scripts/generate_traffic.py --num-requests 100
python scripts/trigger_test_alerts.py --type all
```
* **What is this?** The `generate_traffic.py` script attempts to hit your API with hundreds of randomized feature bounds to mimic chaotic user activity. `trigger_test_alerts.py` pushes deliberately malformed payloads to force 500 API crash errors, additionally pushing dummy threshold data explicitly into the Pushgateway to induce metrics-derived Drift Alerting signals.
* **Why is it implemented?** Testing dashboards manually via `curl` requests is incredibly painful. This handles creating anomalies autonomously so developers can confirm metric displays function accurately and Slack connection hooks resolve seamlessly in `AlertManager`.
* **How to verify?** Load up the Grafana **System Health** dashboard and **Model Performance** dashboard. You should visibly notice the API volume rising accompanied by an instant surge of 500-level prediction failures occurring sequentially.

---

### 7. Run Batch Monitoring

```bash
python -m src.monitoring.batch_monitor \
       --model-name construction_duration \
       --reference-data-path data/processed/train_features.csv \
       --feature-columns "feat1,feat2" \
       --target-column "duration" \
       --hours 24
```

* **What is this?** This manually runs drift detection with **EvidentlyAI**. It loads recent prediction traffic from the PostgreSQL prediction logger, compares feature distributions against the reference CSV, and (when available) compares **prediction distributions** against the MLflow reference predictions artifact (`reference_data/reference_predictions.parquet`) logged during training.
* **Why is it implemented?** Concept drift is ubiquitous in machine learning. As production inputs diverge from baseline models, mathematical boundaries must be checked to ascertain validity. If divergence is excessive, telemetry scores flag an alert to trigger retraining.
* **How to verify?** The terminal prints a structured JSON payload showing data drift and (if enabled) prediction drift. In Grafana, the **Data Drift** dashboard updates via PushGateway metrics. If the model was trained *before* reference predictions were logged or no `champion` alias exists, prediction drift will be skipped (feature drift still runs).

---

### 8. Start Batch Monitoring Scheduler Daemon

```bash
export MONITORING_MODEL_NAME=construction_duration
export MONITORING_REFERENCE_DATA_PATH=data/processed/train_features.csv
export MONITORING_FEATURE_COLUMNS="feat1,feat2"
export MONITORING_LOOKBACK_HOURS=24
export BATCH_MONITOR_CRON_HOUR=2
export BATCH_MONITOR_CRON_MINUTE=0
python scripts/schedule_monitoring.py
```

* **What is this?** This starts the **APScheduler** daemon, which runs a batch monitoring job on a cron schedule (default: daily at 02:00 UTC). The scheduler uses PostgreSQL for job persistence so scheduled jobs survive process restarts.
* **Why is it implemented?** Critical for long-run server architecture where humans cannot be expected to run command line prompts iteratively for daily concept-drift checks. It ensures autonomy.
* **How to verify?** On startup you should see an **immediate run** log (the daemon executes once at boot), followed by a line indicating the scheduler has started (e.g., `Monitoring scheduler started — cron=02:00 UTC daily`). The process then blocks until SIGINT/SIGTERM. You can also confirm the heartbeat metric `mlops_batch_job_last_run_timestamp` in Prometheus or Grafana.

---

### 9. Verify DVC Remote (Shared Datasets)

```bash
dvc remote list
dvc push
dvc pull
```

* **What is this?** Verifies that the DVC remote is correctly configured to use DagsHub Storage. `dvc remote list` should show `origin` pointing to the DagsHub S3-compatible endpoint.
* **Prerequisites:** Each team member must run the DagsHub credential setup commands (see DagsHub UI → Remote → Data → DVC) to populate `.dvc/config.local` with their access token.
* **How to verify?** `dvc remote list` outputs `origin	s3://dvc`. `dvc push` uploads tracked files to DagsHub. `dvc pull` downloads them on another machine.

---

### 10. Switch to Grafana Cloud Dashboards

```bash
# 1. Set env vars in .env
GRAFANA_CLOUD_PROM_URL=https://<your-cloud-endpoint>/api/prom/push
GRAFANA_CLOUD_PROM_USERNAME=<your-cloud-username>

# 2. Place API token in secrets file
echo "<your-api-token>" > .secrets/grafana_cloud_metrics_token

# 3. Restart the monitoring stack
docker compose -f deployment/docker-compose.yml up -d
```

* **What is this?** Switches from local-only Prometheus to cloud mode. When `GRAFANA_CLOUD_PROM_URL` is set, the Prometheus entrypoint script auto-selects `prometheus.cloud.yml` which includes `remote_write` to forward metrics to Grafana Cloud. Each developer's metrics are tagged with their `PROMETHEUS_DEVELOPER` label to avoid collisions.
* **How to verify?** Check Prometheus logs: `docker logs deployment-prometheus-1 --tail 5`. You should see `[entrypoint] Cloud mode — remote_write enabled`. Then log into Grafana Cloud → Explore → query `up{developer="your-name"}`.
