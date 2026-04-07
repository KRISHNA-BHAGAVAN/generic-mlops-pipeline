# Team Collaboration Shared Resources Plan

This document is the **implementation plan** for enabling *team collaboration*
while each developer still runs code locally:

- Shared datasets (DVC on DagsHub storage)
- Shared experiment runs / artifacts (MLflow on DagsHub)
- Shared observability dashboards (Grafana Cloud)
- Shared stateful logging (PostgreSQL instead of SQLite)

The intent is to **mimic production** while keeping local dev ergonomics.

---

## Decisions (What we agreed on)

1) **DVC remote = DagsHub Storage (S3‑compatible / DVC remote)**  
   Team members must be able to `dvc pull` the same datasets/models.

2) **Dashboards = Grafana Cloud** (free tier for now)  
   Each dev runs Prometheus locally and forwards metrics via `remote_write`.
   Switching back to self-hosted Grafana should be a config toggle.

3) **Database = PostgreSQL everywhere** (local + production)  
   No rollback to SQLite; SQLite is not suitable for concurrent team usage.

4) **Keep Pushgateway + EvidentlyAI**  
   - Pushgateway stays for *batch* monitoring metrics.  
   - EvidentlyAI stays as a library that computes drift; outputs go to Prometheus/MLflow.

---

## Target Architecture (Collab Dev Mode)

Per developer (local laptop):
- FastAPI service exposes `/metrics`
- Batch monitor pushes batch metrics → local Pushgateway
- Local Prometheus scrapes:
  - FastAPI `/metrics`
  - Pushgateway `/metrics`
- Local Prometheus `remote_write`s to Grafana Cloud (shared)

Shared services (team-wide):
- Grafana Cloud (dashboards + long-term metric store)
- DagsHub (Git + MLflow + DVC storage)
- PostgreSQL (prediction logs + scheduler job store)

---

# Phase 1 — Shared Datasets (DVC → DagsHub Storage)

## Why
Right now `.dvc/config` is empty and DVC cache is local-only, so teammates cannot
`dvc pull` the same data artifacts.

## What to implement
Configure a DVC remote that points to DagsHub Storage so the repo has a shared
source of truth for data and models.

## How (recommended workflow)
In DagsHub UI:
1) Go to repository homepage
2) Click **Remote** → **Data** → **DVC**
3) Copy the provided commands (DagsHub generates repo-specific values)

Example (DagsHub DVC remote via HTTPS basic auth):
```bash
dvc remote add origin https://dagshub.com/<user>/<repo>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <user>
dvc remote modify origin --local password <token>
```

Example (DagsHub S3-compatible bucket as DVC remote):
```bash
dvc remote add origin s3://dvc
dvc remote modify origin endpointurl https://dagshub.com/<user>/<repo>.s3
dvc remote modify origin --local access_key_id <token>
dvc remote modify origin --local secret_access_key <token>
```

Important:
- Use `--local` for credentials so secrets land in `.dvc/config.local` (not committed).
- Add `dvc remote default origin` (or `dvc remote add -d origin ...`) so `dvc pull/push` works without flags.

## Deliverables
- `.dvc/config` contains the remote (no secrets)
- Team onboarding doc snippet: “run the DagsHub Remote → Data → DVC commands”
- Sanity check: teammate can `dvc pull` successfully on a new machine

---

# Phase 2 — Shared State (SQLite → PostgreSQL)

## TL;DR (Simple Terms)
Not a huge problem: it’s a **moderate refactor** where we replace direct `sqlite3`
usage with SQLAlchemy so `PredictionLogger` can write to Postgres.

## Why
SQLite is a single-file DB and is not suitable for collaborative, concurrent writes
(multiple devs/services writing at once). Postgres is the right baseline for “prod-like dev”.

## What to implement

### 2.1 Dependencies
- Add:
  - `SQLAlchemy>=2.0`
  - `psycopg[binary]` (PostgreSQL driver)

### 2.2 Environment configuration
Prefer URL-first, with a host/port fallback for convenience:
```bash
# Preferred: explicit SQLAlchemy URL
PREDICTION_DB_URL=postgresql+psycopg://user:pass@host:5432/dbname

# Optional: host/port pattern (builder constructs URL internally)
DB_HOST=
DB_PORT=
DB_NAME=
DB_USER=
DB_PASSWORD=

# Scheduler job store uses SQLAlchemyJobStore, also Postgres:
SCHEDULER_DB_URL=postgresql+psycopg://user:pass@host:5432/dbname
```

Rules:
- `PREDICTION_DB_URL` required either directly or derivable from `DB_*`.
- No SQLite fallback (Postgres-only across local + prod).

### 2.3 Code changes

**Prediction logging**
- Refactor `src/monitoring/prediction_logger.py`:
  - Use SQLAlchemy engine (`create_engine`) instead of `sqlite3.connect`
  - Create table if missing
  - Insert / select using SQLAlchemy Core

**Scheduler persistence**
- `src/monitoring/scheduler.py` already uses APScheduler `SQLAlchemyJobStore`.
  Ensure `SCHEDULER_DB_URL` points to Postgres (local docker now, Supabase later).

### 2.4 Local Postgres (for dev)
Add a local Postgres service via Docker (either a new compose file or extend existing compose),
so every dev can run:
```bash
docker compose up -d postgres
```
and point `.env` at it. Later, swapping to Supabase = change creds only.

## Deliverables
- `.env.example` documents Postgres env vars (and removes SQLite-oriented vars)
- Prediction logger + scheduler jobstore both use Postgres
- Team onboarding: “start postgres container; run stack”

---

# Phase 3 — Shared Dashboards (Grafana Cloud + Local Prometheus)

## Why
Grafana Cloud is the shared “single pane of glass” for the team. It does **not**
scrape your targets; you still need Prometheus (or Alloy) to scrape and forward.

## What to implement

### 3.1 Prometheus `remote_write` (Option A: you already run Prometheus)
Add a `remote_write` block in Prometheus configuration that points to Grafana Cloud.

**Security requirement**
Do NOT commit the Grafana Cloud API token into git. Use a file and mount it:
- Create a local secret file (gitignored), e.g. `.secrets/grafana_cloud_metrics_token`
- Use `password_file` in Prometheus config

Example:
```yaml
remote_write:
  - url: https://<cloud-hosted-prom-remote-write-endpoint>/api/prom/push
    basic_auth:
      username: <cloud-username>
      password_file: /run/secrets/grafana_cloud_metrics_token
```

### 3.2 Label every developer’s Prometheus stream
Multiple devs will remote_write into the same stack. Without a distinguishing label,
their series will collide and dashboards will be confusing.

Use Prometheus `external_labels` to add identity, e.g.:
```yaml
global:
  external_labels:
    env: dev
    developer: ${PROMETHEUS_DEVELOPER}
```

Notes:
- `external_labels` are applied to remote_write samples.
- Prometheus supports environment variable references inside `external_labels`.

### 3.3 “Cloud mode” vs “Local mode” toggle
Prometheus config should support two modes:
- **Local mode**: no `remote_write`, dashboards view local Prometheus
- **Cloud mode**: `remote_write` enabled, dashboards are shared in Grafana Cloud

Implementation options:
1) Two config files:
   - `monitoring/prometheus.yml` (local-only)
   - `monitoring/prometheus.cloud.yml` (includes `remote_write` + `external_labels`)
2) One config + provisioning script that renders config from template at container start.

## Deliverables
- A documented flow to obtain remote_write URL/username/token from Grafana Cloud UI
- Prometheus config changes + docker-compose mounts for secret file
- Docs: how to switch between local vs cloud dashboards

---

# Phase 4 — Keep Pushgateway + EvidentlyAI (No Cloud Replacements Needed)

## Pushgateway
Keep using Pushgateway for batch job metrics:
- Batch monitor pushes → Pushgateway
- Prometheus scrapes Pushgateway
- Prometheus remote_write forwards to Grafana Cloud

## EvidentlyAI
EvidentlyAI remains the drift computation library:
- Computes drift metrics in the batch job
- Drift outputs are published to Prometheus/MLflow (shared systems)

---

# Implementation Checklist (What the agent builder must change)

## Data (DVC)
- [ ] Configure DVC remote in `.dvc/config` (no secrets)
- [ ] Document DagsHub “Remote → Data → DVC” onboarding commands

## Database (Postgres-only)
- [ ] Add `SQLAlchemy>=2.0` and `psycopg[binary]`
- [ ] Refactor `src/monitoring/prediction_logger.py` to use Postgres
- [ ] Ensure APScheduler jobstore uses `SCHEDULER_DB_URL` (Postgres)
- [ ] Add/extend docker-compose with a Postgres service for dev
- [ ] Update `.env.example` with Postgres vars (URL + DB_* pattern)

## Monitoring (Grafana Cloud)
- [ ] Add `remote_write` config pointing to Grafana Cloud
- [ ] Use `password_file` + mounted secret for the token
- [ ] Add `external_labels` with `${PROMETHEUS_DEVELOPER}`
- [ ] Document local mode vs cloud mode switching

---

# References
- DagsHub Storage / DVC remote examples:
  - https://dagshub.com/docs/feature_guide/dagshub_storage/
  - https://dagshub.com/docs/integration_guide/dvc/
- DVC remote storage concepts:
  - https://dvc.org/doc/user-guide/data-management/remote-storage
- SQLAlchemy 2.0 engine + transaction patterns:
  - https://docs.sqlalchemy.org/en/20/core/engines.html
  - https://docs.sqlalchemy.org/en/20/core/connections.html
- Prometheus configuration (`password_file`, `external_labels`, `remote_write`):
  - https://prometheus.io/docs/prometheus/latest/configuration/configuration/
- Prometheus Pushgateway best practices:
  - https://prometheus.io/docs/practices/pushing/
- Grafana Cloud: sending Prometheus metrics:
  - https://grafana.com/docs/grafana-cloud/send-data/metrics/metrics-prometheus/
