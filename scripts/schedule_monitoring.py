"""
Batch monitoring daemon using APScheduler BackgroundScheduler.

Starts a long-running process that triggers batch monitoring on a cron
schedule (default: daily at 02:00 UTC).  The scheduler is backed by
SQLite for persistence so scheduled jobs survive process restarts.

Configuration via environment variables (or .env file):
    MONITORING_MODEL_NAME          Required. Registered model name.
    MONITORING_TASK_TYPE           regression | classification  (default: regression)
    MONITORING_REFERENCE_DATA_PATH Required. Path to reference CSV.
    MONITORING_FEATURE_COLUMNS     Required. Comma-separated feature names.
    MONITORING_TARGET_COLUMN       Optional. Target column name.
    MONITORING_LOOKBACK_HOURS      Hours of predictions to analyse (default: 24)
    BATCH_MONITOR_CRON_HOUR        Hour of day for cron trigger, UTC (default: 2)
    BATCH_MONITOR_CRON_MINUTE      Minute for cron trigger, UTC (default: 0)
    SCHEDULER_DB_URL               SQLAlchemy URL for job persistence
                                   (default: resolved from PREDICTION_DB_URL / DB_* env vars)
    PROMETHEUS_PUSHGATEWAY_URL     PushGateway URL (default: http://localhost:9091)

Cron hook (one-shot, for system cron / K8s CronJob):
    python -m src.monitoring.batch_monitor \\
        --model-name <NAME> --task-type regression \\
        --reference-data-path <PATH> --feature-columns <COLS>

Daemon usage:
    python scripts/schedule_monitoring.py
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading

from src.monitoring.batch_monitor import BatchMonitor
from src.monitoring.scheduler import start_monitoring_scheduler
from src.utils.helpers import load_env

# Load .env before importing anything that reads env vars
load_env()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("schedule_monitoring")


def _require_env(name: str) -> str:
    """Return env var value or exit with a clear error message."""
    val = os.getenv(name, "").strip()
    if not val:
        logger.error(
            f"Required environment variable '{name}' is not set. "
            "Set it in .env or export it before starting the scheduler."
        )
        sys.exit(1)
    return val


def _build_monitor() -> BatchMonitor:
    """Construct BatchMonitor from environment variables."""
    model_name = _require_env("MONITORING_MODEL_NAME")
    reference_data_path = _require_env("MONITORING_REFERENCE_DATA_PATH")
    feature_columns_env = _require_env("MONITORING_FEATURE_COLUMNS")

    task_type = os.getenv("MONITORING_TASK_TYPE", "regression")
    target_column = os.getenv("MONITORING_TARGET_COLUMN") or None
    features = [f.strip() for f in feature_columns_env.split(",") if f.strip()]

    return BatchMonitor(
        model_name=model_name,
        task_type=task_type,
        reference_data_path=reference_data_path,
        feature_columns=features,
        target_column=target_column,
    )


def _run_monitoring_job() -> None:
    """Wrapper that reads MONITORING_LOOKBACK_HOURS and calls monitor.run()."""
    hours = int(os.getenv("MONITORING_LOOKBACK_HOURS", "24"))
    try:
        monitor = _build_monitor()
        results = monitor.run(hours=hours)
        logger.info(f"Batch monitoring completed: status={results.get('status')}")
    except Exception as exc:
        logger.error(f"Batch monitoring job raised an exception: {exc}", exc_info=True)


def main() -> None:
    """Entry point: validate config, run once at startup, then start the daemon."""
    logger.info("═" * 60)
    logger.info("  MLOps Batch Monitoring Daemon")
    logger.info("═" * 60)

    # Validate required env vars before starting the scheduler
    _build_monitor()  # will sys.exit if misconfigured

    cron_hour = int(os.getenv("BATCH_MONITOR_CRON_HOUR", "2"))
    cron_minute = int(os.getenv("BATCH_MONITOR_CRON_MINUTE", "0"))
    scheduler_db_url = os.getenv("SCHEDULER_DB_URL", "").strip()
    if not scheduler_db_url:
        from src.monitoring.database import get_db_url
        scheduler_db_url = get_db_url()

    logger.info(f"Schedule   : daily at {cron_hour:02d}:{cron_minute:02d} UTC")
    logger.info(f"Job store  : {scheduler_db_url}")
    logger.info(f"Model      : {os.getenv('MONITORING_MODEL_NAME')}")
    logger.info(
        f"Features   : {os.getenv('MONITORING_FEATURE_COLUMNS', '').split(',')}"
    )

    # Run once immediately at startup so ops can verify it works
    logger.info("Running batch monitoring once at startup…")
    _run_monitoring_job()

    # Start the proper BackgroundScheduler daemon
    scheduler = start_monitoring_scheduler(
        run_fn=_run_monitoring_job,
        cron_hour=cron_hour,
        cron_minute=cron_minute,
        scheduler_db_url=scheduler_db_url,
    )

    # ── Block main thread cleanly ──
    # Use an Event so SIGTERM / SIGINT trigger a clean wakeup.
    stop_event = threading.Event()

    def _handle_signal(signum, frame):
        logger.info(f"Received signal {signum} — shutting down…")
        stop_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    logger.info("Daemon running. Press Ctrl+C or send SIGTERM to stop.")
    stop_event.wait()  # blocks until signal received

    logger.info("Monitoring scheduler stopped.")


if __name__ == "__main__":
    main()
