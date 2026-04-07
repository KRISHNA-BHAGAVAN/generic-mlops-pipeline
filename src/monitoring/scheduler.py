"""
APScheduler daemon module for batch monitoring.

Provides a production-grade ``BackgroundScheduler`` wired with:

- ``SQLAlchemyJobStore`` for persistence across restarts.
- ``ThreadPoolExecutor`` capped at 1 worker to prevent concurrent runs.
- ``job_defaults``: ``coalesce=True``, ``max_instances=1``,
  ``misfire_grace_time=3600`` (1 hour tolerance for late starts).
- Explicit ``timezone=utc`` (APScheduler 3.x requires this for cron triggers).

Lifecycle
---------
1. Call :func:`start_monitoring_scheduler` to start the background daemon.
2. The caller's main thread should block (e.g. via ``signal.pause()``).
3. The scheduler shuts down cleanly on process exit via ``atexit``.

For system-level cron / K8s CronJob, use the one-shot CLI entrypoint in
``src/monitoring/batch_monitor.py`` instead.
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def create_monitoring_scheduler(
    scheduler_db_url: str | None = None,
):
    """
    Build and return a configured ``BackgroundScheduler``.

    Does **not** start the scheduler.  Call ``.start()`` on the returned
    instance (or use :func:`start_monitoring_scheduler` which handles the
    full lifecycle).

    Args:
        scheduler_db_url: SQLAlchemy-compatible URL for job persistence.
                          Schedules survive process restarts.
                          Resolved from ``SCHEDULER_DB_URL`` or ``PREDICTION_DB_URL``
                          environment variables if not provided.

    Returns:
        Configured (not yet started) ``BackgroundScheduler`` instance.
    """
    import os

    from pytz import utc
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
    from apscheduler.executors.pool import ThreadPoolExecutor

    if scheduler_db_url is None:
        from src.monitoring.database import get_db_url
        scheduler_db_url = os.getenv("SCHEDULER_DB_URL", "").strip() or get_db_url()

    jobstores = {
        "default": SQLAlchemyJobStore(url=scheduler_db_url),
    }

    # Single-worker thread pool: batch monitoring is I/O-heavy and must not
    # run concurrently (prevented also by max_instances=1 below).
    executors = {
        "default": ThreadPoolExecutor(max_workers=1),
    }

    job_defaults = {
        # If multiple fire times accumulated while scheduler was offline,
        # run only once (the 'latest' missed time).
        "coalesce": True,
        # Never allow more than one concurrent instance of the same job.
        "max_instances": 1,
        # Allow the job to start up to 1 hour late before treating it as a
        # misfire.  Useful when the host wakes from sleep or a brief outage.
        "misfire_grace_time": 3600,
    }

    scheduler = BackgroundScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=utc,  # APScheduler 3.x requires an explicit timezone for cron
    )

    logger.info(
        f"BackgroundScheduler created — jobstore: {scheduler_db_url}, timezone: UTC"
    )
    return scheduler


def start_monitoring_scheduler(
    run_fn: Callable,
    cron_hour: int = 2,
    cron_minute: int = 0,
    scheduler_db_url: str | None = None,
):
    """
    Create, configure, and start the monitoring background scheduler.

    Registers the batch monitoring job as a cron trigger (default: daily at
    02:00 UTC) and registers a clean shutdown hook via ``atexit``.

    Args:
        run_fn: The callable to run on schedule (e.g. ``monitor.run``).
        cron_hour: Hour of day to run (0–23, UTC).  Default: 2.
        cron_minute: Minute within hour (0–59, UTC).  Default: 0.
        scheduler_db_url: SQLAlchemy URL for job persistence.
                          Resolved from env if not provided.

    Returns:
        Running ``BackgroundScheduler`` instance.
    """
    scheduler = create_monitoring_scheduler(scheduler_db_url)

    scheduler.add_job(
        func=run_fn,
        trigger="cron",
        hour=cron_hour,
        minute=cron_minute,
        id="batch_monitoring_job",
        # replace_existing avoids DuplicateJobError when the persisted job is
        # re-loaded from the SQLite store on restart.
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        f"Monitoring scheduler started — cron={cron_hour:02d}:{cron_minute:02d} UTC daily"
    )

    # Ensure the background thread is cleaned up on normal process exit.
    atexit.register(lambda: _shutdown_scheduler(scheduler))

    return scheduler


def _shutdown_scheduler(scheduler) -> None:
    """Shut down the scheduler gracefully (called by atexit)."""
    try:
        if scheduler.running:
            scheduler.shutdown(wait=False)
            logger.info("Monitoring scheduler shut down.")
    except Exception as e:
        logger.warning(f"Scheduler shutdown error (ignored): {e}")
