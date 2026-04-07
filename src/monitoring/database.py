"""
Shared database engine factory and table definitions.

Centralises all PostgreSQL connection logic:

- ``get_db_url()`` resolves the connection URL from environment variables.
- ``get_engine()`` creates (and caches) a SQLAlchemy ``Engine``.
- ``predictions_table`` is the SQLAlchemy ``Table`` used by ``PredictionLogger``.
- ``ensure_tables()`` materialises the schema (``CREATE TABLE IF NOT EXISTS``).

The module deliberately uses **SQLAlchemy Core** (not ORM) to keep
dependencies lightweight and to match the existing imperative style.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine as _create_engine,
)
from sqlalchemy.engine import Engine

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Schema ──────────────────────────────────────────────────────────────────

metadata = MetaData()

predictions_table = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("model_name", String(256), nullable=False),
    Column("model_version", String(64), nullable=False),
    Column("features_json", Text, nullable=False),
    Column("prediction", Text, nullable=False),
    Column("prediction_proba", Text, nullable=True),
    Column("timestamp", String(64), nullable=False),
    Column("latency_seconds", Float, nullable=True),
)

# Composite index for fast lookups by model + time
Index("idx_predictions_model", predictions_table.c.model_name, predictions_table.c.timestamp)


# ── URL resolution ──────────────────────────────────────────────────────────

def get_db_url(env_var: str = "PREDICTION_DB_URL") -> str:
    """
    Resolve a database URL from environment variables.

    Priority:
    1. ``env_var`` (default ``PREDICTION_DB_URL``) — explicit SQLAlchemy URL.
    2. ``DB_HOST``, ``DB_PORT``, ``DB_NAME``, ``DB_USER``, ``DB_PASSWORD``
       — used to construct ``postgresql+psycopg://…`` automatically.

    Raises:
        RuntimeError: If neither method yields a usable URL.
    """
    url = os.getenv(env_var, "").strip()
    if url:
        return url

    # Fallback: build from individual DB_* vars
    host = os.getenv("DB_HOST", "").strip()
    port = os.getenv("DB_PORT", "5432").strip()
    name = os.getenv("DB_NAME", "").strip()
    user = os.getenv("DB_USER", "").strip()
    password = os.getenv("DB_PASSWORD", "").strip()

    if host and name and user:
        constructed = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{name}"
        return constructed

    raise RuntimeError(
        f"Database URL not configured. Set '{env_var}' "
        "or provide DB_HOST, DB_NAME, and DB_USER environment variables."
    )


# ── Engine factory ──────────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def get_engine(db_url: Optional[str] = None) -> Engine:
    """
    Create (or return cached) SQLAlchemy ``Engine``.

    Args:
        db_url: Explicit database URL.  When ``None``, resolved via
                :func:`get_db_url`.

    Returns:
        Configured ``Engine`` instance.
    """
    resolved_url = db_url or get_db_url()
    engine = _create_engine(
        resolved_url,
        pool_pre_ping=True,       # check connections on checkout
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    logger.info(f"SQLAlchemy engine created — {engine.url.render_as_string(hide_password=True)}")
    return engine


def ensure_tables(engine: Optional[Engine] = None) -> None:
    """
    Create all tables defined in ``metadata`` if they do not exist.

    Args:
        engine: Engine to use.  Defaults to ``get_engine()``.
    """
    eng = engine or get_engine()
    metadata.create_all(eng)
    logger.info("Database tables ensured (CREATE IF NOT EXISTS)")
