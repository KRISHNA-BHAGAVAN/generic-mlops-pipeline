"""
MLflow model registry and experiment logging module.

Handles all interactions with MLflow: logging experiments, registering
models, promoting versions, and setting validation statuses.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config.validate_config import ExperimentConfig
from src.pipelines.exceptions import RegistryError
from src.utils.helpers import get_mlflow_tracking_uri, load_env, safe_dict_value, utc_now_iso
from src.utils.logger import get_logger

logger = get_logger(__name__)


def setup_mlflow(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    """
    Configure MLflow tracking URI and experiment.

    Uses DagsHub credentials from environment if tracking_uri is not provided.

    Args:
        tracking_uri: MLflow tracking URI. Reads from env if None.
        experiment_name: MLflow experiment name. Reads from env if None.
    """
    load_env()

    uri = tracking_uri or get_mlflow_tracking_uri()
    mlflow.set_tracking_uri(uri)

    # Set DagsHub credentials if available
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if username and password:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    if experiment_name:
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            # Handle the case where the experiment exists but is soft-deleted
            if "deleted experiment" in str(e).lower():
                from mlflow.tracking.client import MlflowClient
                client = MlflowClient()
                exp = client.get_experiment_by_name(experiment_name)
                
                if exp and exp.lifecycle_stage == "deleted":
                    try:
                        logger.info(f"Attempting to restore deleted MLflow experiment: {experiment_name}")
                        client.restore_experiment(exp.experiment_id)
                        mlflow.set_experiment(experiment_name)
                        logger.info(f"Successfully restored and set experiment: {experiment_name}")
                    except Exception as restore_err:
                        import time
                        fallback_name = f"{experiment_name}_v{int(time.time())}"
                        logger.warning(
                            f"Could not restore experiment '{experiment_name}': {restore_err}. "
                            f"Creating fallback experiment: {fallback_name}"
                        )
                        mlflow.set_experiment(fallback_name)
                else:
                    raise
            else:
                raise

    logger.info(f"MLflow configured: URI={uri}")


def log_experiment_run(
    config: ExperimentConfig,
    model,
    metadata: Dict[str, Any],
    metrics: Dict[str, float],
    evaluation_plots: Dict[str, str],
) -> tuple[str, str]:
    """
    Log a complete experiment run to MLflow.

    Logs parameters, metrics, tags, model artifacts, evaluation plots,
    and the config file itself. Must be called within an active MLflow run.

    Args:
        config: ExperimentConfig object.
        model: Trained sklearn model.
        metadata: Metadata dict from the train function.
        metrics: Dict of metric_name -> value.
        evaluation_plots: Dict of artifact_name -> file_path.

    Returns:
        Tuple of (run_id, model_uri) — the run's ID and the logged model URI
        that can be passed directly to mlflow.register_model().
    """
    run = mlflow.active_run()
    if run is None:
        raise RegistryError("No active MLflow run. Call mlflow.start_run() first.")

    run_id = run.info.run_id
    logger.info(f"Logging experiment run: {run_id}")

    # ── Log configuration parameters ──
    mlflow.log_param("task_type", config.task_type)
    mlflow.log_param("model_type", config.model_type)
    mlflow.log_param("dataset_source", config.dataset_source)
    mlflow.log_param("target_column", config.target_column)
    mlflow.log_param("n_features", len(config.feature_columns))
    mlflow.log_param("feature_columns", str(config.feature_columns))
    mlflow.log_param("test_size", config.test_size)
    mlflow.log_param("split_strategy", config.split_strategy)
    mlflow.log_param("random_state", config.random_state)

    # Log model hyperparameters
    for key, value in config.model_params.items():
        safe_val = safe_dict_value(value)
        if safe_val is not None:
            mlflow.log_param(f"model_param_{key}", safe_val)

    # ── Log evaluation metrics ──
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, (int, float)):
            mlflow.log_metric(metric_name, metric_value)

    # Log training duration
    training_duration = metadata.get("training_duration_seconds", 0)
    mlflow.log_metric("training_duration_seconds", training_duration)
    mlflow.log_metric("training_samples", metadata.get("training_samples", 0))

    # ── Log tags ──
    mlflow.set_tag("user", config.user)
    mlflow.set_tag("experiment_name", config.experiment_name)
    mlflow.set_tag("task_type", config.task_type)
    mlflow.set_tag("validation_status", "pending")
    mlflow.set_tag("run_datetime", utc_now_iso())

    if config.dvc_version:
        mlflow.set_tag("dvc_version", config.dvc_version)

    for key, value in config.mlflow_tags.items():
        mlflow.set_tag(f"custom_{key}", value)

    # ── Log artifacts ──

    # Log evaluation plots
    for artifact_name, file_path in evaluation_plots.items():
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path, artifact_path="evaluation_plots")

    # Log config as artifact
    _log_config_artifact(config)

    # ── Log model with signature ──
    X_sample = metadata.get("X_sample")
    y_pred_sample = metadata.get("y_pred_sample")

    signature = None
    input_example = None
    if X_sample is not None and y_pred_sample is not None:
        try:
            signature = mlflow.models.infer_signature(
                model_input=X_sample,
                model_output=y_pred_sample,
            )
            input_example = X_sample.head(1)
        except Exception as e:
            logger.warning(f"Could not infer model signature: {e}")

    logged_model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=input_example,
    )

    model_uri = logged_model_info.model_uri
    logger.info(
        f"Run {run_id} logged: "
        f"{len(metrics)} metrics, {len(evaluation_plots)} plots "
        f"(model_uri={model_uri})"
    )
    return run_id, model_uri


def _log_config_artifact(config: ExperimentConfig) -> None:
    """Save the experiment config as a YAML artifact."""
    import yaml

    tmp_dir = tempfile.mkdtemp(prefix="mlops_config_")
    config_path = os.path.join(tmp_dir, "experiment_config.yaml")

    config_dict = config.model_dump()
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    mlflow.log_artifact(config_path, artifact_path="config")
    shutil.rmtree(tmp_dir, ignore_errors=True)


def log_reference_predictions_artifact(
    model,
    X_ref,
    artifact_dir: str = "reference_data",
    filename: str = "reference_predictions.parquet",
) -> None:
    """
    Persist true reference predictions as an MLflow artifact.

    Runs model.predict(X_ref) to produce predictions on the training/reference
    dataset and saves a parquet file containing all feature columns plus a
    ``prediction`` column. This file is the ground-truth baseline used by the
    batch monitor for *prediction drift* detection (i.e. comparing how the
    model's output distribution has shifted, not comparing against labels).

    Must be called within an active MLflow run.

    Args:
        model: Trained sklearn-compatible estimator.
        X_ref: Reference feature DataFrame (typically X_train or a held-out
               validation set used during training).
        artifact_dir: MLflow artifact sub-directory to store the file in.
        filename: Parquet filename.
    """
    import pandas as pd

    run = mlflow.active_run()
    if run is None:
        logger.warning(
            "log_reference_predictions_artifact called with no active MLflow run — skipping."
        )
        return

    try:
        ref_preds = model.predict(X_ref)
        ref_df = X_ref.copy().reset_index(drop=True)
        ref_df["prediction"] = ref_preds

        tmp_dir = tempfile.mkdtemp(prefix="mlops_refpreds_")
        out_path = os.path.join(tmp_dir, filename)
        ref_df.to_parquet(out_path, index=False)

        mlflow.log_artifact(out_path, artifact_path=artifact_dir)
        logger.info(
            f"Reference predictions artifact logged: "
            f"{artifact_dir}/{filename} ({len(ref_df)} rows, run={run.info.run_id})"
        )

    except Exception as e:
        logger.warning(f"Could not log reference predictions artifact: {e}")
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def resolve_model_uri_from_run(run_id: str) -> str:
    """
    Discover the correct model URI for a given MLflow run ID.

    In MLflow 3.x, log_model() returns a models:/m-<hash> URI that
    may differ from the legacy runs:/<run_id>/<path> pattern. This
    function queries the MLflow backend to find the actual URI.

    Strategy:
        1. Try search_logged_models() filtered by run_id
        2. Fall back to artifact listing to find model paths
        3. Last resort: construct legacy runs:/<run_id>/model URI

    Args:
        run_id: MLflow run ID.

    Returns:
        The model URI string.

    Raises:
        RegistryError: If no model can be found for the run.
    """
    client = MlflowClient()

    # Strategy 1: Use search_logged_models (MLflow 3.x API)
    try:
        run = client.get_run(run_id)
        experiment_id = run.info.experiment_id

        results = mlflow.search_logged_models(
            experiment_ids=[experiment_id],
            output_format="list",
        )

        # Filter results to our specific run_id
        for model_info in results:
            if getattr(model_info, "source_run_id", None) == run_id:
                model_uri = model_info.model_uri
                logger.info(f"Resolved model URI via search_logged_models: {model_uri}")
                return model_uri

    except Exception as e:
        logger.debug(f"search_logged_models not available or failed: {e}")

    # Strategy 2: List artifacts to find model directory
    try:
        artifacts = client.list_artifacts(run_id)
        model_artifact_names = []
        for artifact in artifacts:
            # Model artifacts contain MLmodel file
            if artifact.is_dir:
                sub_artifacts = client.list_artifacts(run_id, artifact.path)
                sub_names = [a.path.split("/")[-1] for a in sub_artifacts]
                if "MLmodel" in sub_names:
                    model_artifact_names.append(artifact.path)

        if model_artifact_names:
            # Prefer "model" if it exists, otherwise take the first one
            chosen = "model" if "model" in model_artifact_names else model_artifact_names[0]
            model_uri = f"runs:/{run_id}/{chosen}"
            logger.info(f"Resolved model URI via artifact listing: {model_uri}")
            return model_uri

    except Exception as e:
        logger.debug(f"Artifact listing failed: {e}")

    # Strategy 3: Legacy fallback
    model_uri = f"runs:/{run_id}/model"
    logger.warning(
        f"Could not discover model URI for run {run_id}. "
        f"Falling back to legacy URI: {model_uri}"
    )
    return model_uri


def register_model(
    model_uri: str,
    model_name: str,
    client: MlflowClient | None = None,
) -> Any:
    """
    Register a model from a successful run in the MLflow registry.

    Args:
        model_uri: Model URI returned by mlflow.sklearn.log_model()
                   (e.g. 'runs:/<run_id>/model' or a models:/ URI).
        model_name: Name for the registered model.
        client: MLflowClient instance (created if None).

    Returns:
        ModelVersion object.

    Raises:
        RegistryError: If registration fails.
    """
    if client is None:
        client = MlflowClient()

    try:
        model_version = mlflow.register_model(model_uri, name=model_name)
    except Exception as e:
        raise RegistryError(f"Failed to register model: {e}") from e

    # Tag as pending validation
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="validation_status",
        value="pending",
    )

    logger.info(
        f"Model registered: {model_name} v{model_version.version} "
        f"(uri={model_uri})"
    )
    return model_version


def promote_model(
    model_name: str,
    version: int | str,
    alias: str,
    client: MlflowClient | None = None,
) -> None:
    """
    Assign an alias to a model version (e.g. "champion", "candidate").

    Args:
        model_name: Registered model name.
        version: Model version number.
        alias: Alias to assign.
        client: MLflowClient instance.

    Raises:
        RegistryError: If promotion fails.
    """
    if client is None:
        client = MlflowClient()

    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(version),
        )
    except Exception as e:
        raise RegistryError(f"Failed to promote model: {e}") from e

    logger.info(f"Model {model_name} v{version} promoted to alias '{alias}'")


def approve_model(
    model_name: str,
    version: int | str,
    client: MlflowClient | None = None,
) -> None:
    """
    Mark a model version as approved for production.

    Args:
        model_name: Registered model name.
        version: Model version number.
        client: MLflowClient instance.
    """
    if client is None:
        client = MlflowClient()

    client.set_model_version_tag(
        name=model_name,
        version=str(version),
        key="validation_status",
        value="passed",
    )
    client.set_model_version_tag(
        name=model_name,
        version=str(version),
        key="approved_at",
        value=utc_now_iso(),
    )

    logger.info(f"Model {model_name} v{version} approved")
