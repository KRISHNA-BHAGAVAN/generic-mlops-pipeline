"""
Training pipeline CLI.

The main entry point for running ML experiments. Orchestrates the full
workflow: load config → validate → load data → preprocess → split →
train → evaluate → log to MLflow.

Usage:
    python -m src.pipelines.train_pipeline --config configs/regression/construction_duration_v1.yaml
    python -m src.pipelines.train_pipeline --config configs/regression/construction_duration_v1.yaml --dry-run
"""

from __future__ import annotations

import sys
import time

import click
import mlflow

from src.config.load_config import load_config
from src.config.validate_config import validate_config_against_data
from src.data.load_data import load_dataset
from src.data.validate import validate_dataset
from src.features.build_features import prepare_features, split_data
from src.models.evaluate import evaluate_model, generate_evaluation_plots
from src.models.registry import (
    log_dataset_input,
    log_experiment_run,
    log_reference_predictions_artifact,
    setup_mlflow,
)
from src.models.train import train_model
from src.pipelines.exceptions import ConfigError, DatasetError, MLOpsException, ModelError
from src.utils.logger import get_logger, setup_root_logger

logger = get_logger(__name__)


def _parse_key_value_pairs(pairs: tuple[str, ...]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.UsageError(
                "Registry tags must be provided as key=value pairs."
            )
        key, value = pair.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to experiment config YAML file",
)
@click.option(
    "--mlflow-tracking-uri",
    type=str,
    default=None,
    help="MLflow tracking URI (overrides env var)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate config and data without running the experiment",
)
@click.option(
    "--register",
    is_flag=True,
    help="Auto-register the trained model if experiment succeeds",
)
@click.option(
    "--register-description",
    default=None,
    help="Optional description for the registered model version",
)
@click.option(
    "--register-alias",
    type=click.Choice(["champion", "candidate", "staging", "production"]),
    default=None,
    help="Optional alias to assign after registration",
)
@click.option(
    "--register-tag",
    "register_tags",
    multiple=True,
    help="Optional registry tag in key=value format (can be repeated)",
)
@click.option(
    "--register-created-by",
    default=None,
    help="Optional created_by metadata for the registered model version",
)
def main(
    config_path: str,
    mlflow_tracking_uri: str | None,
    dry_run: bool,
    register: bool,
    register_description: str | None,
    register_alias: str | None,
    register_tags: tuple[str, ...],
    register_created_by: str | None,
):
    """
    Run a machine learning experiment.

    Executes the full pipeline: config validation → data loading →
    preprocessing → training → evaluation → MLflow logging.

    Examples:
        # Run regression experiment
        python -m src.pipelines.train_pipeline \\
            --config configs/regression/construction_duration_v1.yaml

        # Validate only (no training)
        python -m src.pipelines.train_pipeline \\
            --config configs/regression/construction_duration_v1.yaml --dry-run

        # Run and auto-register
        python -m src.pipelines.train_pipeline \\
            --config configs/classification/construction_risk_v1.yaml --register
    """
    setup_root_logger()
    pipeline_start = time.time()

    try:
        # ── Step 1: Load and validate config ──
        click.echo("━" * 60)
        click.echo("🔧 Step 1: Loading config...")
        exp_config = load_config(config_path)
        click.echo(
            f"   ✓ Config loaded: {exp_config.experiment_name} "
            f"(task={exp_config.task_type}, model={exp_config.model_type})"
        )

        # ── Step 2: Load dataset ──
        click.echo("\n📊 Step 2: Loading dataset...")
        df = load_dataset(exp_config.dataset_source)
        click.echo(f"   ✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

        # ── Step 3: Validate config against data ──
        click.echo("\n✅ Step 3: Validating config against data...")
        quality_report = validate_dataset(df, exp_config)
        click.echo(
            f"   ✓ Validation passed. "
            f"Nulls: {quality_report['null_feature_count']}, "
            f"Duplicates: {quality_report['duplicate_count']}"
        )

        if dry_run:
            click.echo("━" * 60)
            click.echo("✓ Dry run complete — config and data are valid.")
            click.echo(f"  Target stats: {quality_report['target_stats']}")
            return

        # ── Step 4: Prepare features ──
        click.echo("\n🔬 Step 4: Preparing features...")
        X, y, preprocessing_artifacts = prepare_features(df, exp_config)
        click.echo(f"   ✓ Features prepared: {X.shape[1]} features")

        # ── Step 5: Split data ──
        click.echo("\n✂️  Step 5: Splitting data...")
        X_train, X_test, y_train, y_test = split_data(X, y, exp_config)
        click.echo(f"   ✓ Train: {len(X_train)}, Test: {len(X_test)}")

        # ── Step 6: Setup MLflow ──
        click.echo("\n📡 Step 6: Setting up MLflow...")
        setup_mlflow(
            tracking_uri=mlflow_tracking_uri,
            experiment_name=exp_config.experiment_name,
            experiment_description=exp_config.experiment_description,
            experiment_tags=exp_config.experiment_tags,
        )
        click.echo("   ✓ MLflow configured")

        # ── Step 7: Train model ──
        click.echo(f"\n🏋️  Step 7: Training {exp_config.model_type}...")
        mlflow.start_run(
            run_name=exp_config.run_name or exp_config.experiment_name,
            tags=exp_config.mlflow_tags or None,
            description=exp_config.run_description,
        )

        # ── Step 7b: Log dataset lineage ──
        log_dataset_input(exp_config, df)

        model, metadata = train_model(X_train, y_train, exp_config)
        click.echo(
            f"   ✓ Model trained in {metadata['training_duration_seconds']:.2f}s"
        )

        # ── Step 8: Evaluate ──
        click.echo("\n📈 Step 8: Evaluating model...")
        metrics = evaluate_model(model, X_test, y_test, exp_config)
        for name, value in metrics.items():
            click.echo(f"   • {name}: {value:.4f}")

        # ── Step 9: Generate plots ──
        click.echo("\n🎨 Step 9: Generating evaluation plots...")
        y_pred = model.predict(X_test)
        plots = generate_evaluation_plots(model, X_test, y_test, y_pred, exp_config)
        click.echo(f"   ✓ Generated {len(plots)} plots")

        # ── Step 10: Log to MLflow ──
        click.echo("\n💾 Step 10: Logging to MLflow...")
        run_id, model_uri = log_experiment_run(exp_config, model, metadata, metrics, plots)
        click.echo(f"   ✓ MLflow run ID: {run_id}")

        # ── Step 10b: Persist reference predictions artifact ──
        # Saves model.predict(X_train) alongside features so batch_monitor
        # can use it as the TRUE prediction drift baseline (not ground-truth labels).
        click.echo("\n📌 Step 10b: Logging reference predictions artifact...")
        log_reference_predictions_artifact(model, X_train)
        click.echo("   ✓ Reference predictions artifact logged to MLflow")

        # ── Step 11: Optional registration ──
        if register and exp_config.registry_name:
            click.echo("📦 Step 11: Registering model...")
            from src.models.registry import register_model, promote_model

            model_version = register_model(
                model_uri,
                exp_config.registry_name,
                description=register_description or exp_config.registry_description,
                version_tags=(
                    _parse_key_value_pairs(register_tags)
                    if register_tags
                    else exp_config.registry_tags or None
                ),
                registered_model_tags=(
                    _parse_key_value_pairs(register_tags)
                    if register_tags
                    else exp_config.registry_tags or None
                ),
                created_by=register_created_by or exp_config.registry_created_by or exp_config.user,
            )
            click.echo(
                f"   ✓ Registered: {exp_config.registry_name} "
                f"v{model_version.version}"
            )

            alias = register_alias or exp_config.registry_alias
            if alias:
                promote_model(exp_config.registry_name, model_version.version, alias)
                click.echo(
                    f"   ✓ Promoted: {exp_config.registry_name} "
                    f"v{model_version.version} → '{alias}'"
                )

        mlflow.end_run()

        # ── Summary ──
        pipeline_duration = time.time() - pipeline_start
        click.echo("━" * 60)
        click.echo(f"✅ Pipeline completed in {pipeline_duration:.1f}s")
        click.echo(f"   Run ID:     {run_id}")
        click.echo(f"   Experiment: {exp_config.experiment_name}")
        click.echo(f"   Model:      {exp_config.model_type}")
        click.echo("━" * 60)

    except (ConfigError, DatasetError, ModelError) as e:
        click.echo(f"\n✗ Pipeline failed: {e}", err=True)
        if mlflow.active_run():
            mlflow.log_param("error", str(e)[:250])
            mlflow.end_run(status="FAILED")
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n✗ Unexpected error: {e}", err=True)
        if mlflow.active_run():
            mlflow.log_param("error", str(e)[:250])
            mlflow.end_run(status="FAILED")
        raise

if __name__ == "__main__":
    main()