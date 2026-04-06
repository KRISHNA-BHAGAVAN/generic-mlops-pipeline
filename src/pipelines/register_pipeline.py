"""
Model registration pipeline CLI.

Register a trained model from a completed MLflow run into the
model registry, and optionally promote it to a specific alias.

Usage:
    # Using model URI (recommended, from log_experiment_run output):
    python -m src.pipelines.register_pipeline \\
        --model-uri "runs:/<run_id>/model" --model-name construction_duration

    # Using run ID (legacy, constructs URI as runs:/<run_id>/model):
    python -m src.pipelines.register_pipeline \\
        --run-id <run_id> --model-name construction_duration
"""

from __future__ import annotations

import click

from src.models.registry import (
    approve_model,
    promote_model,
    register_model,
    resolve_model_uri_from_run,
    setup_mlflow,
)
from src.utils.logger import get_logger, setup_root_logger

logger = get_logger(__name__)


@click.command()
@click.option("--model-uri", default=None, help="Model URI from log_model (preferred)")
@click.option("--run-id", default=None, help="MLflow run ID (auto-discovers model URI)")
@click.option("--model-name", required=True, help="Name for the registered model")
@click.option(
    "--alias",
    default=None,
    type=click.Choice(["champion", "candidate", "staging"]),
    help="Optional alias to assign",
)
@click.option("--approve/--no-approve", default=False, help="Mark as approved")
@click.option("--mlflow-tracking-uri", default=None, help="MLflow tracking URI")
def main(
    model_uri: str | None,
    run_id: str | None,
    model_name: str,
    alias: str | None,
    approve: bool,
    mlflow_tracking_uri: str | None,
):
    """
    Register a model from a completed MLflow run.

    Example:
        python -m src.pipelines.register_pipeline \\
            --model-uri "models:/m-abc123" \\
            --model-name construction_duration \\
            --alias champion \\
            --approve

    Or using run-id (auto-discovers the model URI):
        python -m src.pipelines.register_pipeline \\
            --run-id abc123 \\
            --model-name construction_duration \\
            --alias champion
    """
    setup_root_logger()

    click.echo("━" * 60)
    click.echo("📦 Registering model...")

    setup_mlflow(tracking_uri=mlflow_tracking_uri)

    # Resolve model URI
    if model_uri is None:
        if run_id is None:
            raise click.UsageError("Either --model-uri or --run-id must be provided.")
        click.echo(f"   🔍 Discovering model URI for run: {run_id}...")
        model_uri = resolve_model_uri_from_run(run_id)
        click.echo(f"   ✓ Resolved model URI: {model_uri}")

    # Register
    model_version = register_model(model_uri, model_name)
    click.echo(
        f"   ✓ Registered: {model_name} v{model_version.version}"
    )

    # Approve
    if approve:
        approve_model(model_name, model_version.version)
        click.echo(f"   ✓ Approved: {model_name} v{model_version.version}")

    # Promote
    if alias:
        promote_model(model_name, model_version.version, alias)
        click.echo(
            f"   ✓ Promoted: {model_name} v{model_version.version} → '{alias}'"
        )

    click.echo("━" * 60)
    click.echo("✅ Registration complete")


if __name__ == "__main__":
    main()

