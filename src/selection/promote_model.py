"""
Model promotion CLI/functions.

Provides both programmatic and command-line interfaces for
promoting a model version to an alias (champion, candidate, etc.).
"""

from __future__ import annotations

import click
from mlflow.tracking import MlflowClient

from src.models.registry import approve_model, promote_model, setup_mlflow
from src.utils.logger import get_logger

logger = get_logger(__name__)


@click.command("promote")
@click.option("--model-name", required=True, help="Registered model name")
@click.option("--version", required=True, type=int, help="Model version number")
@click.option(
    "--alias",
    required=True,
    type=click.Choice(["champion", "candidate", "staging", "production"]),
    help="Alias to assign",
)
@click.option("--approve/--no-approve", default=False, help="Also mark as approved")
@click.option("--mlflow-tracking-uri", default=None, help="MLflow tracking URI")
def promote_cli(
    model_name: str,
    version: int,
    alias: str,
    approve: bool,
    mlflow_tracking_uri: str | None,
):
    """
    Promote a model version to an alias.

    Example:
        python -m src.selection.promote_model \\
            --model-name construction_duration \\
            --version 3 \\
            --alias champion \\
            --approve
    """
    setup_mlflow(tracking_uri=mlflow_tracking_uri)
    client = MlflowClient()

    if approve:
        approve_model(model_name, version, client)
        click.echo(f"✓ Model {model_name} v{version} approved")

    promote_model(model_name, version, alias, client)
    click.echo(f"✓ Model {model_name} v{version} promoted to '{alias}'")


if __name__ == "__main__":
    promote_cli()
