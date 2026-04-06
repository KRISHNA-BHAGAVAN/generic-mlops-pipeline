"""
Generic MLOps Pipeline — Main Entry Point

This module delegates to the appropriate sub-command based on usage.
For the full CLI, use the specific pipeline modules:

    # Training
    python -m src.pipelines.train_pipeline --config <config.yaml>

    # Register a model
    python -m src.pipelines.register_pipeline --run-id <id> --model-name <name>

    # Promote a model
    python -m src.selection.promote_model --model-name <name> --version <v> --alias champion

    # Start inference service
    uvicorn deployment.app:app --host 0.0.0.0 --port 8000
"""

import sys
import click

from src.utils.logger import setup_root_logger


@click.group()
def cli():
    """Generic MLOps Pipeline CLI."""
    setup_root_logger()


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), required=True)
@click.option("--dry-run", is_flag=True)
@click.option("--register", is_flag=True)
@click.option("--mlflow-tracking-uri", default=None)
def train(config_path, dry_run, register, mlflow_tracking_uri):
    """Run a training experiment."""
    from src.pipelines.train_pipeline import main as train_main
    # Build args for the Click command
    args = ["--config", config_path]
    if dry_run:
        args.append("--dry-run")
    if register:
        args.append("--register")
    if mlflow_tracking_uri:
        args.extend(["--mlflow-tracking-uri", mlflow_tracking_uri])
    train_main(args, standalone_mode=False)


@cli.command()
def serve():
    """Start the FastAPI inference service."""
    import uvicorn
    uvicorn.run("deployment.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    cli()
