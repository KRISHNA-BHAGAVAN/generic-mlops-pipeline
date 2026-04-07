"""
Run ranking and selection module.

Queries MLflow for top-performing runs within an experiment,
sorted by a specified metric. Used to identify the best model
candidates for registration and promotion.
"""

from __future__ import annotations

from typing import Any, Dict, List

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Metrics where lower is better
LOWER_IS_BETTER = {"mse", "rmse", "mae"}


def rank_runs(
    experiment_name: str,
    metric: str,
    task_type: str | None = None,
    top_k: int = 5,
    client: MlflowClient | None = None,
) -> List[Dict[str, Any]]:
    """
    Find and rank top-performing runs by metric.

    Args:
        experiment_name: MLflow experiment name.
        metric: Metric to rank by (e.g. "r2", "f1", "mse").
        task_type: Optional task type filter.
        top_k: Return top K runs.
        client: MLflowClient instance.

    Returns:
        List of run dictionaries sorted by metric (best first).
    """
    if client is None:
        client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        logger.warning(f"Experiment '{experiment_name}' not found")
        return []

    # Build filter string
    filter_parts = []
    if task_type:
        filter_parts.append(f"tags.task_type = '{task_type}'")

    filter_string = " AND ".join(filter_parts) if filter_parts else ""

    # Determine sort order
    order = "ASC" if metric in LOWER_IS_BETTER else "DESC"

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        max_results=top_k * 2,  # Fetch extra in case of failed runs
        order_by=[f"metrics.{metric} {order}"],
    )

    result = []
    for run in runs:
        metric_value = run.data.metrics.get(metric)
        if metric_value is None:
            continue

        result.append({
            "run_id": run.info.run_id,
            "experiment_name": experiment.name,
            "run_name": run.info.run_name,
            "metric": metric,
            "value": metric_value,
            "status": run.info.status,
            "user": run.data.tags.get("user", "unknown"),
            "model_type": run.data.params.get("model_type", "unknown"),
            "task_type": run.data.tags.get("task_type", "unknown"),
            "validation_status": run.data.tags.get("validation_status", "unknown"),
            "all_metrics": dict(run.data.metrics),
        })

        if len(result) >= top_k:
            break

    logger.info(
        f"Found {len(result)} runs for experiment='{experiment_name}', "
        f"metric='{metric}' (sorted {order})"
    )
    return result


def get_best_run(
    experiment_name: str,
    metric: str,
    task_type: str | None = None,
    client: MlflowClient | None = None,
) -> Dict[str, Any] | None:
    """
    Get the single best run for an experiment by metric.

    Args:
        experiment_name: MLflow experiment name.
        metric: Metric to optimize.
        task_type: Optional task type filter.
        client: MLflowClient instance.

    Returns:
        Best run dictionary, or None if no runs found.
    """
    runs = rank_runs(experiment_name, metric, task_type, top_k=1, client=client)
    return runs[0] if runs else None


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--experiment-name", required=True, help="MLflow experiment name")
    @click.option("--metric", required=True, help="Metric to rank by (e.g. mse, r2)")
    @click.option("--task-type", default=None, help="Optional task type filter")
    @click.option("--top-k", default=5, type=int, help="Number of runs to return")
    @click.option("--auto-promote", is_flag=True, help="Auto promote the best run to candidate")
    @click.option("--model-name", default=None, help="Registered model name (required if auto-promote)")
    def main(experiment_name, metric, task_type, top_k, auto_promote, model_name):
        """Rank MLflow runs and optionally auto-promote the best one."""
        runs = rank_runs(experiment_name, metric, task_type, top_k)
        
        if not runs:
            logger.warning(f"No runs found for experiment '{experiment_name}'")
            return
            
        print(f"Top {len(runs)} runs for metric '{metric}':")
        for i, run in enumerate(runs):
            print(f"{i+1}. Run {run['run_id']} ({run['run_name']}) | {metric}: {run['value']:.4f} | status: {run['status']}")
            
        if auto_promote:
            if not model_name:
                logger.error("--model-name is required when using --auto-promote")
                return
                
            best_run = runs[0]
            print(f"\nAuto-promoting best run {best_run['run_id']} to model '{model_name}' as candidate...")
            from src.selection.promote_model import promote_model
            from src.models.registry import setup_mlflow
            
            # promote_model registers and aliases the model
            setup_mlflow(experiment_name)
            promote_model(
                run_id=best_run['run_id'],
                model_name=model_name,
                alias="candidate"
            )
            print("Promotion complete.")
            
    main()
