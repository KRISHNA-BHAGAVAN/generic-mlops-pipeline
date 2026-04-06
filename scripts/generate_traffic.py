"""
Synthetic traffic generator for testing the inference service.

Generates random prediction requests to populate metrics in
Prometheus and test Grafana dashboards.

Usage:
    python scripts/generate_traffic.py --num-requests 100 --model-name construction_duration
"""

import argparse
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def generate_regression_features() -> dict:
    """Generate random features matching the construction regression model."""
    return {
        "Labor_Required": random.randint(1, 20),
        "Equipment_Units": random.randint(1, 10),
        "Material_Cost_USD": round(random.uniform(1000, 50000), 2),
        "Start_Constraint": random.randint(0, 30),
        "Resource_Constraint_Score": round(random.uniform(0, 1), 2),
        "Site_Constraint_Score": round(random.uniform(0, 1), 2),
        "Dependency_Count": random.randint(0, 8),
    }


def generate_classification_features() -> dict:
    """Generate random features matching the construction classification model."""
    return {
        "Task_Duration_Days": random.randint(1, 120),
        "Labor_Required": random.randint(1, 20),
        "Equipment_Units": random.randint(1, 10),
        "Material_Cost_USD": round(random.uniform(1000, 50000), 2),
        "Start_Constraint": random.randint(0, 30),
        "Resource_Constraint_Score": round(random.uniform(0, 1), 2),
        "Site_Constraint_Score": round(random.uniform(0, 1), 2),
        "Dependency_Count": random.randint(0, 8),
    }


def make_request(
    base_url: str,
    model_name: str,
    model_alias: str,
    task_type: str,
) -> dict:
    """Make a single prediction request."""
    if task_type == "regression":
        features = generate_regression_features()
    else:
        features = generate_classification_features()

    payload = {
        "features": features,
        "model_name": model_name,
        "model_alias": model_alias,
    }

    try:
        response = httpx.post(
            f"{base_url}/predict",
            json=payload,
            timeout=10.0,
        )
        return {
            "status": response.status_code,
            "success": response.status_code == 200,
            "latency": response.elapsed.total_seconds(),
        }
    except Exception as e:
        return {
            "status": 0,
            "success": False,
            "error": str(e),
            "latency": 0,
        }


def generate_traffic(
    num_requests: int = 100,
    base_url: str = "http://localhost:8000",
    model_name: str = "construction_duration",
    model_alias: str = "champion",
    task_type: str = "regression",
    max_workers: int = 10,
    delay: float = 0.1,
):
    """
    Generate synthetic traffic to test the inference service.

    Args:
        num_requests: Number of requests to send.
        base_url: FastAPI service URL.
        model_name: Registered model name.
        model_alias: Model alias (champion, candidate).
        task_type: "regression" or "classification".
        max_workers: Maximum concurrent workers.
        delay: Delay between requests (seconds).
    """
    print(f"Generating {num_requests} requests to {base_url}...")
    print(f"Model: {model_name}@{model_alias} ({task_type})")
    print("-" * 50)

    results = {"success": 0, "failure": 0, "latencies": []}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(num_requests):
            future = executor.submit(
                make_request, base_url, model_name, model_alias, task_type
            )
            futures.append(future)

            if delay > 0:
                time.sleep(delay)

        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                results["success"] += 1
                results["latencies"].append(result["latency"])
            else:
                results["failure"] += 1

    total_time = time.time() - start_time
    success_rate = results["success"] / num_requests * 100

    # Summary
    print("-" * 50)
    print(f"Total requests:  {num_requests}")
    print(f"Success:         {results['success']} ({success_rate:.1f}%)")
    print(f"Failed:          {results['failure']}")
    print(f"Total time:      {total_time:.2f}s")
    print(f"Throughput:      {num_requests / total_time:.1f} req/s")

    if results["latencies"]:
        avg_latency = sum(results["latencies"]) / len(results["latencies"])
        max_latency = max(results["latencies"])
        min_latency = min(results["latencies"])
        print(f"Avg latency:     {avg_latency * 1000:.1f}ms")
        print(f"Min latency:     {min_latency * 1000:.1f}ms")
        print(f"Max latency:     {max_latency * 1000:.1f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic traffic")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model-name", type=str, default="construction_duration")
    parser.add_argument("--model-alias", type=str, default="champion")
    parser.add_argument("--task-type", type=str, default="regression")
    parser.add_argument("--max-workers", type=int, default=10)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    generate_traffic(
        num_requests=args.num_requests,
        base_url=args.base_url,
        model_name=args.model_name,
        model_alias=args.model_alias,
        task_type=args.task_type,
        max_workers=args.max_workers,
        delay=args.delay,
    )
