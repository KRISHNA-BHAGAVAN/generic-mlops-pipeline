"""
Script to trigger Prometheus alerts for testing purposes.
"""

import sys
import time
import json
import logging
import requests
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def trigger_high_error_rate(api_url: str, count: int = 50):
    """
    Trigger HighErrorRate and PredictionErrors by sending malformed payloads 
    that might cause 500s or prediction errors.
    """
    logger.info(f"Triggering High Error Rate (sending {count} bad requests)...")
    url = f"{api_url}/predict"
    
    # We send invalid JSON or wrong data types to generate errors
    for i in range(count):
        try:
            # We explicitly send a payload that breaks internal processing
            response = requests.post(
                url,
                json={"features": {"bad_feature": "I will cause an error"}},
                timeout=1
            )
            logger.info(f"Sent request {i+1}/{count} - Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Connection error: {e}")
            break
        time.sleep(0.1)

def trigger_drift_alert(pushgateway_url: str):
    """
    Trigger DataDriftDetected and HighDriftColumnCount directly 
    by pushing dummy metrics to Pushgateway.
    """
    logger.info(f"Triggering Drift Alerts via Pushgateway at {pushgateway_url}...")
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        registry = CollectorRegistry()
        
        drift_gauge = Gauge("mlops_data_drift_detected", "...", registry=registry)
        drift_gauge.set(1)

        count_gauge = Gauge("mlops_drifted_columns_count", "...", registry=registry)
        count_gauge.set(10) # > 5 will trigger HighDriftColumnCount
        
        push_to_gateway(pushgateway_url, job="batch-monitoring", registry=registry)
        logger.info("Successfully pushed drift alert metrics.")
    except Exception as e:
        logger.error(f"Failed to push drift metrics: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trigger Prometheus alerts for testing")
    parser.add_argument("--api-url", default="http://localhost:8000", help="FastAPI prediction service URL")
    parser.add_argument("--pushgateway-url", default="http://localhost:9091", help="Prometheus Pushgateway URL")
    parser.add_argument("--type", choices=['api', 'drift', 'all'], default='all', help="Alerts to trigger")
    
    args = parser.parse_args()
    
    logger.info("Ensure that Prometheus and the FastAPI service are running!")
    
    if args.type in ['api', 'all']:
        trigger_high_error_rate(args.api_url)
        
    if args.type in ['drift', 'all']:
        trigger_drift_alert(args.pushgateway_url)

    logger.info("Done. Check Prometheus/Grafana and AlertManager for triggered alerts.")
