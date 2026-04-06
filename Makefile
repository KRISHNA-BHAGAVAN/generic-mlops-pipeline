.PHONY: help train train-dry register promote test docker-build docker-run monitoring-up monitoring-down install clean

help:  ## Show this help message
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Generic MLOps Pipeline — Available Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install all dependencies using uv
	uv sync

train:  ## Run training pipeline (CONFIG=path/to/config.yaml)
	python -m src.pipelines.train_pipeline --config $(CONFIG)

train-dry:  ## Validate config without running (CONFIG=path/to/config.yaml)
	python -m src.pipelines.train_pipeline --config $(CONFIG) --dry-run

train-register:  ## Run training + auto-register (CONFIG=path/to/config.yaml)
	python -m src.pipelines.train_pipeline --config $(CONFIG) --register

register:  ## Register a model (RUN_ID=xxx MODEL_NAME=yyy)
	python -m src.pipelines.register_pipeline --run-id $(RUN_ID) --model-name $(MODEL_NAME)

promote:  ## Promote model (MODEL_NAME=xxx VERSION=n ALIAS=champion)
	python -m src.selection.promote_model --model-name $(MODEL_NAME) --version $(VERSION) --alias $(ALIAS)

test:  ## Run all tests
	pytest tests/ -v

serve:  ## Start FastAPI inference service
	uvicorn deployment.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:  ## Build Docker image for inference service
	docker build -t mlops-inference:latest -f deployment/Dockerfile .

docker-run:  ## Run inference service in Docker
	docker run \
		--env-file .env \
		-p 8000:8000 \
		mlops-inference:latest

monitoring-up:  ## Start monitoring stack (Prometheus, Grafana, AlertManager)
	docker compose -f deployment/docker-compose.yml up -d

monitoring-down:  ## Stop monitoring stack
	docker compose -f deployment/docker-compose.yml down

monitoring-logs:  ## View monitoring stack logs
	docker compose -f deployment/docker-compose.yml logs -f

clean:  ## Clean up temporary files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .ruff_cache htmlcov *.egg-info dist build
