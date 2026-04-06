FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency installation
RUN pip install --no-cache-dir uv

# Copy dependency files + README (hatchling build requires README.md)
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies
RUN uv pip install --system --no-cache .

# Copy application code
COPY src/ ./src/
COPY deployment/app.py ./deployment/app.py

# Expose ports: 8000 (API) + 8001 (Prometheus metrics)
EXPOSE 8000

# Environment variables (overridden at runtime via .env or Docker Compose)
ENV MLFLOW_TRACKING_URI=""
ENV MLFLOW_TRACKING_USERNAME=""
ENV MLFLOW_TRACKING_PASSWORD=""
ENV APP_PORT=8000
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Start FastAPI service
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
