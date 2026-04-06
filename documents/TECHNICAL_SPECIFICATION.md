# Technical Specification Supplement
## Generic MLOps Pipeline – Implementation Details for AI Builders

**Purpose:** Fill critical gaps in the PRD to enable unambiguous implementation by Claude Code, Antigravity, or other AI builders.

---

## 1. Model Interface Contract

### 1.1 Model Training Signature

All task-type training functions must follow this signature:

```python
from typing import Tuple, Dict, Any
from sklearn.base import BaseEstimator
import pandas as pd
import numpy as np

def train_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: "ExperimentConfig",
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a regression model.
    
    Args:
        X_train: Feature matrix (rows=samples, cols=features)
        y_train: Target vector (numeric)
        config: Experiment configuration object
    
    Returns:
        Tuple of:
        - trained_model: sklearn-compatible estimator with .predict(X) method
        - metadata: dict with keys:
            - "feature_names": list of feature column names
            - "target_name": str
            - "training_samples": int
    
    Raises:
        ValueError: If config is invalid for this task
        RuntimeError: If training fails (e.g., singular matrix)
    """
    model = RandomForestRegressor(
        n_estimators=config.model_params.get("n_estimators", 100),
        max_depth=config.model_params.get("max_depth", None),
    )
    model.fit(X_train, y_train)
    
    return model, {
        "feature_names": X_train.columns.tolist(),
        "target_name": config.target_column,
        "training_samples": len(X_train),
    }


def train_classification(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: "ExperimentConfig",
) -> Tuple[BaseEstimator, Dict[str, Any]]:
    """
    Train a classification model.
    
    Args:
        X_train: Feature matrix
        y_train: Target vector (categorical/binary)
        config: Experiment configuration
    
    Returns:
        Tuple of (model, metadata)
    """
    # Similar structure to regression
    pass
```

### 1.2 Model Prediction Signature

```python
def predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions from a trained model.
    
    Args:
        model: Trained estimator (from train_regression/train_classification)
        X: Feature matrix (must have same columns as training data)
    
    Returns:
        np.ndarray: Predictions (shape: (n_samples,))
    
    Raises:
        ValueError: If X columns don't match model.feature_names_in_
    """
    return model.predict(X)
```

### 1.3 Required Model Attributes

After training, every model MUST have these attributes (set by sklearn or manually):

```python
model.feature_names_in_  # np.ndarray of feature names
model.n_features_in_     # int, number of features
model.classes_           # (Classification only) array of class labels

# Custom attribute added by pipeline:
model._metadata = {
    "task_type": "regression" or "classification",
    "target_column": "column_name",
    "training_samples": 1000,
}
```

### 1.4 Model Factory Pattern

```python
from enum import Enum

class ModelType(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST_REGRESSION = "random_forest_regression"
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST_CLASSIFICATION = "random_forest_classification"

def get_model_class(model_type: str, task_type: str) -> type:
    """
    Return the appropriate model class based on task and model type.
    
    Args:
        model_type: str from config.model_type
        task_type: "regression" or "classification"
    
    Returns:
        Uninstantiated model class (e.g., RandomForestRegressor)
    
    Raises:
        ValueError: If model_type is not supported for task_type
    """
    FACTORY = {
        ("regression", "linear_regression"): LinearRegression,
        ("regression", "random_forest_regression"): RandomForestRegressor,
        ("classification", "logistic_regression"): LogisticRegression,
        ("classification", "random_forest_classification"): RandomForestClassifier,
    }
    
    key = (task_type, model_type)
    if key not in FACTORY:
        raise ValueError(
            f"Model {model_type} not supported for task {task_type}. "
            f"Allowed: {list(FACTORY.keys())}"
        )
    return FACTORY[key]
```

---

## 2. Config Schema & Validation

### 2.1 Pydantic Config Schema

```python
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
from enum import Enum

class TaskType(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

class SplitStrategy(str, Enum):
    RANDOM = "random"
    TEMPORAL = "temporal"
    STRATIFIED = "stratified"

class PreprocessingStep(BaseModel):
    type: str  # "normalize", "one_hot_encode", etc.
    columns: Optional[List[str]] = None
    params: Optional[Dict] = None

class ExperimentConfig(BaseModel):
    # Metadata
    experiment_name: str = Field(..., min_length=1, max_length=100)
    user: str = Field(..., min_length=1, max_length=50)
    
    # Task definition
    task_type: TaskType
    target_column: str
    feature_columns: List[str] = Field(..., min_items=1)
    
    # Dataset
    dataset_source: str  # Path or DVC reference
    dvc_version: Optional[str] = None  # e.g., "abc123def456"
    
    # Model
    model_type: str
    model_params: Dict[str, any] = Field(default_factory=dict)
    
    # Preprocessing
    preprocessing: Optional[List[PreprocessingStep]] = None
    
    # Splitting
    split_strategy: SplitStrategy = SplitStrategy.RANDOM
    test_size: float = Field(0.2, ge=0.01, le=0.5)
    val_size: float = Field(0.2, ge=0.01, le=0.5)
    random_state: int = 42
    date_column: Optional[str] = None  # For temporal split
    
    # Metrics
    metrics: List[str] = Field(..., min_items=1)
    
    # MLflow
    mlflow_tags: Dict[str, str] = Field(default_factory=dict)
    
    # Registry
    registry_name: Optional[str] = None
    
    @validator("target_column")
    def target_not_in_features(cls, v, values):
        if "feature_columns" in values and v in values["feature_columns"]:
            raise ValueError("target_column cannot be in feature_columns")
        return v
    
    @validator("metrics")
    def validate_metrics_for_task(cls, v, values):
        if "task_type" not in values:
            return v
        
        task_type = values["task_type"]
        allowed_metrics = METRICS_REGISTRY.get(task_type.value, {}).keys()
        
        invalid = [m for m in v if m not in allowed_metrics]
        if invalid:
            raise ValueError(
                f"Metrics {invalid} not allowed for task {task_type}. "
                f"Allowed: {list(allowed_metrics)}"
            )
        return v
    
    @validator("model_type")
    def validate_model_for_task(cls, v, values):
        if "task_type" not in values:
            return v
        
        task_type = values["task_type"].value
        try:
            get_model_class(v, task_type)
        except ValueError as e:
            raise ValueError(str(e))
        return v
    
    class Config:
        use_enum_values = False

# Usage:
config = ExperimentConfig.parse_file("configs/regression/exp_001.yaml")
```

### 2.2 Config Validation Rules

```python
VALIDATION_RULES = {
    "regression": {
        "allowed_metrics": ["mse", "rmse", "mae", "r2"],
        "forbidden_metrics": ["accuracy", "precision", "recall", "f1", "auc"],
        "allowed_model_types": [
            "linear_regression",
            "random_forest_regression",
        ],
        "target_dtype": ["numeric"],  # int or float
    },
    "classification": {
        "allowed_metrics": ["accuracy", "precision", "recall", "f1", "auc", "confusion_matrix"],
        "forbidden_metrics": ["mse", "rmse", "mae", "r2"],
        "allowed_model_types": [
            "logistic_regression",
            "random_forest_classification",
        ],
        "target_dtype": ["categorical", "binary"],
    },
}

def validate_config_against_data(config: ExperimentConfig, df: pd.DataFrame):
    """Validate config against actual dataset."""
    
    # Check target exists
    if config.target_column not in df.columns:
        raise ValueError(f"Target column '{config.target_column}' not in dataset")
    
    # Check features exist
    missing = set(config.feature_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Feature columns {missing} not in dataset")
    
    # Check target dtype matches task_type
    target_dtype = df[config.target_column].dtype
    rules = VALIDATION_RULES[config.task_type.value]
    
    if config.task_type == TaskType.REGRESSION:
        if not pd.api.types.is_numeric_dtype(target_dtype):
            raise ValueError(
                f"Regression requires numeric target, got {target_dtype}"
            )
    elif config.task_type == TaskType.CLASSIFICATION:
        if target_dtype == "object" or pd.api.types.is_categorical_dtype(target_dtype):
            pass  # OK
        elif len(df[config.target_column].unique()) < 3:
            pass  # Binary classification
        else:
            raise ValueError(
                f"Classification target must be categorical, got {target_dtype}"
            )
```

### 2.3 Config File Examples

**regression/customer_churn_v1.yaml:**
```yaml
experiment_name: "customer_churn_prediction_v1"
user: "alice"
task_type: "regression"
dataset_source: "data/processed/customer_data.csv"
dvc_version: "abc123def456"
target_column: "lifetime_value"
feature_columns:
  - "age"
  - "account_tenure_months"
  - "support_tickets"
  - "purchase_frequency"
model_type: "random_forest_regression"
model_params:
  n_estimators: 100
  max_depth: 10
  min_samples_split: 5
  random_state: 42
preprocessing:
  - type: "normalize"
    columns: ["age", "account_tenure_months"]
split_strategy: "random"
test_size: 0.2
val_size: 0.2
random_state: 42
metrics:
  - "mse"
  - "rmse"
  - "mae"
  - "r2"
mlflow_tags:
  team: "growth"
  priority: "high"
  model_family: "random_forest"
registry_name: "customer_churn"
```

**classification/fraud_detection_v1.yaml:**
```yaml
experiment_name: "fraud_detection_v1"
user: "bob"
task_type: "classification"
dataset_source: "data/processed/transactions.csv"
dvc_version: "def456abc123"
target_column: "is_fraud"
feature_columns:
  - "transaction_amount"
  - "merchant_category"
  - "hours_since_last"
  - "device_type"
model_type: "random_forest_classification"
model_params:
  n_estimators: 100
  max_depth: 15
  class_weight: "balanced"
  random_state: 42
split_strategy: "stratified"
test_size: 0.2
val_size: 0.1
random_state: 42
metrics:
  - "accuracy"
  - "precision"
  - "recall"
  - "f1"
  - "auc"
mlflow_tags:
  team: "security"
  priority: "critical"
registry_name: "fraud_detector"
```

---

## 3. MLflow Logging Specification

### 3.1 Logging Schema

```python
from mlflow import log_metric, log_param, set_tag, log_artifact
from datetime import datetime
import json

def log_experiment_run(
    config: ExperimentConfig,
    model,
    metadata: Dict,
    metrics: Dict[str, float],
    evaluation_plots: Dict[str, str],  # artifact_name -> file_path
):
    """
    Log a complete experiment run to MLflow.
    
    Args:
        config: ExperimentConfig object
        model: Trained model object
        metadata: Dict from train_* function
        metrics: Dict of metric_name -> value
        evaluation_plots: Dict of artifact_name -> local_file_path
    """
    
    # Log configuration parameters
    log_param("task_type", config.task_type.value)
    log_param("model_type", config.model_type)
    log_param("dataset_source", config.dataset_source)
    log_param("target_column", config.target_column)
    log_param("n_features", len(config.feature_columns))
    log_param("test_size", config.test_size)
    log_param("split_strategy", config.split_strategy.value)
    
    # Log model hyperparameters
    for key, value in config.model_params.items():
        if isinstance(value, (int, float, str, bool)):
            log_param(f"model_param_{key}", value)
    
    # Log evaluation metrics
    for metric_name, metric_value in metrics.items():
        log_metric(metric_name, metric_value)
    
    # Log metadata
    set_tag("user", config.user)
    set_tag("experiment_name", config.experiment_name)
    set_tag("task_type", config.task_type.value)
    set_tag("validation_status", "pending")  # Initial status
    
    # Log MLflow-specific tags
    for key, value in config.mlflow_tags.items():
        set_tag(f"custom_{key}", value)
    
    # Log DVC version
    if config.dvc_version:
        set_tag("dvc_version", config.dvc_version)
    
    # Log timestamp
    set_tag("run_datetime", datetime.utcnow().isoformat())
    
    # Log artifacts
    for artifact_name, file_path in evaluation_plots.items():
        log_artifact(file_path, artifact_path=artifact_name)
    
    # Log model signature
    signature = mlflow.models.infer_signature(
        model_input=metadata.get("X_sample"),  # Small sample for signature
        model_output=metadata.get("y_pred_sample"),
    )
    
    # Log model
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        signature=signature,
        input_example=metadata.get("X_sample"),
    )

# Logging template in training pipeline:
mlflow.start_run(run_name=config.experiment_name)
try:
    model, metadata = train_regression(X_train, y_train, config)
    metrics = evaluate_regression(model, X_test, y_test)
    log_experiment_run(config, model, metadata, metrics, {})
    mlflow.end_run()
except Exception as e:
    mlflow.log_param("error", str(e))
    mlflow.end_run(status="FAILED")
    raise
```

### 3.2 MLflow Logging Checklist

```yaml
Params (MLflow):
  ✓ task_type
  ✓ model_type
  ✓ dataset_source
  ✓ target_column
  ✓ n_features
  ✓ All model_params_*
  ✓ test_size
  ✓ split_strategy

Metrics (MLflow):
  ✓ All evaluation metrics (mse, rmse, accuracy, etc.)
  ✓ Training time (seconds)
  ✓ Model size (bytes)

Tags (MLflow):
  ✓ user
  ✓ experiment_name
  ✓ task_type
  ✓ validation_status (starts as "pending")
  ✓ dvc_version
  ✓ run_datetime
  ✓ custom_* (from mlflow_tags in config)

Artifacts (MLflow):
  ✓ model/ (trained model via sklearn.log_model)
  ✓ model/signature.json (auto-generated)
  ✓ evaluation_plots/ (confusion matrix, feature importance, etc.)
  ✓ config.yaml (copy of experiment config)

Model Signature:
  ✓ Input schema (feature columns + dtypes)
  ✓ Output schema (prediction dtype)
```

---

## 4. Evaluation & Metrics Specification

### 4.1 Metrics Registry

```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import numpy as np

METRICS_REGISTRY = {
    "regression": {
        "mse": lambda y_true, y_pred: mean_squared_error(y_true, y_pred),
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
        "r2": lambda y_true, y_pred: r2_score(y_true, y_pred),
    },
    "classification": {
        "accuracy": lambda y_true, y_pred: accuracy_score(y_true, y_pred),
        "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="weighted"),
        "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="weighted"),
        "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
        "auc": lambda y_true, y_pred: roc_auc_score(y_true, y_pred),
    },
}

def evaluate(
    task_type: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str],
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        task_type: "regression" or "classification"
        y_true: True labels/targets
        y_pred: Predicted labels/targets
        metrics: List of metric names to compute
    
    Returns:
        Dict of metric_name -> metric_value
    
    Raises:
        ValueError: If metric not in METRICS_REGISTRY
    """
    result = {}
    registry = METRICS_REGISTRY.get(task_type, {})
    
    for metric_name in metrics:
        if metric_name not in registry:
            raise ValueError(
                f"Unknown metric '{metric_name}' for task {task_type}. "
                f"Available: {list(registry.keys())}"
            )
        
        metric_fn = registry[metric_name]
        result[metric_name] = metric_fn(y_true, y_pred)
    
    return result
```

### 4.2 Evaluation Pipeline

```python
def evaluate_regression(model, X_test, y_test, config: ExperimentConfig):
    """Evaluate regression model and return metrics."""
    y_pred = model.predict(X_test)
    metrics = evaluate("regression", y_test, y_pred, config.metrics)
    return metrics

def evaluate_classification(model, X_test, y_test, config: ExperimentConfig):
    """Evaluate classification model and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    metrics = evaluate("classification", y_test, y_pred, config.metrics)
    return metrics
```

---

## 5. FastAPI Serving Specification

### 5.1 Request & Response Schemas

```python
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import numpy as np

class PredictionRequest(BaseModel):
    """Request body for /predict endpoint."""
    features: Dict[str, float | str | int]  # Column name -> value
    model_alias: Optional[str] = "champion"  # Which version to use
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "age": 35,
                    "income": 50000,
                    "account_tenure_months": 24,
                },
                "model_alias": "champion",
            }
        }

class PredictionResponse(BaseModel):
    """Response body for /predict endpoint."""
    prediction: float | int | str  # Numeric for regression, class for classification
    prediction_proba: Optional[Dict[str, float]] = None  # For classification
    model_version: str  # Which model version was used
    model_uri: str  # MLflow URI of the model
    timestamp: str  # ISO timestamp
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 0.85,
                "model_version": "3",
                "model_uri": "models:/fraud_detector@champion",
                "timestamp": "2026-04-06T10:30:00Z",
            }
        }

class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    status_code: int
```

### 5.2 FastAPI Service Implementation

```python
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import mlflow
import pandas as pd
from datetime import datetime
import logging

# Global model cache
_model_cache = {}
_model_signatures = {}

async def load_model_by_alias(model_name: str, alias: str = "champion"):
    """
    Load a model from MLflow registry by alias.
    
    Args:
        model_name: Name of registered model
        alias: Model alias (e.g., "champion", "candidate")
    
    Returns:
        Loaded model object
    
    Raises:
        HTTPException: If model not found or loading fails
    """
    cache_key = f"{model_name}@{alias}"
    
    # Return cached model if available
    if cache_key in _model_cache:
        return _model_cache[cache_key]
    
    try:
        # Load model from registry
        model_uri = f"models:/{model_name}@{alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Cache it
        _model_cache[cache_key] = model
        
        return model
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Failed to load model {model_name}@{alias}: {str(e)}"
        )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager."""
    # Startup: preload default model
    try:
        await load_model_by_alias("fraud_detector", "champion")
    except HTTPException:
        logging.warning("Could not preload champion model")
    yield
    # Shutdown: cleanup
    _model_cache.clear()

app = FastAPI(
    title="MLOps Inference Service",
    description="Serve models from MLflow registry",
    lifespan=lifespan,
)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Generate a prediction using the specified model.
    
    Args:
        request: PredictionRequest with features and model_alias
    
    Returns:
        PredictionResponse with prediction and metadata
    
    Raises:
        400: Invalid request format
        404: Model or alias not found
        503: Model loading or prediction failed
    """
    try:
        # Convert request features to DataFrame
        X = pd.DataFrame([request.features])
        
        # Load model (simplified; in practice, determine model_name from request)
        model_name = "customer_churn"  # Could come from request header
        model = await load_model_by_alias(model_name, request.model_alias)
        
        # Get model version info
        client = mlflow.tracking.MlflowClient()
        registered_model = client.get_registered_model(model_name)
        # Find version by alias
        version = None
        for v in registered_model.aliases:
            if v.alias == request.model_alias:
                version = v.version
                break
        
        # Predict
        prediction = model.predict(X)[0]
        
        return PredictionResponse(
            prediction=float(prediction),
            model_version=str(version),
            model_uri=f"models:/{model_name}@{request.model_alias}",
            timestamp=datetime.utcnow().isoformat(),
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Model alias not found: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
```

### 5.3 Docker Configuration

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install -r --no-cache

# Copy application code
COPY src/ ./src/
COPY deployment/app.py ./

# Expose port
EXPOSE 8000

# Environment variables (to be overridden at runtime)
ENV MLFLOW_TRACKING_URI=http://localhost:5000
ENV MLFLOW_TRACKING_USERNAME=""
ENV MLFLOW_TRACKING_PASSWORD=""

# Start FastAPI service
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Build and Run:**
```bash
# Build
docker build -t mlops-inference:latest .

# Run with DagsHub MLflow
docker run \
    -e MLFLOW_TRACKING_URI="https://dagshub.com/USERNAME/REPO/mlflow" \
    -e MLFLOW_TRACKING_USERNAME="USERNAME" \
    -e MLFLOW_TRACKING_PASSWORD="TOKEN" \
    -p 8000:8000 \
    mlops-inference:latest
```

---

## 6. Model Registry & Promotion Workflow

### 6.1 Registration & Tagging

```python
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion

def register_model(run_id: str, model_name: str, client: MlflowClient = None):
    """
    Register a model from a successful run.
    
    Args:
        run_id: MLflow run ID
        model_name: Name for registered model
        client: MLflowClient instance
    
    Returns:
        ModelVersion object
    """
    if client is None:
        client = mlflow.tracking.MlflowClient()
    
    model_uri = f"runs:/{run_id}/model"
    
    try:
        model_version = mlflow.register_model(model_uri, name=model_name)
    except Exception as e:
        raise RuntimeError(f"Failed to register model: {str(e)}")
    
    # Tag as pending validation
    client.set_model_version_tag(
        name=model_name,
        version=model_version.version,
        key="validation_status",
        value="pending"
    )
    
    return model_version

def promote_model(
    model_name: str,
    version: int,
    alias: str,  # "candidate", "champion", "production"
    client: MlflowClient = None,
):
    """
    Assign an alias to a model version.
    
    Args:
        model_name: Registered model name
        version: Model version number
        alias: Alias to assign
        client: MLflowClient instance
    """
    if client is None:
        client = mlflow.tracking.MlflowClient()
    
    try:
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=version,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to promote model: {str(e)}")

def approve_model(
    model_name: str,
    version: int,
    client: MlflowClient = None,
):
    """
    Mark a model as approved for production.
    
    Args:
        model_name: Registered model name
        version: Model version number
        client: MLflowClient instance
    """
    if client is None:
        client = mlflow.tracking.MlflowClient()
    
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="validation_status",
        value="passed"
    )
    
    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="approved_at",
        value=datetime.utcnow().isoformat()
    )
```

### 6.2 Selection & Ranking

```python
def rank_runs(
    experiment_name: str,
    metric: str,
    task_type: str,
    top_k: int = 5,
) -> List[Dict]:
    """
    Find and rank top-performing runs by metric.
    
    Args:
        experiment_name: MLflow experiment name
        metric: Metric to rank by (e.g., "r2", "f1")
        task_type: Task type (regression, classification)
        top_k: Return top K runs
    
    Returns:
        List of run dicts sorted by metric (descending for r2, f1; ascending for mse)
    """
    client = mlflow.tracking.MlflowClient()
    
    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"Experiment {experiment_name} not found")
    
    # Search runs filtered by task_type
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.task_type = '{task_type}'",
        max_results=10000,
        order_by=[f"metrics.{metric} DESC"],
    )
    
    result = []
    for run in runs[:top_k]:
        result.append({
            "run_id": run.info.run_id,
            "experiment_name": experiment.name,
            "metric": metric,
            "value": run.data.metrics.get(metric),
            "tags": run.data.tags,
            "params": run.data.params,
        })
    
    return result
```

---

## 7. Training Pipeline CLI

### 7.1 CLI Entry Point

```python
# src/pipelines/train_pipeline.py

import click
import mlflow
import pandas as pd
from pathlib import Path
from src.config.load_config import load_config, validate_config
from src.models.factory import get_model_class, train_regression, train_classification
from src.models.evaluate import evaluate
from src.models.registry import log_experiment_run

@click.command()
@click.option(
    "--config",
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
    help="Validate config without running experiment",
)
def main(config, mlflow_tracking_uri, dry_run):
    """
    Run a machine learning experiment.
    
    Example:
        python -m src.pipelines.train_pipeline \\
            --config configs/regression/exp_001.yaml
    """
    # Set MLflow tracking
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
    
    # Load and validate config
    click.echo("Loading config...")
    exp_config = load_config(config)
    
    click.echo("Validating config...")
    validate_config(exp_config)
    
    if dry_run:
        click.echo("✓ Config is valid. (Dry run, exiting.)")
        return
    
    # Load dataset
    click.echo(f"Loading dataset from {exp_config.dataset_source}...")
    df = pd.read_csv(exp_config.dataset_source)
    validate_config_against_data(exp_config, df)
    
    # Prepare features and target
    X = df[exp_config.feature_columns]
    y = df[exp_config.target_column]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=exp_config.test_size,
        random_state=exp_config.random_state,
    )
    
    # Train model
    click.echo(f"Training {exp_config.model_type}...")
    mlflow.start_run(run_name=exp_config.experiment_name)
    
    try:
        if exp_config.task_type == "regression":
            model, metadata = train_regression(X_train, y_train, exp_config)
        elif exp_config.task_type == "classification":
            model, metadata = train_classification(X_train, y_train, exp_config)
        
        # Evaluate
        click.echo("Evaluating...")
        metrics = evaluate(exp_config.task_type, model, X_test, y_test, exp_config)
        
        # Log to MLflow
        click.echo("Logging to MLflow...")
        log_experiment_run(exp_config, model, metadata, metrics, {})
        
        click.echo(f"✓ Run completed. MLflow run ID: {mlflow.active_run().info.run_id}")
        mlflow.end_run()
    
    except Exception as e:
        click.echo(f"✗ Training failed: {str(e)}", err=True)
        mlflow.log_param("error", str(e))
        mlflow.end_run(status="FAILED")
        raise

if __name__ == "__main__":
    main()
```

### 7.2 Makefile

```makefile
.PHONY: train test clean docker-build docker-run help

help:
	@echo "Available targets:"
	@echo "  make train CONFIG=path/to/config.yaml      Run experiment"
	@echo "  make test                                   Run tests"
	@echo "  make docker-build                           Build Docker image"
	@echo "  make docker-run                             Run inference service"

train:
	python -m src.pipelines.train_pipeline --config $(CONFIG)

train-dry:
	python -m src.pipelines.train_pipeline --config $(CONFIG) --dry-run

test:
	pytest tests/ -v

docker-build:
	docker build -t mlops-inference:latest .

docker-run:
	docker run \
		-e MLFLOW_TRACKING_URI=$(MLFLOW_TRACKING_URI) \
		-e MLFLOW_TRACKING_USERNAME=$(MLFLOW_USERNAME) \
		-e MLFLOW_TRACKING_PASSWORD=$(MLFLOW_PASSWORD) \
		-p 8000:8000 \
		mlops-inference:latest
```

---

## 8. Error Handling Strategy

```python
# src/pipelines/exceptions.py

class MLOpsException(Exception):
    """Base exception for MLOps pipeline."""
    pass

class ConfigError(MLOpsException):
    """Configuration is invalid."""
    pass

class DatasetError(MLOpsException):
    """Dataset loading or validation failed."""
    pass

class ModelError(MLOpsException):
    """Model training or evaluation failed."""
    pass

class RegistryError(MLOpsException):
    """MLflow registry operation failed."""
    pass

# Error handling map
ERROR_HANDLING = {
    "dataset_not_found": {
        "status": 404,
        "message": "Dataset not found. Ensure DVC is pulled: `dvc pull`",
        "recovery": "user",
    },
    "invalid_config": {
        "status": 422,
        "message": "Configuration is invalid. Check error details.",
        "recovery": "user",
    },
    "mlflow_unreachable": {
        "status": 503,
        "message": "MLflow server unreachable. Check MLFLOW_TRACKING_URI.",
        "recovery": "infra",
    },
    "model_training_failed": {
        "status": 500,
        "message": "Model training failed. Check MLflow for details.",
        "recovery": "investigation",
    },
}
```

---

## 9. Repository Setup Commands

```bash
# Clone repository
git clone https://github.com/team/generic-mlops-pipeline.git
cd generic-mlops-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set DagsHub credentials
export MLFLOW_TRACKING_URI=https://dagshub.com/USERNAME/REPO/mlflow
export MLFLOW_TRACKING_USERNAME=USERNAME
export MLFLOW_TRACKING_PASSWORD=<token>

# Pull DVC data
dvc pull

# Run experiment
python -m src.pipelines.train_pipeline --config configs/regression/exp_001.yaml

# Inspect results
mlflow ui  # Opens at http://localhost:5000
```

---

## 10. Complete Workflow Example

```bash
# 1. Pull latest code and data
git pull
dvc pull

# 2. Create experiment config
cat > configs/regression/exp_002_alice_v2.yaml << EOF
experiment_name: "customer_churn_v2"
user: "alice"
task_type: "regression"
...
EOF

# 3. Validate config
python -m src.pipelines.train_pipeline --config configs/regression/exp_002_alice_v2.yaml --dry-run

# 4. Run experiment
python -m src.pipelines.train_pipeline --config configs/regression/exp_002_alice_v2.yaml

# 5. Inspect in MLflow (open browser)
mlflow ui

# 6. Register top model (via MLflow UI or CLI)
mlflow models create --name customer_churn

# 7. Tag and promote (via Python script or MLflow API)
python << EOF
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.set_registered_model_alias("customer_churn", "champion", version="3")
EOF

# 8. Deploy (restart inference service)
docker pull mlops-inference:latest
docker-compose down
docker-compose up -d
```

---

## Summary

This specification supplements the PRD with:

1. ✅ Explicit model interface contracts
2. ✅ Pydantic config validation schema
3. ✅ Config examples (YAML)
4. ✅ MLflow logging checklist
5. ✅ Metrics registry and evaluation spec
6. ✅ FastAPI serving with schemas
7. ✅ Model registration and promotion code
8. ✅ Training pipeline CLI
9. ✅ Error handling strategy
10. ✅ End-to-end workflow example

**With this specification, AI builders can implement the pipeline without ambiguity.**

