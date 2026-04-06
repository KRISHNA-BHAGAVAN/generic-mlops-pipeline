"""
Inference request/response schemas for the FastAPI serving endpoint.

These Pydantic models define the API contract for the prediction service.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""

    features: Dict[str, Union[float, int, str]] = Field(
        ...,
        description="Feature column name -> value mapping",
    )
    model_name: str = Field(
        ...,
        description="Name of the registered model to use",
    )
    model_alias: str = Field(
        default="champion",
        description="Model alias to load (e.g. 'champion', 'candidate')",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": {
                        "Labor_Required": 14,
                        "Equipment_Units": 6,
                        "Material_Cost_USD": 16789.73,
                        "Start_Constraint": 0,
                        "Resource_Constraint_Score": 0.41,
                        "Site_Constraint_Score": 0.59,
                        "Dependency_Count": 4,
                    },
                    "model_name": "construction_duration",
                    "model_alias": "champion",
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response body for the /predict endpoint."""

    prediction: Union[float, int, str] = Field(
        ...,
        description="Model prediction (numeric for regression, class for classification)",
    )
    prediction_proba: Optional[Dict[str, float]] = Field(
        default=None,
        description="Class probabilities (classification only)",
    )
    model_name: str = Field(..., description="Registered model name used")
    model_version: str = Field(..., description="Model version number")
    model_uri: str = Field(..., description="MLflow model URI")
    timestamp: str = Field(..., description="ISO-8601 prediction timestamp")


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""

    instances: List[Dict[str, Union[float, int, str]]] = Field(
        ...,
        min_length=1,
        description="List of feature dictionaries for batch prediction",
    )
    model_name: str = Field(..., description="Registered model name")
    model_alias: str = Field(default="champion", description="Model alias")


class BatchPredictionResponse(BaseModel):
    """Response body for batch predictions."""

    predictions: List[Union[float, int, str]] = Field(
        ..., description="List of predictions"
    )
    model_name: str
    model_version: str
    count: int = Field(..., description="Number of predictions made")
    timestamp: str


class HealthResponse(BaseModel):
    """Response body for health check endpoint."""

    status: str
    timestamp: str
    models_loaded: int = 0


class ErrorResponse(BaseModel):
    """Error response body."""

    error: str
    detail: str
    status_code: int
