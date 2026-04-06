"""
Pipeline-specific exceptions for the MLOps pipeline.

Provides a hierarchy of exceptions to distinguish between
configuration, dataset, model, and registry errors.
"""


class MLOpsException(Exception):
    """Base exception for all MLOps pipeline errors."""
    pass


class ConfigError(MLOpsException):
    """Configuration is invalid or cannot be loaded."""
    pass


class DatasetError(MLOpsException):
    """Dataset loading, validation, or transformation failed."""
    pass


class ModelError(MLOpsException):
    """Model training, evaluation, or prediction failed."""
    pass


class RegistryError(MLOpsException):
    """MLflow registry operation failed."""
    pass


class ServingError(MLOpsException):
    """Model serving or inference failed."""
    pass
