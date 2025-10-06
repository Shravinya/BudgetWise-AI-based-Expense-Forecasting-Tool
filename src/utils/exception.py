"""
Custom exceptions for the Budget Tool project.
These help in handling specific error scenarios gracefully.
"""

class BudgetToolException(Exception):
    """
    Base exception class for all Budget Tool specific errors.
    """
    pass


class DataIngestionError(BudgetToolException):
    """
    Raised when there's an issue during data ingestion (e.g., file not found, invalid format).
    """
    def __init__(self, message: str):
        self.message = f"Data Ingestion Error: {message}"
        super().__init__(self.message)


class DataPreprocessingError(BudgetToolException):
    """
    Raised when preprocessing fails (e.g., missing columns, outlier handling issues).
    """
    def __init__(self, message: str):
        self.message = f"Data Preprocessing Error: {message}"
        super().__init__(self.message)


class EDAError(BudgetToolException):
    """
    Raised during EDA generation (e.g., plotting failures due to missing data).
    """
    def __init__(self, message: str):
        self.message = f"EDA Error: {message}"
        super().__init__(self.message)


class FeatureEngineeringError(BudgetToolException):
    """
    Raised when feature engineering encounters issues (e.g., insufficient data for lags).
    """
    def __init__(self, message: str):
        self.message = f"Feature Engineering Error: {message}"
        super().__init__(self.message)


class ModelTrainingError(BudgetToolException):
    """
    Raised during model training (e.g., convergence failure, invalid parameters).
    """
    def __init__(self, model_name: str, message: str):
        self.message = f"Model Training Error ({model_name}): {message}"
        super().__init__(self.message)


class ModelEvaluationError(BudgetToolException):
    """
    Raised during model evaluation (e.g., metric computation failure).
    """
    def __init__(self, message: str):
        self.message = f"Model Evaluation Error: {message}"
        super().__init__(self.message)


class BudgetRecommendationError(BudgetToolException):
    """
    Raised when generating recommendations (e.g., no expense data).
    """
    def __init__(self, message: str):
        self.message = f"Budget Recommendation Error: {message}"
        super().__init__(self.message)


class AppError(BudgetToolException):
    """
    Raised for Streamlit app-specific issues (e.g., file upload failure).
    """
    def __init__(self, message: str):
        self.message = f"App Error: {message}"
        super().__init__(self.message)