class FormFinderError(Exception):
    """Base exception class for FormFinder2 application."""
    pass


class APIError(FormFinderError):
    """Raised when an API call fails after retries or receives an invalid response."""
    pass

class FeatureError(FormFinderError):
    """Raised when feature extraction/parsing fails or fields are missing."""
    pass

class FeatureComputationError(FormFinderError):
    """Raised when feature computation fails."""
    
    def __init__(self, message: str, fixture_id: int = None, feature_type: str = None):
        super().__init__(message)
        self.fixture_id = fixture_id
        self.feature_type = feature_type

class DataQualityError(FormFinderError):
    """Raised when data quality is below acceptable thresholds."""
    
    def __init__(self, message: str, quality_score: float = None, threshold: float = None):
        super().__init__(message)
        self.quality_score = quality_score
        self.threshold = threshold

class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""
    
    def __init__(self, message: str = "API rate limit exceeded", retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


class DatabaseError(FormFinderError):
    """Raised when database operations fail."""
    pass


class ConfigurationError(FormFinderError):
    """Raised when configuration is invalid or missing."""
    pass


class TrainingError(FormFinderError):
    """Raised when model training fails."""
    pass


class ModelValidationError(TrainingError):
    """Raised when model validation fails."""
    
    def __init__(self, message: str, metric_name: str = None, metric_value: float = None, threshold: float = None):
        super().__init__(message)
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.threshold = threshold


class WorkflowError(FormFinderError):
    """Raised when workflow execution fails."""
    pass


class SchedulerError(FormFinderError):
    """Raised when scheduler operations fail."""
    pass


class TaskExecutionError(FormFinderError):
    """Raised when scheduled task execution fails."""
    
    def __init__(self, message: str, task_id: str = None, task_type: str = None):
        super().__init__(message)
        self.task_id = task_id
        self.task_type = task_type


class HealthCheckError(FormFinderError):
    """Raised when health check operations fail."""
    pass


class AlertError(FormFinderError):
    """Raised when alert operations fail."""
    pass


class MetricsError(FormFinderError):
    """Raised when metrics collection fails."""
    pass


class NotificationError(FormFinderError):
    """Raised when notification sending fails."""
    pass