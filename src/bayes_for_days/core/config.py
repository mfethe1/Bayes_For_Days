"""
Configuration management for Bayes For Days platform.

This module handles all configuration settings, environment variables,
and application parameters using Pydantic settings management.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, Field, validator
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class OptimizationMode(str, Enum):
    """Optimization modes."""
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"


class DatabaseType(str, Enum):
    """Database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    
    All settings can be overridden using environment variables with the
    prefix 'BFD_' (e.g., BFD_DEBUG=true).
    """
    
    # Application settings
    app_name: str = Field(default="Bayes For Days", description="Application name")
    version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    
    # Server settings
    host: str = Field(default="localhost", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of worker processes")
    reload: bool = Field(default=False, description="Auto-reload on code changes")
    
    # Database settings
    database_type: DatabaseType = Field(default=DatabaseType.SQLITE, description="Database type")
    database_url: str = Field(default="sqlite:///./bayes_for_days.db", description="Database URL")
    database_echo: bool = Field(default=False, description="Echo SQL queries")
    
    # Redis settings (for caching and task queue)
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    
    # Optimization settings
    default_optimization_mode: OptimizationMode = Field(
        default=OptimizationMode.BAYESIAN, 
        description="Default optimization mode"
    )
    max_iterations: int = Field(default=100, description="Maximum optimization iterations")
    convergence_tolerance: float = Field(default=1e-6, description="Convergence tolerance")
    n_initial_points: int = Field(default=10, description="Number of initial random points")
    
    # Gaussian Process settings
    gp_kernel: str = Field(default="matern", description="GP kernel type")
    gp_noise_variance: float = Field(default=1e-6, description="GP noise variance")
    gp_length_scale_bounds: tuple = Field(default=(1e-5, 1e5), description="GP length scale bounds")
    
    # Acquisition function settings
    acquisition_function: str = Field(default="expected_improvement", description="Acquisition function")
    acquisition_optimizer: str = Field(default="lbfgs", description="Acquisition optimizer")
    n_restarts: int = Field(default=10, description="Number of optimization restarts")
    
    # Multi-objective settings
    pareto_front_size: int = Field(default=100, description="Maximum Pareto front size")
    hypervolume_ref_point: Optional[List[float]] = Field(
        default=None, 
        description="Reference point for hypervolume calculation"
    )
    
    # Ensemble settings
    ensemble_models: List[str] = Field(
        default=["gaussian_process", "random_forest", "neural_network"],
        description="Models to include in ensemble"
    )
    ensemble_weights: Optional[List[float]] = Field(
        default=None,
        description="Ensemble model weights"
    )
    
    # Data settings
    max_file_size: int = Field(default=100 * 1024 * 1024, description="Maximum file size in bytes")
    allowed_file_types: List[str] = Field(
        default=[".csv", ".xlsx", ".json"],
        description="Allowed file types"
    )
    data_validation: bool = Field(default=True, description="Enable data validation")
    
    # Security settings
    secret_key: str = Field(default="your-secret-key-here", description="Secret key for JWT")
    access_token_expire_minutes: int = Field(default=30, description="Access token expiration")
    
    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    models_dir: Path = Field(default=Path("models"), description="Models directory")
    
    class Config:
        env_prefix = "BFD_"
        env_file = ".env"
        case_sensitive = False
    
    @validator("data_dir", "logs_dir", "models_dir")
    def create_directories(cls, v):
        """Create directories if they don't exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator("ensemble_weights")
    def validate_ensemble_weights(cls, v, values):
        """Validate ensemble weights."""
        if v is not None:
            ensemble_models = values.get("ensemble_models", [])
            if len(v) != len(ensemble_models):
                raise ValueError("Ensemble weights must match number of models")
            if abs(sum(v) - 1.0) > 1e-6:
                raise ValueError("Ensemble weights must sum to 1.0")
        return v
    
    def get_database_url(self) -> str:
        """Get the database URL."""
        return self.database_url
    
    def get_redis_url(self) -> str:
        """Get the Redis URL."""
        return self.redis_url
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.dict()


# Global settings instance
settings = Settings()
