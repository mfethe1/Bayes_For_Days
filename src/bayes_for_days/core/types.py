"""
Type definitions and data structures for Bayes For Days platform.

This module contains all the custom types, enums, and data structures
used throughout the optimization platform.
"""

from enum import Enum
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime


class ObjectiveType(str, Enum):
    """Types of optimization objectives."""
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class AcquisitionFunction(str, Enum):
    """Available acquisition functions."""
    EXPECTED_IMPROVEMENT = "expected_improvement"
    UPPER_CONFIDENCE_BOUND = "upper_confidence_bound"
    PROBABILITY_OF_IMPROVEMENT = "probability_of_improvement"
    EXPECTED_PARETO_IMPROVEMENT = "expected_pareto_improvement"
    HYPERVOLUME_IMPROVEMENT = "hypervolume_improvement"
    QNEHVI = "qnehvi"  # q-Noisy Expected Hypervolume Improvement


class ModelType(str, Enum):
    """Available surrogate model types."""
    GAUSSIAN_PROCESS = "gaussian_process"
    RANDOM_FOREST = "random_forest"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"


class OptimizationStatus(str, Enum):
    """Optimization status."""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"
    STOPPED = "stopped"


class ParameterType(str, Enum):
    """Parameter types for optimization variables."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


@dataclass
class Parameter:
    """Definition of an optimization parameter."""
    name: str
    type: ParameterType
    bounds: Optional[Tuple[float, float]] = None
    categories: Optional[List[str]] = None
    default: Optional[Any] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """Validate parameter definition."""
        if self.type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
            if self.bounds is None:
                raise ValueError(f"Bounds required for {self.type} parameter")
        elif self.type in [ParameterType.CATEGORICAL, ParameterType.ORDINAL]:
            if self.categories is None:
                raise ValueError(f"Categories required for {self.type} parameter")


class ParameterSpace(BaseModel):
    """Definition of the optimization parameter space."""
    parameters: List[Parameter] = Field(..., description="List of parameters")
    constraints: Optional[List[str]] = Field(default=None, description="Parameter constraints")
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get bounds for continuous and integer parameters."""
        bounds = []
        for param in self.parameters:
            if param.type in [ParameterType.CONTINUOUS, ParameterType.INTEGER]:
                bounds.append(param.bounds)
        return bounds
    
    def get_categorical_dims(self) -> List[int]:
        """Get indices of categorical parameters."""
        return [i for i, param in enumerate(self.parameters) 
                if param.type in [ParameterType.CATEGORICAL, ParameterType.ORDINAL]]
    
    def get_dimension(self) -> int:
        """Get the dimensionality of the parameter space."""
        return len(self.parameters)


class Objective(BaseModel):
    """Definition of an optimization objective."""
    name: str = Field(..., description="Objective name")
    type: ObjectiveType = Field(..., description="Objective type (minimize/maximize)")
    weight: float = Field(default=1.0, description="Objective weight")
    target: Optional[float] = Field(default=None, description="Target value")
    tolerance: Optional[float] = Field(default=None, description="Tolerance for target")
    description: Optional[str] = Field(default=None, description="Objective description")


class ExperimentPoint(BaseModel):
    """A single experimental point with parameters and results."""
    id: Optional[str] = Field(default=None, description="Unique identifier")
    parameters: Dict[str, Any] = Field(..., description="Parameter values")
    objectives: Dict[str, float] = Field(..., description="Objective values")
    constraints: Optional[Dict[str, float]] = Field(default=None, description="Constraint values")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp")
    is_feasible: bool = Field(default=True, description="Feasibility flag")
    uncertainty: Optional[Dict[str, float]] = Field(default=None, description="Prediction uncertainty")


class OptimizationResult(BaseModel):
    """Result of an optimization run."""
    best_point: ExperimentPoint = Field(..., description="Best point found")
    pareto_front: Optional[List[ExperimentPoint]] = Field(default=None, description="Pareto front")
    all_points: List[ExperimentPoint] = Field(..., description="All evaluated points")
    n_iterations: int = Field(..., description="Number of iterations")
    status: OptimizationStatus = Field(..., description="Optimization status")
    convergence_history: List[float] = Field(default_factory=list, description="Convergence history")
    hypervolume_history: Optional[List[float]] = Field(default=None, description="Hypervolume history")
    execution_time: float = Field(..., description="Execution time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ModelPrediction(BaseModel):
    """Prediction from a surrogate model."""
    mean: Union[float, np.ndarray] = Field(..., description="Predicted mean")
    variance: Union[float, np.ndarray] = Field(..., description="Predicted variance")
    std: Union[float, np.ndarray] = Field(..., description="Predicted standard deviation")
    confidence_interval: Optional[Tuple[float, float]] = Field(
        default=None, description="Confidence interval"
    )
    model_type: ModelType = Field(..., description="Model type used for prediction")


class AcquisitionValue(BaseModel):
    """Value from an acquisition function."""
    value: float = Field(..., description="Acquisition function value")
    gradient: Optional[np.ndarray] = Field(default=None, description="Gradient")
    function_type: AcquisitionFunction = Field(..., description="Acquisition function type")
    parameters: Dict[str, Any] = Field(..., description="Parameters used")


class ValidationResult(BaseModel):
    """Result of data validation."""
    is_valid: bool = Field(..., description="Whether data is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    n_rows: int = Field(..., description="Number of data rows")
    n_columns: int = Field(..., description="Number of data columns")
    missing_values: Dict[str, int] = Field(default_factory=dict, description="Missing values per column")


class ExperimentConfig(BaseModel):
    """Configuration for an optimization experiment."""
    name: str = Field(..., description="Experiment name")
    description: Optional[str] = Field(default=None, description="Experiment description")
    parameter_space: ParameterSpace = Field(..., description="Parameter space definition")
    objectives: List[Objective] = Field(..., description="Optimization objectives")
    constraints: Optional[List[str]] = Field(default=None, description="Constraints")
    acquisition_function: AcquisitionFunction = Field(
        default=AcquisitionFunction.EXPECTED_IMPROVEMENT,
        description="Acquisition function"
    )
    model_type: ModelType = Field(default=ModelType.GAUSSIAN_PROCESS, description="Surrogate model type")
    n_initial_points: int = Field(default=10, description="Number of initial points")
    max_iterations: int = Field(default=100, description="Maximum iterations")
    convergence_tolerance: float = Field(default=1e-6, description="Convergence tolerance")
    random_seed: Optional[int] = Field(default=None, description="Random seed")
    parallel_evaluations: int = Field(default=1, description="Number of parallel evaluations")


# Type aliases for commonly used types
ParameterDict = Dict[str, Any]
ObjectiveDict = Dict[str, float]
ConstraintDict = Dict[str, float]
MetadataDict = Dict[str, Any]

# Function type aliases
ObjectiveFunction = Callable[[ParameterDict], ObjectiveDict]
ConstraintFunction = Callable[[ParameterDict], ConstraintDict]
