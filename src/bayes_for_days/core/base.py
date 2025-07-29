"""
Base classes and abstract interfaces for Bayes For Days platform.

This module defines the fundamental abstract base classes that establish
the interface contracts for all major components of the system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
from pydantic import BaseModel as PydanticBaseModel

from bayes_for_days.core.types import (
    ExperimentPoint,
    OptimizationResult,
    ModelPrediction,
    AcquisitionValue,
    ParameterSpace,
    ExperimentConfig,
    ValidationResult,
    ParameterDict,
    ObjectiveDict,
)


class BaseModel(PydanticBaseModel):
    """
    Base model class with common configuration.
    
    Extends Pydantic BaseModel with custom configuration
    for the Bayes For Days platform.
    """
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        validate_assignment = True
        use_enum_values = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist(),
            np.integer: lambda v: int(v),
            np.floating: lambda v: float(v),
        }


class BaseSurrogateModel(ABC):
    """
    Abstract base class for surrogate models.
    
    All surrogate models (GP, RF, NN, etc.) must implement this interface.
    """
    
    def __init__(self, parameter_space: ParameterSpace, **kwargs):
        """Initialize the surrogate model."""
        self.parameter_space = parameter_space
        self.is_fitted = False
        self.training_data: List[ExperimentPoint] = []
    
    @abstractmethod
    def fit(self, data: List[ExperimentPoint]) -> None:
        """
        Fit the surrogate model to experimental data.
        
        Args:
            data: List of experimental points with parameters and objectives
        """
        pass
    
    @abstractmethod
    def predict(
        self, 
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[ModelPrediction, List[ModelPrediction]]:
        """
        Make predictions at given parameter values.
        
        Args:
            parameters: Parameter values for prediction
            
        Returns:
            Model predictions with uncertainty estimates
        """
        pass
    
    @abstractmethod
    def update(self, new_data: List[ExperimentPoint]) -> None:
        """
        Update the model with new experimental data.
        
        Args:
            new_data: New experimental points to incorporate
        """
        pass
    
    def get_training_data(self) -> List[ExperimentPoint]:
        """Get the current training data."""
        return self.training_data.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "type": self.__class__.__name__,
            "is_fitted": self.is_fitted,
            "n_training_points": len(self.training_data),
            "parameter_space_dim": self.parameter_space.get_dimension(),
        }


class BaseAcquisitionFunction(ABC):
    """
    Abstract base class for acquisition functions.
    
    All acquisition functions must implement this interface.
    """
    
    def __init__(self, surrogate_model: BaseSurrogateModel, **kwargs):
        """Initialize the acquisition function."""
        self.surrogate_model = surrogate_model
        self.kwargs = kwargs
    
    @abstractmethod
    def evaluate(
        self, 
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate the acquisition function at given parameters.
        
        Args:
            parameters: Parameter values for evaluation
            
        Returns:
            Acquisition function values
        """
        pass
    
    @abstractmethod
    def optimize(
        self, 
        n_candidates: int = 1,
        bounds: Optional[List[tuple]] = None,
        **kwargs
    ) -> List[ParameterDict]:
        """
        Optimize the acquisition function to find next evaluation points.
        
        Args:
            n_candidates: Number of candidate points to return
            bounds: Parameter bounds for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            List of optimal parameter dictionaries
        """
        pass


class BaseOptimizer(ABC):
    """
    Abstract base class for optimization algorithms.
    
    All optimizers (Bayesian, genetic, hybrid) must implement this interface.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize the optimizer."""
        self.config = config
        self.history: List[ExperimentPoint] = []
        self.is_running = False
        self.current_iteration = 0
    
    @abstractmethod
    def optimize(
        self, 
        objective_function: callable,
        initial_data: Optional[List[ExperimentPoint]] = None
    ) -> OptimizationResult:
        """
        Run the optimization algorithm.
        
        Args:
            objective_function: Function to optimize
            initial_data: Optional initial experimental data
            
        Returns:
            Optimization results
        """
        pass
    
    @abstractmethod
    def suggest_next_points(self, n_points: int = 1) -> List[ParameterDict]:
        """
        Suggest next points to evaluate.
        
        Args:
            n_points: Number of points to suggest
            
        Returns:
            List of suggested parameter dictionaries
        """
        pass
    
    def add_observations(self, points: List[ExperimentPoint]) -> None:
        """
        Add new observations to the optimizer.
        
        Args:
            points: New experimental points
        """
        self.history.extend(points)
    
    def get_best_point(self) -> Optional[ExperimentPoint]:
        """Get the current best point."""
        if not self.history:
            return None
        
        # For single objective, return point with best objective value
        if len(self.config.objectives) == 1:
            objective_name = self.config.objectives[0].name
            objective_type = self.config.objectives[0].type
            
            if objective_type.value == "minimize":
                return min(self.history, key=lambda p: p.objectives[objective_name])
            else:
                return max(self.history, key=lambda p: p.objectives[objective_name])
        
        # For multi-objective, return a representative point from Pareto front
        pareto_front = self._compute_pareto_front()
        return pareto_front[0] if pareto_front else None
    
    def _compute_pareto_front(self) -> List[ExperimentPoint]:
        """Compute the Pareto front from current history."""
        if not self.history:
            return []
        
        # Simple Pareto front computation
        pareto_front = []
        for point in self.history:
            is_dominated = False
            for other in self.history:
                if self._dominates(other, point):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(point)
        
        return pareto_front
    
    def _dominates(self, point1: ExperimentPoint, point2: ExperimentPoint) -> bool:
        """Check if point1 dominates point2."""
        better_in_all = True
        better_in_at_least_one = False
        
        for objective in self.config.objectives:
            obj_name = objective.name
            obj_type = objective.type
            
            val1 = point1.objectives[obj_name]
            val2 = point2.objectives[obj_name]
            
            if obj_type.value == "minimize":
                if val1 > val2:
                    better_in_all = False
                elif val1 < val2:
                    better_in_at_least_one = True
            else:  # maximize
                if val1 < val2:
                    better_in_all = False
                elif val1 > val2:
                    better_in_at_least_one = True
        
        return better_in_all and better_in_at_least_one


class BaseDataManager(ABC):
    """
    Abstract base class for data management.
    
    Handles data import, export, validation, and preprocessing.
    """
    
    @abstractmethod
    def load_data(self, file_path: str, **kwargs) -> List[ExperimentPoint]:
        """
        Load experimental data from file.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional loading parameters
            
        Returns:
            List of experimental points
        """
        pass
    
    @abstractmethod
    def save_data(
        self, 
        data: List[ExperimentPoint], 
        file_path: str, 
        **kwargs
    ) -> None:
        """
        Save experimental data to file.
        
        Args:
            data: Experimental data to save
            file_path: Output file path
            **kwargs: Additional saving parameters
        """
        pass
    
    @abstractmethod
    def validate_data(
        self, 
        data: List[ExperimentPoint],
        parameter_space: ParameterSpace
    ) -> ValidationResult:
        """
        Validate experimental data.
        
        Args:
            data: Data to validate
            parameter_space: Expected parameter space
            
        Returns:
            Validation results
        """
        pass
    
    @abstractmethod
    def preprocess_data(
        self, 
        data: List[ExperimentPoint],
        **kwargs
    ) -> List[ExperimentPoint]:
        """
        Preprocess experimental data.
        
        Args:
            data: Raw experimental data
            **kwargs: Preprocessing parameters
            
        Returns:
            Preprocessed experimental data
        """
        pass
