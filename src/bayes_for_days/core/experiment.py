"""
Experiment management for Bayes For Days platform.

This module handles the definition, execution, and management of
optimization experiments, including result tracking and persistence.
"""

from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from uuid import uuid4
import json
from pathlib import Path

from bayes_for_days.core.base import BaseModel
from bayes_for_days.core.types import (
    ExperimentPoint,
    ExperimentConfig,
    OptimizationResult,
    OptimizationStatus,
    ParameterDict,
    ObjectiveDict,
    MetadataDict,
)


class ExperimentResult(BaseModel):
    """
    Container for experiment results with metadata and persistence.
    """
    
    id: str
    config: ExperimentConfig
    optimization_result: OptimizationResult
    created_at: datetime
    completed_at: Optional[datetime] = None
    metadata: Optional[MetadataDict] = None
    
    def save_to_file(self, file_path: str) -> None:
        """Save experiment result to JSON file."""
        data = self.dict()
        # Convert datetime objects to ISO format strings
        data['created_at'] = self.created_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'ExperimentResult':
        """Load experiment result from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert ISO format strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        
        return cls(**data)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the experiment results."""
        result = self.optimization_result
        
        summary = {
            'experiment_id': self.id,
            'experiment_name': self.config.name,
            'status': result.status,
            'n_iterations': result.n_iterations,
            'n_objectives': len(self.config.objectives),
            'n_parameters': self.config.parameter_space.get_dimension(),
            'execution_time': result.execution_time,
            'created_at': self.created_at,
            'completed_at': self.completed_at,
        }
        
        # Add best point information
        if result.best_point:
            summary['best_objectives'] = result.best_point.objectives
            summary['best_parameters'] = result.best_point.parameters
        
        # Add Pareto front information for multi-objective
        if result.pareto_front:
            summary['pareto_front_size'] = len(result.pareto_front)
        
        return summary


class Experiment:
    """
    Main experiment class for managing optimization runs.
    
    This class orchestrates the entire optimization process, from
    initialization through execution to result collection.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.id = str(uuid4())
        self.created_at = datetime.now()
        self.status = OptimizationStatus.NOT_STARTED
        self.history: List[ExperimentPoint] = []
        self.current_iteration = 0
        self.optimizer: Optional[Any] = None  # Will be set during run
        self.metadata: Dict[str, Any] = {}
        
        # Callbacks for monitoring
        self.iteration_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []
    
    def add_iteration_callback(self, callback: Callable) -> None:
        """Add callback to be called after each iteration."""
        self.iteration_callbacks.append(callback)
    
    def add_completion_callback(self, callback: Callable) -> None:
        """Add callback to be called when experiment completes."""
        self.completion_callbacks.append(callback)
    
    def add_initial_data(self, data: List[ExperimentPoint]) -> None:
        """
        Add initial experimental data.
        
        Args:
            data: Initial experimental points
        """
        self.history.extend(data)
        self.metadata['n_initial_points'] = len(data)
    
    def evaluate_objective(
        self, 
        parameters: ParameterDict,
        objective_function: Callable[[ParameterDict], ObjectiveDict]
    ) -> ExperimentPoint:
        """
        Evaluate objective function at given parameters.
        
        Args:
            parameters: Parameter values
            objective_function: Function to evaluate
            
        Returns:
            Experiment point with results
        """
        try:
            objectives = objective_function(parameters)
            
            # Validate that all configured objectives are present
            expected_objectives = {obj.name for obj in self.config.objectives}
            actual_objectives = set(objectives.keys())
            
            if not expected_objectives.issubset(actual_objectives):
                missing = expected_objectives - actual_objectives
                raise ValueError(f"Missing objectives: {missing}")
            
            point = ExperimentPoint(
                id=str(uuid4()),
                parameters=parameters,
                objectives=objectives,
                timestamp=datetime.now(),
                is_feasible=True,  # TODO: Add constraint checking
            )
            
            self.history.append(point)
            return point
            
        except Exception as e:
            # Create a failed evaluation point
            point = ExperimentPoint(
                id=str(uuid4()),
                parameters=parameters,
                objectives={obj.name: float('inf') for obj in self.config.objectives},
                timestamp=datetime.now(),
                is_feasible=False,
                metadata={'error': str(e)},
            )
            
            self.history.append(point)
            return point
    
    def run(
        self, 
        objective_function: Callable[[ParameterDict], ObjectiveDict],
        optimizer: Optional[Any] = None
    ) -> ExperimentResult:
        """
        Run the optimization experiment.
        
        Args:
            objective_function: Function to optimize
            optimizer: Optional custom optimizer (will create default if None)
            
        Returns:
            Experiment results
        """
        from bayes_for_days.core.optimizer import BayesianOptimizer
        
        self.status = OptimizationStatus.RUNNING
        start_time = datetime.now()
        
        try:
            # Use provided optimizer or create default
            if optimizer is None:
                optimizer = BayesianOptimizer(self.config)
            
            self.optimizer = optimizer
            
            # Run optimization
            optimization_result = optimizer.optimize(
                objective_function=objective_function,
                initial_data=self.history.copy() if self.history else None
            )
            
            # Update status based on result
            self.status = optimization_result.status
            
            # Create experiment result
            end_time = datetime.now()
            result = ExperimentResult(
                id=self.id,
                config=self.config,
                optimization_result=optimization_result,
                created_at=self.created_at,
                completed_at=end_time,
                metadata=self.metadata,
            )
            
            # Call completion callbacks
            for callback in self.completion_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    print(f"Warning: Completion callback failed: {e}")
            
            return result
            
        except Exception as e:
            self.status = OptimizationStatus.FAILED
            self.metadata['error'] = str(e)
            
            # Create failed result
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            optimization_result = OptimizationResult(
                best_point=self.history[-1] if self.history else None,
                all_points=self.history,
                n_iterations=self.current_iteration,
                status=OptimizationStatus.FAILED,
                execution_time=execution_time,
                metadata={'error': str(e)},
            )
            
            result = ExperimentResult(
                id=self.id,
                config=self.config,
                optimization_result=optimization_result,
                created_at=self.created_at,
                completed_at=end_time,
                metadata=self.metadata,
            )
            
            return result
    
    def get_current_best(self) -> Optional[ExperimentPoint]:
        """Get the current best point."""
        if not self.history:
            return None
        
        # For single objective
        if len(self.config.objectives) == 1:
            objective = self.config.objectives[0]
            feasible_points = [p for p in self.history if p.is_feasible]
            
            if not feasible_points:
                return None
            
            if objective.type.value == "minimize":
                return min(feasible_points, key=lambda p: p.objectives[objective.name])
            else:
                return max(feasible_points, key=lambda p: p.objectives[objective.name])
        
        # For multi-objective, return first point from Pareto front
        pareto_front = self._compute_pareto_front()
        return pareto_front[0] if pareto_front else None
    
    def _compute_pareto_front(self) -> List[ExperimentPoint]:
        """Compute Pareto front from current history."""
        feasible_points = [p for p in self.history if p.is_feasible]
        
        if not feasible_points:
            return []
        
        pareto_front = []
        for point in feasible_points:
            is_dominated = False
            for other in feasible_points:
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
    
    def get_status(self) -> Dict[str, Any]:
        """Get current experiment status."""
        return {
            'id': self.id,
            'name': self.config.name,
            'status': self.status,
            'current_iteration': self.current_iteration,
            'max_iterations': self.config.max_iterations,
            'n_evaluations': len(self.history),
            'n_feasible': len([p for p in self.history if p.is_feasible]),
            'created_at': self.created_at,
            'current_best': self.get_current_best(),
        }
