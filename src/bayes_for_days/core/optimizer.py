"""
Main optimizer implementation for Bayes For Days platform.

This module contains the primary BayesianOptimizer class that orchestrates
the optimization process using surrogate models and acquisition functions.
"""

from typing import List, Dict, Any, Optional, Callable
import time
import numpy as np
from datetime import datetime

from bayes_for_days.core.base import BaseOptimizer
from bayes_for_days.core.types import (
    ExperimentPoint,
    ExperimentConfig,
    OptimizationResult,
    OptimizationStatus,
    ParameterDict,
    ObjectiveDict,
    AcquisitionFunction,
    ModelType,
)


class BayesianOptimizer(BaseOptimizer):
    """
    Main Bayesian optimization implementation.
    
    This class implements the core Bayesian optimization loop with
    support for multi-objective optimization, ensemble models,
    and adaptive acquisition function selection.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            config: Experiment configuration
        """
        super().__init__(config)
        
        self.surrogate_model = None
        self.acquisition_function = None
        self.convergence_history: List[float] = []
        self.hypervolume_history: List[float] = []
        
        # Initialize components based on configuration
        self._initialize_surrogate_model()
        self._initialize_acquisition_function()
    
    def _initialize_surrogate_model(self) -> None:
        """Initialize the surrogate model based on configuration."""
        from bayes_for_days.models.gaussian_process import GaussianProcessModel
        from bayes_for_days.models.random_forest import RandomForestModel
        from bayes_for_days.models.neural_network import NeuralNetworkModel
        from bayes_for_days.models.ensemble import EnsembleModel
        
        model_type = self.config.model_type
        
        if model_type == ModelType.GAUSSIAN_PROCESS:
            self.surrogate_model = GaussianProcessModel(
                parameter_space=self.config.parameter_space
            )
        elif model_type == ModelType.RANDOM_FOREST:
            self.surrogate_model = RandomForestModel(
                parameter_space=self.config.parameter_space
            )
        elif model_type == ModelType.NEURAL_NETWORK:
            self.surrogate_model = NeuralNetworkModel(
                parameter_space=self.config.parameter_space
            )
        elif model_type == ModelType.ENSEMBLE:
            self.surrogate_model = EnsembleModel(
                parameter_space=self.config.parameter_space,
                models=['gaussian_process', 'random_forest', 'neural_network']
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _initialize_acquisition_function(self) -> None:
        """Initialize the acquisition function based on configuration."""
        from bayes_for_days.optimization.acquisition import (
            ExpectedImprovement,
            UpperConfidenceBound,
            ProbabilityOfImprovement,
            ExpectedParetoImprovement,
            HypervolumeImprovement,
        )
        
        acq_func_type = self.config.acquisition_function
        
        if acq_func_type == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            self.acquisition_function = ExpectedImprovement(
                surrogate_model=self.surrogate_model
            )
        elif acq_func_type == AcquisitionFunction.UPPER_CONFIDENCE_BOUND:
            self.acquisition_function = UpperConfidenceBound(
                surrogate_model=self.surrogate_model
            )
        elif acq_func_type == AcquisitionFunction.PROBABILITY_OF_IMPROVEMENT:
            self.acquisition_function = ProbabilityOfImprovement(
                surrogate_model=self.surrogate_model
            )
        elif acq_func_type == AcquisitionFunction.EXPECTED_PARETO_IMPROVEMENT:
            self.acquisition_function = ExpectedParetoImprovement(
                surrogate_model=self.surrogate_model
            )
        elif acq_func_type == AcquisitionFunction.HYPERVOLUME_IMPROVEMENT:
            self.acquisition_function = HypervolumeImprovement(
                surrogate_model=self.surrogate_model
            )
        else:
            raise ValueError(f"Unsupported acquisition function: {acq_func_type}")
    
    def optimize(
        self, 
        objective_function: Callable[[ParameterDict], ObjectiveDict],
        initial_data: Optional[List[ExperimentPoint]] = None
    ) -> OptimizationResult:
        """
        Run the Bayesian optimization loop.
        
        Args:
            objective_function: Function to optimize
            initial_data: Optional initial experimental data
            
        Returns:
            Optimization results
        """
        start_time = time.time()
        self.is_running = True
        
        try:
            # Initialize with initial data or generate initial points
            if initial_data:
                self.history = initial_data.copy()
            else:
                self._generate_initial_points(objective_function)
            
            # Fit initial surrogate model
            if self.history:
                self.surrogate_model.fit(self.history)
            
            # Main optimization loop
            for iteration in range(self.config.max_iterations):
                self.current_iteration = iteration
                
                # Check convergence
                if self._check_convergence():
                    status = OptimizationStatus.CONVERGED
                    break
                
                # Suggest next point(s)
                next_points = self.suggest_next_points(
                    n_points=self.config.parallel_evaluations
                )
                
                # Evaluate objective function
                new_points = []
                for params in next_points:
                    try:
                        objectives = objective_function(params)
                        point = ExperimentPoint(
                            parameters=params,
                            objectives=objectives,
                            timestamp=datetime.now(),
                            is_feasible=True,  # TODO: Add constraint checking
                        )
                        new_points.append(point)
                    except Exception as e:
                        # Handle failed evaluations
                        point = ExperimentPoint(
                            parameters=params,
                            objectives={obj.name: float('inf') for obj in self.config.objectives},
                            timestamp=datetime.now(),
                            is_feasible=False,
                            metadata={'error': str(e)},
                        )
                        new_points.append(point)
                
                # Add new points to history
                self.add_observations(new_points)
                
                # Update surrogate model
                self.surrogate_model.update(new_points)
                
                # Update convergence tracking
                self._update_convergence_tracking()
                
                # Check for early stopping conditions
                if self._should_stop_early():
                    status = OptimizationStatus.STOPPED
                    break
            else:
                status = OptimizationStatus.MAX_ITERATIONS
            
            # Compute final results
            execution_time = time.time() - start_time
            best_point = self.get_best_point()
            pareto_front = self._compute_pareto_front() if len(self.config.objectives) > 1 else None
            
            result = OptimizationResult(
                best_point=best_point,
                pareto_front=pareto_front,
                all_points=self.history,
                n_iterations=self.current_iteration + 1,
                status=status,
                convergence_history=self.convergence_history,
                hypervolume_history=self.hypervolume_history if pareto_front else None,
                execution_time=execution_time,
                metadata={
                    'model_type': self.config.model_type,
                    'acquisition_function': self.config.acquisition_function,
                    'n_objectives': len(self.config.objectives),
                    'n_parameters': self.config.parameter_space.get_dimension(),
                }
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Return failed result
            result = OptimizationResult(
                best_point=self.get_best_point(),
                pareto_front=None,
                all_points=self.history,
                n_iterations=self.current_iteration,
                status=OptimizationStatus.FAILED,
                convergence_history=self.convergence_history,
                execution_time=execution_time,
                metadata={'error': str(e)},
            )
            
            return result
        
        finally:
            self.is_running = False
    
    def suggest_next_points(self, n_points: int = 1) -> List[ParameterDict]:
        """
        Suggest next points to evaluate using acquisition function.
        
        Args:
            n_points: Number of points to suggest
            
        Returns:
            List of suggested parameter dictionaries
        """
        if not self.surrogate_model.is_fitted:
            # If model not fitted, use random sampling
            return self._generate_random_points(n_points)
        
        # Use acquisition function to find optimal points
        bounds = self.config.parameter_space.get_bounds()
        suggested_points = self.acquisition_function.optimize(
            n_candidates=n_points,
            bounds=bounds
        )
        
        return suggested_points
    
    def _generate_initial_points(self, objective_function: Callable) -> None:
        """Generate initial points using Latin Hypercube Sampling."""
        from bayes_for_days.utils.sampling import latin_hypercube_sampling
        
        n_points = self.config.n_initial_points
        bounds = self.config.parameter_space.get_bounds()
        
        # Generate initial parameter sets
        initial_params = latin_hypercube_sampling(
            bounds=bounds,
            n_samples=n_points,
            random_seed=self.config.random_seed
        )
        
        # Evaluate initial points
        for params_array in initial_params:
            # Convert array to parameter dictionary
            params_dict = {}
            for i, param in enumerate(self.config.parameter_space.parameters):
                params_dict[param.name] = params_array[i]
            
            try:
                objectives = objective_function(params_dict)
                point = ExperimentPoint(
                    parameters=params_dict,
                    objectives=objectives,
                    timestamp=datetime.now(),
                    is_feasible=True,
                )
                self.history.append(point)
            except Exception as e:
                # Handle failed initial evaluations
                point = ExperimentPoint(
                    parameters=params_dict,
                    objectives={obj.name: float('inf') for obj in self.config.objectives},
                    timestamp=datetime.now(),
                    is_feasible=False,
                    metadata={'error': str(e)},
                )
                self.history.append(point)
    
    def _generate_random_points(self, n_points: int) -> List[ParameterDict]:
        """Generate random points within parameter bounds."""
        bounds = self.config.parameter_space.get_bounds()
        random_points = []
        
        for _ in range(n_points):
            params_dict = {}
            for i, param in enumerate(self.config.parameter_space.parameters):
                if param.bounds:
                    low, high = param.bounds
                    value = np.random.uniform(low, high)
                    params_dict[param.name] = value
                else:
                    # Handle categorical parameters
                    if param.categories:
                        value = np.random.choice(param.categories)
                        params_dict[param.name] = value
            
            random_points.append(params_dict)
        
        return random_points
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        if len(self.convergence_history) < 5:
            return False
        
        # Check if improvement has stagnated
        recent_improvements = self.convergence_history[-5:]
        improvement_variance = np.var(recent_improvements)
        
        return improvement_variance < self.config.convergence_tolerance
    
    def _update_convergence_tracking(self) -> None:
        """Update convergence tracking metrics."""
        if not self.history:
            return
        
        # For single objective, track best objective value
        if len(self.config.objectives) == 1:
            best_point = self.get_best_point()
            if best_point:
                objective_name = self.config.objectives[0].name
                best_value = best_point.objectives[objective_name]
                self.convergence_history.append(best_value)
        
        # For multi-objective, track hypervolume
        else:
            pareto_front = self._compute_pareto_front()
            if pareto_front:
                hypervolume = self._compute_hypervolume(pareto_front)
                self.hypervolume_history.append(hypervolume)
                self.convergence_history.append(hypervolume)
    
    def _compute_hypervolume(self, pareto_front: List[ExperimentPoint]) -> float:
        """Compute hypervolume of Pareto front."""
        # Simplified hypervolume calculation
        # In practice, would use more sophisticated algorithms
        if not pareto_front:
            return 0.0
        
        # Use reference point from config or compute from data
        ref_point = self.config.hypervolume_ref_point
        if ref_point is None:
            # Compute reference point as worst values in each objective
            ref_point = []
            for objective in self.config.objectives:
                obj_name = objective.name
                values = [p.objectives[obj_name] for p in self.history if p.is_feasible]
                if values:
                    if objective.type.value == "minimize":
                        ref_point.append(max(values) * 1.1)
                    else:
                        ref_point.append(min(values) * 0.9)
                else:
                    ref_point.append(0.0)
        
        # Simple hypervolume approximation
        volume = 0.0
        for point in pareto_front:
            point_volume = 1.0
            for i, objective in enumerate(self.config.objectives):
                obj_name = objective.name
                obj_value = point.objectives[obj_name]
                
                if objective.type.value == "minimize":
                    contribution = max(0, ref_point[i] - obj_value)
                else:
                    contribution = max(0, obj_value - ref_point[i])
                
                point_volume *= contribution
            
            volume += point_volume
        
        return volume
    
    def _should_stop_early(self) -> bool:
        """Check if optimization should stop early."""
        # Could implement various early stopping criteria
        # For now, just return False
        return False
