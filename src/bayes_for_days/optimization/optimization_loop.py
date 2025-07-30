"""
Basic optimization loop integration for Bayesian optimization.

This module implements the complete optimization loop that integrates:
- Gaussian Process surrogate models
- Acquisition functions for experiment selection
- Experiment scheduling and execution
- Convergence detection and stopping criteria
- Result tracking and analysis

Based on:
- Jones et al. (1998) "Efficient Global Optimization of Expensive Black-Box Functions"
- Shahriari et al. (2016) "Taking the Human Out of the Loop: A Review of Bayesian Optimization"
- Modern Bayesian optimization best practices
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import time

from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    OptimizationResult,
    AcquisitionFunction as AcquisitionFunctionType,
)
from bayes_for_days.core.base import BaseAcquisitionFunction, BaseSurrogateModel
from bayes_for_days.models.gaussian_process import GaussianProcessModel
from bayes_for_days.acquisition.expected_improvement import ExpectedImprovement
from bayes_for_days.acquisition.upper_confidence_bound import UpperConfidenceBound

logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """
    Configuration for Bayesian optimization loop.
    
    Defines all parameters needed to run a complete
    optimization campaign.
    """
    # Basic parameters
    max_iterations: int = 100
    initial_experiments: int = 5
    batch_size: int = 1
    
    # Convergence criteria
    convergence_tolerance: float = 1e-6
    max_iterations_without_improvement: int = 10
    improvement_threshold: float = 1e-4
    
    # Resource constraints
    max_budget: Optional[float] = None
    max_time_hours: Optional[float] = None
    
    # Acquisition function parameters
    acquisition_function: str = "expected_improvement"
    acquisition_params: Dict[str, Any] = field(default_factory=dict)
    
    # Model parameters
    model_type: str = "gaussian_process"
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Experiment scheduling
    parallel_experiments: bool = False
    experiment_timeout_minutes: float = 60.0
    
    # Logging and output
    verbose: bool = True
    save_intermediate_results: bool = True
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        
        if self.initial_experiments <= 0:
            raise ValueError("initial_experiments must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


class ConvergenceDetector:
    """
    Convergence detection for optimization loops.
    
    Monitors optimization progress and determines when
    to terminate based on various criteria.
    """
    
    def __init__(
        self,
        tolerance: float = 1e-6,
        max_iterations_without_improvement: int = 10,
        improvement_threshold: float = 1e-4,
        **kwargs
    ):
        """
        Initialize convergence detector.
        
        Args:
            tolerance: Absolute tolerance for convergence
            max_iterations_without_improvement: Max iterations without improvement
            improvement_threshold: Minimum improvement to count as progress
            **kwargs: Additional parameters
        """
        self.tolerance = tolerance
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.improvement_threshold = improvement_threshold
        
        # State tracking
        self.best_value = None
        self.iterations_without_improvement = 0
        self.objective_history: List[float] = []
        self.improvement_history: List[float] = []
        
        logger.info("Initialized convergence detector")
    
    def update(self, new_objective_value: float) -> bool:
        """
        Update convergence detector with new objective value.
        
        Args:
            new_objective_value: New objective function value
            
        Returns:
            True if convergence is detected
        """
        self.objective_history.append(new_objective_value)
        
        # Check for improvement
        if self.best_value is None or new_objective_value > self.best_value:
            improvement = (new_objective_value - self.best_value 
                          if self.best_value is not None else new_objective_value)
            
            if improvement >= self.improvement_threshold:
                self.best_value = new_objective_value
                self.iterations_without_improvement = 0
                self.improvement_history.append(improvement)
                logger.debug(f"Improvement detected: {improvement:.6f}")
            else:
                self.iterations_without_improvement += 1
        else:
            self.iterations_without_improvement += 1
        
        # Check convergence criteria
        return self._check_convergence()
    
    def _check_convergence(self) -> bool:
        """Check if convergence criteria are met."""
        # No improvement criterion
        if self.iterations_without_improvement >= self.max_iterations_without_improvement:
            logger.info(f"Convergence: {self.iterations_without_improvement} iterations without improvement")
            return True
        
        # Objective value stability
        if len(self.objective_history) >= 5:
            recent_values = self.objective_history[-5:]
            if max(recent_values) - min(recent_values) < self.tolerance:
                logger.info("Convergence: Objective values stabilized")
                return True
        
        return False
    
    def get_convergence_info(self) -> Dict[str, Any]:
        """Get information about convergence status."""
        return {
            'best_value': self.best_value,
            'iterations_without_improvement': self.iterations_without_improvement,
            'total_iterations': len(self.objective_history),
            'total_improvement': sum(self.improvement_history),
            'is_converged': self._check_convergence(),
            'objective_history': self.objective_history.copy(),
        }


class ExperimentScheduler:
    """
    Scheduler for managing experiment execution.
    
    Handles experiment queuing, execution, and result collection
    with support for parallel execution and timeouts.
    """
    
    def __init__(
        self,
        parallel_experiments: bool = False,
        experiment_timeout_minutes: float = 60.0,
        **kwargs
    ):
        """
        Initialize experiment scheduler.
        
        Args:
            parallel_experiments: Whether to run experiments in parallel
            experiment_timeout_minutes: Timeout for individual experiments
            **kwargs: Additional parameters
        """
        self.parallel_experiments = parallel_experiments
        self.experiment_timeout_minutes = experiment_timeout_minutes
        self.config = kwargs
        
        # Experiment queue and results
        self.pending_experiments: List[Dict[str, float]] = []
        self.running_experiments: List[Dict[str, Any]] = []
        self.completed_experiments: List[ExperimentPoint] = []
        
        logger.info(f"Initialized experiment scheduler (parallel: {parallel_experiments})")
    
    def schedule_experiments(self, experiment_params: List[Dict[str, float]]):
        """
        Schedule experiments for execution.
        
        Args:
            experiment_params: List of parameter dictionaries for experiments
        """
        self.pending_experiments.extend(experiment_params)
        logger.info(f"Scheduled {len(experiment_params)} experiments")
    
    def execute_experiments(
        self,
        objective_function: Callable[[Dict[str, float]], float]
    ) -> List[ExperimentPoint]:
        """
        Execute scheduled experiments.
        
        Args:
            objective_function: Function to evaluate experiments
            
        Returns:
            List of completed experiment results
        """
        if not self.pending_experiments:
            return []
        
        results = []
        
        if self.parallel_experiments:
            results = self._execute_parallel(objective_function)
        else:
            results = self._execute_sequential(objective_function)
        
        self.completed_experiments.extend(results)
        self.pending_experiments.clear()
        
        logger.info(f"Completed {len(results)} experiments")
        
        return results
    
    def _execute_sequential(
        self,
        objective_function: Callable[[Dict[str, float]], float]
    ) -> List[ExperimentPoint]:
        """Execute experiments sequentially."""
        results = []
        
        for i, params in enumerate(self.pending_experiments):
            try:
                start_time = time.time()
                
                # Execute experiment
                objective_value = objective_function(params)
                
                execution_time = time.time() - start_time
                
                # Create experiment point
                experiment_point = ExperimentPoint(
                    parameters=params,
                    objectives={"objective": objective_value},
                    is_feasible=True,
                    metadata={
                        "execution_time_seconds": execution_time,
                        "experiment_index": i,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                
                results.append(experiment_point)
                
                logger.debug(f"Experiment {i+1}/{len(self.pending_experiments)} completed: {objective_value:.6f}")
                
            except Exception as e:
                logger.error(f"Experiment {i+1} failed: {e}")
                
                # Create failed experiment point
                experiment_point = ExperimentPoint(
                    parameters=params,
                    objectives={"objective": float('-inf')},
                    is_feasible=False,
                    metadata={
                        "error": str(e),
                        "experiment_index": i,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                
                results.append(experiment_point)
        
        return results
    
    def _execute_parallel(
        self,
        objective_function: Callable[[Dict[str, float]], float]
    ) -> List[ExperimentPoint]:
        """Execute experiments in parallel (simplified implementation)."""
        # For now, fall back to sequential execution
        # In practice, would use ThreadPoolExecutor or similar
        logger.warning("Parallel execution not fully implemented, using sequential")
        return self._execute_sequential(objective_function)
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        return {
            'pending_experiments': len(self.pending_experiments),
            'running_experiments': len(self.running_experiments),
            'completed_experiments': len(self.completed_experiments),
            'parallel_experiments': self.parallel_experiments,
        }


class BayesianOptimizationLoop:
    """
    Complete Bayesian optimization loop implementation.
    
    Integrates surrogate models, acquisition functions, experiment
    scheduling, and convergence detection in a unified framework.
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        objective_function: Callable[[Dict[str, float]], float],
        config: OptimizationConfig,
        surrogate_model: Optional[BaseSurrogateModel] = None,
        acquisition_function: Optional[BaseAcquisitionFunction] = None,
        **kwargs
    ):
        """
        Initialize Bayesian optimization loop.
        
        Args:
            parameter_space: Parameter space definition
            objective_function: Objective function to optimize
            config: Optimization configuration
            surrogate_model: Surrogate model (optional, will create default)
            acquisition_function: Acquisition function (optional, will create default)
            **kwargs: Additional parameters
        """
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        self.config = config
        
        # Initialize surrogate model
        if surrogate_model is None:
            self.surrogate_model = self._create_default_surrogate_model()
        else:
            self.surrogate_model = surrogate_model
        
        # Initialize acquisition function
        if acquisition_function is None:
            self.acquisition_function = self._create_default_acquisition_function()
        else:
            self.acquisition_function = acquisition_function
        
        # Initialize components
        self.convergence_detector = ConvergenceDetector(
            tolerance=config.convergence_tolerance,
            max_iterations_without_improvement=config.max_iterations_without_improvement,
            improvement_threshold=config.improvement_threshold
        )
        
        self.experiment_scheduler = ExperimentScheduler(
            parallel_experiments=config.parallel_experiments,
            experiment_timeout_minutes=config.experiment_timeout_minutes
        )
        
        # Optimization state
        self.iteration = 0
        self.is_converged = False
        self.all_experiments: List[ExperimentPoint] = []
        self.best_experiment: Optional[ExperimentPoint] = None
        self.optimization_start_time: Optional[datetime] = None
        
        logger.info("Initialized Bayesian optimization loop")
    
    def _create_default_surrogate_model(self) -> BaseSurrogateModel:
        """Create default surrogate model."""
        model_params = self.config.model_params.copy()
        model_params.setdefault('n_inducing_points', 50)
        
        return GaussianProcessModel(
            parameter_space=self.parameter_space,
            **model_params
        )
    
    def _create_default_acquisition_function(self) -> BaseAcquisitionFunction:
        """Create default acquisition function."""
        acq_params = self.config.acquisition_params.copy()
        
        if self.config.acquisition_function == "expected_improvement":
            acq_params.setdefault('xi', 0.01)
            return ExpectedImprovement(
                surrogate_model=self.surrogate_model,
                **acq_params
            )
        elif self.config.acquisition_function == "upper_confidence_bound":
            acq_params.setdefault('beta', 2.0)
            return UpperConfidenceBound(
                surrogate_model=self.surrogate_model,
                **acq_params
            )
        else:
            raise ValueError(f"Unknown acquisition function: {self.config.acquisition_function}")
    
    def optimize(self) -> OptimizationResult:
        """
        Run the complete Bayesian optimization loop.
        
        Returns:
            Optimization result with best parameters and performance metrics
        """
        self.optimization_start_time = datetime.now()
        
        try:
            # Generate initial experiments
            self._generate_initial_experiments()
            
            # Main optimization loop
            while not self._should_terminate():
                self.iteration += 1
                
                if self.config.verbose:
                    logger.info(f"Starting optimization iteration {self.iteration}")
                
                # Select next experiments using acquisition function
                next_experiments = self._select_next_experiments()
                
                # Schedule and execute experiments
                self.experiment_scheduler.schedule_experiments(next_experiments)
                new_results = self.experiment_scheduler.execute_experiments(self.objective_function)
                
                # Update optimization state
                self._update_optimization_state(new_results)
                
                # Update surrogate model
                self._update_surrogate_model()
                
                # Check convergence
                if self.best_experiment and self.best_experiment.objectives:
                    best_objective = list(self.best_experiment.objectives.values())[0]
                    self.is_converged = self.convergence_detector.update(best_objective)
                
                if self.config.verbose:
                    self._log_iteration_summary()
            
            # Create optimization result
            result = self._create_optimization_result()
            
            if self.config.verbose:
                logger.info(f"Optimization completed after {self.iteration} iterations")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _generate_initial_experiments(self):
        """Generate initial experiments using space-filling design."""
        # Simple random sampling for initial experiments
        initial_params = []
        
        for _ in range(self.config.initial_experiments):
            params = {}
            for param in self.parameter_space.parameters:
                if param.type.value == 'continuous':
                    if param.bounds:
                        low, high = param.bounds
                        value = np.random.uniform(low, high)
                    else:
                        value = np.random.normal(0, 1)
                elif param.type.value == 'categorical':
                    if param.categories:
                        value = np.random.choice(param.categories)
                    else:
                        value = "default"
                else:
                    value = 0.0
                
                params[param.name] = value
            
            initial_params.append(params)
        
        # Execute initial experiments
        self.experiment_scheduler.schedule_experiments(initial_params)
        initial_results = self.experiment_scheduler.execute_experiments(self.objective_function)
        
        # Update state
        self._update_optimization_state(initial_results)
        
        # Fit initial surrogate model
        self._update_surrogate_model()
        
        logger.info(f"Completed {len(initial_results)} initial experiments")
    
    def _select_next_experiments(self) -> List[Dict[str, float]]:
        """Select next experiments using acquisition function."""
        # Get parameter bounds for optimization
        bounds = self.parameter_space.get_bounds()
        
        # Optimize acquisition function
        next_params = self.acquisition_function.optimize(
            n_candidates=self.config.batch_size,
            bounds=bounds
        )
        
        return next_params
    
    def _update_optimization_state(self, new_experiments: List[ExperimentPoint]):
        """Update optimization state with new experiments."""
        self.all_experiments.extend(new_experiments)
        
        # Update best experiment
        for exp in new_experiments:
            if exp.objectives and exp.is_feasible:
                obj_value = list(exp.objectives.values())[0]
                
                if (self.best_experiment is None or 
                    obj_value > list(self.best_experiment.objectives.values())[0]):
                    self.best_experiment = exp
    
    def _update_surrogate_model(self):
        """Update surrogate model with all experimental data."""
        if self.all_experiments:
            self.surrogate_model.fit(self.all_experiments)
    
    def _should_terminate(self) -> bool:
        """Check if optimization should terminate."""
        # Max iterations
        if self.iteration >= self.config.max_iterations:
            logger.info("Terminating: Maximum iterations reached")
            return True
        
        # Convergence
        if self.is_converged:
            logger.info("Terminating: Convergence detected")
            return True
        
        # Time limit
        if (self.config.max_time_hours and self.optimization_start_time and
            (datetime.now() - self.optimization_start_time).total_seconds() / 3600 > self.config.max_time_hours):
            logger.info("Terminating: Time limit reached")
            return True
        
        # Budget limit (simplified)
        if (self.config.max_budget and 
            len(self.all_experiments) * 100 > self.config.max_budget):  # Assume $100 per experiment
            logger.info("Terminating: Budget exhausted")
            return True
        
        return False
    
    def _log_iteration_summary(self):
        """Log summary of current iteration."""
        if self.best_experiment and self.best_experiment.objectives:
            best_value = list(self.best_experiment.objectives.values())[0]
            logger.info(f"Iteration {self.iteration}: Best value = {best_value:.6f}, "
                       f"Total experiments = {len(self.all_experiments)}")
    
    def _create_optimization_result(self) -> OptimizationResult:
        """Create optimization result."""
        # Calculate execution time
        execution_time = (
            (datetime.now() - self.optimization_start_time).total_seconds()
            if self.optimization_start_time else 0.0
        )
        
        # Get convergence info
        convergence_info = self.convergence_detector.get_convergence_info()
        
        return OptimizationResult(
            best_parameters=self.best_experiment.parameters if self.best_experiment else {},
            best_objective_value=(
                list(self.best_experiment.objectives.values())[0] 
                if self.best_experiment and self.best_experiment.objectives else None
            ),
            n_iterations=self.iteration,
            n_function_evaluations=len(self.all_experiments),
            execution_time_seconds=execution_time,
            is_converged=self.is_converged,
            all_experiments=self.all_experiments.copy(),
            metadata={
                'convergence_info': convergence_info,
                'config': self.config.__dict__,
                'scheduler_status': self.experiment_scheduler.get_scheduler_status(),
            }
        )
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status."""
        return {
            'iteration': self.iteration,
            'total_experiments': len(self.all_experiments),
            'is_converged': self.is_converged,
            'best_objective_value': (
                list(self.best_experiment.objectives.values())[0] 
                if self.best_experiment and self.best_experiment.objectives else None
            ),
            'convergence_info': self.convergence_detector.get_convergence_info(),
            'scheduler_status': self.experiment_scheduler.get_scheduler_status(),
        }
