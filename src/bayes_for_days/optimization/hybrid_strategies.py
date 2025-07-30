"""
Hybrid optimization strategies combining Bayesian optimization with other methods.

This module implements hybrid approaches that combine the strengths of:
- Bayesian optimization with genetic algorithms
- Multi-start local optimization
- Simulated annealing integration
- Particle swarm optimization hybrids
- Gradient-based refinement

Based on:
- Jones et al. (1998) "Efficient Global Optimization"
- Hybrid optimization literature
- Modern multi-strategy optimization approaches
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from scipy.optimize import minimize, differential_evolution
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time

from bayes_for_days.core.base import BaseAcquisitionFunction, BaseSurrogateModel
from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    OptimizationResult,
)
from bayes_for_days.optimization.multi_objective import NSGAIIOptimizer

logger = logging.getLogger(__name__)


@dataclass
class HybridConfig:
    """
    Configuration for hybrid optimization strategies.
    
    Defines parameters for combining different optimization
    methods and switching criteria between them.
    """
    # Strategy selection
    primary_strategy: str = "bayesian"  # Primary optimization strategy
    secondary_strategies: List[str] = field(default_factory=lambda: ["genetic", "local"])
    
    # Switching criteria
    switch_after_iterations: int = 20
    switch_on_stagnation: bool = True
    stagnation_threshold: float = 1e-6
    stagnation_patience: int = 5
    
    # Strategy-specific parameters
    bayesian_config: Dict[str, Any] = field(default_factory=dict)
    genetic_config: Dict[str, Any] = field(default_factory=dict)
    local_config: Dict[str, Any] = field(default_factory=dict)
    
    # Resource allocation
    max_total_evaluations: int = 200
    evaluation_budget_ratios: Dict[str, float] = field(default_factory=lambda: {
        "bayesian": 0.6,
        "genetic": 0.3,
        "local": 0.1
    })
    
    # Performance tracking
    track_performance: bool = True
    performance_window: int = 10


class OptimizationStrategy(ABC):
    """
    Abstract base class for optimization strategies.
    
    Defines the interface for individual optimization
    methods that can be combined in hybrid approaches.
    """
    
    @abstractmethod
    def optimize(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameter_space: ParameterSpace,
        initial_points: Optional[List[ExperimentPoint]] = None,
        max_evaluations: int = 50,
        **kwargs
    ) -> OptimizationResult:
        """
        Run optimization strategy.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            initial_points: Initial experiment points
            max_evaluations: Maximum function evaluations
            **kwargs: Strategy-specific parameters
            
        Returns:
            Optimization result
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get name of the optimization strategy."""
        pass


class BayesianStrategy(OptimizationStrategy):
    """
    Bayesian optimization strategy wrapper.
    
    Wraps Bayesian optimization for use in hybrid frameworks.
    """
    
    def __init__(
        self,
        surrogate_model: BaseSurrogateModel,
        acquisition_function: BaseAcquisitionFunction,
        **kwargs
    ):
        """
        Initialize Bayesian strategy.
        
        Args:
            surrogate_model: Surrogate model for predictions
            acquisition_function: Acquisition function for point selection
            **kwargs: Additional parameters
        """
        self.surrogate_model = surrogate_model
        self.acquisition_function = acquisition_function
        self.config = kwargs
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameter_space: ParameterSpace,
        initial_points: Optional[List[ExperimentPoint]] = None,
        max_evaluations: int = 50,
        **kwargs
    ) -> OptimizationResult:
        """Run Bayesian optimization."""
        from bayes_for_days.optimization.optimization_loop import (
            BayesianOptimizationLoop, OptimizationConfig
        )
        
        # Create configuration
        config = OptimizationConfig(
            max_iterations=max_evaluations,
            initial_experiments=len(initial_points) if initial_points else 5,
            batch_size=1,
            verbose=False,
            **kwargs
        )
        
        # Create optimizer
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective_function,
            config=config,
            surrogate_model=self.surrogate_model,
            acquisition_function=self.acquisition_function
        )
        
        # Add initial points if provided
        if initial_points:
            optimizer.all_experiments.extend(initial_points)
            optimizer._update_surrogate_model()
        
        # Run optimization
        return optimizer.optimize()
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "bayesian"


class GeneticStrategy(OptimizationStrategy):
    """
    Genetic algorithm strategy wrapper.
    
    Uses differential evolution as the genetic algorithm implementation.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize genetic strategy.
        
        Args:
            **kwargs: Genetic algorithm parameters
        """
        self.config = kwargs
        self.config.setdefault('popsize', 15)
        self.config.setdefault('mutation', (0.5, 1))
        self.config.setdefault('recombination', 0.7)
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameter_space: ParameterSpace,
        initial_points: Optional[List[ExperimentPoint]] = None,
        max_evaluations: int = 50,
        **kwargs
    ) -> OptimizationResult:
        """Run genetic algorithm optimization."""
        # Get parameter bounds
        bounds = parameter_space.get_bounds()
        if not bounds:
            raise ValueError("Parameter bounds are required for genetic algorithm")
        
        # Convert objective function to work with arrays
        def array_objective(x):
            param_dict = {}
            param_names = [param.name for param in parameter_space.parameters]
            
            for i, param_name in enumerate(param_names):
                if i < len(x):
                    param_dict[param_name] = x[i]
            
            return -objective_function(param_dict)  # Minimize negative
        
        # Set up initial population if initial points provided
        init_population = None
        if initial_points:
            init_pop = []
            param_names = [param.name for param in parameter_space.parameters]
            
            for point in initial_points:
                individual = []
                for param_name in param_names:
                    if param_name in point.parameters:
                        individual.append(point.parameters[param_name])
                    else:
                        # Use midpoint of bounds as default
                        low, high = bounds[len(individual)]
                        individual.append((low + high) / 2)
                init_pop.append(individual)
            
            if init_pop:
                init_population = np.array(init_pop)
        
        # Run differential evolution
        start_time = time.time()
        
        try:
            result = differential_evolution(
                array_objective,
                bounds,
                maxiter=max_evaluations // self.config['popsize'],
                popsize=self.config['popsize'],
                mutation=self.config['mutation'],
                recombination=self.config['recombination'],
                init=init_population,
                **kwargs
            )
            
            # Convert result back to parameter dict
            param_names = [param.name for param in parameter_space.parameters]
            best_parameters = {
                param_names[i]: result.x[i] 
                for i in range(min(len(param_names), len(result.x)))
            }
            
            # Create optimization result
            optimization_result = OptimizationResult(
                best_parameters=best_parameters,
                best_objective_value=-result.fun,  # Convert back from minimization
                n_iterations=result.nit,
                n_function_evaluations=result.nfev,
                execution_time_seconds=time.time() - start_time,
                is_converged=result.success,
                all_experiments=[],  # DE doesn't track individual experiments
                metadata={
                    'strategy': 'genetic',
                    'algorithm': 'differential_evolution',
                    'success': result.success,
                    'message': result.message,
                }
            )
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Genetic algorithm optimization failed: {e}")
            
            # Return fallback result
            param_names = [param.name for param in parameter_space.parameters]
            fallback_params = {}
            for i, param_name in enumerate(param_names):
                if i < len(bounds):
                    low, high = bounds[i]
                    fallback_params[param_name] = (low + high) / 2
            
            return OptimizationResult(
                best_parameters=fallback_params,
                best_objective_value=float('-inf'),
                n_iterations=0,
                n_function_evaluations=0,
                execution_time_seconds=time.time() - start_time,
                is_converged=False,
                all_experiments=[],
                metadata={'strategy': 'genetic', 'error': str(e)}
            )
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "genetic"


class LocalStrategy(OptimizationStrategy):
    """
    Local optimization strategy wrapper.
    
    Uses gradient-based local optimization methods.
    """
    
    def __init__(self, method: str = "L-BFGS-B", **kwargs):
        """
        Initialize local strategy.
        
        Args:
            method: Optimization method for scipy.optimize.minimize
            **kwargs: Additional parameters for local optimizer
        """
        self.method = method
        self.config = kwargs
        self.config.setdefault('options', {'maxiter': 100})
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameter_space: ParameterSpace,
        initial_points: Optional[List[ExperimentPoint]] = None,
        max_evaluations: int = 50,
        **kwargs
    ) -> OptimizationResult:
        """Run local optimization."""
        bounds = parameter_space.get_bounds()
        if not bounds:
            raise ValueError("Parameter bounds are required for local optimization")
        
        # Convert objective function
        def array_objective(x):
            param_dict = {}
            param_names = [param.name for param in parameter_space.parameters]
            
            for i, param_name in enumerate(param_names):
                if i < len(x):
                    param_dict[param_name] = x[i]
            
            return -objective_function(param_dict)  # Minimize negative
        
        # Determine starting points
        starting_points = []
        
        if initial_points:
            param_names = [param.name for param in parameter_space.parameters]
            for point in initial_points:
                start_point = []
                for param_name in param_names:
                    if param_name in point.parameters:
                        start_point.append(point.parameters[param_name])
                    else:
                        low, high = bounds[len(start_point)]
                        start_point.append((low + high) / 2)
                starting_points.append(start_point)
        
        # Add random starting points if needed
        n_starts = max(1, max_evaluations // 20)  # Multiple starts
        while len(starting_points) < n_starts:
            random_start = []
            for low, high in bounds:
                random_start.append(np.random.uniform(low, high))
            starting_points.append(random_start)
        
        # Run multi-start local optimization
        best_result = None
        best_value = float('inf')
        total_evaluations = 0
        start_time = time.time()
        
        for i, start_point in enumerate(starting_points):
            if total_evaluations >= max_evaluations:
                break
            
            try:
                # Limit evaluations for this start
                remaining_evals = max_evaluations - total_evaluations
                local_config = self.config.copy()
                local_config['options'] = local_config.get('options', {}).copy()
                local_config['options']['maxfun'] = min(
                    remaining_evals, 
                    local_config['options'].get('maxfun', remaining_evals)
                )
                
                result = minimize(
                    array_objective,
                    start_point,
                    method=self.method,
                    bounds=bounds,
                    **local_config
                )
                
                total_evaluations += result.nfev
                
                if result.fun < best_value:
                    best_value = result.fun
                    best_result = result
                    
            except Exception as e:
                logger.warning(f"Local optimization start {i} failed: {e}")
                continue
        
        # Convert best result
        if best_result is not None:
            param_names = [param.name for param in parameter_space.parameters]
            best_parameters = {
                param_names[i]: best_result.x[i] 
                for i in range(min(len(param_names), len(best_result.x)))
            }
            
            optimization_result = OptimizationResult(
                best_parameters=best_parameters,
                best_objective_value=-best_result.fun,
                n_iterations=best_result.nit,
                n_function_evaluations=total_evaluations,
                execution_time_seconds=time.time() - start_time,
                is_converged=best_result.success,
                all_experiments=[],
                metadata={
                    'strategy': 'local',
                    'method': self.method,
                    'n_starts': len(starting_points),
                    'success': best_result.success,
                }
            )
        else:
            # Fallback result
            param_names = [param.name for param in parameter_space.parameters]
            fallback_params = {
                param_names[i]: (bounds[i][0] + bounds[i][1]) / 2
                for i in range(min(len(param_names), len(bounds)))
            }
            
            optimization_result = OptimizationResult(
                best_parameters=fallback_params,
                best_objective_value=float('-inf'),
                n_iterations=0,
                n_function_evaluations=total_evaluations,
                execution_time_seconds=time.time() - start_time,
                is_converged=False,
                all_experiments=[],
                metadata={'strategy': 'local', 'error': 'All starts failed'}
            )
        
        return optimization_result
    
    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return "local"


class HybridOptimizer:
    """
    Hybrid optimizer combining multiple optimization strategies.
    
    Coordinates the execution of different optimization methods
    with intelligent switching and resource allocation.
    """
    
    def __init__(
        self,
        strategies: Dict[str, OptimizationStrategy],
        config: HybridConfig,
        **kwargs
    ):
        """
        Initialize hybrid optimizer.
        
        Args:
            strategies: Dictionary of optimization strategies
            config: Hybrid optimization configuration
            **kwargs: Additional parameters
        """
        self.strategies = strategies
        self.config = config
        
        # Validate configuration
        if config.primary_strategy not in strategies:
            raise ValueError(f"Primary strategy '{config.primary_strategy}' not found in strategies")
        
        for strategy_name in config.secondary_strategies:
            if strategy_name not in strategies:
                raise ValueError(f"Secondary strategy '{strategy_name}' not found in strategies")
        
        # Optimization state
        self.current_strategy = config.primary_strategy
        self.strategy_history: List[str] = []
        self.performance_history: List[Dict[str, float]] = []
        self.total_evaluations = 0
        self.best_result: Optional[OptimizationResult] = None
        
        logger.info(f"Initialized hybrid optimizer with {len(strategies)} strategies")
    
    def optimize(
        self,
        objective_function: Callable[[Dict[str, float]], float],
        parameter_space: ParameterSpace,
        **kwargs
    ) -> OptimizationResult:
        """
        Run hybrid optimization.
        
        Args:
            objective_function: Function to optimize
            parameter_space: Parameter space definition
            **kwargs: Additional parameters
            
        Returns:
            Combined optimization result
        """
        start_time = time.time()
        all_experiments: List[ExperimentPoint] = []
        
        # Initialize with primary strategy
        self.current_strategy = self.config.primary_strategy
        current_points: List[ExperimentPoint] = []
        
        try:
            while self.total_evaluations < self.config.max_total_evaluations:
                # Determine evaluation budget for current strategy
                remaining_evaluations = self.config.max_total_evaluations - self.total_evaluations
                strategy_budget = self._calculate_strategy_budget(remaining_evaluations)
                
                if strategy_budget <= 0:
                    break
                
                logger.info(f"Running {self.current_strategy} strategy with {strategy_budget} evaluations")
                
                # Run current strategy
                strategy_result = self.strategies[self.current_strategy].optimize(
                    objective_function=objective_function,
                    parameter_space=parameter_space,
                    initial_points=current_points,
                    max_evaluations=strategy_budget,
                    **kwargs
                )
                
                # Update state
                self.total_evaluations += strategy_result.n_function_evaluations
                all_experiments.extend(strategy_result.all_experiments)
                self.strategy_history.append(self.current_strategy)
                
                # Update best result
                if (self.best_result is None or 
                    strategy_result.best_objective_value > self.best_result.best_objective_value):
                    self.best_result = strategy_result
                
                # Record performance
                if self.config.track_performance:
                    self._record_performance(strategy_result)
                
                # Prepare points for next strategy
                if strategy_result.all_experiments:
                    current_points = strategy_result.all_experiments[-5:]  # Use recent points
                else:
                    # Create experiment point from best result
                    best_point = ExperimentPoint(
                        parameters=strategy_result.best_parameters,
                        objectives={"objective": strategy_result.best_objective_value},
                        is_feasible=True,
                        metadata={"strategy": self.current_strategy}
                    )
                    current_points = [best_point]
                
                # Decide on strategy switching
                if self._should_switch_strategy():
                    self.current_strategy = self._select_next_strategy()
                    logger.info(f"Switching to {self.current_strategy} strategy")
            
            # Create combined result
            combined_result = OptimizationResult(
                best_parameters=self.best_result.best_parameters if self.best_result else {},
                best_objective_value=(
                    self.best_result.best_objective_value 
                    if self.best_result else float('-inf')
                ),
                n_iterations=len(self.strategy_history),
                n_function_evaluations=self.total_evaluations,
                execution_time_seconds=time.time() - start_time,
                is_converged=self.best_result.is_converged if self.best_result else False,
                all_experiments=all_experiments,
                metadata={
                    'hybrid_optimizer': True,
                    'strategy_history': self.strategy_history,
                    'performance_history': self.performance_history,
                    'strategies_used': list(set(self.strategy_history)),
                }
            )
            
            logger.info(f"Hybrid optimization completed with {self.total_evaluations} evaluations")
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Hybrid optimization failed: {e}")
            
            # Return best result found so far
            if self.best_result:
                self.best_result.metadata['error'] = str(e)
                return self.best_result
            else:
                # Return fallback result
                param_names = [param.name for param in parameter_space.parameters]
                fallback_params = {name: 0.0 for name in param_names}
                
                return OptimizationResult(
                    best_parameters=fallback_params,
                    best_objective_value=float('-inf'),
                    n_iterations=0,
                    n_function_evaluations=self.total_evaluations,
                    execution_time_seconds=time.time() - start_time,
                    is_converged=False,
                    all_experiments=all_experiments,
                    metadata={'error': str(e)}
                )
    
    def _calculate_strategy_budget(self, remaining_evaluations: int) -> int:
        """Calculate evaluation budget for current strategy."""
        strategy_ratio = self.config.evaluation_budget_ratios.get(self.current_strategy, 0.1)
        budget = int(remaining_evaluations * strategy_ratio)
        
        # Ensure minimum budget
        min_budget = 5
        budget = max(min_budget, min(budget, remaining_evaluations))
        
        return budget
    
    def _should_switch_strategy(self) -> bool:
        """Determine if strategy should be switched."""
        # Switch after fixed number of iterations
        if len(self.strategy_history) >= self.config.switch_after_iterations:
            return True
        
        # Switch on stagnation
        if self.config.switch_on_stagnation and len(self.performance_history) >= self.config.stagnation_patience:
            recent_performance = self.performance_history[-self.config.stagnation_patience:]
            performance_values = [p.get('best_value', float('-inf')) for p in recent_performance]
            
            if len(performance_values) > 1:
                improvement = max(performance_values) - min(performance_values)
                if improvement < self.config.stagnation_threshold:
                    return True
        
        return False
    
    def _select_next_strategy(self) -> str:
        """Select next optimization strategy."""
        # Simple round-robin selection
        available_strategies = [self.config.primary_strategy] + self.config.secondary_strategies
        
        # Find current strategy index
        try:
            current_index = available_strategies.index(self.current_strategy)
            next_index = (current_index + 1) % len(available_strategies)
            return available_strategies[next_index]
        except ValueError:
            # Fallback to primary strategy
            return self.config.primary_strategy
    
    def _record_performance(self, result: OptimizationResult):
        """Record performance metrics for current strategy."""
        performance_record = {
            'strategy': self.current_strategy,
            'best_value': result.best_objective_value,
            'n_evaluations': result.n_function_evaluations,
            'execution_time': result.execution_time_seconds,
            'converged': result.is_converged,
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of hybrid optimization process."""
        return {
            'total_evaluations': self.total_evaluations,
            'strategies_used': list(set(self.strategy_history)),
            'strategy_history': self.strategy_history,
            'n_strategy_switches': len(set(self.strategy_history)) - 1,
            'best_objective_value': (
                self.best_result.best_objective_value 
                if self.best_result else None
            ),
            'performance_history': self.performance_history,
        }
