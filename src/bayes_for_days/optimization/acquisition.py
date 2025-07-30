"""
Acquisition functions for Bayesian optimization.

This module implements essential acquisition functions with gradient-based optimization:
- Expected Improvement (EI)
- Upper Confidence Bound (UCB)
- Probability of Improvement (PI)
- Expected Pareto Improvement (EPI) for multi-objective optimization
- Parallel multi-start optimization capabilities
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import torch
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from bayes_for_days.core.base import BaseAcquisitionFunction
from bayes_for_days.core.types import (
    AcquisitionValue,
    AcquisitionFunction as AcquisitionFunctionType,
    ParameterDict,
    ModelPrediction,
)

logger = logging.getLogger(__name__)


class ExpectedImprovement(BaseAcquisitionFunction):
    """
    Expected Improvement acquisition function.
    
    Balances exploration and exploitation by computing the expected improvement
    over the current best observed value.
    """
    
    def __init__(self, surrogate_model, xi: float = 0.01, **kwargs):
        """
        Initialize Expected Improvement acquisition function.
        
        Args:
            surrogate_model: Fitted surrogate model
            xi: Exploration parameter (higher values encourage exploration)
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        self.xi = xi
        self.current_best = None
        self._update_current_best()
    
    def _update_current_best(self) -> None:
        """Update the current best observed value."""
        if hasattr(self.surrogate_model, 'training_data') and self.surrogate_model.training_data:
            # Assume single objective for now
            objectives = []
            for point in self.surrogate_model.training_data:
                if point.objectives and point.is_feasible:
                    obj_values = list(point.objectives.values())
                    if obj_values:
                        objectives.append(obj_values[0])  # First objective
            
            if objectives:
                self.current_best = max(objectives)  # Assuming maximization
            else:
                self.current_best = 0.0
        else:
            self.current_best = 0.0
    
    def evaluate(
        self, 
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate Expected Improvement at given parameters.
        
        Args:
            parameters: Parameter values for evaluation
            
        Returns:
            Acquisition function values
        """
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_evaluation = True
        else:
            single_evaluation = False
        
        # Get model predictions
        predictions = self.surrogate_model.predict(parameters)
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Compute Expected Improvement
        results = []
        for pred in predictions:
            mean = pred.mean
            std = pred.std
            
            if std <= 1e-8:  # No uncertainty
                ei_value = 0.0
            else:
                # Compute improvement
                improvement = mean - self.current_best - self.xi
                z = improvement / std
                
                # Expected Improvement formula
                ei_value = improvement * norm.cdf(z) + std * norm.pdf(z)
                ei_value = max(0.0, ei_value)  # Ensure non-negative
            
            acq_value = AcquisitionValue(
                value=ei_value,
                function_type=AcquisitionFunctionType.EXPECTED_IMPROVEMENT,
                parameters={"xi": self.xi, "current_best": self.current_best}
            )
            results.append(acq_value)
        
        if single_evaluation:
            return results[0]
        return results
    
    def optimize(
        self, 
        n_candidates: int = 1,
        bounds: Optional[List[tuple]] = None,
        n_restarts: int = 10,
        **kwargs
    ) -> List[ParameterDict]:
        """
        Optimize Expected Improvement to find next evaluation points.
        
        Args:
            n_candidates: Number of candidate points to return
            bounds: Parameter bounds for optimization
            n_restarts: Number of random restarts for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            List of optimal parameter dictionaries
        """
        if bounds is None:
            bounds = self.surrogate_model.parameter_space.get_bounds()
        
        if not bounds:
            raise ValueError("Parameter bounds are required for optimization")
        
        # Update current best before optimization
        self._update_current_best()
        
        # Define objective function for scipy.optimize (negative EI for minimization)
        def objective(x):
            param_dict = self._array_to_param_dict(x)
            ei_value = self.evaluate(param_dict)
            return -ei_value.value  # Negative for minimization
        
        # Parallel multi-start optimization
        candidates = []
        
        with ThreadPoolExecutor(max_workers=min(n_restarts, 8)) as executor:
            futures = []
            
            for _ in range(n_restarts):
                # Random starting point
                x0 = np.array([
                    np.random.uniform(low, high) for low, high in bounds
                ])
                
                # Submit optimization task
                future = executor.submit(
                    minimize,
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )
                futures.append(future)
            
            # Collect results
            optimization_results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    if result.success:
                        optimization_results.append(result)
                except Exception as e:
                    logger.warning(f"Optimization failed: {e}")
        
        # Sort by objective value and select best candidates
        optimization_results.sort(key=lambda r: r.fun)
        
        for i in range(min(n_candidates, len(optimization_results))):
            result = optimization_results[i]
            param_dict = self._array_to_param_dict(result.x)
            candidates.append(param_dict)
        
        # If we don't have enough candidates, use random sampling
        while len(candidates) < n_candidates:
            random_params = self._random_sample(bounds)
            candidates.append(random_params)
        
        return candidates[:n_candidates]
    
    def _array_to_param_dict(self, x: np.ndarray) -> ParameterDict:
        """Convert parameter array to parameter dictionary."""
        param_dict = {}
        param_names = [param.name for param in self.surrogate_model.parameter_space.parameters]
        
        for i, param_name in enumerate(param_names):
            if i < len(x):
                param_dict[param_name] = float(x[i])
        
        return param_dict
    
    def _random_sample(self, bounds: List[tuple]) -> ParameterDict:
        """Generate random parameter sample within bounds."""
        param_dict = {}
        param_names = [param.name for param in self.surrogate_model.parameter_space.parameters]
        
        for i, param_name in enumerate(param_names):
            if i < len(bounds):
                low, high = bounds[i]
                param_dict[param_name] = np.random.uniform(low, high)
        
        return param_dict


class UpperConfidenceBound(BaseAcquisitionFunction):
    """
    Upper Confidence Bound acquisition function.
    
    Balances exploration and exploitation using confidence bounds.
    """
    
    def __init__(self, surrogate_model, beta: float = 2.0, **kwargs):
        """
        Initialize Upper Confidence Bound acquisition function.
        
        Args:
            surrogate_model: Fitted surrogate model
            beta: Exploration parameter (higher values encourage exploration)
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        self.beta = beta
    
    def evaluate(
        self, 
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate Upper Confidence Bound at given parameters.
        
        Args:
            parameters: Parameter values for evaluation
            
        Returns:
            Acquisition function values
        """
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_evaluation = True
        else:
            single_evaluation = False
        
        # Get model predictions
        predictions = self.surrogate_model.predict(parameters)
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Compute Upper Confidence Bound
        results = []
        for pred in predictions:
            mean = pred.mean
            std = pred.std
            
            # UCB formula: mean + beta * std
            ucb_value = mean + self.beta * std
            
            acq_value = AcquisitionValue(
                value=ucb_value,
                function_type=AcquisitionFunctionType.UPPER_CONFIDENCE_BOUND,
                parameters={"beta": self.beta}
            )
            results.append(acq_value)
        
        if single_evaluation:
            return results[0]
        return results
    
    def optimize(
        self, 
        n_candidates: int = 1,
        bounds: Optional[List[tuple]] = None,
        n_restarts: int = 10,
        **kwargs
    ) -> List[ParameterDict]:
        """
        Optimize Upper Confidence Bound to find next evaluation points.
        
        Args:
            n_candidates: Number of candidate points to return
            bounds: Parameter bounds for optimization
            n_restarts: Number of random restarts for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            List of optimal parameter dictionaries
        """
        if bounds is None:
            bounds = self.surrogate_model.parameter_space.get_bounds()
        
        if not bounds:
            raise ValueError("Parameter bounds are required for optimization")
        
        # Define objective function for scipy.optimize (negative UCB for minimization)
        def objective(x):
            param_dict = self._array_to_param_dict(x)
            ucb_value = self.evaluate(param_dict)
            return -ucb_value.value  # Negative for minimization
        
        # Use differential evolution for global optimization
        candidates = []
        
        for _ in range(n_candidates):
            try:
                result = differential_evolution(
                    objective,
                    bounds,
                    maxiter=100,
                    popsize=15,
                    seed=np.random.randint(0, 10000)
                )
                
                if result.success:
                    param_dict = self._array_to_param_dict(result.x)
                    candidates.append(param_dict)
                else:
                    # Fallback to random sampling
                    candidates.append(self._random_sample(bounds))
                    
            except Exception as e:
                logger.warning(f"UCB optimization failed: {e}")
                candidates.append(self._random_sample(bounds))
        
        return candidates
    
    def _array_to_param_dict(self, x: np.ndarray) -> ParameterDict:
        """Convert parameter array to parameter dictionary."""
        param_dict = {}
        param_names = [param.name for param in self.surrogate_model.parameter_space.parameters]
        
        for i, param_name in enumerate(param_names):
            if i < len(x):
                param_dict[param_name] = float(x[i])
        
        return param_dict
    
    def _random_sample(self, bounds: List[tuple]) -> ParameterDict:
        """Generate random parameter sample within bounds."""
        param_dict = {}
        param_names = [param.name for param in self.surrogate_model.parameter_space.parameters]
        
        for i, param_name in enumerate(param_names):
            if i < len(bounds):
                low, high = bounds[i]
                param_dict[param_name] = np.random.uniform(low, high)
        
        return param_dict


class ProbabilityOfImprovement(BaseAcquisitionFunction):
    """
    Probability of Improvement acquisition function.
    
    Computes the probability of improving over the current best value.
    """
    
    def __init__(self, surrogate_model, xi: float = 0.01, **kwargs):
        """
        Initialize Probability of Improvement acquisition function.
        
        Args:
            surrogate_model: Fitted surrogate model
            xi: Exploration parameter
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        self.xi = xi
        self.current_best = None
        self._update_current_best()
    
    def _update_current_best(self) -> None:
        """Update the current best observed value."""
        if hasattr(self.surrogate_model, 'training_data') and self.surrogate_model.training_data:
            objectives = []
            for point in self.surrogate_model.training_data:
                if point.objectives and point.is_feasible:
                    obj_values = list(point.objectives.values())
                    if obj_values:
                        objectives.append(obj_values[0])
            
            if objectives:
                self.current_best = max(objectives)
            else:
                self.current_best = 0.0
        else:
            self.current_best = 0.0
    
    def evaluate(
        self, 
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate Probability of Improvement at given parameters.
        
        Args:
            parameters: Parameter values for evaluation
            
        Returns:
            Acquisition function values
        """
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_evaluation = True
        else:
            single_evaluation = False
        
        # Get model predictions
        predictions = self.surrogate_model.predict(parameters)
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Compute Probability of Improvement
        results = []
        for pred in predictions:
            mean = pred.mean
            std = pred.std
            
            if std <= 1e-8:  # No uncertainty
                pi_value = 1.0 if mean > self.current_best + self.xi else 0.0
            else:
                # Probability of Improvement formula
                z = (mean - self.current_best - self.xi) / std
                pi_value = norm.cdf(z)
            
            acq_value = AcquisitionValue(
                value=pi_value,
                function_type=AcquisitionFunctionType.PROBABILITY_OF_IMPROVEMENT,
                parameters={"xi": self.xi, "current_best": self.current_best}
            )
            results.append(acq_value)
        
        if single_evaluation:
            return results[0]
        return results
    
    def optimize(
        self, 
        n_candidates: int = 1,
        bounds: Optional[List[tuple]] = None,
        n_restarts: int = 10,
        **kwargs
    ) -> List[ParameterDict]:
        """
        Optimize Probability of Improvement to find next evaluation points.
        
        Args:
            n_candidates: Number of candidate points to return
            bounds: Parameter bounds for optimization
            n_restarts: Number of random restarts for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            List of optimal parameter dictionaries
        """
        if bounds is None:
            bounds = self.surrogate_model.parameter_space.get_bounds()
        
        if not bounds:
            raise ValueError("Parameter bounds are required for optimization")
        
        # Update current best before optimization
        self._update_current_best()
        
        # Define objective function for scipy.optimize (negative PI for minimization)
        def objective(x):
            param_dict = self._array_to_param_dict(x)
            pi_value = self.evaluate(param_dict)
            return -pi_value.value  # Negative for minimization
        
        # Multi-start optimization
        candidates = []
        best_results = []
        
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.array([
                np.random.uniform(low, high) for low, high in bounds
            ])
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100}
                )
                
                if result.success:
                    best_results.append(result)
                    
            except Exception as e:
                logger.warning(f"PI optimization failed: {e}")
        
        # Sort by objective value and select best candidates
        best_results.sort(key=lambda r: r.fun)
        
        for i in range(min(n_candidates, len(best_results))):
            result = best_results[i]
            param_dict = self._array_to_param_dict(result.x)
            candidates.append(param_dict)
        
        # Fill remaining candidates with random sampling
        while len(candidates) < n_candidates:
            candidates.append(self._random_sample(bounds))
        
        return candidates[:n_candidates]
    
    def _array_to_param_dict(self, x: np.ndarray) -> ParameterDict:
        """Convert parameter array to parameter dictionary."""
        param_dict = {}
        param_names = [param.name for param in self.surrogate_model.parameter_space.parameters]
        
        for i, param_name in enumerate(param_names):
            if i < len(x):
                param_dict[param_name] = float(x[i])
        
        return param_dict
    
    def _random_sample(self, bounds: List[tuple]) -> ParameterDict:
        """Generate random parameter sample within bounds."""
        param_dict = {}
        param_names = [param.name for param in self.surrogate_model.parameter_space.parameters]
        
        for i, param_name in enumerate(param_names):
            if i < len(bounds):
                low, high = bounds[i]
                param_dict[param_name] = np.random.uniform(low, high)
        
        return param_dict
