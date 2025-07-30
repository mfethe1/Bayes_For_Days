"""
Expected Improvement acquisition function for Bayesian optimization.

This module implements the Expected Improvement (EI) acquisition function,
one of the most popular and effective acquisition functions for Bayesian optimization.
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any
from scipy.stats import norm
from abc import ABC, abstractmethod

from bayes_for_days.core.types import ParameterDict, AcquisitionValue
from bayes_for_days.core.base import BaseAcquisitionFunction, BaseSurrogateModel


class ExpectedImprovement(BaseAcquisitionFunction):
    """
    Expected Improvement acquisition function.
    
    The Expected Improvement acquisition function balances exploration and exploitation
    by computing the expected improvement over the current best observed value.
    
    EI(x) = E[max(f_min - f(x), 0)]
    
    where f_min is the current best (minimum) observed value.
    """
    
    def __init__(
        self, 
        surrogate_model: BaseSurrogateModel,
        xi: float = 0.01,
        minimize: bool = True,
        **kwargs
    ):
        """
        Initialize Expected Improvement acquisition function.
        
        Args:
            surrogate_model: Fitted surrogate model for predictions
            xi: Exploration parameter (higher values encourage more exploration)
            minimize: Whether to minimize (True) or maximize (False) the objective
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        self.xi = xi
        self.minimize = minimize
        self.current_best = None
        
    def set_current_best(self, best_value: float) -> None:
        """
        Set the current best observed value.
        
        Args:
            best_value: Current best objective value
        """
        self.current_best = best_value
    
    def evaluate(
        self, 
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate Expected Improvement at given parameters.
        
        Args:
            parameters: Parameter values for evaluation
            
        Returns:
            Expected Improvement values
        """
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_input = True
        else:
            single_input = False
        
        # Get predictions from surrogate model
        predictions = self.surrogate_model.predict(parameters)
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Calculate Expected Improvement for each point
        ei_values = []
        for pred in predictions:
            ei_value = self._calculate_ei(pred.mean, pred.std)
            ei_values.append(AcquisitionValue(
                value=ei_value,
                gradient=None  # Gradient computation not implemented yet
            ))
        
        return ei_values[0] if single_input else ei_values
    
    def _calculate_ei(self, mean: float, std: float) -> float:
        """
        Calculate Expected Improvement value.
        
        Args:
            mean: Predicted mean
            std: Predicted standard deviation
            
        Returns:
            Expected Improvement value
        """
        if self.current_best is None:
            raise ValueError("Current best value not set. Call set_current_best() first.")
        
        if std <= 0:
            return 0.0
        
        # Calculate improvement
        if self.minimize:
            improvement = self.current_best - mean - self.xi
        else:
            improvement = mean - self.current_best - self.xi
        
        # Standardize improvement
        z = improvement / std
        
        # Calculate Expected Improvement
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)
        
        return max(ei, 0.0)
    
    def optimize(
        self, 
        n_candidates: int = 1,
        bounds: Optional[List[tuple]] = None,
        **kwargs
    ) -> List[ParameterDict]:
        """
        Optimize the Expected Improvement function to find next evaluation points.
        
        Args:
            n_candidates: Number of candidate points to return
            bounds: Parameter bounds for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            List of optimal parameter dictionaries
        """
        # Simple grid search implementation for demonstration
        # In practice, you would use more sophisticated optimization
        
        if bounds is None:
            # Use default bounds from parameter space
            bounds = [(0, 1)] * 2  # Default 2D unit square
        
        # Create grid of candidate points
        n_grid = kwargs.get('n_grid', 100)
        candidates = []
        
        if len(bounds) == 1:
            # 1D case
            x_vals = np.linspace(bounds[0][0], bounds[0][1], n_grid)
            for x in x_vals:
                candidates.append({'x0': x})
        elif len(bounds) == 2:
            # 2D case
            x1_vals = np.linspace(bounds[0][0], bounds[0][1], int(np.sqrt(n_grid)))
            x2_vals = np.linspace(bounds[1][0], bounds[1][1], int(np.sqrt(n_grid)))
            for x1 in x1_vals:
                for x2 in x2_vals:
                    candidates.append({'x0': x1, 'x1': x2})
        else:
            # Higher dimensional case - random sampling
            for _ in range(n_grid):
                candidate = {}
                for i, (low, high) in enumerate(bounds):
                    candidate[f'x{i}'] = np.random.uniform(low, high)
                candidates.append(candidate)
        
        # Evaluate EI at all candidates
        ei_values = self.evaluate(candidates)
        if not isinstance(ei_values, list):
            ei_values = [ei_values]
        
        # Sort by EI value (descending)
        candidate_ei_pairs = list(zip(candidates, ei_values))
        candidate_ei_pairs.sort(key=lambda x: x[1].value, reverse=True)
        
        # Return top n_candidates
        return [pair[0] for pair in candidate_ei_pairs[:n_candidates]]


class ProbabilityOfImprovement(BaseAcquisitionFunction):
    """
    Probability of Improvement acquisition function.
    
    PI(x) = P(f(x) < f_min - xi)
    
    where f_min is the current best observed value and xi is an exploration parameter.
    """
    
    def __init__(
        self, 
        surrogate_model: BaseSurrogateModel,
        xi: float = 0.01,
        minimize: bool = True,
        **kwargs
    ):
        """
        Initialize Probability of Improvement acquisition function.
        
        Args:
            surrogate_model: Fitted surrogate model for predictions
            xi: Exploration parameter
            minimize: Whether to minimize (True) or maximize (False) the objective
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        self.xi = xi
        self.minimize = minimize
        self.current_best = None
    
    def set_current_best(self, best_value: float) -> None:
        """Set the current best observed value."""
        self.current_best = best_value
    
    def evaluate(
        self, 
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """Evaluate Probability of Improvement at given parameters."""
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_input = True
        else:
            single_input = False
        
        # Get predictions from surrogate model
        predictions = self.surrogate_model.predict(parameters)
        if not isinstance(predictions, list):
            predictions = [predictions]
        
        # Calculate PI for each point
        pi_values = []
        for pred in predictions:
            pi_value = self._calculate_pi(pred.mean, pred.std)
            pi_values.append(AcquisitionValue(
                value=pi_value,
                gradient=None
            ))
        
        return pi_values[0] if single_input else pi_values
    
    def _calculate_pi(self, mean: float, std: float) -> float:
        """Calculate Probability of Improvement value."""
        if self.current_best is None:
            raise ValueError("Current best value not set. Call set_current_best() first.")
        
        if std <= 0:
            return 0.0
        
        # Calculate improvement threshold
        if self.minimize:
            improvement = self.current_best - mean - self.xi
        else:
            improvement = mean - self.current_best - self.xi
        
        # Standardize improvement
        z = improvement / std
        
        # Calculate Probability of Improvement
        pi = norm.cdf(z)
        
        return pi
    
    def optimize(
        self, 
        n_candidates: int = 1,
        bounds: Optional[List[tuple]] = None,
        **kwargs
    ) -> List[ParameterDict]:
        """Optimize PI function (similar to EI optimization)."""
        # Use same optimization strategy as EI
        ei_optimizer = ExpectedImprovement(self.surrogate_model, self.xi, self.minimize)
        ei_optimizer.current_best = self.current_best
        return ei_optimizer.optimize(n_candidates, bounds, **kwargs)


# Convenience function for creating acquisition functions
def create_acquisition_function(
    name: str,
    surrogate_model: BaseSurrogateModel,
    **kwargs
) -> BaseAcquisitionFunction:
    """
    Create an acquisition function by name.
    
    Args:
        name: Name of acquisition function ('ei', 'pi')
        surrogate_model: Fitted surrogate model
        **kwargs: Additional parameters
        
    Returns:
        Acquisition function instance
    """
    name = name.lower()
    
    if name in ['ei', 'expected_improvement']:
        return ExpectedImprovement(surrogate_model, **kwargs)
    elif name in ['pi', 'probability_of_improvement']:
        return ProbabilityOfImprovement(surrogate_model, **kwargs)
    else:
        raise ValueError(f"Unknown acquisition function: {name}")
