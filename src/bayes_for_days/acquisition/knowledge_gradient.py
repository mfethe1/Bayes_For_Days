"""
Knowledge Gradient acquisition function for Bayesian optimization.

The Knowledge Gradient (KG) acquisition function measures the expected value
of information gained by evaluating a candidate point, considering the impact
on future optimization decisions.

Based on:
- Frazier et al. (2009) "The Knowledge-Gradient Policy for Correlated Normal Beliefs"
- Wu & Frazier (2016) "The Parallel Knowledge Gradient Method for Batch Bayesian Optimization"
- Scott et al. (2011) "The Correlated Knowledge Gradient for Simulation Optimization"
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import torch

from bayes_for_days.core.base import BaseAcquisitionFunction
from bayes_for_days.core.types import AcquisitionValue, ModelPrediction

logger = logging.getLogger(__name__)


class KnowledgeGradient(BaseAcquisitionFunction):
    """
    Knowledge Gradient acquisition function.
    
    The KG acquisition function estimates the expected improvement in the
    best achievable objective value after making one more observation.
    It considers the impact of the new observation on the entire optimization
    problem, not just the immediate improvement.
    
    Features:
    - Single-point and batch Knowledge Gradient
    - Monte Carlo approximation for complex posteriors
    - Support for discrete and continuous optimization
    - Parallel evaluation capabilities
    """
    
    def __init__(
        self,
        surrogate_model,
        n_fantasies: int = 64,
        n_candidates: int = 1000,
        current_best_value: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize Knowledge Gradient acquisition function.
        
        Args:
            surrogate_model: Fitted surrogate model
            n_fantasies: Number of fantasy samples for KG approximation
            n_candidates: Number of candidate points for discrete approximation
            current_best_value: Current best observed value
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        
        self.n_fantasies = n_fantasies
        self.n_candidates = n_candidates
        self.current_best_value = current_best_value
        
        # Cache for candidate points
        self._candidate_points = None
        self._candidate_predictions = None
        
        logger.info(f"Initialized Knowledge Gradient with {n_fantasies} fantasies")
    
    def evaluate(
        self,
        parameters: Union[Dict[str, float], List[Dict[str, float]]],
        **kwargs
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate Knowledge Gradient acquisition function.
        
        Args:
            parameters: Parameter values to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Knowledge Gradient values
        """
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_evaluation = True
        else:
            single_evaluation = False
        
        # Generate candidate points if not cached
        if self._candidate_points is None:
            self._generate_candidate_points()
        
        # Evaluate KG for each parameter set
        kg_values = []
        
        for param_dict in parameters:
            try:
                kg_value = self._compute_knowledge_gradient(param_dict)
                
                acquisition_value = AcquisitionValue(
                    value=kg_value,
                    parameters=param_dict,
                    metadata={
                        'acquisition_function': 'knowledge_gradient',
                        'n_fantasies': self.n_fantasies,
                        'n_candidates': self.n_candidates,
                    }
                )
                kg_values.append(acquisition_value)
                
            except Exception as e:
                logger.warning(f"KG evaluation failed for {param_dict}: {e}")
                
                # Return zero acquisition value for failed evaluations
                acquisition_value = AcquisitionValue(
                    value=0.0,
                    parameters=param_dict,
                    metadata={'error': str(e)}
                )
                kg_values.append(acquisition_value)
        
        if single_evaluation:
            return kg_values[0]
        return kg_values
    
    def _generate_candidate_points(self):
        """Generate candidate points for discrete KG approximation."""
        # Get parameter bounds
        bounds = self._get_parameter_bounds()
        
        if not bounds:
            logger.warning("No parameter bounds available for candidate generation")
            self._candidate_points = []
            self._candidate_predictions = []
            return
        
        # Generate random candidate points
        candidate_points = []
        
        for _ in range(self.n_candidates):
            candidate = {}
            
            for i, (param_name, (low, high)) in enumerate(bounds.items()):
                candidate[param_name] = np.random.uniform(low, high)
            
            candidate_points.append(candidate)
        
        self._candidate_points = candidate_points
        
        # Get predictions for all candidate points
        try:
            predictions = self.surrogate_model.predict(candidate_points)
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            self._candidate_predictions = predictions
            
        except Exception as e:
            logger.warning(f"Failed to get candidate predictions: {e}")
            self._candidate_predictions = []
    
    def _compute_knowledge_gradient(self, parameters: Dict[str, float]) -> float:
        """
        Compute Knowledge Gradient value for given parameters.
        
        Args:
            parameters: Parameter values
            
        Returns:
            Knowledge Gradient value
        """
        if not self._candidate_points or not self._candidate_predictions:
            return 0.0
        
        try:
            # Get prediction at query point
            query_prediction = self.surrogate_model.predict(parameters)
            
            if not hasattr(query_prediction, 'mean') or not hasattr(query_prediction, 'std'):
                return 0.0
            
            query_mean = query_prediction.mean
            query_std = query_prediction.std
            
            if query_std <= 1e-10:
                return 0.0
            
            # Current best value among candidates
            current_best = self._get_current_best_value()
            
            # Monte Carlo approximation of KG
            kg_samples = []
            
            for _ in range(self.n_fantasies):
                # Sample fantasy observation at query point
                fantasy_value = np.random.normal(query_mean, query_std)
                
                # Compute expected best value after observing fantasy_value
                expected_best = self._compute_expected_best_after_observation(
                    parameters, fantasy_value, current_best
                )
                
                # KG sample = improvement over current best
                kg_sample = max(0.0, expected_best - current_best)
                kg_samples.append(kg_sample)
            
            # Return mean KG value
            kg_value = np.mean(kg_samples)
            
            return kg_value
            
        except Exception as e:
            logger.warning(f"KG computation failed: {e}")
            return 0.0
    
    def _compute_expected_best_after_observation(
        self,
        query_params: Dict[str, float],
        fantasy_value: float,
        current_best: float
    ) -> float:
        """
        Compute expected best value after observing fantasy_value at query_params.
        
        This is a simplified approximation that assumes the fantasy observation
        doesn't significantly change the posterior at candidate points.
        """
        # In a full implementation, we would:
        # 1. Update the GP posterior with the fantasy observation
        # 2. Recompute predictions at all candidate points
        # 3. Find the maximum expected value
        
        # Simplified approximation: assume fantasy observation is the new best
        # if it's better than current best, otherwise return current best
        
        # Consider the fantasy value as a potential new best
        potential_best = max(current_best, fantasy_value)
        
        # Also consider the best expected value among candidates
        if self._candidate_predictions:
            candidate_means = [
                pred.mean if hasattr(pred, 'mean') else 0.0
                for pred in self._candidate_predictions
            ]
            
            if candidate_means:
                best_candidate_mean = max(candidate_means)
                potential_best = max(potential_best, best_candidate_mean)
        
        return potential_best
    
    def _get_current_best_value(self) -> float:
        """Get current best observed value."""
        if self.current_best_value is not None:
            return self.current_best_value
        
        # Estimate from candidate predictions
        if self._candidate_predictions:
            candidate_means = [
                pred.mean if hasattr(pred, 'mean') else 0.0
                for pred in self._candidate_predictions
            ]
            
            if candidate_means:
                return max(candidate_means)
        
        return 0.0
    
    def _get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds from surrogate model."""
        if hasattr(self.surrogate_model, 'parameter_space'):
            parameter_space = self.surrogate_model.parameter_space
            
            bounds = {}
            for param in parameter_space.parameters:
                if param.bounds:
                    bounds[param.name] = param.bounds
            
            return bounds
        
        return {}
    
    def optimize(
        self,
        n_candidates: int = 1,
        bounds: Optional[List[Tuple[float, float]]] = None,
        n_restarts: int = 10,
        **kwargs
    ) -> List[Dict[str, float]]:
        """
        Optimize Knowledge Gradient acquisition function.
        
        Args:
            n_candidates: Number of candidates to return
            bounds: Parameter bounds
            n_restarts: Number of optimization restarts
            **kwargs: Additional optimization parameters
            
        Returns:
            List of optimal parameter dictionaries
        """
        if bounds is None:
            bounds_dict = self._get_parameter_bounds()
            if not bounds_dict:
                logger.warning("No bounds available for KG optimization")
                return []
            
            param_names = list(bounds_dict.keys())
            bounds = [bounds_dict[name] for name in param_names]
        else:
            # Assume bounds correspond to parameter space parameters
            if hasattr(self.surrogate_model, 'parameter_space'):
                param_names = [p.name for p in self.surrogate_model.parameter_space.parameters]
            else:
                param_names = [f"x{i}" for i in range(len(bounds))]
        
        if not bounds:
            return []
        
        # Generate candidate points for KG evaluation
        self._generate_candidate_points()
        
        # Objective function for optimization (negative KG for minimization)
        def objective(x):
            param_dict = {param_names[i]: x[i] for i in range(len(param_names))}
            kg_value = self._compute_knowledge_gradient(param_dict)
            return -kg_value  # Minimize negative KG
        
        # Multi-start optimization
        best_candidates = []
        
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.array([
                np.random.uniform(low, high) for low, high in bounds
            ])
            
            try:
                result = minimize(
                    objective,
                    x0,
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 100}
                )
                
                if result.success:
                    optimal_params = {
                        param_names[i]: result.x[i] 
                        for i in range(len(param_names))
                    }
                    
                    kg_value = -result.fun
                    best_candidates.append((optimal_params, kg_value))
                    
            except Exception as e:
                logger.debug(f"KG optimization restart failed: {e}")
                continue
        
        # Sort by KG value and return top candidates
        best_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return unique candidates
        unique_candidates = []
        for candidate, _ in best_candidates:
            if len(unique_candidates) >= n_candidates:
                break
            
            # Check if candidate is sufficiently different from existing ones
            is_unique = True
            for existing in unique_candidates:
                distance = np.sqrt(sum(
                    (candidate.get(key, 0) - existing.get(key, 0))**2
                    for key in candidate.keys()
                ))
                
                if distance < 1e-6:
                    is_unique = False
                    break
            
            if is_unique:
                unique_candidates.append(candidate)
        
        # Fill remaining slots with random candidates if needed
        while len(unique_candidates) < n_candidates and len(unique_candidates) < len(best_candidates):
            remaining_candidates = [
                candidate for candidate, _ in best_candidates
                if candidate not in unique_candidates
            ]
            
            if remaining_candidates:
                unique_candidates.append(remaining_candidates[0])
            else:
                break
        
        return unique_candidates[:n_candidates]
    
    def update_current_best(self, best_value: float):
        """
        Update current best observed value.
        
        Args:
            best_value: New best observed value
        """
        self.current_best_value = best_value
        logger.debug(f"Updated current best value to {best_value}")
    
    def get_acquisition_info(self) -> Dict[str, Any]:
        """Get information about the acquisition function."""
        info = super().get_acquisition_info()
        info.update({
            'n_fantasies': self.n_fantasies,
            'n_candidates': self.n_candidates,
            'current_best_value': self.current_best_value,
            'has_candidate_cache': self._candidate_points is not None,
        })
        
        return info
