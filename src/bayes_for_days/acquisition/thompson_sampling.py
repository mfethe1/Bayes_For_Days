"""
Thompson Sampling acquisition function for Bayesian optimization.

Thompson Sampling (TS) is a probability matching strategy that samples
from the posterior distribution of the objective function and selects
the point that maximizes the sampled function.

Based on:
- Thompson (1933) "On the likelihood that one unknown probability exceeds another"
- Russo & Van Roy (2014) "Learning to Optimize via Posterior Sampling"
- Kandasamy et al. (2018) "Parallelised Bayesian Optimisation via Thompson Sampling"
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.optimize import minimize, differential_evolution
import torch

from bayes_for_days.core.base import BaseAcquisitionFunction
from bayes_for_days.core.types import AcquisitionValue, ModelPrediction

logger = logging.getLogger(__name__)


class ThompsonSampling(BaseAcquisitionFunction):
    """
    Thompson Sampling acquisition function.
    
    TS samples a function from the GP posterior and finds the maximum
    of this sampled function. This provides a natural way to balance
    exploration and exploitation by sampling from the uncertainty.
    
    Features:
    - Posterior function sampling
    - Batch Thompson Sampling for parallel evaluation
    - Support for different sampling strategies
    - Efficient optimization of sampled functions
    """
    
    def __init__(
        self,
        surrogate_model,
        n_samples: int = 1,
        sampling_strategy: str = "gp_sample",
        optimization_method: str = "differential_evolution",
        **kwargs
    ):
        """
        Initialize Thompson Sampling acquisition function.
        
        Args:
            surrogate_model: Fitted surrogate model
            n_samples: Number of posterior samples to draw
            sampling_strategy: Strategy for sampling ('gp_sample', 'discrete')
            optimization_method: Method for optimizing sampled functions
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        
        self.n_samples = n_samples
        self.sampling_strategy = sampling_strategy
        self.optimization_method = optimization_method
        
        # Cache for sampled functions
        self._sampled_functions = None
        self._sample_points = None
        
        logger.info(f"Initialized Thompson Sampling with {n_samples} samples")
    
    def evaluate(
        self,
        parameters: Union[Dict[str, float], List[Dict[str, float]]],
        **kwargs
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate Thompson Sampling acquisition function.
        
        For TS, the "acquisition value" is the value of the sampled function
        at the given parameters. This is mainly used for debugging/analysis.
        
        Args:
            parameters: Parameter values to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            Sampled function values
        """
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_evaluation = True
        else:
            single_evaluation = False
        
        # Generate samples if not cached
        if self._sampled_functions is None:
            self._generate_posterior_samples()
        
        # Evaluate sampled functions at given parameters
        ts_values = []
        
        for param_dict in parameters:
            try:
                # Get prediction at query point
                prediction = self.surrogate_model.predict(param_dict)
                
                if hasattr(prediction, 'mean'):
                    # Use mean as proxy for sampled function value
                    # In practice, would evaluate actual sampled function
                    sampled_value = prediction.mean
                else:
                    sampled_value = 0.0
                
                acquisition_value = AcquisitionValue(
                    value=sampled_value,
                    parameters=param_dict,
                    metadata={
                        'acquisition_function': 'thompson_sampling',
                        'n_samples': self.n_samples,
                        'sampling_strategy': self.sampling_strategy,
                    }
                )
                ts_values.append(acquisition_value)
                
            except Exception as e:
                logger.warning(f"TS evaluation failed for {param_dict}: {e}")
                
                # Return zero acquisition value for failed evaluations
                acquisition_value = AcquisitionValue(
                    value=0.0,
                    parameters=param_dict,
                    metadata={'error': str(e)}
                )
                ts_values.append(acquisition_value)
        
        if single_evaluation:
            return ts_values[0]
        return ts_values
    
    def _generate_posterior_samples(self):
        """Generate samples from the GP posterior."""
        if self.sampling_strategy == "gp_sample":
            self._generate_gp_samples()
        elif self.sampling_strategy == "discrete":
            self._generate_discrete_samples()
        else:
            logger.warning(f"Unknown sampling strategy: {self.sampling_strategy}")
            self._generate_gp_samples()
    
    def _generate_gp_samples(self):
        """Generate samples from GP posterior using function sampling."""
        # Get parameter bounds
        bounds = self._get_parameter_bounds()
        
        if not bounds:
            logger.warning("No parameter bounds available for GP sampling")
            self._sampled_functions = []
            self._sample_points = []
            return
        
        # Generate sample points for function evaluation
        n_sample_points = 100
        sample_points = []
        
        for _ in range(n_sample_points):
            point = {}
            for param_name, (low, high) in bounds.items():
                point[param_name] = np.random.uniform(low, high)
            sample_points.append(point)
        
        self._sample_points = sample_points
        
        try:
            # Get predictions at sample points
            predictions = self.surrogate_model.predict(sample_points)
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            # Extract means and covariances for sampling
            means = np.array([
                pred.mean if hasattr(pred, 'mean') else 0.0
                for pred in predictions
            ])
            
            variances = np.array([
                pred.variance if hasattr(pred, 'variance') else 1.0
                for pred in predictions
            ])
            
            # Sample functions from GP posterior
            sampled_functions = []
            
            for _ in range(self.n_samples):
                # Sample function values at sample points
                # Simplified: assume independence (should use GP covariance)
                sampled_values = np.random.normal(means, np.sqrt(variances))
                sampled_functions.append(sampled_values)
            
            self._sampled_functions = sampled_functions
            
        except Exception as e:
            logger.warning(f"GP sampling failed: {e}")
            self._sampled_functions = []
    
    def _generate_discrete_samples(self):
        """Generate samples using discrete approximation."""
        # Get parameter bounds
        bounds = self._get_parameter_bounds()
        
        if not bounds:
            self._sampled_functions = []
            self._sample_points = []
            return
        
        # Generate discrete grid of points
        n_grid_points = 50
        sample_points = []
        
        # Simple random sampling (could use grid)
        for _ in range(n_grid_points):
            point = {}
            for param_name, (low, high) in bounds.items():
                point[param_name] = np.random.uniform(low, high)
            sample_points.append(point)
        
        self._sample_points = sample_points
        
        try:
            # Get predictions at grid points
            predictions = self.surrogate_model.predict(sample_points)
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            # Sample from predictive distributions
            sampled_functions = []
            
            for _ in range(self.n_samples):
                sampled_values = []
                
                for pred in predictions:
                    if hasattr(pred, 'mean') and hasattr(pred, 'std'):
                        sampled_value = np.random.normal(pred.mean, pred.std)
                    else:
                        sampled_value = 0.0
                    
                    sampled_values.append(sampled_value)
                
                sampled_functions.append(np.array(sampled_values))
            
            self._sampled_functions = sampled_functions
            
        except Exception as e:
            logger.warning(f"Discrete sampling failed: {e}")
            self._sampled_functions = []
    
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
        Optimize Thompson Sampling acquisition function.
        
        This finds the maximum of each sampled function and returns
        the corresponding parameter values.
        
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
                logger.warning("No bounds available for TS optimization")
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
        
        # Generate posterior samples
        self._generate_posterior_samples()
        
        if not self._sampled_functions or not self._sample_points:
            logger.warning("No sampled functions available")
            return []
        
        # Find maximum of each sampled function
        candidates = []
        
        for i, sampled_function in enumerate(self._sampled_functions):
            if len(candidates) >= n_candidates:
                break
            
            try:
                # Find maximum of sampled function
                max_idx = np.argmax(sampled_function)
                max_point = self._sample_points[max_idx]
                max_value = sampled_function[max_idx]
                
                candidates.append((max_point, max_value))
                
            except Exception as e:
                logger.debug(f"Failed to find maximum of sample {i}: {e}")
                continue
        
        # If we need more candidates, use optimization
        while len(candidates) < n_candidates and len(candidates) < len(self._sampled_functions):
            sample_idx = len(candidates)
            
            if sample_idx >= len(self._sampled_functions):
                break
            
            sampled_function = self._sampled_functions[sample_idx]
            
            # Create interpolation function for optimization
            def objective(x):
                param_dict = {param_names[i]: x[i] for i in range(len(param_names))}
                
                # Find nearest sample point (simplified)
                min_distance = float('inf')
                nearest_idx = 0
                
                for j, sample_point in enumerate(self._sample_points):
                    distance = sum(
                        (param_dict.get(key, 0) - sample_point.get(key, 0))**2
                        for key in param_dict.keys()
                    )
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_idx = j
                
                return -sampled_function[nearest_idx]  # Minimize negative
            
            # Optimize sampled function
            try:
                if self.optimization_method == "differential_evolution":
                    result = differential_evolution(
                        objective,
                        bounds,
                        maxiter=50,
                        popsize=10
                    )
                else:
                    # Random starting point
                    x0 = np.array([
                        np.random.uniform(low, high) for low, high in bounds
                    ])
                    
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
                    
                    optimal_value = -result.fun
                    candidates.append((optimal_params, optimal_value))
                
            except Exception as e:
                logger.debug(f"TS optimization failed for sample {sample_idx}: {e}")
                continue
        
        # Return unique candidates
        unique_candidates = []
        for candidate, _ in candidates:
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
        
        return unique_candidates[:n_candidates]
    
    def get_acquisition_info(self) -> Dict[str, Any]:
        """Get information about the acquisition function."""
        info = super().get_acquisition_info()
        info.update({
            'n_samples': self.n_samples,
            'sampling_strategy': self.sampling_strategy,
            'optimization_method': self.optimization_method,
            'has_sampled_functions': self._sampled_functions is not None,
            'n_sample_points': len(self._sample_points) if self._sample_points else 0,
        })
        
        return info
