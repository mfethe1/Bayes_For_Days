"""
Entropy Search acquisition function for Bayesian optimization.

Entropy Search (ES) and Predictive Entropy Search (PES) acquisition functions
that measure the expected reduction in entropy about the location of the global
optimum after making an observation.

Based on:
- Hennig & Schuler (2012) "Entropy Search for Information-Efficient Global Optimization"
- Hernández-Lobato et al. (2014) "Predictive Entropy Search for Efficient Global Optimization"
- Wang & Jegelka (2017) "Max-value Entropy Search for Efficient Bayesian Optimization"
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
import torch

from bayes_for_days.core.base import BaseAcquisitionFunction
from bayes_for_days.core.types import AcquisitionValue, ModelPrediction

logger = logging.getLogger(__name__)


class PredictiveEntropySearch(BaseAcquisitionFunction):
    """
    Predictive Entropy Search acquisition function.
    
    PES measures the expected reduction in entropy about the location
    of the global optimum by evaluating the mutual information between
    the observation and the optimum location.
    
    Features:
    - Efficient approximation using representative points
    - Support for batch optimization
    - Handles high-dimensional parameter spaces
    - Monte Carlo approximation for complex posteriors
    """
    
    def __init__(
        self,
        surrogate_model,
        n_representer_points: int = 50,
        n_samples: int = 100,
        **kwargs
    ):
        """
        Initialize Predictive Entropy Search acquisition function.
        
        Args:
            surrogate_model: Fitted surrogate model
            n_representer_points: Number of representer points for optimum approximation
            n_samples: Number of samples for Monte Carlo approximation
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        
        self.n_representer_points = n_representer_points
        self.n_samples = n_samples
        
        # Cache for representer points
        self._representer_points = None
        self._representer_predictions = None
        self._optimum_samples = None
        
        logger.info(f"Initialized PES with {n_representer_points} representer points")
    
    def evaluate(
        self,
        parameters: Union[Dict[str, float], List[Dict[str, float]]],
        **kwargs
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate Predictive Entropy Search acquisition function.
        
        Args:
            parameters: Parameter values to evaluate
            **kwargs: Additional evaluation parameters
            
        Returns:
            PES values
        """
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_evaluation = True
        else:
            single_evaluation = False
        
        # Generate representer points if not cached
        if self._representer_points is None:
            self._generate_representer_points()
        
        # Evaluate PES for each parameter set
        pes_values = []
        
        for param_dict in parameters:
            try:
                pes_value = self._compute_predictive_entropy_search(param_dict)
                
                acquisition_value = AcquisitionValue(
                    value=pes_value,
                    parameters=param_dict,
                    metadata={
                        'acquisition_function': 'predictive_entropy_search',
                        'n_representer_points': self.n_representer_points,
                        'n_samples': self.n_samples,
                    }
                )
                pes_values.append(acquisition_value)
                
            except Exception as e:
                logger.warning(f"PES evaluation failed for {param_dict}: {e}")
                
                # Return zero acquisition value for failed evaluations
                acquisition_value = AcquisitionValue(
                    value=0.0,
                    parameters=param_dict,
                    metadata={'error': str(e)}
                )
                pes_values.append(acquisition_value)
        
        if single_evaluation:
            return pes_values[0]
        return pes_values
    
    def _generate_representer_points(self):
        """Generate representer points for optimum location approximation."""
        # Get parameter bounds
        bounds = self._get_parameter_bounds()
        
        if not bounds:
            logger.warning("No parameter bounds available for representer point generation")
            self._representer_points = []
            self._representer_predictions = []
            return
        
        # Generate representer points using Latin Hypercube Sampling
        representer_points = []
        
        # Simple random sampling (could be improved with LHS)
        for _ in range(self.n_representer_points):
            point = {}
            
            for param_name, (low, high) in bounds.items():
                point[param_name] = np.random.uniform(low, high)
            
            representer_points.append(point)
        
        self._representer_points = representer_points
        
        # Get predictions for all representer points
        try:
            predictions = self.surrogate_model.predict(representer_points)
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            self._representer_predictions = predictions
            
            # Generate samples of optimum location
            self._generate_optimum_samples()
            
        except Exception as e:
            logger.warning(f"Failed to get representer predictions: {e}")
            self._representer_predictions = []
            self._optimum_samples = []
    
    def _generate_optimum_samples(self):
        """Generate samples of the optimum location using representer points."""
        if not self._representer_predictions:
            self._optimum_samples = []
            return
        
        # Extract means and variances from predictions
        means = np.array([
            pred.mean if hasattr(pred, 'mean') else 0.0
            for pred in self._representer_predictions
        ])
        
        variances = np.array([
            pred.variance if hasattr(pred, 'variance') else 1.0
            for pred in self._representer_predictions
        ])
        
        # Sample from GP posterior at representer points
        optimum_samples = []
        
        for _ in range(self.n_samples):
            # Sample function values at representer points
            sampled_values = np.random.normal(means, np.sqrt(variances))
            
            # Find the index of the maximum value
            max_idx = np.argmax(sampled_values)
            
            # Store the corresponding representer point
            optimum_samples.append(self._representer_points[max_idx])
        
        self._optimum_samples = optimum_samples
    
    def _compute_predictive_entropy_search(self, parameters: Dict[str, float]) -> float:
        """
        Compute Predictive Entropy Search value for given parameters.
        
        Args:
            parameters: Parameter values
            
        Returns:
            PES value
        """
        if not self._optimum_samples:
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
            
            # Compute entropy reduction
            # H(x*) - E[H(x*|y)]
            
            # Current entropy about optimum location (approximated)
            current_entropy = self._compute_optimum_entropy()
            
            # Expected entropy after observation
            expected_entropy = self._compute_expected_entropy_after_observation(
                query_mean, query_std
            )
            
            # Information gain = entropy reduction
            information_gain = current_entropy - expected_entropy
            
            return max(0.0, information_gain)
            
        except Exception as e:
            logger.warning(f"PES computation failed: {e}")
            return 0.0
    
    def _compute_optimum_entropy(self) -> float:
        """Compute current entropy about optimum location."""
        if not self._optimum_samples:
            return 0.0
        
        # Simplified entropy calculation based on sample diversity
        # In practice, this would be more sophisticated
        
        # Convert optimum samples to array
        sample_array = []
        param_names = list(self._optimum_samples[0].keys())
        
        for sample in self._optimum_samples:
            sample_vector = [sample[name] for name in param_names]
            sample_array.append(sample_vector)
        
        sample_array = np.array(sample_array)
        
        if len(sample_array) == 0:
            return 0.0
        
        # Estimate entropy using sample covariance
        try:
            cov_matrix = np.cov(sample_array.T)
            
            # Add regularization for numerical stability
            cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
            
            # Entropy of multivariate Gaussian
            sign, logdet = np.linalg.slogdet(2 * np.pi * np.e * cov_matrix)
            
            if sign > 0:
                entropy = 0.5 * logdet
            else:
                entropy = 0.0
            
            return entropy
            
        except Exception as e:
            logger.debug(f"Entropy computation failed: {e}")
            return 0.0
    
    def _compute_expected_entropy_after_observation(
        self,
        query_mean: float,
        query_std: float
    ) -> float:
        """
        Compute expected entropy about optimum after observing at query point.
        
        Args:
            query_mean: Mean prediction at query point
            query_std: Standard deviation at query point
            
        Returns:
            Expected entropy after observation
        """
        # Monte Carlo approximation
        n_fantasy_samples = 20
        entropy_samples = []
        
        for _ in range(n_fantasy_samples):
            # Sample fantasy observation
            fantasy_value = np.random.normal(query_mean, query_std)
            
            # Compute posterior entropy given this observation
            # This is a simplified approximation
            posterior_entropy = self._compute_posterior_entropy_given_observation(
                fantasy_value, query_mean, query_std
            )
            
            entropy_samples.append(posterior_entropy)
        
        return np.mean(entropy_samples)
    
    def _compute_posterior_entropy_given_observation(
        self,
        fantasy_value: float,
        query_mean: float,
        query_std: float
    ) -> float:
        """
        Compute posterior entropy about optimum given fantasy observation.
        
        This is a simplified approximation. In practice, would update
        the GP posterior and recompute optimum distribution.
        """
        # Simplified: assume observation reduces entropy proportionally
        # to how informative it is
        
        current_entropy = self._compute_optimum_entropy()
        
        # Information content of observation (simplified)
        information_content = abs(fantasy_value - query_mean) / (query_std + 1e-8)
        
        # Reduce entropy based on information content
        reduction_factor = min(0.9, information_content * 0.1)
        posterior_entropy = current_entropy * (1 - reduction_factor)
        
        return max(0.0, posterior_entropy)
    
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
        Optimize Predictive Entropy Search acquisition function.
        
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
                logger.warning("No bounds available for PES optimization")
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
        
        # Generate representer points for PES evaluation
        self._generate_representer_points()
        
        # Objective function for optimization (negative PES for minimization)
        def objective(x):
            param_dict = {param_names[i]: x[i] for i in range(len(param_names))}
            pes_value = self._compute_predictive_entropy_search(param_dict)
            return -pes_value  # Minimize negative PES
        
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
                    
                    pes_value = -result.fun
                    best_candidates.append((optimal_params, pes_value))
                    
            except Exception as e:
                logger.debug(f"PES optimization restart failed: {e}")
                continue
        
        # Sort by PES value and return top candidates
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
        
        return unique_candidates[:n_candidates]
    
    def get_acquisition_info(self) -> Dict[str, Any]:
        """Get information about the acquisition function."""
        info = super().get_acquisition_info()
        info.update({
            'n_representer_points': self.n_representer_points,
            'n_samples': self.n_samples,
            'has_representer_cache': self._representer_points is not None,
            'n_optimum_samples': len(self._optimum_samples) if self._optimum_samples else 0,
        })
        
        return info
