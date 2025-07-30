"""
Multi-objective acquisition functions for Bayesian optimization.

This module implements state-of-the-art multi-objective acquisition functions:
- Expected Pareto Improvement (EPI)
- q-Noisy Expected Hypervolume Improvement (qNEHVI)
- Multi-objective variants of standard acquisition functions
- Batch optimization support for parallel evaluation

Based on:
- Keane (2006) "Statistical improvement criteria for use in multiobjective design optimization"
- Daulton et al. (2020) "Differentiable Expected Hypervolume Improvement for Parallel Multi-Objective Bayesian Optimization"
- Latest 2024-2025 research in multi-objective Bayesian optimization
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import torch
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, multivariate_normal
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from bayes_for_days.core.base import BaseAcquisitionFunction
from bayes_for_days.core.types import (
    AcquisitionValue,
    AcquisitionFunction as AcquisitionFunctionType,
    ParameterDict,
    ModelPrediction,
    ExperimentPoint,
)
from bayes_for_days.optimization.pareto import ParetoFrontManager, HypervolumeCalculator

logger = logging.getLogger(__name__)


class ExpectedParetoImprovement(BaseAcquisitionFunction):
    """
    Expected Pareto Improvement (EPI) acquisition function.
    
    Computes the expected improvement in the Pareto front by considering
    the probability that a new point will be non-dominated by the current
    Pareto front and the magnitude of improvement it provides.
    
    Based on Keane (2006) and extended for modern multi-objective BO.
    """
    
    def __init__(
        self,
        surrogate_model,
        pareto_front: Optional[List[ExperimentPoint]] = None,
        reference_point: Optional[List[float]] = None,
        xi: float = 0.01,
        **kwargs
    ):
        """
        Initialize Expected Pareto Improvement acquisition function.
        
        Args:
            surrogate_model: Multi-output surrogate model
            pareto_front: Current Pareto front points
            reference_point: Reference point for hypervolume calculation
            xi: Exploration parameter
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        self.xi = xi
        self.reference_point = reference_point
        
        # Initialize Pareto front manager
        self.pareto_manager = ParetoFrontManager(
            reference_point=reference_point,
            archive_history=False
        )
        
        if pareto_front:
            self.pareto_manager.update(pareto_front)
        
        # Cache for efficiency
        self._pareto_objectives_cache = None
        self._hypervolume_cache = None
        
        logger.info(f"Initialized EPI with {len(self.pareto_manager.pareto_front)} Pareto points")
    
    def update_pareto_front(self, new_points: List[ExperimentPoint]):
        """
        Update the Pareto front with new points.
        
        Args:
            new_points: New experiment points to consider
        """
        self.pareto_manager.update(new_points)
        self._invalidate_cache()
        
        logger.debug(f"Updated Pareto front: {len(self.pareto_manager.pareto_front)} points")
    
    def _invalidate_cache(self):
        """Invalidate cached values when Pareto front changes."""
        self._pareto_objectives_cache = None
        self._hypervolume_cache = None
    
    def _get_pareto_objectives(self) -> np.ndarray:
        """Get objectives matrix from current Pareto front."""
        if self._pareto_objectives_cache is None:
            pareto_front = self.pareto_manager.get_pareto_front()
            
            if not pareto_front or not pareto_front[0].objectives:
                return np.array([])
            
            # Extract objective values
            obj_names = list(pareto_front[0].objectives.keys())
            objectives = []
            
            for point in pareto_front:
                if point.objectives:
                    obj_values = [point.objectives.get(name, 0.0) for name in obj_names]
                    objectives.append(obj_values)
            
            self._pareto_objectives_cache = np.array(objectives)
        
        return self._pareto_objectives_cache
    
    def evaluate(
        self,
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate Expected Pareto Improvement at given parameters.
        
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
        
        # Get current Pareto objectives
        pareto_objectives = self._get_pareto_objectives()
        
        if pareto_objectives.size == 0:
            # No Pareto front yet - use standard improvement
            results = []
            for pred in predictions:
                epi_value = pred.mean + self.xi  # Simple improvement
                
                acq_value = AcquisitionValue(
                    value=epi_value,
                    function_type=AcquisitionFunctionType.EXPECTED_PARETO_IMPROVEMENT,
                    parameters={"xi": self.xi, "pareto_size": 0}
                )
                results.append(acq_value)
        else:
            # Compute EPI for each prediction
            results = []
            for pred in predictions:
                epi_value = self._compute_epi(pred, pareto_objectives)
                
                acq_value = AcquisitionValue(
                    value=epi_value,
                    function_type=AcquisitionFunctionType.EXPECTED_PARETO_IMPROVEMENT,
                    parameters={
                        "xi": self.xi,
                        "pareto_size": len(pareto_objectives),
                        "n_objectives": pareto_objectives.shape[1] if pareto_objectives.size > 0 else 0
                    }
                )
                results.append(acq_value)
        
        if single_evaluation:
            return results[0]
        return results
    
    def _compute_epi(self, prediction: ModelPrediction, pareto_objectives: np.ndarray) -> float:
        """
        Compute Expected Pareto Improvement for a single prediction.
        
        Args:
            prediction: Model prediction with mean and uncertainty
            pareto_objectives: Current Pareto front objectives (N x M matrix)
            
        Returns:
            EPI value
        """
        if not hasattr(prediction, 'mean') or not hasattr(prediction, 'std'):
            return 0.0
        
        # For multi-objective, we need to handle multiple outputs
        # Assume prediction contains multiple objectives
        if isinstance(prediction.mean, (list, np.ndarray)):
            pred_means = np.array(prediction.mean)
            pred_stds = np.array(prediction.std) if hasattr(prediction, 'std') else np.ones_like(pred_means) * 0.1
        else:
            # Single objective case - convert to multi-objective
            pred_means = np.array([prediction.mean])
            pred_stds = np.array([prediction.std])
        
        if len(pred_means) != pareto_objectives.shape[1]:
            logger.warning(f"Dimension mismatch: prediction has {len(pred_means)} objectives, "
                          f"Pareto front has {pareto_objectives.shape[1]}")
            return 0.0
        
        # Monte Carlo approximation of EPI
        n_samples = 1000
        epi_sum = 0.0
        
        for _ in range(n_samples):
            # Sample from prediction distribution
            sample_objectives = np.random.normal(pred_means, pred_stds)
            
            # Check if sample dominates any Pareto point or is non-dominated
            improvement = self._compute_pareto_improvement(sample_objectives, pareto_objectives)
            epi_sum += improvement
        
        epi_value = epi_sum / n_samples
        return max(0.0, epi_value)
    
    def _compute_pareto_improvement(
        self,
        sample_objectives: np.ndarray,
        pareto_objectives: np.ndarray
    ) -> float:
        """
        Compute improvement contribution of a sample point.
        
        Args:
            sample_objectives: Sampled objective values
            pareto_objectives: Current Pareto front objectives
            
        Returns:
            Improvement value
        """
        # Check if sample is dominated by any Pareto point
        dominated = False
        dominates_count = 0
        
        for pareto_point in pareto_objectives:
            # Check domination (assuming maximization)
            if np.all(pareto_point >= sample_objectives) and np.any(pareto_point > sample_objectives):
                dominated = True
                break
            elif np.all(sample_objectives >= pareto_point) and np.any(sample_objectives > pareto_point):
                dominates_count += 1
        
        if dominated:
            return 0.0
        
        # Compute improvement based on:
        # 1. Number of points dominated
        # 2. Distance from Pareto front
        # 3. Hypervolume contribution (simplified)
        
        domination_improvement = dominates_count
        
        # Distance-based improvement
        if len(pareto_objectives) > 0:
            distances = np.linalg.norm(pareto_objectives - sample_objectives, axis=1)
            min_distance = np.min(distances)
            distance_improvement = 1.0 / (1.0 + min_distance)  # Closer points get higher value
        else:
            distance_improvement = 1.0
        
        # Combine improvements
        total_improvement = domination_improvement + distance_improvement + self.xi
        
        return total_improvement
    
    def optimize(
        self,
        n_candidates: int = 1,
        bounds: Optional[List[tuple]] = None,
        n_restarts: int = 10,
        **kwargs
    ) -> List[ParameterDict]:
        """
        Optimize Expected Pareto Improvement to find next evaluation points.
        
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
        
        # Define objective function for scipy.optimize (negative EPI for minimization)
        def objective(x):
            param_dict = self._array_to_param_dict(x)
            epi_value = self.evaluate(param_dict)
            return -epi_value.value  # Negative for minimization
        
        # Multi-start optimization
        candidates = []
        best_results = []
        
        # Use parallel optimization for efficiency
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
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    if result.success:
                        best_results.append(result)
                except Exception as e:
                    logger.warning(f"EPI optimization failed: {e}")
        
        # Sort by objective value and select best candidates
        best_results.sort(key=lambda r: r.fun)
        
        for i in range(min(n_candidates, len(best_results))):
            result = best_results[i]
            param_dict = self._array_to_param_dict(result.x)
            candidates.append(param_dict)
        
        # Fill remaining candidates with random sampling if needed
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


class qNoisyExpectedHypervolumeImprovement(BaseAcquisitionFunction):
    """
    q-Noisy Expected Hypervolume Improvement (qNEHVI) acquisition function.
    
    State-of-the-art acquisition function for multi-objective Bayesian optimization
    with support for:
    - Batch optimization (q > 1)
    - Noisy observations
    - Efficient hypervolume computation
    - Parallel evaluation
    
    Based on Daulton et al. (2020).
    """
    
    def __init__(
        self,
        surrogate_model,
        pareto_front: Optional[List[ExperimentPoint]] = None,
        reference_point: Optional[List[float]] = None,
        noise_level: float = 0.0,
        **kwargs
    ):
        """
        Initialize qNEHVI acquisition function.
        
        Args:
            surrogate_model: Multi-output surrogate model
            pareto_front: Current Pareto front points
            reference_point: Reference point for hypervolume calculation
            noise_level: Observation noise level
            **kwargs: Additional parameters
        """
        super().__init__(surrogate_model, **kwargs)
        self.noise_level = noise_level
        self.reference_point = reference_point
        
        # Initialize hypervolume calculator
        self.hv_calculator = HypervolumeCalculator(reference_point)
        
        # Initialize Pareto front manager
        self.pareto_manager = ParetoFrontManager(
            reference_point=reference_point,
            archive_history=False
        )
        
        if pareto_front:
            self.pareto_manager.update(pareto_front)
        
        # Current hypervolume
        self.current_hv = self._compute_current_hypervolume()
        
        logger.info(f"Initialized qNEHVI with {len(self.pareto_manager.pareto_front)} Pareto points, "
                   f"HV={self.current_hv:.4f}")
    
    def _compute_current_hypervolume(self) -> float:
        """Compute hypervolume of current Pareto front."""
        pareto_front = self.pareto_manager.get_pareto_front()
        if not pareto_front:
            return 0.0
        
        return self.hv_calculator.calculate(pareto_front, self.reference_point)
    
    def update_pareto_front(self, new_points: List[ExperimentPoint]):
        """Update the Pareto front with new points."""
        self.pareto_manager.update(new_points)
        self.current_hv = self._compute_current_hypervolume()
        
        logger.debug(f"Updated Pareto front: {len(self.pareto_manager.pareto_front)} points, "
                    f"HV={self.current_hv:.4f}")
    
    def evaluate(
        self,
        parameters: Union[ParameterDict, List[ParameterDict]]
    ) -> Union[AcquisitionValue, List[AcquisitionValue]]:
        """
        Evaluate qNEHVI at given parameters.
        
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
        
        # Compute qNEHVI for batch
        qnehvi_value = self._compute_qnehvi_batch(predictions)
        
        # Create acquisition values
        results = []
        for i, pred in enumerate(predictions):
            # For batch evaluation, distribute the total qNEHVI value
            individual_value = qnehvi_value / len(predictions) if len(predictions) > 1 else qnehvi_value
            
            acq_value = AcquisitionValue(
                value=individual_value,
                function_type=AcquisitionFunctionType.Q_NOISY_EXPECTED_HYPERVOLUME_IMPROVEMENT,
                parameters={
                    "noise_level": self.noise_level,
                    "current_hv": self.current_hv,
                    "batch_size": len(predictions),
                    "pareto_size": len(self.pareto_manager.pareto_front)
                }
            )
            results.append(acq_value)
        
        if single_evaluation:
            return results[0]
        return results
    
    def _compute_qnehvi_batch(self, predictions: List[ModelPrediction]) -> float:
        """
        Compute qNEHVI for a batch of predictions.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            qNEHVI value for the batch
        """
        if not predictions:
            return 0.0
        
        # Monte Carlo approximation
        n_samples = 500  # Reduced for efficiency
        hv_improvements = []
        
        for _ in range(n_samples):
            # Sample from each prediction
            sample_objectives = []
            
            for pred in predictions:
                if isinstance(pred.mean, (list, np.ndarray)):
                    pred_means = np.array(pred.mean)
                    pred_stds = np.array(pred.std) if hasattr(pred, 'std') else np.ones_like(pred_means) * 0.1
                else:
                    pred_means = np.array([pred.mean])
                    pred_stds = np.array([pred.std])
                
                # Add noise
                if self.noise_level > 0:
                    pred_stds = np.sqrt(pred_stds**2 + self.noise_level**2)
                
                # Sample objectives
                sample = np.random.normal(pred_means, pred_stds)
                sample_objectives.append(sample)
            
            # Compute hypervolume improvement for this sample
            hv_improvement = self._compute_sample_hv_improvement(sample_objectives)
            hv_improvements.append(hv_improvement)
        
        # Return expected hypervolume improvement
        qnehvi_value = np.mean(hv_improvements)
        return max(0.0, qnehvi_value)
    
    def _compute_sample_hv_improvement(self, sample_objectives: List[np.ndarray]) -> float:
        """
        Compute hypervolume improvement for a sample.
        
        Args:
            sample_objectives: List of sampled objective vectors
            
        Returns:
            Hypervolume improvement
        """
        # Create temporary experiment points from samples
        temp_points = []
        current_pareto = self.pareto_manager.get_pareto_front()
        
        # Add current Pareto points
        temp_points.extend(current_pareto)
        
        # Add sample points
        for i, sample in enumerate(sample_objectives):
            if len(sample) > 0:
                # Create objectives dict
                obj_dict = {f"obj_{j}": float(sample[j]) for j in range(len(sample))}
                
                temp_point = ExperimentPoint(
                    parameters={f"x_{i}": 0.0},  # Dummy parameters
                    objectives=obj_dict,
                    is_feasible=True
                )
                temp_points.append(temp_point)
        
        # Compute new hypervolume
        if temp_points:
            new_hv = self.hv_calculator.calculate(temp_points, self.reference_point)
            hv_improvement = new_hv - self.current_hv
        else:
            hv_improvement = 0.0
        
        return hv_improvement
    
    def optimize(
        self,
        n_candidates: int = 1,
        bounds: Optional[List[tuple]] = None,
        n_restarts: int = 5,  # Reduced for efficiency
        **kwargs
    ) -> List[ParameterDict]:
        """
        Optimize qNEHVI to find next evaluation points.
        
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
        
        # For batch optimization, use sequential greedy approach
        candidates = []
        
        for i in range(n_candidates):
            # Define objective function
            def objective(x):
                param_dict = self._array_to_param_dict(x)
                
                # Evaluate current candidates + new candidate
                all_params = candidates + [param_dict]
                qnehvi_value = self.evaluate(all_params)
                
                if isinstance(qnehvi_value, list):
                    return -sum(acq.value for acq in qnehvi_value)
                else:
                    return -qnehvi_value.value
            
            # Optimize for this candidate
            best_result = None
            best_value = float('inf')
            
            for _ in range(n_restarts):
                x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
                
                try:
                    result = minimize(
                        objective,
                        x0,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 50}  # Reduced for efficiency
                    )
                    
                    if result.success and result.fun < best_value:
                        best_result = result
                        best_value = result.fun
                        
                except Exception as e:
                    logger.warning(f"qNEHVI optimization failed: {e}")
            
            # Add best candidate
            if best_result is not None:
                param_dict = self._array_to_param_dict(best_result.x)
                candidates.append(param_dict)
            else:
                # Fallback to random sampling
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
