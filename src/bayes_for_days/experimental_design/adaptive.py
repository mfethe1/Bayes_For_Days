"""
Bayesian Adaptive Experimental Design (BAED) for optimal experiment selection.

This module implements information-theoretic experimental design with:
- Expected information gain maximization
- Utility functions for different design objectives
- Adaptive design updating strategies
- Integration with Bayesian optimization framework

Based on:
- Chaloner & Verdinelli (1995) "Bayesian experimental design: A review"
- Ryan et al. (2016) "A review of modern computational algorithms for Bayesian optimal design"
- Latest 2024-2025 research in adaptive experimental design
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.stats import entropy
from scipy.special import logsumexp
import torch
from abc import ABC, abstractmethod

from bayes_for_days.experimental_design.variables import ExperimentalVariable
from bayes_for_days.experimental_design.design_strategies import DesignStrategy
from bayes_for_days.core.types import ExperimentPoint, ModelPrediction

logger = logging.getLogger(__name__)


class UtilityFunction(ABC):
    """
    Abstract base class for utility functions in BAED.
    
    Utility functions quantify the value of information gained
    from conducting an experiment at specific parameter values.
    """
    
    @abstractmethod
    def evaluate(
        self,
        candidate_params: Dict[str, float],
        model_prediction: ModelPrediction,
        current_data: List[ExperimentPoint],
        **kwargs
    ) -> float:
        """
        Evaluate utility of conducting experiment at candidate parameters.
        
        Args:
            candidate_params: Candidate experimental parameters
            model_prediction: Model prediction at candidate parameters
            current_data: Current experimental data
            **kwargs: Additional parameters
            
        Returns:
            Utility value (higher is better)
        """
        pass


class ExpectedInformationGain(UtilityFunction):
    """
    Expected Information Gain utility function.
    
    Computes the expected reduction in uncertainty about model parameters
    or predictions by conducting an experiment at the candidate location.
    """
    
    def __init__(
        self,
        information_type: str = "mutual",
        n_samples: int = 100,
        **kwargs
    ):
        """
        Initialize Expected Information Gain utility.
        
        Args:
            information_type: Type of information ('mutual', 'entropy_reduction')
            n_samples: Number of Monte Carlo samples for approximation
            **kwargs: Additional parameters
        """
        self.information_type = information_type
        self.n_samples = n_samples
        self.config = kwargs
        
        logger.info(f"Initialized EIG utility with type: {information_type}")
    
    def evaluate(
        self,
        candidate_params: Dict[str, float],
        model_prediction: ModelPrediction,
        current_data: List[ExperimentPoint],
        **kwargs
    ) -> float:
        """
        Evaluate expected information gain.
        
        Args:
            candidate_params: Candidate experimental parameters
            model_prediction: Model prediction at candidate parameters
            current_data: Current experimental data
            **kwargs: Additional parameters
            
        Returns:
            Expected information gain value
        """
        if self.information_type == "mutual":
            return self._compute_mutual_information(
                candidate_params, model_prediction, current_data, **kwargs
            )
        elif self.information_type == "entropy_reduction":
            return self._compute_entropy_reduction(
                candidate_params, model_prediction, current_data, **kwargs
            )
        else:
            raise ValueError(f"Unknown information type: {self.information_type}")
    
    def _compute_mutual_information(
        self,
        candidate_params: Dict[str, float],
        model_prediction: ModelPrediction,
        current_data: List[ExperimentPoint],
        **kwargs
    ) -> float:
        """
        Compute mutual information between parameters and potential observation.
        
        I(θ; y) = H(y) - H(y|θ)
        """
        # Monte Carlo approximation of mutual information
        mi_samples = []
        
        for _ in range(self.n_samples):
            # Sample potential observation from predictive distribution
            if hasattr(model_prediction, 'mean') and hasattr(model_prediction, 'std'):
                y_sample = np.random.normal(model_prediction.mean, model_prediction.std)
            else:
                y_sample = 0.0
            
            # Compute log likelihood ratio (simplified)
            # In practice, this would involve sampling from posterior
            log_ratio = self._compute_log_likelihood_ratio(
                candidate_params, y_sample, current_data
            )
            mi_samples.append(log_ratio)
        
        # Expected mutual information
        mutual_info = np.mean(mi_samples)
        return max(0.0, mutual_info)
    
    def _compute_entropy_reduction(
        self,
        candidate_params: Dict[str, float],
        model_prediction: ModelPrediction,
        current_data: List[ExperimentPoint],
        **kwargs
    ) -> float:
        """
        Compute expected entropy reduction in posterior distribution.
        
        ER = H(θ|D) - E[H(θ|D,y)]
        """
        # Current posterior entropy (approximated)
        current_entropy = self._estimate_posterior_entropy(current_data)
        
        # Expected posterior entropy after new observation
        expected_new_entropy = 0.0
        
        for _ in range(self.n_samples):
            # Sample potential observation
            if hasattr(model_prediction, 'mean') and hasattr(model_prediction, 'std'):
                y_sample = np.random.normal(model_prediction.mean, model_prediction.std)
            else:
                y_sample = 0.0
            
            # Estimate posterior entropy after observing y_sample
            new_entropy = self._estimate_posterior_entropy_with_new_data(
                current_data, candidate_params, y_sample
            )
            expected_new_entropy += new_entropy
        
        expected_new_entropy /= self.n_samples
        
        # Entropy reduction
        entropy_reduction = current_entropy - expected_new_entropy
        return max(0.0, entropy_reduction)
    
    def _compute_log_likelihood_ratio(
        self,
        params: Dict[str, float],
        observation: float,
        data: List[ExperimentPoint]
    ) -> float:
        """Compute log likelihood ratio (simplified approximation)."""
        # This is a simplified version - in practice would use actual model
        # For now, return a reasonable approximation based on prediction uncertainty
        return abs(observation) * 0.1  # Placeholder
    
    def _estimate_posterior_entropy(self, data: List[ExperimentPoint]) -> float:
        """Estimate entropy of current posterior distribution."""
        # Simplified entropy estimation based on data size
        # In practice, would use actual posterior samples or approximation
        n_data = len(data)
        if n_data == 0:
            return 10.0  # High entropy for no data
        else:
            # Entropy decreases with more data
            return max(0.1, 10.0 / np.sqrt(n_data))
    
    def _estimate_posterior_entropy_with_new_data(
        self,
        current_data: List[ExperimentPoint],
        new_params: Dict[str, float],
        new_observation: float
    ) -> float:
        """Estimate posterior entropy after adding new data point."""
        # Simplified - entropy should decrease with new informative data
        current_entropy = self._estimate_posterior_entropy(current_data)
        information_content = abs(new_observation) * 0.1  # Simplified
        new_entropy = current_entropy * (1 - information_content / 10.0)
        return max(0.01, new_entropy)


class ModelDiscrimination(UtilityFunction):
    """
    Model discrimination utility function.
    
    Maximizes the ability to distinguish between competing models
    by selecting experiments that lead to different predictions.
    """
    
    def __init__(
        self,
        competing_models: List[Any],
        discrimination_metric: str = "kl_divergence",
        **kwargs
    ):
        """
        Initialize model discrimination utility.
        
        Args:
            competing_models: List of competing models to discriminate
            discrimination_metric: Metric for discrimination ('kl_divergence', 'hellinger')
            **kwargs: Additional parameters
        """
        self.competing_models = competing_models
        self.discrimination_metric = discrimination_metric
        self.config = kwargs
        
        logger.info(f"Initialized model discrimination with {len(competing_models)} models")
    
    def evaluate(
        self,
        candidate_params: Dict[str, float],
        model_prediction: ModelPrediction,
        current_data: List[ExperimentPoint],
        **kwargs
    ) -> float:
        """
        Evaluate model discrimination utility.
        
        Args:
            candidate_params: Candidate experimental parameters
            model_prediction: Prediction from primary model
            current_data: Current experimental data
            **kwargs: Additional parameters
            
        Returns:
            Model discrimination utility value
        """
        if len(self.competing_models) < 2:
            return 0.0
        
        # Get predictions from all competing models
        model_predictions = []
        for model in self.competing_models:
            try:
                pred = model.predict(candidate_params)
                model_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
                continue
        
        if len(model_predictions) < 2:
            return 0.0
        
        # Compute discrimination metric
        if self.discrimination_metric == "kl_divergence":
            return self._compute_kl_discrimination(model_predictions)
        elif self.discrimination_metric == "hellinger":
            return self._compute_hellinger_discrimination(model_predictions)
        else:
            raise ValueError(f"Unknown discrimination metric: {self.discrimination_metric}")
    
    def _compute_kl_discrimination(self, predictions: List[ModelPrediction]) -> float:
        """Compute KL divergence-based discrimination."""
        total_discrimination = 0.0
        n_pairs = 0
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i = predictions[i]
                pred_j = predictions[j]
                
                # Compute KL divergence between predictions (simplified)
                if hasattr(pred_i, 'mean') and hasattr(pred_j, 'mean'):
                    mean_diff = abs(pred_i.mean - pred_j.mean)
                    std_i = getattr(pred_i, 'std', 0.1)
                    std_j = getattr(pred_j, 'std', 0.1)
                    
                    # Simplified KL divergence approximation
                    kl_div = mean_diff / (std_i + std_j + 1e-8)
                    total_discrimination += kl_div
                    n_pairs += 1
        
        return total_discrimination / max(1, n_pairs)
    
    def _compute_hellinger_discrimination(self, predictions: List[ModelPrediction]) -> float:
        """Compute Hellinger distance-based discrimination."""
        total_discrimination = 0.0
        n_pairs = 0
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                pred_i = predictions[i]
                pred_j = predictions[j]
                
                # Compute Hellinger distance (simplified)
                if hasattr(pred_i, 'mean') and hasattr(pred_j, 'mean'):
                    mean_diff = abs(pred_i.mean - pred_j.mean)
                    hellinger = np.sqrt(mean_diff / 2.0)
                    total_discrimination += hellinger
                    n_pairs += 1
        
        return total_discrimination / max(1, n_pairs)


class BayesianAdaptiveDesign(DesignStrategy):
    """
    Bayesian Adaptive Experimental Design (BAED) strategy.
    
    Sequentially selects experiments to maximize information gain
    or other utility functions using Bayesian inference.
    
    Features:
    - Multiple utility functions (EIG, model discrimination)
    - Adaptive design updating
    - Integration with surrogate models
    - Batch experiment selection
    """
    
    def __init__(
        self,
        variables: List[ExperimentalVariable],
        surrogate_model: Any,
        utility_function: UtilityFunction,
        acquisition_optimizer: str = "differential_evolution",
        n_candidates: int = 1000,
        batch_size: int = 1,
        **kwargs
    ):
        """
        Initialize Bayesian Adaptive Design.
        
        Args:
            variables: List of experimental variables
            surrogate_model: Surrogate model for predictions
            utility_function: Utility function for experiment selection
            acquisition_optimizer: Optimizer for utility maximization
            n_candidates: Number of candidate points to consider
            batch_size: Number of experiments to select simultaneously
            **kwargs: Additional parameters
        """
        super().__init__(variables, **kwargs)
        
        self.surrogate_model = surrogate_model
        self.utility_function = utility_function
        self.acquisition_optimizer = acquisition_optimizer
        self.n_candidates = n_candidates
        self.batch_size = batch_size
        
        # Current experimental data
        self.current_data: List[ExperimentPoint] = []
        
        # Optimization history
        self.design_history: List[np.ndarray] = []
        self.utility_history: List[float] = []
        
        logger.info(f"Initialized BAED with {utility_function.__class__.__name__} utility")
    
    def update_data(self, new_data: List[ExperimentPoint]):
        """
        Update current experimental data.
        
        Args:
            new_data: New experimental data points
        """
        self.current_data.extend(new_data)
        
        # Update surrogate model if it supports incremental learning
        if hasattr(self.surrogate_model, 'update'):
            self.surrogate_model.update(new_data)
        elif hasattr(self.surrogate_model, 'fit'):
            # Refit with all data
            self.surrogate_model.fit(self.current_data)
        
        logger.info(f"Updated BAED with {len(new_data)} new data points")
    
    def generate_design(self, n_experiments: int, **kwargs) -> np.ndarray:
        """
        Generate adaptive experimental design.
        
        Args:
            n_experiments: Number of experiments to generate
            **kwargs: Additional parameters
            
        Returns:
            Adaptive design matrix
        """
        design_points = []
        
        # Generate experiments sequentially or in batches
        n_batches = (n_experiments + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            remaining_experiments = min(self.batch_size, n_experiments - len(design_points))
            
            if remaining_experiments <= 0:
                break
            
            # Select next batch of experiments
            batch_points = self._select_next_batch(remaining_experiments)
            design_points.extend(batch_points)
            
            # Simulate adding these points to data (for next batch selection)
            # In practice, you would conduct experiments and get real results
            simulated_data = self._simulate_experiment_results(batch_points)
            self.current_data.extend(simulated_data)
        
        design_matrix = np.array(design_points)
        
        logger.info(f"Generated adaptive design with {len(design_points)} experiments")
        
        return design_matrix
    
    def _select_next_batch(self, batch_size: int) -> List[List[float]]:
        """
        Select next batch of experiments using utility maximization.
        
        Args:
            batch_size: Number of experiments to select
            
        Returns:
            List of selected experiment parameter vectors
        """
        # Generate candidate points
        candidate_points = self._generate_candidate_points()
        
        # Evaluate utility for each candidate
        utilities = []
        for candidate in candidate_points:
            candidate_dict = self._array_to_param_dict(candidate)
            
            # Get model prediction
            try:
                prediction = self.surrogate_model.predict(candidate_dict)
                if isinstance(prediction, list):
                    prediction = prediction[0]  # Take first prediction if multi-output
                
                # Evaluate utility
                utility = self.utility_function.evaluate(
                    candidate_dict, prediction, self.current_data
                )
                utilities.append(utility)
                
            except Exception as e:
                logger.warning(f"Utility evaluation failed: {e}")
                utilities.append(0.0)
        
        # Select top candidates
        utilities = np.array(utilities)
        top_indices = np.argsort(utilities)[-batch_size:]
        
        selected_points = [candidate_points[i].tolist() for i in top_indices]
        selected_utilities = [utilities[i] for i in top_indices]
        
        # Store history
        self.utility_history.extend(selected_utilities)
        
        logger.debug(f"Selected batch with utilities: {selected_utilities}")
        
        return selected_points
    
    def _generate_candidate_points(self) -> np.ndarray:
        """Generate candidate points for utility evaluation."""
        candidate_points = []
        
        for _ in range(self.n_candidates):
            point = []
            for var in self.variables:
                if var.bounds:
                    low, high = var.bounds
                    value = np.random.uniform(low, high)
                elif var.baseline_value is not None:
                    # Sample around baseline
                    std = abs(var.baseline_value) * 0.1
                    value = np.random.normal(var.baseline_value, std)
                else:
                    value = np.random.normal(0, 1)
                
                point.append(value)
            
            candidate_points.append(point)
        
        return np.array(candidate_points)
    
    def _array_to_param_dict(self, param_array: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        param_dict = {}
        for i, var in enumerate(self.variables):
            if i < len(param_array):
                param_dict[var.name] = float(param_array[i])
        return param_dict
    
    def _simulate_experiment_results(self, experiment_points: List[List[float]]) -> List[ExperimentPoint]:
        """
        Simulate experiment results for design planning.
        
        In practice, this would be replaced by actual experiments.
        """
        simulated_data = []
        
        for point in experiment_points:
            param_dict = self._array_to_param_dict(np.array(point))
            
            # Simulate result using model prediction + noise
            try:
                prediction = self.surrogate_model.predict(param_dict)
                if isinstance(prediction, list):
                    prediction = prediction[0]
                
                # Add noise to prediction
                if hasattr(prediction, 'mean') and hasattr(prediction, 'std'):
                    simulated_result = np.random.normal(prediction.mean, prediction.std)
                else:
                    simulated_result = np.random.normal(0, 1)
                
                exp_point = ExperimentPoint(
                    parameters=param_dict,
                    objectives={"objective": simulated_result},
                    is_feasible=True,
                    metadata={"simulated": True}
                )
                simulated_data.append(exp_point)
                
            except Exception as e:
                logger.warning(f"Simulation failed: {e}")
        
        return simulated_data
    
    def get_design_statistics(self) -> Dict[str, Any]:
        """Get statistics about the adaptive design process."""
        return {
            'n_experiments_conducted': len(self.current_data),
            'n_design_iterations': len(self.design_history),
            'utility_history': self.utility_history.copy(),
            'mean_utility': np.mean(self.utility_history) if self.utility_history else 0.0,
            'utility_trend': np.polyfit(range(len(self.utility_history)), self.utility_history, 1)[0] 
                           if len(self.utility_history) > 1 else 0.0,
            'batch_size': self.batch_size,
            'n_candidates': self.n_candidates,
        }
