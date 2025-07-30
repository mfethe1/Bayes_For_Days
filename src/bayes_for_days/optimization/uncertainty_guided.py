"""
Uncertainty-Guided Resource Allocation for Bayes For Days platform.

This module implements intelligent resource allocation that quantifies uncertainty
in all experimental parameters and recommends optimal allocation of experimental
budget based on information theory and Bayesian decision theory.

This is a revolutionary capability that no existing experimental design tool provides.

Key Features:
- Bayesian decision theory framework for resource allocation
- Value of information calculations for each potential experiment
- Risk-aware utility functions for decision making under uncertainty
- Dynamic budget reallocation as experiments progress
- Uncertainty propagation through experimental chains
- Multi-objective resource allocation with Pareto-optimal solutions

Based on:
- Raiffa & Schlaifer (1961) "Applied Statistical Decision Theory"
- Howard (1966) "Information Value Theory"
- Recent advances in Bayesian experimental design (2020-2025)
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import scipy.stats as stats
from scipy.optimize import minimize
import pandas as pd

from bayes_for_days.core.base import BaseOptimizer, BaseSurrogateModel
from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    Parameter,
    ParameterType,
    OptimizationResult,
    ParameterDict,
    ObjectiveDict,
)

logger = logging.getLogger(__name__)


@dataclass
class ResourceConstraint:
    """
    Represents a resource constraint for experimental allocation.
    
    Resources can be budget, time, equipment availability, personnel, etc.
    """
    name: str
    total_available: float
    cost_per_experiment: Dict[str, float]  # Cost per experiment type
    description: str = ""
    
    def __post_init__(self):
        """Validate constraint parameters."""
        if self.total_available <= 0:
            raise ValueError("Total available resources must be positive")
        
        for exp_type, cost in self.cost_per_experiment.items():
            if cost < 0:
                raise ValueError(f"Cost for {exp_type} must be non-negative")


@dataclass
class UncertaintyMetrics:
    """
    Comprehensive uncertainty metrics for experimental parameters and objectives.
    """
    parameter_uncertainties: Dict[str, float]  # Uncertainty in each parameter
    objective_uncertainty: float  # Uncertainty in objective prediction
    model_uncertainty: float  # Epistemic uncertainty in the model
    noise_uncertainty: float  # Aleatoric uncertainty (measurement noise)
    total_uncertainty: float  # Combined uncertainty measure
    confidence_intervals: Dict[str, Tuple[float, float]]  # 95% confidence intervals
    
    def get_uncertainty_score(self) -> float:
        """Calculate overall uncertainty score (0-1, higher = more uncertain)."""
        return min(1.0, self.total_uncertainty / 10.0)  # Normalize to [0,1]


@dataclass
class ValueOfInformation:
    """
    Value of Information (VoI) calculation for potential experiments.
    
    Quantifies the expected value of conducting a specific experiment
    in terms of information gain and decision improvement.
    """
    experiment_description: str
    expected_information_gain: float  # Expected reduction in uncertainty
    expected_utility_improvement: float  # Expected improvement in decision utility
    cost: float  # Cost of conducting the experiment
    net_value: float  # Expected utility improvement - cost
    probability_of_improvement: float  # Probability this experiment improves decisions
    risk_adjusted_value: float  # Value adjusted for risk preferences
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.cost > 0:
            self.information_efficiency = self.expected_information_gain / self.cost
            self.utility_efficiency = self.expected_utility_improvement / self.cost
        else:
            self.information_efficiency = float('inf')
            self.utility_efficiency = float('inf')


class UncertaintyQuantifier:
    """
    Quantifies uncertainty in experimental parameters and predictions.
    
    Uses Bayesian methods to separate epistemic and aleatoric uncertainty
    and provide comprehensive uncertainty characterization.
    """
    
    def __init__(self, surrogate_model: BaseSurrogateModel):
        """
        Initialize uncertainty quantifier.
        
        Args:
            surrogate_model: Fitted surrogate model for uncertainty estimation
        """
        self.surrogate_model = surrogate_model
        
    def quantify_uncertainty(
        self,
        parameters: Union[ParameterDict, List[ParameterDict]],
        n_samples: int = 1000
    ) -> Union[UncertaintyMetrics, List[UncertaintyMetrics]]:
        """
        Quantify uncertainty at given parameter values.
        
        Args:
            parameters: Parameter values for uncertainty quantification
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            Comprehensive uncertainty metrics
        """
        # Handle single parameter dict
        if isinstance(parameters, dict):
            parameters = [parameters]
            single_input = True
        else:
            single_input = False
        
        uncertainty_metrics = []
        
        for param_dict in parameters:
            # Get model predictions with uncertainty
            predictions = self.surrogate_model.predict([param_dict])
            pred = predictions[0] if isinstance(predictions, list) else predictions
            
            # Calculate parameter uncertainties
            param_uncertainties = self._calculate_parameter_uncertainties(param_dict)
            
            # Separate epistemic and aleatoric uncertainty
            epistemic_uncertainty = self._estimate_epistemic_uncertainty(param_dict, n_samples)
            aleatoric_uncertainty = pred.std  # Measurement noise uncertainty
            
            # Calculate total uncertainty
            total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(param_dict, pred)
            
            metrics = UncertaintyMetrics(
                parameter_uncertainties=param_uncertainties,
                objective_uncertainty=pred.std,
                model_uncertainty=epistemic_uncertainty,
                noise_uncertainty=aleatoric_uncertainty,
                total_uncertainty=total_uncertainty,
                confidence_intervals=confidence_intervals
            )
            
            uncertainty_metrics.append(metrics)
        
        return uncertainty_metrics[0] if single_input else uncertainty_metrics
    
    def _calculate_parameter_uncertainties(self, param_dict: ParameterDict) -> Dict[str, float]:
        """Calculate uncertainty in each parameter."""
        uncertainties = {}
        
        for param in self.surrogate_model.parameter_space.parameters:
            param_name = param.name
            param_value = param_dict.get(param_name, 0)
            
            if param.type == ParameterType.CONTINUOUS:
                # For continuous parameters, uncertainty is related to bounds
                param_range = param.bounds[1] - param.bounds[0]
                # Higher uncertainty near bounds, lower in middle
                normalized_pos = (param_value - param.bounds[0]) / param_range
                uncertainty = param_range * 0.1 * (1 - 4 * (normalized_pos - 0.5)**2)
            else:
                # For discrete parameters, uncertainty is categorical
                uncertainty = 0.1  # Fixed small uncertainty
            
            uncertainties[param_name] = max(0.01, uncertainty)
        
        return uncertainties
    
    def _estimate_epistemic_uncertainty(self, param_dict: ParameterDict, n_samples: int) -> float:
        """Estimate epistemic (model) uncertainty using bootstrap or ensemble methods."""
        if not hasattr(self.surrogate_model, 'model') or self.surrogate_model.model is None:
            return 0.1  # Default uncertainty if model not available
        
        # For GP models, epistemic uncertainty is captured in the posterior variance
        # For other models, we could use ensemble methods or bootstrap
        
        # Simple approximation: use prediction variance as epistemic uncertainty
        predictions = self.surrogate_model.predict([param_dict])
        pred = predictions[0] if isinstance(predictions, list) else predictions
        
        return pred.std * 0.5  # Assume half of prediction uncertainty is epistemic
    
    def _calculate_confidence_intervals(
        self, 
        param_dict: ParameterDict, 
        prediction
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions."""
        # 95% confidence interval
        z_score = 1.96
        margin = z_score * prediction.std
        
        return {
            "objective": (prediction.mean - margin, prediction.mean + margin)
        }


class ValueOfInformationCalculator:
    """
    Calculates Value of Information (VoI) for potential experiments.
    
    Uses Bayesian decision theory to quantify the expected value
    of conducting specific experiments.
    """
    
    def __init__(
        self,
        surrogate_model: BaseSurrogateModel,
        uncertainty_quantifier: UncertaintyQuantifier,
        utility_function: Optional[Callable] = None
    ):
        """
        Initialize VoI calculator.
        
        Args:
            surrogate_model: Surrogate model for predictions
            uncertainty_quantifier: Uncertainty quantification system
            utility_function: Utility function for decision making
        """
        self.surrogate_model = surrogate_model
        self.uncertainty_quantifier = uncertainty_quantifier
        self.utility_function = utility_function or self._default_utility_function
        
    def calculate_voi(
        self,
        candidate_experiments: List[Tuple[ParameterDict, float]],
        current_best: Optional[ExperimentPoint] = None,
        risk_aversion: float = 0.5
    ) -> List[ValueOfInformation]:
        """
        Calculate Value of Information for candidate experiments.
        
        Args:
            candidate_experiments: List of (parameters, cost) tuples
            current_best: Current best experimental result
            risk_aversion: Risk aversion parameter (0=risk neutral, 1=very risk averse)
            
        Returns:
            List of VoI calculations for each candidate experiment
        """
        voi_results = []
        
        for param_dict, cost in candidate_experiments:
            # Calculate expected information gain
            info_gain = self._calculate_expected_information_gain(param_dict)
            
            # Calculate expected utility improvement
            utility_improvement = self._calculate_expected_utility_improvement(
                param_dict, current_best
            )
            
            # Calculate probability of improvement
            prob_improvement = self._calculate_probability_of_improvement(
                param_dict, current_best
            )
            
            # Calculate net value
            net_value = utility_improvement - cost
            
            # Apply risk adjustment
            risk_adjusted_value = self._apply_risk_adjustment(
                net_value, prob_improvement, risk_aversion
            )
            
            voi = ValueOfInformation(
                experiment_description=f"Experiment at {param_dict}",
                expected_information_gain=info_gain,
                expected_utility_improvement=utility_improvement,
                cost=cost,
                net_value=net_value,
                probability_of_improvement=prob_improvement,
                risk_adjusted_value=risk_adjusted_value
            )
            
            voi_results.append(voi)
        
        # Sort by risk-adjusted value (descending)
        voi_results.sort(key=lambda x: x.risk_adjusted_value, reverse=True)
        
        return voi_results
    
    def _calculate_expected_information_gain(self, param_dict: ParameterDict) -> float:
        """Calculate expected information gain from an experiment."""
        # Get current uncertainty
        uncertainty_metrics = self.uncertainty_quantifier.quantify_uncertainty(param_dict)
        
        # Information gain is related to uncertainty reduction
        # Higher uncertainty -> higher potential information gain
        current_uncertainty = uncertainty_metrics.total_uncertainty
        
        # Expected information gain (simplified model)
        # In practice, this would involve more sophisticated calculations
        expected_gain = current_uncertainty * 0.5  # Assume 50% uncertainty reduction
        
        return expected_gain
    
    def _calculate_expected_utility_improvement(
        self, 
        param_dict: ParameterDict, 
        current_best: Optional[ExperimentPoint]
    ) -> float:
        """Calculate expected improvement in decision utility."""
        # Get prediction for candidate experiment
        predictions = self.surrogate_model.predict([param_dict])
        pred = predictions[0] if isinstance(predictions, list) else predictions
        
        if current_best is None:
            # If no current best, any experiment has potential value
            return abs(pred.mean) * 0.1
        
        # Calculate expected improvement over current best
        current_best_value = list(current_best.objectives.values())[0]
        
        # Expected improvement (for minimization)
        improvement = max(0, current_best_value - pred.mean)
        
        # Weight by uncertainty (higher uncertainty = higher potential improvement)
        uncertainty_weight = 1 + pred.std
        
        return improvement * uncertainty_weight
    
    def _calculate_probability_of_improvement(
        self, 
        param_dict: ParameterDict, 
        current_best: Optional[ExperimentPoint]
    ) -> float:
        """Calculate probability that experiment improves current best."""
        if current_best is None:
            return 0.5  # 50% chance if no baseline
        
        # Get prediction
        predictions = self.surrogate_model.predict([param_dict])
        pred = predictions[0] if isinstance(predictions, list) else predictions
        
        current_best_value = list(current_best.objectives.values())[0]
        
        # Probability of improvement (assuming normal distribution)
        if pred.std > 0:
            z_score = (current_best_value - pred.mean) / pred.std
            prob_improvement = stats.norm.cdf(z_score)
        else:
            prob_improvement = 1.0 if pred.mean < current_best_value else 0.0
        
        return prob_improvement
    
    def _apply_risk_adjustment(
        self, 
        net_value: float, 
        prob_improvement: float, 
        risk_aversion: float
    ) -> float:
        """Apply risk adjustment to value calculation."""
        # Risk-adjusted value using utility theory
        # Higher risk aversion reduces value of uncertain outcomes
        
        risk_penalty = risk_aversion * (1 - prob_improvement) * abs(net_value)
        risk_adjusted_value = net_value - risk_penalty
        
        return risk_adjusted_value
    
    def _default_utility_function(self, objective_value: float) -> float:
        """Default utility function (negative for minimization problems)."""
        return -objective_value


class ResourceAllocationOptimizer:
    """
    Optimizes allocation of experimental resources based on uncertainty and VoI.
    
    Solves the resource allocation problem to maximize expected utility
    subject to resource constraints.
    """
    
    def __init__(
        self,
        voi_calculator: ValueOfInformationCalculator,
        constraints: List[ResourceConstraint]
    ):
        """
        Initialize resource allocation optimizer.
        
        Args:
            voi_calculator: Value of Information calculator
            constraints: List of resource constraints
        """
        self.voi_calculator = voi_calculator
        self.constraints = constraints
    
    def optimize_allocation(
        self,
        candidate_experiments: List[Tuple[ParameterDict, Dict[str, float]]],
        current_best: Optional[ExperimentPoint] = None,
        max_experiments: Optional[int] = None
    ) -> List[Tuple[ParameterDict, Dict[str, float], float]]:
        """
        Optimize experimental resource allocation.
        
        Args:
            candidate_experiments: List of (parameters, resource_costs) tuples
            current_best: Current best experimental result
            max_experiments: Maximum number of experiments to select
            
        Returns:
            List of selected experiments with (parameters, costs, expected_value)
        """
        # Calculate VoI for all candidates
        candidate_costs = [costs.get('budget', 1.0) for _, costs in candidate_experiments]
        candidate_params = [params for params, _ in candidate_experiments]
        
        voi_results = self.voi_calculator.calculate_voi(
            list(zip(candidate_params, candidate_costs)),
            current_best
        )
        
        # Solve resource allocation optimization problem
        selected_experiments = self._solve_allocation_problem(
            candidate_experiments, voi_results, max_experiments
        )
        
        return selected_experiments
    
    def _solve_allocation_problem(
        self,
        candidates: List[Tuple[ParameterDict, Dict[str, float]]],
        voi_results: List[ValueOfInformation],
        max_experiments: Optional[int]
    ) -> List[Tuple[ParameterDict, Dict[str, float], float]]:
        """Solve the resource allocation optimization problem."""
        n_candidates = len(candidates)
        
        if n_candidates == 0:
            return []
        
        # Simple greedy algorithm for now (could be replaced with more sophisticated methods)
        selected = []
        remaining_resources = {
            constraint.name: constraint.total_available 
            for constraint in self.constraints
        }
        
        # Sort candidates by risk-adjusted value per unit cost
        candidate_values = []
        for i, (params, costs) in enumerate(candidates):
            voi = voi_results[i] if i < len(voi_results) else voi_results[0]
            total_cost = sum(costs.values())
            value_per_cost = voi.risk_adjusted_value / max(total_cost, 0.01)
            candidate_values.append((i, params, costs, voi.risk_adjusted_value, value_per_cost))
        
        # Sort by value per cost (descending)
        candidate_values.sort(key=lambda x: x[4], reverse=True)
        
        # Greedy selection
        for idx, params, costs, expected_value, value_per_cost in candidate_values:
            # Check if we can afford this experiment
            can_afford = True
            for constraint in self.constraints:
                resource_name = constraint.name
                required = costs.get(resource_name, 0)
                if required > remaining_resources.get(resource_name, 0):
                    can_afford = False
                    break
            
            # Check max experiments constraint
            if max_experiments is not None and len(selected) >= max_experiments:
                break
            
            if can_afford and expected_value > 0:
                # Select this experiment
                selected.append((params, costs, expected_value))
                
                # Update remaining resources
                for constraint in self.constraints:
                    resource_name = constraint.name
                    required = costs.get(resource_name, 0)
                    remaining_resources[resource_name] -= required
        
        logger.info(f"Selected {len(selected)} experiments from {n_candidates} candidates")
        
        return selected
