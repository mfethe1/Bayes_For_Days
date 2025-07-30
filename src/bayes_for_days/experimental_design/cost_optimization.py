"""
Cost-aware and resource-optimized experimental design.

This module implements experimental design strategies that consider:
- Reagent costs and material consumption
- Instrument time and availability
- Batch optimization for efficiency
- Resource allocation strategies
- Multi-objective cost-performance trade-offs

Based on:
- Atkinson et al. (2007) "Optimum Experimental Designs"
- Industrial design of experiments best practices
- Laboratory resource management principles
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from scipy.optimize import minimize, differential_evolution
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from bayes_for_days.experimental_design.variables import ExperimentalVariable
from bayes_for_days.experimental_design.design_strategies import DesignStrategy
from bayes_for_days.experimental_design.laboratory_constraints import StockSolution

logger = logging.getLogger(__name__)


@dataclass
class ReagentCost:
    """
    Reagent cost information for cost-aware design.
    
    Tracks costs, availability, and consumption rates
    for experimental reagents.
    """
    name: str
    cost_per_unit: float  # Cost per unit (e.g., $/mL, $/g)
    unit: str  # Unit of measurement
    available_quantity: float
    minimum_order_quantity: float = 0.0
    lead_time_days: int = 0
    shelf_life_days: Optional[int] = None
    waste_factor: float = 1.1  # Factor accounting for waste (>1.0)
    
    def calculate_cost(self, required_quantity: float) -> float:
        """Calculate total cost for required quantity including waste."""
        actual_quantity = required_quantity * self.waste_factor
        return actual_quantity * self.cost_per_unit
    
    def is_available(self, required_quantity: float) -> bool:
        """Check if required quantity is available."""
        actual_quantity = required_quantity * self.waste_factor
        return actual_quantity <= self.available_quantity


@dataclass
class InstrumentTime:
    """
    Instrument time and availability information.
    
    Tracks instrument usage costs, availability windows,
    and scheduling constraints.
    """
    instrument_name: str
    cost_per_hour: float
    available_hours_per_day: float = 8.0
    setup_time_hours: float = 0.5
    cleanup_time_hours: float = 0.5
    batch_capacity: int = 1  # Number of samples per batch
    maintenance_days: List[str] = field(default_factory=list)
    
    def calculate_time_cost(self, n_experiments: int, time_per_experiment: float) -> Tuple[float, float]:
        """
        Calculate total time and cost for experiments.
        
        Args:
            n_experiments: Number of experiments
            time_per_experiment: Time per experiment (hours)
            
        Returns:
            Tuple of (total_time_hours, total_cost)
        """
        n_batches = (n_experiments + self.batch_capacity - 1) // self.batch_capacity
        
        # Total time includes setup, experiments, and cleanup for each batch
        experiment_time = n_experiments * time_per_experiment
        overhead_time = n_batches * (self.setup_time_hours + self.cleanup_time_hours)
        total_time = experiment_time + overhead_time
        
        total_cost = total_time * self.cost_per_hour
        
        return total_time, total_cost


class CostFunction(ABC):
    """
    Abstract base class for cost functions in experimental design.
    
    Defines interface for computing costs of experimental designs
    considering various resource constraints.
    """
    
    @abstractmethod
    def evaluate_cost(
        self,
        design_matrix: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> float:
        """
        Evaluate total cost of experimental design.
        
        Args:
            design_matrix: Design matrix (n_experiments x n_variables)
            variable_names: Names of variables
            **kwargs: Additional parameters
            
        Returns:
            Total cost
        """
        pass
    
    @abstractmethod
    def get_cost_breakdown(
        self,
        design_matrix: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """
        Get detailed cost breakdown.
        
        Args:
            design_matrix: Design matrix
            variable_names: Names of variables
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with cost breakdown by category
        """
        pass


class ReagentCostFunction(CostFunction):
    """
    Cost function based on reagent consumption and costs.
    
    Calculates costs based on reagent usage for each experiment
    in the design matrix.
    """
    
    def __init__(
        self,
        reagent_costs: Dict[str, ReagentCost],
        consumption_rates: Dict[str, float],  # variable_name -> consumption per unit
        **kwargs
    ):
        """
        Initialize reagent cost function.
        
        Args:
            reagent_costs: Dictionary of reagent cost information
            consumption_rates: Consumption rates for each variable
            **kwargs: Additional parameters
        """
        self.reagent_costs = reagent_costs
        self.consumption_rates = consumption_rates
        self.config = kwargs
        
        logger.info(f"Initialized reagent cost function with {len(reagent_costs)} reagents")
    
    def evaluate_cost(
        self,
        design_matrix: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> float:
        """Evaluate total reagent cost."""
        total_cost = 0.0
        
        for j, var_name in enumerate(variable_names):
            if var_name in self.consumption_rates and j < design_matrix.shape[1]:
                # Calculate total consumption for this variable
                consumption_per_unit = self.consumption_rates[var_name]
                total_consumption = np.sum(design_matrix[:, j] * consumption_per_unit)
                
                # Find corresponding reagent cost
                reagent_name = var_name  # Assume variable name matches reagent name
                if reagent_name in self.reagent_costs:
                    reagent_cost = self.reagent_costs[reagent_name]
                    cost = reagent_cost.calculate_cost(total_consumption)
                    total_cost += cost
        
        return total_cost
    
    def get_cost_breakdown(
        self,
        design_matrix: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Get detailed reagent cost breakdown."""
        cost_breakdown = {}
        
        for j, var_name in enumerate(variable_names):
            if var_name in self.consumption_rates and j < design_matrix.shape[1]:
                consumption_per_unit = self.consumption_rates[var_name]
                total_consumption = np.sum(design_matrix[:, j] * consumption_per_unit)
                
                reagent_name = var_name
                if reagent_name in self.reagent_costs:
                    reagent_cost = self.reagent_costs[reagent_name]
                    cost = reagent_cost.calculate_cost(total_consumption)
                    cost_breakdown[f"reagent_{reagent_name}"] = cost
        
        return cost_breakdown


class InstrumentCostFunction(CostFunction):
    """
    Cost function based on instrument time and usage.
    
    Calculates costs based on instrument time requirements
    for experimental designs.
    """
    
    def __init__(
        self,
        instrument_times: Dict[str, InstrumentTime],
        experiment_durations: Dict[str, float],  # variable_name -> time per experiment
        **kwargs
    ):
        """
        Initialize instrument cost function.
        
        Args:
            instrument_times: Dictionary of instrument time information
            experiment_durations: Time requirements for each variable/experiment type
            **kwargs: Additional parameters
        """
        self.instrument_times = instrument_times
        self.experiment_durations = experiment_durations
        self.config = kwargs
        
        logger.info(f"Initialized instrument cost function with {len(instrument_times)} instruments")
    
    def evaluate_cost(
        self,
        design_matrix: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> float:
        """Evaluate total instrument cost."""
        total_cost = 0.0
        n_experiments = design_matrix.shape[0]
        
        # Calculate cost for each instrument
        for instrument_name, instrument_time in self.instrument_times.items():
            # Determine time per experiment (simplified)
            avg_time_per_exp = np.mean(list(self.experiment_durations.values()))
            
            total_time, cost = instrument_time.calculate_time_cost(
                n_experiments, avg_time_per_exp
            )
            total_cost += cost
        
        return total_cost
    
    def get_cost_breakdown(
        self,
        design_matrix: np.ndarray,
        variable_names: List[str],
        **kwargs
    ) -> Dict[str, float]:
        """Get detailed instrument cost breakdown."""
        cost_breakdown = {}
        n_experiments = design_matrix.shape[0]
        
        for instrument_name, instrument_time in self.instrument_times.items():
            avg_time_per_exp = np.mean(list(self.experiment_durations.values()))
            
            total_time, cost = instrument_time.calculate_time_cost(
                n_experiments, avg_time_per_exp
            )
            
            cost_breakdown[f"instrument_{instrument_name}"] = cost
            cost_breakdown[f"time_{instrument_name}_hours"] = total_time
        
        return cost_breakdown


class CostAwareDesign(DesignStrategy):
    """
    Cost-aware experimental design strategy.
    
    Optimizes experimental designs considering both statistical
    efficiency and resource costs.
    
    Features:
    - Multi-objective optimization (efficiency vs. cost)
    - Resource constraint handling
    - Batch optimization for cost reduction
    - Pareto-optimal design selection
    """
    
    def __init__(
        self,
        variables: List[ExperimentalVariable],
        cost_functions: List[CostFunction],
        efficiency_function: Optional[Callable] = None,
        cost_weight: float = 0.5,
        max_budget: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize cost-aware design.
        
        Args:
            variables: List of experimental variables
            cost_functions: List of cost functions to consider
            efficiency_function: Function to evaluate design efficiency
            cost_weight: Weight for cost vs. efficiency trade-off (0-1)
            max_budget: Maximum allowed budget
            **kwargs: Additional parameters
        """
        super().__init__(variables, **kwargs)
        
        self.cost_functions = cost_functions
        self.efficiency_function = efficiency_function or self._default_efficiency_function
        self.cost_weight = cost_weight
        self.max_budget = max_budget
        
        logger.info(f"Initialized cost-aware design with {len(cost_functions)} cost functions")
    
    def _default_efficiency_function(self, design_matrix: np.ndarray) -> float:
        """Default efficiency function based on design size."""
        # Simple efficiency based on number of experiments and space-filling
        n_experiments, n_variables = design_matrix.shape
        
        if n_experiments <= n_variables:
            return 0.0
        
        # Space-filling efficiency (simplified)
        min_distances = []
        for i in range(n_experiments):
            distances = []
            for j in range(n_experiments):
                if i != j:
                    dist = np.linalg.norm(design_matrix[i] - design_matrix[j])
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))
        
        # Higher minimum distance = better space-filling
        efficiency = np.mean(min_distances) if min_distances else 0.0
        return efficiency
    
    def generate_design(self, n_experiments: int, **kwargs) -> np.ndarray:
        """
        Generate cost-aware experimental design.
        
        Args:
            n_experiments: Number of experiments to generate
            **kwargs: Additional parameters
            
        Returns:
            Cost-optimized design matrix
        """
        # Generate initial candidate designs
        candidate_designs = self._generate_candidate_designs(n_experiments)
        
        # Evaluate cost-efficiency trade-offs
        best_design = None
        best_score = float('-inf')
        
        for design in candidate_designs:
            # Check budget constraint
            if self.max_budget is not None:
                total_cost = self._evaluate_total_cost(design)
                if total_cost > self.max_budget:
                    continue
            
            # Evaluate combined score
            score = self._evaluate_design_score(design)
            
            if score > best_score:
                best_score = score
                best_design = design
        
        if best_design is None:
            logger.warning("No feasible design found within budget constraints")
            # Return simple design as fallback
            best_design = self._generate_simple_design(n_experiments)
        
        logger.info(f"Generated cost-aware design with score: {best_score:.4f}")
        
        return best_design
    
    def _generate_candidate_designs(self, n_experiments: int) -> List[np.ndarray]:
        """Generate candidate designs for evaluation."""
        candidates = []
        
        # Generate multiple random designs
        for _ in range(10):  # Generate 10 candidates
            design = []
            for _ in range(n_experiments):
                experiment = []
                for var in self.variables:
                    if var.bounds:
                        low, high = var.bounds
                        value = np.random.uniform(low, high)
                    else:
                        value = np.random.normal(0, 1)
                    experiment.append(value)
                design.append(experiment)
            
            candidates.append(np.array(design))
        
        return candidates
    
    def _generate_simple_design(self, n_experiments: int) -> np.ndarray:
        """Generate simple fallback design."""
        design = []
        for _ in range(n_experiments):
            experiment = []
            for var in self.variables:
                if var.bounds:
                    low, high = var.bounds
                    value = (low + high) / 2  # Use midpoint
                elif var.baseline_value is not None:
                    value = var.baseline_value
                else:
                    value = 0.0
                experiment.append(value)
            design.append(experiment)
        
        return np.array(design)
    
    def _evaluate_total_cost(self, design_matrix: np.ndarray) -> float:
        """Evaluate total cost of design."""
        total_cost = 0.0
        
        for cost_function in self.cost_functions:
            cost = cost_function.evaluate_cost(design_matrix, self.variable_names)
            total_cost += cost
        
        return total_cost
    
    def _evaluate_design_score(self, design_matrix: np.ndarray) -> float:
        """
        Evaluate combined cost-efficiency score.
        
        Args:
            design_matrix: Design matrix to evaluate
            
        Returns:
            Combined score (higher is better)
        """
        # Evaluate efficiency
        efficiency = self.efficiency_function(design_matrix)
        
        # Evaluate cost (normalize by dividing by max budget or typical cost)
        total_cost = self._evaluate_total_cost(design_matrix)
        
        # Normalize cost (lower cost is better)
        if self.max_budget is not None and self.max_budget > 0:
            normalized_cost = total_cost / self.max_budget
        else:
            normalized_cost = total_cost / 1000.0  # Arbitrary normalization
        
        # Combined score: weighted combination of efficiency and (negative) cost
        score = (1 - self.cost_weight) * efficiency - self.cost_weight * normalized_cost
        
        return score
    
    def get_design_analysis(self, design_matrix: np.ndarray) -> Dict[str, Any]:
        """
        Get detailed analysis of design cost and efficiency.
        
        Args:
            design_matrix: Design matrix to analyze
            
        Returns:
            Dictionary with detailed analysis
        """
        analysis = {
            'n_experiments': design_matrix.shape[0],
            'n_variables': design_matrix.shape[1],
            'efficiency': self.efficiency_function(design_matrix),
            'total_cost': self._evaluate_total_cost(design_matrix),
            'cost_breakdown': {},
            'within_budget': True,
        }
        
        # Get cost breakdown from each cost function
        for i, cost_function in enumerate(self.cost_functions):
            breakdown = cost_function.get_cost_breakdown(design_matrix, self.variable_names)
            analysis['cost_breakdown'][f'cost_function_{i}'] = breakdown
        
        # Check budget constraint
        if self.max_budget is not None:
            analysis['within_budget'] = analysis['total_cost'] <= self.max_budget
            analysis['budget_utilization'] = analysis['total_cost'] / self.max_budget
        
        # Combined score
        analysis['design_score'] = self._evaluate_design_score(design_matrix)
        
        return analysis


class BatchOptimizer:
    """
    Optimizer for batching experiments to reduce costs.
    
    Groups experiments into batches to minimize setup costs,
    reagent waste, and instrument time.
    """
    
    def __init__(
        self,
        batch_constraints: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize batch optimizer.
        
        Args:
            batch_constraints: Constraints for batching (e.g., max_batch_size)
            **kwargs: Additional parameters
        """
        self.batch_constraints = batch_constraints
        self.config = kwargs
        
        logger.info("Initialized batch optimizer")
    
    def optimize_batches(
        self,
        design_matrix: np.ndarray,
        variable_names: List[str]
    ) -> List[List[int]]:
        """
        Optimize experiment batching.
        
        Args:
            design_matrix: Design matrix
            variable_names: Variable names
            
        Returns:
            List of batches (each batch is list of experiment indices)
        """
        n_experiments = design_matrix.shape[0]
        max_batch_size = self.batch_constraints.get('max_batch_size', 10)
        
        # Simple batching strategy: group similar experiments
        batches = []
        remaining_indices = list(range(n_experiments))
        
        while remaining_indices:
            # Start new batch
            current_batch = [remaining_indices.pop(0)]
            
            # Add similar experiments to batch
            while len(current_batch) < max_batch_size and remaining_indices:
                # Find most similar remaining experiment
                best_similarity = -1
                best_idx = None
                best_remaining_idx = None
                
                for i, remaining_idx in enumerate(remaining_indices):
                    similarity = self._compute_similarity(
                        design_matrix[current_batch[0]],
                        design_matrix[remaining_idx]
                    )
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_idx = remaining_idx
                        best_remaining_idx = i
                
                if best_idx is not None:
                    current_batch.append(best_idx)
                    remaining_indices.pop(best_remaining_idx)
                else:
                    break
            
            batches.append(current_batch)
        
        logger.info(f"Optimized into {len(batches)} batches")
        
        return batches
    
    def _compute_similarity(self, exp1: np.ndarray, exp2: np.ndarray) -> float:
        """Compute similarity between two experiments."""
        # Simple Euclidean distance-based similarity
        distance = np.linalg.norm(exp1 - exp2)
        similarity = 1.0 / (1.0 + distance)  # Convert distance to similarity
        return similarity
