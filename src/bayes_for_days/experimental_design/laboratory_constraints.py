"""
Laboratory-specific design constraints for practical experimental design.

This module implements constraints that reflect real laboratory limitations:
- Mixture design constraints (sum-to-100%)
- Stock solution limitations and dilution calculations
- Practical laboratory workflow considerations
- Resource availability and scheduling constraints
- Equipment-specific limitations

Based on:
- Cornell (2002) "Experiments with Mixtures"
- Montgomery (2017) "Design and Analysis of Experiments"
- Real-world laboratory workflow requirements
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from bayes_for_days.experimental_design.variables import ExperimentalVariable, VariableType, UnitType

logger = logging.getLogger(__name__)


@dataclass
class StockSolution:
    """
    Stock solution definition for laboratory constraints.
    
    Represents available stock solutions with their concentrations,
    volumes, and dilution capabilities.
    """
    name: str
    concentration: float
    unit: UnitType
    available_volume: float  # mL
    min_dilution_factor: float = 1.0
    max_dilution_factor: float = 1000.0
    cost_per_ml: float = 0.0
    expiry_date: Optional[str] = None
    
    def can_achieve_concentration(self, target_concentration: float, target_unit: UnitType) -> bool:
        """Check if stock can achieve target concentration through dilution."""
        # Convert target to stock units if needed
        if target_unit != self.unit:
            # Simplified conversion - in practice would use UnitConversion
            if self.unit == UnitType.MOLARITY and target_unit == UnitType.MILLIMOLAR:
                target_in_stock_units = target_concentration / 1000.0
            elif self.unit == UnitType.MILLIMOLAR and target_unit == UnitType.MOLARITY:
                target_in_stock_units = target_concentration * 1000.0
            else:
                return False  # Unsupported conversion
        else:
            target_in_stock_units = target_concentration
        
        # Check if achievable through dilution
        min_achievable = self.concentration / self.max_dilution_factor
        max_achievable = self.concentration / self.min_dilution_factor
        
        return min_achievable <= target_in_stock_units <= max_achievable
    
    def calculate_dilution_volume(
        self, 
        target_concentration: float, 
        target_volume: float,
        target_unit: UnitType
    ) -> Tuple[float, float]:
        """
        Calculate required stock and diluent volumes.
        
        Args:
            target_concentration: Desired final concentration
            target_volume: Desired final volume (mL)
            target_unit: Unit of target concentration
            
        Returns:
            Tuple of (stock_volume_ml, diluent_volume_ml)
        """
        # Convert target to stock units
        if target_unit != self.unit:
            if self.unit == UnitType.MOLARITY and target_unit == UnitType.MILLIMOLAR:
                target_in_stock_units = target_concentration / 1000.0
            elif self.unit == UnitType.MILLIMOLAR and target_unit == UnitType.MOLARITY:
                target_in_stock_units = target_concentration * 1000.0
            else:
                raise ValueError(f"Cannot convert from {target_unit} to {self.unit}")
        else:
            target_in_stock_units = target_concentration
        
        # Calculate dilution: C1*V1 = C2*V2
        stock_volume = (target_in_stock_units * target_volume) / self.concentration
        diluent_volume = target_volume - stock_volume
        
        if stock_volume > self.available_volume:
            raise ValueError(f"Required stock volume ({stock_volume:.2f} mL) exceeds available ({self.available_volume:.2f} mL)")
        
        if diluent_volume < 0:
            raise ValueError("Target concentration higher than stock concentration")
        
        return stock_volume, diluent_volume


class LaboratoryConstraint(ABC):
    """
    Abstract base class for laboratory constraints.
    
    Defines the interface for constraints that must be satisfied
    in practical experimental designs.
    """
    
    @abstractmethod
    def is_satisfied(self, experiment_params: Dict[str, float]) -> bool:
        """
        Check if constraint is satisfied by experiment parameters.
        
        Args:
            experiment_params: Dictionary of parameter values
            
        Returns:
            True if constraint is satisfied
        """
        pass
    
    @abstractmethod
    def get_violation_magnitude(self, experiment_params: Dict[str, float]) -> float:
        """
        Get magnitude of constraint violation.
        
        Args:
            experiment_params: Dictionary of parameter values
            
        Returns:
            Violation magnitude (0 if satisfied, positive if violated)
        """
        pass
    
    @abstractmethod
    def get_constraint_description(self) -> str:
        """Get human-readable description of the constraint."""
        pass


class MixtureConstraint(LaboratoryConstraint):
    """
    Mixture design constraint ensuring components sum to 100%.
    
    Handles mixture experiments where component fractions must
    sum to a specified total (typically 100%).
    """
    
    def __init__(
        self,
        component_variables: List[str],
        target_sum: float = 100.0,
        tolerance: float = 1e-6,
        **kwargs
    ):
        """
        Initialize mixture constraint.
        
        Args:
            component_variables: Names of mixture component variables
            target_sum: Target sum for components (default 100%)
            tolerance: Tolerance for constraint satisfaction
            **kwargs: Additional parameters
        """
        self.component_variables = component_variables
        self.target_sum = target_sum
        self.tolerance = tolerance
        self.config = kwargs
        
        logger.info(f"Initialized mixture constraint for {len(component_variables)} components")
    
    def is_satisfied(self, experiment_params: Dict[str, float]) -> bool:
        """Check if mixture constraint is satisfied."""
        total = sum(experiment_params.get(var, 0.0) for var in self.component_variables)
        return abs(total - self.target_sum) <= self.tolerance
    
    def get_violation_magnitude(self, experiment_params: Dict[str, float]) -> float:
        """Get magnitude of mixture constraint violation."""
        total = sum(experiment_params.get(var, 0.0) for var in self.component_variables)
        return abs(total - self.target_sum)
    
    def get_constraint_description(self) -> str:
        """Get description of mixture constraint."""
        return f"Sum of {', '.join(self.component_variables)} must equal {self.target_sum}"
    
    def normalize_components(self, experiment_params: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize mixture components to satisfy constraint.
        
        Args:
            experiment_params: Original parameter values
            
        Returns:
            Normalized parameter values
        """
        normalized_params = experiment_params.copy()
        
        # Calculate current sum
        current_sum = sum(experiment_params.get(var, 0.0) for var in self.component_variables)
        
        if current_sum > 0:
            # Normalize to target sum
            normalization_factor = self.target_sum / current_sum
            
            for var in self.component_variables:
                if var in normalized_params:
                    normalized_params[var] *= normalization_factor
        
        return normalized_params


class StockSolutionConstraint(LaboratoryConstraint):
    """
    Stock solution availability constraint.
    
    Ensures that required concentrations can be achieved using
    available stock solutions and dilution procedures.
    """
    
    def __init__(
        self,
        concentration_variables: Dict[str, str],  # var_name -> stock_name
        stock_solutions: Dict[str, StockSolution],
        **kwargs
    ):
        """
        Initialize stock solution constraint.
        
        Args:
            concentration_variables: Mapping of variable names to stock solution names
            stock_solutions: Available stock solutions
            **kwargs: Additional parameters
        """
        self.concentration_variables = concentration_variables
        self.stock_solutions = stock_solutions
        self.config = kwargs
        
        logger.info(f"Initialized stock solution constraint for {len(concentration_variables)} variables")
    
    def is_satisfied(self, experiment_params: Dict[str, float]) -> bool:
        """Check if stock solution constraint is satisfied."""
        for var_name, stock_name in self.concentration_variables.items():
            if var_name in experiment_params:
                target_concentration = experiment_params[var_name]
                
                if stock_name in self.stock_solutions:
                    stock = self.stock_solutions[stock_name]
                    
                    # Assume same units for simplicity
                    if not stock.can_achieve_concentration(target_concentration, stock.unit):
                        return False
                else:
                    return False  # Stock not available
        
        return True
    
    def get_violation_magnitude(self, experiment_params: Dict[str, float]) -> float:
        """Get magnitude of stock solution constraint violation."""
        total_violation = 0.0
        
        for var_name, stock_name in self.concentration_variables.items():
            if var_name in experiment_params:
                target_concentration = experiment_params[var_name]
                
                if stock_name in self.stock_solutions:
                    stock = self.stock_solutions[stock_name]
                    
                    # Calculate how far outside achievable range
                    min_achievable = stock.concentration / stock.max_dilution_factor
                    max_achievable = stock.concentration / stock.min_dilution_factor
                    
                    if target_concentration < min_achievable:
                        total_violation += min_achievable - target_concentration
                    elif target_concentration > max_achievable:
                        total_violation += target_concentration - max_achievable
                else:
                    total_violation += target_concentration  # Penalty for unavailable stock
        
        return total_violation
    
    def get_constraint_description(self) -> str:
        """Get description of stock solution constraint."""
        return f"Concentrations must be achievable using available stock solutions"
    
    def calculate_preparation_protocol(
        self, 
        experiment_params: Dict[str, float],
        target_volume: float = 10.0  # mL
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate preparation protocol for experiment.
        
        Args:
            experiment_params: Experiment parameter values
            target_volume: Target final volume for each solution
            
        Returns:
            Dictionary with preparation instructions for each variable
        """
        protocols = {}
        
        for var_name, stock_name in self.concentration_variables.items():
            if var_name in experiment_params and stock_name in self.stock_solutions:
                target_concentration = experiment_params[var_name]
                stock = self.stock_solutions[stock_name]
                
                try:
                    stock_vol, diluent_vol = stock.calculate_dilution_volume(
                        target_concentration, target_volume, stock.unit
                    )
                    
                    protocols[var_name] = {
                        'stock_solution': stock_name,
                        'stock_volume_ml': stock_vol,
                        'diluent_volume_ml': diluent_vol,
                        'final_volume_ml': target_volume,
                        'final_concentration': target_concentration,
                        'dilution_factor': stock.concentration / target_concentration
                    }
                    
                except ValueError as e:
                    protocols[var_name] = {
                        'error': str(e),
                        'stock_solution': stock_name,
                        'target_concentration': target_concentration
                    }
        
        return protocols


class VolumeConstraint(LaboratoryConstraint):
    """
    Total volume constraint for experiments.
    
    Ensures that total volume requirements don't exceed
    available capacity or practical limits.
    """
    
    def __init__(
        self,
        volume_variables: List[str],
        max_total_volume: float,
        min_total_volume: float = 0.0,
        **kwargs
    ):
        """
        Initialize volume constraint.
        
        Args:
            volume_variables: Names of variables contributing to total volume
            max_total_volume: Maximum allowed total volume
            min_total_volume: Minimum required total volume
            **kwargs: Additional parameters
        """
        self.volume_variables = volume_variables
        self.max_total_volume = max_total_volume
        self.min_total_volume = min_total_volume
        self.config = kwargs
        
        logger.info(f"Initialized volume constraint: {min_total_volume}-{max_total_volume} mL")
    
    def is_satisfied(self, experiment_params: Dict[str, float]) -> bool:
        """Check if volume constraint is satisfied."""
        total_volume = sum(experiment_params.get(var, 0.0) for var in self.volume_variables)
        return self.min_total_volume <= total_volume <= self.max_total_volume
    
    def get_violation_magnitude(self, experiment_params: Dict[str, float]) -> float:
        """Get magnitude of volume constraint violation."""
        total_volume = sum(experiment_params.get(var, 0.0) for var in self.volume_variables)
        
        if total_volume < self.min_total_volume:
            return self.min_total_volume - total_volume
        elif total_volume > self.max_total_volume:
            return total_volume - self.max_total_volume
        else:
            return 0.0
    
    def get_constraint_description(self) -> str:
        """Get description of volume constraint."""
        return f"Total volume must be between {self.min_total_volume} and {self.max_total_volume} mL"


class EquipmentConstraint(LaboratoryConstraint):
    """
    Equipment-specific constraints.
    
    Handles limitations imposed by specific laboratory equipment
    such as temperature ranges, pressure limits, etc.
    """
    
    def __init__(
        self,
        equipment_name: str,
        parameter_limits: Dict[str, Tuple[float, float]],
        **kwargs
    ):
        """
        Initialize equipment constraint.
        
        Args:
            equipment_name: Name of the equipment
            parameter_limits: Dictionary of parameter_name -> (min, max) limits
            **kwargs: Additional parameters
        """
        self.equipment_name = equipment_name
        self.parameter_limits = parameter_limits
        self.config = kwargs
        
        logger.info(f"Initialized equipment constraint for {equipment_name}")
    
    def is_satisfied(self, experiment_params: Dict[str, float]) -> bool:
        """Check if equipment constraint is satisfied."""
        for param_name, (min_val, max_val) in self.parameter_limits.items():
            if param_name in experiment_params:
                value = experiment_params[param_name]
                if not (min_val <= value <= max_val):
                    return False
        return True
    
    def get_violation_magnitude(self, experiment_params: Dict[str, float]) -> float:
        """Get magnitude of equipment constraint violation."""
        total_violation = 0.0
        
        for param_name, (min_val, max_val) in self.parameter_limits.items():
            if param_name in experiment_params:
                value = experiment_params[param_name]
                
                if value < min_val:
                    total_violation += min_val - value
                elif value > max_val:
                    total_violation += value - max_val
        
        return total_violation
    
    def get_constraint_description(self) -> str:
        """Get description of equipment constraint."""
        limits_str = ", ".join([
            f"{param}: [{min_val}, {max_val}]" 
            for param, (min_val, max_val) in self.parameter_limits.items()
        ])
        return f"{self.equipment_name} limits: {limits_str}"


class LaboratoryConstraintManager:
    """
    Manager for multiple laboratory constraints.
    
    Coordinates constraint checking, violation calculation,
    and constraint-aware design generation.
    """
    
    def __init__(self, constraints: List[LaboratoryConstraint]):
        """
        Initialize constraint manager.
        
        Args:
            constraints: List of laboratory constraints
        """
        self.constraints = constraints
        
        logger.info(f"Initialized constraint manager with {len(constraints)} constraints")
    
    def check_feasibility(self, experiment_params: Dict[str, float]) -> bool:
        """
        Check if experiment parameters satisfy all constraints.
        
        Args:
            experiment_params: Dictionary of parameter values
            
        Returns:
            True if all constraints are satisfied
        """
        return all(constraint.is_satisfied(experiment_params) for constraint in self.constraints)
    
    def get_total_violation(self, experiment_params: Dict[str, float]) -> float:
        """
        Get total constraint violation magnitude.
        
        Args:
            experiment_params: Dictionary of parameter values
            
        Returns:
            Total violation magnitude across all constraints
        """
        return sum(constraint.get_violation_magnitude(experiment_params) for constraint in self.constraints)
    
    def get_constraint_violations(self, experiment_params: Dict[str, float]) -> Dict[str, float]:
        """
        Get individual constraint violations.
        
        Args:
            experiment_params: Dictionary of parameter values
            
        Returns:
            Dictionary mapping constraint descriptions to violation magnitudes
        """
        violations = {}
        
        for constraint in self.constraints:
            violation = constraint.get_violation_magnitude(experiment_params)
            if violation > 0:
                violations[constraint.get_constraint_description()] = violation
        
        return violations
    
    def filter_feasible_designs(self, design_matrix: np.ndarray, variable_names: List[str]) -> np.ndarray:
        """
        Filter design matrix to keep only feasible experiments.
        
        Args:
            design_matrix: Design matrix (n_experiments x n_variables)
            variable_names: Names of variables corresponding to columns
            
        Returns:
            Filtered design matrix with only feasible experiments
        """
        feasible_indices = []
        
        for i, experiment in enumerate(design_matrix):
            experiment_params = {
                var_name: experiment[j] 
                for j, var_name in enumerate(variable_names) 
                if j < len(experiment)
            }
            
            if self.check_feasibility(experiment_params):
                feasible_indices.append(i)
        
        if not feasible_indices:
            logger.warning("No feasible experiments found in design matrix")
            return np.array([])
        
        feasible_design = design_matrix[feasible_indices]
        
        logger.info(f"Filtered design: {len(feasible_indices)}/{len(design_matrix)} experiments feasible")
        
        return feasible_design
    
    def repair_infeasible_design(
        self, 
        experiment_params: Dict[str, float],
        max_iterations: int = 100
    ) -> Optional[Dict[str, float]]:
        """
        Attempt to repair infeasible experiment parameters.
        
        Args:
            experiment_params: Infeasible parameter values
            max_iterations: Maximum repair iterations
            
        Returns:
            Repaired parameter values or None if repair failed
        """
        if self.check_feasibility(experiment_params):
            return experiment_params
        
        # Simple repair strategy: minimize total violation
        param_names = list(experiment_params.keys())
        initial_values = np.array([experiment_params[name] for name in param_names])
        
        def objective(x):
            params = {name: x[i] for i, name in enumerate(param_names)}
            return self.get_total_violation(params)
        
        try:
            result = minimize(
                objective,
                initial_values,
                method='L-BFGS-B',
                options={'maxiter': max_iterations}
            )
            
            if result.success and result.fun < 1e-6:
                repaired_params = {name: result.x[i] for i, name in enumerate(param_names)}
                return repaired_params
            else:
                return None
                
        except Exception as e:
            logger.warning(f"Constraint repair failed: {e}")
            return None
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of all constraints."""
        return {
            'n_constraints': len(self.constraints),
            'constraint_types': [type(c).__name__ for c in self.constraints],
            'constraint_descriptions': [c.get_constraint_description() for c in self.constraints],
        }
