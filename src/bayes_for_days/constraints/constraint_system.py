"""
Comprehensive constraint handling system for Bayes For Days.

This module provides a unified framework for handling various types of
constraints in optimization problems:
- Equality and inequality constraints
- Linear and nonlinear constraints
- Box constraints (parameter bounds)
- Custom constraint functions
- Constraint-aware acquisition functions

Based on:
- Gelbart et al. (2014) "Bayesian Optimization with Unknown Constraints"
- Gardner et al. (2014) "Bayesian Optimization with Inequality Constraints"
- Hernández-Lobato et al. (2016) "A General Framework for Constrained Bayesian Optimization"
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ConstraintType(Enum):
    """Types of constraints."""
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    BOX = "box"
    CUSTOM = "custom"


class ConstraintMethod(Enum):
    """Methods for handling constraints."""
    PENALTY = "penalty"
    BARRIER = "barrier"
    FEASIBILITY_RULE = "feasibility_rule"
    CONSTRAINT_AWARE = "constraint_aware"


@dataclass
class ConstraintViolation:
    """Information about constraint violation."""
    constraint_name: str
    constraint_type: ConstraintType
    violation_amount: float
    is_violated: bool
    parameters: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None


class BaseConstraint(ABC):
    """
    Abstract base class for constraints.
    
    All constraint types inherit from this class and implement
    the evaluate method to check constraint satisfaction.
    """
    
    def __init__(
        self,
        name: str,
        constraint_type: ConstraintType,
        tolerance: float = 1e-6,
        **kwargs
    ):
        """
        Initialize base constraint.
        
        Args:
            name: Constraint name
            constraint_type: Type of constraint
            tolerance: Tolerance for constraint satisfaction
            **kwargs: Additional parameters
        """
        self.name = name
        self.constraint_type = constraint_type
        self.tolerance = tolerance
        self.metadata = kwargs
    
    @abstractmethod
    def evaluate(self, parameters: Dict[str, float]) -> ConstraintViolation:
        """
        Evaluate constraint at given parameters.
        
        Args:
            parameters: Parameter values
            
        Returns:
            Constraint violation information
        """
        pass
    
    def is_satisfied(self, parameters: Dict[str, float]) -> bool:
        """
        Check if constraint is satisfied.
        
        Args:
            parameters: Parameter values
            
        Returns:
            True if constraint is satisfied
        """
        violation = self.evaluate(parameters)
        return not violation.is_violated
    
    def get_violation_amount(self, parameters: Dict[str, float]) -> float:
        """
        Get constraint violation amount.
        
        Args:
            parameters: Parameter values
            
        Returns:
            Violation amount (0 if satisfied)
        """
        violation = self.evaluate(parameters)
        return violation.violation_amount


class LinearConstraint(BaseConstraint):
    """
    Linear constraint: a^T x <= b (inequality) or a^T x = b (equality).
    """
    
    def __init__(
        self,
        name: str,
        coefficients: Dict[str, float],
        rhs: float,
        constraint_type: ConstraintType = ConstraintType.INEQUALITY,
        tolerance: float = 1e-6
    ):
        """
        Initialize linear constraint.
        
        Args:
            name: Constraint name
            coefficients: Linear coefficients for each parameter
            rhs: Right-hand side value
            constraint_type: EQUALITY or INEQUALITY
            tolerance: Tolerance for constraint satisfaction
        """
        super().__init__(name, constraint_type, tolerance)
        
        self.coefficients = coefficients
        self.rhs = rhs
    
    def evaluate(self, parameters: Dict[str, float]) -> ConstraintViolation:
        """Evaluate linear constraint."""
        # Compute linear combination
        lhs = sum(
            self.coefficients.get(param_name, 0.0) * param_value
            for param_name, param_value in parameters.items()
        )
        
        if self.constraint_type == ConstraintType.EQUALITY:
            # Equality constraint: |a^T x - b| <= tolerance
            violation_amount = abs(lhs - self.rhs)
            is_violated = violation_amount > self.tolerance
        
        else:
            # Inequality constraint: a^T x <= b
            violation_amount = max(0.0, lhs - self.rhs)
            is_violated = violation_amount > self.tolerance
        
        return ConstraintViolation(
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            violation_amount=violation_amount,
            is_violated=is_violated,
            parameters=parameters,
            metadata={
                'lhs_value': lhs,
                'rhs_value': self.rhs,
                'coefficients': self.coefficients
            }
        )


class NonlinearConstraint(BaseConstraint):
    """
    Nonlinear constraint defined by a custom function.
    """
    
    def __init__(
        self,
        name: str,
        constraint_function: Callable[[Dict[str, float]], float],
        constraint_type: ConstraintType = ConstraintType.INEQUALITY,
        tolerance: float = 1e-6,
        target_value: float = 0.0
    ):
        """
        Initialize nonlinear constraint.
        
        Args:
            name: Constraint name
            constraint_function: Function that evaluates constraint
            constraint_type: EQUALITY or INEQUALITY
            tolerance: Tolerance for constraint satisfaction
            target_value: Target value for constraint (default 0)
        """
        super().__init__(name, constraint_type, tolerance)
        
        self.constraint_function = constraint_function
        self.target_value = target_value
    
    def evaluate(self, parameters: Dict[str, float]) -> ConstraintViolation:
        """Evaluate nonlinear constraint."""
        try:
            # Evaluate constraint function
            function_value = self.constraint_function(parameters)
            
            if self.constraint_type == ConstraintType.EQUALITY:
                # Equality constraint: |g(x) - target| <= tolerance
                violation_amount = abs(function_value - self.target_value)
                is_violated = violation_amount > self.tolerance
            
            else:
                # Inequality constraint: g(x) <= target
                violation_amount = max(0.0, function_value - self.target_value)
                is_violated = violation_amount > self.tolerance
            
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                violation_amount=violation_amount,
                is_violated=is_violated,
                parameters=parameters,
                metadata={
                    'function_value': function_value,
                    'target_value': self.target_value
                }
            )
            
        except Exception as e:
            logger.warning(f"Error evaluating constraint {self.name}: {e}")
            
            # Return violation for failed evaluation
            return ConstraintViolation(
                constraint_name=self.name,
                constraint_type=self.constraint_type,
                violation_amount=float('inf'),
                is_violated=True,
                parameters=parameters,
                metadata={'error': str(e)}
            )


class BoxConstraint(BaseConstraint):
    """
    Box constraint: lower <= x <= upper for each parameter.
    """
    
    def __init__(
        self,
        name: str,
        bounds: Dict[str, Tuple[float, float]],
        tolerance: float = 1e-6
    ):
        """
        Initialize box constraint.
        
        Args:
            name: Constraint name
            bounds: Parameter bounds {param_name: (lower, upper)}
            tolerance: Tolerance for constraint satisfaction
        """
        super().__init__(name, ConstraintType.BOX, tolerance)
        
        self.bounds = bounds
    
    def evaluate(self, parameters: Dict[str, float]) -> ConstraintViolation:
        """Evaluate box constraint."""
        total_violation = 0.0
        violations = {}
        
        for param_name, param_value in parameters.items():
            if param_name in self.bounds:
                lower, upper = self.bounds[param_name]
                
                # Check lower bound
                lower_violation = max(0.0, lower - param_value)
                
                # Check upper bound
                upper_violation = max(0.0, param_value - upper)
                
                param_violation = lower_violation + upper_violation
                total_violation += param_violation
                
                if param_violation > self.tolerance:
                    violations[param_name] = {
                        'lower_violation': lower_violation,
                        'upper_violation': upper_violation,
                        'bounds': (lower, upper),
                        'value': param_value
                    }
        
        is_violated = total_violation > self.tolerance
        
        return ConstraintViolation(
            constraint_name=self.name,
            constraint_type=self.constraint_type,
            violation_amount=total_violation,
            is_violated=is_violated,
            parameters=parameters,
            metadata={
                'parameter_violations': violations,
                'bounds': self.bounds
            }
        )


class ConstraintManager:
    """
    Manager for handling multiple constraints.
    
    Provides unified interface for evaluating multiple constraints,
    computing total violations, and applying constraint handling methods.
    """
    
    def __init__(
        self,
        constraints: List[BaseConstraint],
        method: ConstraintMethod = ConstraintMethod.PENALTY,
        penalty_weight: float = 1000.0
    ):
        """
        Initialize constraint manager.
        
        Args:
            constraints: List of constraints
            method: Constraint handling method
            penalty_weight: Weight for penalty method
        """
        self.constraints = constraints
        self.method = method
        self.penalty_weight = penalty_weight
        
        logger.info(f"Initialized constraint manager with {len(constraints)} constraints")
    
    def evaluate_all(self, parameters: Dict[str, float]) -> List[ConstraintViolation]:
        """
        Evaluate all constraints.
        
        Args:
            parameters: Parameter values
            
        Returns:
            List of constraint violations
        """
        violations = []
        
        for constraint in self.constraints:
            try:
                violation = constraint.evaluate(parameters)
                violations.append(violation)
            except Exception as e:
                logger.warning(f"Error evaluating constraint {constraint.name}: {e}")
                
                # Create error violation
                error_violation = ConstraintViolation(
                    constraint_name=constraint.name,
                    constraint_type=constraint.constraint_type,
                    violation_amount=float('inf'),
                    is_violated=True,
                    parameters=parameters,
                    metadata={'error': str(e)}
                )
                violations.append(error_violation)
        
        return violations
    
    def is_feasible(self, parameters: Dict[str, float]) -> bool:
        """
        Check if parameters satisfy all constraints.
        
        Args:
            parameters: Parameter values
            
        Returns:
            True if all constraints are satisfied
        """
        violations = self.evaluate_all(parameters)
        return all(not violation.is_violated for violation in violations)
    
    def get_total_violation(self, parameters: Dict[str, float]) -> float:
        """
        Get total constraint violation.
        
        Args:
            parameters: Parameter values
            
        Returns:
            Total violation amount
        """
        violations = self.evaluate_all(parameters)
        return sum(violation.violation_amount for violation in violations)
    
    def apply_penalty(
        self,
        objective_value: float,
        parameters: Dict[str, float]
    ) -> float:
        """
        Apply penalty method to objective value.
        
        Args:
            objective_value: Original objective value
            parameters: Parameter values
            
        Returns:
            Penalized objective value
        """
        if self.method != ConstraintMethod.PENALTY:
            return objective_value
        
        total_violation = self.get_total_violation(parameters)
        
        if total_violation > 0:
            penalty = self.penalty_weight * total_violation
            return objective_value - penalty
        
        return objective_value
    
    def apply_feasibility_rule(
        self,
        candidates: List[Tuple[Dict[str, float], float]]
    ) -> List[Tuple[Dict[str, float], float]]:
        """
        Apply feasibility rule to candidate solutions.
        
        Feasibility rule:
        1. Feasible solutions are always preferred over infeasible ones
        2. Among feasible solutions, prefer better objective value
        3. Among infeasible solutions, prefer less constraint violation
        
        Args:
            candidates: List of (parameters, objective_value) tuples
            
        Returns:
            Sorted candidates according to feasibility rule
        """
        if self.method != ConstraintMethod.FEASIBILITY_RULE:
            return candidates
        
        # Evaluate feasibility for all candidates
        candidate_info = []
        
        for parameters, objective_value in candidates:
            is_feasible = self.is_feasible(parameters)
            total_violation = self.get_total_violation(parameters)
            
            candidate_info.append({
                'parameters': parameters,
                'objective_value': objective_value,
                'is_feasible': is_feasible,
                'total_violation': total_violation
            })
        
        # Sort according to feasibility rule
        def sort_key(candidate):
            if candidate['is_feasible']:
                # Feasible: sort by objective value (descending for maximization)
                return (1, -candidate['objective_value'])
            else:
                # Infeasible: sort by violation (ascending)
                return (0, candidate['total_violation'])
        
        sorted_candidates = sorted(candidate_info, key=sort_key, reverse=True)
        
        # Return as original format
        return [
            (candidate['parameters'], candidate['objective_value'])
            for candidate in sorted_candidates
        ]
    
    def get_constraint_info(self) -> Dict[str, Any]:
        """Get information about constraints."""
        return {
            'n_constraints': len(self.constraints),
            'constraint_types': [c.constraint_type.value for c in self.constraints],
            'constraint_names': [c.name for c in self.constraints],
            'method': self.method.value,
            'penalty_weight': self.penalty_weight,
        }
    
    def add_constraint(self, constraint: BaseConstraint):
        """Add a new constraint."""
        self.constraints.append(constraint)
        logger.info(f"Added constraint: {constraint.name}")
    
    def remove_constraint(self, constraint_name: str) -> bool:
        """
        Remove constraint by name.
        
        Args:
            constraint_name: Name of constraint to remove
            
        Returns:
            True if constraint was removed
        """
        for i, constraint in enumerate(self.constraints):
            if constraint.name == constraint_name:
                del self.constraints[i]
                logger.info(f"Removed constraint: {constraint_name}")
                return True
        
        logger.warning(f"Constraint not found: {constraint_name}")
        return False
    
    def get_constraint_summary(self, parameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Get summary of constraint evaluation.
        
        Args:
            parameters: Parameter values
            
        Returns:
            Summary of constraint violations
        """
        violations = self.evaluate_all(parameters)
        
        summary = {
            'is_feasible': all(not v.is_violated for v in violations),
            'total_violation': sum(v.violation_amount for v in violations),
            'n_violated_constraints': sum(1 for v in violations if v.is_violated),
            'constraint_details': []
        }
        
        for violation in violations:
            summary['constraint_details'].append({
                'name': violation.constraint_name,
                'type': violation.constraint_type.value,
                'is_violated': violation.is_violated,
                'violation_amount': violation.violation_amount,
                'metadata': violation.metadata
            })
        
        return summary
