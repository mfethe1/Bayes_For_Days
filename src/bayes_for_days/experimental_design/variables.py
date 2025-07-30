"""
Variable definition and parameter specification system for experimental design.

This module provides comprehensive tools for defining experimental variables
with support for:
- Multiple variable types (continuous, categorical, ordinal, mixture)
- Parameter dependencies and constraint relationships
- Multi-unit support with automatic conversion
- Baseline values and variation range specifications
- Laboratory-specific parameter grouping and relationships

Based on:
- Design of Experiments (DoE) best practices
- JMP and Minitab variable specification systems
- Real laboratory workflow requirements
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import re

logger = logging.getLogger(__name__)


class VariableType(Enum):
    """Types of experimental variables."""
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    INTEGER = "integer"
    MIXTURE = "mixture"  # Components that must sum to 100%
    DERIVED = "derived"  # Calculated from other variables


class UnitType(Enum):
    """Supported unit types for experimental variables."""
    # Concentration units
    MOLARITY = "M"
    MILLIMOLAR = "mM"
    MICROMOLAR = "μM"
    NANOMOLAR = "nM"
    
    # Mass/volume units
    WEIGHT_PERCENT = "wt%"
    VOLUME_PERCENT = "vol%"
    MASS_PER_VOLUME = "mg/mL"
    GRAMS_PER_LITER = "g/L"
    
    # Parts per notation
    PPM = "ppm"
    PPB = "ppb"
    PARTS_PER_THOUSAND = "ppt"
    
    # Physical units
    TEMPERATURE_CELSIUS = "°C"
    TEMPERATURE_KELVIN = "K"
    PRESSURE_BAR = "bar"
    PRESSURE_PSI = "psi"
    TIME_MINUTES = "min"
    TIME_HOURS = "h"
    
    # Dimensionless
    RATIO = "ratio"
    PERCENTAGE = "%"
    DIMENSIONLESS = "dimensionless"
    
    # Raw numbers
    COUNT = "count"
    RAW = "raw"


@dataclass
class VariationRange:
    """
    Specification for how a variable can vary from its baseline.
    
    Supports multiple ways to specify variation:
    - Percentage variation (±20% from baseline)
    - Absolute ranges (0.1M to 0.5M)
    - Relative multipliers (0.5x to 2x baseline)
    """
    variation_type: str  # "percentage", "absolute", "multiplier"
    
    # For percentage variation
    percentage_range: Optional[Tuple[float, float]] = None  # (-20, +50) for -20% to +50%
    
    # For absolute variation
    absolute_min: Optional[float] = None
    absolute_max: Optional[float] = None
    
    # For multiplier variation
    multiplier_min: Optional[float] = None  # 0.5 for 0.5x baseline
    multiplier_max: Optional[float] = None  # 2.0 for 2x baseline
    
    # Constraints
    enforce_positive: bool = True  # Ensure values stay positive
    round_to_digits: Optional[int] = None  # Round to specified decimal places
    
    def __post_init__(self):
        """Validate variation range specification."""
        if self.variation_type == "percentage":
            if self.percentage_range is None:
                raise ValueError("percentage_range must be specified for percentage variation")
        elif self.variation_type == "absolute":
            if self.absolute_min is None or self.absolute_max is None:
                raise ValueError("absolute_min and absolute_max must be specified for absolute variation")
        elif self.variation_type == "multiplier":
            if self.multiplier_min is None or self.multiplier_max is None:
                raise ValueError("multiplier_min and multiplier_max must be specified for multiplier variation")
        else:
            raise ValueError(f"Unknown variation_type: {self.variation_type}")
    
    def apply_to_baseline(self, baseline_value: float) -> Tuple[float, float]:
        """
        Apply variation range to baseline value to get min/max bounds.
        
        Args:
            baseline_value: Baseline value to vary from
            
        Returns:
            Tuple of (min_value, max_value)
        """
        if self.variation_type == "percentage":
            min_pct, max_pct = self.percentage_range
            min_val = baseline_value * (1 + min_pct / 100)
            max_val = baseline_value * (1 + max_pct / 100)
        
        elif self.variation_type == "absolute":
            min_val = self.absolute_min
            max_val = self.absolute_max
        
        elif self.variation_type == "multiplier":
            min_val = baseline_value * self.multiplier_min
            max_val = baseline_value * self.multiplier_max
        
        else:
            raise ValueError(f"Unknown variation_type: {self.variation_type}")
        
        # Apply constraints
        if self.enforce_positive:
            min_val = max(0.0, min_val)
            max_val = max(0.0, max_val)
        
        # Ensure min <= max
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        
        # Apply rounding
        if self.round_to_digits is not None:
            min_val = round(min_val, self.round_to_digits)
            max_val = round(max_val, self.round_to_digits)
        
        return min_val, max_val


@dataclass
class UnitConversion:
    """Unit conversion utilities for experimental variables."""
    
    # Concentration conversion factors (to Molarity)
    CONCENTRATION_TO_MOLARITY = {
        UnitType.MOLARITY: 1.0,
        UnitType.MILLIMOLAR: 1e-3,
        UnitType.MICROMOLAR: 1e-6,
        UnitType.NANOMOLAR: 1e-9,
    }
    
    # Temperature conversion functions
    @staticmethod
    def celsius_to_kelvin(celsius: float) -> float:
        """Convert Celsius to Kelvin."""
        return celsius + 273.15
    
    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """Convert Kelvin to Celsius."""
        return kelvin - 273.15
    
    # Pressure conversion functions
    @staticmethod
    def bar_to_psi(bar: float) -> float:
        """Convert bar to PSI."""
        return bar * 14.5038
    
    @staticmethod
    def psi_to_bar(psi: float) -> float:
        """Convert PSI to bar."""
        return psi / 14.5038
    
    @classmethod
    def convert_concentration(
        cls, 
        value: float, 
        from_unit: UnitType, 
        to_unit: UnitType
    ) -> float:
        """
        Convert between concentration units.
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            
        Returns:
            Converted value
        """
        if from_unit == to_unit:
            return value
        
        # Convert to molarity first, then to target unit
        if from_unit in cls.CONCENTRATION_TO_MOLARITY:
            molarity = value * cls.CONCENTRATION_TO_MOLARITY[from_unit]
            
            if to_unit in cls.CONCENTRATION_TO_MOLARITY:
                return molarity / cls.CONCENTRATION_TO_MOLARITY[to_unit]
        
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
    
    @classmethod
    def convert_temperature(
        cls,
        value: float,
        from_unit: UnitType,
        to_unit: UnitType
    ) -> float:
        """Convert between temperature units."""
        if from_unit == to_unit:
            return value
        
        if from_unit == UnitType.TEMPERATURE_CELSIUS and to_unit == UnitType.TEMPERATURE_KELVIN:
            return cls.celsius_to_kelvin(value)
        elif from_unit == UnitType.TEMPERATURE_KELVIN and to_unit == UnitType.TEMPERATURE_CELSIUS:
            return cls.kelvin_to_celsius(value)
        
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")
    
    @classmethod
    def convert_pressure(
        cls,
        value: float,
        from_unit: UnitType,
        to_unit: UnitType
    ) -> float:
        """Convert between pressure units."""
        if from_unit == to_unit:
            return value
        
        if from_unit == UnitType.PRESSURE_BAR and to_unit == UnitType.PRESSURE_PSI:
            return cls.bar_to_psi(value)
        elif from_unit == UnitType.PRESSURE_PSI and to_unit == UnitType.PRESSURE_BAR:
            return cls.psi_to_bar(value)
        
        raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")


class ExperimentalVariable:
    """
    Comprehensive experimental variable definition.
    
    Supports all aspects of laboratory variable specification including
    units, baselines, variation ranges, dependencies, and constraints.
    """
    
    def __init__(
        self,
        name: str,
        variable_type: VariableType,
        unit: UnitType = UnitType.DIMENSIONLESS,
        baseline_value: Optional[float] = None,
        variation_range: Optional[VariationRange] = None,
        categories: Optional[List[str]] = None,
        description: Optional[str] = None,
        group: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize experimental variable.
        
        Args:
            name: Variable name (must be valid identifier)
            variable_type: Type of variable
            unit: Unit of measurement
            baseline_value: Baseline/reference value
            variation_range: How the variable can vary from baseline
            categories: Categories for categorical/ordinal variables
            description: Human-readable description
            group: Variable group for organization
            **kwargs: Additional metadata
        """
        self.name = self._validate_name(name)
        self.variable_type = variable_type
        self.unit = unit
        self.baseline_value = baseline_value
        self.variation_range = variation_range
        self.categories = categories or []
        self.description = description or ""
        self.group = group
        self.metadata = kwargs
        
        # Derived properties
        self.bounds: Optional[Tuple[float, float]] = None
        self.dependencies: List[str] = []
        self.constraints: List[Callable] = []
        
        # Validate configuration
        self._validate_configuration()
        
        # Calculate bounds if possible
        if self.baseline_value is not None and self.variation_range is not None:
            self.bounds = self.variation_range.apply_to_baseline(self.baseline_value)
    
    def _validate_name(self, name: str) -> str:
        """Validate variable name is a valid identifier."""
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
            raise ValueError(f"Variable name '{name}' is not a valid identifier")
        return name
    
    def _validate_configuration(self):
        """Validate variable configuration."""
        if self.variable_type in [VariableType.CATEGORICAL, VariableType.ORDINAL]:
            if not self.categories:
                raise ValueError(f"Categories must be specified for {self.variable_type} variables")
        
        if self.variable_type == VariableType.CONTINUOUS:
            if self.baseline_value is None:
                logger.warning(f"No baseline value specified for continuous variable '{self.name}'")
        
        if self.variable_type == VariableType.MIXTURE:
            if self.unit not in [UnitType.PERCENTAGE, UnitType.WEIGHT_PERCENT, UnitType.VOLUME_PERCENT]:
                logger.warning(f"Mixture variable '{self.name}' should use percentage units")
    
    def add_dependency(self, other_variable: str, relationship: str = "affects"):
        """
        Add dependency on another variable.
        
        Args:
            other_variable: Name of variable this depends on
            relationship: Type of relationship (affects, requires, excludes)
        """
        dependency = f"{other_variable}:{relationship}"
        if dependency not in self.dependencies:
            self.dependencies.append(dependency)
    
    def add_constraint(self, constraint_func: Callable[[float], bool], description: str = ""):
        """
        Add constraint function.
        
        Args:
            constraint_func: Function that returns True if value is valid
            description: Description of the constraint
        """
        constraint_func.description = description
        self.constraints.append(constraint_func)
    
    def is_valid_value(self, value: Union[float, str]) -> bool:
        """
        Check if a value is valid for this variable.
        
        Args:
            value: Value to check
            
        Returns:
            True if value is valid
        """
        try:
            if self.variable_type in [VariableType.CATEGORICAL, VariableType.ORDINAL]:
                return str(value) in self.categories
            
            elif self.variable_type in [VariableType.CONTINUOUS, VariableType.INTEGER, VariableType.MIXTURE]:
                numeric_value = float(value)
                
                # Check bounds
                if self.bounds:
                    min_val, max_val = self.bounds
                    if not (min_val <= numeric_value <= max_val):
                        return False
                
                # Check constraints
                for constraint in self.constraints:
                    if not constraint(numeric_value):
                        return False
                
                return True
            
            return False
            
        except (ValueError, TypeError):
            return False
    
    def generate_levels(self, n_levels: int = 3) -> List[Union[float, str]]:
        """
        Generate experimental levels for this variable.
        
        Args:
            n_levels: Number of levels to generate
            
        Returns:
            List of experimental levels
        """
        if self.variable_type in [VariableType.CATEGORICAL, VariableType.ORDINAL]:
            return self.categories[:n_levels] if len(self.categories) >= n_levels else self.categories
        
        elif self.variable_type in [VariableType.CONTINUOUS, VariableType.MIXTURE]:
            if self.bounds:
                min_val, max_val = self.bounds
                if n_levels == 1:
                    return [self.baseline_value or (min_val + max_val) / 2]
                else:
                    levels = np.linspace(min_val, max_val, n_levels)
                    return levels.tolist()
            elif self.baseline_value is not None:
                return [self.baseline_value]
            else:
                return [0.0]
        
        elif self.variable_type == VariableType.INTEGER:
            if self.bounds:
                min_val, max_val = self.bounds
                if n_levels == 1:
                    return [int(self.baseline_value or (min_val + max_val) / 2)]
                else:
                    levels = np.linspace(min_val, max_val, n_levels)
                    return [int(round(level)) for level in levels]
            elif self.baseline_value is not None:
                return [int(self.baseline_value)]
            else:
                return [0]
        
        return []
    
    def convert_to_unit(self, value: float, target_unit: UnitType) -> float:
        """
        Convert value to different unit.
        
        Args:
            value: Value in current unit
            target_unit: Target unit
            
        Returns:
            Converted value
        """
        if self.unit == target_unit:
            return value
        
        # Use appropriate conversion based on unit types
        if self.unit in UnitConversion.CONCENTRATION_TO_MOLARITY and target_unit in UnitConversion.CONCENTRATION_TO_MOLARITY:
            return UnitConversion.convert_concentration(value, self.unit, target_unit)
        
        elif self.unit in [UnitType.TEMPERATURE_CELSIUS, UnitType.TEMPERATURE_KELVIN]:
            return UnitConversion.convert_temperature(value, self.unit, target_unit)
        
        elif self.unit in [UnitType.PRESSURE_BAR, UnitType.PRESSURE_PSI]:
            return UnitConversion.convert_pressure(value, self.unit, target_unit)
        
        else:
            raise ValueError(f"Cannot convert from {self.unit} to {target_unit}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert variable to dictionary for serialization."""
        return {
            'name': self.name,
            'variable_type': self.variable_type.value,
            'unit': self.unit.value,
            'baseline_value': self.baseline_value,
            'variation_range': self.variation_range.__dict__ if self.variation_range else None,
            'categories': self.categories,
            'description': self.description,
            'group': self.group,
            'bounds': self.bounds,
            'dependencies': self.dependencies,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentalVariable':
        """Create variable from dictionary."""
        # Convert enums
        variable_type = VariableType(data['variable_type'])
        unit = UnitType(data['unit'])
        
        # Reconstruct variation range
        variation_range = None
        if data.get('variation_range'):
            variation_range = VariationRange(**data['variation_range'])
        
        # Create variable
        variable = cls(
            name=data['name'],
            variable_type=variable_type,
            unit=unit,
            baseline_value=data.get('baseline_value'),
            variation_range=variation_range,
            categories=data.get('categories'),
            description=data.get('description'),
            group=data.get('group'),
            **data.get('metadata', {})
        )
        
        # Restore derived properties
        variable.bounds = tuple(data['bounds']) if data.get('bounds') else None
        variable.dependencies = data.get('dependencies', [])
        
        return variable
    
    def __repr__(self) -> str:
        """String representation of variable."""
        return (f"ExperimentalVariable(name='{self.name}', "
                f"type={self.variable_type.value}, "
                f"unit={self.unit.value}, "
                f"baseline={self.baseline_value})")


class VariableGroup:
    """
    Group of related experimental variables.
    
    Supports variable grouping for organization and dependency management.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize variable group.
        
        Args:
            name: Group name
            description: Group description
        """
        self.name = name
        self.description = description
        self.variables: Dict[str, ExperimentalVariable] = {}
        self.group_constraints: List[Callable] = []
    
    def add_variable(self, variable: ExperimentalVariable):
        """Add variable to group."""
        variable.group = self.name
        self.variables[variable.name] = variable
    
    def remove_variable(self, variable_name: str):
        """Remove variable from group."""
        if variable_name in self.variables:
            del self.variables[variable_name]
    
    def add_group_constraint(self, constraint_func: Callable, description: str = ""):
        """
        Add constraint that applies to the entire group.
        
        Args:
            constraint_func: Function that takes dict of variable values
            description: Constraint description
        """
        constraint_func.description = description
        self.group_constraints.append(constraint_func)
    
    def validate_values(self, values: Dict[str, Union[float, str]]) -> bool:
        """
        Validate a set of values for all variables in the group.
        
        Args:
            values: Dictionary of variable_name -> value
            
        Returns:
            True if all values are valid
        """
        # Check individual variable constraints
        for var_name, value in values.items():
            if var_name in self.variables:
                if not self.variables[var_name].is_valid_value(value):
                    return False
        
        # Check group constraints
        for constraint in self.group_constraints:
            if not constraint(values):
                return False
        
        return True
    
    def get_variable_names(self) -> List[str]:
        """Get list of variable names in this group."""
        return list(self.variables.keys())
    
    def __len__(self) -> int:
        """Number of variables in group."""
        return len(self.variables)
    
    def __iter__(self):
        """Iterate over variables."""
        return iter(self.variables.values())
    
    def __repr__(self) -> str:
        """String representation of group."""
        return f"VariableGroup(name='{self.name}', n_variables={len(self.variables)})"
