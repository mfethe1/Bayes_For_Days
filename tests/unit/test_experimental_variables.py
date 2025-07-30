"""
Unit tests for experimental variable definition system.

Tests the comprehensive variable specification including:
- Variable types (continuous, categorical, ordinal, mixture)
- Unit types and conversions
- Variation ranges (percentage, absolute, multiplier)
- Variable groups and dependencies
- Constraint validation and level generation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from bayes_for_days.experimental_design.variables import (
    ExperimentalVariable,
    VariableType,
    UnitType,
    VariationRange,
    UnitConversion,
    VariableGroup,
)


class TestVariationRange:
    """Test suite for VariationRange class."""
    
    def test_percentage_variation_initialization(self):
        """Test percentage variation range initialization."""
        var_range = VariationRange(
            variation_type="percentage",
            percentage_range=(-20, 30)
        )
        
        assert var_range.variation_type == "percentage"
        assert var_range.percentage_range == (-20, 30)
    
    def test_absolute_variation_initialization(self):
        """Test absolute variation range initialization."""
        var_range = VariationRange(
            variation_type="absolute",
            absolute_min=1.0,
            absolute_max=10.0
        )
        
        assert var_range.variation_type == "absolute"
        assert var_range.absolute_min == 1.0
        assert var_range.absolute_max == 10.0
    
    def test_multiplier_variation_initialization(self):
        """Test multiplier variation range initialization."""
        var_range = VariationRange(
            variation_type="multiplier",
            multiplier_min=0.5,
            multiplier_max=2.0
        )
        
        assert var_range.variation_type == "multiplier"
        assert var_range.multiplier_min == 0.5
        assert var_range.multiplier_max == 2.0
    
    def test_invalid_variation_type(self):
        """Test that invalid variation type raises error."""
        with pytest.raises(ValueError, match="Unknown variation_type"):
            VariationRange(variation_type="invalid")
    
    def test_percentage_variation_missing_range(self):
        """Test that percentage variation without range raises error."""
        with pytest.raises(ValueError, match="percentage_range must be specified"):
            VariationRange(variation_type="percentage")
    
    def test_apply_percentage_variation(self):
        """Test applying percentage variation to baseline."""
        var_range = VariationRange(
            variation_type="percentage",
            percentage_range=(-20, 30)
        )
        
        min_val, max_val = var_range.apply_to_baseline(100.0)
        
        assert min_val == 80.0  # 100 * (1 - 20/100)
        assert max_val == 130.0  # 100 * (1 + 30/100)
    
    def test_apply_absolute_variation(self):
        """Test applying absolute variation."""
        var_range = VariationRange(
            variation_type="absolute",
            absolute_min=5.0,
            absolute_max=15.0
        )
        
        min_val, max_val = var_range.apply_to_baseline(10.0)
        
        assert min_val == 5.0
        assert max_val == 15.0
    
    def test_apply_multiplier_variation(self):
        """Test applying multiplier variation."""
        var_range = VariationRange(
            variation_type="multiplier",
            multiplier_min=0.5,
            multiplier_max=2.0
        )
        
        min_val, max_val = var_range.apply_to_baseline(10.0)
        
        assert min_val == 5.0  # 10 * 0.5
        assert max_val == 20.0  # 10 * 2.0
    
    def test_enforce_positive_constraint(self):
        """Test that positive constraint is enforced."""
        var_range = VariationRange(
            variation_type="percentage",
            percentage_range=(-150, 50),  # Would give negative values
            enforce_positive=True
        )
        
        min_val, max_val = var_range.apply_to_baseline(10.0)
        
        assert min_val >= 0.0  # Should be clipped to 0
        assert max_val == 15.0  # 10 * 1.5
    
    def test_rounding_constraint(self):
        """Test value rounding."""
        var_range = VariationRange(
            variation_type="percentage",
            percentage_range=(-33, 33),
            round_to_digits=2
        )
        
        min_val, max_val = var_range.apply_to_baseline(3.0)
        
        # 3 * 0.67 = 2.01, 3 * 1.33 = 3.99
        assert min_val == 2.01
        assert max_val == 3.99


class TestUnitConversion:
    """Test suite for UnitConversion class."""
    
    def test_concentration_conversion(self):
        """Test concentration unit conversions."""
        # Molarity to millimolar
        result = UnitConversion.convert_concentration(
            1.0, UnitType.MOLARITY, UnitType.MILLIMOLAR
        )
        assert result == 1000.0
        
        # Millimolar to micromolar
        result = UnitConversion.convert_concentration(
            1.0, UnitType.MILLIMOLAR, UnitType.MICROMOLAR
        )
        assert result == 1000.0
    
    def test_temperature_conversion(self):
        """Test temperature unit conversions."""
        # Celsius to Kelvin
        result = UnitConversion.convert_temperature(
            25.0, UnitType.TEMPERATURE_CELSIUS, UnitType.TEMPERATURE_KELVIN
        )
        assert result == 298.15
        
        # Kelvin to Celsius
        result = UnitConversion.convert_temperature(
            273.15, UnitType.TEMPERATURE_KELVIN, UnitType.TEMPERATURE_CELSIUS
        )
        assert result == 0.0
    
    def test_pressure_conversion(self):
        """Test pressure unit conversions."""
        # Bar to PSI
        result = UnitConversion.convert_pressure(
            1.0, UnitType.PRESSURE_BAR, UnitType.PRESSURE_PSI
        )
        assert abs(result - 14.5038) < 0.001
        
        # PSI to Bar
        result = UnitConversion.convert_pressure(
            14.5038, UnitType.PRESSURE_PSI, UnitType.PRESSURE_BAR
        )
        assert abs(result - 1.0) < 0.001
    
    def test_same_unit_conversion(self):
        """Test conversion between same units."""
        result = UnitConversion.convert_concentration(
            5.0, UnitType.MOLARITY, UnitType.MOLARITY
        )
        assert result == 5.0
    
    def test_invalid_conversion(self):
        """Test that invalid conversions raise errors."""
        with pytest.raises(ValueError, match="Cannot convert"):
            UnitConversion.convert_concentration(
                1.0, UnitType.MOLARITY, UnitType.TEMPERATURE_CELSIUS
            )


class TestExperimentalVariable:
    """Test suite for ExperimentalVariable class."""
    
    def test_continuous_variable_initialization(self):
        """Test continuous variable initialization."""
        var_range = VariationRange(
            variation_type="percentage",
            percentage_range=(-20, 30)
        )
        
        variable = ExperimentalVariable(
            name="temperature",
            variable_type=VariableType.CONTINUOUS,
            unit=UnitType.TEMPERATURE_CELSIUS,
            baseline_value=80.0,
            variation_range=var_range,
            description="Reaction temperature"
        )
        
        assert variable.name == "temperature"
        assert variable.variable_type == VariableType.CONTINUOUS
        assert variable.unit == UnitType.TEMPERATURE_CELSIUS
        assert variable.baseline_value == 80.0
        assert variable.bounds == (64.0, 104.0)  # 80 * (0.8, 1.3)
    
    def test_categorical_variable_initialization(self):
        """Test categorical variable initialization."""
        variable = ExperimentalVariable(
            name="solvent",
            variable_type=VariableType.CATEGORICAL,
            categories=["water", "ethanol", "acetone"],
            description="Reaction solvent"
        )
        
        assert variable.name == "solvent"
        assert variable.variable_type == VariableType.CATEGORICAL
        assert variable.categories == ["water", "ethanol", "acetone"]
    
    def test_invalid_variable_name(self):
        """Test that invalid variable names raise errors."""
        with pytest.raises(ValueError, match="not a valid identifier"):
            ExperimentalVariable(
                name="invalid-name",  # Hyphens not allowed
                variable_type=VariableType.CONTINUOUS
            )
    
    def test_categorical_without_categories(self):
        """Test that categorical variables require categories."""
        with pytest.raises(ValueError, match="Categories must be specified"):
            ExperimentalVariable(
                name="test_var",
                variable_type=VariableType.CATEGORICAL
            )
    
    def test_generate_levels_continuous(self):
        """Test level generation for continuous variables."""
        var_range = VariationRange(
            variation_type="absolute",
            absolute_min=1.0,
            absolute_max=10.0
        )
        
        variable = ExperimentalVariable(
            name="concentration",
            variable_type=VariableType.CONTINUOUS,
            baseline_value=5.0,
            variation_range=var_range
        )
        
        levels = variable.generate_levels(5)
        
        assert len(levels) == 5
        assert levels[0] == 1.0
        assert levels[-1] == 10.0
        assert all(1.0 <= level <= 10.0 for level in levels)
    
    def test_generate_levels_categorical(self):
        """Test level generation for categorical variables."""
        variable = ExperimentalVariable(
            name="solvent",
            variable_type=VariableType.CATEGORICAL,
            categories=["water", "ethanol", "acetone"]
        )
        
        levels = variable.generate_levels(3)
        
        assert len(levels) == 3
        assert set(levels) == set(["water", "ethanol", "acetone"])
    
    def test_generate_levels_integer(self):
        """Test level generation for integer variables."""
        var_range = VariationRange(
            variation_type="absolute",
            absolute_min=1.0,
            absolute_max=10.0
        )
        
        variable = ExperimentalVariable(
            name="cycles",
            variable_type=VariableType.INTEGER,
            baseline_value=5,
            variation_range=var_range
        )
        
        levels = variable.generate_levels(5)
        
        assert len(levels) == 5
        assert all(isinstance(level, int) for level in levels)
        assert all(1 <= level <= 10 for level in levels)
    
    def test_is_valid_value_continuous(self):
        """Test value validation for continuous variables."""
        var_range = VariationRange(
            variation_type="absolute",
            absolute_min=1.0,
            absolute_max=10.0
        )
        
        variable = ExperimentalVariable(
            name="concentration",
            variable_type=VariableType.CONTINUOUS,
            variation_range=var_range
        )
        
        assert variable.is_valid_value(5.0) is True
        assert variable.is_valid_value(0.5) is False  # Below minimum
        assert variable.is_valid_value(15.0) is False  # Above maximum
    
    def test_is_valid_value_categorical(self):
        """Test value validation for categorical variables."""
        variable = ExperimentalVariable(
            name="solvent",
            variable_type=VariableType.CATEGORICAL,
            categories=["water", "ethanol", "acetone"]
        )
        
        assert variable.is_valid_value("water") is True
        assert variable.is_valid_value("methanol") is False
    
    def test_unit_conversion(self):
        """Test unit conversion for variables."""
        variable = ExperimentalVariable(
            name="temperature",
            variable_type=VariableType.CONTINUOUS,
            unit=UnitType.TEMPERATURE_CELSIUS,
            baseline_value=25.0
        )
        
        kelvin_value = variable.convert_to_unit(25.0, UnitType.TEMPERATURE_KELVIN)
        assert kelvin_value == 298.15
    
    def test_add_dependency(self):
        """Test adding variable dependencies."""
        variable = ExperimentalVariable(
            name="pressure",
            variable_type=VariableType.CONTINUOUS
        )
        
        variable.add_dependency("temperature", "affects")
        
        assert "temperature:affects" in variable.dependencies
    
    def test_add_constraint(self):
        """Test adding variable constraints."""
        variable = ExperimentalVariable(
            name="concentration",
            variable_type=VariableType.CONTINUOUS
        )
        
        def positive_constraint(value):
            return value > 0
        
        variable.add_constraint(positive_constraint, "Must be positive")
        
        assert len(variable.constraints) == 1
        assert variable.constraints[0](5.0) is True
        assert variable.constraints[0](-1.0) is False
    
    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        var_range = VariationRange(
            variation_type="percentage",
            percentage_range=(-20, 30)
        )
        
        original = ExperimentalVariable(
            name="temperature",
            variable_type=VariableType.CONTINUOUS,
            unit=UnitType.TEMPERATURE_CELSIUS,
            baseline_value=80.0,
            variation_range=var_range,
            description="Test variable"
        )
        
        # Convert to dict and back
        var_dict = original.to_dict()
        restored = ExperimentalVariable.from_dict(var_dict)
        
        assert restored.name == original.name
        assert restored.variable_type == original.variable_type
        assert restored.unit == original.unit
        assert restored.baseline_value == original.baseline_value
        assert restored.description == original.description


class TestVariableGroup:
    """Test suite for VariableGroup class."""
    
    def test_group_initialization(self):
        """Test variable group initialization."""
        group = VariableGroup("reaction_conditions", "Main reaction parameters")
        
        assert group.name == "reaction_conditions"
        assert group.description == "Main reaction parameters"
        assert len(group.variables) == 0
    
    def test_add_remove_variables(self):
        """Test adding and removing variables from group."""
        group = VariableGroup("test_group")
        
        variable = ExperimentalVariable(
            name="temperature",
            variable_type=VariableType.CONTINUOUS
        )
        
        # Add variable
        group.add_variable(variable)
        
        assert len(group) == 1
        assert "temperature" in group.variables
        assert variable.group == "test_group"
        
        # Remove variable
        group.remove_variable("temperature")
        
        assert len(group) == 0
        assert "temperature" not in group.variables
    
    def test_group_constraint(self):
        """Test group-level constraints."""
        group = VariableGroup("mixture_group")
        
        # Add constraint that sum of fractions must equal 100
        def sum_constraint(values):
            total = sum(values.values())
            return abs(total - 100.0) < 1e-6
        
        group.add_group_constraint(sum_constraint, "Sum must equal 100%")
        
        # Test validation
        valid_values = {"component_a": 60.0, "component_b": 40.0}
        invalid_values = {"component_a": 60.0, "component_b": 50.0}
        
        assert group.validate_values(valid_values) is True
        assert group.validate_values(invalid_values) is False
    
    def test_group_iteration(self):
        """Test iterating over variables in group."""
        group = VariableGroup("test_group")
        
        var1 = ExperimentalVariable("var1", VariableType.CONTINUOUS)
        var2 = ExperimentalVariable("var2", VariableType.CATEGORICAL, categories=["A", "B"])
        
        group.add_variable(var1)
        group.add_variable(var2)
        
        variables = list(group)
        assert len(variables) == 2
        assert var1 in variables
        assert var2 in variables
    
    def test_get_variable_names(self):
        """Test getting variable names from group."""
        group = VariableGroup("test_group")
        
        var1 = ExperimentalVariable("temperature", VariableType.CONTINUOUS)
        var2 = ExperimentalVariable("pressure", VariableType.CONTINUOUS)
        
        group.add_variable(var1)
        group.add_variable(var2)
        
        names = group.get_variable_names()
        assert set(names) == {"temperature", "pressure"}
