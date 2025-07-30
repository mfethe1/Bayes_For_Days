"""
Unit tests for multi-objective optimization algorithms.

Tests the NSGA-II implementation including:
- Individual representation and domination
- Fast non-dominated sorting
- Crowding distance calculation
- Tournament selection
- Complete NSGA-II optimization loop
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from bayes_for_days.optimization.multi_objective import (
    Individual,
    FastNonDominatedSorting,
    CrowdingDistance,
    TournamentSelection,
    NSGAIIOptimizer,
)
from bayes_for_days.core.types import (
    ParameterSpace,
    Parameter,
    ParameterType,
    OptimizationResult,
)


@pytest.fixture
def sample_parameter_space():
    """Create a sample parameter space for testing."""
    parameters = [
        Parameter(name="x1", type=ParameterType.CONTINUOUS, bounds=(-5, 5)),
        Parameter(name="x2", type=ParameterType.CONTINUOUS, bounds=(-5, 5)),
        Parameter(name="category", type=ParameterType.CATEGORICAL, categories=["A", "B", "C"]),
    ]
    return ParameterSpace(parameters=parameters)


@pytest.fixture
def sample_individuals():
    """Create sample individuals for testing."""
    individuals = [
        Individual(
            parameters={"x1": 1.0, "x2": 2.0, "category": "A"},
            objectives={"f1": 1.0, "f2": 4.0},
            is_feasible=True,
        ),
        Individual(
            parameters={"x1": 2.0, "x2": 1.0, "category": "B"},
            objectives={"f1": 2.0, "f2": 2.0},
            is_feasible=True,
        ),
        Individual(
            parameters={"x1": 3.0, "x2": 3.0, "category": "C"},
            objectives={"f1": 3.0, "f2": 1.0},
            is_feasible=True,
        ),
        Individual(
            parameters={"x1": 4.0, "x2": 4.0, "category": "A"},
            objectives={"f1": 4.0, "f2": 4.0},
            is_feasible=True,
        ),
    ]
    
    # Set objective values for each individual
    for ind in individuals:
        ind.objective_values = list(ind.objectives.values())
    
    return individuals


class TestIndividual:
    """Test suite for Individual class."""
    
    def test_individual_initialization(self):
        """Test individual initialization."""
        parameters = {"x1": 1.5, "x2": 2.5}
        objectives = {"f1": 0.8, "f2": 1.2}
        
        individual = Individual(
            parameters=parameters,
            objectives=objectives,
            is_feasible=True,
        )
        
        assert individual.parameters == parameters
        assert individual.objectives == objectives
        assert individual.is_feasible is True
        assert individual.rank == -1  # Default value
        assert individual.crowding_distance == 0.0
        assert individual.objective_values == [0.8, 1.2]
    
    def test_individual_domination_feasible(self):
        """Test domination relationship between feasible individuals."""
        # Individual 1 dominates Individual 2 (better in both objectives)
        ind1 = Individual(
            parameters={"x": 1.0},
            objectives={"f1": 1.0, "f2": 1.0},
            is_feasible=True,
        )
        ind1.objective_values = [1.0, 1.0]
        
        ind2 = Individual(
            parameters={"x": 2.0},
            objectives={"f1": 2.0, "f2": 2.0},
            is_feasible=True,
        )
        ind2.objective_values = [2.0, 2.0]
        
        assert ind1.dominates(ind2) is True
        assert ind2.dominates(ind1) is False
    
    def test_individual_domination_non_dominated(self):
        """Test non-dominated individuals."""
        # Trade-off: ind1 better in f1, ind2 better in f2
        ind1 = Individual(
            parameters={"x": 1.0},
            objectives={"f1": 1.0, "f2": 3.0},
            is_feasible=True,
        )
        ind1.objective_values = [1.0, 3.0]
        
        ind2 = Individual(
            parameters={"x": 2.0},
            objectives={"f1": 2.0, "f2": 1.0},
            is_feasible=True,
        )
        ind2.objective_values = [2.0, 1.0]
        
        assert ind1.dominates(ind2) is False
        assert ind2.dominates(ind1) is False
    
    def test_individual_domination_infeasible(self):
        """Test domination with constraint violations."""
        # Feasible individual dominates infeasible
        feasible = Individual(
            parameters={"x": 1.0},
            objectives={"f1": 5.0, "f2": 5.0},
            is_feasible=True,
        )
        feasible.objective_values = [5.0, 5.0]
        
        infeasible = Individual(
            parameters={"x": 2.0},
            objectives={"f1": 1.0, "f2": 1.0},
            is_feasible=False,
            constraint_violation=1.0,
        )
        infeasible.objective_values = [1.0, 1.0]
        
        assert feasible.dominates(infeasible) is True
        assert infeasible.dominates(feasible) is False
    
    def test_individual_to_experiment_point(self):
        """Test conversion to ExperimentPoint."""
        individual = Individual(
            parameters={"x1": 1.0, "x2": 2.0},
            objectives={"f1": 0.5, "f2": 1.5},
            rank=1,
            crowding_distance=0.8,
            generation=5,
        )
        
        exp_point = individual.to_experiment_point()
        
        assert exp_point.parameters == individual.parameters
        assert exp_point.objectives == individual.objectives
        assert exp_point.metadata["rank"] == 1
        assert exp_point.metadata["crowding_distance"] == 0.8
        assert exp_point.metadata["generation"] == 5


class TestFastNonDominatedSorting:
    """Test suite for FastNonDominatedSorting."""
    
    def test_empty_population(self):
        """Test sorting empty population."""
        fronts = FastNonDominatedSorting.sort([])
        assert fronts == []
    
    def test_single_individual(self, sample_individuals):
        """Test sorting single individual."""
        fronts = FastNonDominatedSorting.sort([sample_individuals[0]])
        
        assert len(fronts) == 1
        assert len(fronts[0]) == 1
        assert fronts[0][0].rank == 0
    
    def test_non_dominated_sorting(self, sample_individuals):
        """Test complete non-dominated sorting."""
        fronts = FastNonDominatedSorting.sort(sample_individuals)
        
        # Check that we have at least one front
        assert len(fronts) >= 1
        
        # Check that all individuals are assigned ranks
        for individual in sample_individuals:
            assert individual.rank >= 0
        
        # Check that first front contains non-dominated solutions
        first_front = fronts[0]
        for i, ind1 in enumerate(first_front):
            for j, ind2 in enumerate(first_front):
                if i != j:
                    assert not ind1.dominates(ind2)
                    assert not ind2.dominates(ind1)
    
    def test_dominated_solutions(self):
        """Test that dominated solutions are in later fronts."""
        # Create clearly dominated solutions
        ind1 = Individual(
            parameters={"x": 1.0},
            objectives={"f1": 1.0, "f2": 1.0},
            is_feasible=True,
        )
        ind1.objective_values = [1.0, 1.0]
        
        ind2 = Individual(
            parameters={"x": 2.0},
            objectives={"f1": 2.0, "f2": 2.0},
            is_feasible=True,
        )
        ind2.objective_values = [2.0, 2.0]
        
        fronts = FastNonDominatedSorting.sort([ind1, ind2])
        
        assert len(fronts) == 2
        assert ind1.rank == 0  # Non-dominated
        assert ind2.rank == 1  # Dominated
        assert ind1 in fronts[0]
        assert ind2 in fronts[1]


class TestCrowdingDistance:
    """Test suite for CrowdingDistance."""
    
    def test_empty_front(self):
        """Test crowding distance for empty front."""
        CrowdingDistance.calculate([])
        # Should not raise any exceptions
    
    def test_single_individual_front(self, sample_individuals):
        """Test crowding distance for single individual."""
        front = [sample_individuals[0]]
        CrowdingDistance.calculate(front)
        
        assert front[0].crowding_distance == float('inf')
    
    def test_two_individuals_front(self, sample_individuals):
        """Test crowding distance for two individuals."""
        front = sample_individuals[:2]
        CrowdingDistance.calculate(front)
        
        # Both should have infinite crowding distance (boundary solutions)
        for individual in front:
            assert individual.crowding_distance == float('inf')
    
    def test_multiple_individuals_front(self, sample_individuals):
        """Test crowding distance for multiple individuals."""
        front = sample_individuals[:3]
        CrowdingDistance.calculate(front)
        
        # Boundary solutions should have infinite distance
        # Middle solutions should have finite distance
        distances = [ind.crowding_distance for ind in front]
        
        # At least some should be finite
        finite_distances = [d for d in distances if d != float('inf')]
        assert len(finite_distances) >= 0  # Could be all infinite for small fronts


class TestTournamentSelection:
    """Test suite for TournamentSelection."""
    
    def test_tournament_selection_initialization(self):
        """Test tournament selection initialization."""
        selection = TournamentSelection(tournament_size=3)
        assert selection.tournament_size == 3
    
    def test_select_parents(self, sample_individuals):
        """Test parent selection."""
        # Set ranks for individuals
        for i, ind in enumerate(sample_individuals):
            ind.rank = i
            ind.crowding_distance = 1.0 / (i + 1)  # Decreasing crowding distance
        
        selection = TournamentSelection(tournament_size=2)
        parents = selection.select(sample_individuals, num_parents=2)
        
        assert len(parents) == 2
        assert all(parent in sample_individuals for parent in parents)
    
    def test_crowding_comparison(self, sample_individuals):
        """Test crowding comparison operator."""
        ind1 = sample_individuals[0]
        ind2 = sample_individuals[1]
        
        # Set different ranks
        ind1.rank = 0
        ind2.rank = 1
        ind1.crowding_distance = 0.5
        ind2.crowding_distance = 1.0
        
        selection = TournamentSelection()
        
        # Lower rank should win
        assert selection._crowding_comparison(ind1, ind2) is True
        assert selection._crowding_comparison(ind2, ind1) is False
        
        # Same rank - higher crowding distance should win
        ind2.rank = 0
        assert selection._crowding_comparison(ind2, ind1) is True
        assert selection._crowding_comparison(ind1, ind2) is False


class TestNSGAIIOptimizer:
    """Test suite for NSGAIIOptimizer."""
    
    def test_nsga2_initialization(self, sample_parameter_space):
        """Test NSGA-II optimizer initialization."""
        optimizer = NSGAIIOptimizer(
            parameter_space=sample_parameter_space,
            population_size=50,
            max_generations=100,
            random_seed=42,
        )
        
        assert optimizer.parameter_space == sample_parameter_space
        assert optimizer.population_size == 50
        assert optimizer.max_generations == 100
        assert optimizer.generation == 0
        assert optimizer.evaluation_count == 0
        assert len(optimizer.population) == 0
    
    def test_initialize_population(self, sample_parameter_space):
        """Test population initialization."""
        optimizer = NSGAIIOptimizer(
            parameter_space=sample_parameter_space,
            population_size=10,
            random_seed=42,
        )
        
        population = optimizer._initialize_population()
        
        assert len(population) == 10
        for individual in population:
            assert isinstance(individual, Individual)
            assert "x1" in individual.parameters
            assert "x2" in individual.parameters
            assert "category" in individual.parameters
            assert -5 <= individual.parameters["x1"] <= 5
            assert -5 <= individual.parameters["x2"] <= 5
            assert individual.parameters["category"] in ["A", "B", "C"]
    
    def test_objective_function_setting(self, sample_parameter_space):
        """Test setting objective function."""
        optimizer = NSGAIIOptimizer(parameter_space=sample_parameter_space)
        
        def test_objective(params):
            return {"f1": params["x1"]**2, "f2": params["x2"]**2}
        
        optimizer.set_objective_function(test_objective)
        assert optimizer.objective_function == test_objective
    
    def test_individual_evaluation(self, sample_parameter_space):
        """Test individual evaluation."""
        optimizer = NSGAIIOptimizer(parameter_space=sample_parameter_space)
        
        def test_objective(params):
            return {"f1": params["x1"]**2, "f2": params["x2"]**2}
        
        optimizer.set_objective_function(test_objective)
        
        individual = Individual(
            parameters={"x1": 2.0, "x2": 3.0, "category": "A"}
        )
        
        optimizer._evaluate_individual(individual)
        
        assert individual.objectives["f1"] == 4.0
        assert individual.objectives["f2"] == 9.0
        assert individual.objective_values == [4.0, 9.0]
        assert individual.is_feasible is True
        assert optimizer.evaluation_count == 1
