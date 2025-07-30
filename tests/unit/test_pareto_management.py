"""
Unit tests for Pareto frontier management system.

Tests the comprehensive Pareto front management including:
- Hypervolume calculation for different dimensions
- Spread and spacing metrics
- Pareto front filtering and updates
- Quality metrics computation
- Front size management and archiving
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from bayes_for_days.optimization.pareto import (
    ParetoFrontManager,
    HypervolumeCalculator,
    SpreadCalculator,
    ParetoFrontMetrics,
)
from bayes_for_days.core.types import ExperimentPoint


@pytest.fixture
def sample_experiment_points():
    """Create sample experiment points with known Pareto relationships."""
    points = [
        # Pareto optimal points (trade-offs)
        ExperimentPoint(
            parameters={"x1": 1.0, "x2": 1.0},
            objectives={"f1": 1.0, "f2": 4.0},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"x1": 2.0, "x2": 2.0},
            objectives={"f1": 2.0, "f2": 3.0},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"x1": 3.0, "x2": 3.0},
            objectives={"f1": 3.0, "f2": 2.0},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"x1": 4.0, "x2": 4.0},
            objectives={"f1": 4.0, "f2": 1.0},
            is_feasible=True,
        ),
        # Dominated point
        ExperimentPoint(
            parameters={"x1": 5.0, "x2": 5.0},
            objectives={"f1": 3.5, "f2": 3.5},
            is_feasible=True,
        ),
        # Infeasible point
        ExperimentPoint(
            parameters={"x1": 6.0, "x2": 6.0},
            objectives={"f1": 0.5, "f2": 0.5},
            is_feasible=False,
        ),
    ]
    return points


@pytest.fixture
def simple_2d_points():
    """Create simple 2D points for basic testing."""
    return [
        ExperimentPoint(
            parameters={"x": 1.0},
            objectives={"f1": 1.0, "f2": 3.0},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"x": 2.0},
            objectives={"f1": 2.0, "f2": 2.0},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"x": 3.0},
            objectives={"f1": 3.0, "f2": 1.0},
            is_feasible=True,
        ),
    ]


class TestHypervolumeCalculator:
    """Test suite for HypervolumeCalculator."""
    
    def test_hypervolume_calculator_initialization(self):
        """Test hypervolume calculator initialization."""
        ref_point = [5.0, 5.0]
        calc = HypervolumeCalculator(reference_point=ref_point)
        assert calc.reference_point == ref_point
    
    def test_hypervolume_empty_front(self):
        """Test hypervolume calculation for empty front."""
        calc = HypervolumeCalculator()
        hypervolume = calc.calculate([])
        assert hypervolume == 0.0
    
    def test_hypervolume_1d(self):
        """Test hypervolume calculation for 1D case."""
        points = [
            ExperimentPoint(
                parameters={"x": 1.0},
                objectives={"f1": 2.0},
                is_feasible=True,
            ),
        ]
        
        calc = HypervolumeCalculator()
        hypervolume = calc.calculate(points, reference_point=[5.0])
        
        # For 1D: hypervolume = reference_point - best_value = 5.0 - 2.0 = 3.0
        assert hypervolume == 3.0
    
    def test_hypervolume_2d(self, simple_2d_points):
        """Test hypervolume calculation for 2D case."""
        calc = HypervolumeCalculator()
        hypervolume = calc.calculate(simple_2d_points, reference_point=[5.0, 5.0])
        
        # Should be positive for non-dominated points
        assert hypervolume > 0.0
    
    def test_hypervolume_3d(self):
        """Test hypervolume calculation for 3D case."""
        points = [
            ExperimentPoint(
                parameters={"x": 1.0},
                objectives={"f1": 1.0, "f2": 1.0, "f3": 1.0},
                is_feasible=True,
            ),
            ExperimentPoint(
                parameters={"x": 2.0},
                objectives={"f1": 2.0, "f2": 2.0, "f3": 2.0},
                is_feasible=True,
            ),
        ]
        
        calc = HypervolumeCalculator()
        hypervolume = calc.calculate(points, reference_point=[5.0, 5.0, 5.0])
        
        assert hypervolume > 0.0
    
    def test_hypervolume_monte_carlo(self):
        """Test hypervolume calculation using Monte Carlo for high dimensions."""
        # Create 5D points
        points = [
            ExperimentPoint(
                parameters={"x": 1.0},
                objectives={f"f{i}": 1.0 for i in range(5)},
                is_feasible=True,
            ),
        ]
        
        calc = HypervolumeCalculator()
        hypervolume = calc.calculate(points, reference_point=[3.0] * 5)
        
        # Should use Monte Carlo approximation for >3 objectives
        assert hypervolume >= 0.0
    
    def test_reference_point_computation(self, simple_2d_points):
        """Test automatic reference point computation."""
        calc = HypervolumeCalculator()
        
        # Extract objectives matrix
        objectives = calc._extract_objectives(simple_2d_points)
        ref_point = calc._compute_reference_point(objectives)
        
        # Reference point should be above all objective values
        assert ref_point[0] > 3.0  # Max f1 is 3.0
        assert ref_point[1] > 3.0  # Max f2 is 3.0


class TestSpreadCalculator:
    """Test suite for SpreadCalculator."""
    
    def test_spread_empty_front(self):
        """Test spread calculation for empty front."""
        spread = SpreadCalculator.calculate_spread([])
        assert spread == 0.0
    
    def test_spread_single_point(self, simple_2d_points):
        """Test spread calculation for single point."""
        spread = SpreadCalculator.calculate_spread([simple_2d_points[0]])
        assert spread == 0.0
    
    def test_spread_multiple_points(self, simple_2d_points):
        """Test spread calculation for multiple points."""
        spread = SpreadCalculator.calculate_spread(simple_2d_points)
        assert spread >= 0.0
    
    def test_spacing_empty_front(self):
        """Test spacing calculation for empty front."""
        spacing = SpreadCalculator.calculate_spacing([])
        assert spacing == 0.0
    
    def test_spacing_single_point(self, simple_2d_points):
        """Test spacing calculation for single point."""
        spacing = SpreadCalculator.calculate_spacing([simple_2d_points[0]])
        assert spacing == 0.0
    
    def test_spacing_multiple_points(self, simple_2d_points):
        """Test spacing calculation for multiple points."""
        spacing = SpreadCalculator.calculate_spacing(simple_2d_points)
        assert spacing >= 0.0
    
    def test_extract_objectives(self, simple_2d_points):
        """Test objective extraction from experiment points."""
        objectives = SpreadCalculator._extract_objectives(simple_2d_points)
        
        assert objectives.shape == (3, 2)  # 3 points, 2 objectives
        assert np.array_equal(objectives[0], [1.0, 3.0])
        assert np.array_equal(objectives[1], [2.0, 2.0])
        assert np.array_equal(objectives[2], [3.0, 1.0])


class TestParetoFrontManager:
    """Test suite for ParetoFrontManager."""
    
    def test_manager_initialization(self):
        """Test Pareto front manager initialization."""
        manager = ParetoFrontManager(
            max_size=50,
            reference_point=[10.0, 10.0],
            archive_history=True,
        )
        
        assert manager.max_size == 50
        assert manager.reference_point == [10.0, 10.0]
        assert manager.archive_history is True
        assert len(manager.pareto_front) == 0
        assert len(manager.front_history) == 0
        assert manager.update_count == 0
    
    def test_update_empty_points(self):
        """Test updating with empty point list."""
        manager = ParetoFrontManager()
        result = manager.update([])
        
        assert result["updated"] is False
        assert "No new points" in result["reason"]
    
    def test_update_with_points(self, sample_experiment_points):
        """Test updating with experiment points."""
        manager = ParetoFrontManager(archive_history=True)
        
        # Initial update
        result = manager.update(sample_experiment_points)
        
        assert result["updated"] is True
        assert result["old_size"] == 0
        assert result["new_size"] > 0
        assert result["points_added"] == len(sample_experiment_points)
        assert "metrics" in result
        
        # Check that infeasible points are filtered out
        pareto_front = manager.get_pareto_front()
        assert all(point.is_feasible for point in pareto_front)
    
    def test_pareto_filtering(self, sample_experiment_points):
        """Test that dominated solutions are filtered out."""
        manager = ParetoFrontManager()
        manager.update(sample_experiment_points)
        
        pareto_front = manager.get_pareto_front()
        
        # Should have fewer points than input (dominated point should be removed)
        feasible_points = [p for p in sample_experiment_points if p.is_feasible]
        assert len(pareto_front) <= len(feasible_points)
        
        # Check that no point in the front dominates another
        for i, point1 in enumerate(pareto_front):
            for j, point2 in enumerate(pareto_front):
                if i != j:
                    assert not manager._dominates(point1, point2)
    
    def test_domination_check(self):
        """Test domination relationship checking."""
        manager = ParetoFrontManager()
        
        # Point 1 dominates Point 2 (better in both objectives)
        point1 = ExperimentPoint(
            parameters={"x": 1.0},
            objectives={"f1": 1.0, "f2": 1.0},
            is_feasible=True,
        )
        point2 = ExperimentPoint(
            parameters={"x": 2.0},
            objectives={"f1": 2.0, "f2": 2.0},
            is_feasible=True,
        )
        
        assert manager._dominates(point1, point2) is True
        assert manager._dominates(point2, point1) is False
        
        # Non-dominated points (trade-off)
        point3 = ExperimentPoint(
            parameters={"x": 3.0},
            objectives={"f1": 1.0, "f2": 3.0},
            is_feasible=True,
        )
        point4 = ExperimentPoint(
            parameters={"x": 4.0},
            objectives={"f1": 3.0, "f2": 1.0},
            is_feasible=True,
        )
        
        assert manager._dominates(point3, point4) is False
        assert manager._dominates(point4, point3) is False
    
    def test_max_size_enforcement(self, sample_experiment_points):
        """Test that max size is enforced."""
        manager = ParetoFrontManager(max_size=2)
        manager.update(sample_experiment_points)
        
        pareto_front = manager.get_pareto_front()
        assert len(pareto_front) <= 2
    
    def test_history_archiving(self, sample_experiment_points):
        """Test that front history is archived."""
        manager = ParetoFrontManager(archive_history=True)
        
        # First update
        manager.update(sample_experiment_points[:3])
        assert len(manager.front_history) == 0  # No history for first update
        
        # Second update
        manager.update(sample_experiment_points[3:])
        assert len(manager.front_history) == 1  # Previous front archived
    
    def test_metrics_calculation(self, simple_2d_points):
        """Test quality metrics calculation."""
        manager = ParetoFrontManager(reference_point=[5.0, 5.0])
        manager.update(simple_2d_points)
        
        metrics = manager.calculate_metrics()
        
        assert isinstance(metrics, ParetoFrontMetrics)
        assert metrics.front_size == len(manager.pareto_front)
        assert metrics.hypervolume >= 0.0
        assert metrics.spread >= 0.0
        assert metrics.spacing >= 0.0
        assert metrics.computation_time >= 0.0
    
    def test_statistics_retrieval(self, simple_2d_points):
        """Test comprehensive statistics retrieval."""
        manager = ParetoFrontManager(max_size=10, archive_history=True)
        manager.update(simple_2d_points)
        
        stats = manager.get_statistics()
        
        assert "current_front_size" in stats
        assert "max_size" in stats
        assert "update_count" in stats
        assert "total_points_processed" in stats
        assert "history_length" in stats
        assert "metrics" in stats
        
        assert stats["current_front_size"] == len(manager.pareto_front)
        assert stats["max_size"] == 10
        assert stats["update_count"] == 1
        assert stats["total_points_processed"] == len(simple_2d_points)
    
    def test_crowding_distance_calculation(self, simple_2d_points):
        """Test crowding distance calculation for diversity."""
        manager = ParetoFrontManager()
        
        # Extract objectives matrix
        objectives = manager._extract_objectives_matrix(simple_2d_points)
        distances = manager._calculate_crowding_distances(objectives)
        
        assert len(distances) == len(simple_2d_points)
        assert all(d >= 0.0 for d in distances)
        
        # Boundary solutions should have infinite distance
        assert distances[0] == float('inf') or distances[-1] == float('inf')
    
    def test_front_size_reduction(self, sample_experiment_points):
        """Test front size reduction while maintaining diversity."""
        manager = ParetoFrontManager()
        
        # Create a large front
        large_front = sample_experiment_points * 3  # Duplicate points
        reduced_front = manager._reduce_front_size(large_front, target_size=3)
        
        assert len(reduced_front) <= 3
        assert all(isinstance(point, ExperimentPoint) for point in reduced_front)


class TestParetoFrontMetrics:
    """Test suite for ParetoFrontMetrics."""
    
    def test_metrics_initialization(self):
        """Test metrics dataclass initialization."""
        metrics = ParetoFrontMetrics()
        
        assert metrics.hypervolume == 0.0
        assert metrics.spread == 0.0
        assert metrics.spacing == 0.0
        assert metrics.front_size == 0
        assert metrics.computation_time == 0.0
    
    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = ParetoFrontMetrics(
            hypervolume=1.5,
            spread=0.8,
            spacing=0.3,
            front_size=10,
            reference_point=[5.0, 5.0],
        )
        
        metrics_dict = metrics.to_dict()
        
        assert metrics_dict["hypervolume"] == 1.5
        assert metrics_dict["spread"] == 0.8
        assert metrics_dict["spacing"] == 0.3
        assert metrics_dict["front_size"] == 10
        assert metrics_dict["reference_point"] == [5.0, 5.0]
