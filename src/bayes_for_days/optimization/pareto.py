"""
Pareto frontier management and quality metrics for multi-objective optimization.

This module provides comprehensive tools for managing Pareto fronts including:
- Efficient Pareto front storage and updates
- Dominated solution filtering
- Quality metrics (hypervolume, spread, convergence)
- Pareto front visualization utilities
- Integration with multi-objective optimization algorithms

Based on:
- Zitzler et al. (2003) "Performance assessment of multiobjective optimizers"
- Latest 2024-2025 research in multi-objective optimization metrics
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from bayes_for_days.core.types import (
    ExperimentPoint,
    OptimizationResult,
)

logger = logging.getLogger(__name__)


@dataclass
class ParetoFrontMetrics:
    """
    Comprehensive metrics for evaluating Pareto front quality.
    
    Includes standard multi-objective optimization metrics for
    convergence, diversity, and overall quality assessment.
    """
    hypervolume: float = 0.0
    spread: float = 0.0
    spacing: float = 0.0
    convergence_metric: float = 0.0
    diversity_metric: float = 0.0
    
    # Additional metrics
    front_size: int = 0
    dominated_hypervolume: float = 0.0
    epsilon_indicator: float = 0.0
    
    # Computational metadata
    computation_time: float = 0.0
    reference_point: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            'hypervolume': self.hypervolume,
            'spread': self.spread,
            'spacing': self.spacing,
            'convergence_metric': self.convergence_metric,
            'diversity_metric': self.diversity_metric,
            'front_size': self.front_size,
            'dominated_hypervolume': self.dominated_hypervolume,
            'epsilon_indicator': self.epsilon_indicator,
            'computation_time': self.computation_time,
            'reference_point': self.reference_point,
        }


class HypervolumeCalculator:
    """
    Hypervolume calculation for Pareto front quality assessment.
    
    Implements efficient hypervolume computation algorithms:
    - WFG algorithm for 2-3 objectives
    - Monte Carlo approximation for higher dimensions
    - Incremental hypervolume updates
    """
    
    def __init__(self, reference_point: Optional[List[float]] = None):
        """
        Initialize hypervolume calculator.
        
        Args:
            reference_point: Reference point for hypervolume calculation
                           (should be dominated by all Pareto points)
        """
        self.reference_point = reference_point
    
    def calculate(
        self, 
        pareto_front: List[ExperimentPoint],
        reference_point: Optional[List[float]] = None
    ) -> float:
        """
        Calculate hypervolume of Pareto front.
        
        Args:
            pareto_front: List of Pareto-optimal points
            reference_point: Reference point (overrides instance default)
            
        Returns:
            Hypervolume value
        """
        if not pareto_front:
            return 0.0
        
        # Extract objective values
        objectives_matrix = self._extract_objectives(pareto_front)
        if objectives_matrix.size == 0:
            return 0.0
        
        # Determine reference point
        ref_point = reference_point or self.reference_point
        if ref_point is None:
            ref_point = self._compute_reference_point(objectives_matrix)
        
        ref_point = np.array(ref_point)
        
        # Check dimensions
        n_objectives = objectives_matrix.shape[1]
        
        if n_objectives == 1:
            return self._hypervolume_1d(objectives_matrix, ref_point)
        elif n_objectives == 2:
            return self._hypervolume_2d(objectives_matrix, ref_point)
        elif n_objectives == 3:
            return self._hypervolume_3d(objectives_matrix, ref_point)
        else:
            # Use Monte Carlo for higher dimensions
            return self._hypervolume_monte_carlo(objectives_matrix, ref_point)
    
    def _extract_objectives(self, pareto_front: List[ExperimentPoint]) -> np.ndarray:
        """Extract objective values from experiment points."""
        if not pareto_front or not pareto_front[0].objectives:
            return np.array([])
        
        # Get objective names from first point
        obj_names = list(pareto_front[0].objectives.keys())
        
        # Extract values for all points
        objectives = []
        for point in pareto_front:
            if point.objectives:
                obj_values = [point.objectives.get(name, 0.0) for name in obj_names]
                objectives.append(obj_values)
        
        return np.array(objectives)
    
    def _compute_reference_point(self, objectives: np.ndarray) -> List[float]:
        """
        Compute reference point as worst values in each objective.
        
        Assumes minimization problem - reference point should be dominated
        by all Pareto points.
        """
        # Add small margin to ensure reference point is dominated
        margin = 0.1
        ref_point = np.max(objectives, axis=0) + margin
        return ref_point.tolist()
    
    def _hypervolume_1d(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate hypervolume for 1D case."""
        if objectives.shape[0] == 0:
            return 0.0
        
        # For 1D, hypervolume is just the distance from reference point
        best_value = np.min(objectives[:, 0])
        return max(0.0, ref_point[0] - best_value)
    
    def _hypervolume_2d(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate hypervolume for 2D case using sweep line algorithm."""
        if objectives.shape[0] == 0:
            return 0.0
        
        # Sort by first objective
        sorted_indices = np.argsort(objectives[:, 0])
        sorted_objectives = objectives[sorted_indices]
        
        hypervolume = 0.0
        prev_x = ref_point[0]
        
        for i, point in enumerate(sorted_objectives):
            x, y = point
            
            # Check if point is dominated by reference point
            if x >= ref_point[0] or y >= ref_point[1]:
                continue
            
            # Add rectangle area
            width = prev_x - x
            height = ref_point[1] - y
            
            if width > 0 and height > 0:
                hypervolume += width * height
            
            prev_x = x
        
        return hypervolume
    
    def _hypervolume_3d(self, objectives: np.ndarray, ref_point: np.ndarray) -> float:
        """Calculate hypervolume for 3D case using inclusion-exclusion."""
        if objectives.shape[0] == 0:
            return 0.0
        
        # Filter dominated points
        non_dominated = self._filter_dominated_points(objectives)
        
        if len(non_dominated) == 0:
            return 0.0
        
        hypervolume = 0.0
        
        # Use inclusion-exclusion principle
        for i in range(1, len(non_dominated) + 1):
            for combination in itertools.combinations(non_dominated, i):
                # Calculate intersection volume
                intersection = np.maximum.reduce(combination, axis=0)
                
                # Check if intersection is dominated by reference point
                if np.all(intersection < ref_point):
                    volume = np.prod(ref_point - intersection)
                    
                    # Add or subtract based on inclusion-exclusion
                    if i % 2 == 1:
                        hypervolume += volume
                    else:
                        hypervolume -= volume
        
        return max(0.0, hypervolume)
    
    def _hypervolume_monte_carlo(
        self, 
        objectives: np.ndarray, 
        ref_point: np.ndarray,
        n_samples: int = 100000
    ) -> float:
        """
        Calculate hypervolume using Monte Carlo approximation.
        
        Suitable for high-dimensional objective spaces (>3 objectives).
        """
        if objectives.shape[0] == 0:
            return 0.0
        
        # Define bounding box
        min_bounds = np.min(objectives, axis=0)
        max_bounds = ref_point
        
        # Check if bounding box is valid
        if np.any(min_bounds >= max_bounds):
            return 0.0
        
        # Generate random samples in bounding box
        n_objectives = len(ref_point)
        samples = np.random.uniform(
            low=min_bounds,
            high=max_bounds,
            size=(n_samples, n_objectives)
        )
        
        # Count samples dominated by at least one Pareto point
        dominated_count = 0
        
        for sample in samples:
            # Check if sample is dominated by any Pareto point
            dominated = np.any(np.all(objectives <= sample, axis=1))
            if dominated:
                dominated_count += 1
        
        # Calculate hypervolume as fraction of bounding box
        bounding_volume = np.prod(max_bounds - min_bounds)
        hypervolume = (dominated_count / n_samples) * bounding_volume
        
        return hypervolume
    
    def _filter_dominated_points(self, objectives: np.ndarray) -> List[np.ndarray]:
        """Filter out dominated points from objective matrix."""
        non_dominated = []
        
        for i, point1 in enumerate(objectives):
            is_dominated = False
            
            for j, point2 in enumerate(objectives):
                if i != j:
                    # Check if point1 is dominated by point2
                    if np.all(point2 <= point1) and np.any(point2 < point1):
                        is_dominated = True
                        break
            
            if not is_dominated:
                non_dominated.append(point1)
        
        return non_dominated


class SpreadCalculator:
    """
    Calculate spread (diversity) metrics for Pareto fronts.
    
    Implements various diversity measures:
    - Spread metric (Deb et al.)
    - Spacing metric
    - Distribution uniformity
    """
    
    @staticmethod
    def calculate_spread(pareto_front: List[ExperimentPoint]) -> float:
        """
        Calculate spread metric for Pareto front diversity.
        
        Based on Deb et al. (2002) NSGA-II paper.
        """
        if len(pareto_front) < 2:
            return 0.0
        
        objectives = SpreadCalculator._extract_objectives(pareto_front)
        if objectives.size == 0:
            return 0.0
        
        n_points, n_objectives = objectives.shape
        
        if n_points < 2:
            return 0.0
        
        # Calculate distances between consecutive points
        distances = []
        
        for i in range(n_objectives):
            # Sort by objective i
            sorted_indices = np.argsort(objectives[:, i])
            sorted_obj = objectives[sorted_indices]
            
            # Calculate distances between consecutive points
            for j in range(len(sorted_obj) - 1):
                dist = np.linalg.norm(sorted_obj[j + 1] - sorted_obj[j])
                distances.append(dist)
        
        if not distances:
            return 0.0
        
        # Calculate spread as coefficient of variation
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        if mean_distance == 0:
            return 0.0
        
        spread = std_distance / mean_distance
        return spread
    
    @staticmethod
    def calculate_spacing(pareto_front: List[ExperimentPoint]) -> float:
        """
        Calculate spacing metric for Pareto front uniformity.
        
        Measures how evenly distributed the solutions are.
        """
        if len(pareto_front) < 2:
            return 0.0
        
        objectives = SpreadCalculator._extract_objectives(pareto_front)
        if objectives.size == 0:
            return 0.0
        
        n_points = objectives.shape[0]
        
        if n_points < 2:
            return 0.0
        
        # Calculate minimum distance to nearest neighbor for each point
        min_distances = []
        
        for i in range(n_points):
            distances_to_others = []
            
            for j in range(n_points):
                if i != j:
                    dist = np.linalg.norm(objectives[i] - objectives[j])
                    distances_to_others.append(dist)
            
            if distances_to_others:
                min_distances.append(min(distances_to_others))
        
        if not min_distances:
            return 0.0
        
        # Calculate spacing as standard deviation of minimum distances
        spacing = np.std(min_distances)
        return spacing
    
    @staticmethod
    def _extract_objectives(pareto_front: List[ExperimentPoint]) -> np.ndarray:
        """Extract objective values from experiment points."""
        if not pareto_front or not pareto_front[0].objectives:
            return np.array([])
        
        obj_names = list(pareto_front[0].objectives.keys())
        objectives = []
        
        for point in pareto_front:
            if point.objectives:
                obj_values = [point.objectives.get(name, 0.0) for name in obj_names]
                objectives.append(obj_values)
        
        return np.array(objectives)


class ParetoFrontManager:
    """
    Comprehensive Pareto front management system.
    
    Features:
    - Efficient Pareto front storage and updates
    - Dominated solution filtering
    - Quality metrics computation
    - Pareto front archiving and history
    - Integration with optimization algorithms
    """
    
    def __init__(
        self,
        max_size: Optional[int] = None,
        reference_point: Optional[List[float]] = None,
        archive_history: bool = True,
    ):
        """
        Initialize Pareto front manager.
        
        Args:
            max_size: Maximum size of Pareto front (None for unlimited)
            reference_point: Reference point for hypervolume calculation
            archive_history: Whether to keep history of Pareto fronts
        """
        self.max_size = max_size
        self.reference_point = reference_point
        self.archive_history = archive_history
        
        # Current Pareto front
        self.pareto_front: List[ExperimentPoint] = []
        
        # History of Pareto fronts (if archiving enabled)
        self.front_history: List[List[ExperimentPoint]] = []
        
        # Quality metrics calculators
        self.hypervolume_calc = HypervolumeCalculator(reference_point)
        
        # Statistics
        self.update_count = 0
        self.total_points_processed = 0
        
        logger.info(f"Initialized Pareto front manager with max_size={max_size}")
    
    def update(self, new_points: List[ExperimentPoint]) -> Dict[str, Any]:
        """
        Update Pareto front with new points.
        
        Args:
            new_points: New experiment points to consider
            
        Returns:
            Update statistics and metrics
        """
        start_time = time.time()
        
        if not new_points:
            return {"updated": False, "reason": "No new points provided"}
        
        # Combine current front with new points
        all_points = self.pareto_front + new_points
        
        # Filter to get new Pareto front
        new_front = self._filter_pareto_front(all_points)
        
        # Archive current front if history is enabled
        if self.archive_history and self.pareto_front:
            self.front_history.append(self.pareto_front.copy())
        
        # Update statistics
        old_size = len(self.pareto_front)
        self.pareto_front = new_front
        new_size = len(self.pareto_front)
        
        self.update_count += 1
        self.total_points_processed += len(new_points)
        
        # Apply size limit if specified
        if self.max_size and len(self.pareto_front) > self.max_size:
            self.pareto_front = self._reduce_front_size(self.pareto_front, self.max_size)
        
        update_time = time.time() - start_time
        
        # Calculate metrics for new front
        metrics = self.calculate_metrics()
        
        update_stats = {
            "updated": True,
            "old_size": old_size,
            "new_size": len(self.pareto_front),
            "points_added": len(new_points),
            "update_time": update_time,
            "metrics": metrics.to_dict(),
        }
        
        logger.debug(f"Updated Pareto front: {old_size} -> {len(self.pareto_front)} points")
        
        return update_stats
    
    def _filter_pareto_front(self, points: List[ExperimentPoint]) -> List[ExperimentPoint]:
        """
        Filter points to extract Pareto front.
        
        Uses efficient domination checking to identify non-dominated solutions.
        """
        if not points:
            return []
        
        # Filter out infeasible points
        feasible_points = [p for p in points if p.is_feasible]
        
        if not feasible_points:
            return []
        
        pareto_front = []
        
        for i, point1 in enumerate(feasible_points):
            is_dominated = False
            
            for j, point2 in enumerate(feasible_points):
                if i != j and self._dominates(point2, point1):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(point1)
        
        return pareto_front
    
    def _dominates(self, point1: ExperimentPoint, point2: ExperimentPoint) -> bool:
        """
        Check if point1 dominates point2.
        
        Assumes maximization problem (higher objective values are better).
        """
        if not point1.objectives or not point2.objectives:
            return False
        
        # Get common objectives
        common_objectives = set(point1.objectives.keys()) & set(point2.objectives.keys())
        
        if not common_objectives:
            return False
        
        at_least_as_good = True
        strictly_better = False
        
        for obj_name in common_objectives:
            val1 = point1.objectives[obj_name]
            val2 = point2.objectives[obj_name]
            
            if val1 < val2:  # point1 is worse in this objective
                at_least_as_good = False
                break
            elif val1 > val2:  # point1 is better in this objective
                strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def _reduce_front_size(
        self, 
        front: List[ExperimentPoint], 
        target_size: int
    ) -> List[ExperimentPoint]:
        """
        Reduce Pareto front size while maintaining diversity.
        
        Uses crowding distance to select most diverse solutions.
        """
        if len(front) <= target_size:
            return front
        
        # Calculate crowding distances (simplified version)
        objectives = self._extract_objectives_matrix(front)
        
        if objectives.size == 0:
            return front[:target_size]
        
        crowding_distances = self._calculate_crowding_distances(objectives)
        
        # Sort by crowding distance (descending) and select top solutions
        sorted_indices = np.argsort(crowding_distances)[::-1]
        selected_indices = sorted_indices[:target_size]
        
        return [front[i] for i in selected_indices]
    
    def _extract_objectives_matrix(self, points: List[ExperimentPoint]) -> np.ndarray:
        """Extract objectives as matrix for calculations."""
        if not points or not points[0].objectives:
            return np.array([])
        
        obj_names = list(points[0].objectives.keys())
        objectives = []
        
        for point in points:
            if point.objectives:
                obj_values = [point.objectives.get(name, 0.0) for name in obj_names]
                objectives.append(obj_values)
        
        return np.array(objectives)
    
    def _calculate_crowding_distances(self, objectives: np.ndarray) -> np.ndarray:
        """Calculate crowding distances for diversity preservation."""
        n_points, n_objectives = objectives.shape
        distances = np.zeros(n_points)
        
        for obj_idx in range(n_objectives):
            # Sort by this objective
            sorted_indices = np.argsort(objectives[:, obj_idx])
            
            # Boundary solutions get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calculate range
            obj_range = objectives[sorted_indices[-1], obj_idx] - objectives[sorted_indices[0], obj_idx]
            
            if obj_range == 0:
                continue
            
            # Calculate distances for intermediate solutions
            for i in range(1, n_points - 1):
                idx = sorted_indices[i]
                if distances[idx] != float('inf'):
                    distance = (objectives[sorted_indices[i + 1], obj_idx] - 
                              objectives[sorted_indices[i - 1], obj_idx]) / obj_range
                    distances[idx] += distance
        
        return distances
    
    def calculate_metrics(self) -> ParetoFrontMetrics:
        """Calculate comprehensive quality metrics for current Pareto front."""
        start_time = time.time()
        
        metrics = ParetoFrontMetrics()
        metrics.front_size = len(self.pareto_front)
        
        if not self.pareto_front:
            metrics.computation_time = time.time() - start_time
            return metrics
        
        # Calculate hypervolume
        try:
            metrics.hypervolume = self.hypervolume_calc.calculate(self.pareto_front)
        except Exception as e:
            logger.warning(f"Hypervolume calculation failed: {e}")
            metrics.hypervolume = 0.0
        
        # Calculate spread and spacing
        try:
            metrics.spread = SpreadCalculator.calculate_spread(self.pareto_front)
            metrics.spacing = SpreadCalculator.calculate_spacing(self.pareto_front)
        except Exception as e:
            logger.warning(f"Diversity metrics calculation failed: {e}")
            metrics.spread = 0.0
            metrics.spacing = 0.0
        
        metrics.computation_time = time.time() - start_time
        metrics.reference_point = self.reference_point
        
        return metrics
    
    def get_pareto_front(self) -> List[ExperimentPoint]:
        """Get current Pareto front."""
        return self.pareto_front.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the Pareto front manager."""
        metrics = self.calculate_metrics()
        
        return {
            "current_front_size": len(self.pareto_front),
            "max_size": self.max_size,
            "update_count": self.update_count,
            "total_points_processed": self.total_points_processed,
            "history_length": len(self.front_history),
            "metrics": metrics.to_dict(),
            "reference_point": self.reference_point,
        }
