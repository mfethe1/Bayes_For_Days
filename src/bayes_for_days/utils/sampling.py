"""
Sampling utilities for experimental design.

This module provides various sampling methods for generating initial
experimental designs and space-filling samples.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.stats import qmc
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


def latin_hypercube_sampling(
    bounds: List[Tuple[float, float]],
    n_samples: int,
    random_seed: Optional[int] = None,
    criterion: str = 'maximin'
) -> np.ndarray:
    """
    Generate Latin Hypercube samples within specified bounds.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        criterion: Optimization criterion ('maximin', 'correlation', 'ratio')
        
    Returns:
        Array of shape (n_samples, n_dimensions) with samples
    """
    if not bounds:
        raise ValueError("Bounds cannot be empty")
    
    n_dimensions = len(bounds)
    
    # Create Latin Hypercube sampler
    if random_seed is not None:
        np.random.seed(random_seed)
    
    sampler = qmc.LatinHypercube(d=n_dimensions, seed=random_seed)
    
    # Generate samples in [0, 1]^d
    unit_samples = sampler.random(n_samples)
    
    # Optimize the design if criterion is specified
    if criterion == 'maximin':
        # Optimize for maximin distance
        best_samples = unit_samples
        best_min_dist = _compute_min_distance(unit_samples)
        
        # Try multiple random starts
        for _ in range(10):
            candidate_samples = sampler.random(n_samples)
            min_dist = _compute_min_distance(candidate_samples)
            if min_dist > best_min_dist:
                best_samples = candidate_samples
                best_min_dist = min_dist
        
        unit_samples = best_samples
    
    # Scale to actual bounds
    scaled_samples = np.zeros_like(unit_samples)
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = low + unit_samples[:, i] * (high - low)
    
    logger.info(f"Generated {n_samples} Latin Hypercube samples in {n_dimensions}D space")
    return scaled_samples


def sobol_sampling(
    bounds: List[Tuple[float, float]],
    n_samples: int,
    random_seed: Optional[int] = None,
    scramble: bool = True
) -> np.ndarray:
    """
    Generate Sobol sequence samples within specified bounds.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        n_samples: Number of samples to generate
        random_seed: Random seed for scrambling
        scramble: Whether to scramble the sequence
        
    Returns:
        Array of shape (n_samples, n_dimensions) with samples
    """
    if not bounds:
        raise ValueError("Bounds cannot be empty")
    
    n_dimensions = len(bounds)
    
    # Create Sobol sampler
    sampler = qmc.Sobol(d=n_dimensions, scramble=scramble, seed=random_seed)
    
    # Generate samples in [0, 1]^d
    unit_samples = sampler.random(n_samples)
    
    # Scale to actual bounds
    scaled_samples = np.zeros_like(unit_samples)
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = low + unit_samples[:, i] * (high - low)
    
    logger.info(f"Generated {n_samples} Sobol samples in {n_dimensions}D space")
    return scaled_samples


def halton_sampling(
    bounds: List[Tuple[float, float]],
    n_samples: int,
    random_seed: Optional[int] = None,
    scramble: bool = True
) -> np.ndarray:
    """
    Generate Halton sequence samples within specified bounds.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        n_samples: Number of samples to generate
        random_seed: Random seed for scrambling
        scramble: Whether to scramble the sequence
        
    Returns:
        Array of shape (n_samples, n_dimensions) with samples
    """
    if not bounds:
        raise ValueError("Bounds cannot be empty")
    
    n_dimensions = len(bounds)
    
    # Create Halton sampler
    sampler = qmc.Halton(d=n_dimensions, scramble=scramble, seed=random_seed)
    
    # Generate samples in [0, 1]^d
    unit_samples = sampler.random(n_samples)
    
    # Scale to actual bounds
    scaled_samples = np.zeros_like(unit_samples)
    for i, (low, high) in enumerate(bounds):
        scaled_samples[:, i] = low + unit_samples[:, i] * (high - low)
    
    logger.info(f"Generated {n_samples} Halton samples in {n_dimensions}D space")
    return scaled_samples


def random_sampling(
    bounds: List[Tuple[float, float]],
    n_samples: int,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate random samples within specified bounds.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_dimensions) with samples
    """
    if not bounds:
        raise ValueError("Bounds cannot be empty")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_dimensions = len(bounds)
    samples = np.zeros((n_samples, n_dimensions))
    
    for i, (low, high) in enumerate(bounds):
        samples[:, i] = np.random.uniform(low, high, n_samples)
    
    logger.info(f"Generated {n_samples} random samples in {n_dimensions}D space")
    return samples


def grid_sampling(
    bounds: List[Tuple[float, float]],
    n_points_per_dim: int
) -> np.ndarray:
    """
    Generate grid samples within specified bounds.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        n_points_per_dim: Number of points per dimension
        
    Returns:
        Array of shape (n_points_per_dim^n_dimensions, n_dimensions) with samples
    """
    if not bounds:
        raise ValueError("Bounds cannot be empty")
    
    # Create grid points for each dimension
    grid_points = []
    for low, high in bounds:
        points = np.linspace(low, high, n_points_per_dim)
        grid_points.append(points)
    
    # Create meshgrid
    mesh = np.meshgrid(*grid_points, indexing='ij')
    
    # Flatten and combine
    n_dimensions = len(bounds)
    n_total_points = n_points_per_dim ** n_dimensions
    samples = np.zeros((n_total_points, n_dimensions))
    
    for i in range(n_dimensions):
        samples[:, i] = mesh[i].flatten()
    
    logger.info(f"Generated {n_total_points} grid samples in {n_dimensions}D space")
    return samples


def maximin_sampling(
    bounds: List[Tuple[float, float]],
    n_samples: int,
    n_candidates: int = 1000,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate maximin distance samples within specified bounds.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        n_samples: Number of samples to generate
        n_candidates: Number of candidate samples to consider
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_dimensions) with samples
    """
    if not bounds:
        raise ValueError("Bounds cannot be empty")
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate candidate samples
    candidates = random_sampling(bounds, n_candidates, random_seed)
    
    # Select samples using maximin criterion
    selected_indices = []
    remaining_indices = list(range(n_candidates))
    
    # Select first point randomly
    first_idx = np.random.choice(remaining_indices)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    
    # Select remaining points to maximize minimum distance
    for _ in range(n_samples - 1):
        best_idx = None
        best_min_dist = -1
        
        for candidate_idx in remaining_indices:
            # Compute minimum distance to already selected points
            min_dist = float('inf')
            for selected_idx in selected_indices:
                dist = np.linalg.norm(candidates[candidate_idx] - candidates[selected_idx])
                min_dist = min(min_dist, dist)
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_idx = candidate_idx
        
        if best_idx is not None:
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
    
    selected_samples = candidates[selected_indices]
    
    logger.info(f"Generated {n_samples} maximin samples in {len(bounds)}D space")
    return selected_samples


def _compute_min_distance(samples: np.ndarray) -> float:
    """Compute minimum pairwise distance in a sample set."""
    n_samples = samples.shape[0]
    min_dist = float('inf')
    
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dist = np.linalg.norm(samples[i] - samples[j])
            min_dist = min(min_dist, dist)
    
    return min_dist


def adaptive_sampling(
    bounds: List[Tuple[float, float]],
    n_samples: int,
    existing_samples: Optional[np.ndarray] = None,
    criterion: str = 'maximin',
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate adaptive samples that complement existing samples.
    
    Args:
        bounds: List of (min, max) tuples for each dimension
        n_samples: Number of new samples to generate
        existing_samples: Existing samples to complement
        criterion: Sampling criterion ('maximin', 'space_filling')
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of shape (n_samples, n_dimensions) with new samples
    """
    if not bounds:
        raise ValueError("Bounds cannot be empty")
    
    if existing_samples is None:
        # No existing samples, use standard sampling
        return latin_hypercube_sampling(bounds, n_samples, random_seed)
    
    if criterion == 'maximin':
        # Generate candidates and select those that maximize distance to existing samples
        n_candidates = max(1000, n_samples * 10)
        candidates = latin_hypercube_sampling(bounds, n_candidates, random_seed)
        
        selected_indices = []
        remaining_indices = list(range(n_candidates))
        
        for _ in range(n_samples):
            best_idx = None
            best_min_dist = -1
            
            for candidate_idx in remaining_indices:
                candidate = candidates[candidate_idx]
                
                # Compute minimum distance to existing samples
                min_dist_existing = float('inf')
                for existing_sample in existing_samples:
                    dist = np.linalg.norm(candidate - existing_sample)
                    min_dist_existing = min(min_dist_existing, dist)
                
                # Compute minimum distance to already selected samples
                min_dist_selected = float('inf')
                for selected_idx in selected_indices:
                    dist = np.linalg.norm(candidate - candidates[selected_idx])
                    min_dist_selected = min(min_dist_selected, dist)
                
                # Overall minimum distance
                overall_min_dist = min(min_dist_existing, min_dist_selected)
                
                if overall_min_dist > best_min_dist:
                    best_min_dist = overall_min_dist
                    best_idx = candidate_idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
        
        new_samples = candidates[selected_indices]
    
    else:
        # Default to Latin Hypercube
        new_samples = latin_hypercube_sampling(bounds, n_samples, random_seed)
    
    logger.info(f"Generated {n_samples} adaptive samples complementing {len(existing_samples)} existing samples")
    return new_samples
