"""
Utilities module for Bayes For Days platform.

This module contains utility functions and helper classes
used throughout the optimization platform.

Components:
- sampling: Space-filling sampling methods for experimental design
- validation: Data validation utilities
- metrics: Performance and quality metrics
- helpers: General helper functions
"""

from bayes_for_days.utils.sampling import (
    latin_hypercube_sampling,
    sobol_sampling,
    halton_sampling,
    random_sampling,
    grid_sampling,
    maximin_sampling,
    adaptive_sampling,
)

__all__ = [
    "latin_hypercube_sampling",
    "sobol_sampling",
    "halton_sampling",
    "random_sampling",
    "grid_sampling",
    "maximin_sampling",
    "adaptive_sampling",
]