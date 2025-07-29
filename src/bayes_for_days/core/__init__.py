"""
Core module for Bayes For Days optimization platform.

This module contains the fundamental classes and interfaces that form the backbone
of the Bayesian experimental design optimization system.

Components:
- config: Configuration management and settings
- experiment: Experiment definition and result handling
- optimizer: Main optimization engine interface
- base: Base classes and abstract interfaces
- types: Type definitions and data structures
"""

from bayes_for_days.core.config import Settings
from bayes_for_days.core.experiment import Experiment, ExperimentResult
from bayes_for_days.core.optimizer import BayesianOptimizer
from bayes_for_days.core.base import BaseModel, BaseOptimizer
from bayes_for_days.core.types import (
    ObjectiveType,
    OptimizationResult,
    ParameterSpace,
    AcquisitionFunction,
)

__all__ = [
    "Settings",
    "Experiment",
    "ExperimentResult", 
    "BayesianOptimizer",
    "BaseModel",
    "BaseOptimizer",
    "ObjectiveType",
    "OptimizationResult",
    "ParameterSpace",
    "AcquisitionFunction",
]
