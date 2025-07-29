"""
Bayes For Days: Comprehensive Bayesian Experimental Design Optimization Platform

A sophisticated platform for multi-objective Bayesian optimization with adaptive learning
capabilities, designed for experimental design and optimization scenarios.

Key Features:
- Multi-objective Bayesian optimization with Pareto frontier exploration
- Hybrid optimization strategies combining Bayesian and genetic algorithms
- Ensemble learning with Gaussian Processes, Random Forests, and Neural Networks
- Advanced experimental design methodologies (D-optimal, Latin Hypercube, RSM)
- Interactive web-based visualization and analysis tools
- Robust data management with CSV import/export capabilities
- Adaptive learning systems with uncertainty quantification

Modules:
- core: Core optimization engine and base classes
- optimization: Bayesian optimization, genetic algorithms, and hybrid strategies
- models: Surrogate models (GP, RF, NN) and ensemble methods
- data: Data management, validation, and preprocessing
- api: RESTful API endpoints and WebSocket communication
- web: Frontend interface and visualization components
- utils: Utility functions and helper classes
- visualization: Interactive plotting and analysis tools
"""

__version__ = "0.1.0"
__author__ = "Bayes For Days Team"
__email__ = "team@bayesfordays.com"

# Core imports for easy access
from bayes_for_days.core.config import Settings
from bayes_for_days.core.experiment import Experiment, ExperimentResult
from bayes_for_days.core.optimizer import BayesianOptimizer

# Version info
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "Settings",
    "Experiment", 
    "ExperimentResult",
    "BayesianOptimizer",
]


def get_version() -> str:
    """Get the current version of Bayes For Days."""
    return __version__


def get_info() -> dict:
    """Get package information."""
    return {
        "name": "bayes-for-days",
        "version": __version__,
        "author": __author__,
        "email": __email__,
        "description": "Comprehensive Bayesian experimental design optimization platform",
        "homepage": "https://github.com/mfethe1/Bayes_For_Days",
    }
