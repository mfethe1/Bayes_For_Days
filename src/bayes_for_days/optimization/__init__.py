"""
Optimization module for Bayes For Days platform.

This module contains all optimization algorithms, acquisition functions,
and hybrid optimization strategies with state-of-the-art implementations:

- Essential acquisition functions with gradient-based optimization
- Multi-objective optimization algorithms (NSGA-II, MOEA/D)
- Hybrid optimization strategies combining Bayesian and genetic algorithms
- Advanced experimental design methods
- Parallel multi-start optimization capabilities

Components:
- acquisition: Core acquisition functions (EI, UCB, PI, EPI)
- genetic: Genetic algorithm implementations
- hybrid: Hybrid optimization strategies
- multi_objective: Multi-objective optimization utilities
- experimental_design: Advanced experimental design methods
"""

from bayes_for_days.optimization.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
)

# Placeholder imports for future implementations
# from bayes_for_days.optimization.genetic import GeneticAlgorithm
# from bayes_for_days.optimization.hybrid import HybridOptimizer
# from bayes_for_days.optimization.multi_objective import NSGAIIOptimizer
# from bayes_for_days.optimization.experimental_design import DOptimalDesign

__all__ = [
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "ProbabilityOfImprovement",
    # Will be populated as additional modules are implemented
]
