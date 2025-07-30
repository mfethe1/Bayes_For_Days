"""
Multi-objective optimization algorithms for Bayes For Days platform.

This module implements state-of-the-art multi-objective optimization algorithms:
- NSGA-II (Non-dominated Sorting Genetic Algorithm II)
- Pareto frontier management and quality metrics
- Constraint handling for multi-objective problems
- Integration with Bayesian optimization framework

Based on:
- Deb et al. (2002) "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- Latest 2024-2025 research in multi-objective optimization
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from bayes_for_days.core.base import BaseOptimizer
from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    Parameter,
    ParameterType,
    OptimizationResult,
    ConstraintFunction,
)

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """
    Individual in the multi-objective optimization population.
    
    Represents a single solution with parameters, objectives, constraints,
    and ranking information for NSGA-II algorithm.
    """
    parameters: Dict[str, float]
    objectives: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, float] = field(default_factory=dict)
    
    # NSGA-II specific attributes
    rank: int = -1
    crowding_distance: float = 0.0
    constraint_violation: float = 0.0
    is_feasible: bool = True
    
    # Metadata
    generation: int = 0
    evaluation_time: float = 0.0
    
    def __post_init__(self):
        """Initialize derived attributes."""
        self.objective_values = list(self.objectives.values()) if self.objectives else []
        self.constraint_values = list(self.constraints.values()) if self.constraints else []
    
    def dominates(self, other: 'Individual') -> bool:
        """
        Check if this individual dominates another individual.
        
        For minimization problems:
        - At least as good in all objectives
        - Strictly better in at least one objective
        - Both individuals must be feasible or this one has less constraint violation
        """
        if not self.is_feasible and not other.is_feasible:
            # Both infeasible - compare constraint violations
            return self.constraint_violation < other.constraint_violation
        elif not self.is_feasible:
            # This is infeasible, other is feasible
            return False
        elif not other.is_feasible:
            # This is feasible, other is infeasible
            return True
        
        # Both feasible - compare objectives
        if not self.objective_values or not other.objective_values:
            return False
        
        at_least_as_good = True
        strictly_better = False
        
        for i, (obj1, obj2) in enumerate(zip(self.objective_values, other.objective_values)):
            if obj1 > obj2:  # Assuming maximization (convert if needed)
                at_least_as_good = False
                break
            elif obj1 < obj2:
                strictly_better = True
        
        return at_least_as_good and strictly_better
    
    def to_experiment_point(self) -> ExperimentPoint:
        """Convert individual to ExperimentPoint for integration."""
        return ExperimentPoint(
            parameters=self.parameters.copy(),
            objectives=self.objectives.copy(),
            constraints=self.constraints.copy(),
            is_feasible=self.is_feasible,
            metadata={
                'rank': self.rank,
                'crowding_distance': self.crowding_distance,
                'generation': self.generation,
            }
        )


class FastNonDominatedSorting:
    """
    Fast non-dominated sorting algorithm for NSGA-II.
    
    Implements the O(MN²) algorithm from Deb et al. (2002) where:
    - M is the number of objectives
    - N is the population size
    """
    
    @staticmethod
    def sort(population: List[Individual]) -> List[List[Individual]]:
        """
        Perform fast non-dominated sorting on population.
        
        Args:
            population: List of individuals to sort
            
        Returns:
            List of fronts, where each front is a list of non-dominated individuals
        """
        if not population:
            return []
        
        # Initialize data structures
        fronts = []
        dominated_solutions = {i: [] for i in range(len(population))}
        domination_count = {i: 0 for i in range(len(population))}
        
        # First front
        first_front = []
        
        # For each individual
        for i, individual_i in enumerate(population):
            for j, individual_j in enumerate(population):
                if i != j:
                    if individual_i.dominates(individual_j):
                        # i dominates j
                        dominated_solutions[i].append(j)
                    elif individual_j.dominates(individual_i):
                        # j dominates i
                        domination_count[i] += 1
            
            # If no one dominates this individual, it belongs to first front
            if domination_count[i] == 0:
                individual_i.rank = 0
                first_front.append(individual_i)
        
        fronts.append(first_front)
        
        # Find subsequent fronts
        front_index = 0
        while fronts[front_index]:
            next_front = []
            
            for individual_i in fronts[front_index]:
                i = population.index(individual_i)
                
                # For each individual dominated by i
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    
                    # If j is no longer dominated by anyone
                    if domination_count[j] == 0:
                        population[j].rank = front_index + 1
                        next_front.append(population[j])
            
            if next_front:
                fronts.append(next_front)
            front_index += 1
        
        return fronts


class CrowdingDistance:
    """
    Crowding distance calculation for NSGA-II.
    
    Maintains diversity in the population by calculating the density
    of solutions surrounding each individual.
    """
    
    @staticmethod
    def calculate(front: List[Individual]) -> None:
        """
        Calculate crowding distance for individuals in a front.
        
        Args:
            front: List of individuals in the same front
        """
        if not front:
            return
        
        front_size = len(front)
        
        # Initialize crowding distance
        for individual in front:
            individual.crowding_distance = 0.0
        
        if front_size <= 2:
            # Boundary solutions have infinite crowding distance
            for individual in front:
                individual.crowding_distance = float('inf')
            return
        
        # Get number of objectives
        if not front[0].objective_values:
            return
        
        num_objectives = len(front[0].objective_values)
        
        # For each objective
        for obj_index in range(num_objectives):
            # Sort front by this objective
            front.sort(key=lambda x: x.objective_values[obj_index])
            
            # Boundary solutions have infinite crowding distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate objective range
            obj_min = front[0].objective_values[obj_index]
            obj_max = front[-1].objective_values[obj_index]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue  # All solutions have same objective value
            
            # Calculate crowding distance for intermediate solutions
            for i in range(1, front_size - 1):
                if front[i].crowding_distance != float('inf'):
                    distance = (front[i + 1].objective_values[obj_index] - 
                              front[i - 1].objective_values[obj_index]) / obj_range
                    front[i].crowding_distance += distance


class TournamentSelection:
    """
    Tournament selection for NSGA-II with crowding comparison.
    
    Selects individuals based on:
    1. Rank (lower is better)
    2. Crowding distance (higher is better for same rank)
    """
    
    def __init__(self, tournament_size: int = 2):
        """
        Initialize tournament selection.
        
        Args:
            tournament_size: Number of individuals in each tournament
        """
        self.tournament_size = tournament_size
    
    def select(self, population: List[Individual], num_parents: int) -> List[Individual]:
        """
        Select parents using tournament selection.
        
        Args:
            population: Population to select from
            num_parents: Number of parents to select
            
        Returns:
            Selected parents
        """
        parents = []
        
        for _ in range(num_parents):
            # Select random individuals for tournament
            tournament = random.sample(population, 
                                     min(self.tournament_size, len(population)))
            
            # Find best individual in tournament
            best = tournament[0]
            for individual in tournament[1:]:
                if self._crowding_comparison(individual, best):
                    best = individual
            
            parents.append(best)
        
        return parents
    
    def _crowding_comparison(self, ind1: Individual, ind2: Individual) -> bool:
        """
        Crowding comparison operator.
        
        Returns True if ind1 is better than ind2 based on:
        1. Rank (lower is better)
        2. Crowding distance (higher is better for same rank)
        """
        if ind1.rank < ind2.rank:
            return True
        elif ind1.rank > ind2.rank:
            return False
        else:
            # Same rank - compare crowding distance
            return ind1.crowding_distance > ind2.crowding_distance


class NSGAIIOptimizer(BaseOptimizer):
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.
    
    Features:
    - Fast non-dominated sorting
    - Crowding distance for diversity preservation
    - Tournament selection with crowding comparison
    - Constraint handling with penalty methods
    - Support for 2-10 objectives
    - Integration with Bayesian optimization framework
    """
    
    def __init__(
        self,
        parameter_space: ParameterSpace,
        population_size: int = 100,
        max_generations: int = 100,
        crossover_probability: float = 0.9,
        mutation_probability: float = 0.1,
        tournament_size: int = 2,
        constraint_functions: Optional[List[ConstraintFunction]] = None,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize NSGA-II optimizer.
        
        Args:
            parameter_space: Parameter space definition
            population_size: Size of the population
            max_generations: Maximum number of generations
            crossover_probability: Probability of crossover
            mutation_probability: Probability of mutation
            tournament_size: Tournament size for selection
            constraint_functions: List of constraint functions
            random_seed: Random seed for reproducibility
            **kwargs: Additional configuration options
        """
        super().__init__(parameter_space, **kwargs)
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.tournament_size = tournament_size
        self.constraint_functions = constraint_functions or []
        
        # Set random seed
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        # Algorithm components
        self.sorter = FastNonDominatedSorting()
        self.crowding_distance = CrowdingDistance()
        self.selection = TournamentSelection(tournament_size)
        
        # Population and statistics
        self.population: List[Individual] = []
        self.pareto_front: List[Individual] = []
        self.generation = 0
        self.evaluation_count = 0
        
        # Objective function (to be set by user)
        self.objective_function: Optional[Callable] = None
        
        logger.info(f"Initialized NSGA-II with population size {population_size}, "
                   f"max generations {max_generations}")
    
    def set_objective_function(self, objective_function: Callable) -> None:
        """
        Set the objective function for optimization.
        
        Args:
            objective_function: Function that takes parameters dict and returns
                               dict of objective values
        """
        self.objective_function = objective_function
    
    def _initialize_population(self) -> List[Individual]:
        """Initialize random population."""
        population = []
        
        for _ in range(self.population_size):
            # Generate random parameters
            parameters = {}
            for param in self.parameter_space.parameters:
                if param.type == ParameterType.CONTINUOUS:
                    if param.bounds:
                        low, high = param.bounds
                        parameters[param.name] = np.random.uniform(low, high)
                    else:
                        parameters[param.name] = np.random.normal(0, 1)
                elif param.type == ParameterType.CATEGORICAL:
                    if param.categories:
                        parameters[param.name] = np.random.choice(param.categories)
                    else:
                        parameters[param.name] = 0
                elif param.type == ParameterType.INTEGER:
                    if param.bounds:
                        low, high = param.bounds
                        parameters[param.name] = np.random.randint(low, high + 1)
                    else:
                        parameters[param.name] = np.random.randint(0, 10)
            
            individual = Individual(parameters=parameters, generation=0)
            population.append(individual)
        
        return population

    def _evaluate_individual(self, individual: Individual) -> None:
        """
        Evaluate an individual's objectives and constraints.

        Args:
            individual: Individual to evaluate
        """
        if self.objective_function is None:
            raise ValueError("Objective function not set. Use set_objective_function().")

        start_time = time.time()

        try:
            # Evaluate objectives
            objectives = self.objective_function(individual.parameters)
            if isinstance(objectives, dict):
                individual.objectives = objectives
            else:
                # Assume single objective
                individual.objectives = {"objective": objectives}

            individual.objective_values = list(individual.objectives.values())

            # Evaluate constraints
            constraint_violation = 0.0
            for constraint_func in self.constraint_functions:
                violation = constraint_func(individual.parameters)
                individual.constraints[constraint_func.__name__] = violation
                if violation > 0:  # Constraint violated
                    constraint_violation += violation

            individual.constraint_violation = constraint_violation
            individual.is_feasible = constraint_violation <= 1e-6

            individual.evaluation_time = time.time() - start_time
            self.evaluation_count += 1

        except Exception as e:
            logger.error(f"Error evaluating individual: {e}")
            # Set default values for failed evaluation
            individual.objectives = {"objective": float('inf')}
            individual.objective_values = [float('inf')]
            individual.is_feasible = False
            individual.constraint_violation = float('inf')

    def _evaluate_population(self, population: List[Individual]) -> None:
        """
        Evaluate entire population.

        Args:
            population: Population to evaluate
        """
        # Parallel evaluation for better performance
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self._evaluate_individual, ind)
                      for ind in population]

            for future in as_completed(futures):
                try:
                    future.result(timeout=30)  # 30 second timeout per evaluation
                except Exception as e:
                    logger.warning(f"Individual evaluation failed: {e}")

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Simulated Binary Crossover (SBX) for continuous variables.

        Args:
            parent1: First parent
            parent2: Second parent

        Returns:
            Two offspring individuals
        """
        eta_c = 20  # Distribution index for crossover

        child1_params = {}
        child2_params = {}

        for param in self.parameter_space.parameters:
            param_name = param.name

            if param.type == ParameterType.CONTINUOUS:
                if np.random.random() <= self.crossover_probability:
                    # SBX crossover
                    p1_val = parent1.parameters[param_name]
                    p2_val = parent2.parameters[param_name]

                    if abs(p1_val - p2_val) > 1e-14:
                        if p1_val > p2_val:
                            p1_val, p2_val = p2_val, p1_val

                        # Calculate beta
                        rand = np.random.random()
                        if rand <= 0.5:
                            beta = (2 * rand) ** (1.0 / (eta_c + 1))
                        else:
                            beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta_c + 1))

                        # Generate offspring
                        c1_val = 0.5 * ((1 + beta) * p1_val + (1 - beta) * p2_val)
                        c2_val = 0.5 * ((1 - beta) * p1_val + (1 + beta) * p2_val)

                        # Apply bounds
                        if param.bounds:
                            low, high = param.bounds
                            c1_val = np.clip(c1_val, low, high)
                            c2_val = np.clip(c2_val, low, high)

                        child1_params[param_name] = c1_val
                        child2_params[param_name] = c2_val
                    else:
                        # Parents are identical
                        child1_params[param_name] = p1_val
                        child2_params[param_name] = p2_val
                else:
                    # No crossover - copy parents
                    child1_params[param_name] = parent1.parameters[param_name]
                    child2_params[param_name] = parent2.parameters[param_name]

            elif param.type == ParameterType.CATEGORICAL:
                # Uniform crossover for categorical
                if np.random.random() <= 0.5:
                    child1_params[param_name] = parent1.parameters[param_name]
                    child2_params[param_name] = parent2.parameters[param_name]
                else:
                    child1_params[param_name] = parent2.parameters[param_name]
                    child2_params[param_name] = parent1.parameters[param_name]

            elif param.type == ParameterType.INTEGER:
                # Integer crossover (similar to continuous but with rounding)
                if np.random.random() <= self.crossover_probability:
                    p1_val = parent1.parameters[param_name]
                    p2_val = parent2.parameters[param_name]

                    # Simple average with random perturbation
                    avg = (p1_val + p2_val) / 2
                    perturbation = np.random.randint(-1, 2)  # -1, 0, or 1

                    c1_val = int(avg + perturbation)
                    c2_val = int(avg - perturbation)

                    # Apply bounds
                    if param.bounds:
                        low, high = param.bounds
                        c1_val = np.clip(c1_val, low, high)
                        c2_val = np.clip(c2_val, low, high)

                    child1_params[param_name] = c1_val
                    child2_params[param_name] = c2_val
                else:
                    child1_params[param_name] = parent1.parameters[param_name]
                    child2_params[param_name] = parent2.parameters[param_name]

        child1 = Individual(parameters=child1_params, generation=self.generation + 1)
        child2 = Individual(parameters=child2_params, generation=self.generation + 1)

        return child1, child2

    def _mutate(self, individual: Individual) -> Individual:
        """
        Polynomial mutation for continuous variables.

        Args:
            individual: Individual to mutate

        Returns:
            Mutated individual
        """
        eta_m = 20  # Distribution index for mutation

        mutated_params = individual.parameters.copy()

        for param in self.parameter_space.parameters:
            param_name = param.name

            if np.random.random() <= self.mutation_probability:
                if param.type == ParameterType.CONTINUOUS:
                    # Polynomial mutation
                    val = mutated_params[param_name]

                    if param.bounds:
                        low, high = param.bounds
                        delta1 = (val - low) / (high - low)
                        delta2 = (high - val) / (high - low)

                        rand = np.random.random()
                        mut_pow = 1.0 / (eta_m + 1.0)

                        if rand <= 0.5:
                            xy = 1.0 - delta1
                            val_new = val + (high - low) * ((2.0 * rand + (1.0 - 2.0 * rand) *
                                                           (xy ** (eta_m + 1.0))) ** mut_pow - 1.0)
                        else:
                            xy = 1.0 - delta2
                            val_new = val + (high - low) * (1.0 - (2.0 * (1.0 - rand) +
                                                           2.0 * (rand - 0.5) *
                                                           (xy ** (eta_m + 1.0))) ** mut_pow)

                        mutated_params[param_name] = np.clip(val_new, low, high)
                    else:
                        # No bounds - Gaussian mutation
                        mutated_params[param_name] = val + np.random.normal(0, 0.1)

                elif param.type == ParameterType.CATEGORICAL:
                    # Random categorical mutation
                    if param.categories:
                        mutated_params[param_name] = np.random.choice(param.categories)

                elif param.type == ParameterType.INTEGER:
                    # Integer mutation
                    val = mutated_params[param_name]
                    perturbation = np.random.randint(-2, 3)  # -2, -1, 0, 1, 2
                    new_val = val + perturbation

                    if param.bounds:
                        low, high = param.bounds
                        new_val = np.clip(new_val, low, high)

                    mutated_params[param_name] = new_val

        return Individual(parameters=mutated_params, generation=self.generation + 1)

    def optimize(
        self,
        objective_function: Callable,
        max_evaluations: Optional[int] = None,
        convergence_tolerance: float = 1e-6,
        verbose: bool = True
    ) -> OptimizationResult:
        """
        Run NSGA-II optimization.

        Args:
            objective_function: Function to optimize
            max_evaluations: Maximum number of function evaluations
            convergence_tolerance: Convergence tolerance for stopping
            verbose: Whether to print progress

        Returns:
            Optimization result with Pareto front
        """
        self.objective_function = objective_function

        if max_evaluations is None:
            max_evaluations = self.population_size * self.max_generations

        start_time = time.time()

        # Initialize population
        if verbose:
            logger.info("Initializing population...")

        self.population = self._initialize_population()
        self._evaluate_population(self.population)

        # Initial sorting and crowding distance
        fronts = self.sorter.sort(self.population)
        for front in fronts:
            self.crowding_distance.calculate(front)

        self.pareto_front = fronts[0] if fronts else []

        if verbose:
            logger.info(f"Generation 0: {len(self.pareto_front)} solutions in Pareto front")

        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation

            if self.evaluation_count >= max_evaluations:
                if verbose:
                    logger.info(f"Reached maximum evaluations: {max_evaluations}")
                break

            # Selection and reproduction
            parents = self.selection.select(self.population, self.population_size)

            # Create offspring
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                parent1 = parents[i]
                parent2 = parents[i + 1] if i + 1 < len(parents) else parents[0]

                # Crossover
                child1, child2 = self._crossover(parent1, parent2)

                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                offspring.extend([child1, child2])

            # Evaluate offspring
            self._evaluate_population(offspring)

            # Combine parent and offspring populations
            combined_population = self.population + offspring

            # Non-dominated sorting
            fronts = self.sorter.sort(combined_population)

            # Environmental selection
            new_population = []
            front_index = 0

            while (len(new_population) + len(fronts[front_index]) <= self.population_size and
                   front_index < len(fronts)):
                # Calculate crowding distance for this front
                self.crowding_distance.calculate(fronts[front_index])
                new_population.extend(fronts[front_index])
                front_index += 1

            # If we need to partially fill from the next front
            if len(new_population) < self.population_size and front_index < len(fronts):
                remaining_slots = self.population_size - len(new_population)
                last_front = fronts[front_index]

                # Calculate crowding distance for last front
                self.crowding_distance.calculate(last_front)

                # Sort by crowding distance (descending)
                last_front.sort(key=lambda x: x.crowding_distance, reverse=True)

                # Add best individuals from last front
                new_population.extend(last_front[:remaining_slots])

            self.population = new_population
            self.pareto_front = fronts[0] if fronts else []

            if verbose and (generation + 1) % 10 == 0:
                logger.info(f"Generation {generation + 1}: "
                           f"{len(self.pareto_front)} solutions in Pareto front, "
                           f"{self.evaluation_count} evaluations")

            # Check convergence (simplified)
            if generation > 10 and self._check_convergence(convergence_tolerance):
                if verbose:
                    logger.info(f"Converged at generation {generation + 1}")
                break

        optimization_time = time.time() - start_time

        # Create result
        result = OptimizationResult(
            best_parameters=[ind.parameters for ind in self.pareto_front],
            best_objectives=[ind.objectives for ind in self.pareto_front],
            pareto_front=[ind.to_experiment_point() for ind in self.pareto_front],
            optimization_history=[],  # Could be populated with generation statistics
            convergence_data={
                'generations': self.generation + 1,
                'evaluations': self.evaluation_count,
                'optimization_time': optimization_time,
                'pareto_front_size': len(self.pareto_front),
            },
            metadata={
                'algorithm': 'NSGA-II',
                'population_size': self.population_size,
                'crossover_probability': self.crossover_probability,
                'mutation_probability': self.mutation_probability,
            }
        )

        if verbose:
            logger.info(f"Optimization completed in {optimization_time:.2f} seconds")
            logger.info(f"Final Pareto front size: {len(self.pareto_front)}")

        return result

    def _check_convergence(self, tolerance: float) -> bool:
        """
        Check if the algorithm has converged.

        Simple convergence check based on Pareto front stability.
        Could be enhanced with more sophisticated metrics.
        """
        # For now, just return False to run full generations
        # In practice, you might check:
        # - Hypervolume improvement
        # - Pareto front movement
        # - Objective value changes
        return False

    def get_pareto_front(self) -> List[ExperimentPoint]:
        """Get current Pareto front as ExperimentPoint objects."""
        return [ind.to_experiment_point() for ind in self.pareto_front]

    def get_population_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current population."""
        if not self.population:
            return {}

        # Calculate statistics
        feasible_count = sum(1 for ind in self.population if ind.is_feasible)
        avg_constraint_violation = np.mean([ind.constraint_violation for ind in self.population])

        # Objective statistics
        objective_stats = {}
        if self.population[0].objective_values:
            num_objectives = len(self.population[0].objective_values)
            for i in range(num_objectives):
                obj_values = [ind.objective_values[i] for ind in self.population
                             if ind.is_feasible]
                if obj_values:
                    objective_stats[f'objective_{i}'] = {
                        'mean': np.mean(obj_values),
                        'std': np.std(obj_values),
                        'min': np.min(obj_values),
                        'max': np.max(obj_values),
                    }

        return {
            'generation': self.generation,
            'population_size': len(self.population),
            'feasible_count': feasible_count,
            'feasible_ratio': feasible_count / len(self.population),
            'avg_constraint_violation': avg_constraint_violation,
            'pareto_front_size': len(self.pareto_front),
            'evaluation_count': self.evaluation_count,
            'objective_statistics': objective_stats,
        }
