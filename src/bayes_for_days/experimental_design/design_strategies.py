"""
Design strategies for experimental design.

This module implements various experimental design strategies:
- D-optimal design for parameter estimation
- Latin Hypercube Sampling with optimization
- Factorial designs (full/fractional)
- Response Surface Methodology (RSM)
- Space-filling designs with maximin criteria

Based on:
- Atkinson et al. (2007) "Optimum Experimental Designs"
- Montgomery (2017) "Design and Analysis of Experiments"
- Jones & Johnson (2009) "Design and analysis for the Gaussian process model"
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import det, inv
from sklearn.preprocessing import StandardScaler
import itertools
from abc import ABC, abstractmethod

from bayes_for_days.experimental_design.variables import ExperimentalVariable, VariableGroup

logger = logging.getLogger(__name__)


class DesignStrategy(ABC):
    """
    Abstract base class for experimental design strategies.
    
    Defines the interface that all design strategies must implement.
    """
    
    def __init__(self, variables: List[ExperimentalVariable], **kwargs):
        """
        Initialize design strategy.
        
        Args:
            variables: List of experimental variables
            **kwargs: Strategy-specific parameters
        """
        self.variables = variables
        self.variable_names = [var.name for var in variables]
        self.n_variables = len(variables)
        self.config = kwargs
        
        # Validate variables
        self._validate_variables()
    
    def _validate_variables(self):
        """Validate that variables are suitable for this design strategy."""
        if not self.variables:
            raise ValueError("No variables provided for design strategy")
    
    @abstractmethod
    def generate_design(self, n_experiments: int, **kwargs) -> np.ndarray:
        """
        Generate experimental design.
        
        Args:
            n_experiments: Number of experiments to generate
            **kwargs: Additional parameters
            
        Returns:
            Design matrix (n_experiments x n_variables)
        """
        pass
    
    def get_design_info(self) -> Dict[str, Any]:
        """Get information about the design strategy."""
        return {
            'strategy_type': self.__class__.__name__,
            'n_variables': self.n_variables,
            'variable_names': self.variable_names,
            'config': self.config,
        }


class DOptimalDesign(DesignStrategy):
    """
    D-optimal experimental design for parameter estimation.
    
    Maximizes the determinant of the information matrix (Fisher Information Matrix)
    to minimize the volume of the confidence ellipsoid for parameter estimates.
    
    Features:
    - Candidate point generation and selection
    - Exchange algorithms for design optimization
    - Support for linear and nonlinear models
    - Efficiency criteria evaluation
    """
    
    def __init__(
        self,
        variables: List[ExperimentalVariable],
        model_matrix_func: Optional[Callable] = None,
        candidate_size: int = 1000,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        **kwargs
    ):
        """
        Initialize D-optimal design.
        
        Args:
            variables: List of experimental variables
            model_matrix_func: Function to compute model matrix from design points
            candidate_size: Size of candidate point set
            max_iterations: Maximum iterations for optimization
            tolerance: Convergence tolerance
            **kwargs: Additional parameters
        """
        super().__init__(variables, **kwargs)
        
        self.model_matrix_func = model_matrix_func or self._default_model_matrix
        self.candidate_size = candidate_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        
        # Generate candidate points
        self.candidate_points = self._generate_candidate_points()
        
        logger.info(f"Initialized D-optimal design with {len(self.candidate_points)} candidate points")
    
    def _default_model_matrix(self, design_points: np.ndarray) -> np.ndarray:
        """
        Default model matrix for linear model with interactions.
        
        Args:
            design_points: Design points (n_points x n_variables)
            
        Returns:
            Model matrix including intercept, main effects, and interactions
        """
        n_points, n_vars = design_points.shape
        
        # Start with intercept
        X = [np.ones(n_points)]
        
        # Main effects
        for i in range(n_vars):
            X.append(design_points[:, i])
        
        # Two-way interactions
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                X.append(design_points[:, i] * design_points[:, j])
        
        # Quadratic terms (for RSM)
        if self.config.get('include_quadratic', False):
            for i in range(n_vars):
                X.append(design_points[:, i] ** 2)
        
        return np.column_stack(X)
    
    def _generate_candidate_points(self) -> np.ndarray:
        """Generate candidate points for D-optimal design selection."""
        candidate_points = []
        
        for _ in range(self.candidate_size):
            point = []
            for var in self.variables:
                if var.variable_type.value in ['continuous', 'integer']:
                    if var.bounds:
                        low, high = var.bounds
                        if var.variable_type.value == 'integer':
                            value = np.random.randint(low, high + 1)
                        else:
                            value = np.random.uniform(low, high)
                    else:
                        value = np.random.normal(0, 1)
                    point.append(value)
                elif var.variable_type.value == 'categorical':
                    if var.categories:
                        # Encode categorically as integers
                        value = np.random.randint(0, len(var.categories))
                    else:
                        value = 0
                    point.append(value)
                else:
                    point.append(0.0)
            
            candidate_points.append(point)
        
        return np.array(candidate_points)
    
    def generate_design(self, n_experiments: int, **kwargs) -> np.ndarray:
        """
        Generate D-optimal experimental design.
        
        Args:
            n_experiments: Number of experiments to generate
            **kwargs: Additional parameters
            
        Returns:
            D-optimal design matrix
        """
        if n_experiments > len(self.candidate_points):
            logger.warning(f"Requested {n_experiments} experiments but only have "
                          f"{len(self.candidate_points)} candidates. Using all candidates.")
            n_experiments = len(self.candidate_points)
        
        # Use exchange algorithm to find D-optimal design
        design_indices = self._exchange_algorithm(n_experiments)
        design_points = self.candidate_points[design_indices]
        
        # Evaluate design efficiency
        efficiency = self._evaluate_d_efficiency(design_points)
        
        logger.info(f"Generated D-optimal design with {n_experiments} experiments, "
                   f"D-efficiency: {efficiency:.4f}")
        
        return design_points
    
    def _exchange_algorithm(self, n_experiments: int) -> np.ndarray:
        """
        Exchange algorithm for D-optimal design selection.
        
        Args:
            n_experiments: Number of experiments to select
            
        Returns:
            Indices of selected candidate points
        """
        # Start with random selection
        current_indices = np.random.choice(
            len(self.candidate_points), 
            size=n_experiments, 
            replace=False
        )
        
        current_design = self.candidate_points[current_indices]
        current_det = self._compute_determinant(current_design)
        
        # Exchange algorithm
        for iteration in range(self.max_iterations):
            improved = False
            
            # Try to exchange each point in the design
            for i in range(n_experiments):
                best_det = current_det
                best_candidate = None
                
                # Try all candidate points not in current design
                available_indices = np.setdiff1d(
                    np.arange(len(self.candidate_points)), 
                    current_indices
                )
                
                for candidate_idx in available_indices:
                    # Create trial design by replacing point i
                    trial_indices = current_indices.copy()
                    trial_indices[i] = candidate_idx
                    trial_design = self.candidate_points[trial_indices]
                    
                    # Compute determinant
                    trial_det = self._compute_determinant(trial_design)
                    
                    if trial_det > best_det:
                        best_det = trial_det
                        best_candidate = candidate_idx
                
                # Make exchange if improvement found
                if best_candidate is not None:
                    current_indices[i] = best_candidate
                    current_det = best_det
                    improved = True
            
            # Check convergence
            if not improved:
                logger.debug(f"Exchange algorithm converged at iteration {iteration}")
                break
        
        return current_indices
    
    def _compute_determinant(self, design_points: np.ndarray) -> float:
        """
        Compute determinant of information matrix.
        
        Args:
            design_points: Design points
            
        Returns:
            Determinant value (0 if singular)
        """
        try:
            X = self.model_matrix_func(design_points)
            
            # Information matrix: X'X
            info_matrix = X.T @ X
            
            # Add small regularization to avoid singularity
            info_matrix += np.eye(info_matrix.shape[0]) * 1e-8
            
            # Compute determinant
            det_value = det(info_matrix)
            
            return max(0.0, det_value)
            
        except (np.linalg.LinAlgError, ValueError):
            return 0.0
    
    def _evaluate_d_efficiency(self, design_points: np.ndarray) -> float:
        """
        Evaluate D-efficiency of design.
        
        Args:
            design_points: Design points
            
        Returns:
            D-efficiency (0-1, where 1 is optimal)
        """
        try:
            X = self.model_matrix_func(design_points)
            n_points, n_params = X.shape
            
            if n_points < n_params:
                return 0.0
            
            # Information matrix
            info_matrix = X.T @ X
            det_value = det(info_matrix)
            
            # D-efficiency relative to saturated design
            max_det = n_points ** n_params  # Theoretical maximum
            efficiency = (det_value / max_det) ** (1.0 / n_params)
            
            return min(1.0, max(0.0, efficiency))
            
        except (np.linalg.LinAlgError, ValueError):
            return 0.0
    
    def evaluate_design_criteria(self, design_points: np.ndarray) -> Dict[str, float]:
        """
        Evaluate multiple design criteria.
        
        Args:
            design_points: Design points to evaluate
            
        Returns:
            Dictionary of design criteria values
        """
        try:
            X = self.model_matrix_func(design_points)
            info_matrix = X.T @ X
            
            # Add regularization
            info_matrix += np.eye(info_matrix.shape[0]) * 1e-8
            
            # D-criterion (determinant)
            d_criterion = det(info_matrix)
            
            # A-criterion (trace of inverse)
            try:
                a_criterion = np.trace(inv(info_matrix))
            except np.linalg.LinAlgError:
                a_criterion = float('inf')
            
            # E-criterion (smallest eigenvalue)
            eigenvals = np.linalg.eigvals(info_matrix)
            e_criterion = np.min(eigenvals)
            
            # G-criterion (maximum prediction variance)
            try:
                inv_info = inv(info_matrix)
                pred_vars = np.diag(X @ inv_info @ X.T)
                g_criterion = np.max(pred_vars)
            except np.linalg.LinAlgError:
                g_criterion = float('inf')
            
            return {
                'D_criterion': d_criterion,
                'A_criterion': a_criterion,
                'E_criterion': e_criterion,
                'G_criterion': g_criterion,
                'D_efficiency': self._evaluate_d_efficiency(design_points),
            }
            
        except Exception as e:
            logger.warning(f"Error evaluating design criteria: {e}")
            return {
                'D_criterion': 0.0,
                'A_criterion': float('inf'),
                'E_criterion': 0.0,
                'G_criterion': float('inf'),
                'D_efficiency': 0.0,
            }


class LatinHypercubeSampling(DesignStrategy):
    """
    Latin Hypercube Sampling with optimization.
    
    Generates space-filling designs with good projection properties
    and optional optimization for improved space-filling criteria.
    """
    
    def __init__(
        self,
        variables: List[ExperimentalVariable],
        criterion: str = "maximin",
        optimization: bool = True,
        max_iterations: int = 1000,
        **kwargs
    ):
        """
        Initialize Latin Hypercube Sampling.
        
        Args:
            variables: List of experimental variables
            criterion: Space-filling criterion ('maximin', 'correlation')
            optimization: Whether to optimize the design
            max_iterations: Maximum optimization iterations
            **kwargs: Additional parameters
        """
        super().__init__(variables, **kwargs)
        
        self.criterion = criterion
        self.optimization = optimization
        self.max_iterations = max_iterations
        
        logger.info(f"Initialized LHS with criterion: {criterion}")
    
    def generate_design(self, n_experiments: int, **kwargs) -> np.ndarray:
        """
        Generate Latin Hypercube design.
        
        Args:
            n_experiments: Number of experiments to generate
            **kwargs: Additional parameters
            
        Returns:
            Latin Hypercube design matrix
        """
        # Generate initial LHS design
        design = self._generate_basic_lhs(n_experiments)
        
        # Optimize if requested
        if self.optimization:
            design = self._optimize_lhs(design)
        
        # Scale to variable bounds
        design = self._scale_to_bounds(design)
        
        logger.info(f"Generated LHS design with {n_experiments} experiments")
        
        return design
    
    def _generate_basic_lhs(self, n_experiments: int) -> np.ndarray:
        """Generate basic Latin Hypercube design."""
        design = np.zeros((n_experiments, self.n_variables))
        
        for j in range(self.n_variables):
            # Generate random permutation
            perm = np.random.permutation(n_experiments)
            
            # Generate uniform random values within each interval
            uniform_vals = np.random.uniform(0, 1, n_experiments)
            
            # Create LHS values
            lhs_vals = (perm + uniform_vals) / n_experiments
            
            design[:, j] = lhs_vals
        
        return design
    
    def _optimize_lhs(self, design: np.ndarray) -> np.ndarray:
        """Optimize LHS design using specified criterion."""
        if self.criterion == "maximin":
            return self._optimize_maximin(design)
        elif self.criterion == "correlation":
            return self._optimize_correlation(design)
        else:
            logger.warning(f"Unknown criterion: {self.criterion}")
            return design
    
    def _optimize_maximin(self, design: np.ndarray) -> np.ndarray:
        """Optimize design using maximin criterion."""
        best_design = design.copy()
        best_criterion = self._compute_maximin_criterion(design)
        
        for iteration in range(self.max_iterations):
            # Try random perturbations
            trial_design = self._perturb_design(best_design)
            trial_criterion = self._compute_maximin_criterion(trial_design)
            
            if trial_criterion > best_criterion:
                best_design = trial_design
                best_criterion = trial_criterion
        
        return best_design
    
    def _compute_maximin_criterion(self, design: np.ndarray) -> float:
        """Compute maximin criterion (minimum pairwise distance)."""
        distances = pdist(design)
        return np.min(distances) if len(distances) > 0 else 0.0
    
    def _optimize_correlation(self, design: np.ndarray) -> np.ndarray:
        """Optimize design to minimize correlations between variables."""
        best_design = design.copy()
        best_criterion = self._compute_correlation_criterion(design)
        
        for iteration in range(self.max_iterations):
            trial_design = self._perturb_design(best_design)
            trial_criterion = self._compute_correlation_criterion(trial_design)
            
            if trial_criterion < best_criterion:  # Minimize correlation
                best_design = trial_design
                best_criterion = trial_criterion
        
        return best_design
    
    def _compute_correlation_criterion(self, design: np.ndarray) -> float:
        """Compute correlation criterion (sum of absolute correlations)."""
        corr_matrix = np.corrcoef(design.T)
        
        # Sum of absolute off-diagonal correlations
        n_vars = corr_matrix.shape[0]
        total_corr = 0.0
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                total_corr += abs(corr_matrix[i, j])
        
        return total_corr
    
    def _perturb_design(self, design: np.ndarray) -> np.ndarray:
        """Apply small random perturbation to design."""
        perturbed = design.copy()
        
        # Randomly select points and variables to perturb
        n_points, n_vars = design.shape
        n_perturbations = max(1, n_points // 10)
        
        for _ in range(n_perturbations):
            i = np.random.randint(0, n_points)
            j = np.random.randint(0, n_vars)
            
            # Small random perturbation
            perturbation = np.random.normal(0, 0.01)
            perturbed[i, j] = np.clip(perturbed[i, j] + perturbation, 0, 1)
        
        return perturbed
    
    def _scale_to_bounds(self, design: np.ndarray) -> np.ndarray:
        """Scale design from [0,1] to variable bounds."""
        scaled_design = design.copy()
        
        for j, var in enumerate(self.variables):
            if var.bounds:
                low, high = var.bounds
                scaled_design[:, j] = low + design[:, j] * (high - low)
            elif var.variable_type.value == 'categorical':
                # Map to categorical indices
                if var.categories:
                    n_categories = len(var.categories)
                    scaled_design[:, j] = np.floor(design[:, j] * n_categories)
                    scaled_design[:, j] = np.clip(scaled_design[:, j], 0, n_categories - 1)
        
        return scaled_design


class FactorialDesign(DesignStrategy):
    """
    Factorial experimental designs (full and fractional).

    Generates factorial designs for studying main effects and interactions:
    - Full factorial designs (all combinations)
    - Fractional factorial designs (subset with confounding)
    - Central composite designs for RSM
    - Box-Behnken designs
    """

    def __init__(
        self,
        variables: List[ExperimentalVariable],
        design_type: str = "full",
        resolution: Optional[int] = None,
        center_points: int = 1,
        **kwargs
    ):
        """
        Initialize factorial design.

        Args:
            variables: List of experimental variables
            design_type: Type of design ('full', 'fractional', 'ccd', 'box_behnken')
            resolution: Resolution for fractional factorial (III, IV, V)
            center_points: Number of center points to add
            **kwargs: Additional parameters
        """
        super().__init__(variables, **kwargs)

        self.design_type = design_type
        self.resolution = resolution
        self.center_points = center_points

        # Validate variables for factorial design
        self._validate_factorial_variables()

        logger.info(f"Initialized {design_type} factorial design")

    def _validate_factorial_variables(self):
        """Validate variables are suitable for factorial design."""
        for var in self.variables:
            if var.variable_type.value not in ['continuous', 'categorical', 'ordinal']:
                logger.warning(f"Variable {var.name} may not be suitable for factorial design")

    def generate_design(self, n_experiments: Optional[int] = None, **kwargs) -> np.ndarray:
        """
        Generate factorial design.

        Args:
            n_experiments: Number of experiments (ignored for full factorial)
            **kwargs: Additional parameters

        Returns:
            Factorial design matrix
        """
        if self.design_type == "full":
            design = self._generate_full_factorial()
        elif self.design_type == "fractional":
            design = self._generate_fractional_factorial(n_experiments)
        elif self.design_type == "ccd":
            design = self._generate_central_composite()
        elif self.design_type == "box_behnken":
            design = self._generate_box_behnken()
        else:
            raise ValueError(f"Unknown design type: {self.design_type}")

        # Add center points
        if self.center_points > 0:
            design = self._add_center_points(design)

        # Scale to variable bounds
        design = self._scale_to_bounds(design)

        logger.info(f"Generated {self.design_type} factorial design with {len(design)} experiments")

        return design

    def _generate_full_factorial(self) -> np.ndarray:
        """Generate full factorial design."""
        # Determine levels for each variable
        levels_per_var = []
        for var in self.variables:
            if var.variable_type.value == 'categorical':
                n_levels = len(var.categories) if var.categories else 2
            else:
                n_levels = 2  # Default to 2-level for continuous
            levels_per_var.append(n_levels)

        # Generate all combinations
        level_combinations = list(itertools.product(*[range(n) for n in levels_per_var]))

        # Convert to design matrix
        design = np.array(level_combinations, dtype=float)

        # Convert to [-1, 1] coding for continuous variables
        for j, var in enumerate(self.variables):
            if var.variable_type.value in ['continuous', 'integer']:
                max_level = levels_per_var[j] - 1
                if max_level > 0:
                    design[:, j] = 2 * (design[:, j] / max_level) - 1

        return design

    def _generate_fractional_factorial(self, n_experiments: Optional[int]) -> np.ndarray:
        """Generate fractional factorial design."""
        # For simplicity, generate 2^(k-p) fractional factorial
        k = self.n_variables

        if n_experiments is None:
            # Default to half-fraction
            p = 1
            n_experiments = 2 ** (k - p)
        else:
            # Determine fraction size
            p = k - int(np.log2(n_experiments))
            p = max(0, p)

        # Generate base design (first k-p factors)
        base_factors = k - p
        base_design = self._generate_full_factorial_subset(base_factors)

        # Generate additional factors using generators
        if p > 0:
            additional_factors = self._generate_additional_factors(base_design, p)
            design = np.column_stack([base_design, additional_factors])
        else:
            design = base_design

        return design

    def _generate_full_factorial_subset(self, n_factors: int) -> np.ndarray:
        """Generate full factorial for subset of factors."""
        n_runs = 2 ** n_factors
        design = np.zeros((n_runs, n_factors))

        for j in range(n_factors):
            period = 2 ** (n_factors - j - 1)
            for i in range(n_runs):
                design[i, j] = 1 if (i // period) % 2 == 0 else -1

        return design

    def _generate_additional_factors(self, base_design: np.ndarray, p: int) -> np.ndarray:
        """Generate additional factors for fractional factorial."""
        n_runs, base_factors = base_design.shape
        additional = np.zeros((n_runs, p))

        # Simple generators (can be improved)
        for j in range(p):
            if j == 0 and base_factors >= 2:
                # First additional factor = interaction of first two base factors
                additional[:, j] = base_design[:, 0] * base_design[:, 1]
            elif j == 1 and base_factors >= 3:
                # Second additional factor = interaction of first three base factors
                additional[:, j] = base_design[:, 0] * base_design[:, 1] * base_design[:, 2]
            else:
                # Default to interaction of all base factors
                additional[:, j] = np.prod(base_design, axis=1)

        return additional

    def _generate_central_composite(self) -> np.ndarray:
        """Generate Central Composite Design (CCD)."""
        k = self.n_variables

        # Factorial portion (2^k or 2^(k-p))
        factorial_design = self._generate_full_factorial()

        # Axial points
        alpha = (2 ** k) ** 0.25  # Rotatability condition
        axial_points = np.zeros((2 * k, k))

        for i in range(k):
            axial_points[2 * i, i] = alpha
            axial_points[2 * i + 1, i] = -alpha

        # Combine factorial and axial points
        design = np.vstack([factorial_design, axial_points])

        return design

    def _generate_box_behnken(self) -> np.ndarray:
        """Generate Box-Behnken design."""
        k = self.n_variables

        if k < 3:
            raise ValueError("Box-Behnken design requires at least 3 variables")

        # Generate all combinations of 2 factors at ±1, others at 0
        design_points = []

        for i in range(k):
            for j in range(i + 1, k):
                # Four combinations for factors i and j
                for level_i in [-1, 1]:
                    for level_j in [-1, 1]:
                        point = np.zeros(k)
                        point[i] = level_i
                        point[j] = level_j
                        design_points.append(point)

        return np.array(design_points)

    def _add_center_points(self, design: np.ndarray) -> np.ndarray:
        """Add center points to design."""
        center_point = np.zeros((1, self.n_variables))
        center_points = np.repeat(center_point, self.center_points, axis=0)

        return np.vstack([design, center_points])

    def _scale_to_bounds(self, design: np.ndarray) -> np.ndarray:
        """Scale design from coded levels to actual variable bounds."""
        scaled_design = design.copy()

        for j, var in enumerate(self.variables):
            if var.variable_type.value in ['continuous', 'integer']:
                if var.bounds:
                    low, high = var.bounds
                    # Scale from [-1, 1] to [low, high]
                    scaled_design[:, j] = low + (design[:, j] + 1) * (high - low) / 2
                elif var.baseline_value is not None:
                    # Use baseline ± some range
                    range_val = abs(var.baseline_value) * 0.2  # 20% range
                    scaled_design[:, j] = var.baseline_value + design[:, j] * range_val
            elif var.variable_type.value == 'categorical':
                if var.categories:
                    # Map coded levels to category indices
                    n_categories = len(var.categories)
                    # Map from [-1, 1] to [0, n_categories-1]
                    scaled_design[:, j] = np.round(
                        (design[:, j] + 1) * (n_categories - 1) / 2
                    ).astype(int)
                    scaled_design[:, j] = np.clip(scaled_design[:, j], 0, n_categories - 1)

        return scaled_design


class ResponseSurfaceDesign(DesignStrategy):
    """
    Response Surface Methodology (RSM) designs.

    Specialized designs for fitting second-order polynomial models
    and optimizing responses.
    """

    def __init__(
        self,
        variables: List[ExperimentalVariable],
        design_type: str = "ccd",
        alpha: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize RSM design.

        Args:
            variables: List of experimental variables
            design_type: RSM design type ('ccd', 'box_behnken', 'doehlert')
            alpha: Axial distance for CCD (None for automatic)
            **kwargs: Additional parameters
        """
        super().__init__(variables, **kwargs)

        self.design_type = design_type
        self.alpha = alpha

        logger.info(f"Initialized RSM design: {design_type}")

    def generate_design(self, n_experiments: Optional[int] = None, **kwargs) -> np.ndarray:
        """Generate RSM design."""
        if self.design_type == "ccd":
            factorial_design = FactorialDesign(self.variables, design_type="ccd")
            design = factorial_design.generate_design()
        elif self.design_type == "box_behnken":
            factorial_design = FactorialDesign(self.variables, design_type="box_behnken")
            design = factorial_design.generate_design()
        else:
            raise ValueError(f"Unknown RSM design type: {self.design_type}")

        return design
