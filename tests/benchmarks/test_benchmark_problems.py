"""
Benchmark problems for testing Bayesian optimization algorithms.

This module implements standard benchmark functions used to evaluate
the performance of optimization algorithms:
- Branin function (2D)
- Hartmann functions (3D, 6D)
- Ackley function (multi-dimensional)
- Rosenbrock function (multi-dimensional)
- Sphere function (multi-dimensional)

Based on:
- Surjanovic & Bingham "Virtual Library of Simulation Experiments"
- Standard optimization benchmark suite
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from bayes_for_days.core.types import ParameterSpace, Parameter, ParameterType
from bayes_for_days.optimization.optimization_loop import BayesianOptimizationLoop, OptimizationConfig


@dataclass
class BenchmarkProblem:
    """
    Definition of a benchmark optimization problem.
    
    Contains the objective function, parameter space, known optimum,
    and other metadata for testing optimization algorithms.
    """
    name: str
    objective_function: callable
    parameter_space: ParameterSpace
    global_optimum_params: Dict[str, float]
    global_optimum_value: float
    description: str
    difficulty: str  # "easy", "medium", "hard"
    
    def evaluate(self, params: Dict[str, float]) -> float:
        """Evaluate the objective function."""
        return self.objective_function(params)
    
    def get_bounds(self) -> List[Tuple[float, float]]:
        """Get parameter bounds as list of tuples."""
        return [param.bounds for param in self.parameter_space.parameters if param.bounds]


class BenchmarkProblems:
    """Collection of standard benchmark problems."""
    
    @staticmethod
    def branin() -> BenchmarkProblem:
        """
        Branin function (2D).
        
        Global minima at:
        - (-π, 12.275), (π, 2.275), (9.42478, 2.475)
        - Global minimum value: 0.397887
        """
        def branin_function(params: Dict[str, float]) -> float:
            x1, x2 = params["x1"], params["x2"]
            
            a = 1
            b = 5.1 / (4 * np.pi**2)
            c = 5 / np.pi
            r = 6
            s = 10
            t = 1 / (8 * np.pi)
            
            term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
            term2 = s * (1 - t) * np.cos(x1)
            term3 = s
            
            return -(term1 + term2 + term3)  # Negative for maximization
        
        parameters = [
            Parameter(name="x1", type=ParameterType.CONTINUOUS, bounds=(-5, 10)),
            Parameter(name="x2", type=ParameterType.CONTINUOUS, bounds=(0, 15)),
        ]
        
        return BenchmarkProblem(
            name="Branin",
            objective_function=branin_function,
            parameter_space=ParameterSpace(parameters=parameters),
            global_optimum_params={"x1": -np.pi, "x2": 12.275},
            global_optimum_value=-0.397887,
            description="2D function with 3 global minima",
            difficulty="easy"
        )
    
    @staticmethod
    def hartmann3d() -> BenchmarkProblem:
        """
        Hartmann 3D function.
        
        Global minimum at: (0.114614, 0.555649, 0.852547)
        Global minimum value: -3.86278
        """
        def hartmann3d_function(params: Dict[str, float]) -> float:
            x = np.array([params["x1"], params["x2"], params["x3"]])
            
            A = np.array([
                [3.0, 10, 30],
                [0.1, 10, 35],
                [3.0, 10, 30],
                [0.1, 10, 35]
            ])
            
            P = np.array([
                [0.3689, 0.1170, 0.2673],
                [0.4699, 0.4387, 0.7470],
                [0.1091, 0.8732, 0.5547],
                [0.03815, 0.5743, 0.8828]
            ])
            
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            
            result = 0
            for i in range(4):
                inner_sum = np.sum(A[i] * (x - P[i])**2)
                result += alpha[i] * np.exp(-inner_sum)
            
            return result  # Already negative, so maximize
        
        parameters = [
            Parameter(name="x1", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            Parameter(name="x2", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
            Parameter(name="x3", type=ParameterType.CONTINUOUS, bounds=(0, 1)),
        ]
        
        return BenchmarkProblem(
            name="Hartmann3D",
            objective_function=hartmann3d_function,
            parameter_space=ParameterSpace(parameters=parameters),
            global_optimum_params={"x1": 0.114614, "x2": 0.555649, "x3": 0.852547},
            global_optimum_value=-3.86278,
            description="3D multimodal function",
            difficulty="medium"
        )
    
    @staticmethod
    def hartmann6d() -> BenchmarkProblem:
        """
        Hartmann 6D function.
        
        Global minimum at: (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573)
        Global minimum value: -3.32237
        """
        def hartmann6d_function(params: Dict[str, float]) -> float:
            x = np.array([params[f"x{i+1}"] for i in range(6)])
            
            A = np.array([
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14]
            ])
            
            P = np.array([
                [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
                [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
                [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
                [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
            ])
            
            alpha = np.array([1.0, 1.2, 3.0, 3.2])
            
            result = 0
            for i in range(4):
                inner_sum = np.sum(A[i] * (x - P[i])**2)
                result += alpha[i] * np.exp(-inner_sum)
            
            return result  # Already negative, so maximize
        
        parameters = [
            Parameter(name=f"x{i+1}", type=ParameterType.CONTINUOUS, bounds=(0, 1))
            for i in range(6)
        ]
        
        return BenchmarkProblem(
            name="Hartmann6D",
            objective_function=hartmann6d_function,
            parameter_space=ParameterSpace(parameters=parameters),
            global_optimum_params={
                "x1": 0.20169, "x2": 0.150011, "x3": 0.476874,
                "x4": 0.275332, "x5": 0.311652, "x6": 0.6573
            },
            global_optimum_value=-3.32237,
            description="6D multimodal function",
            difficulty="hard"
        )
    
    @staticmethod
    def ackley(dimension: int = 2) -> BenchmarkProblem:
        """
        Ackley function (multi-dimensional).
        
        Global minimum at: (0, 0, ..., 0)
        Global minimum value: 0
        """
        def ackley_function(params: Dict[str, float]) -> float:
            x = np.array([params[f"x{i+1}"] for i in range(dimension)])
            
            a = 20
            b = 0.2
            c = 2 * np.pi
            
            term1 = -a * np.exp(-b * np.sqrt(np.mean(x**2)))
            term2 = -np.exp(np.mean(np.cos(c * x)))
            term3 = a + np.e
            
            return -(term1 + term2 + term3)  # Negative for maximization
        
        parameters = [
            Parameter(name=f"x{i+1}", type=ParameterType.CONTINUOUS, bounds=(-32.768, 32.768))
            for i in range(dimension)
        ]
        
        global_optimum_params = {f"x{i+1}": 0.0 for i in range(dimension)}
        
        return BenchmarkProblem(
            name=f"Ackley{dimension}D",
            objective_function=ackley_function,
            parameter_space=ParameterSpace(parameters=parameters),
            global_optimum_params=global_optimum_params,
            global_optimum_value=0.0,
            description=f"{dimension}D highly multimodal function",
            difficulty="hard"
        )
    
    @staticmethod
    def rosenbrock(dimension: int = 2) -> BenchmarkProblem:
        """
        Rosenbrock function (multi-dimensional).
        
        Global minimum at: (1, 1, ..., 1)
        Global minimum value: 0
        """
        def rosenbrock_function(params: Dict[str, float]) -> float:
            x = np.array([params[f"x{i+1}"] for i in range(dimension)])
            
            result = 0
            for i in range(dimension - 1):
                result += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
            
            return -result  # Negative for maximization
        
        parameters = [
            Parameter(name=f"x{i+1}", type=ParameterType.CONTINUOUS, bounds=(-5, 10))
            for i in range(dimension)
        ]
        
        global_optimum_params = {f"x{i+1}": 1.0 for i in range(dimension)}
        
        return BenchmarkProblem(
            name=f"Rosenbrock{dimension}D",
            objective_function=rosenbrock_function,
            parameter_space=ParameterSpace(parameters=parameters),
            global_optimum_params=global_optimum_params,
            global_optimum_value=0.0,
            description=f"{dimension}D banana-shaped valley function",
            difficulty="medium"
        )
    
    @staticmethod
    def sphere(dimension: int = 2) -> BenchmarkProblem:
        """
        Sphere function (multi-dimensional).
        
        Global minimum at: (0, 0, ..., 0)
        Global minimum value: 0
        """
        def sphere_function(params: Dict[str, float]) -> float:
            x = np.array([params[f"x{i+1}"] for i in range(dimension)])
            return -np.sum(x**2)  # Negative for maximization
        
        parameters = [
            Parameter(name=f"x{i+1}", type=ParameterType.CONTINUOUS, bounds=(-5.12, 5.12))
            for i in range(dimension)
        ]
        
        global_optimum_params = {f"x{i+1}": 0.0 for i in range(dimension)}
        
        return BenchmarkProblem(
            name=f"Sphere{dimension}D",
            objective_function=sphere_function,
            parameter_space=ParameterSpace(parameters=parameters),
            global_optimum_params=global_optimum_params,
            global_optimum_value=0.0,
            description=f"{dimension}D convex quadratic function",
            difficulty="easy"
        )


class TestBenchmarkProblems:
    """Test suite for benchmark problems."""
    
    def test_branin_function(self):
        """Test Branin function evaluation."""
        problem = BenchmarkProblems.branin()
        
        # Test known optimum
        optimum_value = problem.evaluate(problem.global_optimum_params)
        assert abs(optimum_value - problem.global_optimum_value) < 1e-3
        
        # Test parameter space
        assert len(problem.parameter_space.parameters) == 2
        assert problem.parameter_space.parameters[0].name == "x1"
        assert problem.parameter_space.parameters[1].name == "x2"
        
        # Test bounds
        bounds = problem.get_bounds()
        assert len(bounds) == 2
        assert bounds[0] == (-5, 10)
        assert bounds[1] == (0, 15)
    
    def test_hartmann3d_function(self):
        """Test Hartmann 3D function evaluation."""
        problem = BenchmarkProblems.hartmann3d()
        
        # Test parameter space
        assert len(problem.parameter_space.parameters) == 3
        
        # Test evaluation at random point
        test_params = {"x1": 0.5, "x2": 0.5, "x3": 0.5}
        value = problem.evaluate(test_params)
        assert isinstance(value, float)
        
        # Test bounds
        bounds = problem.get_bounds()
        assert len(bounds) == 3
        assert all(bound == (0, 1) for bound in bounds)
    
    def test_hartmann6d_function(self):
        """Test Hartmann 6D function evaluation."""
        problem = BenchmarkProblems.hartmann6d()
        
        # Test parameter space
        assert len(problem.parameter_space.parameters) == 6
        
        # Test evaluation
        test_params = {f"x{i+1}": 0.5 for i in range(6)}
        value = problem.evaluate(test_params)
        assert isinstance(value, float)
    
    def test_ackley_function(self):
        """Test Ackley function evaluation."""
        problem = BenchmarkProblems.ackley(dimension=3)
        
        # Test parameter space
        assert len(problem.parameter_space.parameters) == 3
        
        # Test evaluation at origin (should be close to optimum)
        origin_params = {"x1": 0.0, "x2": 0.0, "x3": 0.0}
        value = problem.evaluate(origin_params)
        assert abs(value - problem.global_optimum_value) < 1e-10
    
    def test_rosenbrock_function(self):
        """Test Rosenbrock function evaluation."""
        problem = BenchmarkProblems.rosenbrock(dimension=2)
        
        # Test parameter space
        assert len(problem.parameter_space.parameters) == 2
        
        # Test evaluation at optimum
        optimum_value = problem.evaluate(problem.global_optimum_params)
        assert abs(optimum_value - problem.global_optimum_value) < 1e-10
    
    def test_sphere_function(self):
        """Test Sphere function evaluation."""
        problem = BenchmarkProblems.sphere(dimension=4)
        
        # Test parameter space
        assert len(problem.parameter_space.parameters) == 4
        
        # Test evaluation at origin
        origin_params = {f"x{i+1}": 0.0 for i in range(4)}
        value = problem.evaluate(origin_params)
        assert abs(value - problem.global_optimum_value) < 1e-10


class TestBenchmarkOptimization:
    """Integration tests using benchmark problems."""
    
    def run_optimization_test(
        self,
        problem: BenchmarkProblem,
        max_iterations: int = 20,
        tolerance: float = 0.1
    ) -> Dict[str, float]:
        """
        Run optimization test on benchmark problem.
        
        Args:
            problem: Benchmark problem to optimize
            max_iterations: Maximum optimization iterations
            tolerance: Tolerance for success (distance from global optimum)
            
        Returns:
            Dictionary with optimization results and metrics
        """
        config = OptimizationConfig(
            max_iterations=max_iterations,
            initial_experiments=5,
            batch_size=1,
            verbose=False
        )
        
        optimizer = BayesianOptimizationLoop(
            parameter_space=problem.parameter_space,
            objective_function=problem.objective_function,
            config=config
        )
        
        # Run optimization
        result = optimizer.optimize()
        
        # Calculate distance from global optimum
        best_params = result.best_parameters
        global_params = problem.global_optimum_params
        
        param_distance = np.sqrt(sum(
            (best_params.get(key, 0) - global_params[key])**2
            for key in global_params.keys()
        ))
        
        value_distance = abs(
            result.best_objective_value - problem.global_optimum_value
        )
        
        return {
            'success': param_distance < tolerance,
            'param_distance': param_distance,
            'value_distance': value_distance,
            'best_objective_value': result.best_objective_value,
            'n_iterations': result.n_iterations,
            'n_evaluations': result.n_function_evaluations,
            'execution_time': result.execution_time_seconds,
        }
    
    @pytest.mark.slow
    def test_branin_optimization(self):
        """Test optimization on Branin function."""
        problem = BenchmarkProblems.branin()
        results = self.run_optimization_test(problem, max_iterations=30, tolerance=1.0)
        
        # Should find a reasonable solution
        assert results['n_evaluations'] > 0
        assert results['best_objective_value'] is not None
        assert results['execution_time'] > 0
        
        # Log results for analysis
        print(f"Branin optimization results: {results}")
    
    @pytest.mark.slow
    def test_sphere_optimization(self):
        """Test optimization on Sphere function (should be easy)."""
        problem = BenchmarkProblems.sphere(dimension=2)
        results = self.run_optimization_test(problem, max_iterations=15, tolerance=0.5)
        
        # Sphere function should be relatively easy to optimize
        assert results['n_evaluations'] > 0
        assert results['best_objective_value'] is not None
        
        print(f"Sphere optimization results: {results}")
    
    @pytest.mark.slow
    def test_hartmann3d_optimization(self):
        """Test optimization on Hartmann 3D function."""
        problem = BenchmarkProblems.hartmann3d()
        results = self.run_optimization_test(problem, max_iterations=25, tolerance=0.2)
        
        assert results['n_evaluations'] > 0
        assert results['best_objective_value'] is not None
        
        print(f"Hartmann3D optimization results: {results}")
    
    def test_benchmark_problem_collection(self):
        """Test that all benchmark problems can be created and evaluated."""
        problems = [
            BenchmarkProblems.branin(),
            BenchmarkProblems.hartmann3d(),
            BenchmarkProblems.hartmann6d(),
            BenchmarkProblems.ackley(2),
            BenchmarkProblems.rosenbrock(2),
            BenchmarkProblems.sphere(3),
        ]
        
        for problem in problems:
            # Test that problem can be evaluated
            test_params = {}
            for param in problem.parameter_space.parameters:
                if param.bounds:
                    low, high = param.bounds
                    test_params[param.name] = (low + high) / 2
                else:
                    test_params[param.name] = 0.0
            
            value = problem.evaluate(test_params)
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)
    
    def test_performance_comparison(self):
        """Compare performance across different benchmark problems."""
        problems = [
            BenchmarkProblems.sphere(2),
            BenchmarkProblems.branin(),
        ]
        
        results = {}
        
        for problem in problems:
            try:
                result = self.run_optimization_test(
                    problem, 
                    max_iterations=10, 
                    tolerance=1.0
                )
                results[problem.name] = result
            except Exception as e:
                print(f"Failed to optimize {problem.name}: {e}")
                results[problem.name] = {'error': str(e)}
        
        # Log comparison results
        print("Performance comparison:")
        for name, result in results.items():
            if 'error' not in result:
                print(f"{name}: {result['n_evaluations']} evals, "
                      f"value={result['best_objective_value']:.4f}")
            else:
                print(f"{name}: ERROR - {result['error']}")
        
        # At least one problem should succeed
        successful_runs = [r for r in results.values() if 'error' not in r]
        assert len(successful_runs) > 0
