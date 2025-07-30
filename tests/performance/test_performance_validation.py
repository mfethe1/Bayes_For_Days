"""
Performance validation tests for Bayes For Days platform.

This module validates that the platform meets performance requirements:
- Optimization convergence within specified iterations
- Model training and prediction speed benchmarks
- Memory usage and scalability limits
- Multi-objective optimization efficiency
- Real-world performance scenarios

Performance targets:
- GP model fitting: <30s for 100 points, <5min for 1000 points
- Predictions: <1s for single point, <10s for 100 points
- Multi-objective optimization: <30s for 100-point Pareto front
- Memory usage: <2GB for typical problems
"""

import pytest
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import gc

from bayes_for_days.core.types import (
    ParameterSpace,
    Parameter,
    ParameterType,
    ExperimentPoint,
)
from bayes_for_days.models.gaussian_process import GaussianProcessModel
from bayes_for_days.models.ensemble import EnsembleSurrogateModel, SimpleAveraging
from bayes_for_days.acquisition.expected_improvement import ExpectedImprovement
from bayes_for_days.optimization.multi_objective import NSGAIIOptimizer
from bayes_for_days.optimization.optimization_loop import (
    BayesianOptimizationLoop,
    OptimizationConfig,
)


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    execution_time: float
    memory_usage_mb: float
    peak_memory_mb: float
    cpu_usage_percent: float
    success: bool
    error_message: str = ""
    additional_metrics: Dict[str, Any] = None


class PerformanceMonitor:
    """Monitor system performance during test execution."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def update_peak_memory(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory)
        
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics."""
        if self.start_time is None:
            raise ValueError("Monitoring not started")
        
        execution_time = time.time() - self.start_time
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_usage = current_memory - self.start_memory
        cpu_usage = self.process.cpu_percent()
        
        return PerformanceMetrics(
            execution_time=execution_time,
            memory_usage_mb=memory_usage,
            peak_memory_mb=self.peak_memory,
            cpu_usage_percent=cpu_usage,
            success=True
        )


class TestPerformanceValidation:
    """Performance validation test suite."""
    
    def create_test_data(self, n_points: int, n_dims: int) -> List[ExperimentPoint]:
        """Create test data for performance testing."""
        data = []
        
        for i in range(n_points):
            # Generate random parameters
            params = {f"x{j}": np.random.uniform(-5, 5) for j in range(n_dims)}
            
            # Simple quadratic objective with noise
            x = np.array(list(params.values()))
            objective_value = -np.sum(x**2) + np.random.normal(0, 0.1)
            
            experiment_point = ExperimentPoint(
                parameters=params,
                objectives={"objective": objective_value},
                is_feasible=True,
                metadata={"test_id": i}
            )
            data.append(experiment_point)
        
        return data
    
    def create_parameter_space(self, n_dims: int) -> ParameterSpace:
        """Create parameter space for testing."""
        parameters = [
            Parameter(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(-5, 5))
            for i in range(n_dims)
        ]
        return ParameterSpace(parameters=parameters)
    
    @pytest.mark.performance
    def test_gp_model_fitting_performance(self):
        """Test GP model fitting performance with different data sizes."""
        test_cases = [
            (50, 2, 10.0),    # 50 points, 2D, <10s
            (100, 3, 30.0),   # 100 points, 3D, <30s
            (200, 5, 60.0),   # 200 points, 5D, <60s
        ]
        
        results = {}
        
        for n_points, n_dims, time_limit in test_cases:
            print(f"\nTesting GP fitting: {n_points} points, {n_dims}D")
            
            # Create test data
            parameter_space = self.create_parameter_space(n_dims)
            training_data = self.create_test_data(n_points, n_dims)
            
            # Create GP model
            gp_model = GaussianProcessModel(
                parameter_space=parameter_space,
                n_inducing_points=min(50, n_points // 2)
            )
            
            # Monitor performance
            monitor = PerformanceMonitor()
            monitor.start_monitoring()
            
            try:
                # Fit model
                gp_model.fit(training_data)
                
                # Update memory tracking
                monitor.update_peak_memory()
                
                # Get metrics
                metrics = monitor.get_metrics()
                
                # Validate performance
                assert metrics.execution_time < time_limit, (
                    f"GP fitting took {metrics.execution_time:.2f}s, "
                    f"expected <{time_limit}s"
                )
                
                assert metrics.peak_memory_mb < 1000, (
                    f"Peak memory {metrics.peak_memory_mb:.1f}MB exceeds 1GB limit"
                )
                
                results[f"{n_points}_{n_dims}D"] = metrics
                
                print(f"  Time: {metrics.execution_time:.2f}s")
                print(f"  Memory: {metrics.memory_usage_mb:.1f}MB")
                print(f"  Peak Memory: {metrics.peak_memory_mb:.1f}MB")
                
            except Exception as e:
                pytest.fail(f"GP fitting failed for {n_points} points, {n_dims}D: {e}")
            
            # Clean up
            del gp_model, training_data
            gc.collect()
        
        # Overall performance summary
        print(f"\nGP Model Fitting Performance Summary:")
        for case, metrics in results.items():
            print(f"  {case}: {metrics.execution_time:.2f}s, {metrics.peak_memory_mb:.1f}MB")
    
    @pytest.mark.performance
    def test_gp_prediction_performance(self):
        """Test GP prediction performance."""
        # Create model and training data
        parameter_space = self.create_parameter_space(3)
        training_data = self.create_test_data(100, 3)
        
        gp_model = GaussianProcessModel(parameter_space=parameter_space)
        gp_model.fit(training_data)
        
        # Test single prediction
        test_params = {"x0": 1.0, "x1": 2.0, "x2": -1.0}
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        prediction = gp_model.predict(test_params)
        
        metrics = monitor.get_metrics()
        
        assert metrics.execution_time < 1.0, (
            f"Single prediction took {metrics.execution_time:.4f}s, expected <1s"
        )
        
        print(f"Single prediction: {metrics.execution_time:.4f}s")
        
        # Test batch prediction
        batch_params = [
            {f"x{j}": np.random.uniform(-5, 5) for j in range(3)}
            for _ in range(100)
        ]
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        batch_predictions = gp_model.predict(batch_params)
        
        metrics = monitor.get_metrics()
        
        assert metrics.execution_time < 10.0, (
            f"Batch prediction took {metrics.execution_time:.2f}s, expected <10s"
        )
        
        assert len(batch_predictions) == 100
        
        print(f"Batch prediction (100 points): {metrics.execution_time:.2f}s")
    
    @pytest.mark.performance
    def test_ensemble_model_performance(self):
        """Test ensemble model performance."""
        parameter_space = self.create_parameter_space(3)
        training_data = self.create_test_data(100, 3)
        
        # Create ensemble with multiple base models
        base_models = [
            GaussianProcessModel(parameter_space=parameter_space, kernel_type="matern"),
            GaussianProcessModel(parameter_space=parameter_space, kernel_type="rbf"),
            GaussianProcessModel(parameter_space=parameter_space, kernel_type="matern", ard=False),
        ]
        
        ensemble_model = EnsembleSurrogateModel(
            base_models=base_models,
            parameter_space=parameter_space,
            ensemble_method=SimpleAveraging()
        )
        
        # Test fitting performance
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        ensemble_model.fit(training_data)
        
        fit_metrics = monitor.get_metrics()
        
        # Should be reasonable even with multiple models
        assert fit_metrics.execution_time < 120.0, (
            f"Ensemble fitting took {fit_metrics.execution_time:.2f}s, expected <120s"
        )
        
        print(f"Ensemble fitting (3 models): {fit_metrics.execution_time:.2f}s")
        
        # Test prediction performance
        test_params = {"x0": 1.0, "x1": 2.0, "x2": -1.0}
        
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        prediction = ensemble_model.predict(test_params)
        
        pred_metrics = monitor.get_metrics()
        
        assert pred_metrics.execution_time < 2.0, (
            f"Ensemble prediction took {pred_metrics.execution_time:.4f}s, expected <2s"
        )
        
        print(f"Ensemble prediction: {pred_metrics.execution_time:.4f}s")
    
    @pytest.mark.performance
    def test_multi_objective_optimization_performance(self):
        """Test multi-objective optimization performance."""
        parameter_space = self.create_parameter_space(4)
        
        # Multi-objective function
        def multi_objective(params):
            x = np.array([params[f"x{i}"] for i in range(4)])
            
            # Two conflicting objectives
            obj1 = -np.sum(x**2)  # Minimize sum of squares
            obj2 = -np.sum((x - 2)**2)  # Minimize distance from (2,2,2,2)
            
            return [obj1, obj2]
        
        # Create NSGA-II optimizer
        optimizer = NSGAIIOptimizer(
            parameter_space=parameter_space,
            population_size=50,
            max_generations=20,
            crossover_prob=0.9,
            mutation_prob=0.1
        )
        
        # Monitor performance
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        result = optimizer.optimize(
            objective_function=multi_objective,
            n_objectives=2
        )
        
        metrics = monitor.get_metrics()
        
        # Performance requirements
        assert metrics.execution_time < 60.0, (
            f"Multi-objective optimization took {metrics.execution_time:.2f}s, expected <60s"
        )
        
        assert len(result.pareto_front) > 0, "No Pareto front found"
        assert len(result.pareto_front) <= 50, "Pareto front too large"
        
        # Check quality of Pareto front
        pareto_size = len(result.pareto_front)
        assert pareto_size >= 5, f"Pareto front too small: {pareto_size}"
        
        print(f"Multi-objective optimization:")
        print(f"  Time: {metrics.execution_time:.2f}s")
        print(f"  Pareto front size: {pareto_size}")
        print(f"  Function evaluations: {result.n_function_evaluations}")
    
    @pytest.mark.performance
    def test_bayesian_optimization_convergence(self):
        """Test Bayesian optimization convergence performance."""
        parameter_space = self.create_parameter_space(3)
        
        # Test function with known optimum
        def objective(params):
            x = np.array([params[f"x{i}"] for i in range(3)])
            # Optimum at (1, 1, 1) with value 0
            return -np.sum((x - 1)**2)
        
        # Create optimization configuration
        config = OptimizationConfig(
            max_iterations=30,
            initial_experiments=5,
            batch_size=1,
            convergence_tolerance=1e-3,
            verbose=False
        )
        
        # Create optimizer
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config
        )
        
        # Monitor performance
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        result = optimizer.optimize()
        
        metrics = monitor.get_metrics()
        
        # Performance requirements
        assert metrics.execution_time < 120.0, (
            f"Bayesian optimization took {metrics.execution_time:.2f}s, expected <120s"
        )
        
        # Convergence requirements
        best_value = result.best_objective_value
        optimal_value = 0.0  # Known optimum
        
        # Should get close to optimum
        assert best_value > optimal_value - 0.5, (
            f"Optimization result {best_value:.4f} too far from optimum {optimal_value}"
        )
        
        # Should converge reasonably quickly
        assert result.n_function_evaluations <= 35, (
            f"Too many function evaluations: {result.n_function_evaluations}"
        )
        
        print(f"Bayesian optimization convergence:")
        print(f"  Time: {metrics.execution_time:.2f}s")
        print(f"  Best value: {best_value:.4f} (optimum: {optimal_value})")
        print(f"  Function evaluations: {result.n_function_evaluations}")
        print(f"  Converged: {result.is_converged}")
    
    @pytest.mark.performance
    def test_memory_usage_scalability(self):
        """Test memory usage with increasing problem sizes."""
        memory_results = {}
        
        test_cases = [
            (50, 2),
            (100, 3),
            (200, 4),
            (500, 5),
        ]
        
        for n_points, n_dims in test_cases:
            print(f"\nTesting memory usage: {n_points} points, {n_dims}D")
            
            # Force garbage collection
            gc.collect()
            
            # Get baseline memory
            process = psutil.Process(os.getpid())
            baseline_memory = process.memory_info().rss / 1024 / 1024
            
            # Create and fit model
            parameter_space = self.create_parameter_space(n_dims)
            training_data = self.create_test_data(n_points, n_dims)
            
            gp_model = GaussianProcessModel(
                parameter_space=parameter_space,
                n_inducing_points=min(100, n_points // 2)
            )
            
            gp_model.fit(training_data)
            
            # Measure peak memory
            peak_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = peak_memory - baseline_memory
            
            # Memory requirements (rough guidelines)
            expected_memory_mb = n_points * n_dims * 0.1  # Very rough estimate
            max_allowed_memory = max(100, expected_memory_mb * 5)  # 5x safety factor
            
            assert memory_increase < max_allowed_memory, (
                f"Memory usage {memory_increase:.1f}MB exceeds limit {max_allowed_memory:.1f}MB"
            )
            
            memory_results[f"{n_points}_{n_dims}D"] = memory_increase
            
            print(f"  Memory increase: {memory_increase:.1f}MB")
            
            # Clean up
            del gp_model, training_data
            gc.collect()
        
        # Check memory scaling
        print(f"\nMemory Usage Scaling:")
        for case, memory in memory_results.items():
            print(f"  {case}: {memory:.1f}MB")
        
        # Memory should scale reasonably
        small_case_memory = memory_results["50_2D"]
        large_case_memory = memory_results["500_5D"]
        
        # Large case should not use more than 20x memory of small case
        memory_ratio = large_case_memory / small_case_memory
        assert memory_ratio < 20, (
            f"Memory scaling too steep: {memory_ratio:.1f}x increase"
        )
    
    @pytest.mark.performance
    def test_acquisition_function_optimization_performance(self):
        """Test acquisition function optimization performance."""
        parameter_space = self.create_parameter_space(4)
        training_data = self.create_test_data(50, 4)
        
        # Create and fit GP model
        gp_model = GaussianProcessModel(parameter_space=parameter_space)
        gp_model.fit(training_data)
        
        # Create acquisition function
        acquisition = ExpectedImprovement(surrogate_model=gp_model)
        
        # Test acquisition optimization
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        
        candidates = acquisition.optimize(
            n_candidates=5,
            bounds=parameter_space.get_bounds(),
            n_restarts=10
        )
        
        metrics = monitor.get_metrics()
        
        # Performance requirements
        assert metrics.execution_time < 30.0, (
            f"Acquisition optimization took {metrics.execution_time:.2f}s, expected <30s"
        )
        
        assert len(candidates) == 5, f"Expected 5 candidates, got {len(candidates)}"
        
        # Validate candidates are within bounds
        bounds = parameter_space.get_bounds()
        for candidate in candidates:
            for i, param_name in enumerate([f"x{j}" for j in range(4)]):
                if param_name in candidate:
                    value = candidate[param_name]
                    low, high = bounds[i]
                    assert low <= value <= high, (
                        f"Candidate {param_name}={value} outside bounds [{low}, {high}]"
                    )
        
        print(f"Acquisition optimization:")
        print(f"  Time: {metrics.execution_time:.2f}s")
        print(f"  Candidates found: {len(candidates)}")
    
    def test_performance_summary(self):
        """Generate performance summary report."""
        print("\n" + "="*60)
        print("BAYES FOR DAYS - PERFORMANCE VALIDATION SUMMARY")
        print("="*60)
        
        # This would typically aggregate results from other tests
        # For now, just print the performance targets
        
        performance_targets = {
            "GP Model Fitting": {
                "100 points, 3D": "<30s",
                "200 points, 5D": "<60s",
                "Memory usage": "<1GB"
            },
            "Predictions": {
                "Single prediction": "<1s",
                "Batch (100 points)": "<10s"
            },
            "Multi-objective Optimization": {
                "50 population, 20 generations": "<60s",
                "Pareto front size": "5-50 points"
            },
            "Bayesian Optimization": {
                "Convergence": "<30 iterations",
                "Total time": "<120s"
            },
            "Memory Scalability": {
                "Scaling factor": "<20x for 10x data increase"
            }
        }
        
        print("\nPerformance Targets:")
        for category, targets in performance_targets.items():
            print(f"\n{category}:")
            for metric, target in targets.items():
                print(f"  {metric}: {target}")
        
        print("\nAll performance tests should validate against these targets.")
        print("="*60)


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-m", "performance", "--tb=short"])
