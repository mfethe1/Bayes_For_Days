"""
Unit tests for Bayesian optimization loop integration.

Tests the complete optimization loop including:
- Convergence detection
- Experiment scheduling
- Integration with surrogate models and acquisition functions
- Stopping criteria and termination conditions
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from bayes_for_days.optimization.optimization_loop import (
    OptimizationConfig,
    ConvergenceDetector,
    ExperimentScheduler,
    BayesianOptimizationLoop,
)
from bayes_for_days.core.types import (
    ParameterSpace,
    Parameter,
    ParameterType,
    ExperimentPoint,
)


class TestOptimizationConfig:
    """Test suite for OptimizationConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizationConfig()
        
        assert config.max_iterations == 100
        assert config.initial_experiments == 5
        assert config.batch_size == 1
        assert config.convergence_tolerance == 1e-6
        assert config.acquisition_function == "expected_improvement"
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = OptimizationConfig(
            max_iterations=50,
            initial_experiments=10,
            batch_size=3,
            acquisition_function="upper_confidence_bound"
        )
        
        assert config.max_iterations == 50
        assert config.initial_experiments == 10
        assert config.batch_size == 3
        assert config.acquisition_function == "upper_confidence_bound"
    
    def test_invalid_config(self):
        """Test that invalid configurations raise errors."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            OptimizationConfig(max_iterations=0)
        
        with pytest.raises(ValueError, match="initial_experiments must be positive"):
            OptimizationConfig(initial_experiments=-1)
        
        with pytest.raises(ValueError, match="batch_size must be positive"):
            OptimizationConfig(batch_size=0)


class TestConvergenceDetector:
    """Test suite for ConvergenceDetector class."""
    
    def test_initialization(self):
        """Test convergence detector initialization."""
        detector = ConvergenceDetector(
            tolerance=1e-4,
            max_iterations_without_improvement=5,
            improvement_threshold=1e-3
        )
        
        assert detector.tolerance == 1e-4
        assert detector.max_iterations_without_improvement == 5
        assert detector.improvement_threshold == 1e-3
        assert detector.best_value is None
        assert detector.iterations_without_improvement == 0
    
    def test_improvement_detection(self):
        """Test detection of improvements."""
        detector = ConvergenceDetector(improvement_threshold=0.1)
        
        # First update should not trigger convergence
        converged = detector.update(1.0)
        assert not converged
        assert detector.best_value == 1.0
        assert detector.iterations_without_improvement == 0
        
        # Significant improvement
        converged = detector.update(1.2)
        assert not converged
        assert detector.best_value == 1.2
        assert detector.iterations_without_improvement == 0
        
        # Small improvement (below threshold)
        converged = detector.update(1.21)
        assert not converged
        assert detector.best_value == 1.2  # Should not update
        assert detector.iterations_without_improvement == 1
    
    def test_convergence_by_no_improvement(self):
        """Test convergence detection by lack of improvement."""
        detector = ConvergenceDetector(
            max_iterations_without_improvement=3,
            improvement_threshold=0.1
        )
        
        # Initial value
        detector.update(1.0)
        
        # No improvements for 3 iterations
        for i in range(3):
            converged = detector.update(0.9)  # Worse value
            if i < 2:
                assert not converged
            else:
                assert converged  # Should converge on 3rd iteration
    
    def test_convergence_by_stability(self):
        """Test convergence detection by objective value stability."""
        detector = ConvergenceDetector(tolerance=1e-3)
        
        # Add 5 very similar values
        base_value = 1.0
        for i in range(5):
            value = base_value + i * 1e-4  # Very small differences
            converged = detector.update(value)
            
            if i < 4:
                assert not converged
            else:
                assert converged  # Should converge when stability is detected
    
    def test_convergence_info(self):
        """Test convergence information retrieval."""
        detector = ConvergenceDetector()
        
        # Add some values
        detector.update(1.0)
        detector.update(1.5)
        detector.update(1.3)
        
        info = detector.get_convergence_info()
        
        assert info['best_value'] == 1.5
        assert info['total_iterations'] == 3
        assert len(info['objective_history']) == 3
        assert 'is_converged' in info


class TestExperimentScheduler:
    """Test suite for ExperimentScheduler class."""
    
    def test_initialization(self):
        """Test experiment scheduler initialization."""
        scheduler = ExperimentScheduler(
            parallel_experiments=True,
            experiment_timeout_minutes=30.0
        )
        
        assert scheduler.parallel_experiments is True
        assert scheduler.experiment_timeout_minutes == 30.0
        assert len(scheduler.pending_experiments) == 0
        assert len(scheduler.completed_experiments) == 0
    
    def test_schedule_experiments(self):
        """Test experiment scheduling."""
        scheduler = ExperimentScheduler()
        
        experiments = [
            {"x": 1.0, "y": 2.0},
            {"x": 3.0, "y": 4.0},
        ]
        
        scheduler.schedule_experiments(experiments)
        
        assert len(scheduler.pending_experiments) == 2
        assert scheduler.pending_experiments[0] == {"x": 1.0, "y": 2.0}
        assert scheduler.pending_experiments[1] == {"x": 3.0, "y": 4.0}
    
    def test_execute_experiments_sequential(self):
        """Test sequential experiment execution."""
        scheduler = ExperimentScheduler(parallel_experiments=False)
        
        # Mock objective function
        def mock_objective(params):
            return params["x"] + params["y"]
        
        # Schedule experiments
        experiments = [
            {"x": 1.0, "y": 2.0},
            {"x": 3.0, "y": 4.0},
        ]
        scheduler.schedule_experiments(experiments)
        
        # Execute experiments
        results = scheduler.execute_experiments(mock_objective)
        
        assert len(results) == 2
        assert len(scheduler.pending_experiments) == 0
        assert len(scheduler.completed_experiments) == 2
        
        # Check results
        assert results[0].parameters == {"x": 1.0, "y": 2.0}
        assert results[0].objectives["objective"] == 3.0
        assert results[0].is_feasible is True
        
        assert results[1].parameters == {"x": 3.0, "y": 4.0}
        assert results[1].objectives["objective"] == 7.0
        assert results[1].is_feasible is True
    
    def test_execute_experiments_with_failure(self):
        """Test experiment execution with failures."""
        scheduler = ExperimentScheduler()
        
        # Mock objective function that fails
        def failing_objective(params):
            if params["x"] > 2.0:
                raise ValueError("Experiment failed")
            return params["x"] + params["y"]
        
        # Schedule experiments
        experiments = [
            {"x": 1.0, "y": 2.0},  # Should succeed
            {"x": 3.0, "y": 4.0},  # Should fail
        ]
        scheduler.schedule_experiments(experiments)
        
        # Execute experiments
        results = scheduler.execute_experiments(failing_objective)
        
        assert len(results) == 2
        
        # First experiment should succeed
        assert results[0].is_feasible is True
        assert results[0].objectives["objective"] == 3.0
        
        # Second experiment should fail
        assert results[1].is_feasible is False
        assert results[1].objectives["objective"] == float('-inf')
        assert "error" in results[1].metadata
    
    def test_scheduler_status(self):
        """Test scheduler status reporting."""
        scheduler = ExperimentScheduler()
        
        # Initial status
        status = scheduler.get_scheduler_status()
        assert status['pending_experiments'] == 0
        assert status['completed_experiments'] == 0
        
        # Schedule some experiments
        scheduler.schedule_experiments([{"x": 1.0}])
        status = scheduler.get_scheduler_status()
        assert status['pending_experiments'] == 1
        
        # Execute experiments
        scheduler.execute_experiments(lambda p: p["x"])
        status = scheduler.get_scheduler_status()
        assert status['pending_experiments'] == 0
        assert status['completed_experiments'] == 1


class TestBayesianOptimizationLoop:
    """Test suite for BayesianOptimizationLoop class."""
    
    def create_test_parameter_space(self):
        """Create a simple parameter space for testing."""
        parameters = [
            Parameter(name="x", type=ParameterType.CONTINUOUS, bounds=(-5, 5)),
            Parameter(name="y", type=ParameterType.CONTINUOUS, bounds=(-5, 5)),
        ]
        return ParameterSpace(parameters=parameters)
    
    def create_test_objective(self):
        """Create a simple test objective function."""
        def objective(params):
            x, y = params["x"], params["y"]
            # Simple quadratic with optimum at (1, 1)
            return -(x - 1)**2 - (y - 1)**2
        return objective
    
    def test_initialization(self):
        """Test optimization loop initialization."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective()
        config = OptimizationConfig(max_iterations=10, initial_experiments=3)
        
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config
        )
        
        assert optimizer.parameter_space == parameter_space
        assert optimizer.objective_function == objective
        assert optimizer.config == config
        assert optimizer.iteration == 0
        assert optimizer.is_converged is False
        assert len(optimizer.all_experiments) == 0
    
    def test_default_model_creation(self):
        """Test creation of default surrogate model."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective()
        config = OptimizationConfig()
        
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config
        )
        
        assert optimizer.surrogate_model is not None
        assert hasattr(optimizer.surrogate_model, 'fit')
        assert hasattr(optimizer.surrogate_model, 'predict')
    
    def test_default_acquisition_creation(self):
        """Test creation of default acquisition function."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective()
        config = OptimizationConfig(acquisition_function="expected_improvement")
        
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config
        )
        
        assert optimizer.acquisition_function is not None
        assert hasattr(optimizer.acquisition_function, 'evaluate')
        assert hasattr(optimizer.acquisition_function, 'optimize')
    
    def test_unknown_acquisition_function(self):
        """Test error handling for unknown acquisition function."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective()
        config = OptimizationConfig(acquisition_function="unknown_function")
        
        with pytest.raises(ValueError, match="Unknown acquisition function"):
            BayesianOptimizationLoop(
                parameter_space=parameter_space,
                objective_function=objective,
                config=config
            )
    
    @patch('bayes_for_days.models.gaussian_process.GaussianProcessModel')
    def test_optimization_loop_execution(self, mock_gp_class):
        """Test complete optimization loop execution."""
        # Mock the GP model
        mock_gp = Mock()
        mock_gp.fit.return_value = None
        mock_gp.predict.return_value = Mock(mean=0.0, std=1.0)
        mock_gp_class.return_value = mock_gp
        
        # Mock acquisition function
        mock_acquisition = Mock()
        mock_acquisition.optimize.return_value = [{"x": 1.0, "y": 1.0}]
        
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective()
        config = OptimizationConfig(
            max_iterations=3,
            initial_experiments=2,
            batch_size=1
        )
        
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config,
            acquisition_function=mock_acquisition
        )
        
        # Run optimization
        result = optimizer.optimize()
        
        # Check results
        assert result is not None
        assert result.n_iterations <= config.max_iterations
        assert result.n_function_evaluations >= config.initial_experiments
        assert result.best_parameters is not None
        assert result.best_objective_value is not None
        assert len(result.all_experiments) >= config.initial_experiments
    
    def test_termination_by_max_iterations(self):
        """Test termination by maximum iterations."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective()
        config = OptimizationConfig(max_iterations=2, initial_experiments=1)
        
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config
        )
        
        # Mock components to prevent actual model fitting
        optimizer.surrogate_model = Mock()
        optimizer.surrogate_model.fit.return_value = None
        optimizer.acquisition_function = Mock()
        optimizer.acquisition_function.optimize.return_value = [{"x": 0.0, "y": 0.0}]
        
        result = optimizer.optimize()
        
        assert result.n_iterations == config.max_iterations
        assert not result.is_converged  # Should terminate by max iterations, not convergence
    
    def test_optimization_status(self):
        """Test optimization status reporting."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective()
        config = OptimizationConfig()
        
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config
        )
        
        # Initial status
        status = optimizer.get_optimization_status()
        
        assert status['iteration'] == 0
        assert status['total_experiments'] == 0
        assert status['is_converged'] is False
        assert status['best_objective_value'] is None
        assert 'convergence_info' in status
        assert 'scheduler_status' in status


class TestIntegration:
    """Integration tests for the complete optimization system."""
    
    def test_simple_optimization_problem(self):
        """Test optimization on a simple quadratic function."""
        # Define parameter space
        parameters = [
            Parameter(name="x", type=ParameterType.CONTINUOUS, bounds=(-2, 2)),
        ]
        parameter_space = ParameterSpace(parameters=parameters)
        
        # Define objective function (simple quadratic with optimum at x=1)
        def objective(params):
            x = params["x"]
            return -(x - 1)**2
        
        # Configure optimization
        config = OptimizationConfig(
            max_iterations=5,
            initial_experiments=3,
            batch_size=1,
            convergence_tolerance=1e-3,
            verbose=False
        )
        
        # Run optimization
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config
        )
        
        # Mock the surrogate model to avoid actual GP fitting
        optimizer.surrogate_model = Mock()
        optimizer.surrogate_model.fit.return_value = None
        
        # Mock acquisition function to suggest points near optimum
        mock_acquisition = Mock()
        mock_acquisition.optimize.return_value = [{"x": 1.0 + np.random.normal(0, 0.1)}]
        optimizer.acquisition_function = mock_acquisition
        
        result = optimizer.optimize()
        
        # Check that optimization completed
        assert result is not None
        assert result.n_function_evaluations >= config.initial_experiments
        assert result.best_parameters is not None
        assert "x" in result.best_parameters
        assert result.execution_time_seconds > 0
    
    def test_multi_dimensional_optimization(self):
        """Test optimization on a multi-dimensional problem."""
        # Define 2D parameter space
        parameters = [
            Parameter(name="x1", type=ParameterType.CONTINUOUS, bounds=(-3, 3)),
            Parameter(name="x2", type=ParameterType.CONTINUOUS, bounds=(-3, 3)),
        ]
        parameter_space = ParameterSpace(parameters=parameters)
        
        # Define 2D objective function (Rosenbrock-like)
        def objective(params):
            x1, x2 = params["x1"], params["x2"]
            return -(100 * (x2 - x1**2)**2 + (1 - x1)**2)
        
        config = OptimizationConfig(
            max_iterations=3,
            initial_experiments=5,
            verbose=False
        )
        
        optimizer = BayesianOptimizationLoop(
            parameter_space=parameter_space,
            objective_function=objective,
            config=config
        )
        
        # Mock components
        optimizer.surrogate_model = Mock()
        optimizer.surrogate_model.fit.return_value = None
        optimizer.acquisition_function = Mock()
        optimizer.acquisition_function.optimize.return_value = [{"x1": 1.0, "x2": 1.0}]
        
        result = optimizer.optimize()
        
        assert result is not None
        assert "x1" in result.best_parameters
        assert "x2" in result.best_parameters
        assert len(result.all_experiments) >= config.initial_experiments
