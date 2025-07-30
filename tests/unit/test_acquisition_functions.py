"""
Unit tests for acquisition functions.

Tests the acquisition function implementations including:
- Expected Improvement (EI)
- Upper Confidence Bound (UCB)
- Probability of Improvement (PI)
- Optimization capabilities and performance
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from datetime import datetime

from bayes_for_days.optimization.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    ProbabilityOfImprovement,
)
from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    Parameter,
    ParameterType,
    ModelPrediction,
    ModelType,
    AcquisitionValue,
    AcquisitionFunction as AcquisitionFunctionType,
)


@pytest.fixture
def mock_surrogate_model():
    """Create a mock surrogate model for testing."""
    model = Mock()
    
    # Mock parameter space
    parameters = [
        Parameter(name="x1", type=ParameterType.CONTINUOUS, bounds=(-5, 5)),
        Parameter(name="x2", type=ParameterType.CONTINUOUS, bounds=(-5, 5)),
    ]
    model.parameter_space = ParameterSpace(parameters=parameters)
    
    # Mock training data
    model.training_data = [
        ExperimentPoint(
            parameters={"x1": 0.0, "x2": 0.0},
            objectives={"objective": 0.8},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"x1": 1.0, "x2": 1.0},
            objectives={"objective": 0.6},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"x1": -1.0, "x2": -1.0},
            objectives={"objective": 0.4},
            is_feasible=True,
        ),
    ]
    
    # Mock predict method
    def mock_predict(parameters):
        if isinstance(parameters, dict):
            parameters = [parameters]
        
        predictions = []
        for param_dict in parameters:
            x1 = param_dict.get("x1", 0.0)
            x2 = param_dict.get("x2", 0.0)
            
            # Simple synthetic prediction
            mean = 0.5 + 0.1 * (x1 + x2)
            variance = 0.1 + 0.05 * abs(x1 + x2)
            std = np.sqrt(variance)
            
            pred = ModelPrediction(
                mean=mean,
                variance=variance,
                std=std,
                confidence_interval=(mean - 1.96 * std, mean + 1.96 * std),
                model_type=ModelType.GAUSSIAN_PROCESS,
            )
            predictions.append(pred)
        
        return predictions[0] if len(predictions) == 1 else predictions
    
    model.predict = mock_predict
    return model


class TestExpectedImprovement:
    """Test suite for Expected Improvement acquisition function."""
    
    def test_initialization(self, mock_surrogate_model):
        """Test EI initialization with various parameters."""
        ei = ExpectedImprovement(mock_surrogate_model, xi=0.01)
        
        assert ei.surrogate_model == mock_surrogate_model
        assert ei.xi == 0.01
        assert ei.current_best == 0.8  # Best from mock training data
    
    def test_initialization_custom_xi(self, mock_surrogate_model):
        """Test EI initialization with custom exploration parameter."""
        ei = ExpectedImprovement(mock_surrogate_model, xi=0.1)
        assert ei.xi == 0.1
    
    def test_evaluate_single_point(self, mock_surrogate_model):
        """Test EI evaluation for a single parameter set."""
        ei = ExpectedImprovement(mock_surrogate_model, xi=0.01)
        
        test_params = {"x1": 2.0, "x2": 2.0}
        result = ei.evaluate(test_params)
        
        assert isinstance(result, AcquisitionValue)
        assert result.function_type == AcquisitionFunctionType.EXPECTED_IMPROVEMENT
        assert result.value >= 0.0  # EI should be non-negative
        assert "xi" in result.parameters
        assert "current_best" in result.parameters
    
    def test_evaluate_multiple_points(self, mock_surrogate_model):
        """Test EI evaluation for multiple parameter sets."""
        ei = ExpectedImprovement(mock_surrogate_model, xi=0.01)
        
        test_params_list = [
            {"x1": 1.0, "x2": 1.0},
            {"x1": 2.0, "x2": 2.0},
            {"x1": -1.0, "x2": -1.0},
        ]
        results = ei.evaluate(test_params_list)
        
        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, AcquisitionValue)
            assert result.value >= 0.0
    
    def test_evaluate_zero_uncertainty(self, mock_surrogate_model):
        """Test EI behavior when model has zero uncertainty."""
        # Mock predict to return zero variance
        def zero_uncertainty_predict(parameters):
            if isinstance(parameters, dict):
                parameters = [parameters]
            
            predictions = []
            for _ in parameters:
                pred = ModelPrediction(
                    mean=0.9,  # Higher than current best
                    variance=1e-10,  # Near zero
                    std=1e-5,
                    model_type=ModelType.GAUSSIAN_PROCESS,
                )
                predictions.append(pred)
            
            return predictions[0] if len(predictions) == 1 else predictions
        
        mock_surrogate_model.predict = zero_uncertainty_predict
        ei = ExpectedImprovement(mock_surrogate_model, xi=0.01)
        
        result = ei.evaluate({"x1": 0.0, "x2": 0.0})
        assert result.value >= 0.0
    
    def test_optimize_candidates(self, mock_surrogate_model):
        """Test optimization to find candidate points."""
        ei = ExpectedImprovement(mock_surrogate_model, xi=0.01)
        
        candidates = ei.optimize(n_candidates=3, n_restarts=5)
        
        assert isinstance(candidates, list)
        assert len(candidates) == 3
        for candidate in candidates:
            assert isinstance(candidate, dict)
            assert "x1" in candidate
            assert "x2" in candidate
            assert -5 <= candidate["x1"] <= 5
            assert -5 <= candidate["x2"] <= 5
    
    def test_optimize_with_bounds(self, mock_surrogate_model):
        """Test optimization with custom bounds."""
        ei = ExpectedImprovement(mock_surrogate_model, xi=0.01)
        
        custom_bounds = [(-2, 2), (-2, 2)]
        candidates = ei.optimize(n_candidates=2, bounds=custom_bounds, n_restarts=3)
        
        assert len(candidates) == 2
        for candidate in candidates:
            assert -2 <= candidate["x1"] <= 2
            assert -2 <= candidate["x2"] <= 2


class TestUpperConfidenceBound:
    """Test suite for Upper Confidence Bound acquisition function."""
    
    def test_initialization(self, mock_surrogate_model):
        """Test UCB initialization."""
        ucb = UpperConfidenceBound(mock_surrogate_model, beta=2.0)
        
        assert ucb.surrogate_model == mock_surrogate_model
        assert ucb.beta == 2.0
    
    def test_evaluate_single_point(self, mock_surrogate_model):
        """Test UCB evaluation for a single parameter set."""
        ucb = UpperConfidenceBound(mock_surrogate_model, beta=2.0)
        
        test_params = {"x1": 1.0, "x2": 1.0}
        result = ucb.evaluate(test_params)
        
        assert isinstance(result, AcquisitionValue)
        assert result.function_type == AcquisitionFunctionType.UPPER_CONFIDENCE_BOUND
        assert "beta" in result.parameters
        
        # UCB should be mean + beta * std
        prediction = mock_surrogate_model.predict(test_params)
        expected_ucb = prediction.mean + 2.0 * prediction.std
        assert abs(result.value - expected_ucb) < 1e-6
    
    def test_evaluate_multiple_points(self, mock_surrogate_model):
        """Test UCB evaluation for multiple parameter sets."""
        ucb = UpperConfidenceBound(mock_surrogate_model, beta=1.5)
        
        test_params_list = [
            {"x1": 0.0, "x2": 0.0},
            {"x1": 1.0, "x2": 1.0},
        ]
        results = ucb.evaluate(test_params_list)
        
        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, AcquisitionValue)
    
    def test_optimize_candidates(self, mock_surrogate_model):
        """Test UCB optimization."""
        ucb = UpperConfidenceBound(mock_surrogate_model, beta=2.0)
        
        candidates = ucb.optimize(n_candidates=2, n_restarts=3)
        
        assert isinstance(candidates, list)
        assert len(candidates) == 2
        for candidate in candidates:
            assert isinstance(candidate, dict)
            assert "x1" in candidate
            assert "x2" in candidate


class TestProbabilityOfImprovement:
    """Test suite for Probability of Improvement acquisition function."""
    
    def test_initialization(self, mock_surrogate_model):
        """Test PI initialization."""
        pi = ProbabilityOfImprovement(mock_surrogate_model, xi=0.01)
        
        assert pi.surrogate_model == mock_surrogate_model
        assert pi.xi == 0.01
        assert pi.current_best == 0.8  # Best from mock training data
    
    def test_evaluate_single_point(self, mock_surrogate_model):
        """Test PI evaluation for a single parameter set."""
        pi = ProbabilityOfImprovement(mock_surrogate_model, xi=0.01)
        
        test_params = {"x1": 2.0, "x2": 2.0}
        result = pi.evaluate(test_params)
        
        assert isinstance(result, AcquisitionValue)
        assert result.function_type == AcquisitionFunctionType.PROBABILITY_OF_IMPROVEMENT
        assert 0.0 <= result.value <= 1.0  # PI should be a probability
        assert "xi" in result.parameters
        assert "current_best" in result.parameters
    
    def test_evaluate_high_mean(self, mock_surrogate_model):
        """Test PI when predicted mean is much higher than current best."""
        # Mock predict to return high mean
        def high_mean_predict(parameters):
            pred = ModelPrediction(
                mean=2.0,  # Much higher than current best (0.8)
                variance=0.01,
                std=0.1,
                model_type=ModelType.GAUSSIAN_PROCESS,
            )
            return pred
        
        mock_surrogate_model.predict = high_mean_predict
        pi = ProbabilityOfImprovement(mock_surrogate_model, xi=0.01)
        
        result = pi.evaluate({"x1": 0.0, "x2": 0.0})
        assert result.value > 0.9  # Should have high probability of improvement
    
    def test_evaluate_low_mean(self, mock_surrogate_model):
        """Test PI when predicted mean is lower than current best."""
        # Mock predict to return low mean
        def low_mean_predict(parameters):
            pred = ModelPrediction(
                mean=0.2,  # Much lower than current best (0.8)
                variance=0.01,
                std=0.1,
                model_type=ModelType.GAUSSIAN_PROCESS,
            )
            return pred
        
        mock_surrogate_model.predict = low_mean_predict
        pi = ProbabilityOfImprovement(mock_surrogate_model, xi=0.01)
        
        result = pi.evaluate({"x1": 0.0, "x2": 0.0})
        assert result.value < 0.1  # Should have low probability of improvement
    
    def test_optimize_candidates(self, mock_surrogate_model):
        """Test PI optimization."""
        pi = ProbabilityOfImprovement(mock_surrogate_model, xi=0.01)
        
        candidates = pi.optimize(n_candidates=2, n_restarts=3)
        
        assert isinstance(candidates, list)
        assert len(candidates) == 2
        for candidate in candidates:
            assert isinstance(candidate, dict)
            assert "x1" in candidate
            assert "x2" in candidate


class TestAcquisitionFunctionCommon:
    """Test common functionality across acquisition functions."""
    
    def test_array_to_param_dict(self, mock_surrogate_model):
        """Test conversion from array to parameter dictionary."""
        ei = ExpectedImprovement(mock_surrogate_model)
        
        x = np.array([1.5, -2.3])
        param_dict = ei._array_to_param_dict(x)
        
        assert isinstance(param_dict, dict)
        assert param_dict["x1"] == 1.5
        assert param_dict["x2"] == -2.3
    
    def test_random_sample(self, mock_surrogate_model):
        """Test random sampling within bounds."""
        ei = ExpectedImprovement(mock_surrogate_model)
        
        bounds = [(-2, 2), (-3, 3)]
        sample = ei._random_sample(bounds)
        
        assert isinstance(sample, dict)
        assert "x1" in sample
        assert "x2" in sample
        assert -2 <= sample["x1"] <= 2
        assert -3 <= sample["x2"] <= 3
    
    def test_optimization_timeout_handling(self, mock_surrogate_model):
        """Test that optimization handles timeouts gracefully."""
        # This test would require mocking the optimization to timeout
        # For now, just test that optimization completes in reasonable time
        ei = ExpectedImprovement(mock_surrogate_model)
        
        import time
        start_time = time.time()
        candidates = ei.optimize(n_candidates=1, n_restarts=2)
        end_time = time.time()
        
        assert len(candidates) == 1
        assert end_time - start_time < 10  # Should complete within 10 seconds
