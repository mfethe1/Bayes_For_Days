"""
Unit tests for Gaussian Process surrogate models.

Tests the GaussianProcessModel implementation including:
- Model initialization and configuration
- Data fitting and training
- Prediction with uncertainty quantification
- Model persistence (save/load)
- Error handling and edge cases
"""

import pytest
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
import tempfile

from bayes_for_days.models.gaussian_process import (
    GaussianProcessModel,
    SVGPModel,
    FocalizedLoss,
)
from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    Parameter,
    ParameterType,
    ModelPrediction,
    ModelType,
)


@pytest.fixture
def sample_parameter_space():
    """Create a sample parameter space for testing."""
    parameters = [
        Parameter(name="temperature", type=ParameterType.CONTINUOUS, bounds=(20, 100)),
        Parameter(name="pressure", type=ParameterType.CONTINUOUS, bounds=(1, 10)),
        Parameter(name="catalyst", type=ParameterType.CATEGORICAL, categories=["A", "B", "C"]),
    ]
    return ParameterSpace(parameters=parameters)


@pytest.fixture
def sample_experiment_data():
    """Create sample experimental data for testing."""
    np.random.seed(42)
    data = []
    
    for i in range(50):
        temp = np.random.uniform(20, 100)
        pressure = np.random.uniform(1, 10)
        catalyst = np.random.choice(["A", "B", "C"])
        
        # Simple synthetic function
        yield_val = (
            0.5 + 0.3 * (temp / 100) + 0.2 * (pressure / 10) + 
            0.1 * (ord(catalyst) - ord("A")) + np.random.normal(0, 0.05)
        )
        
        point = ExperimentPoint(
            id=f"exp_{i:03d}",
            parameters={"temperature": temp, "pressure": pressure, "catalyst": catalyst},
            objectives={"yield": max(0, min(1, yield_val))},
            timestamp=datetime.now(),
            is_feasible=True,
        )
        data.append(point)
    
    return data


@pytest.fixture
def small_experiment_data():
    """Create small dataset for quick testing."""
    data = [
        ExperimentPoint(
            parameters={"temperature": 50.0, "pressure": 5.0, "catalyst": "A"},
            objectives={"yield": 0.7},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"temperature": 80.0, "pressure": 8.0, "catalyst": "B"},
            objectives={"yield": 0.9},
            is_feasible=True,
        ),
        ExperimentPoint(
            parameters={"temperature": 30.0, "pressure": 2.0, "catalyst": "C"},
            objectives={"yield": 0.4},
            is_feasible=True,
        ),
    ]
    return data


class TestGaussianProcessModel:
    """Test suite for GaussianProcessModel."""
    
    def test_initialization(self, sample_parameter_space):
        """Test model initialization with various configurations."""
        # Test basic initialization
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            n_inducing_points=50,
            kernel_type="matern",
            ard=True,
        )
        
        assert model.parameter_space == sample_parameter_space
        assert model.n_inducing_points == 50
        assert model.kernel_type == "matern"
        assert model.ard is True
        assert not model.is_fitted
        assert model.input_dim == 3  # temperature, pressure, catalyst
    
    def test_initialization_with_gpu(self, sample_parameter_space):
        """Test initialization with GPU device if available."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            device=device,
        )
        
        assert str(model.device) == device
    
    def test_initialization_with_focus_regions(self, sample_parameter_space):
        """Test initialization with focus regions for focalized loss."""
        focus_regions = np.array([[50.0, 5.0, 0], [80.0, 8.0, 1]])  # Encoded categorical
        
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            focus_regions=focus_regions,
        )
        
        assert model.focalized_loss is not None
    
    def test_invalid_kernel_type(self, sample_parameter_space):
        """Test that invalid kernel type raises error."""
        with pytest.raises(ValueError, match="Unsupported kernel type"):
            GaussianProcessModel(
                parameter_space=sample_parameter_space,
                kernel_type="invalid_kernel",
            )
    
    def test_fit_with_valid_data(self, sample_parameter_space, sample_experiment_data):
        """Test fitting model with valid experimental data."""
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            n_inducing_points=20,  # Smaller for faster testing
        )
        
        # Should not raise any exceptions
        model.fit(sample_experiment_data)
        
        assert model.is_fitted
        assert model.model is not None
        assert model.likelihood is not None
        assert model.train_x is not None
        assert model.train_y is not None
        assert len(model.training_data) == len(sample_experiment_data)
    
    def test_fit_with_empty_data(self, sample_parameter_space):
        """Test that fitting with empty data raises error."""
        model = GaussianProcessModel(parameter_space=sample_parameter_space)
        
        with pytest.raises(ValueError, match="No data provided"):
            model.fit([])
    
    def test_predict_single_point(self, sample_parameter_space, small_experiment_data):
        """Test prediction for a single parameter set."""
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            n_inducing_points=10,
        )
        model.fit(small_experiment_data)
        
        test_params = {"temperature": 60.0, "pressure": 6.0, "catalyst": "B"}
        prediction = model.predict(test_params)
        
        assert isinstance(prediction, ModelPrediction)
        assert prediction.model_type == ModelType.GAUSSIAN_PROCESS
        assert isinstance(prediction.mean, float)
        assert isinstance(prediction.variance, float)
        assert isinstance(prediction.std, float)
        assert prediction.variance >= 0
        assert prediction.std >= 0
        assert prediction.confidence_interval is not None
        assert len(prediction.confidence_interval) == 2
    
    def test_predict_multiple_points(self, sample_parameter_space, small_experiment_data):
        """Test prediction for multiple parameter sets."""
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            n_inducing_points=10,
        )
        model.fit(small_experiment_data)
        
        test_params_list = [
            {"temperature": 60.0, "pressure": 6.0, "catalyst": "A"},
            {"temperature": 70.0, "pressure": 7.0, "catalyst": "B"},
        ]
        predictions = model.predict(test_params_list)
        
        assert isinstance(predictions, list)
        assert len(predictions) == 2
        for pred in predictions:
            assert isinstance(pred, ModelPrediction)
            assert pred.model_type == ModelType.GAUSSIAN_PROCESS
    
    def test_predict_without_fitting(self, sample_parameter_space):
        """Test that prediction without fitting raises error."""
        model = GaussianProcessModel(parameter_space=sample_parameter_space)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict({"temperature": 50.0, "pressure": 5.0, "catalyst": "A"})
    
    def test_update_model(self, sample_parameter_space, small_experiment_data):
        """Test updating model with new data."""
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            n_inducing_points=10,
        )
        
        # Initial fit
        initial_data = small_experiment_data[:2]
        model.fit(initial_data)
        assert len(model.training_data) == 2
        
        # Update with new data
        new_data = small_experiment_data[2:]
        model.update(new_data)
        assert len(model.training_data) == 3
    
    def test_get_model_info(self, sample_parameter_space, small_experiment_data):
        """Test getting model information."""
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            n_inducing_points=10,
        )
        
        # Before fitting
        info = model.get_model_info()
        assert "type" in info
        assert "is_fitted" in info
        assert info["is_fitted"] is False
        
        # After fitting
        model.fit(small_experiment_data)
        info = model.get_model_info()
        assert info["is_fitted"] is True
        assert "lengthscales" in info
        assert "outputscale" in info
        assert "noise" in info
    
    def test_save_and_load_model(self, sample_parameter_space, small_experiment_data):
        """Test saving and loading trained model."""
        model = GaussianProcessModel(
            parameter_space=sample_parameter_space,
            n_inducing_points=10,
        )
        model.fit(small_experiment_data)
        
        # Test prediction before save
        test_params = {"temperature": 60.0, "pressure": 6.0, "catalyst": "A"}
        pred_before = model.predict(test_params)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
            model.save_model(tmp_file.name)
            
            # Create new model and load
            new_model = GaussianProcessModel(parameter_space=sample_parameter_space)
            new_model.load_model(tmp_file.name)
            
            # Test that loaded model gives same predictions
            pred_after = new_model.predict(test_params)
            
            assert abs(pred_before.mean - pred_after.mean) < 1e-6
            assert abs(pred_before.variance - pred_after.variance) < 1e-6
            
            # Clean up
            Path(tmp_file.name).unlink()
    
    def test_save_unfitted_model_raises_error(self, sample_parameter_space):
        """Test that saving unfitted model raises error."""
        model = GaussianProcessModel(parameter_space=sample_parameter_space)
        
        with pytest.raises(ValueError, match="Cannot save unfitted model"):
            model.save_model("dummy_path.pt")


class TestSVGPModel:
    """Test suite for SVGPModel."""
    
    def test_svgp_initialization(self):
        """Test SVGP model initialization."""
        inducing_points = torch.randn(20, 2)
        model = SVGPModel(
            inducing_points=inducing_points,
            input_dim=2,
            kernel_type="matern",
            ard=True,
        )
        
        assert model.input_dim == 2
        assert model.kernel_type == "matern"
        assert model.ard is True
    
    def test_svgp_forward_pass(self):
        """Test forward pass through SVGP model."""
        inducing_points = torch.randn(10, 2)
        model = SVGPModel(
            inducing_points=inducing_points,
            input_dim=2,
        )
        
        # Test forward pass
        test_x = torch.randn(5, 2)
        output = model(test_x)
        
        assert hasattr(output, "mean")
        assert hasattr(output, "covariance_matrix")
        assert output.mean.shape == (5,)


class TestFocalizedLoss:
    """Test suite for FocalizedLoss."""
    
    def test_focalized_loss_initialization(self):
        """Test focalized loss initialization."""
        focus_regions = torch.randn(3, 2)
        loss_fn = FocalizedLoss(focus_regions=focus_regions, focus_weight=2.0)
        
        assert loss_fn.focus_regions is not None
        assert loss_fn.focus_weight == 2.0
    
    def test_compute_focus_weights(self):
        """Test computation of focus weights."""
        focus_regions = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        loss_fn = FocalizedLoss(focus_regions=focus_regions)
        
        test_points = torch.tensor([[0.1, 0.1], [2.0, 2.0]])
        weights = loss_fn.compute_focus_weights(test_points)
        
        assert weights.shape == (2,)
        assert weights[0] > weights[1]  # Closer point should have higher weight
    
    def test_focalized_loss_call(self):
        """Test calling focalized loss function."""
        focus_regions = torch.tensor([[0.0, 0.0]])
        loss_fn = FocalizedLoss(focus_regions=focus_regions)
        
        elbo = torch.tensor([1.0, 2.0])
        x = torch.tensor([[0.1, 0.1], [1.0, 1.0]])
        
        weighted_loss = loss_fn(elbo, x)
        assert isinstance(weighted_loss, torch.Tensor)
        assert weighted_loss.dim() == 0  # Scalar output
