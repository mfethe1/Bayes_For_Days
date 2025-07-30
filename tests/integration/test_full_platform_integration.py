"""
Full platform integration tests for Bayes For Days.

This module tests the complete integration of all platform components:
- Multi-objective optimization with experimental design
- Ensemble models with hybrid optimization strategies
- Laboratory constraints with cost optimization
- End-to-end workflow validation
- Performance and scalability testing
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import time
from typing import Dict, List, Any

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
from bayes_for_days.optimization.hybrid_strategies import (
    HybridOptimizer,
    HybridConfig,
    BayesianStrategy,
    GeneticStrategy,
    LocalStrategy,
)
from bayes_for_days.experimental_design.variables import (
    ExperimentalVariable,
    VariableType,
    UnitType,
    VariationRange,
)
from bayes_for_days.experimental_design.design_strategies import (
    DOptimalDesign,
    LatinHypercubeSampling,
)
from bayes_for_days.experimental_design.laboratory_constraints import (
    LaboratoryConstraintManager,
    MixtureConstraint,
    VolumeConstraint,
)
from bayes_for_days.experimental_design.cost_optimization import (
    CostAwareDesign,
    ReagentCostFunction,
    ReagentCost,
)


class TestFullPlatformIntegration:
    """Integration tests for the complete Bayes For Days platform."""
    
    def create_test_parameter_space(self) -> ParameterSpace:
        """Create a comprehensive parameter space for testing."""
        parameters = [
            Parameter(name="temperature", type=ParameterType.CONTINUOUS, bounds=(60, 100)),
            Parameter(name="pressure", type=ParameterType.CONTINUOUS, bounds=(1, 10)),
            Parameter(name="catalyst_conc", type=ParameterType.CONTINUOUS, bounds=(0.1, 5.0)),
            Parameter(name="solvent", type=ParameterType.CATEGORICAL, categories=["water", "ethanol", "acetone"]),
        ]
        return ParameterSpace(parameters=parameters)
    
    def create_test_experimental_variables(self) -> List[ExperimentalVariable]:
        """Create experimental variables for design testing."""
        variables = [
            ExperimentalVariable(
                name="temperature",
                variable_type=VariableType.CONTINUOUS,
                unit=UnitType.TEMPERATURE_CELSIUS,
                baseline_value=80.0,
                variation_range=VariationRange(
                    variation_type="absolute",
                    absolute_min=60.0,
                    absolute_max=100.0
                )
            ),
            ExperimentalVariable(
                name="pressure",
                variable_type=VariableType.CONTINUOUS,
                unit=UnitType.PRESSURE_BAR,
                baseline_value=5.0,
                variation_range=VariationRange(
                    variation_type="absolute",
                    absolute_min=1.0,
                    absolute_max=10.0
                )
            ),
            ExperimentalVariable(
                name="catalyst_conc",
                variable_type=VariableType.CONTINUOUS,
                unit=UnitType.MILLIMOLAR,
                baseline_value=2.0,
                variation_range=VariationRange(
                    variation_type="absolute",
                    absolute_min=0.1,
                    absolute_max=5.0
                )
            ),
        ]
        return variables
    
    def create_test_objective_function(self) -> callable:
        """Create a multi-modal test objective function."""
        def objective(params: Dict[str, float]) -> float:
            temp = params.get("temperature", 80)
            pressure = params.get("pressure", 5)
            catalyst = params.get("catalyst_conc", 2)
            
            # Multi-modal function with noise
            term1 = -0.1 * (temp - 85)**2
            term2 = -0.2 * (pressure - 6)**2
            term3 = -0.5 * (catalyst - 3)**2
            
            # Add interaction terms
            interaction = 0.01 * temp * pressure * catalyst
            
            # Add noise
            noise = np.random.normal(0, 0.1)
            
            return term1 + term2 + term3 + interaction + noise
        
        return objective
    
    def create_multi_objective_function(self) -> callable:
        """Create a multi-objective test function."""
        def multi_objective(params: Dict[str, float]) -> Dict[str, float]:
            temp = params.get("temperature", 80)
            pressure = params.get("pressure", 5)
            catalyst = params.get("catalyst_conc", 2)
            
            # Objective 1: Yield (maximize)
            yield_obj = -(temp - 90)**2 - (pressure - 7)**2 + catalyst * 2
            
            # Objective 2: Cost (minimize, so negate for maximization)
            cost_obj = -(temp * 0.1 + pressure * 0.2 + catalyst * 10)
            
            return {
                "yield": yield_obj,
                "cost": cost_obj
            }
        
        return multi_objective
    
    def test_basic_bayesian_optimization_integration(self):
        """Test basic Bayesian optimization with GP model."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective_function()
        
        # Create GP model
        gp_model = GaussianProcessModel(parameter_space=parameter_space)
        
        # Create acquisition function
        acquisition = ExpectedImprovement(surrogate_model=gp_model)
        
        # Generate initial data
        initial_data = []
        for i in range(5):
            params = {
                "temperature": np.random.uniform(60, 100),
                "pressure": np.random.uniform(1, 10),
                "catalyst_conc": np.random.uniform(0.1, 5.0),
                "solvent": np.random.choice(["water", "ethanol", "acetone"])
            }
            
            obj_value = objective(params)
            
            experiment_point = ExperimentPoint(
                parameters=params,
                objectives={"objective": obj_value},
                is_feasible=True
            )
            initial_data.append(experiment_point)
        
        # Fit model
        gp_model.fit(initial_data)
        
        # Test prediction
        test_params = {"temperature": 85, "pressure": 6, "catalyst_conc": 3, "solvent": "water"}
        prediction = gp_model.predict(test_params)
        
        assert prediction is not None
        assert hasattr(prediction, 'mean')
        assert hasattr(prediction, 'std')
        
        # Test acquisition function
        acq_value = acquisition.evaluate(test_params)
        assert acq_value is not None
        assert acq_value.value >= 0  # EI should be non-negative
    
    def test_ensemble_model_integration(self):
        """Test ensemble model with multiple base models."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective_function()
        
        # Create multiple base models
        base_models = [
            GaussianProcessModel(parameter_space=parameter_space, kernel_type="matern"),
            GaussianProcessModel(parameter_space=parameter_space, kernel_type="rbf"),
        ]
        
        # Create ensemble model
        ensemble_model = EnsembleSurrogateModel(
            base_models=base_models,
            parameter_space=parameter_space,
            ensemble_method=SimpleAveraging()
        )
        
        # Generate training data
        training_data = []
        for i in range(10):
            params = {
                "temperature": np.random.uniform(60, 100),
                "pressure": np.random.uniform(1, 10),
                "catalyst_conc": np.random.uniform(0.1, 5.0),
                "solvent": np.random.choice(["water", "ethanol", "acetone"])
            }
            
            obj_value = objective(params)
            
            experiment_point = ExperimentPoint(
                parameters=params,
                objectives={"objective": obj_value},
                is_feasible=True
            )
            training_data.append(experiment_point)
        
        # Fit ensemble model
        ensemble_model.fit(training_data)
        
        # Test prediction
        test_params = {"temperature": 85, "pressure": 6, "catalyst_conc": 3, "solvent": "water"}
        prediction = ensemble_model.predict(test_params)
        
        assert prediction is not None
        assert hasattr(prediction, 'mean')
        assert hasattr(prediction, 'std')
        assert prediction.model_type.value == "ensemble"
        
        # Check that ensemble combines multiple models
        model_info = ensemble_model.get_model_info()
        assert model_info['n_base_models'] == 2
        assert len(model_info['model_weights']) == 2
    
    def test_multi_objective_optimization_integration(self):
        """Test multi-objective optimization with NSGA-II."""
        parameter_space = self.create_test_parameter_space()
        multi_objective = self.create_multi_objective_function()
        
        # Create NSGA-II optimizer
        optimizer = NSGAIIOptimizer(
            parameter_space=parameter_space,
            population_size=20,
            max_generations=5,  # Small for testing
            crossover_prob=0.9,
            mutation_prob=0.1
        )
        
        # Define objective function for NSGA-II
        def nsga_objective(params):
            objectives = multi_objective(params)
            return [objectives["yield"], objectives["cost"]]
        
        # Run optimization
        result = optimizer.optimize(
            objective_function=nsga_objective,
            n_objectives=2
        )
        
        assert result is not None
        assert result.pareto_front is not None
        assert len(result.pareto_front) > 0
        assert result.n_function_evaluations > 0
        
        # Check Pareto front quality
        pareto_front = result.pareto_front
        assert len(pareto_front) <= 20  # Should not exceed population size
        
        # Verify non-domination
        for i, ind1 in enumerate(pareto_front):
            for j, ind2 in enumerate(pareto_front):
                if i != j:
                    # No individual should dominate another in Pareto front
                    assert not ind1.dominates(ind2)
    
    def test_experimental_design_integration(self):
        """Test experimental design strategies integration."""
        variables = self.create_test_experimental_variables()
        
        # Test D-optimal design
        d_optimal = DOptimalDesign(variables, candidate_size=100)
        d_design = d_optimal.generate_design(n_experiments=10)
        
        assert d_design.shape == (10, 3)  # 10 experiments, 3 variables
        assert np.all(d_design[:, 0] >= 60) and np.all(d_design[:, 0] <= 100)  # Temperature bounds
        assert np.all(d_design[:, 1] >= 1) and np.all(d_design[:, 1] <= 10)    # Pressure bounds
        assert np.all(d_design[:, 2] >= 0.1) and np.all(d_design[:, 2] <= 5.0) # Catalyst bounds
        
        # Test Latin Hypercube Sampling
        lhs = LatinHypercubeSampling(variables, criterion="maximin")
        lhs_design = lhs.generate_design(n_experiments=8)
        
        assert lhs_design.shape == (8, 3)
        
        # Check space-filling property (simplified)
        min_distances = []
        for i in range(len(lhs_design)):
            distances = []
            for j in range(len(lhs_design)):
                if i != j:
                    dist = np.linalg.norm(lhs_design[i] - lhs_design[j])
                    distances.append(dist)
            if distances:
                min_distances.append(min(distances))
        
        # LHS should have reasonable space-filling
        assert np.mean(min_distances) > 0
    
    def test_laboratory_constraints_integration(self):
        """Test laboratory constraints with experimental design."""
        variables = self.create_test_experimental_variables()
        
        # Create mixture constraint (components must sum to 100%)
        mixture_constraint = MixtureConstraint(
            component_variables=["temperature", "pressure"],  # Dummy mixture for testing
            target_sum=100.0,
            tolerance=1e-3
        )
        
        # Create volume constraint
        volume_constraint = VolumeConstraint(
            volume_variables=["catalyst_conc"],
            max_total_volume=10.0,
            min_total_volume=1.0
        )
        
        # Create constraint manager
        constraint_manager = LaboratoryConstraintManager([
            mixture_constraint,
            volume_constraint
        ])
        
        # Test constraint checking
        test_params = {
            "temperature": 50.0,
            "pressure": 50.0,  # Sum = 100, satisfies mixture constraint
            "catalyst_conc": 5.0  # Within volume constraint
        }
        
        assert constraint_manager.check_feasibility(test_params)
        
        # Test infeasible parameters
        infeasible_params = {
            "temperature": 60.0,
            "pressure": 60.0,  # Sum = 120, violates mixture constraint
            "catalyst_conc": 15.0  # Exceeds volume constraint
        }
        
        assert not constraint_manager.check_feasibility(infeasible_params)
        
        # Test violation calculation
        violations = constraint_manager.get_constraint_violations(infeasible_params)
        assert len(violations) > 0
    
    def test_cost_aware_design_integration(self):
        """Test cost-aware experimental design."""
        variables = self.create_test_experimental_variables()
        
        # Create reagent costs
        reagent_costs = {
            "temperature": ReagentCost(
                name="heating_cost",
                cost_per_unit=0.1,  # $0.1 per degree
                unit="celsius",
                available_quantity=1000
            ),
            "catalyst_conc": ReagentCost(
                name="catalyst",
                cost_per_unit=50.0,  # $50 per mM
                unit="mM",
                available_quantity=100
            )
        }
        
        # Create cost function
        consumption_rates = {
            "temperature": 1.0,  # 1 unit per degree
            "catalyst_conc": 1.0  # 1 unit per mM
        }
        
        reagent_cost_function = ReagentCostFunction(
            reagent_costs=reagent_costs,
            consumption_rates=consumption_rates
        )
        
        # Create cost-aware design
        cost_aware_design = CostAwareDesign(
            variables=variables,
            cost_functions=[reagent_cost_function],
            cost_weight=0.5,
            max_budget=1000.0
        )
        
        # Generate design
        design = cost_aware_design.generate_design(n_experiments=5)
        
        assert design.shape == (5, 3)
        
        # Analyze design
        analysis = cost_aware_design.get_design_analysis(design)
        
        assert 'total_cost' in analysis
        assert 'efficiency' in analysis
        assert 'within_budget' in analysis
        assert analysis['within_budget']  # Should be within budget
        assert analysis['total_cost'] <= 1000.0
    
    def test_hybrid_optimization_integration(self):
        """Test hybrid optimization strategies."""
        parameter_space = self.create_test_parameter_space()
        objective = self.create_test_objective_function()
        
        # Create base models for Bayesian strategy
        gp_model = GaussianProcessModel(parameter_space=parameter_space)
        acquisition = ExpectedImprovement(surrogate_model=gp_model)
        
        # Create optimization strategies
        strategies = {
            "bayesian": BayesianStrategy(gp_model, acquisition),
            "genetic": GeneticStrategy(popsize=10),
            "local": LocalStrategy(method="L-BFGS-B")
        }
        
        # Create hybrid configuration
        config = HybridConfig(
            primary_strategy="bayesian",
            secondary_strategies=["genetic", "local"],
            max_total_evaluations=30,  # Small for testing
            switch_after_iterations=2,
            evaluation_budget_ratios={
                "bayesian": 0.5,
                "genetic": 0.3,
                "local": 0.2
            }
        )
        
        # Create hybrid optimizer
        hybrid_optimizer = HybridOptimizer(strategies, config)
        
        # Run optimization
        result = hybrid_optimizer.optimize(objective, parameter_space)
        
        assert result is not None
        assert result.best_parameters is not None
        assert result.best_objective_value is not None
        assert result.n_function_evaluations <= config.max_total_evaluations
        
        # Check that multiple strategies were used
        metadata = result.metadata
        assert 'hybrid_optimizer' in metadata
        assert 'strategy_history' in metadata
        assert len(metadata['strategies_used']) > 1  # Should use multiple strategies
    
    @pytest.mark.slow
    def test_end_to_end_workflow(self):
        """Test complete end-to-end optimization workflow."""
        # This test combines multiple components in a realistic workflow
        
        # 1. Define problem
        parameter_space = self.create_test_parameter_space()
        variables = self.create_test_experimental_variables()
        objective = self.create_test_objective_function()
        
        # 2. Generate initial experimental design
        lhs = LatinHypercubeSampling(variables, criterion="maximin")
        initial_design = lhs.generate_design(n_experiments=8)
        
        # Convert design to experiment points
        initial_experiments = []
        for i, design_point in enumerate(initial_design):
            params = {
                "temperature": design_point[0],
                "pressure": design_point[1],
                "catalyst_conc": design_point[2],
                "solvent": "water"  # Default categorical value
            }
            
            obj_value = objective(params)
            
            experiment_point = ExperimentPoint(
                parameters=params,
                objectives={"objective": obj_value},
                is_feasible=True,
                metadata={"design_method": "lhs", "experiment_id": i}
            )
            initial_experiments.append(experiment_point)
        
        # 3. Create ensemble model
        base_models = [
            GaussianProcessModel(parameter_space=parameter_space, kernel_type="matern"),
            GaussianProcessModel(parameter_space=parameter_space, kernel_type="rbf"),
        ]
        
        ensemble_model = EnsembleSurrogateModel(
            base_models=base_models,
            parameter_space=parameter_space,
            ensemble_method=SimpleAveraging()
        )
        
        # 4. Fit ensemble model
        ensemble_model.fit(initial_experiments)
        
        # 5. Create acquisition function
        acquisition = ExpectedImprovement(surrogate_model=ensemble_model)
        
        # 6. Run adaptive optimization
        all_experiments = initial_experiments.copy()
        
        for iteration in range(5):  # 5 adaptive iterations
            # Select next experiment
            candidates = acquisition.optimize(n_candidates=1)
            
            if candidates:
                next_params = candidates[0]
                next_obj_value = objective(next_params)
                
                next_experiment = ExperimentPoint(
                    parameters=next_params,
                    objectives={"objective": next_obj_value},
                    is_feasible=True,
                    metadata={"iteration": iteration, "method": "adaptive"}
                )
                
                all_experiments.append(next_experiment)
                
                # Update model
                ensemble_model.fit(all_experiments)
        
        # 7. Analyze results
        best_experiment = max(
            all_experiments,
            key=lambda x: list(x.objectives.values())[0]
        )
        
        assert len(all_experiments) == 8 + 5  # Initial + adaptive
        assert best_experiment is not None
        assert best_experiment.objectives is not None
        
        # Check improvement over initial design
        initial_best = max(
            initial_experiments,
            key=lambda x: list(x.objectives.values())[0]
        )
        
        best_value = list(best_experiment.objectives.values())[0]
        initial_best_value = list(initial_best.objectives.values())[0]
        
        # Should show some improvement (or at least not get worse)
        assert best_value >= initial_best_value - 0.5  # Allow for noise
        
        print(f"End-to-end workflow completed:")
        print(f"  Initial best: {initial_best_value:.4f}")
        print(f"  Final best: {best_value:.4f}")
        print(f"  Improvement: {best_value - initial_best_value:.4f}")
        print(f"  Total experiments: {len(all_experiments)}")
    
    def test_performance_scalability(self):
        """Test platform performance with larger problems."""
        # Create larger parameter space
        parameters = [
            Parameter(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(-5, 5))
            for i in range(10)  # 10-dimensional problem
        ]
        parameter_space = ParameterSpace(parameters=parameters)
        
        # Simple quadratic objective
        def objective(params):
            x = np.array([params[f"x{i}"] for i in range(10)])
            return -np.sum(x**2)  # Negative for maximization
        
        # Create GP model
        gp_model = GaussianProcessModel(parameter_space=parameter_space)
        
        # Generate training data
        training_data = []
        for i in range(50):  # Larger training set
            params = {f"x{j}": np.random.uniform(-5, 5) for j in range(10)}
            obj_value = objective(params)
            
            experiment_point = ExperimentPoint(
                parameters=params,
                objectives={"objective": obj_value},
                is_feasible=True
            )
            training_data.append(experiment_point)
        
        # Time model fitting
        start_time = time.time()
        gp_model.fit(training_data)
        fit_time = time.time() - start_time
        
        # Time prediction
        test_params = {f"x{i}": 0.0 for i in range(10)}
        
        start_time = time.time()
        prediction = gp_model.predict(test_params)
        predict_time = time.time() - start_time
        
        # Performance assertions
        assert fit_time < 30.0  # Should fit within 30 seconds
        assert predict_time < 1.0  # Should predict within 1 second
        assert prediction is not None
        
        print(f"Performance test results:")
        print(f"  Model fitting time: {fit_time:.2f} seconds")
        print(f"  Prediction time: {predict_time:.4f} seconds")
        print(f"  Training data size: {len(training_data)}")
        print(f"  Parameter space dimension: {len(parameters)}")


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
