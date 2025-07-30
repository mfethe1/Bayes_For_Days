#!/usr/bin/env python3
"""
Uncertainty-Guided Resource Allocation Demonstration

This script demonstrates the revolutionary uncertainty-guided resource allocation
capabilities of Bayes For Days, showing how it can intelligently allocate
experimental resources based on uncertainty quantification and value of information.

This is a capability that no existing experimental design tool provides.

Example Scenario: Materials Science Optimization
- Limited budget for experiments
- Different experiment types with different costs and information value
- Need to balance exploration vs exploitation under resource constraints
- Uncertainty-aware decision making for optimal resource allocation

The system automatically:
1. Quantifies uncertainty in all experimental parameters
2. Calculates value of information for potential experiments
3. Optimizes resource allocation to maximize expected utility
4. Provides risk-adjusted recommendations
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Bayes For Days components
import sys
sys.path.insert(0, 'src')

from bayes_for_days.optimization.uncertainty_guided import (
    UncertaintyQuantifier,
    ValueOfInformationCalculator,
    ResourceAllocationOptimizer,
    ResourceConstraint,
    UncertaintyMetrics,
    ValueOfInformation
)
from bayes_for_days.models.gaussian_process import GaussianProcessModel
from bayes_for_days.core.types import (
    ParameterSpace,
    Parameter,
    ParameterType,
    ExperimentPoint
)

def create_materials_science_scenario():
    """
    Create a realistic materials science optimization scenario.
    
    We're optimizing a new alloy with 3 parameters:
    - temperature: Processing temperature (800-1200°C)
    - pressure: Processing pressure (1-10 atm)
    - composition: Alloy composition ratio (0.1-0.9)
    
    Objective: Maximize tensile strength (higher is better)
    
    Different experiment types:
    - Computational simulation: Low cost, medium accuracy
    - Lab-scale synthesis: Medium cost, high accuracy
    - Pilot-scale production: High cost, very high accuracy
    """
    
    # Define parameter space
    parameters = [
        Parameter(
            name="temperature",
            type=ParameterType.CONTINUOUS,
            bounds=(800.0, 1200.0),
            description="Processing temperature in Celsius"
        ),
        Parameter(
            name="pressure",
            type=ParameterType.CONTINUOUS,
            bounds=(1.0, 10.0),
            description="Processing pressure in atmospheres"
        ),
        Parameter(
            name="composition",
            type=ParameterType.CONTINUOUS,
            bounds=(0.1, 0.9),
            description="Alloy composition ratio"
        )
    ]
    
    parameter_space = ParameterSpace(parameters=parameters)
    
    # Define resource constraints
    constraints = [
        ResourceConstraint(
            name="budget",
            total_available=50000.0,  # $50,000 total budget
            cost_per_experiment={
                "computational": 100.0,    # $100 per simulation
                "lab_scale": 2000.0,       # $2,000 per lab experiment
                "pilot_scale": 15000.0     # $15,000 per pilot experiment
            },
            description="Total experimental budget in USD"
        ),
        ResourceConstraint(
            name="time",
            total_available=30.0,  # 30 days available
            cost_per_experiment={
                "computational": 0.5,     # 0.5 days per simulation
                "lab_scale": 3.0,         # 3 days per lab experiment
                "pilot_scale": 10.0       # 10 days per pilot experiment
            },
            description="Total time available in days"
        )
    ]
    
    return parameter_space, constraints

def materials_tensile_strength(params: Dict[str, float], experiment_type: str = "lab_scale") -> float:
    """
    Simulate tensile strength measurement for materials optimization.
    
    This function simulates a realistic materials science objective where:
    - Higher tensile strength is better (maximize)
    - Different experiment types have different noise levels and biases
    - The true optimum is around temp=1000°C, pressure=5 atm, composition=0.6
    """
    temp = params["temperature"]
    pressure = params["pressure"]
    composition = params["composition"]
    
    # True underlying function (unknown to optimizer)
    # Optimal around temp=1000, pressure=5, composition=0.6
    true_strength = (
        100 +  # Base strength
        50 * np.exp(-0.001 * (temp - 1000)**2) +  # Temperature effect
        30 * np.exp(-0.5 * (pressure - 5)**2) +   # Pressure effect
        40 * np.exp(-2 * (composition - 0.6)**2) + # Composition effect
        10 * np.sin(0.01 * temp) * np.cos(pressure) +  # Interaction terms
        5 * composition * (1 - composition) * temp / 1000  # Complex interaction
    )
    
    # Add experiment-type-specific noise and bias
    if experiment_type == "computational":
        # High noise, systematic bias
        noise = np.random.normal(0, 15.0)
        bias = -10  # Computational methods underestimate strength
        measured_strength = true_strength + bias + noise
        
    elif experiment_type == "lab_scale":
        # Medium noise, small bias
        noise = np.random.normal(0, 8.0)
        bias = -2
        measured_strength = true_strength + bias + noise
        
    elif experiment_type == "pilot_scale":
        # Low noise, minimal bias
        noise = np.random.normal(0, 3.0)
        bias = 0
        measured_strength = true_strength + bias + noise
        
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    # Ensure positive strength
    return max(10.0, measured_strength)

def generate_initial_data(parameter_space: ParameterSpace, n_points: int = 8) -> List[ExperimentPoint]:
    """Generate initial experimental data."""
    initial_data = []
    
    # Use Latin Hypercube Sampling for initial points
    from bayes_for_days.utils.sampling import latin_hypercube_sampling_parameter_space
    
    param_samples = latin_hypercube_sampling_parameter_space(
        parameter_space=parameter_space,
        n_samples=n_points,
        random_seed=42
    )
    
    # Evaluate at different experiment types (mostly lab-scale for initial data)
    experiment_types = ["computational", "lab_scale", "lab_scale", "lab_scale", 
                       "lab_scale", "lab_scale", "pilot_scale", "computational"]
    
    for i, param_dict in enumerate(param_samples):
        exp_type = experiment_types[i % len(experiment_types)]
        strength = materials_tensile_strength(param_dict, exp_type)
        
        experiment_point = ExperimentPoint(
            parameters=param_dict,
            objectives={"tensile_strength": strength},
            metadata={"experiment_type": exp_type, "iteration": -1}
        )
        
        initial_data.append(experiment_point)
    
    return initial_data

def run_uncertainty_guided_optimization():
    """Run the uncertainty-guided resource allocation demonstration."""
    
    print("=" * 80)
    print("🔬 UNCERTAINTY-GUIDED RESOURCE ALLOCATION")
    print("=" * 80)
    print()
    
    # Create scenario
    parameter_space, constraints = create_materials_science_scenario()
    
    print("📋 Optimization Scenario:")
    print("   Objective: Maximize tensile strength of new alloy")
    print("   Parameters: Temperature, Pressure, Composition")
    print("   Resource Constraints:")
    for constraint in constraints:
        print(f"     • {constraint.name}: {constraint.total_available} units available")
        for exp_type, cost in constraint.cost_per_experiment.items():
            print(f"       - {exp_type}: {cost} units per experiment")
    print()
    
    # Generate initial data
    print("🚀 Generating initial experimental data...")
    initial_data = generate_initial_data(parameter_space)
    
    print(f"Generated {len(initial_data)} initial experiments:")
    for i, point in enumerate(initial_data):
        exp_type = point.metadata.get("experiment_type", "unknown")
        strength = point.objectives["tensile_strength"]
        print(f"   {i+1}. {exp_type}: Strength = {strength:.1f} MPa")
    print()
    
    # Fit surrogate model
    print("🤖 Fitting surrogate model...")
    surrogate_model = GaussianProcessModel(parameter_space=parameter_space)
    surrogate_model.fit(initial_data)
    print("✅ Surrogate model fitted successfully")
    print()
    
    # Initialize uncertainty quantification
    print("📊 Initializing uncertainty quantification...")
    uncertainty_quantifier = UncertaintyQuantifier(surrogate_model)
    
    # Initialize VoI calculator
    voi_calculator = ValueOfInformationCalculator(
        surrogate_model=surrogate_model,
        uncertainty_quantifier=uncertainty_quantifier
    )
    
    # Initialize resource allocation optimizer
    resource_optimizer = ResourceAllocationOptimizer(
        voi_calculator=voi_calculator,
        constraints=constraints
    )
    print("✅ Uncertainty-guided system initialized")
    print()
    
    # Generate candidate experiments
    print("🎯 Generating candidate experiments...")
    n_candidates = 20
    
    # Use Latin Hypercube Sampling for candidates
    from bayes_for_days.utils.sampling import latin_hypercube_sampling_parameter_space
    
    candidate_params = latin_hypercube_sampling_parameter_space(
        parameter_space=parameter_space,
        n_samples=n_candidates,
        random_seed=123
    )
    
    # Create candidate experiments with different types and costs
    candidate_experiments = []
    experiment_types = ["computational", "lab_scale", "pilot_scale"]
    
    for i, params in enumerate(candidate_params):
        # Vary experiment types
        exp_type = experiment_types[i % len(experiment_types)]
        
        # Get costs for this experiment type
        costs = {}
        for constraint in constraints:
            costs[constraint.name] = constraint.cost_per_experiment[exp_type]
        
        candidate_experiments.append((params, costs))
    
    print(f"Generated {len(candidate_experiments)} candidate experiments")
    print()
    
    # Quantify uncertainty for a few example points
    print("📈 Uncertainty Analysis Examples:")
    example_points = candidate_params[:3]
    
    for i, params in enumerate(example_points):
        uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(params)
        
        print(f"   Point {i+1}: T={params['temperature']:.0f}°C, P={params['pressure']:.1f}atm, C={params['composition']:.2f}")
        print(f"     Total uncertainty: {uncertainty_metrics.total_uncertainty:.3f}")
        print(f"     Model uncertainty: {uncertainty_metrics.model_uncertainty:.3f}")
        print(f"     Noise uncertainty: {uncertainty_metrics.noise_uncertainty:.3f}")
        print(f"     Uncertainty score: {uncertainty_metrics.get_uncertainty_score():.3f}")
        print()
    
    # Optimize resource allocation
    print("🎯 Optimizing resource allocation...")
    current_best = max(initial_data, key=lambda p: p.objectives["tensile_strength"])
    
    selected_experiments = resource_optimizer.optimize_allocation(
        candidate_experiments=candidate_experiments,
        current_best=current_best,
        max_experiments=10
    )
    
    print(f"✅ Selected {len(selected_experiments)} optimal experiments")
    print()
    
    # Display results
    print("📊 OPTIMAL RESOURCE ALLOCATION:")
    total_budget_used = 0
    total_time_used = 0
    
    for i, (params, costs, expected_value) in enumerate(selected_experiments):
        budget_cost = costs.get("budget", 0)
        time_cost = costs.get("time", 0)
        total_budget_used += budget_cost
        total_time_used += time_cost
        
        print(f"   Experiment {i+1}:")
        print(f"     Parameters: T={params['temperature']:.0f}°C, P={params['pressure']:.1f}atm, C={params['composition']:.2f}")
        print(f"     Budget cost: ${budget_cost:,.0f}")
        print(f"     Time cost: {time_cost:.1f} days")
        print(f"     Expected value: {expected_value:.2f}")
        print()
    
    print("📈 Resource Utilization Summary:")
    budget_constraint = next(c for c in constraints if c.name == "budget")
    time_constraint = next(c for c in constraints if c.name == "time")
    
    budget_utilization = total_budget_used / budget_constraint.total_available
    time_utilization = total_time_used / time_constraint.total_available
    
    print(f"   Budget utilization: ${total_budget_used:,.0f} / ${budget_constraint.total_available:,.0f} ({budget_utilization:.1%})")
    print(f"   Time utilization: {total_time_used:.1f} / {time_constraint.total_available:.1f} days ({time_utilization:.1%})")
    print()
    
    # Create visualizations
    create_resource_allocation_visualizations(
        selected_experiments, constraints, candidate_experiments, uncertainty_quantifier
    )
    
    return selected_experiments, constraints

def create_resource_allocation_visualizations(
    selected_experiments, 
    constraints, 
    all_candidates, 
    uncertainty_quantifier
):
    """Create comprehensive visualizations of resource allocation."""
    
    print("📊 Creating resource allocation visualizations...")
    
    # Create multi-panel plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Uncertainty-Guided Resource Allocation Results', fontsize=16, fontweight='bold')
    
    # 1. Budget vs Time allocation
    selected_budgets = [costs.get("budget", 0) for _, costs, _ in selected_experiments]
    selected_times = [costs.get("time", 0) for _, costs, _ in selected_experiments]
    selected_values = [value for _, _, value in selected_experiments]
    
    scatter = axes[0, 0].scatter(selected_budgets, selected_times, 
                                c=selected_values, cmap='viridis', 
                                s=100, alpha=0.7, edgecolors='black')
    axes[0, 0].set_xlabel('Budget Cost ($)')
    axes[0, 0].set_ylabel('Time Cost (days)')
    axes[0, 0].set_title('Selected Experiments: Cost vs Value')
    plt.colorbar(scatter, ax=axes[0, 0], label='Expected Value')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Parameter space coverage
    selected_temps = [params['temperature'] for params, _, _ in selected_experiments]
    selected_pressures = [params['pressure'] for params, _, _ in selected_experiments]
    
    axes[0, 1].scatter(selected_temps, selected_pressures, 
                      c='red', s=100, alpha=0.7, label='Selected', edgecolors='black')
    
    # Show all candidates in background
    all_temps = [params['temperature'] for params, _ in all_candidates]
    all_pressures = [params['pressure'] for params, _ in all_candidates]
    axes[0, 1].scatter(all_temps, all_pressures, 
                      c='lightblue', s=30, alpha=0.5, label='Candidates')
    
    axes[0, 1].set_xlabel('Temperature (°C)')
    axes[0, 1].set_ylabel('Pressure (atm)')
    axes[0, 1].set_title('Parameter Space Coverage')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Resource utilization
    budget_constraint = next(c for c in constraints if c.name == "budget")
    time_constraint = next(c for c in constraints if c.name == "time")
    
    total_budget_used = sum(selected_budgets)
    total_time_used = sum(selected_times)
    
    resources = ['Budget', 'Time']
    used = [total_budget_used / budget_constraint.total_available * 100,
            total_time_used / time_constraint.total_available * 100]
    available = [100 - used[0], 100 - used[1]]
    
    x = np.arange(len(resources))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, used, width, label='Used', color='orange', alpha=0.7)
    axes[1, 0].bar(x + width/2, available, width, label='Available', color='lightblue', alpha=0.7)
    
    axes[1, 0].set_xlabel('Resource Type')
    axes[1, 0].set_ylabel('Utilization (%)')
    axes[1, 0].set_title('Resource Utilization')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(resources)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 100)
    
    # 4. Uncertainty vs Expected Value
    sample_params = [params for params, _ in all_candidates[:10]]  # Sample for visualization
    uncertainties = []
    
    for params in sample_params:
        uncertainty_metrics = uncertainty_quantifier.quantify_uncertainty(params)
        uncertainties.append(uncertainty_metrics.total_uncertainty)
    
    # Find which of these are selected
    selected_indices = []
    for i, params in enumerate(sample_params):
        for sel_params, _, _ in selected_experiments:
            if (abs(params['temperature'] - sel_params['temperature']) < 1 and
                abs(params['pressure'] - sel_params['pressure']) < 0.1 and
                abs(params['composition'] - sel_params['composition']) < 0.01):
                selected_indices.append(i)
                break
    
    colors = ['red' if i in selected_indices else 'blue' for i in range(len(sample_params))]
    sizes = [100 if i in selected_indices else 50 for i in range(len(sample_params))]
    
    axes[1, 1].scatter(uncertainties, [1] * len(uncertainties), 
                      c=colors, s=sizes, alpha=0.7)
    axes[1, 1].set_xlabel('Total Uncertainty')
    axes[1, 1].set_ylabel('Experiments')
    axes[1, 1].set_title('Uncertainty Distribution')
    axes[1, 1].set_yticks([])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='red', alpha=0.7, label='Selected'),
                      Patch(facecolor='blue', alpha=0.7, label='Not Selected')]
    axes[1, 1].legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig('uncertainty_guided_resource_allocation.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved to: uncertainty_guided_resource_allocation.png")

def main():
    """Run the complete uncertainty-guided resource allocation demonstration."""
    
    try:
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Run optimization
        selected_experiments, constraints = run_uncertainty_guided_optimization()
        
        print("🎉 Uncertainty-Guided Resource Allocation Demonstration Completed Successfully!")
        print()
        print("Key Advantages Demonstrated:")
        print("✅ Intelligent uncertainty quantification across all parameters")
        print("✅ Value of Information calculations for optimal experiment selection")
        print("✅ Risk-aware resource allocation under multiple constraints")
        print("✅ Automatic balancing of exploration vs exploitation")
        print("✅ Multi-objective resource optimization (budget, time, information)")
        print()
        print("This capability is revolutionary and not available in any existing tool!")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in uncertainty-guided optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
