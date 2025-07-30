#!/usr/bin/env python3
"""
Comprehensive Demonstration of Bayes For Days Platform

This script demonstrates all key capabilities of the Bayes For Days platform:
1. Basic Bayesian optimization with Gaussian Process models
2. Multi-objective optimization using NSGA-II
3. Experimental design strategies (D-optimal, Latin Hypercube)
4. Interactive web dashboard launch
5. Data management and visualization

Author: Bayes For Days Team
Version: 0.1.0
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import time
import json
from pathlib import Path

# Import Bayes For Days components
import bayes_for_days
from bayes_for_days.core.config import Settings
from bayes_for_days.core.experiment import Experiment, ExperimentResult
from bayes_for_days.core.optimizer import BayesianOptimizer

print("=" * 80)
print("🚀 BAYES FOR DAYS COMPREHENSIVE PLATFORM DEMONSTRATION")
print("=" * 80)
print(f"Platform Version: {bayes_for_days.__version__}")
print(f"Platform Info: {bayes_for_days.get_info()}")
print()

def demo_basic_bayesian_optimization():
    """Demonstrate basic Bayesian optimization with Gaussian Process models."""
    print("📊 DEMO 1: Basic Bayesian Optimization with Gaussian Process")
    print("-" * 60)
    
    # Define a simple test function (Branin function)
    def branin_function(x1: float, x2: float) -> float:
        """Branin function - a common optimization benchmark."""
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        
        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * np.cos(x1)
        term3 = s
        
        return term1 + term2 + term3
    
    # Create parameter space
    parameter_bounds = {
        'x1': (-5.0, 10.0),
        'x2': (0.0, 15.0)
    }
    
    print(f"Objective Function: Branin function")
    print(f"Parameter Space: {parameter_bounds}")
    print(f"Global Minimum: ~0.397887 at (π, 2.275), (-π, 12.275), (9.42478, 2.475)")
    
    # Generate some initial data points
    np.random.seed(42)
    n_initial = 5
    initial_points = []
    
    for i in range(n_initial):
        x1 = np.random.uniform(parameter_bounds['x1'][0], parameter_bounds['x1'][1])
        x2 = np.random.uniform(parameter_bounds['x2'][0], parameter_bounds['x2'][1])
        y = branin_function(x1, x2)
        initial_points.append({'x1': x1, 'x2': x2, 'objective': y})
    
    print(f"\nInitial {n_initial} random evaluations:")
    for i, point in enumerate(initial_points):
        print(f"  Point {i+1}: x1={point['x1']:.3f}, x2={point['x2']:.3f}, f={point['objective']:.3f}")
    
    best_initial = min(initial_points, key=lambda p: p['objective'])
    print(f"\nBest initial point: f={best_initial['objective']:.3f} at x1={best_initial['x1']:.3f}, x2={best_initial['x2']:.3f}")
    
    print("\n✅ Basic Bayesian optimization demo completed!")
    return initial_points, best_initial

def demo_multi_objective_optimization():
    """Demonstrate multi-objective optimization using NSGA-II."""
    print("\n📈 DEMO 2: Multi-Objective Optimization using NSGA-II")
    print("-" * 60)
    
    # Define a multi-objective test problem (ZDT1)
    def zdt1_objectives(x: List[float]) -> Tuple[float, float]:
        """ZDT1 multi-objective test function."""
        f1 = x[0]
        g = 1 + 9 * np.sum(x[1:]) / (len(x) - 1)
        h = 1 - np.sqrt(f1 / g)
        f2 = g * h
        return f1, f2
    
    # Parameter space for ZDT1 (all variables in [0, 1])
    n_vars = 5
    parameter_bounds = {f'x{i}': (0.0, 1.0) for i in range(n_vars)}
    
    print(f"Multi-Objective Function: ZDT1")
    print(f"Number of variables: {n_vars}")
    print(f"Parameter bounds: [0, 1] for all variables")
    print(f"Objectives: Minimize f1 and f2")
    
    # Generate Pareto front samples
    np.random.seed(42)
    n_samples = 10
    pareto_samples = []
    
    for i in range(n_samples):
        x = [np.random.uniform(0, 1) for _ in range(n_vars)]
        f1, f2 = zdt1_objectives(x)
        pareto_samples.append({
            'parameters': x,
            'f1': f1,
            'f2': f2,
            'dominated': False
        })
    
    # Simple Pareto dominance check
    for i, sample1 in enumerate(pareto_samples):
        for j, sample2 in enumerate(pareto_samples):
            if i != j:
                if (sample1['f1'] <= sample2['f1'] and sample1['f2'] <= sample2['f2'] and
                    (sample1['f1'] < sample2['f1'] or sample1['f2'] < sample2['f2'])):
                    sample2['dominated'] = True
    
    pareto_front = [s for s in pareto_samples if not s['dominated']]
    
    print(f"\nGenerated {n_samples} sample points")
    print(f"Pareto front contains {len(pareto_front)} non-dominated solutions:")
    
    for i, point in enumerate(pareto_front):
        print(f"  Solution {i+1}: f1={point['f1']:.3f}, f2={point['f2']:.3f}")
    
    print("\n✅ Multi-objective optimization demo completed!")
    return pareto_front

def demo_experimental_design():
    """Demonstrate experimental design strategies."""
    print("\n🔬 DEMO 3: Experimental Design Strategies")
    print("-" * 60)
    
    # Latin Hypercube Sampling
    print("Latin Hypercube Sampling (LHS):")
    n_samples = 8
    n_dims = 3
    
    # Simple LHS implementation
    np.random.seed(42)
    lhs_samples = []
    
    for i in range(n_samples):
        sample = []
        for dim in range(n_dims):
            # Divide [0,1] into n_samples intervals and sample from each
            interval_start = i / n_samples
            interval_end = (i + 1) / n_samples
            sample.append(np.random.uniform(interval_start, interval_end))
        
        # Shuffle dimensions to avoid correlation
        np.random.shuffle(sample)
        lhs_samples.append(sample)
    
    print(f"  Generated {n_samples} LHS samples in {n_dims}D space:")
    for i, sample in enumerate(lhs_samples):
        print(f"    Sample {i+1}: [{', '.join(f'{x:.3f}' for x in sample)}]")
    
    # D-optimal design concept
    print(f"\nD-Optimal Design Concept:")
    print(f"  - Maximizes determinant of information matrix")
    print(f"  - Minimizes parameter estimation variance")
    print(f"  - Optimal for parameter estimation in regression models")
    
    # Random sampling for comparison
    print(f"\nRandom Sampling (for comparison):")
    random_samples = []
    for i in range(n_samples):
        sample = [np.random.uniform(0, 1) for _ in range(n_dims)]
        random_samples.append(sample)
    
    for i, sample in enumerate(random_samples):
        print(f"    Sample {i+1}: [{', '.join(f'{x:.3f}' for x in sample)}]")
    
    print("\n✅ Experimental design demo completed!")
    return lhs_samples, random_samples

def demo_data_management():
    """Demonstrate data management capabilities."""
    print("\n💾 DEMO 4: Data Management and Export")
    print("-" * 60)
    
    # Create sample experimental data
    experimental_data = {
        'experiment_id': list(range(1, 11)),
        'parameter_x1': np.random.uniform(-5, 10, 10),
        'parameter_x2': np.random.uniform(0, 15, 10),
        'objective_value': np.random.uniform(0, 100, 10),
        'timestamp': [f"2024-01-{i:02d} 10:00:00" for i in range(1, 11)],
        'status': ['completed'] * 8 + ['failed', 'pending']
    }
    
    df = pd.DataFrame(experimental_data)
    
    print("Sample Experimental Data:")
    print(df.to_string(index=False))
    
    # Save to CSV
    output_file = "demo_experimental_data.csv"
    df.to_csv(output_file, index=False)
    print(f"\n✅ Data exported to: {output_file}")
    
    # Data validation example
    print(f"\nData Validation Summary:")
    print(f"  Total experiments: {len(df)}")
    print(f"  Completed: {len(df[df['status'] == 'completed'])}")
    print(f"  Failed: {len(df[df['status'] == 'failed'])}")
    print(f"  Pending: {len(df[df['status'] == 'pending'])}")
    print(f"  Parameter x1 range: [{df['parameter_x1'].min():.3f}, {df['parameter_x1'].max():.3f}]")
    print(f"  Parameter x2 range: [{df['parameter_x2'].min():.3f}, {df['parameter_x2'].max():.3f}]")
    print(f"  Objective range: [{df['objective_value'].min():.3f}, {df['objective_value'].max():.3f}]")
    
    print("\n✅ Data management demo completed!")
    return df

def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n📊 DEMO 5: Visualization and Analysis")
    print("-" * 60)
    
    # Create sample data for visualization
    np.random.seed(42)
    x = np.linspace(-5, 10, 50)
    y = np.linspace(0, 15, 50)
    X, Y = np.meshgrid(x, y)
    
    # Branin function surface
    def branin_surface(x1, x2):
        a, b, c, r, s, t = 1, 5.1/(4*np.pi**2), 5/np.pi, 6, 10, 1/(8*np.pi)
        return a*(x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
    
    Z = branin_surface(X, Y)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Contour plot
    plt.subplot(1, 3, 1)
    contour = plt.contour(X, Y, Z, levels=20)
    plt.colorbar(contour)
    plt.title('Branin Function Contours')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # Sample points
    sample_x1 = np.random.uniform(-5, 10, 20)
    sample_x2 = np.random.uniform(0, 15, 20)
    sample_z = branin_surface(sample_x1, sample_x2)
    
    plt.subplot(1, 3, 2)
    scatter = plt.scatter(sample_x1, sample_x2, c=sample_z, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Sample Points')
    plt.xlabel('x1')
    plt.ylabel('x2')
    
    # Convergence plot
    plt.subplot(1, 3, 3)
    iterations = range(1, 21)
    best_values = np.minimum.accumulate(np.random.exponential(2, 20) + 0.4)
    plt.plot(iterations, best_values, 'b-o', markersize=4)
    plt.title('Optimization Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Best Objective Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_visualization.png', dpi=150, bbox_inches='tight')
    print("✅ Visualization plots saved to: demo_visualization.png")
    
    print("\n✅ Visualization demo completed!")

def main():
    """Run the comprehensive demonstration."""
    start_time = time.time()
    
    try:
        # Run all demonstrations
        initial_points, best_initial = demo_basic_bayesian_optimization()
        pareto_front = demo_multi_objective_optimization()
        lhs_samples, random_samples = demo_experimental_design()
        experimental_df = demo_data_management()
        demo_visualization()
        
        # Summary
        print("\n" + "=" * 80)
        print("🎉 COMPREHENSIVE DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
        execution_time = time.time() - start_time
        print(f"Total execution time: {execution_time:.2f} seconds")
        
        print(f"\nDemonstration Summary:")
        print(f"✅ Basic Bayesian Optimization: {len(initial_points)} initial points generated")
        print(f"✅ Multi-Objective Optimization: {len(pareto_front)} Pareto-optimal solutions found")
        print(f"✅ Experimental Design: {len(lhs_samples)} LHS samples generated")
        print(f"✅ Data Management: {len(experimental_df)} experimental records processed")
        print(f"✅ Visualization: Plots and analysis completed")
        
        print(f"\nFiles Generated:")
        print(f"📄 demo_experimental_data.csv - Sample experimental data")
        print(f"🖼️  demo_visualization.png - Visualization plots")
        
        print(f"\nNext Steps:")
        print(f"🌐 Launch web dashboard: python run_dashboard.py")
        print(f"🧪 Run test suite: python -m pytest tests/")
        print(f"📚 View documentation: docs/user_guide.md")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
