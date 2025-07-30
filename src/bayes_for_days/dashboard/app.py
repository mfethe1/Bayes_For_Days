"""
Web dashboard for Bayes For Days platform.

This module provides a web-based interface for:
- Interactive optimization campaign management
- Real-time visualization of optimization progress
- Parameter space exploration and analysis
- Experimental design configuration
- Results analysis and reporting

Built with Streamlit for rapid development and deployment.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Any
import json
import time
from datetime import datetime

# Import Bayes For Days components
from bayes_for_days.core.types import (
    ParameterSpace,
    Parameter,
    ParameterType,
    ExperimentPoint,
)
from bayes_for_days.models.gaussian_process import GaussianProcessModel
from bayes_for_days.acquisition.expected_improvement import ExpectedImprovement
from bayes_for_days.optimization.optimization_loop import (
    BayesianOptimizationLoop,
    OptimizationConfig,
)
from bayes_for_days.optimization.multi_objective import NSGAIIOptimizer

logger = logging.getLogger(__name__)


class BayesForDaysDashboard:
    """
    Main dashboard application for Bayes For Days platform.
    
    Provides interactive web interface for optimization campaigns,
    visualization, and analysis.
    """
    
    def __init__(self):
        """Initialize dashboard application."""
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="Bayes For Days",
            page_icon="🧬",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .status-running {
            color: #28a745;
        }
        .status-complete {
            color: #007bff;
        }
        .status-error {
            color: #dc3545;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'optimization_results' not in st.session_state:
            st.session_state.optimization_results = []
        
        if 'current_campaign' not in st.session_state:
            st.session_state.current_campaign = None
        
        if 'parameter_space' not in st.session_state:
            st.session_state.parameter_space = None
        
        if 'experiment_data' not in st.session_state:
            st.session_state.experiment_data = []
    
    def run(self):
        """Run the main dashboard application."""
        # Header
        st.markdown('<h1 class="main-header">🧬 Bayes For Days</h1>', unsafe_allow_html=True)
        st.markdown("**Comprehensive Bayesian Experimental Design Platform**")
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            [
                "🏠 Home",
                "⚙️ Configure Optimization",
                "🚀 Run Campaign",
                "📊 Results Analysis",
                "🔬 Experimental Design",
                "📈 Multi-Objective",
                "🎯 Model Analysis"
            ]
        )
        
        # Route to appropriate page
        if page == "🏠 Home":
            self.show_home_page()
        elif page == "⚙️ Configure Optimization":
            self.show_configuration_page()
        elif page == "🚀 Run Campaign":
            self.show_campaign_page()
        elif page == "📊 Results Analysis":
            self.show_results_page()
        elif page == "🔬 Experimental Design":
            self.show_design_page()
        elif page == "📈 Multi-Objective":
            self.show_multi_objective_page()
        elif page == "🎯 Model Analysis":
            self.show_model_analysis_page()
    
    def show_home_page(self):
        """Display home page with platform overview."""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🎯 Optimization")
            st.markdown("""
            - Bayesian Optimization
            - Multi-Objective NSGA-II
            - Hybrid Strategies
            - Ensemble Models
            """)
        
        with col2:
            st.markdown("### 🔬 Experimental Design")
            st.markdown("""
            - D-Optimal Design
            - Latin Hypercube Sampling
            - Factorial Designs
            - Laboratory Constraints
            """)
        
        with col3:
            st.markdown("### 📊 Analysis")
            st.markdown("""
            - Real-time Visualization
            - Performance Metrics
            - Model Diagnostics
            - Export Results
            """)
        
        # Quick stats
        st.markdown("---")
        st.markdown("### 📈 Platform Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Campaigns", len(st.session_state.optimization_results))
        
        with col2:
            total_experiments = sum(
                len(result.get('all_experiments', []))
                for result in st.session_state.optimization_results
            )
            st.metric("Total Experiments", total_experiments)
        
        with col3:
            if st.session_state.optimization_results:
                best_value = max(
                    result.get('best_objective_value', 0)
                    for result in st.session_state.optimization_results
                )
                st.metric("Best Objective Value", f"{best_value:.4f}")
            else:
                st.metric("Best Objective Value", "N/A")
        
        with col4:
            if st.session_state.current_campaign:
                st.markdown('<p class="status-running">🟢 Campaign Running</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-complete">⚪ Ready</p>', unsafe_allow_html=True)
    
    def show_configuration_page(self):
        """Display optimization configuration page."""
        st.markdown("## ⚙️ Configure Optimization")
        
        # Parameter space configuration
        st.markdown("### Parameter Space")
        
        # Number of parameters
        n_params = st.number_input("Number of Parameters", min_value=1, max_value=10, value=2)
        
        parameters = []
        
        for i in range(n_params):
            st.markdown(f"#### Parameter {i+1}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                param_name = st.text_input(f"Name", value=f"param_{i+1}", key=f"name_{i}")
            
            with col2:
                param_type = st.selectbox(
                    "Type",
                    ["Continuous", "Integer", "Categorical"],
                    key=f"type_{i}"
                )
            
            with col3:
                if param_type in ["Continuous", "Integer"]:
                    bounds_str = st.text_input(
                        "Bounds (min,max)",
                        value="0,10",
                        key=f"bounds_{i}"
                    )
                    
                    try:
                        bounds = tuple(map(float, bounds_str.split(',')))
                        if len(bounds) != 2:
                            st.error("Bounds must be two values separated by comma")
                            continue
                    except:
                        st.error("Invalid bounds format")
                        continue
                else:
                    categories_str = st.text_input(
                        "Categories (comma-separated)",
                        value="A,B,C",
                        key=f"categories_{i}"
                    )
                    categories = [cat.strip() for cat in categories_str.split(',')]
                    bounds = None
            
            # Create parameter
            if param_type == "Continuous":
                param = Parameter(
                    name=param_name,
                    type=ParameterType.CONTINUOUS,
                    bounds=bounds
                )
            elif param_type == "Integer":
                param = Parameter(
                    name=param_name,
                    type=ParameterType.INTEGER,
                    bounds=bounds
                )
            else:
                param = Parameter(
                    name=param_name,
                    type=ParameterType.CATEGORICAL,
                    categories=categories
                )
            
            parameters.append(param)
        
        # Store parameter space
        if parameters:
            st.session_state.parameter_space = ParameterSpace(parameters=parameters)
            st.success(f"Parameter space configured with {len(parameters)} parameters")
        
        # Optimization configuration
        st.markdown("### Optimization Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            max_iterations = st.number_input("Max Iterations", min_value=1, max_value=200, value=50)
            initial_experiments = st.number_input("Initial Experiments", min_value=1, max_value=50, value=5)
        
        with col2:
            acquisition_function = st.selectbox(
                "Acquisition Function",
                ["Expected Improvement", "Upper Confidence Bound", "Thompson Sampling"]
            )
            
            batch_size = st.number_input("Batch Size", min_value=1, max_value=10, value=1)
        
        # Store configuration
        config = OptimizationConfig(
            max_iterations=max_iterations,
            initial_experiments=initial_experiments,
            batch_size=batch_size,
            acquisition_function=acquisition_function.lower().replace(" ", "_"),
            verbose=True
        )
        
        st.session_state.optimization_config = config
        
        if st.button("Save Configuration"):
            st.success("Configuration saved successfully!")
    
    def show_campaign_page(self):
        """Display campaign execution page."""
        st.markdown("## 🚀 Run Optimization Campaign")
        
        if st.session_state.parameter_space is None:
            st.warning("Please configure parameter space first!")
            return
        
        # Objective function definition
        st.markdown("### Objective Function")
        
        objective_type = st.selectbox(
            "Objective Type",
            ["Test Function", "Custom Function"]
        )
        
        if objective_type == "Test Function":
            test_function = st.selectbox(
                "Test Function",
                ["Quadratic", "Rosenbrock", "Ackley"]
            )
            
            objective_function = self._create_test_function(test_function)
        
        else:
            st.markdown("Define your custom objective function:")
            function_code = st.text_area(
                "Python Function",
                value="""def objective(params):
    # Example: maximize -(x-1)^2 - (y-2)^2
    x = params.get('param_1', 0)
    y = params.get('param_2', 0)
    return -(x-1)**2 - (y-2)**2""",
                height=150
            )
            
            try:
                exec(function_code)
                objective_function = locals()['objective']
            except Exception as e:
                st.error(f"Error in function definition: {e}")
                return
        
        # Campaign controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Start Campaign", type="primary"):
                self._start_optimization_campaign(objective_function)
        
        with col2:
            if st.button("Stop Campaign"):
                st.session_state.current_campaign = None
                st.success("Campaign stopped")
        
        with col3:
            if st.button("Clear Results"):
                st.session_state.optimization_results = []
                st.session_state.experiment_data = []
                st.success("Results cleared")
        
        # Campaign status
        if st.session_state.current_campaign:
            st.markdown("### 📊 Campaign Status")
            
            # Progress metrics
            col1, col2, col3, col4 = st.columns(4)
            
            campaign = st.session_state.current_campaign
            
            with col1:
                st.metric("Iteration", campaign.get('iteration', 0))
            
            with col2:
                st.metric("Experiments", len(campaign.get('all_experiments', [])))
            
            with col3:
                best_value = campaign.get('best_objective_value', 0)
                st.metric("Best Value", f"{best_value:.4f}")
            
            with col4:
                status = "Running" if campaign.get('is_running', False) else "Complete"
                st.metric("Status", status)
            
            # Real-time plot
            if campaign.get('all_experiments'):
                self._plot_optimization_progress(campaign['all_experiments'])
    
    def _create_test_function(self, function_name: str):
        """Create test objective function."""
        if function_name == "Quadratic":
            def objective(params):
                values = list(params.values())
                return -sum((x - 1)**2 for x in values)
        
        elif function_name == "Rosenbrock":
            def objective(params):
                values = list(params.values())
                if len(values) < 2:
                    return 0
                result = 0
                for i in range(len(values) - 1):
                    result += 100 * (values[i+1] - values[i]**2)**2 + (1 - values[i])**2
                return -result
        
        elif function_name == "Ackley":
            def objective(params):
                values = np.array(list(params.values()))
                a, b, c = 20, 0.2, 2 * np.pi
                term1 = -a * np.exp(-b * np.sqrt(np.mean(values**2)))
                term2 = -np.exp(np.mean(np.cos(c * values)))
                return -(term1 + term2 + a + np.e)
        
        else:
            def objective(params):
                return 0
        
        return objective
    
    def _start_optimization_campaign(self, objective_function):
        """Start optimization campaign."""
        try:
            config = getattr(st.session_state, 'optimization_config', OptimizationConfig())
            
            # Create optimizer
            optimizer = BayesianOptimizationLoop(
                parameter_space=st.session_state.parameter_space,
                objective_function=objective_function,
                config=config
            )
            
            # Run optimization (simplified for demo)
            with st.spinner("Running optimization..."):
                result = optimizer.optimize()
            
            # Store results
            result_dict = {
                'timestamp': datetime.now().isoformat(),
                'best_parameters': result.best_parameters,
                'best_objective_value': result.best_objective_value,
                'n_iterations': result.n_iterations,
                'n_function_evaluations': result.n_function_evaluations,
                'execution_time_seconds': result.execution_time_seconds,
                'is_converged': result.is_converged,
                'all_experiments': [
                    {
                        'parameters': exp.parameters,
                        'objectives': exp.objectives,
                        'is_feasible': exp.is_feasible
                    }
                    for exp in result.all_experiments
                ]
            }
            
            st.session_state.optimization_results.append(result_dict)
            st.session_state.current_campaign = result_dict
            
            st.success(f"Campaign completed! Best value: {result.best_objective_value:.4f}")
            
        except Exception as e:
            st.error(f"Campaign failed: {e}")
    
    def _plot_optimization_progress(self, experiments):
        """Plot optimization progress."""
        if not experiments:
            return
        
        # Extract objective values
        objective_values = []
        for exp in experiments:
            if exp.get('objectives'):
                obj_value = list(exp['objectives'].values())[0]
                objective_values.append(obj_value)
        
        if not objective_values:
            return
        
        # Create progress plot
        fig = go.Figure()
        
        # Objective values over time
        fig.add_trace(go.Scatter(
            x=list(range(1, len(objective_values) + 1)),
            y=objective_values,
            mode='lines+markers',
            name='Objective Value',
            line=dict(color='blue')
        ))
        
        # Best value so far
        best_so_far = []
        current_best = float('-inf')
        
        for value in objective_values:
            if value > current_best:
                current_best = value
            best_so_far.append(current_best)
        
        fig.add_trace(go.Scatter(
            x=list(range(1, len(best_so_far) + 1)),
            y=best_so_far,
            mode='lines',
            name='Best So Far',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Optimization Progress",
            xaxis_title="Iteration",
            yaxis_title="Objective Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def show_results_page(self):
        """Display results analysis page."""
        st.markdown("## 📊 Results Analysis")
        
        if not st.session_state.optimization_results:
            st.info("No optimization results available. Run a campaign first!")
            return
        
        # Results summary
        st.markdown("### Campaign Summary")
        
        results_df = pd.DataFrame([
            {
                'Campaign': i + 1,
                'Best Value': result['best_objective_value'],
                'Iterations': result['n_iterations'],
                'Experiments': result['n_function_evaluations'],
                'Time (s)': result['execution_time_seconds'],
                'Converged': result['is_converged']
            }
            for i, result in enumerate(st.session_state.optimization_results)
        ])
        
        st.dataframe(results_df, use_container_width=True)
        
        # Detailed analysis for selected campaign
        if len(st.session_state.optimization_results) > 0:
            campaign_idx = st.selectbox(
                "Select Campaign for Analysis",
                range(len(st.session_state.optimization_results)),
                format_func=lambda x: f"Campaign {x + 1}"
            )
            
            selected_result = st.session_state.optimization_results[campaign_idx]
            
            # Best parameters
            st.markdown("### Best Parameters")
            best_params_df = pd.DataFrame([
                {'Parameter': k, 'Value': v}
                for k, v in selected_result['best_parameters'].items()
            ])
            st.dataframe(best_params_df, use_container_width=True)
            
            # Optimization progress
            st.markdown("### Optimization Progress")
            if selected_result.get('all_experiments'):
                self._plot_optimization_progress(selected_result['all_experiments'])
    
    def show_design_page(self):
        """Display experimental design page."""
        st.markdown("## 🔬 Experimental Design")
        st.info("Experimental design features coming soon!")
    
    def show_multi_objective_page(self):
        """Display multi-objective optimization page."""
        st.markdown("## 📈 Multi-Objective Optimization")
        st.info("Multi-objective optimization features coming soon!")
    
    def show_model_analysis_page(self):
        """Display model analysis page."""
        st.markdown("## 🎯 Model Analysis")
        st.info("Model analysis features coming soon!")


def main():
    """Main entry point for dashboard application."""
    dashboard = BayesForDaysDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
