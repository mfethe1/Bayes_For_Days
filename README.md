# Bayes For Days

A comprehensive Bayesian experimental design platform for multi-objective optimization with cutting-edge methodologies and seamless integration capabilities.

## 🚀 Features

### Multi-Objective Optimization
- **NSGA-II Algorithm**: State-of-the-art multi-objective genetic algorithm
- **Pareto Front Management**: Efficient tracking and analysis of non-dominated solutions
- **Expected Pareto Improvement (EPI)**: Advanced acquisition function for multi-objective Bayesian optimization
- **q-Noisy Expected Hypervolume Improvement (qNEHVI)**: Batch optimization with noise handling
- **Constraint Handling**: Multiple constraint handling methods including feasibility rules and penalty methods

### Gaussian Process Surrogate Models
- **Stochastic Variational Gaussian Processes (SVGP)**: Scalable GP implementation
- **Multiple Kernel Support**: Matérn, RBF, and custom kernels
- **Automatic Relevance Determination (ARD)**: Automatic feature selection
- **Multi-Output GPs**: Handle multiple objectives simultaneously
- **Ensemble Models**: Combine multiple surrogate models for robust predictions

### Experimental Design Strategies
- **D-Optimal Design**: Maximize information for parameter estimation
- **Latin Hypercube Sampling (LHS)**: Space-filling designs with optimization
- **Factorial Designs**: Full and fractional factorial designs
- **Response Surface Methodology (RSM)**: Central composite and Box-Behnken designs
- **Bayesian Adaptive Experimental Design (BAED)**: Information-theoretic experiment selection

### Laboratory Integration
- **Concentration/Value Setting System**: Practical laboratory parameter management
- **Stock Solution Constraints**: Real-world reagent availability and dilution calculations
- **Cost-Aware Design**: Optimize experiments considering reagent costs and instrument time
- **Multi-Stage Workflows**: Sequential experimental campaigns with adaptive stopping

### Advanced Optimization
- **Hybrid Strategies**: Combine Bayesian optimization with genetic algorithms and local search
- **Ensemble Learning**: Multiple model combination with dynamic weighting
- **Acquisition Function Portfolio**: Expected Improvement, Upper Confidence Bound, and more
- **Batch Optimization**: Parallel experiment selection and execution

## 🚀 Features

### Core Capabilities
- **Multi-Objective Bayesian Optimization**: Pareto frontier exploration with advanced acquisition functions
- **Hybrid Optimization Strategies**: Adaptive switching between Bayesian optimization and genetic algorithms
- **Ensemble Learning**: Combines Gaussian Processes, Random Forests, and Neural Networks
- **Advanced Experimental Design**: D-optimal design, Latin Hypercube Sampling, Response Surface Methodology
- **Interactive Visualization**: Web-based dashboard with real-time optimization monitoring
- **Robust Data Management**: CSV import/export with comprehensive validation and preprocessing

### Advanced Features
- **Adaptive Learning Systems**: Continuous model improvement with uncertainty quantification
- **Multi-Fidelity Optimization**: Efficient use of cheap simulations to guide expensive experiments
- **Constraint Handling**: Support for feasibility and safety constraints
- **Real-time Monitoring**: Live experiment tracking with adaptive stopping criteria
- **Scalable Architecture**: Distributed computing support for high-throughput campaigns

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- Git

### Install from Source

```bash
# Clone the repository
git clone https://github.com/mfethe1/Bayes_For_Days.git
cd Bayes_For_Days

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Or install development dependencies
pip install -e ".[dev]"
```

## 🎯 Quick Start

### 1. Command Line Interface

Create a simple optimization experiment:

```bash
# Create experiment configuration interactively
bayes-for-days create-config --name "My Experiment"

# Run optimization from configuration
bayes-for-days optimize experiment_config.json

# Start web server
bayes-for-days serve --host localhost --port 8000
```

### 2. Python API

```python
from bayes_for_days import BayesianOptimizer, Experiment
from bayes_for_days.core.types import *

# Define parameter space
parameters = [
    Parameter(name="temperature", type=ParameterType.CONTINUOUS, bounds=(20, 100)),
    Parameter(name="pressure", type=ParameterType.CONTINUOUS, bounds=(1, 10)),
    Parameter(name="catalyst", type=ParameterType.CATEGORICAL, categories=["A", "B", "C"])
]

# Define objectives
objectives = [
    Objective(name="yield", type=ObjectiveType.MAXIMIZE),
    Objective(name="cost", type=ObjectiveType.MINIMIZE)
]

# Create experiment configuration
config = ExperimentConfig(
    name="Chemical Process Optimization",
    parameter_space=ParameterSpace(parameters=parameters),
    objectives=objectives,
    acquisition_function=AcquisitionFunction.EXPECTED_PARETO_IMPROVEMENT,
    model_type=ModelType.ENSEMBLE,
    max_iterations=100
)

# Define objective function
def evaluate_process(params):
    # Your experimental evaluation code here
    temperature = params["temperature"]
    pressure = params["pressure"]
    catalyst = params["catalyst"]

    # Simulate or run actual experiment
    yield_value = simulate_yield(temperature, pressure, catalyst)
    cost_value = calculate_cost(temperature, pressure, catalyst)

    return {"yield": yield_value, "cost": cost_value}

# Run optimization
experiment = Experiment(config)
result = experiment.run(evaluate_process)

print(f"Best point: {result.optimization_result.best_point}")
print(f"Pareto front size: {len(result.optimization_result.pareto_front)}")
```

## 🏗️ Architecture

The platform follows a modular architecture with clear separation of concerns:

```
bayes_for_days/
├── core/           # Core optimization engine and base classes
├── optimization/   # Bayesian optimization and acquisition functions
├── models/         # Surrogate models (GP, RF, NN, Ensemble)
├── data/           # Data management and validation
├── api/            # RESTful API endpoints
├── web/            # Frontend interface
├── utils/          # Utility functions
└── visualization/  # Interactive plotting and analysis
```

### Key Components

- **Optimization Engine**: Multi-objective Bayesian optimization with hybrid strategies
- **Surrogate Models**: Scalable Gaussian Processes with ensemble methods
- **Acquisition Functions**: Expected Pareto Improvement, Hypervolume Improvement, and more
- **Data Pipeline**: Robust CSV handling with validation and preprocessing
- **Web Interface**: React-based dashboard with real-time updates
- **API Layer**: FastAPI backend with WebSocket support

## 📊 Supported Methods

### Optimization Algorithms
- Bayesian Optimization with Gaussian Processes
- Multi-objective optimization (NSGA-II, MOEA/D)
- Hybrid Bayesian-Genetic algorithms
- Ensemble optimization strategies

### Acquisition Functions
- Expected Improvement (EI)
- Upper Confidence Bound (UCB)
- Expected Pareto Improvement (EPI)
- q-Noisy Expected Hypervolume Improvement (qNEHVI)
- Probability of Improvement (PI)

### Surrogate Models
- Gaussian Processes (GPyTorch backend)
- Random Forests (scikit-learn)
- Neural Networks (PyTorch)
- Ensemble methods with uncertainty aggregation

### Experimental Design
- Latin Hypercube Sampling
- D-optimal design
- Response Surface Methodology
- Bayesian Adaptive Experimental Design (BAED)
- Multi-fidelity approaches

## 🔧 Configuration

The platform uses environment variables and configuration files for customization:

```bash
# Environment variables (prefix with BFD_)
export BFD_DEBUG=true
export BFD_LOG_LEVEL=INFO
export BFD_DATABASE_URL=postgresql://user:pass@localhost/bayes_for_days
export BFD_REDIS_URL=redis://localhost:6379/0
```

See `src/bayes_for_days/core/config.py` for all available settings.

## 📈 Performance Metrics

The platform tracks comprehensive performance metrics:

- **Multi-objective**: Hypervolume indicator, Inverted Generational Distance
- **Convergence**: Simple regret, cumulative regret, convergence rate
- **Uncertainty**: Calibration metrics, coverage probability
- **Efficiency**: Sample efficiency, time to target performance

## 🧪 Examples

Check the `examples/` directory for:
- Simple optimization problems
- Multi-objective optimization
- Constraint handling
- Real-world case studies
- Integration with laboratory equipment

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## 📚 Documentation

- **API Documentation**: Available at `/docs` when running the server
- **User Guide**: Comprehensive tutorials and examples
- **Technical Reference**: Detailed algorithm descriptions
- **Integration Guide**: How to integrate with existing systems

## 🛣️ Roadmap

### Phase 1: Core Platform (Current)
- [x] Project architecture and setup
- [ ] Core data management system
- [ ] Gaussian Process surrogate models
- [ ] Multi-objective optimization core
- [ ] Basic web interface

### Phase 2: Advanced Features
- [ ] Ensemble learning framework
- [ ] Hybrid optimization strategies
- [ ] Advanced experimental design methods
- [ ] Interactive visualization dashboard

### Phase 3: Production Features
- [ ] Performance metrics and validation
- [ ] Deployment and scalability
- [ ] Laboratory integration
- [ ] Advanced uncertainty quantification

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of excellent open-source libraries: BoTorch, GPyTorch, scikit-learn, FastAPI
- Inspired by state-of-the-art research in Bayesian optimization and experimental design
- Thanks to the scientific computing and optimization communities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/mfethe1/Bayes_For_Days/issues)
- **Discussions**: [GitHub Discussions](https://github.com/mfethe1/Bayes_For_Days/discussions)
- **Email**: team@bayesfordays.com

---

**Bayes For Days** - Making every day a good day for Bayesian optimization! 🎯
