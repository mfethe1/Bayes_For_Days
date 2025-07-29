"""
Command-line interface for Bayes For Days platform.

This module provides a CLI for running optimization experiments,
managing data, and interacting with the platform from the command line.
"""

import click
from pathlib import Path
from typing import Optional
import json

from bayes_for_days.core.config import settings
from bayes_for_days import __version__


@click.group()
@click.version_option(version=__version__)
@click.option('--debug/--no-debug', default=False, help='Enable debug mode')
@click.option('--config', type=click.Path(exists=True), help='Configuration file path')
def main(debug: bool, config: Optional[str]) -> None:
    """
    Bayes For Days: Comprehensive Bayesian Experimental Design Optimization Platform
    
    A sophisticated platform for multi-objective Bayesian optimization with adaptive
    learning capabilities, designed for experimental design and optimization scenarios.
    """
    if debug:
        settings.debug = True
        settings.log_level = "DEBUG"
    
    if config:
        # Load configuration from file
        with open(config, 'r') as f:
            config_data = json.load(f)
        
        # Update settings with config file values
        for key, value in config_data.items():
            if hasattr(settings, key):
                setattr(settings, key, value)


@main.command()
@click.option('--host', default='localhost', help='Server host')
@click.option('--port', default=8000, help='Server port')
@click.option('--reload/--no-reload', default=False, help='Auto-reload on code changes')
@click.option('--workers', default=1, help='Number of worker processes')
def serve(host: str, port: int, reload: bool, workers: int) -> None:
    """Start the web server."""
    import uvicorn
    
    click.echo(f"Starting Bayes For Days server on {host}:{port}")
    click.echo(f"Debug mode: {settings.debug}")
    click.echo(f"Workers: {workers}")
    
    uvicorn.run(
        "bayes_for_days.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,
        log_level=settings.log_level.lower(),
    )


@main.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for results')
@click.option('--data', '-d', type=click.Path(exists=True), help='Initial data file')
def optimize(config_file: str, output: Optional[str], data: Optional[str]) -> None:
    """Run an optimization experiment from configuration file."""
    from bayes_for_days.core.experiment import Experiment
    from bayes_for_days.core.types import ExperimentConfig
    from bayes_for_days.data.manager import DataManager
    
    click.echo(f"Loading experiment configuration from {config_file}")
    
    # Load experiment configuration
    with open(config_file, 'r') as f:
        config_data = json.load(f)
    
    config = ExperimentConfig(**config_data)
    
    # Create experiment
    experiment = Experiment(config)
    
    # Load initial data if provided
    if data:
        click.echo(f"Loading initial data from {data}")
        data_manager = DataManager()
        initial_data = data_manager.load_data(data)
        experiment.add_initial_data(initial_data)
    
    # Define objective function (this would need to be customized)
    def objective_function(parameters):
        # This is a placeholder - in practice, this would call
        # external simulation, experiment, or evaluation code
        click.echo(f"Evaluating parameters: {parameters}")
        
        # Example: simple quadratic function
        result = {}
        for obj in config.objectives:
            if obj.name == "objective1":
                result[obj.name] = sum(x**2 for x in parameters.values())
            else:
                result[obj.name] = sum(abs(x) for x in parameters.values())
        
        return result
    
    # Run optimization
    click.echo("Starting optimization...")
    with click.progressbar(length=config.max_iterations, label='Optimizing') as bar:
        def progress_callback(iteration_data):
            bar.update(1)
        
        experiment.add_iteration_callback(progress_callback)
        result = experiment.run(objective_function)
    
    click.echo(f"Optimization completed with status: {result.optimization_result.status}")
    click.echo(f"Best point: {result.optimization_result.best_point}")
    
    # Save results
    if output:
        result.save_to_file(output)
        click.echo(f"Results saved to {output}")
    else:
        # Save to default location
        output_file = f"experiment_{result.id}_results.json"
        result.save_to_file(output_file)
        click.echo(f"Results saved to {output_file}")


@main.command()
@click.argument('data_file', type=click.Path(exists=True))
@click.option('--config', type=click.Path(exists=True), help='Parameter space configuration')
def validate(data_file: str, config: Optional[str]) -> None:
    """Validate experimental data file."""
    from bayes_for_days.data.manager import DataManager
    from bayes_for_days.core.types import ParameterSpace
    
    click.echo(f"Validating data file: {data_file}")
    
    data_manager = DataManager()
    
    # Load data
    try:
        data = data_manager.load_data(data_file)
        click.echo(f"Loaded {len(data)} data points")
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        return
    
    # Load parameter space if provided
    parameter_space = None
    if config:
        with open(config, 'r') as f:
            config_data = json.load(f)
        parameter_space = ParameterSpace(**config_data['parameter_space'])
    
    # Validate data
    if parameter_space:
        validation_result = data_manager.validate_data(data, parameter_space)
        
        if validation_result.is_valid:
            click.echo("✓ Data validation passed")
        else:
            click.echo("✗ Data validation failed")
            for error in validation_result.errors:
                click.echo(f"  Error: {error}")
            for warning in validation_result.warnings:
                click.echo(f"  Warning: {warning}")
    else:
        click.echo("No parameter space provided - skipping validation")
    
    # Show data summary
    click.echo(f"Data summary:")
    click.echo(f"  Rows: {len(data)}")
    if data:
        click.echo(f"  Parameters: {list(data[0].parameters.keys())}")
        click.echo(f"  Objectives: {list(data[0].objectives.keys())}")


@main.command()
@click.option('--name', prompt='Experiment name', help='Name of the experiment')
@click.option('--description', help='Description of the experiment')
@click.option('--output', '-o', type=click.Path(), help='Output configuration file')
def create_config(name: str, description: Optional[str], output: Optional[str]) -> None:
    """Create a new experiment configuration file interactively."""
    from bayes_for_days.core.types import (
        ExperimentConfig, ParameterSpace, Parameter, Objective,
        ParameterType, ObjectiveType, AcquisitionFunction, ModelType
    )
    
    click.echo(f"Creating configuration for experiment: {name}")
    
    # Collect parameters
    parameters = []
    while True:
        param_name = click.prompt("Parameter name (or 'done' to finish)")
        if param_name.lower() == 'done':
            break
        
        param_type = click.prompt(
            "Parameter type",
            type=click.Choice(['continuous', 'integer', 'categorical', 'ordinal'])
        )
        
        if param_type in ['continuous', 'integer']:
            low = click.prompt("Lower bound", type=float)
            high = click.prompt("Upper bound", type=float)
            bounds = (low, high)
            categories = None
        else:
            categories = click.prompt("Categories (comma-separated)").split(',')
            categories = [cat.strip() for cat in categories]
            bounds = None
        
        param = Parameter(
            name=param_name,
            type=ParameterType(param_type),
            bounds=bounds,
            categories=categories,
            description=click.prompt("Parameter description", default="")
        )
        parameters.append(param)
    
    # Collect objectives
    objectives = []
    while True:
        obj_name = click.prompt("Objective name (or 'done' to finish)")
        if obj_name.lower() == 'done':
            break
        
        obj_type = click.prompt(
            "Objective type",
            type=click.Choice(['minimize', 'maximize'])
        )
        
        objective = Objective(
            name=obj_name,
            type=ObjectiveType(obj_type),
            description=click.prompt("Objective description", default="")
        )
        objectives.append(objective)
    
    # Create configuration
    parameter_space = ParameterSpace(parameters=parameters)
    
    config = ExperimentConfig(
        name=name,
        description=description,
        parameter_space=parameter_space,
        objectives=objectives,
        acquisition_function=AcquisitionFunction.EXPECTED_IMPROVEMENT,
        model_type=ModelType.GAUSSIAN_PROCESS,
        n_initial_points=click.prompt("Number of initial points", default=10, type=int),
        max_iterations=click.prompt("Maximum iterations", default=100, type=int),
    )
    
    # Save configuration
    if not output:
        output = f"{name.lower().replace(' ', '_')}_config.json"
    
    with open(output, 'w') as f:
        json.dump(config.dict(), f, indent=2, default=str)
    
    click.echo(f"Configuration saved to {output}")


@main.command()
def info() -> None:
    """Show platform information."""
    from bayes_for_days import get_info
    
    info_data = get_info()
    
    click.echo("Bayes For Days Platform Information")
    click.echo("=" * 40)
    for key, value in info_data.items():
        click.echo(f"{key.capitalize()}: {value}")
    
    click.echo("\nConfiguration:")
    click.echo(f"Debug mode: {settings.debug}")
    click.echo(f"Log level: {settings.log_level}")
    click.echo(f"Data directory: {settings.data_dir}")
    click.echo(f"Models directory: {settings.models_dir}")


if __name__ == '__main__':
    main()
