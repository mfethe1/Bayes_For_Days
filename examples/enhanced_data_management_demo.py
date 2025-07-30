"""
Comprehensive demonstration of the enhanced data management system.

This example showcases the advanced data management capabilities including:
- Domain-specific preprocessing for different experimental domains
- Advanced missing value imputation with uncertainty quantification
- Real-time data quality monitoring
- Data versioning and provenance tracking
- Integration with optimization workflows
"""

import asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, AsyncGenerator

from bayes_for_days.data import EnhancedDataManager, ImputationManager
from bayes_for_days.core.types import (
    ExperimentPoint,
    DataDomain,
    ImputationStrategy,
    DataStreamConfig,
    DataQualityMetrics,
    Parameter,
    ParameterType,
    ParameterSpace,
    Objective,
    ObjectiveType,
)


def create_sample_chemical_data() -> List[ExperimentPoint]:
    """Create sample chemical experimental data with missing values."""
    np.random.seed(42)
    
    data_points = []
    for i in range(50):
        # Chemical parameters
        temperature = np.random.uniform(20, 100) if np.random.random() > 0.1 else None
        pressure = np.random.uniform(1, 10) if np.random.random() > 0.15 else None
        catalyst = np.random.choice(['A', 'B', 'C']) if np.random.random() > 0.05 else None
        ph = np.random.uniform(6, 8) if np.random.random() > 0.2 else None
        
        # Objectives (with some correlation to parameters)
        if temperature and pressure:
            yield_val = 0.5 + 0.3 * (temperature / 100) + 0.2 * (pressure / 10) + np.random.normal(0, 0.1)
            cost = 100 + 2 * temperature + 5 * pressure + np.random.normal(0, 10)
        else:
            yield_val = None
            cost = None
        
        parameters = {}
        if temperature is not None:
            parameters['temperature'] = temperature
        if pressure is not None:
            parameters['pressure'] = pressure
        if catalyst is not None:
            parameters['catalyst'] = catalyst
        if ph is not None:
            parameters['ph'] = ph
        
        objectives = {}
        if yield_val is not None:
            objectives['yield'] = max(0, min(1, yield_val))
        if cost is not None:
            objectives['cost'] = max(50, cost)
        
        point = ExperimentPoint(
            id=f"chem_exp_{i:03d}",
            parameters=parameters,
            objectives=objectives,
            timestamp=datetime.now(),
            is_feasible=len(parameters) >= 3,  # Require at least 3 parameters
            metadata={'experiment_type': 'chemical_synthesis', 'batch': i // 10}
        )
        
        data_points.append(point)
    
    return data_points


def demonstrate_domain_specific_processing():
    """Demonstrate domain-specific data processing."""
    print("=== Domain-Specific Data Processing Demo ===")
    
    # Create sample data
    chemical_data = create_sample_chemical_data()
    print(f"Created {len(chemical_data)} chemical experiment points")
    
    # Initialize data manager for chemical domain
    data_manager = EnhancedDataManager(
        domain=DataDomain.CHEMICAL,
        enable_versioning=True,
        enable_quality_monitoring=True
    )
    
    print(f"Initialized data manager for {data_manager.domain} domain")
    
    # Validate data
    parameter_space = ParameterSpace(parameters=[
        Parameter(name="temperature", type=ParameterType.CONTINUOUS, bounds=(20, 100)),
        Parameter(name="pressure", type=ParameterType.CONTINUOUS, bounds=(1, 10)),
        Parameter(name="catalyst", type=ParameterType.CATEGORICAL, categories=['A', 'B', 'C']),
        Parameter(name="ph", type=ParameterType.CONTINUOUS, bounds=(6, 8)),
    ])
    
    validation_result = data_manager.validate_data(chemical_data, parameter_space)
    print(f"Data validation: {'PASS' if validation_result.is_valid else 'FAIL'}")
    print(f"  Errors: {len(validation_result.errors)}")
    print(f"  Warnings: {len(validation_result.warnings)}")
    print(f"  Missing values: {validation_result.missing_values}")
    
    return data_manager, chemical_data, validation_result


def demonstrate_advanced_imputation():
    """Demonstrate advanced imputation techniques."""
    print("\n=== Advanced Imputation Demo ===")
    
    data_manager, chemical_data, _ = demonstrate_domain_specific_processing()
    
    # Test different imputation strategies
    strategies = [
        ImputationStrategy.MICE,
        ImputationStrategy.KNN,
        ImputationStrategy.DOMAIN_AWARE
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value} imputation:")
        
        try:
            processed_data = data_manager.preprocess_data(
                chemical_data,
                imputation_strategy=strategy.value,
                feature_engineering=True,
                imputation_params={'n_imputations': 3} if strategy == ImputationStrategy.MICE else {}
            )
            
            # Count missing values before and after
            missing_before = sum(1 for point in chemical_data 
                               for param in ['temperature', 'pressure', 'catalyst', 'ph'] 
                               if param not in point.parameters)
            
            missing_after = sum(1 for point in processed_data 
                              for param in ['temperature', 'pressure', 'catalyst', 'ph'] 
                              if param not in point.parameters)
            
            print(f"  Missing values: {missing_before} → {missing_after}")
            print(f"  Imputation quality: {(missing_before - missing_after) / missing_before * 100:.1f}% improvement")
            
            results[strategy] = {
                'processed_data': processed_data,
                'missing_reduction': missing_before - missing_after
            }
            
        except Exception as e:
            print(f"  Error with {strategy.value}: {e}")
    
    return results


def demonstrate_data_versioning():
    """Demonstrate data versioning and provenance tracking."""
    print("\n=== Data Versioning Demo ===")
    
    data_manager, chemical_data, _ = demonstrate_domain_specific_processing()
    
    # Create initial version
    version_1 = data_manager.create_data_version(
        data=chemical_data,
        author="researcher_1",
        message="Initial chemical experimental data"
    )
    
    if version_1:
        print(f"Created version 1: {version_1.version_id}")
        print(f"  Data hash: {version_1.data_hash}")
        print(f"  Timestamp: {version_1.timestamp}")
    
    # Process data and create new version
    processed_data = data_manager.preprocess_data(
        chemical_data,
        imputation_strategy=ImputationStrategy.MICE.value,
        feature_engineering=True
    )
    
    version_2 = data_manager.create_data_version(
        data=processed_data,
        author="researcher_1", 
        message="Data after MICE imputation and feature engineering",
        parent_version=version_1.version_id if version_1 else None
    )
    
    if version_2:
        print(f"Created version 2: {version_2.version_id}")
        print(f"  Parent: {version_2.parent_version}")
        print(f"  Changes: {version_2.changes_summary}")
    
    # List all versions
    versions = data_manager.get_version_history()
    print(f"\nVersion history ({len(versions)} versions):")
    for version in versions:
        print(f"  {version.version_id}: {version.message}")
    
    return versions


async def simulate_data_stream() -> AsyncGenerator[ExperimentPoint, None]:
    """Simulate a real-time data stream from laboratory instruments."""
    for i in range(20):
        # Simulate some delay
        await asyncio.sleep(0.1)
        
        # Create experiment point with occasional quality issues
        temperature = np.random.uniform(20, 100)
        pressure = np.random.uniform(1, 10)
        
        # Introduce quality issues occasionally
        if np.random.random() < 0.1:  # 10% chance of anomaly
            temperature = np.random.uniform(150, 200)  # Out of normal range
        
        if np.random.random() < 0.05:  # 5% chance of missing data
            pressure = None
        
        parameters = {'temperature': temperature}
        if pressure is not None:
            parameters['pressure'] = pressure
        
        # Simulate objectives
        if pressure is not None:
            yield_val = 0.5 + 0.3 * (temperature / 100) + 0.2 * (pressure / 10) + np.random.normal(0, 0.1)
            objectives = {'yield': max(0, min(1, yield_val))}
        else:
            objectives = {}
        
        point = ExperimentPoint(
            id=f"stream_point_{i:03d}",
            parameters=parameters,
            objectives=objectives,
            timestamp=datetime.now(),
            is_feasible=True
        )
        
        yield point


async def demonstrate_quality_monitoring():
    """Demonstrate real-time data quality monitoring."""
    print("\n=== Real-Time Quality Monitoring Demo ===")
    
    # Set up quality monitoring
    stream_config = DataStreamConfig(
        stream_id="chemical_reactor_1",
        source_type="instrument",
        connection_params={"host": "localhost", "port": 8080},
        quality_thresholds=DataQualityMetrics(
            completeness=0.9,
            consistency=0.95,
            validity=0.9
        ),
        monitoring_interval=60,
        buffer_size=10,
        enable_anomaly_detection=True,
        enable_drift_detection=True
    )
    
    data_manager = EnhancedDataManager(
        domain=DataDomain.CHEMICAL,
        enable_quality_monitoring=True
    )
    
    data_manager.setup_quality_monitoring(stream_config)
    print(f"Set up quality monitoring for stream: {stream_config.stream_id}")
    
    # Monitor the simulated data stream
    data_stream = simulate_data_stream()
    quality_metrics = []
    
    print("Monitoring data stream...")
    async for metrics in data_manager.monitor_data_stream(stream_config.stream_id, data_stream):
        quality_metrics.append(metrics)
        print(f"  Quality score: {metrics.overall_quality():.3f} "
              f"(completeness: {metrics.completeness:.3f}, "
              f"consistency: {metrics.consistency:.3f}, "
              f"validity: {metrics.validity:.3f})")
        
        if metrics.overall_quality() < 0.8:
            print(f"  ⚠️  Quality alert! Score below threshold")
    
    print(f"Processed {len(quality_metrics)} quality assessments")
    return quality_metrics


def demonstrate_file_operations():
    """Demonstrate file I/O operations with versioning."""
    print("\n=== File Operations Demo ===")
    
    data_manager, chemical_data, _ = demonstrate_domain_specific_processing()
    
    # Save data to different formats
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_path = output_dir / "chemical_data.csv"
    data_manager.save_data(
        chemical_data, 
        str(csv_path),
        author="demo_user",
        message="Export chemical data to CSV"
    )
    print(f"Saved data to {csv_path}")
    
    # Save as JSON
    json_path = output_dir / "chemical_data.json"
    data_manager.save_data(
        chemical_data,
        str(json_path),
        author="demo_user", 
        message="Export chemical data to JSON"
    )
    print(f"Saved data to {json_path}")
    
    # Load data back
    loaded_data = data_manager.load_data(str(csv_path))
    print(f"Loaded {len(loaded_data)} points from {csv_path}")
    
    return loaded_data


async def main():
    """Run all demonstrations."""
    print("🧪 Enhanced Data Management System Demo")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_domain_specific_processing()
    demonstrate_advanced_imputation()
    demonstrate_data_versioning()
    
    # Run async quality monitoring demo
    await demonstrate_quality_monitoring()
    
    demonstrate_file_operations()
    
    print("\n✅ All demonstrations completed successfully!")
    print("\nKey features demonstrated:")
    print("  • Domain-specific data processing")
    print("  • Advanced imputation with uncertainty quantification")
    print("  • Data versioning and provenance tracking")
    print("  • Real-time quality monitoring")
    print("  • Robust file I/O operations")


if __name__ == "__main__":
    asyncio.run(main())
