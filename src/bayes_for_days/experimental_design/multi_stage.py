"""
Multi-stage experimental design workflows for sequential optimization.

This module implements sequential experimental design with:
- Stage-wise optimization strategies
- Adaptive stopping criteria
- Integration with Bayesian optimization loops
- Progressive refinement of experimental regions
- Multi-fidelity experimental workflows

Based on:
- Jones et al. (1998) "Efficient Global Optimization of Expensive Black-Box Functions"
- Forrester & Keane (2009) "Recent advances in surrogate-based optimization"
- Sequential experimental design methodologies
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from bayes_for_days.experimental_design.variables import ExperimentalVariable
from bayes_for_days.experimental_design.design_strategies import DesignStrategy
from bayes_for_days.experimental_design.adaptive import BayesianAdaptiveDesign
from bayes_for_days.core.types import ExperimentPoint

logger = logging.getLogger(__name__)


class StageType(Enum):
    """Types of experimental stages."""
    SCREENING = "screening"
    OPTIMIZATION = "optimization"
    CONFIRMATION = "confirmation"
    REFINEMENT = "refinement"
    EXPLORATION = "exploration"


class StoppingCriterion(Enum):
    """Stopping criteria for multi-stage designs."""
    MAX_EXPERIMENTS = "max_experiments"
    CONVERGENCE = "convergence"
    IMPROVEMENT_THRESHOLD = "improvement_threshold"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TIME_LIMIT = "time_limit"
    USER_DEFINED = "user_defined"


@dataclass
class StageConfiguration:
    """
    Configuration for a single experimental stage.
    
    Defines the objectives, constraints, and termination
    criteria for each stage of the multi-stage design.
    """
    stage_id: str
    stage_type: StageType
    design_strategy: str  # Name of design strategy to use
    max_experiments: int
    min_experiments: int = 1
    stopping_criteria: List[StoppingCriterion] = field(default_factory=list)
    
    # Stage-specific parameters
    exploration_factor: float = 1.0  # Higher = more exploration
    exploitation_factor: float = 1.0  # Higher = more exploitation
    region_reduction_factor: float = 0.8  # Factor to reduce search region
    
    # Transition criteria
    transition_threshold: float = 0.01  # Improvement threshold for stage transition
    min_improvement_experiments: int = 5  # Min experiments without improvement
    
    # Resource constraints
    max_budget: Optional[float] = None
    max_time_hours: Optional[float] = None
    
    def __post_init__(self):
        """Validate stage configuration."""
        if self.min_experiments > self.max_experiments:
            raise ValueError("min_experiments cannot exceed max_experiments")
        
        if not self.stopping_criteria:
            self.stopping_criteria = [StoppingCriterion.MAX_EXPERIMENTS]


class ExperimentalStage:
    """
    Individual experimental stage in multi-stage workflow.
    
    Manages execution of a single stage including experiment
    selection, data collection, and termination decisions.
    """
    
    def __init__(
        self,
        config: StageConfiguration,
        variables: List[ExperimentalVariable],
        surrogate_model: Any,
        **kwargs
    ):
        """
        Initialize experimental stage.
        
        Args:
            config: Stage configuration
            variables: List of experimental variables
            surrogate_model: Surrogate model for predictions
            **kwargs: Additional parameters
        """
        self.config = config
        self.variables = variables
        self.surrogate_model = surrogate_model
        
        # Stage state
        self.is_active = False
        self.is_complete = False
        self.experiments_conducted = 0
        self.stage_data: List[ExperimentPoint] = []
        
        # Performance tracking
        self.best_objective_value = None
        self.improvement_history: List[float] = []
        self.experiments_since_improvement = 0
        
        # Current search region (can be refined during stage)
        self.current_bounds = self._initialize_bounds()
        
        logger.info(f"Initialized stage {config.stage_id} ({config.stage_type.value})")
    
    def _initialize_bounds(self) -> List[Tuple[float, float]]:
        """Initialize search bounds for this stage."""
        bounds = []
        for var in self.variables:
            if var.bounds:
                bounds.append(var.bounds)
            elif var.baseline_value is not None:
                # Create bounds around baseline
                baseline = var.baseline_value
                range_val = abs(baseline) * 0.5  # 50% range around baseline
                bounds.append((baseline - range_val, baseline + range_val))
            else:
                bounds.append((-1.0, 1.0))  # Default bounds
        
        return bounds
    
    def start_stage(self):
        """Start the experimental stage."""
        self.is_active = True
        self.is_complete = False
        logger.info(f"Started stage {self.config.stage_id}")
    
    def conduct_experiments(self, n_experiments: int) -> List[ExperimentPoint]:
        """
        Conduct experiments for this stage.
        
        Args:
            n_experiments: Number of experiments to conduct
            
        Returns:
            List of experiment results
        """
        if not self.is_active:
            raise ValueError("Stage must be started before conducting experiments")
        
        # Generate experimental design
        design_points = self._generate_stage_design(n_experiments)
        
        # Simulate experiment execution (in practice, would be real experiments)
        experiment_results = self._execute_experiments(design_points)
        
        # Update stage data and state
        self.stage_data.extend(experiment_results)
        self.experiments_conducted += len(experiment_results)
        
        # Update performance tracking
        self._update_performance_tracking(experiment_results)
        
        # Check stopping criteria
        if self._check_stopping_criteria():
            self.complete_stage()
        
        return experiment_results
    
    def _generate_stage_design(self, n_experiments: int) -> np.ndarray:
        """Generate experimental design for this stage."""
        # Select design strategy based on stage type
        if self.config.stage_type == StageType.SCREENING:
            # Use space-filling design for screening
            design = self._generate_screening_design(n_experiments)
        elif self.config.stage_type == StageType.OPTIMIZATION:
            # Use adaptive design for optimization
            design = self._generate_optimization_design(n_experiments)
        elif self.config.stage_type == StageType.CONFIRMATION:
            # Use confirmation design around best points
            design = self._generate_confirmation_design(n_experiments)
        else:
            # Default to random sampling
            design = self._generate_random_design(n_experiments)
        
        return design
    
    def _generate_screening_design(self, n_experiments: int) -> np.ndarray:
        """Generate screening design for factor identification."""
        # Simple Latin Hypercube-like design
        design = []
        
        for _ in range(n_experiments):
            experiment = []
            for i, (low, high) in enumerate(self.current_bounds):
                value = np.random.uniform(low, high)
                experiment.append(value)
            design.append(experiment)
        
        return np.array(design)
    
    def _generate_optimization_design(self, n_experiments: int) -> np.ndarray:
        """Generate optimization design using surrogate model."""
        # Use model-based design (simplified)
        design = []
        
        for _ in range(n_experiments):
            # Generate candidate points
            candidates = []
            for _ in range(100):  # Generate 100 candidates
                candidate = []
                for low, high in self.current_bounds:
                    value = np.random.uniform(low, high)
                    candidate.append(value)
                candidates.append(candidate)
            
            # Select best candidate based on acquisition function (simplified)
            best_candidate = self._select_best_candidate(candidates)
            design.append(best_candidate)
        
        return np.array(design)
    
    def _generate_confirmation_design(self, n_experiments: int) -> np.ndarray:
        """Generate confirmation design around best points."""
        if not self.stage_data:
            return self._generate_random_design(n_experiments)
        
        # Find best experiment so far
        best_experiment = max(
            self.stage_data,
            key=lambda x: list(x.objectives.values())[0] if x.objectives else 0
        )
        
        # Generate points around best experiment
        design = []
        best_params = list(best_experiment.parameters.values())
        
        for _ in range(n_experiments):
            # Add noise to best parameters
            experiment = []
            for i, param_value in enumerate(best_params):
                if i < len(self.current_bounds):
                    low, high = self.current_bounds[i]
                    noise_std = (high - low) * 0.05  # 5% of range
                    noisy_value = np.random.normal(param_value, noise_std)
                    noisy_value = np.clip(noisy_value, low, high)
                    experiment.append(noisy_value)
            design.append(experiment)
        
        return np.array(design)
    
    def _generate_random_design(self, n_experiments: int) -> np.ndarray:
        """Generate random design within current bounds."""
        design = []
        
        for _ in range(n_experiments):
            experiment = []
            for low, high in self.current_bounds:
                value = np.random.uniform(low, high)
                experiment.append(value)
            design.append(experiment)
        
        return np.array(design)
    
    def _select_best_candidate(self, candidates: List[List[float]]) -> List[float]:
        """Select best candidate using acquisition function (simplified)."""
        # Simplified selection based on model prediction
        best_candidate = candidates[0]
        best_score = float('-inf')
        
        for candidate in candidates:
            # Convert to parameter dict
            param_dict = {
                var.name: candidate[i] 
                for i, var in enumerate(self.variables) 
                if i < len(candidate)
            }
            
            try:
                # Get model prediction
                prediction = self.surrogate_model.predict(param_dict)
                if hasattr(prediction, 'mean'):
                    score = prediction.mean
                else:
                    score = 0.0
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
                    
            except Exception:
                continue
        
        return best_candidate
    
    def _execute_experiments(self, design_points: np.ndarray) -> List[ExperimentPoint]:
        """Execute experiments (simulation for now)."""
        results = []
        
        for i, point in enumerate(design_points):
            # Convert to parameter dict
            param_dict = {
                var.name: point[j] 
                for j, var in enumerate(self.variables) 
                if j < len(point)
            }
            
            # Simulate experiment result
            try:
                prediction = self.surrogate_model.predict(param_dict)
                if hasattr(prediction, 'mean') and hasattr(prediction, 'std'):
                    # Add noise to prediction
                    result_value = np.random.normal(prediction.mean, prediction.std * 0.1)
                else:
                    result_value = np.random.normal(0, 1)
                
                experiment_point = ExperimentPoint(
                    parameters=param_dict,
                    objectives={"objective": result_value},
                    is_feasible=True,
                    metadata={
                        "stage_id": self.config.stage_id,
                        "stage_type": self.config.stage_type.value,
                        "experiment_index": i
                    }
                )
                results.append(experiment_point)
                
            except Exception as e:
                logger.warning(f"Experiment simulation failed: {e}")
        
        return results
    
    def _update_performance_tracking(self, new_experiments: List[ExperimentPoint]):
        """Update performance tracking with new experiments."""
        for exp in new_experiments:
            if exp.objectives:
                obj_value = list(exp.objectives.values())[0]
                
                # Update best objective value
                if self.best_objective_value is None or obj_value > self.best_objective_value:
                    improvement = (obj_value - self.best_objective_value 
                                 if self.best_objective_value is not None else obj_value)
                    self.improvement_history.append(improvement)
                    self.best_objective_value = obj_value
                    self.experiments_since_improvement = 0
                else:
                    self.experiments_since_improvement += 1
    
    def _check_stopping_criteria(self) -> bool:
        """Check if any stopping criteria are met."""
        for criterion in self.config.stopping_criteria:
            if criterion == StoppingCriterion.MAX_EXPERIMENTS:
                if self.experiments_conducted >= self.config.max_experiments:
                    logger.info(f"Stage {self.config.stage_id}: Max experiments reached")
                    return True
            
            elif criterion == StoppingCriterion.IMPROVEMENT_THRESHOLD:
                if (self.experiments_since_improvement >= self.config.min_improvement_experiments):
                    logger.info(f"Stage {self.config.stage_id}: No improvement threshold reached")
                    return True
            
            elif criterion == StoppingCriterion.CONVERGENCE:
                if len(self.improvement_history) >= 5:
                    recent_improvements = self.improvement_history[-5:]
                    if all(imp < self.config.transition_threshold for imp in recent_improvements):
                        logger.info(f"Stage {self.config.stage_id}: Convergence achieved")
                        return True
        
        return False
    
    def complete_stage(self):
        """Complete the experimental stage."""
        self.is_active = False
        self.is_complete = True
        logger.info(f"Completed stage {self.config.stage_id} with {self.experiments_conducted} experiments")
    
    def refine_search_region(self):
        """Refine search region based on current results."""
        if not self.stage_data:
            return
        
        # Find best experiments
        sorted_experiments = sorted(
            self.stage_data,
            key=lambda x: list(x.objectives.values())[0] if x.objectives else 0,
            reverse=True
        )
        
        # Take top 20% of experiments
        n_top = max(1, len(sorted_experiments) // 5)
        top_experiments = sorted_experiments[:n_top]
        
        # Calculate new bounds around top experiments
        new_bounds = []
        
        for i, var in enumerate(self.variables):
            param_values = []
            for exp in top_experiments:
                if var.name in exp.parameters:
                    param_values.append(exp.parameters[var.name])
            
            if param_values:
                min_val = min(param_values)
                max_val = max(param_values)
                
                # Expand slightly
                range_expansion = (max_val - min_val) * 0.1
                new_min = min_val - range_expansion
                new_max = max_val + range_expansion
                
                # Ensure within original bounds
                orig_min, orig_max = self.current_bounds[i]
                new_min = max(new_min, orig_min)
                new_max = min(new_max, orig_max)
                
                new_bounds.append((new_min, new_max))
            else:
                new_bounds.append(self.current_bounds[i])
        
        self.current_bounds = new_bounds
        logger.info(f"Refined search region for stage {self.config.stage_id}")
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary of stage performance."""
        return {
            'stage_id': self.config.stage_id,
            'stage_type': self.config.stage_type.value,
            'experiments_conducted': self.experiments_conducted,
            'is_complete': self.is_complete,
            'best_objective_value': self.best_objective_value,
            'total_improvement': sum(self.improvement_history),
            'experiments_since_improvement': self.experiments_since_improvement,
            'current_bounds': self.current_bounds,
        }


class MultiStageWorkflow:
    """
    Multi-stage experimental design workflow manager.
    
    Coordinates execution of multiple experimental stages
    with adaptive transitions and resource management.
    """
    
    def __init__(
        self,
        stages: List[StageConfiguration],
        variables: List[ExperimentalVariable],
        surrogate_model: Any,
        **kwargs
    ):
        """
        Initialize multi-stage workflow.
        
        Args:
            stages: List of stage configurations
            variables: List of experimental variables
            surrogate_model: Surrogate model for predictions
            **kwargs: Additional parameters
        """
        self.stage_configs = stages
        self.variables = variables
        self.surrogate_model = surrogate_model
        self.config = kwargs
        
        # Initialize stages
        self.stages = [
            ExperimentalStage(config, variables, surrogate_model)
            for config in stages
        ]
        
        # Workflow state
        self.current_stage_index = 0
        self.is_complete = False
        self.all_experiment_data: List[ExperimentPoint] = []
        
        # Performance tracking
        self.workflow_start_time = None
        self.total_experiments = 0
        self.total_budget_used = 0.0
        
        logger.info(f"Initialized multi-stage workflow with {len(stages)} stages")
    
    def execute_workflow(self, max_total_experiments: Optional[int] = None) -> List[ExperimentPoint]:
        """
        Execute the complete multi-stage workflow.
        
        Args:
            max_total_experiments: Maximum total experiments across all stages
            
        Returns:
            All experiment results from workflow
        """
        self.workflow_start_time = datetime.now()
        
        while not self.is_complete and self.current_stage_index < len(self.stages):
            current_stage = self.stages[self.current_stage_index]
            
            # Start stage
            current_stage.start_stage()
            
            # Execute stage experiments
            stage_experiments = self._execute_stage(current_stage, max_total_experiments)
            self.all_experiment_data.extend(stage_experiments)
            self.total_experiments += len(stage_experiments)
            
            # Update surrogate model with new data
            self._update_surrogate_model(stage_experiments)
            
            # Check for workflow termination
            if max_total_experiments and self.total_experiments >= max_total_experiments:
                logger.info("Maximum total experiments reached")
                break
            
            # Transition to next stage
            self._transition_to_next_stage()
        
        self.is_complete = True
        logger.info(f"Completed multi-stage workflow with {self.total_experiments} total experiments")
        
        return self.all_experiment_data
    
    def _execute_stage(
        self, 
        stage: ExperimentalStage, 
        max_total_experiments: Optional[int]
    ) -> List[ExperimentPoint]:
        """Execute a single stage."""
        stage_experiments = []
        
        while not stage.is_complete:
            # Determine batch size for this iteration
            remaining_stage_experiments = stage.config.max_experiments - stage.experiments_conducted
            
            if max_total_experiments:
                remaining_total_experiments = max_total_experiments - self.total_experiments
                batch_size = min(5, remaining_stage_experiments, remaining_total_experiments)
            else:
                batch_size = min(5, remaining_stage_experiments)
            
            if batch_size <= 0:
                break
            
            # Conduct batch of experiments
            batch_results = stage.conduct_experiments(batch_size)
            stage_experiments.extend(batch_results)
            
            # Check if we should refine search region
            if (stage.config.stage_type in [StageType.OPTIMIZATION, StageType.REFINEMENT] and
                stage.experiments_conducted % 10 == 0):  # Every 10 experiments
                stage.refine_search_region()
        
        return stage_experiments
    
    def _update_surrogate_model(self, new_experiments: List[ExperimentPoint]):
        """Update surrogate model with new experimental data."""
        try:
            if hasattr(self.surrogate_model, 'update'):
                self.surrogate_model.update(new_experiments)
            elif hasattr(self.surrogate_model, 'fit'):
                # Refit with all data
                self.surrogate_model.fit(self.all_experiment_data + new_experiments)
        except Exception as e:
            logger.warning(f"Failed to update surrogate model: {e}")
    
    def _transition_to_next_stage(self):
        """Transition to the next stage in the workflow."""
        if self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            logger.info(f"Transitioning to stage {self.current_stage_index + 1}")
        else:
            logger.info("All stages completed")
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of entire workflow."""
        stage_summaries = [stage.get_stage_summary() for stage in self.stages]
        
        # Find overall best result
        best_objective = None
        if self.all_experiment_data:
            best_experiment = max(
                self.all_experiment_data,
                key=lambda x: list(x.objectives.values())[0] if x.objectives else 0
            )
            best_objective = list(best_experiment.objectives.values())[0] if best_experiment.objectives else None
        
        return {
            'total_experiments': self.total_experiments,
            'total_stages': len(self.stages),
            'completed_stages': sum(1 for stage in self.stages if stage.is_complete),
            'best_objective_value': best_objective,
            'workflow_complete': self.is_complete,
            'stage_summaries': stage_summaries,
            'execution_time_hours': (
                (datetime.now() - self.workflow_start_time).total_seconds() / 3600
                if self.workflow_start_time else 0
            ),
        }
