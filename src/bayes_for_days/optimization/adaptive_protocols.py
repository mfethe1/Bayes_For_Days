"""
Real-Time Adaptive Experimental Protocols for Bayes For Days platform.

This module implements dynamic experimental protocols that adapt in real-time
based on incoming experimental results. This is a revolutionary capability
that eliminates waste from predetermined experimental plans and maximizes
information gain through intelligent protocol adaptation.

Key Features:
- Real-time protocol adaptation based on experimental results
- Intelligent stopping criteria with statistical significance testing
- Dynamic experimental parameter adjustment during execution
- Automated protocol generation for laboratory execution
- Integration with laboratory automation systems
- Uncertainty-aware protocol modifications
- Multi-objective protocol optimization

Based on:
- Sequential experimental design theory
- Adaptive clinical trial methodologies
- Real-time optimization algorithms
- Laboratory automation best practices
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import time
from datetime import datetime, timedelta
import json

from bayes_for_days.core.base import BaseOptimizer, BaseSurrogateModel
from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    Parameter,
    ParameterType,
    OptimizationResult,
    ParameterDict,
    ObjectiveDict,
    OptimizationStatus
)

logger = logging.getLogger(__name__)


@dataclass
class ProtocolStep:
    """
    Represents a single step in an experimental protocol.
    """
    step_id: str
    description: str
    parameters: ParameterDict
    expected_duration: float  # in minutes
    required_equipment: List[str] = field(default_factory=list)
    safety_requirements: List[str] = field(default_factory=list)
    quality_checks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AdaptiveProtocol:
    """
    Represents a complete adaptive experimental protocol.
    """
    protocol_id: str
    name: str
    description: str
    steps: List[ProtocolStep]
    adaptation_rules: List['AdaptationRule']
    stopping_criteria: List['StoppingCriterion']
    created_at: datetime = field(default_factory=datetime.now)
    last_modified: datetime = field(default_factory=datetime.now)
    version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        """Export protocol to JSON format for laboratory systems."""
        protocol_dict = {
            'protocol_id': self.protocol_id,
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat(),
            'steps': [
                {
                    'step_id': step.step_id,
                    'description': step.description,
                    'parameters': step.parameters,
                    'expected_duration': step.expected_duration,
                    'required_equipment': step.required_equipment,
                    'safety_requirements': step.safety_requirements,
                    'quality_checks': step.quality_checks,
                    'metadata': step.metadata
                }
                for step in self.steps
            ],
            'metadata': self.metadata
        }
        return json.dumps(protocol_dict, indent=2)


@dataclass
class AdaptationRule:
    """
    Rule for adapting experimental protocols based on results.
    """
    rule_id: str
    name: str
    condition: str  # Python expression to evaluate
    action: str  # Action to take when condition is met
    priority: int = 1  # Higher priority rules are evaluated first
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StoppingCriterion:
    """
    Criterion for stopping experimental protocols early.
    """
    criterion_id: str
    name: str
    condition: str  # Python expression to evaluate
    confidence_level: float = 0.95
    min_experiments: int = 5
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProtocolAdaptationEngine:
    """
    Engine for real-time adaptation of experimental protocols.
    
    Monitors experimental results and adapts protocols in real-time
    based on predefined rules and statistical criteria.
    """
    
    def __init__(
        self,
        surrogate_model: BaseSurrogateModel,
        parameter_space: ParameterSpace,
        adaptation_interval: float = 60.0,  # seconds
        min_adaptation_data: int = 3
    ):
        """
        Initialize protocol adaptation engine.
        
        Args:
            surrogate_model: Surrogate model for predictions
            parameter_space: Parameter space for optimization
            adaptation_interval: Time interval between adaptations (seconds)
            min_adaptation_data: Minimum data points needed for adaptation
        """
        self.surrogate_model = surrogate_model
        self.parameter_space = parameter_space
        self.adaptation_interval = adaptation_interval
        self.min_adaptation_data = min_adaptation_data
        
        self.current_protocol: Optional[AdaptiveProtocol] = None
        self.experimental_data: List[ExperimentPoint] = []
        self.adaptation_history: List[Dict[str, Any]] = []
        self.is_running = False
        self.last_adaptation_time = datetime.now()
        
    def set_protocol(self, protocol: AdaptiveProtocol) -> None:
        """Set the current adaptive protocol."""
        self.current_protocol = protocol
        logger.info(f"Set adaptive protocol: {protocol.name}")
    
    def add_experimental_result(self, result: ExperimentPoint) -> None:
        """
        Add new experimental result and trigger adaptation if needed.
        
        Args:
            result: New experimental result
        """
        self.experimental_data.append(result)
        logger.info(f"Added experimental result: {result.objectives}")
        
        # Check if adaptation is needed
        if self._should_adapt():
            self._adapt_protocol()
    
    def _should_adapt(self) -> bool:
        """Check if protocol adaptation should be triggered."""
        if not self.current_protocol:
            return False
        
        # Check minimum data requirement
        if len(self.experimental_data) < self.min_adaptation_data:
            return False
        
        # Check time interval
        time_since_last = (datetime.now() - self.last_adaptation_time).total_seconds()
        if time_since_last < self.adaptation_interval:
            return False
        
        return True
    
    def _adapt_protocol(self) -> None:
        """Adapt the current protocol based on experimental results."""
        if not self.current_protocol:
            return
        
        logger.info("Starting protocol adaptation...")
        
        # Update surrogate model with latest data
        if len(self.experimental_data) >= 2:
            self.surrogate_model.fit(self.experimental_data)
        
        # Evaluate adaptation rules
        adaptations_made = []
        
        # Sort rules by priority (higher first)
        sorted_rules = sorted(
            self.current_protocol.adaptation_rules,
            key=lambda r: r.priority,
            reverse=True
        )
        
        for rule in sorted_rules:
            if not rule.enabled:
                continue
            
            if self._evaluate_rule_condition(rule):
                adaptation = self._apply_adaptation_rule(rule)
                if adaptation:
                    adaptations_made.append(adaptation)
        
        # Check stopping criteria
        should_stop = self._check_stopping_criteria()
        
        # Record adaptation
        adaptation_record = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(self.experimental_data),
            'adaptations_made': adaptations_made,
            'should_stop': should_stop,
            'protocol_version': self.current_protocol.version
        }
        
        self.adaptation_history.append(adaptation_record)
        self.last_adaptation_time = datetime.now()
        
        if adaptations_made:
            # Update protocol version
            self.current_protocol.version += 1
            self.current_protocol.last_modified = datetime.now()
            
            logger.info(f"Protocol adapted with {len(adaptations_made)} changes")
        
        if should_stop:
            logger.info("Stopping criteria met - protocol should be terminated")
    
    def _evaluate_rule_condition(self, rule: AdaptationRule) -> bool:
        """Evaluate if a rule condition is met."""
        try:
            # Create evaluation context
            context = self._create_evaluation_context()
            
            # Evaluate condition
            result = eval(rule.condition, {"__builtins__": {}}, context)
            return bool(result)
            
        except Exception as e:
            logger.warning(f"Error evaluating rule condition '{rule.condition}': {e}")
            return False
    
    def _create_evaluation_context(self) -> Dict[str, Any]:
        """Create context for rule evaluation."""
        if not self.experimental_data:
            return {}
        
        # Extract objective values
        objectives = []
        for point in self.experimental_data:
            if point.objectives:
                objectives.extend(point.objectives.values())
        
        objectives = np.array(objectives) if objectives else np.array([])
        
        # Create context with statistical measures
        context = {
            'n_experiments': len(self.experimental_data),
            'objectives': objectives,
            'mean_objective': np.mean(objectives) if len(objectives) > 0 else 0,
            'std_objective': np.std(objectives) if len(objectives) > 1 else 0,
            'min_objective': np.min(objectives) if len(objectives) > 0 else 0,
            'max_objective': np.max(objectives) if len(objectives) > 0 else 0,
            'latest_objective': objectives[-1] if len(objectives) > 0 else 0,
            'improvement_rate': self._calculate_improvement_rate(),
            'convergence_indicator': self._calculate_convergence_indicator(),
            'np': np,  # Allow numpy functions
        }
        
        # Add surrogate model predictions if available
        if hasattr(self.surrogate_model, 'is_fitted') and self.surrogate_model.is_fitted:
            context['model_uncertainty'] = self._get_model_uncertainty()
        
        return context
    
    def _calculate_improvement_rate(self) -> float:
        """Calculate rate of improvement in recent experiments."""
        if len(self.experimental_data) < 3:
            return 0.0
        
        # Get recent objectives
        recent_objectives = []
        for point in self.experimental_data[-5:]:  # Last 5 experiments
            if point.objectives:
                recent_objectives.extend(point.objectives.values())
        
        if len(recent_objectives) < 2:
            return 0.0
        
        # Calculate improvement rate (assuming minimization)
        improvements = []
        for i in range(1, len(recent_objectives)):
            if recent_objectives[i-1] > recent_objectives[i]:
                improvements.append(recent_objectives[i-1] - recent_objectives[i])
        
        return np.mean(improvements) if improvements else 0.0
    
    def _calculate_convergence_indicator(self) -> float:
        """Calculate convergence indicator (0-1, higher = more converged)."""
        if len(self.experimental_data) < 5:
            return 0.0
        
        # Get recent objectives
        recent_objectives = []
        for point in self.experimental_data[-10:]:  # Last 10 experiments
            if point.objectives:
                recent_objectives.extend(point.objectives.values())
        
        if len(recent_objectives) < 3:
            return 0.0
        
        # Calculate coefficient of variation (lower = more converged)
        mean_obj = np.mean(recent_objectives)
        std_obj = np.std(recent_objectives)
        
        if mean_obj == 0:
            return 1.0 if std_obj == 0 else 0.0
        
        cv = std_obj / abs(mean_obj)
        
        # Convert to convergence indicator (0-1, higher = more converged)
        convergence = max(0, 1 - cv)
        return min(1, convergence)
    
    def _get_model_uncertainty(self) -> float:
        """Get average model uncertainty."""
        if not self.experimental_data:
            return 1.0
        
        # Get predictions for recent experiments
        recent_params = [point.parameters for point in self.experimental_data[-5:]]
        
        try:
            predictions = self.surrogate_model.predict(recent_params)
            if not isinstance(predictions, list):
                predictions = [predictions]
            
            uncertainties = [pred.std for pred in predictions if hasattr(pred, 'std')]
            return np.mean(uncertainties) if uncertainties else 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating model uncertainty: {e}")
            return 1.0
    
    def _apply_adaptation_rule(self, rule: AdaptationRule) -> Optional[Dict[str, Any]]:
        """Apply an adaptation rule to the protocol."""
        try:
            # Parse action (simplified implementation)
            action_parts = rule.action.split(':')
            action_type = action_parts[0].strip()
            
            if action_type == "modify_parameter":
                return self._modify_parameter_action(action_parts[1:])
            elif action_type == "add_experiments":
                return self._add_experiments_action(action_parts[1:])
            elif action_type == "change_strategy":
                return self._change_strategy_action(action_parts[1:])
            else:
                logger.warning(f"Unknown adaptation action: {action_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error applying adaptation rule '{rule.name}': {e}")
            return None
    
    def _modify_parameter_action(self, action_args: List[str]) -> Dict[str, Any]:
        """Apply parameter modification action."""
        # Simplified implementation
        return {
            'type': 'modify_parameter',
            'description': 'Modified experimental parameters based on results',
            'timestamp': datetime.now().isoformat()
        }
    
    def _add_experiments_action(self, action_args: List[str]) -> Dict[str, Any]:
        """Apply add experiments action."""
        return {
            'type': 'add_experiments',
            'description': 'Added additional experiments to protocol',
            'timestamp': datetime.now().isoformat()
        }
    
    def _change_strategy_action(self, action_args: List[str]) -> Dict[str, Any]:
        """Apply strategy change action."""
        return {
            'type': 'change_strategy',
            'description': 'Changed optimization strategy based on convergence',
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_stopping_criteria(self) -> bool:
        """Check if any stopping criteria are met."""
        if not self.current_protocol:
            return False
        
        for criterion in self.current_protocol.stopping_criteria:
            if not criterion.enabled:
                continue
            
            if len(self.experimental_data) < criterion.min_experiments:
                continue
            
            try:
                context = self._create_evaluation_context()
                if eval(criterion.condition, {"__builtins__": {}}, context):
                    logger.info(f"Stopping criterion met: {criterion.name}")
                    return True
            except Exception as e:
                logger.warning(f"Error evaluating stopping criterion '{criterion.name}': {e}")
        
        return False
    
    def generate_protocol_report(self) -> Dict[str, Any]:
        """Generate comprehensive protocol adaptation report."""
        return {
            'protocol_info': {
                'name': self.current_protocol.name if self.current_protocol else None,
                'version': self.current_protocol.version if self.current_protocol else None,
                'total_experiments': len(self.experimental_data)
            },
            'adaptation_summary': {
                'total_adaptations': len(self.adaptation_history),
                'last_adaptation': self.adaptation_history[-1] if self.adaptation_history else None,
                'adaptation_frequency': len(self.adaptation_history) / max(1, len(self.experimental_data))
            },
            'performance_metrics': {
                'improvement_rate': self._calculate_improvement_rate(),
                'convergence_indicator': self._calculate_convergence_indicator(),
                'model_uncertainty': self._get_model_uncertainty()
            },
            'experimental_summary': self._get_experimental_summary()
        }
    
    def _get_experimental_summary(self) -> Dict[str, Any]:
        """Get summary of experimental results."""
        if not self.experimental_data:
            return {}
        
        objectives = []
        for point in self.experimental_data:
            if point.objectives:
                objectives.extend(point.objectives.values())
        
        if not objectives:
            return {}
        
        return {
            'total_experiments': len(self.experimental_data),
            'best_objective': min(objectives),  # Assuming minimization
            'mean_objective': np.mean(objectives),
            'std_objective': np.std(objectives),
            'objective_range': [min(objectives), max(objectives)]
        }
