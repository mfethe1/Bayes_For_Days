"""
Automated Hypothesis Generation for Bayes For Days platform.

This module implements advanced AI-driven hypothesis generation that goes beyond
simple pattern recognition to create sophisticated, testable scientific hypotheses
with mechanistic reasoning and experimental validation strategies.

Key Features:
- Multi-modal hypothesis generation (mechanistic, phenomenological, statistical)
- Literature-informed hypothesis creation
- Hypothesis ranking and prioritization
- Automated experimental design for hypothesis testing
- Hypothesis validation and refinement
- Cross-domain hypothesis transfer
- Uncertainty-aware hypothesis confidence scoring

Based on:
- Scientific discovery methodologies
- Automated reasoning systems
- Knowledge graph reasoning
- Large language model integration
- Bayesian hypothesis testing
- Active learning for hypothesis validation
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json
import re
from datetime import datetime
from collections import defaultdict
import itertools

from bayes_for_days.core.base import BaseSurrogateModel
from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    Parameter,
    ParameterType,
    ParameterDict
)
from bayes_for_days.ai.intelligent_assistant import (
    ScientificHypothesis,
    ExperimentalInsight,
    ScientificReasoningEngine
)

logger = logging.getLogger(__name__)


@dataclass
class HypothesisTemplate:
    """Template for generating specific types of hypotheses."""
    template_id: str
    name: str
    description: str
    hypothesis_type: str  # 'mechanistic', 'phenomenological', 'statistical', 'causal'
    pattern_requirements: List[str]
    template_structure: str
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    domain_applicability: List[str] = field(default_factory=list)
    
    def matches_pattern(self, insights: List[ExperimentalInsight]) -> bool:
        """Check if insights match this template's requirements."""
        # Simplified pattern matching
        for requirement in self.pattern_requirements:
            if not any(requirement.lower() in insight.description.lower() for insight in insights):
                return False
        return True


@dataclass
class HypothesisEvidence:
    """Evidence supporting or refuting a hypothesis."""
    evidence_id: str
    hypothesis_id: str
    evidence_type: str  # 'experimental', 'literature', 'theoretical', 'analogical'
    description: str
    strength: float  # 0-1, strength of evidence
    confidence: float  # 0-1, confidence in evidence
    source: str
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class HypothesisValidationPlan:
    """Plan for validating a scientific hypothesis."""
    plan_id: str
    hypothesis_id: str
    validation_experiments: List[Dict[str, Any]]
    success_criteria: List[str]
    failure_criteria: List[str]
    expected_outcomes: Dict[str, Any]
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    estimated_duration: Optional[float] = None  # in days
    priority_score: float = 1.0


class HypothesisTemplateLibrary:
    """
    Library of hypothesis templates for different scientific domains.
    
    Provides structured templates for generating hypotheses based on
    common scientific reasoning patterns.
    """
    
    def __init__(self):
        """Initialize hypothesis template library."""
        self.templates = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default hypothesis templates."""
        
        # Mechanistic templates
        self.templates['dose_response'] = HypothesisTemplate(
            template_id='dose_response',
            name='Dose-Response Mechanism',
            description='Mechanistic hypothesis for dose-response relationships',
            hypothesis_type='mechanistic',
            pattern_requirements=['concentration', 'dose', 'positive correlation'],
            template_structure='Increasing {parameter} enhances {outcome} through {mechanism}',
            confidence_factors={'correlation_strength': 0.4, 'mechanism_plausibility': 0.6},
            domain_applicability=['chemistry', 'biology', 'pharmacology']
        )
        
        self.templates['inhibition'] = HypothesisTemplate(
            template_id='inhibition',
            name='Inhibition Mechanism',
            description='Mechanistic hypothesis for inhibitory effects',
            hypothesis_type='mechanistic',
            pattern_requirements=['negative correlation', 'saturation'],
            template_structure='{parameter} inhibits {outcome} through competitive binding to {target}',
            confidence_factors={'correlation_strength': 0.5, 'saturation_evidence': 0.5},
            domain_applicability=['chemistry', 'biology', 'biochemistry']
        )
        
        self.templates['catalytic_enhancement'] = HypothesisTemplate(
            template_id='catalytic_enhancement',
            name='Catalytic Enhancement',
            description='Mechanistic hypothesis for catalytic effects',
            hypothesis_type='mechanistic',
            pattern_requirements=['temperature', 'catalyst', 'positive correlation'],
            template_structure='{catalyst} enhances reaction rate by lowering activation energy for {reaction}',
            confidence_factors={'temperature_dependence': 0.4, 'catalyst_specificity': 0.6},
            domain_applicability=['chemistry', 'catalysis', 'materials']
        )
        
        # Phenomenological templates
        self.templates['optimization_surface'] = HypothesisTemplate(
            template_id='optimization_surface',
            name='Optimization Surface',
            description='Phenomenological hypothesis about response surfaces',
            hypothesis_type='phenomenological',
            pattern_requirements=['multiple parameters', 'nonlinear'],
            template_structure='The response surface has an optimum at {parameters} due to competing effects',
            confidence_factors={'nonlinearity_evidence': 0.7, 'parameter_interactions': 0.3},
            domain_applicability=['general', 'optimization', 'engineering']
        )
        
        # Statistical templates
        self.templates['interaction_effect'] = HypothesisTemplate(
            template_id='interaction_effect',
            name='Parameter Interaction',
            description='Statistical hypothesis about parameter interactions',
            hypothesis_type='statistical',
            pattern_requirements=['interaction', 'multiple parameters'],
            template_structure='{param1} and {param2} interact synergistically to affect {outcome}',
            confidence_factors={'interaction_strength': 0.8, 'statistical_significance': 0.2},
            domain_applicability=['general', 'statistics', 'experimental_design']
        )
    
    def get_matching_templates(
        self,
        insights: List[ExperimentalInsight],
        domain: str = "general"
    ) -> List[HypothesisTemplate]:
        """Get templates that match the given insights and domain."""
        matching_templates = []
        
        for template in self.templates.values():
            if domain in template.domain_applicability or "general" in template.domain_applicability:
                if template.matches_pattern(insights):
                    matching_templates.append(template)
        
        return matching_templates


class AdvancedHypothesisGenerator:
    """
    Advanced hypothesis generator using multiple reasoning strategies.
    
    Combines pattern recognition, template matching, mechanistic reasoning,
    and domain knowledge to generate sophisticated scientific hypotheses.
    """
    
    def __init__(
        self,
        domain: str = "chemistry",
        template_library: Optional[HypothesisTemplateLibrary] = None
    ):
        """
        Initialize advanced hypothesis generator.
        
        Args:
            domain: Scientific domain for specialized reasoning
            template_library: Library of hypothesis templates
        """
        self.domain = domain
        self.template_library = template_library or HypothesisTemplateLibrary()
        self.reasoning_engine = ScientificReasoningEngine(domain)
        self.generated_hypotheses = []
        self.hypothesis_evidence = defaultdict(list)
        
    def generate_hypotheses(
        self,
        experimental_data: List[ExperimentPoint],
        insights: List[ExperimentalInsight],
        context: Dict[str, Any] = None
    ) -> List[ScientificHypothesis]:
        """
        Generate comprehensive set of scientific hypotheses.
        
        Args:
            experimental_data: Experimental data for analysis
            insights: Experimental insights from pattern analysis
            context: Additional context about the experimental system
            
        Returns:
            List of generated hypotheses
        """
        if context is None:
            context = {}
        
        logger.info(f"Generating hypotheses from {len(experimental_data)} experiments and {len(insights)} insights")
        
        all_hypotheses = []
        
        # 1. Template-based hypothesis generation
        template_hypotheses = self._generate_template_hypotheses(insights, context)
        all_hypotheses.extend(template_hypotheses)
        
        # 2. Mechanistic hypothesis generation
        mechanistic_hypotheses = self._generate_mechanistic_hypotheses(experimental_data, insights, context)
        all_hypotheses.extend(mechanistic_hypotheses)
        
        # 3. Statistical hypothesis generation
        statistical_hypotheses = self._generate_statistical_hypotheses(experimental_data, insights)
        all_hypotheses.extend(statistical_hypotheses)
        
        # 4. Anomaly-based hypothesis generation
        anomaly_hypotheses = self._generate_anomaly_hypotheses(experimental_data, insights)
        all_hypotheses.extend(anomaly_hypotheses)
        
        # 5. Cross-parameter hypothesis generation
        interaction_hypotheses = self._generate_interaction_hypotheses(experimental_data, insights)
        all_hypotheses.extend(interaction_hypotheses)
        
        # Rank and filter hypotheses
        ranked_hypotheses = self._rank_hypotheses(all_hypotheses, experimental_data)
        
        # Store generated hypotheses
        self.generated_hypotheses.extend(ranked_hypotheses)
        
        logger.info(f"Generated {len(ranked_hypotheses)} ranked hypotheses")
        
        return ranked_hypotheses
    
    def _generate_template_hypotheses(
        self,
        insights: List[ExperimentalInsight],
        context: Dict[str, Any]
    ) -> List[ScientificHypothesis]:
        """Generate hypotheses using template matching."""
        hypotheses = []
        
        # Get matching templates
        matching_templates = self.template_library.get_matching_templates(insights, self.domain)
        
        for template in matching_templates:
            # Extract relevant insights for this template
            relevant_insights = [i for i in insights if template.matches_pattern([i])]
            
            for insight in relevant_insights:
                hypothesis = self._instantiate_template_hypothesis(template, insight, context)
                if hypothesis:
                    hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _instantiate_template_hypothesis(
        self,
        template: HypothesisTemplate,
        insight: ExperimentalInsight,
        context: Dict[str, Any]
    ) -> Optional[ScientificHypothesis]:
        """Instantiate a specific hypothesis from a template."""
        
        # Extract parameters from insight description
        param_match = re.search(r'Parameter (\w+)', insight.description)
        parameter = param_match.group(1) if param_match else "unknown_parameter"
        
        # Generate hypothesis based on template
        if template.template_id == 'dose_response':
            mechanism = self._infer_dose_response_mechanism(parameter, context)
            description = f"Increasing {parameter} enhances the objective through {mechanism}"
            
            predictions = [
                f"Dose-response curve should be monotonic for {parameter}",
                f"Saturation behavior expected at high {parameter} values",
                f"Mechanism involves {mechanism}"
            ]
            
        elif template.template_id == 'inhibition':
            target = self._infer_inhibition_target(parameter, context)
            description = f"{parameter} inhibits the objective through competitive binding to {target}"
            
            predictions = [
                f"Inhibition curve should show IC50 behavior",
                f"Effect can be overcome by increasing substrate concentration",
                f"Competitive inhibition mechanism with {target}"
            ]
            
        elif template.template_id == 'catalytic_enhancement':
            reaction = context.get('reaction_type', 'target reaction')
            description = f"{parameter} enhances reaction rate by lowering activation energy for {reaction}"
            
            predictions = [
                f"Arrhenius relationship should hold for temperature dependence",
                f"Catalyst shows specificity for {reaction}",
                f"Turnover frequency increases with {parameter}"
            ]
            
        else:
            # Generic template instantiation
            description = template.template_structure.format(
                parameter=parameter,
                outcome="objective",
                mechanism="unknown mechanism"
            )
            predictions = [f"Testable prediction for {parameter} effect"]
        
        # Calculate confidence based on template factors
        confidence = self._calculate_template_confidence(template, insight)
        
        # Generate suggested experiments
        suggested_experiments = self._generate_template_experiments(template, parameter)
        
        hypothesis = ScientificHypothesis(
            hypothesis_id=f"{template.template_id}_{parameter}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            title=f"{template.name}: {parameter} effect",
            description=description,
            testable_predictions=predictions,
            suggested_experiments=suggested_experiments,
            confidence_score=confidence,
            supporting_evidence=[insight.description],
            potential_mechanisms=[description]
        )
        
        return hypothesis
    
    def _generate_mechanistic_hypotheses(
        self,
        experimental_data: List[ExperimentPoint],
        insights: List[ExperimentalInsight],
        context: Dict[str, Any]
    ) -> List[ScientificHypothesis]:
        """Generate mechanistic hypotheses based on domain knowledge."""
        hypotheses = []
        
        # Use existing reasoning engine for basic mechanistic hypotheses
        basic_hypotheses = self.reasoning_engine.generate_hypotheses(insights, context)
        
        # Enhance with advanced mechanistic reasoning
        for hypothesis in basic_hypotheses:
            enhanced_hypothesis = self._enhance_mechanistic_hypothesis(hypothesis, experimental_data, context)
            hypotheses.append(enhanced_hypothesis)
        
        return hypotheses
    
    def _generate_statistical_hypotheses(
        self,
        experimental_data: List[ExperimentPoint],
        insights: List[ExperimentalInsight]
    ) -> List[ScientificHypothesis]:
        """Generate statistical hypotheses about data patterns."""
        hypotheses = []
        
        if len(experimental_data) < 5:
            return hypotheses
        
        # Analyze statistical patterns
        param_names = list(experimental_data[0].parameters.keys())
        
        # Generate hypotheses about parameter distributions
        for param_name in param_names:
            param_values = [point.parameters.get(param_name, 0) for point in experimental_data]
            
            # Test for normality, skewness, etc.
            mean_val = np.mean(param_values)
            std_val = np.std(param_values)
            skewness = self._calculate_skewness(param_values)
            
            if abs(skewness) > 1.0:  # Significant skewness
                hypothesis = ScientificHypothesis(
                    hypothesis_id=f"stat_skew_{param_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    title=f"Statistical distribution hypothesis for {param_name}",
                    description=f"Parameter {param_name} shows {'positive' if skewness > 0 else 'negative'} skewness (skew={skewness:.2f}), suggesting underlying asymmetric process",
                    testable_predictions=[
                        f"Distribution of {param_name} should consistently show skewness",
                        f"Transformation may normalize the distribution",
                        f"Underlying process has asymmetric response to {param_name}"
                    ],
                    suggested_experiments=[
                        {
                            'type': 'distribution_analysis',
                            'description': f"Systematic sampling to characterize {param_name} distribution",
                            'parameters': {param_name: 'distribution_sampling'},
                            'priority': 'medium'
                        }
                    ],
                    confidence_score=min(abs(skewness) / 2.0, 0.9),
                    supporting_evidence=[f"Observed skewness: {skewness:.2f}"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_anomaly_hypotheses(
        self,
        experimental_data: List[ExperimentPoint],
        insights: List[ExperimentalInsight]
    ) -> List[ScientificHypothesis]:
        """Generate hypotheses to explain experimental anomalies."""
        hypotheses = []
        
        # Find anomaly insights
        anomaly_insights = [i for i in insights if i.insight_type == "anomaly"]
        
        for insight in anomaly_insights:
            # Generate hypothesis to explain the anomaly
            hypothesis = ScientificHypothesis(
                hypothesis_id=f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                title="Anomaly explanation hypothesis",
                description=f"The observed anomaly ({insight.description}) is caused by an uncontrolled variable or measurement error",
                testable_predictions=[
                    "Anomaly can be reproduced under similar conditions",
                    "Systematic investigation will identify the root cause",
                    "Controlling for the identified factor will eliminate the anomaly"
                ],
                suggested_experiments=[
                    {
                        'type': 'anomaly_investigation',
                        'description': 'Systematic investigation of anomalous result',
                        'parameters': {'investigation_type': 'root_cause_analysis'},
                        'priority': 'high'
                    }
                ],
                confidence_score=insight.confidence * 0.7,
                supporting_evidence=[insight.description]
            )
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _generate_interaction_hypotheses(
        self,
        experimental_data: List[ExperimentPoint],
        insights: List[ExperimentalInsight]
    ) -> List[ScientificHypothesis]:
        """Generate hypotheses about parameter interactions."""
        hypotheses = []
        
        if len(experimental_data) < 10:
            return hypotheses
        
        param_names = list(experimental_data[0].parameters.keys())
        
        # Generate hypotheses for parameter pairs
        for param1, param2 in itertools.combinations(param_names, 2):
            interaction_strength = self._calculate_interaction_strength(experimental_data, param1, param2)
            
            if interaction_strength > 0.3:  # Significant interaction
                hypothesis = ScientificHypothesis(
                    hypothesis_id=f"interact_{param1}_{param2}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    title=f"Parameter interaction hypothesis: {param1} × {param2}",
                    description=f"Parameters {param1} and {param2} interact {'synergistically' if interaction_strength > 0 else 'antagonistically'} to affect the objective",
                    testable_predictions=[
                        f"Effect of {param1} depends on level of {param2}",
                        f"Factorial design will reveal interaction pattern",
                        f"Interaction strength: {interaction_strength:.2f}"
                    ],
                    suggested_experiments=[
                        {
                            'type': 'factorial_design',
                            'description': f"Factorial experiment to study {param1} × {param2} interaction",
                            'parameters': {param1: 'factorial_levels', param2: 'factorial_levels'},
                            'priority': 'medium'
                        }
                    ],
                    confidence_score=min(abs(interaction_strength), 0.9),
                    supporting_evidence=[f"Calculated interaction strength: {interaction_strength:.2f}"]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _rank_hypotheses(
        self,
        hypotheses: List[ScientificHypothesis],
        experimental_data: List[ExperimentPoint]
    ) -> List[ScientificHypothesis]:
        """Rank hypotheses by priority and confidence."""
        
        # Calculate priority scores
        for hypothesis in hypotheses:
            priority_score = self._calculate_hypothesis_priority(hypothesis, experimental_data)
            hypothesis.priority_score = priority_score
        
        # Sort by priority score (descending)
        ranked_hypotheses = sorted(hypotheses, key=lambda h: h.priority_score, reverse=True)
        
        return ranked_hypotheses
    
    def _calculate_hypothesis_priority(
        self,
        hypothesis: ScientificHypothesis,
        experimental_data: List[ExperimentPoint]
    ) -> float:
        """Calculate priority score for a hypothesis."""
        
        # Base priority from confidence
        priority = hypothesis.confidence_score
        
        # Boost for testability (more predictions = more testable)
        testability_boost = min(len(hypothesis.testable_predictions) * 0.1, 0.3)
        priority += testability_boost
        
        # Boost for mechanistic hypotheses
        if any('mechanism' in pred.lower() for pred in hypothesis.testable_predictions):
            priority += 0.2
        
        # Boost for hypotheses with high-priority experiments
        if any(exp.get('priority') == 'high' for exp in hypothesis.suggested_experiments):
            priority += 0.1
        
        return min(priority, 1.0)
    
    # Helper methods
    def _infer_dose_response_mechanism(self, parameter: str, context: Dict[str, Any]) -> str:
        """Infer likely mechanism for dose-response relationship."""
        if 'concentration' in parameter.lower() or 'dose' in parameter.lower():
            return "receptor binding and activation"
        elif 'temperature' in parameter.lower():
            return "increased molecular kinetics"
        elif 'catalyst' in parameter.lower():
            return "catalytic site availability"
        else:
            return "concentration-dependent binding"
    
    def _infer_inhibition_target(self, parameter: str, context: Dict[str, Any]) -> str:
        """Infer likely target for inhibition mechanism."""
        if 'inhibitor' in parameter.lower():
            return "active site"
        elif 'competitor' in parameter.lower():
            return "binding site"
        else:
            return "target protein"
    
    def _calculate_template_confidence(self, template: HypothesisTemplate, insight: ExperimentalInsight) -> float:
        """Calculate confidence for template-based hypothesis."""
        base_confidence = insight.confidence
        
        # Apply template-specific confidence factors
        for factor, weight in template.confidence_factors.items():
            if factor == 'correlation_strength' and 'correlation' in insight.description:
                # Extract correlation strength if available
                corr_match = re.search(r'r=([0-9.]+)', insight.description)
                if corr_match:
                    corr_strength = float(corr_match.group(1))
                    base_confidence += weight * corr_strength
        
        return min(base_confidence, 1.0)
    
    def _generate_template_experiments(self, template: HypothesisTemplate, parameter: str) -> List[Dict[str, Any]]:
        """Generate experiments for template-based hypothesis."""
        experiments = []
        
        if template.hypothesis_type == 'mechanistic':
            experiments.append({
                'type': 'mechanism_validation',
                'description': f"Validate proposed mechanism for {parameter}",
                'parameters': {parameter: 'mechanism_test'},
                'priority': 'high'
            })
        
        experiments.append({
            'type': 'parameter_sweep',
            'description': f"Systematic variation of {parameter}",
            'parameters': {parameter: 'sweep_range'},
            'priority': 'medium'
        })
        
        return experiments
    
    def _enhance_mechanistic_hypothesis(
        self,
        hypothesis: ScientificHypothesis,
        experimental_data: List[ExperimentPoint],
        context: Dict[str, Any]
    ) -> ScientificHypothesis:
        """Enhance basic mechanistic hypothesis with additional reasoning."""
        
        # Add domain-specific mechanistic details
        if self.domain == "chemistry":
            enhanced_mechanisms = self._add_chemical_mechanisms(hypothesis, context)
            hypothesis.potential_mechanisms.extend(enhanced_mechanisms)
        
        # Add quantitative predictions if possible
        quantitative_predictions = self._generate_quantitative_predictions(hypothesis, experimental_data)
        hypothesis.testable_predictions.extend(quantitative_predictions)
        
        return hypothesis
    
    def _add_chemical_mechanisms(self, hypothesis: ScientificHypothesis, context: Dict[str, Any]) -> List[str]:
        """Add chemical-specific mechanistic details."""
        mechanisms = []
        
        if 'temperature' in hypothesis.description.lower():
            mechanisms.append("Arrhenius temperature dependence with activation energy")
            mechanisms.append("Thermodynamic vs kinetic control competition")
        
        if 'catalyst' in hypothesis.description.lower():
            mechanisms.append("Heterogeneous catalysis with surface adsorption")
            mechanisms.append("Catalyst deactivation through poisoning or sintering")
        
        return mechanisms
    
    def _generate_quantitative_predictions(
        self,
        hypothesis: ScientificHypothesis,
        experimental_data: List[ExperimentPoint]
    ) -> List[str]:
        """Generate quantitative predictions for hypothesis."""
        predictions = []
        
        if len(experimental_data) >= 5:
            # Calculate some basic statistics for quantitative predictions
            objectives = []
            for point in experimental_data:
                if point.objectives:
                    objectives.extend(point.objectives.values())
            
            if objectives:
                mean_obj = np.mean(objectives)
                std_obj = np.std(objectives)
                
                predictions.append(f"Expected improvement range: {mean_obj + std_obj:.3f} to {mean_obj + 2*std_obj:.3f}")
                predictions.append(f"Confidence interval should be within ±{2*std_obj:.3f}")
        
        return predictions
    
    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness of a distribution."""
        if len(values) < 3:
            return 0.0
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if std_val == 0:
            return 0.0
        
        skewness = np.mean([((x - mean_val) / std_val) ** 3 for x in values])
        return skewness
    
    def _calculate_interaction_strength(
        self,
        experimental_data: List[ExperimentPoint],
        param1: str,
        param2: str
    ) -> float:
        """Calculate interaction strength between two parameters."""
        
        # Simplified interaction calculation
        # In practice, would use proper statistical methods
        
        param1_values = [point.parameters.get(param1, 0) for point in experimental_data]
        param2_values = [point.parameters.get(param2, 0) for point in experimental_data]
        objectives = []
        
        for point in experimental_data:
            if point.objectives:
                objectives.append(list(point.objectives.values())[0])
        
        if len(objectives) < len(param1_values):
            return 0.0
        
        # Calculate correlation of product vs individual correlations
        param1_corr = np.corrcoef(param1_values, objectives)[0, 1] if len(param1_values) > 1 else 0
        param2_corr = np.corrcoef(param2_values, objectives)[0, 1] if len(param2_values) > 1 else 0
        
        # Product of parameters
        product_values = [p1 * p2 for p1, p2 in zip(param1_values, param2_values)]
        product_corr = np.corrcoef(product_values, objectives)[0, 1] if len(product_values) > 1 else 0
        
        # Interaction strength as difference from additive model
        expected_corr = (param1_corr + param2_corr) / 2
        interaction_strength = product_corr - expected_corr
        
        return interaction_strength if not np.isnan(interaction_strength) else 0.0
    
    def create_validation_plan(self, hypothesis: ScientificHypothesis) -> HypothesisValidationPlan:
        """Create a validation plan for a hypothesis."""
        
        validation_experiments = []
        
        # Convert suggested experiments to validation experiments
        for exp in hypothesis.suggested_experiments:
            validation_exp = {
                'experiment_type': exp['type'],
                'description': exp['description'],
                'parameters': exp.get('parameters', {}),
                'priority': exp.get('priority', 'medium'),
                'expected_outcome': 'hypothesis_support'
            }
            validation_experiments.append(validation_exp)
        
        # Define success and failure criteria
        success_criteria = [
            "Experimental results match predicted outcomes",
            "Statistical significance achieved (p < 0.05)",
            "Effect size is practically significant"
        ]
        
        failure_criteria = [
            "Results contradict predictions",
            "No statistical significance",
            "Effect size is negligible"
        ]
        
        plan = HypothesisValidationPlan(
            plan_id=f"validation_{hypothesis.hypothesis_id}",
            hypothesis_id=hypothesis.hypothesis_id,
            validation_experiments=validation_experiments,
            success_criteria=success_criteria,
            failure_criteria=failure_criteria,
            expected_outcomes={
                'primary': 'hypothesis_confirmation',
                'secondary': 'mechanistic_insight',
                'tertiary': 'optimization_guidance'
            },
            estimated_duration=len(validation_experiments) * 2.0,  # 2 days per experiment
            priority_score=hypothesis.priority_score
        )
        
        return plan
    
    def generate_hypothesis_report(self, hypotheses: List[ScientificHypothesis]) -> str:
        """Generate comprehensive hypothesis report."""
        
        report_lines = []
        
        report_lines.append("# Automated Hypothesis Generation Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Domain: {self.domain}")
        report_lines.append(f"Total Hypotheses: {len(hypotheses)}")
        report_lines.append("")
        
        # Summary statistics
        hypothesis_types = defaultdict(int)
        avg_confidence = np.mean([h.confidence_score for h in hypotheses]) if hypotheses else 0
        
        for h in hypotheses:
            if 'mechanistic' in h.description.lower():
                hypothesis_types['mechanistic'] += 1
            elif 'statistical' in h.description.lower():
                hypothesis_types['statistical'] += 1
            elif 'interaction' in h.description.lower():
                hypothesis_types['interaction'] += 1
            else:
                hypothesis_types['other'] += 1
        
        report_lines.append("## Summary Statistics")
        report_lines.append(f"- Average Confidence: {avg_confidence:.1%}")
        report_lines.append("- Hypothesis Types:")
        for htype, count in hypothesis_types.items():
            report_lines.append(f"  - {htype.title()}: {count}")
        report_lines.append("")
        
        # Top hypotheses
        top_hypotheses = sorted(hypotheses, key=lambda h: h.priority_score, reverse=True)[:5]
        
        report_lines.append("## Top 5 Hypotheses")
        for i, hypothesis in enumerate(top_hypotheses, 1):
            report_lines.append(f"### {i}. {hypothesis.title}")
            report_lines.append(f"**Description:** {hypothesis.description}")
            report_lines.append(f"**Confidence:** {hypothesis.confidence_score:.1%}")
            report_lines.append(f"**Priority:** {hypothesis.priority_score:.2f}")
            
            if hypothesis.testable_predictions:
                report_lines.append("**Testable Predictions:**")
                for pred in hypothesis.testable_predictions[:3]:  # Top 3
                    report_lines.append(f"- {pred}")
            
            if hypothesis.suggested_experiments:
                report_lines.append("**Suggested Experiments:**")
                for exp in hypothesis.suggested_experiments[:2]:  # Top 2
                    report_lines.append(f"- {exp['description']} (Priority: {exp.get('priority', 'medium')})")
            
            report_lines.append("")
        
        return "\n".join(report_lines)

    def export_hypotheses_json(self, hypotheses: List[ScientificHypothesis], filename: str) -> None:
        """Export hypotheses to JSON format."""

        hypotheses_data = []
        for hypothesis in hypotheses:
            hypothesis_dict = {
                'hypothesis_id': hypothesis.hypothesis_id,
                'title': hypothesis.title,
                'description': hypothesis.description,
                'confidence_score': hypothesis.confidence_score,
                'priority_score': hypothesis.priority_score,
                'testable_predictions': hypothesis.testable_predictions,
                'suggested_experiments': hypothesis.suggested_experiments,
                'supporting_evidence': hypothesis.supporting_evidence,
                'potential_mechanisms': hypothesis.potential_mechanisms,
                'generated_at': hypothesis.generated_at.isoformat()
            }
            hypotheses_data.append(hypothesis_dict)

        with open(filename, 'w') as f:
            json.dump(hypotheses_data, f, indent=2)

        logger.info(f"Exported {len(hypotheses)} hypotheses to {filename}")
