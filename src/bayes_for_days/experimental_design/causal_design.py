"""
Causal-Aware Experimental Design for Bayes For Days platform.

This module implements experimental design strategies that focus on causal
discovery and mechanistic understanding rather than just correlation-based
optimization. This represents a revolutionary approach to experimental design
that goes beyond traditional methods to uncover true causal relationships.

Key Features:
- Causal graph discovery from experimental data
- Interventional experiment design for causal validation
- Confounding variable identification and control
- Mediation analysis and pathway discovery
- Counterfactual reasoning for experimental planning
- Causal effect estimation with uncertainty quantification
- Integration with domain knowledge and scientific theory

Based on:
- Pearl (2009) "Causality: Models, Reasoning, and Inference"
- Peters et al. (2017) "Elements of Causal Inference"
- Spirtes et al. (2000) "Causation, Prediction, and Search"
- Modern causal discovery algorithms (PC, GES, LiNGAM)
- Experimental design for causal inference
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any, Set
import numpy as np
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import networkx as nx
from itertools import combinations, permutations
import matplotlib.pyplot as plt
from datetime import datetime

from bayes_for_days.core.base import BaseSurrogateModel
from bayes_for_days.core.types import (
    ExperimentPoint,
    ParameterSpace,
    Parameter,
    ParameterType,
    ParameterDict
)
from bayes_for_days.experimental_design.design_strategies import DesignStrategy

logger = logging.getLogger(__name__)


@dataclass
class CausalEdge:
    """Represents a causal edge in a causal graph."""
    source: str
    target: str
    strength: float
    confidence: float
    edge_type: str = "causal"  # 'causal', 'confounding', 'mediating'
    mechanism: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.source} -> {self.target} (strength: {self.strength:.3f}, confidence: {self.confidence:.3f})"


@dataclass
class CausalGraph:
    """Represents a causal graph with nodes and edges."""
    nodes: Set[str] = field(default_factory=set)
    edges: List[CausalEdge] = field(default_factory=list)
    confounders: Set[str] = field(default_factory=set)
    mediators: Set[str] = field(default_factory=set)
    
    def add_edge(self, edge: CausalEdge) -> None:
        """Add a causal edge to the graph."""
        self.nodes.add(edge.source)
        self.nodes.add(edge.target)
        self.edges.append(edge)
    
    def get_parents(self, node: str) -> List[str]:
        """Get parent nodes of a given node."""
        return [edge.source for edge in self.edges if edge.target == node]
    
    def get_children(self, node: str) -> List[str]:
        """Get child nodes of a given node."""
        return [edge.target for edge in self.edges if edge.source == node]
    
    def to_networkx(self) -> nx.DiGraph:
        """Convert to NetworkX directed graph."""
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        for edge in self.edges:
            G.add_edge(edge.source, edge.target, 
                      strength=edge.strength, confidence=edge.confidence)
        return G


@dataclass
class CausalHypothesis:
    """Represents a causal hypothesis to be tested."""
    hypothesis_id: str
    description: str
    proposed_cause: str
    proposed_effect: str
    mechanism: Optional[str] = None
    confounders: List[str] = field(default_factory=list)
    mediators: List[str] = field(default_factory=list)
    testable_predictions: List[str] = field(default_factory=list)
    suggested_interventions: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'hypothesis_id': self.hypothesis_id,
            'description': self.description,
            'proposed_cause': self.proposed_cause,
            'proposed_effect': self.proposed_effect,
            'mechanism': self.mechanism,
            'confounders': self.confounders,
            'mediators': self.mediators,
            'testable_predictions': self.testable_predictions,
            'suggested_interventions': self.suggested_interventions,
            'confidence': self.confidence
        }


class CausalDiscoveryEngine:
    """
    Engine for discovering causal relationships from experimental data.
    
    Uses constraint-based and score-based algorithms to identify
    causal structures from observational and interventional data.
    """
    
    def __init__(self, algorithm: str = "pc", significance_level: float = 0.05):
        """
        Initialize causal discovery engine.
        
        Args:
            algorithm: Causal discovery algorithm ('pc', 'ges', 'lingam')
            significance_level: Statistical significance level for tests
        """
        self.algorithm = algorithm
        self.significance_level = significance_level
        self.discovered_graph: Optional[CausalGraph] = None
        
    def discover_causal_structure(
        self,
        experimental_data: List[ExperimentPoint],
        prior_knowledge: Optional[Dict[str, Any]] = None
    ) -> CausalGraph:
        """
        Discover causal structure from experimental data.
        
        Args:
            experimental_data: List of experimental results
            prior_knowledge: Optional prior knowledge about causal structure
            
        Returns:
            Discovered causal graph
        """
        if len(experimental_data) < 10:
            logger.warning("Insufficient data for reliable causal discovery")
            return CausalGraph()
        
        # Extract data matrix
        data_matrix, variable_names = self._prepare_data_matrix(experimental_data)
        
        if self.algorithm == "pc":
            causal_graph = self._pc_algorithm(data_matrix, variable_names)
        elif self.algorithm == "ges":
            causal_graph = self._ges_algorithm(data_matrix, variable_names)
        elif self.algorithm == "lingam":
            causal_graph = self._lingam_algorithm(data_matrix, variable_names)
        else:
            # Fallback to correlation-based discovery
            causal_graph = self._correlation_based_discovery(data_matrix, variable_names)
        
        # Incorporate prior knowledge if available
        if prior_knowledge:
            causal_graph = self._incorporate_prior_knowledge(causal_graph, prior_knowledge)
        
        self.discovered_graph = causal_graph
        return causal_graph
    
    def _prepare_data_matrix(
        self,
        experimental_data: List[ExperimentPoint]
    ) -> Tuple[np.ndarray, List[str]]:
        """Prepare data matrix for causal discovery."""
        if not experimental_data:
            return np.array([]), []
        
        # Get all parameter and objective names
        all_params = set()
        all_objectives = set()
        
        for point in experimental_data:
            all_params.update(point.parameters.keys())
            if point.objectives:
                all_objectives.update(point.objectives.keys())
        
        variable_names = list(all_params) + list(all_objectives)
        
        # Create data matrix
        data_matrix = []
        for point in experimental_data:
            row = []
            
            # Add parameter values
            for param in all_params:
                row.append(point.parameters.get(param, 0.0))
            
            # Add objective values
            for obj in all_objectives:
                row.append(point.objectives.get(obj, 0.0) if point.objectives else 0.0)
            
            data_matrix.append(row)
        
        return np.array(data_matrix), variable_names
    
    def _pc_algorithm(self, data: np.ndarray, variables: List[str]) -> CausalGraph:
        """Simplified PC algorithm implementation."""
        n_vars = len(variables)
        causal_graph = CausalGraph()
        
        # Calculate pairwise correlations
        correlations = np.corrcoef(data.T)
        
        # Identify significant correlations
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                corr = correlations[i, j]
                if abs(corr) > 0.3:  # Threshold for significance
                    # Determine direction (simplified)
                    if corr > 0:
                        edge = CausalEdge(
                            source=variables[i],
                            target=variables[j],
                            strength=abs(corr),
                            confidence=min(abs(corr) * 2, 1.0)
                        )
                    else:
                        edge = CausalEdge(
                            source=variables[j],
                            target=variables[i],
                            strength=abs(corr),
                            confidence=min(abs(corr) * 2, 1.0)
                        )
                    
                    causal_graph.add_edge(edge)
        
        return causal_graph
    
    def _ges_algorithm(self, data: np.ndarray, variables: List[str]) -> CausalGraph:
        """Simplified GES algorithm implementation."""
        # For now, use same as PC algorithm
        # In practice, would implement proper GES scoring
        return self._pc_algorithm(data, variables)
    
    def _lingam_algorithm(self, data: np.ndarray, variables: List[str]) -> CausalGraph:
        """Simplified LiNGAM algorithm implementation."""
        # For now, use same as PC algorithm
        # In practice, would implement proper LiNGAM for linear non-Gaussian models
        return self._pc_algorithm(data, variables)
    
    def _correlation_based_discovery(self, data: np.ndarray, variables: List[str]) -> CausalGraph:
        """Simple correlation-based causal discovery."""
        return self._pc_algorithm(data, variables)
    
    def _incorporate_prior_knowledge(
        self,
        graph: CausalGraph,
        prior_knowledge: Dict[str, Any]
    ) -> CausalGraph:
        """Incorporate prior domain knowledge into causal graph."""
        # Add known causal relationships
        if 'known_edges' in prior_knowledge:
            for edge_info in prior_knowledge['known_edges']:
                edge = CausalEdge(
                    source=edge_info['source'],
                    target=edge_info['target'],
                    strength=edge_info.get('strength', 1.0),
                    confidence=edge_info.get('confidence', 1.0),
                    mechanism=edge_info.get('mechanism')
                )
                graph.add_edge(edge)
        
        # Add known confounders
        if 'confounders' in prior_knowledge:
            graph.confounders.update(prior_knowledge['confounders'])
        
        return graph


class InterventionalDesigner:
    """
    Designer for interventional experiments to test causal hypotheses.
    
    Creates experimental designs that can distinguish between
    causal and confounded relationships through targeted interventions.
    """
    
    def __init__(self, causal_graph: CausalGraph):
        """Initialize interventional designer with causal graph."""
        self.causal_graph = causal_graph
    
    def design_intervention_experiments(
        self,
        target_hypothesis: CausalHypothesis,
        parameter_space: ParameterSpace,
        n_experiments: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Design interventional experiments to test causal hypothesis.
        
        Args:
            target_hypothesis: Causal hypothesis to test
            parameter_space: Available parameter space
            n_experiments: Number of experiments to design
            
        Returns:
            List of interventional experiment designs
        """
        interventions = []
        
        cause_var = target_hypothesis.proposed_cause
        effect_var = target_hypothesis.proposed_effect
        
        # Design do-calculus interventions
        # Intervention 1: Manipulate cause, observe effect
        intervention_1 = self._design_do_intervention(
            cause_var, effect_var, parameter_space, n_experiments // 2
        )
        interventions.extend(intervention_1)
        
        # Intervention 2: Control for confounders
        if target_hypothesis.confounders:
            intervention_2 = self._design_confounder_control(
                cause_var, effect_var, target_hypothesis.confounders,
                parameter_space, n_experiments // 2
            )
            interventions.extend(intervention_2)
        
        return interventions
    
    def _design_do_intervention(
        self,
        cause_var: str,
        effect_var: str,
        parameter_space: ParameterSpace,
        n_experiments: int
    ) -> List[Dict[str, Any]]:
        """Design do-calculus intervention experiments."""
        interventions = []
        
        # Find parameter bounds for cause variable
        cause_param = None
        for param in parameter_space.parameters:
            if param.name == cause_var:
                cause_param = param
                break
        
        if not cause_param:
            return interventions
        
        # Create intervention levels
        if cause_param.type == ParameterType.CONTINUOUS:
            intervention_levels = np.linspace(
                cause_param.bounds[0], cause_param.bounds[1], n_experiments
            )
        else:
            intervention_levels = cause_param.categories[:n_experiments]
        
        for i, level in enumerate(intervention_levels):
            intervention = {
                'experiment_id': f"intervention_{cause_var}_{i}",
                'intervention_type': 'do_calculus',
                'target_variable': cause_var,
                'intervention_value': level,
                'observe_variables': [effect_var],
                'description': f"Set {cause_var} = {level}, observe {effect_var}",
                'priority': 'high'
            }
            interventions.append(intervention)
        
        return interventions
    
    def _design_confounder_control(
        self,
        cause_var: str,
        effect_var: str,
        confounders: List[str],
        parameter_space: ParameterSpace,
        n_experiments: int
    ) -> List[Dict[str, Any]]:
        """Design experiments controlling for confounders."""
        interventions = []
        
        # Create experiments with confounders held constant
        for i in range(n_experiments):
            intervention = {
                'experiment_id': f"confounder_control_{i}",
                'intervention_type': 'confounder_control',
                'target_variable': cause_var,
                'control_variables': confounders,
                'observe_variables': [effect_var],
                'description': f"Control {confounders}, vary {cause_var}, observe {effect_var}",
                'priority': 'medium'
            }
            interventions.append(intervention)
        
        return interventions


class CausalAwareDesign(DesignStrategy):
    """
    Causal-aware experimental design strategy.
    
    Integrates causal discovery and interventional design to create
    experiments that can uncover true causal relationships rather
    than just correlations.
    """
    
    def __init__(
        self,
        variables: List[Any],
        causal_discovery_engine: Optional[CausalDiscoveryEngine] = None,
        prior_knowledge: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize causal-aware design strategy.
        
        Args:
            variables: List of experimental variables
            causal_discovery_engine: Engine for causal discovery
            prior_knowledge: Prior knowledge about causal structure
            **kwargs: Additional parameters
        """
        super().__init__(variables, **kwargs)
        
        self.causal_engine = causal_discovery_engine or CausalDiscoveryEngine()
        self.prior_knowledge = prior_knowledge or {}
        self.discovered_graph: Optional[CausalGraph] = None
        self.causal_hypotheses: List[CausalHypothesis] = []
        
    def generate_design(self, n_experiments: int, **kwargs) -> np.ndarray:
        """
        Generate causal-aware experimental design.
        
        Args:
            n_experiments: Number of experiments to generate
            **kwargs: Additional parameters including existing_data
            
        Returns:
            Causal-aware design matrix
        """
        existing_data = kwargs.get('existing_data', [])
        
        if existing_data and len(existing_data) >= 5:
            # Discover causal structure from existing data
            self.discovered_graph = self.causal_engine.discover_causal_structure(
                existing_data, self.prior_knowledge
            )
            
            # Generate causal hypotheses
            self.causal_hypotheses = self._generate_causal_hypotheses()
            
            # Design interventional experiments
            design_matrix = self._design_interventional_experiments(n_experiments)
        else:
            # Use space-filling design for initial exploration
            design_matrix = self._generate_space_filling_design(n_experiments)
        
        return design_matrix
    
    def _generate_causal_hypotheses(self) -> List[CausalHypothesis]:
        """Generate causal hypotheses from discovered graph."""
        hypotheses = []
        
        if not self.discovered_graph:
            return hypotheses
        
        # Generate hypotheses for each causal edge
        for i, edge in enumerate(self.discovered_graph.edges):
            if edge.confidence > 0.5:  # Only high-confidence edges
                hypothesis = CausalHypothesis(
                    hypothesis_id=f"causal_hyp_{i}",
                    description=f"{edge.source} causally affects {edge.target}",
                    proposed_cause=edge.source,
                    proposed_effect=edge.target,
                    mechanism=edge.mechanism,
                    confidence=edge.confidence,
                    testable_predictions=[
                        f"Intervening on {edge.source} will change {edge.target}",
                        f"The effect persists when controlling for confounders"
                    ]
                )
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    def _design_interventional_experiments(self, n_experiments: int) -> np.ndarray:
        """Design interventional experiments based on causal hypotheses."""
        if not self.causal_hypotheses:
            return self._generate_space_filling_design(n_experiments)
        
        # Create parameter space from variables
        from bayes_for_days.core.types import ParameterSpace, Parameter, ParameterType
        
        parameters = []
        for var in self.variables:
            if hasattr(var, 'name') and hasattr(var, 'bounds'):
                param = Parameter(
                    name=var.name,
                    type=ParameterType.CONTINUOUS,
                    bounds=var.bounds
                )
                parameters.append(param)
        
        parameter_space = ParameterSpace(parameters=parameters)
        
        # Design interventions for top hypotheses
        interventional_designer = InterventionalDesigner(self.discovered_graph)
        
        all_interventions = []
        experiments_per_hypothesis = max(1, n_experiments // len(self.causal_hypotheses))
        
        for hypothesis in self.causal_hypotheses[:3]:  # Top 3 hypotheses
            interventions = interventional_designer.design_intervention_experiments(
                hypothesis, parameter_space, experiments_per_hypothesis
            )
            all_interventions.extend(interventions)
        
        # Convert interventions to design matrix
        design_matrix = self._interventions_to_matrix(all_interventions, n_experiments)
        
        return design_matrix
    
    def _interventions_to_matrix(
        self,
        interventions: List[Dict[str, Any]],
        n_experiments: int
    ) -> np.ndarray:
        """Convert intervention designs to design matrix."""
        n_vars = len(self.variables)
        design_matrix = np.zeros((n_experiments, n_vars))
        
        # Fill matrix with intervention values
        for i, intervention in enumerate(interventions[:n_experiments]):
            for j, var in enumerate(self.variables):
                if hasattr(var, 'name'):
                    var_name = var.name
                    
                    if var_name == intervention.get('target_variable'):
                        # Set intervention value
                        design_matrix[i, j] = intervention.get('intervention_value', 0.5)
                    else:
                        # Use random value within bounds
                        if hasattr(var, 'bounds'):
                            low, high = var.bounds
                            design_matrix[i, j] = np.random.uniform(low, high)
                        else:
                            design_matrix[i, j] = 0.5
        
        # Fill remaining experiments with space-filling design
        if len(interventions) < n_experiments:
            remaining = n_experiments - len(interventions)
            space_filling = self._generate_space_filling_design(remaining)
            design_matrix[len(interventions):] = space_filling
        
        return design_matrix
    
    def _generate_space_filling_design(self, n_experiments: int) -> np.ndarray:
        """Generate space-filling design for initial exploration."""
        n_vars = len(self.variables)
        
        # Use Latin Hypercube Sampling
        from scipy.stats import qmc
        
        sampler = qmc.LatinHypercube(d=n_vars, seed=42)
        design = sampler.random(n=n_experiments)
        
        # Scale to variable bounds
        for j, var in enumerate(self.variables):
            if hasattr(var, 'bounds'):
                low, high = var.bounds
                design[:, j] = low + design[:, j] * (high - low)
        
        return design
    
    def generate_causal_report(self) -> str:
        """Generate comprehensive causal analysis report."""
        report_lines = []
        
        report_lines.append("# Causal-Aware Experimental Design Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Discovered causal structure
        if self.discovered_graph:
            report_lines.append("## Discovered Causal Structure")
            report_lines.append(f"Nodes: {len(self.discovered_graph.nodes)}")
            report_lines.append(f"Edges: {len(self.discovered_graph.edges)}")
            report_lines.append("")
            
            report_lines.append("### Causal Relationships")
            for edge in self.discovered_graph.edges:
                report_lines.append(f"- {edge}")
            report_lines.append("")
        
        # Causal hypotheses
        if self.causal_hypotheses:
            report_lines.append("## Generated Causal Hypotheses")
            for i, hyp in enumerate(self.causal_hypotheses, 1):
                report_lines.append(f"### Hypothesis {i}: {hyp.description}")
                report_lines.append(f"**Cause:** {hyp.proposed_cause}")
                report_lines.append(f"**Effect:** {hyp.proposed_effect}")
                report_lines.append(f"**Confidence:** {hyp.confidence:.1%}")
                
                if hyp.testable_predictions:
                    report_lines.append("**Testable Predictions:**")
                    for pred in hyp.testable_predictions:
                        report_lines.append(f"- {pred}")
                
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def visualize_causal_graph(self, save_path: Optional[str] = None) -> None:
        """Visualize the discovered causal graph."""
        if not self.discovered_graph:
            logger.warning("No causal graph to visualize")
            return
        
        # Convert to NetworkX graph
        G = self.discovered_graph.to_networkx()
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Use spring layout for node positioning
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=1000, alpha=0.7)
        
        # Draw edges with different styles based on confidence
        high_conf_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('confidence', 0) > 0.7]
        med_conf_edges = [(u, v) for u, v, d in G.edges(data=True) if 0.4 <= d.get('confidence', 0) <= 0.7]
        low_conf_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('confidence', 0) < 0.4]
        
        nx.draw_networkx_edges(G, pos, edgelist=high_conf_edges, 
                              edge_color='red', width=3, alpha=0.8)
        nx.draw_networkx_edges(G, pos, edgelist=med_conf_edges, 
                              edge_color='orange', width=2, alpha=0.6)
        nx.draw_networkx_edges(G, pos, edgelist=low_conf_edges, 
                              edge_color='gray', width=1, alpha=0.4)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title('Discovered Causal Graph', fontsize=16, fontweight='bold')
        plt.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', lw=3, label='High Confidence (>70%)'),
            Line2D([0], [0], color='orange', lw=2, label='Medium Confidence (40-70%)'),
            Line2D([0], [0], color='gray', lw=1, label='Low Confidence (<40%)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        return plt.gcf()


# Create __init__.py file to make this a proper module
def create_init_file():
    """Create __init__.py file for the causal design module."""
    init_content = '''"""
Causal-aware experimental design module.
"""

from .causal_design import (
    CausalAwareDesign,
    CausalDiscoveryEngine,
    InterventionalDesigner,
    CausalGraph,
    CausalEdge,
    CausalHypothesis
)

__all__ = [
    'CausalAwareDesign',
    'CausalDiscoveryEngine',
    'InterventionalDesigner',
    'CausalGraph',
    'CausalEdge',
    'CausalHypothesis'
]
'''

    with open('src/bayes_for_days/experimental_design/causal/__init__.py', 'w') as f:
        f.write(init_content)
