"""
File: model_architecture_analyzer.py
Author: Generated with Claude
Created: 02/27/2025
Description: Analysis tool for the Mixture-of-Experts language model architecture.
             Provides detailed visualization of model architecture, component dependencies,
             and architectural documentation generation.
"""

import os
import sys
import argparse
import logging
import json
import inspect
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import importlib
import re

import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
from torchinfo import summary

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("model_architecture_analysis.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ArchitectureAnalyzerConfig:
    """Configuration for architecture analysis"""

    output_dir: str = "architecture_analysis"
    model_modules_to_analyze: List[str] = field(
        default_factory=lambda: [
            "expertlm.models.hierarchicalmixtureofexperts",
            "expertlm.models.moelanguage",
            "expertlm.models.specializedexpertnetwork",
            "expertlm.models.expertlayer",
            "expertlm.models.hybridpositionalencoding",
            "expertlm.models.domainspecificattention",
            "expertlm.models.positionalencoding",
        ]
    )
    create_module_diagrams: bool = True
    create_dependency_graphs: bool = True
    create_class_docs: bool = True
    create_architecture_report: bool = True
    model_configs_to_analyze: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "vocab_size": 32000,
                "d_model": 256,
                "n_layers": 2,
                "num_experts": 8,
                "ffn_hidden_dim": 1024,
                "num_heads": 8,
                "max_seq_len": 512,
                "k_experts": 2,
            }
        ]
    )
    dummy_batch_size: int = 2
    dummy_seq_length: int = 64


class ModelArchitectureAnalyzer:
    """Analyzes model architecture and generates documentation"""

    def __init__(self, config: ArchitectureAnalyzerConfig):
        self.config = config

        # Create output directory if it doesn't exist
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

        # Create subdirectories
        os.makedirs(os.path.join(config.output_dir, "diagrams"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "class_docs"), exist_ok=True)
        os.makedirs(os.path.join(config.output_dir, "model_summaries"), exist_ok=True)

        # Track modules and classes we've analyzed
        self.analyzed_modules = {}
        self.analyzed_classes = {}
        self.dependency_graph = nx.DiGraph()

        logger.info(
            f"Initialized architecture analyzer with output to {config.output_dir}"
        )

    def _import_module(self, module_name: str) -> Optional[Any]:
        """Safely import a module"""
        try:
            return importlib.import_module(module_name)
        except ImportError as e:
            logger.error(f"Error importing module {module_name}: {str(e)}")
            return None

    def _get_module_classes(self, module_obj: Any) -> Dict[str, Any]:
        """Extract PyTorch Module classes from a module"""
        classes = {}

        # Inspect all module attributes
        for name, obj in inspect.getmembers(module_obj):
            # Check if it's a class and a subclass of nn.Module
            if (
                inspect.isclass(obj)
                and issubclass(obj, nn.Module)
                and obj.__module__ == module_obj.__name__
            ):
                classes[name] = obj

        return classes

    def _analyze_class_dependencies(self, cls: Any) -> List[str]:
        """Analyze class dependencies by inspecting the constructor"""
        dependencies = []

        # Get the source code of the __init__ method
        try:
            init_method = cls.__init__
            source_code = inspect.getsource(init_method)

            # Look for patterns like: self.xxx = ModuleName(...)
            # or nn.ModuleList([ModuleName(...) for ...])
            module_pattern = r"(?:self\.[a-zA-Z0-9_]+ *= *|nn\.ModuleList\(\[)[a-zA-Z0-9_\.]*([A-Z][a-zA-Z0-9_]+)\("
            modules = re.findall(module_pattern, source_code)

            # Add unique dependencies
            for module_name in modules:
                if module_name not in dependencies and module_name != cls.__name__:
                    dependencies.append(module_name)

        except (IOError, TypeError) as e:
            logger.warning(
                f"Could not analyze dependencies for {cls.__name__}: {str(e)}"
            )

        return dependencies

    def _get_class_docstring(self, cls: Any) -> str:
        """Extract the docstring from a class"""
        doc = inspect.getdoc(cls)
        if doc:
            return doc
        return "No documentation available."

    def _get_method_signature(self, method: Any) -> str:
        """Get method signature as a string"""
        try:
            return str(inspect.signature(method))
        except ValueError:
            return "(unknown signature)"

    def _create_class_documentation(self, cls_name: str, cls: Any) -> str:
        """Create detailed documentation for a class"""
        doc = [f"# {cls_name}\n"]

        # Add class docstring
        doc.append("## Description\n")
        doc.append(self._get_class_docstring(cls) + "\n")

        # List all methods
        doc.append("## Methods\n")

        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            # Skip private methods
            if name.startswith("_") and name != "__init__":
                continue

            # Get method signature and docstring
            signature = self._get_method_signature(method)
            method_doc = inspect.getdoc(method)

            doc.append(f"### `{name}{signature}`\n")

            if method_doc:
                doc.append(method_doc + "\n")
            else:
                doc.append("No documentation available.\n")

        # List all attributes and fields
        doc.append("## Attributes\n")

        # Attempt to analyze source code to find attributes
        try:
            source = inspect.getsource(cls)
            # Look for self.xxx patterns in __init__
            attributes = re.findall(r"self\.([a-zA-Z0-9_]+) *=", source)

            if attributes:
                for attr in sorted(set(attributes)):
                    # Skip private attributes
                    if attr.startswith("_"):
                        continue
                    doc.append(f"- `{attr}`\n")
            else:
                doc.append("No attributes found in source code.\n")

        except (IOError, TypeError):
            doc.append("Could not analyze source code for attributes.\n")

        return "\n".join(doc)

    def analyze_module(self, module_name: str):
        """Analyze a module and its classes"""
        logger.info(f"Analyzing module: {module_name}")

        # Import the module
        module = self._import_module(module_name)
        if not module:
            return

        # Get all module classes that are nn.Modules
        classes = self._get_module_classes(module)
        logger.info(f"Found {len(classes)} PyTorch Module classes in {module_name}")

        # Store module and classes
        self.analyzed_modules[module_name] = module

        # Analyze each class
        for cls_name, cls in classes.items():
            logger.info(f"Analyzing class: {cls_name}")

            # Find dependencies
            dependencies = self._analyze_class_dependencies(cls)

            # Store class info
            self.analyzed_classes[cls_name] = {
                "class": cls,
                "module": module_name,
                "dependencies": dependencies,
            }

            # Add to dependency graph
            self.dependency_graph.add_node(cls_name)
            for dep in dependencies:
                self.dependency_graph.add_edge(cls_name, dep)

            # Create class documentation if enabled
            if self.config.create_class_docs:
                doc_content = self._create_class_documentation(cls_name, cls)

                doc_file = os.path.join(
                    self.config.output_dir, "class_docs", f"{cls_name}.md"
                )
                with open(doc_file, "w") as f:
                    f.write(doc_content)
                logger.info(f"Created documentation for {cls_name}")

    def create_dependency_diagram(self):
        """Create a diagram of class dependencies"""
        if not self.dependency_graph.nodes:
            logger.warning("No dependencies found to create diagram")
            return

        try:
            plt.figure(figsize=(12, 10))

            # Create a hierarchical layout
            pos = nx.spring_layout(self.dependency_graph, seed=42)

            # Draw the graph
            nx.draw(
                self.dependency_graph,
                pos,
                with_labels=True,
                node_color="lightblue",
                node_size=3000,
                font_size=10,
                font_weight="bold",
                arrows=True,
                arrowsize=20,
                edge_color="gray",
            )

            # Add a title
            plt.title("Model Architecture Dependency Graph", fontsize=15)

            # Save the figure
            output_file = os.path.join(
                self.config.output_dir, "diagrams", "dependency_graph.png"
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Dependency diagram saved to {output_file}")

        except Exception as e:
            logger.error(f"Error creating dependency diagram: {str(e)}")

    def create_module_diagram(self, module_name: str):
        """Create a diagram for a specific module's classes"""
        # Find all classes in this module
        module_classes = [
            name
            for name, info in self.analyzed_classes.items()
            if info["module"] == module_name
        ]

        if not module_classes:
            logger.warning(f"No classes found in {module_name} to create diagram")
            return

        try:
            # Create a subgraph with only these classes and their direct dependencies
            subgraph = nx.DiGraph()

            for cls_name in module_classes:
                subgraph.add_node(cls_name)

                # Add dependencies that we have analyzed
                for dep in self.analyzed_classes.get(cls_name, {}).get(
                    "dependencies", []
                ):
                    if dep in self.analyzed_classes:
                        subgraph.add_edge(cls_name, dep)

            if not subgraph.nodes:
                return

            plt.figure(figsize=(10, 8))

            # Create a hierarchical layout
            pos = nx.spring_layout(subgraph, seed=42)

            # Draw the graph
            nx.draw(
                subgraph,
                pos,
                with_labels=True,
                node_color="lightgreen",
                node_size=2500,
                font_size=10,
                font_weight="bold",
                arrows=True,
                arrowsize=15,
                edge_color="gray",
            )

            # Extract module short name for the title
            short_name = module_name.split(".")[-1]
            plt.title(f"{short_name} Module Classes", fontsize=15)

            # Save the figure
            output_file = os.path.join(
                self.config.output_dir, "diagrams", f"{short_name}_classes.png"
            )
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Module diagram saved to {output_file}")

        except Exception as e:
            logger.error(f"Error creating module diagram: {str(e)}")

    def create_model_summary(self, config_dict: Dict[str, Any]):
        """Create a summary of the full model with a given configuration"""
        try:
            # Import the MoELanguageModel
            from expertlm.models.moelanguage import MoELanguageModel

            # Create a model instance
            model = MoELanguageModel(**config_dict)

            # Generate a summary
            batch_size = self.config.dummy_batch_size
            seq_len = self.config.dummy_seq_length

            # Generate a detailed summary
            model_stats = summary(
                model,
                input_size=(batch_size, seq_len),
                depth=4,
                verbose=1,
                col_names=["input_size", "output_size", "num_params", "trainable"],
                col_width=20,
                row_settings=["var_names"],
            )

            # Convert to string representation
            model_summary_str = str(model_stats)

            # Save the summary
            config_name = f"d{config_dict['d_model']}_l{config_dict['n_layers']}_e{config_dict['num_experts']}"
            summary_file = os.path.join(
                self.config.output_dir,
                "model_summaries",
                f"model_summary_{config_name}.txt",
            )

            with open(summary_file, "w") as f:
                f.write(model_summary_str)

            logger.info(f"Model summary saved to {summary_file}")

            return model_stats

        except Exception as e:
            logger.error(f"Error creating model summary: {str(e)}")
            return None

    def analyze_model_parameters(self, config_dict: Dict[str, Any]):
        """Analyze parameter distribution across model components"""
        try:
            # Import the MoELanguageModel
            from expertlm.models.moelanguage import MoELanguageModel

            # Create a model instance
            model = MoELanguageModel(**config_dict)

            # Analyze parameter distribution by component
            param_counts = {}
            total_params = 0

            # Iterate through named modules
            for name, module in model.named_modules():
                if (
                    isinstance(module, nn.Linear)
                    or isinstance(module, nn.Embedding)
                    or isinstance(module, nn.LayerNorm)
                ):
                    # Count parameters in this module
                    module_params = sum(
                        p.numel() for p in module.parameters() if p.requires_grad
                    )

                    # Add to the appropriate component
                    if "token_embedding" in name:
                        component = "Token Embedding"
                    elif "pos_encoding" in name:
                        component = "Positional Encoding"
                    elif "self_attn" in name or "mha" in name:
                        component = "Self-Attention"
                    elif "moe_ffn" in name or "expert" in name:
                        component = "MoE FFN"
                    elif "router" in name:
                        component = "Router"
                    elif "output_projection" in name:
                        component = "Output Projection"
                    elif "layer_norm" in name:
                        component = "Layer Normalization"
                    else:
                        component = "Other"

                    # Increment counters
                    if component not in param_counts:
                        param_counts[component] = 0
                    param_counts[component] += module_params
                    total_params += module_params

            # Calculate percentages
            param_distribution = {
                component: {
                    "count": count,
                    "percentage": (
                        (count / total_params) * 100 if total_params > 0 else 0
                    ),
                }
                for component, count in param_counts.items()
            }

            # Save the analysis
            config_name = f"d{config_dict['d_model']}_l{config_dict['n_layers']}_e{config_dict['num_experts']}"
            analysis_file = os.path.join(
                self.config.output_dir,
                "model_summaries",
                f"param_analysis_{config_name}.json",
            )

            with open(analysis_file, "w") as f:
                json.dump(
                    {
                        "config": config_dict,
                        "total_parameters": total_params,
                        "parameter_distribution": param_distribution,
                    },
                    f,
                    indent=2,
                )

            logger.info(f"Parameter analysis saved to {analysis_file}")

            # Create visualization
            self._create_parameter_distribution_chart(param_distribution, config_name)

            return param_distribution

        except Exception as e:
            logger.error(f"Error analyzing model parameters: {str(e)}")
            return None

    def _create_parameter_distribution_chart(
        self, param_distribution: Dict[str, Dict[str, Any]], config_name: str
    ):
        """Create a pie chart of parameter distribution"""
        try:
            components = []
            counts = []
            percentages = []

            # Extract data
            for component, data in param_distribution.items():
                components.append(component)
                counts.append(data["count"])
                percentages.append(data["percentage"])

            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(
                percentages,
                labels=components,
                autopct="%1.1f%%",
                startangle=90,
                shadow=True,
                explode=[0.05] * len(components),
            )
            plt.axis(
                "equal"
            )  # Equal aspect ratio ensures that pie is drawn as a circle
            plt.title(f"Parameter Distribution (Total: {sum(counts):,})")

            # Save chart
            chart_file = os.path.join(
                self.config.output_dir,
                "diagrams",
                f"param_distribution_{config_name}.png",
            )
            plt.savefig(chart_file, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Parameter distribution chart saved to {chart_file}")

        except Exception as e:
            logger.error(f"Error creating parameter distribution chart: {str(e)}")

    def create_architecture_report(self):
        """Create a comprehensive architecture report"""
        if not self.config.create_architecture_report:
            return

        try:
            logger.info("Generating architecture report...")

            report = []
            report.append("# Mixture-of-Experts Language Model Architecture Report\n")

            # 1. Overview of the architecture
            report.append("## 1. Architecture Overview\n")
            report.append(
                "The Mixture-of-Experts (MoE) Language Model implements a transformer-based architecture with sparse expert routing. "
            )
            report.append(
                "Each token is processed by a subset of available experts, allowing for specialization and efficient scaling.\n"
            )

            # 2. Key components
            report.append("## 2. Key Components\n")

            # Describe main components with links to their documentation
            for cls_name, info in sorted(self.analyzed_classes.items()):
                report.append(f"### {cls_name}\n")

                # Get class docstring
                cls = info["class"]
                docstring = self._get_class_docstring(cls)

                # Add description
                if docstring:
                    # Extract first paragraph of docstring
                    first_para = docstring.split("\n\n")[0]
                    report.append(f"{first_para}\n")

                # Add link to full documentation
                report.append(
                    f"[View detailed documentation](class_docs/{cls_name}.md)\n"
                )

            # 3. Component relationships
            report.append("## 3. Component Relationships\n")
            report.append(
                "The diagram below shows the dependencies between different components in the architecture.\n"
            )
            report.append("![Dependency Graph](diagrams/dependency_graph.png)\n")

            # 4. Parameter analysis
            report.append("## 4. Parameter Analysis\n")
            report.append(
                "The following section shows parameter distribution for different model configurations.\n"
            )

            # Add parameter analysis for each config
            for config_dict in self.config.model_configs_to_analyze:
                config_name = f"d{config_dict['d_model']}_l{config_dict['n_layers']}_e{config_dict['num_experts']}"

                report.append(f"### Configuration: {config_name}\n")
                report.append(f"- Model dimension: {config_dict['d_model']}")
                report.append(f"- Number of layers: {config_dict['n_layers']}")
                report.append(f"- Number of experts: {config_dict['num_experts']}")
                report.append(f"- Number of heads: {config_dict['num_heads']}")
                report.append(f"- Experts per token (k): {config_dict['k_experts']}\n")

                report.append(
                    f"![Parameter Distribution](diagrams/param_distribution_{config_name}.png)\n"
                )

            # 5. Implementation notes
            report.append("## 5. Implementation Notes\n")

            report.append("### Expert Routing\n")
            report.append(
                "The router determines which experts should process each token based on the token's features. "
            )
            report.append("The routing mechanism includes:")
            report.append("- Top-k selection of experts for each token")
            report.append("- Load balancing to prevent expert overutilization")
            report.append("- Capacity factors to handle varying batch sizes")
            report.append("- Expert pruning to eliminate unused experts\n")

            report.append("### Performance Optimizations\n")
            report.append(
                "The implementation includes several performance optimizations:"
            )
            report.append("- Vectorized token processing")
            report.append("- Expert-parallel processing")
            report.append("- Memory-efficient attention")
            report.append("- Gradient checkpointing option")
            report.append("- Specialized experts for different domains\n")

            # Save the report
            report_file = os.path.join(self.config.output_dir, "architecture_report.md")
            with open(report_file, "w") as f:
                f.write("\n".join(report))

            logger.info(f"Architecture report saved to {report_file}")

        except Exception as e:
            logger.error(f"Error creating architecture report: {str(e)}")

    def run_analysis(self):
        """Run the complete architecture analysis"""
        logger.info("Starting architecture analysis...")

        # 1. Analyze all modules
        for module_name in self.config.model_modules_to_analyze:
            self.analyze_module(module_name)

        # 2. Create dependency diagram
        if self.config.create_dependency_graphs:
            self.create_dependency_diagram()

            # Create module-specific diagrams
            for module_name in self.config.model_modules_to_analyze:
                self.create_module_diagram(module_name)

        # 3. Create model summaries for different configurations
        for config_dict in self.config.model_configs_to_analyze:
            self.create_model_summary(config_dict)
            self.analyze_model_parameters(config_dict)

        # 4. Create comprehensive architecture report
        self.create_architecture_report()

        logger.info("Architecture analysis completed")


def main():
    """Main entry point for architecture analyzer"""
    parser = argparse.ArgumentParser(description="Model Architecture Analyzer")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="architecture_analysis",
        help="Directory to save analysis results",
    )
    parser.add_argument(
        "--skip-diagrams",
        action="store_true",
        help="Skip creation of architecture diagrams",
    )
    parser.add_argument(
        "--skip-docs", action="store_true", help="Skip creation of class documentation"
    )

    args = parser.parse_args()

    # Create config
    config = ArchitectureAnalyzerConfig(
        output_dir=args.output_dir,
        create_module_diagrams=not args.skip_diagrams,
        create_dependency_graphs=not args.skip_diagrams,
        create_class_docs=not args.skip_docs,
    )

    # Create analyzer and run analysis
    analyzer = ModelArchitectureAnalyzer(config)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
