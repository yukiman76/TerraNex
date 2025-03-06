import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import defaultdict
import os
import json
from PIL import Image, ImageDraw, ImageFont
# from expertlm.models.moelanguage import MoELanguageModel
from expertlm.models.recurrent_moe_integration import RecurrentMoELanguageModelAdapter
from expertlm.train_rmoe import load_config
from transformers import AutoTokenizer
    

class ModelVisualizer:
    """
    A class to visualize the structure of MoE language models
    that might not work well with torchviz due to conditional execution.
    """
    def __init__(self, model, output_dir='./visualizations'):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def get_model_structure(self):
        """Extract the hierarchical structure of the model"""
        structure = {}
        
        def extract_structure(module, path=''):
            result = {
                'type': type(module).__name__,
                'children': {}
            }
            
            # Add parameters info
            params = {name: param.shape for name, param in module._parameters.items() if param is not None}
            if params:
                result['parameters'] = params
                
            # Add buffers info
            buffers = {name: buffer.shape for name, buffer in module._buffers.items() if buffer is not None}
            if buffers:
                result['buffers'] = buffers
            
            # Process child modules
            for name, child in module.named_children():
                child_path = f"{path}/{name}" if path else name
                result['children'][name] = extract_structure(child, child_path)
            
            return result
        
        structure = extract_structure(self.model)
        return structure
    
    def save_structure_json(self, filename='model_structure.json'):
        """Save the model structure to a JSON file"""
        structure = self.get_model_structure()
        
        # Convert tensors shapes to lists for JSON serialization
        def process_shapes(obj):
            if isinstance(obj, dict):
                result = {}
                for k, v in obj.items():
                    if isinstance(v, torch.Size):
                        result[k] = list(v)
                    elif isinstance(v, dict):
                        result[k] = process_shapes(v)
                    else:
                        result[k] = v
                return result
            return obj
        
        structure = process_shapes(structure)
        
        with open(os.path.join(self.output_dir, filename), 'w') as f:
            json.dump(structure, f, indent=2)
        
        print(f"Model structure saved to {os.path.join(self.output_dir, filename)}")
        return structure
    
    def count_parameters(self):
        """Count trainable parameters in the model"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def visualize_hierarchy(self, filename='model_hierarchy.png'):
        """Create a hierarchical graph visualization of the model architecture"""
        G = nx.DiGraph()
        
        def add_nodes(module, name=''):
            for child_name, child in module.named_children():
                full_name = f"{name}/{child_name}" if name else child_name
                
                # Get parameter count for this module
                param_count = sum(p.numel() for p in child.parameters(True))
                param_str = f"{param_count:,}" if param_count > 0 else "0"
                
                # Create label with module type and parameter count
                label = f"{child_name}\n{type(child).__name__}\nParams: {param_str}"
                
                G.add_node(full_name, label=label, module_type=type(child).__name__)
                
                if name:  # Add edge from parent to child
                    G.add_edge(name, full_name)
                
                # Recursively add children
                add_nodes(child, full_name)
        
        # Start with the model itself
        total_params = self.count_parameters()
        G.add_node('model', label=f"Model: {type(self.model).__name__}\nTotal Params: {total_params:,}", 
                  module_type=type(self.model).__name__)
        
        # Add children
        add_nodes(self.model, 'model')
        
        # Create figure
        plt.figure(figsize=(24, 18))
        
        # Use hierarchical layout
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        
        # Color nodes by type
        module_types = set(nx.get_node_attributes(G, 'module_type').values())
        color_map = {}
        colors = plt.cm.tab20(np.linspace(0, 1, len(module_types)))
        
        for i, module_type in enumerate(module_types):
            color_map[module_type] = colors[i]
        
        node_colors = [color_map[G.nodes[node]['module_type']] for node in G.nodes()]
        
        # Draw the graph
        nx.draw(G, pos, with_labels=False, node_size=3000, node_color=node_colors, 
                font_size=10, arrows=True, arrowsize=20, alpha=0.8)
        
        # Draw labels
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)
        
        # Add legend for module types
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor=color_map[module_type], 
                                    markersize=10, label=module_type)
                          for module_type in color_map]
        
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add title
        plt.title(f"Model Architecture: {type(self.model).__name__}")
        
        # Save and close
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Hierarchical visualization saved to {os.path.join(self.output_dir, filename)}")
    
    def visualize_moe_architecture(self, filename='moe_architecture.png'):
        """Create a specialized visualization for MoE models"""
        # Extract model info
        d_model = getattr(self.model, 'd_model', None)
        vocab_size = None
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding) and 'token' in name.lower():
                vocab_size = module.num_embeddings
                break
        
        # Count layers and experts
        num_layers = 0
        experts_per_layer = {}
        router_info = {}
        
        for name, module in self.model.named_modules():
            if 'expert' in name.lower() and isinstance(module, nn.Linear):
                layer_name = name.split('.')[1] if len(name.split('.')) > 1 else 'unknown'
                if layer_name not in experts_per_layer:
                    experts_per_layer[layer_name] = 0
                experts_per_layer[layer_name] += 1
            
            if 'router' in name.lower() and isinstance(module, nn.Linear):
                layer_name = name.split('.')[1] if len(name.split('.')) > 1 else 'unknown'
                out_features = module.out_features
                router_info[layer_name] = out_features
                
        num_layers = len(experts_per_layer)
        
        # Create an image
        width, height = 1200, 800
        image = Image.new('RGBA', (width, height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        try:
            # Try to load a font, fall back to default if not available
            font_large = ImageFont.truetype("Arial.ttf", 24)
            font_medium = ImageFont.truetype("Arial.ttf", 18)
            font_small = ImageFont.truetype("Arial.ttf", 14)
        except IOError:
            # Use default font
            font_large = ImageFont.load_default()
            font_medium = font_large
            font_small = font_large
        
        # Draw title
        title = f"Mixture of Experts (MoE) Language Model Architecture"
        draw.text((width//2 - 250, 30), title, fill=(0, 0, 0), font=font_large)
        
        # Draw model summary
        summary = f"Model Dimensions: d_model={d_model}, vocab_size={vocab_size}, layers={num_layers}"
        draw.text((width//2 - 200, 70), summary, fill=(0, 0, 0), font=font_medium)
        
        # Draw layers
        layer_height = 80
        y_start = 150
        box_width = 160
        
        # Draw input layer
        draw.rectangle((width//2 - box_width//2, y_start, width//2 + box_width//2, y_start + 50), 
                      outline=(0, 0, 0), fill=(200, 200, 255))
        draw.text((width//2 - 80, y_start + 15), "Input Embeddings", fill=(0, 0, 0), font=font_small)
        
        # Draw transformer layers with MoE FFNs
        for i in range(num_layers):
            y_pos = y_start + 80 + i * (layer_height + 60)
            
            # Layer container
            draw.rectangle((100, y_pos, width - 100, y_pos + layer_height), 
                          outline=(0, 0, 0), fill=(240, 240, 240))
            
            # Layer label
            draw.text((120, y_pos + 10), f"Layer {i}", fill=(0, 0, 0), font=font_medium)
            
            # Self-attention
            attn_x = 280
            draw.rectangle((attn_x, y_pos + 15, attn_x + 150, y_pos + layer_height - 15), 
                          outline=(0, 0, 0), fill=(255, 220, 150))
            draw.text((attn_x + 20, y_pos + 35), "Self-Attention", fill=(0, 0, 0), font=font_small)
            
            # Router
            router_x = attn_x + 200
            draw.rectangle((router_x, y_pos + 25, router_x + 100, y_pos + layer_height - 25), 
                          outline=(0, 0, 0), fill=(255, 150, 150))
            
            num_experts = experts_per_layer.get(f"layer{i}", 0)
            router_text = f"Router\n→ {num_experts} experts"
            draw.text((router_x + 10, y_pos + 35), router_text, fill=(0, 0, 0), font=font_small)
            
            # Experts
            expert_x = router_x + 150
            expert_width = 60
            expert_spacing = 20
            
            for j in range(min(8, num_experts)):  # Draw up to 8 experts to avoid crowding
                x = expert_x + j * (expert_width + expert_spacing)
                draw.rectangle((x, y_pos + 20, x + expert_width, y_pos + layer_height - 20), 
                              outline=(0, 0, 0), fill=(150, 255, 150))
                draw.text((x + 10, y_pos + 35), f"E{j+1}", fill=(0, 0, 0), font=font_small)
            
            # If more experts than shown
            if num_experts > 8:
                draw.text((expert_x + 8 * (expert_width + expert_spacing) + 10, y_pos + 35), 
                         f"... +{num_experts - 8} more", fill=(0, 0, 0), font=font_small)
        
        # Draw output layer
        y_output = y_start + 80 + num_layers * (layer_height + 60)
        draw.rectangle((width//2 - box_width//2, y_output, width//2 + box_width//2, y_output + 50), 
                      outline=(0, 0, 0), fill=(200, 255, 200))
        draw.text((width//2 - 80, y_output + 15), "Output Layer", fill=(0, 0, 0), font=font_small)
        
        # Save the image
        image.save(os.path.join(self.output_dir, filename))
        print(f"MoE architecture visualization saved to {os.path.join(self.output_dir, filename)}")
    
    def visualize_all(self):
        """Generate all visualizations"""
        self.save_structure_json()
        self.visualize_hierarchy()
        self.visualize_moe_architecture()

# Example usage for MoE model visualization
def visualize_moe_model(model, output_dir='./visualizations'):
    """
    Visualize a Mixture of Experts (MoE) language model
    
    Args:
        model: PyTorch MoE model
        output_dir: Directory to save visualizations
    """
    visualizer = ModelVisualizer(model, output_dir)
    visualizer.visualize_all()
    
    # Print model summary
    total_params = visualizer.count_parameters()
    print(f"\nModel Summary:")
    print(f"Type: {type(model).__name__}")
    print(f"Total trainable parameters: {total_params:,}")
    
    # Try to extract important dimensions
    d_model = getattr(model, 'd_model', None)
    if d_model:
        print(f"Dimension (d_model): {d_model}")
    
    # Count MoE specific parts
    num_experts = 0
    num_routers = 0
    
    for name, module in model.named_modules():
        if 'expert' in name.lower() and isinstance(module, nn.Linear):
            num_experts += 1
        if 'router' in name.lower() and isinstance(module, nn.Linear):
            num_routers += 1
    
    print(f"Number of experts: {num_experts}")
    print(f"Number of routers: {num_routers}")

# Modified visualization function that doesn't rely on torchviz
def visualize_model(model, output_dir='./visualizations'):
    """
    Create visualizations of a PyTorch model without using torchviz
    
    Args:
        model: PyTorch model
        output_dir: Directory to save visualizations
    """
    visualize_moe_model(model, output_dir)
    print(f"All visualizations saved to {output_dir}")

# Simplified function to visualize any model, avoiding the problematic torchviz approach
def simple_model_visualizer(model, filename='simple_model_viz.png'):
    """
    Create a simple visualization of model layers using matplotlib
    
    Args:
        model: PyTorch model
        filename: Output filename
    """
    # Collect layer info
    layers = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            layers.append((name, type(module).__name__))
    
    # Count layers by type
    layer_counts = defaultdict(int)
    for _, layer_type in layers:
        layer_counts[layer_type] += 1
    
    # Sort by count
    sorted_layers = sorted(layer_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    labels = [layer_type for layer_type, _ in sorted_layers]
    counts = [count for _, count in sorted_layers]
    
    bars = ax.bar(labels, counts, color='skyblue')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Layer Types in {type(model).__name__}')
    plt.xlabel('Layer Type')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    plt.savefig(filename)
    plt.close()
    
    print(f"Simple model visualization saved to {filename}")
    
    # Also create a module tree text representation
    with open(f"{os.path.splitext(filename)[0]}_tree.txt", 'w') as f:
        def print_module_tree(module, prefix="", file=None):
            for name, child in module.named_children():
                print(f"{prefix}└── {name} ({type(child).__name__})", file=file)
                print_module_tree(child, prefix + "    ", file)
        
        print(f"Model: {type(model).__name__}", file=f)
        print_module_tree(model, prefix="", file=f)
    
    print(f"Model tree structure saved to {os.path.splitext(filename)[0]}_tree.txt")

if __name__ == "__main__":
    # This is a placeholder for your actual model
    # Replace this with your MoE model loading code
    
    # config = {}
    # model = MoELanguageModel(
    #     vocab_size=config.get("vocab_size", 50257),  # default to GPT2 vocab size
    #     d_model=384,
    #     n_layers=config.get("n_layers", 4),
    #     num_experts=config.get("num_experts", 8),
    #     ffn_hidden_dim=config.get("ffn_hidden_dim", 2048),
    #     num_heads=config.get("num_heads", 8),
    #     max_seq_len=1024,
    #     k_experts=config.get("k_experts", 2),
    #     dropout=config.get("dropout", 0.1),
    # )
    # Example usage:
    # Load model from saved files
    import os
    import torch
    from safetensors.torch import load_file
    
    model_path = "moe-model-mps"
    
    # Check if model file exists
    model_file = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")
    
    # Load state dict
    state_dict = load_file(model_file)
    
    # Create model instance with config
    # model = MoELanguageModel(
    #     vocab_size=50257,  # default to GPT2 vocab size
    #     d_model=384,
    #     n_layers=4,
    #     num_experts=8,
    #     ffn_hidden_dim=2048,
    #     num_heads=8,
    #     max_seq_len=1024,
    #     k_experts=2,
    #     dropout=0.1,
    # )
    
    config = load_config("expertlm/utils/model_component_benchmark.py")
    tokenizer_name = config.get("tokenizer", "gpt2")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    model = RecurrentMoELanguageModelAdapter(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        num_experts=config["num_experts"],
        ffn_hidden_dim=config["ffn_hidden_dim"],
        num_heads=config["num_heads"],
        max_seq_len=config["max_seq_len"],
        k_experts=config["k_experts"],
        dropout=config.get("dropout", 0.1),
        router_type=config["router_type"],
        router_hidden_dim=config["router_hidden_dim"],
        add_shared_expert=config["add_shared_expert"],
        shared_expert_weight=config["shared_expert_weight"],
        moe_weight=config["moe_weight"],
        use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
        pad_token_id=tokenizer.pad_token_id,
    )
        
    # Load state dict into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    # Set model to evaluation mode
    model.eval()
    visualize_model(model)
    simple_model_visualizer(model)
    
    print("Run this script with your MoE model loaded.")
    print("Example usage:")
    print("  model = YourMoEModel(...)")
    print("  visualize_model(model)")