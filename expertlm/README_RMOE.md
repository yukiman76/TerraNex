# Recurrent Mixture-of-Experts (RMoE) Integration

Author: Sonny Mir  
Email: sonnym@hotmail.se  
Last Modified: Feb 28, 2024

This document explains how to use the Recurrent Mixture-of-Experts (RMoE) implementation with the expertlm codebase. The RMoE architecture enhances traditional Mixture-of-Experts (MoE) models by incorporating a recurrent router that maintains state across tokens in a sequence, enabling more context-aware routing decisions.

## Overview

The RMoE implementation is located in the `prototype` folder and has been integrated with the expertlm codebase through adapter classes in `expertlm/models/recurrent_moe_integration.py`. This integration allows you to use the RMoE architecture as a drop-in replacement for the existing MoE implementation in expertlm.

## Key Components

1. **Recurrent Router**: Uses GRU or RNN cells to maintain state across tokens, enabling the router to make decisions based on the sequence context rather than treating each token independently.

2. **Expert Layers**: Each expert is implemented as a feed-forward network with a SwiGLU-like activation function, similar to modern transformer architectures.

3. **MoE Block**: Combines the recurrent router with a set of experts, routing each token to the top-k experts based on the router's decisions.

## Files

- `prototype/recurrent_router.py`: Implements the recurrent router with GRU and RNN cell options.
- `prototype/recurrent_moe.py`: Implements the Recurrent MoE layer and block.
- `prototype/recurrent_moe_language_model.py`: Implements a language model that uses the Recurrent MoE architecture.
- `prototype/config.py`: Provides configuration classes for the Recurrent MoE model.
- `expertlm/models/recurrent_moe_integration.py`: Provides adapter classes to integrate the RMoE implementation with the expertlm codebase.
- `expertlm/train_rmoe.py`: Modified version of the training script that uses the RMoE implementation.
- `expertlm/inference_rmoe.py`: Modified version of the inference script that uses the RMoE implementation.
- `expertlm/configs/rmoe_config.yaml`: Sample configuration file for training the RMoE model.

## Usage

### Training

To train a model using the RMoE architecture, use the `train_rmoe.py` script:

```bash
python -m expertlm.train_rmoe --config expertlm/configs/rmoe_config.yaml
```

You can customize the configuration file to adjust the model architecture, training parameters, and data configuration.

### Inference

To run inference with a trained RMoE model, use the `inference_rmoe.py` script:

```bash
python -m expertlm.inference_rmoe --model_path ./output/rmoe --prompt "Once upon a time" --max_new_tokens 100
```

For interactive mode:

```bash
python -m expertlm.inference_rmoe --model_path ./output/rmoe --interactive
```

To analyze expert usage:

```bash
python -m expertlm.inference_rmoe --model_path ./output/rmoe --prompt "Once upon a time" --analyze
```

### Using the RMoE Model in Custom Code

You can use the RMoE model in your custom code by importing the adapter classes:

```python
from expertlm.models import RecurrentMoELanguageModelAdapter, create_rmoe_config_from_dict

# Create model configuration
config = {
    "vocab_size": 50257,
    "d_model": 768,
    "n_layers": 12,
    "num_heads": 12,
    "num_experts": 8,
    "ffn_hidden_dim": 3072,
    "max_seq_len": 1024,
    "k_experts": 2,
    "router_type": "gru",
    "router_hidden_dim": 256,
}

# Create model
model = RecurrentMoELanguageModelAdapter(
    vocab_size=config["vocab_size"],
    d_model=config["d_model"],
    n_layers=config["n_layers"],
    num_experts=config["num_experts"],
    ffn_hidden_dim=config["ffn_hidden_dim"],
    num_heads=config["num_heads"],
    max_seq_len=config["max_seq_len"],
    k_experts=config["k_experts"],
    router_type=config["router_type"],
    router_hidden_dim=config["router_hidden_dim"],
)

# Use the model
# ...
```

## Advantages of Recurrent Routing

1. **Context-Aware Routing**: The router makes decisions based on the sequence context, not just the current token.
2. **Improved Expert Utilization**: More coherent routing leads to better utilization of expert capacity.
3. **Reduced Load Imbalance**: The recurrent nature helps distribute tokens more evenly across experts.
4. **Enhanced Specialization**: Experts can specialize in handling specific types of sequence patterns.

## Implementation Details

### Recurrent Cell Types

- **GRU**: Uses a Gated Recurrent Unit cell for the router, providing a good balance between complexity and performance.
- **RNN**: Uses a simple RNN cell for the router, offering a lighter-weight alternative.

### Expert Selection

The router selects the top-k experts for each token based on the router probabilities. The outputs from these experts are weighted by their corresponding probabilities and combined to form the final output.

### Shared Expert

Optionally, a shared expert can be included that processes all tokens. This provides a balance between specialization (through the MoE experts) and generalization (through the shared expert).

### Auxiliary Loss

The implementation includes an auxiliary load balancing loss that encourages uniform utilization of experts, preventing the "rich get richer" problem where a few experts dominate the routing decisions.

## Configuration Options

The RMoE model supports the following configuration options:

- `router_type`: Type of recurrent cell to use (`"gru"` or `"rnn"`).
- `router_hidden_dim`: Hidden dimension of the router.
- `add_shared_expert`: Whether to add a shared expert that processes all tokens.
- `shared_expert_weight`: Weight for the shared expert output.
- `moe_weight`: Weight for the mixture of experts output.

See the sample configuration file `expertlm/configs/rmoe_config.yaml` for a complete list of configuration options. 