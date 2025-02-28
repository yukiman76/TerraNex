# Recurrent Mixture-of-Experts (RMoE)

This implementation provides a complete Recurrent Mixture-of-Experts (RMoE) architecture with a GRU-based recurrent router, as described in the paper "Layerwise Recurrent Router for Mixture-of-Experts".

## Overview

The RMoE architecture enhances traditional Mixture-of-Experts (MoE) models by incorporating a recurrent router that maintains state across tokens in a sequence. This enables more context-aware routing decisions, leading to improved performance and more coherent expert selection.

## Key Components

### Recurrent Router

The recurrent router uses either GRU or RNN cells to maintain state across tokens, enabling the router to make decisions based on the sequence context rather than treating each token independently. This leads to more coherent expert selection and better utilization of expert capacity.

### Expert Layers

Each expert is implemented as a feed-forward network with a SwiGLU-like activation function, similar to modern transformer architectures. The experts are specialized to handle different aspects of the input data.

### MoE Block

The MoE block combines the recurrent router with a set of experts, routing each token to the top-k experts based on the router's decisions. It also includes options for a shared expert that processes all tokens, providing a balance between specialization and generalization.

## Files

- `recurrent_router.py`: Implements the recurrent router with GRU and RNN cell options.
- `recurrent_moe.py`: Implements the Recurrent MoE layer and block.
- `recurrent_moe_language_model.py`: Implements a language model that uses the Recurrent MoE architecture.
- `config.py`: Provides configuration classes for the Recurrent MoE model.

## Usage

```python
import torch
from prototype.config import RecurrentMoEModelConfig
from prototype.recurrent_moe_language_model import RecurrentMoELanguageModel

# Create a model configuration
config = RecurrentMoEModelConfig(
    vocab_size=32000,
    d_model=768,
    n_layers=12,
    num_heads=12,
    num_experts=8,
    ffn_hidden_dim=3072,
    router_hidden_dim=256,
    max_seq_len=2048,
    k_experts=2,
    router_type="gru",
    add_shared_expert=True,
    shared_expert_weight=0.5,
    moe_weight=0.5,
)

# Create the model
model = RecurrentMoELanguageModel(
    vocab_size=config.vocab_size,
    d_model=config.d_model,
    n_layers=config.n_layers,
    num_heads=config.num_heads,
    num_experts=config.moe_config.num_experts,
    ffn_hidden_dim=config.moe_config.expert_hidden_dim,
    router_hidden_dim=config.moe_config.router_config.hidden_dim,
    max_seq_len=config.max_seq_len,
    k_experts=config.moe_config.k_experts,
    router_type=config.moe_config.router_config.cell_type,
    add_shared_expert=config.moe_config.add_shared_expert,
    shared_expert_weight=config.moe_config.shared_expert_weight,
    moe_weight=config.moe_config.moe_weight,
)

# Generate text
input_ids = torch.randint(0, config.vocab_size, (1, 10))
generated_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)
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

## References

- "Layerwise Recurrent Router for Mixture-of-Experts"
- "Mixture-of-Experts with Expert Choice Routing"
- "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" 