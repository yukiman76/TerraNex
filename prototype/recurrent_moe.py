"""
File: prototype/recurrent_moe.py
Description: Implements a Recurrent Mixture-of-Experts (RMoE) layer that uses
             a recurrent router to make routing decisions based on sequence context.
             
             This implementation features:
             - A GRU-based recurrent router that maintains state across tokens
             - Expert layers with SwiGLU-like activation functions
             - Top-k expert selection for each token
             - Optional shared expert that processes all tokens
             - Auxiliary load balancing loss to encourage uniform expert utilization
             
             The recurrent nature of the router enables more coherent expert selection
             across tokens in a sequence, leading to better specialization and more
             efficient use of model capacity. The GRU cell provides a good balance
             between complexity and performance, capturing relevant context information
             while remaining computationally efficient.

Author: Sonny Mir
Email: sonnym@hotmail.se
Date: Feb 28, 2024
Last Modified: Feb 28, 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from prototype.recurrent_router import RecurrentRouter, GRURouterCell, RNNRouterCell


class ExpertLayer(nn.Module):
    """
    Expert layer implementation for the Mixture-of-Experts model.
    
    Each expert is a feed-forward network with a single hidden layer.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define activation function
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        elif activation.lower() == "silu" or activation.lower() == "swish":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Expert network
        self.up_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.gate_proj = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, output_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the expert layer."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass of the expert layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim) or (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim) or (batch_size, output_dim)
        """
        # SwiGLU activation as used in modern transformer architectures
        hidden = self.activation(self.gate_proj(x)) * self.up_proj(x)
        hidden = self.dropout(hidden)
        output = self.down_proj(hidden)
        
        return output


class RecurrentMoE(nn.Module):
    """
    Recurrent Mixture-of-Experts (RMoE) layer.
    
    This layer uses a recurrent router to make routing decisions based on
    sequence context, enabling more coherent expert selection across tokens.
    """
    
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        router_hidden_dim: int,
        k: int = 2,
        router_type: str = "gru",
        dropout: float = 0.1,
        add_shared_expert: bool = False,
        shared_expert_weight: float = 0.5,
        moe_weight: float = 0.5,
        add_bias: bool = False,
        router_aux_loss_coef: float = 0.001,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.router_hidden_dim = router_hidden_dim
        self.k = k
        self.add_shared_expert = add_shared_expert
        self.shared_expert_weight = shared_expert_weight
        self.moe_weight = moe_weight
        self.add_bias = add_bias
        self.router_aux_loss_coef = router_aux_loss_coef
        
        # Initialize experts
        self.experts = nn.ModuleList([
            ExpertLayer(input_dim, hidden_dim, output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Initialize shared expert if needed
        if add_shared_expert:
            self.shared_expert = ExpertLayer(input_dim, hidden_dim, output_dim, dropout)
        
        # Initialize recurrent router
        self.router = RecurrentRouter(
            input_dim=input_dim,
            hidden_dim=router_hidden_dim,
            num_experts=num_experts,
            cell_type=router_type,
            dropout=dropout
        )
        
        # Add bias if needed
        if add_bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Initialize auxiliary loss tracking
        self.aux_loss = 0.0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_hidden_state: Optional[torch.Tensor] = None,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the Recurrent MoE layer.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, input_dim)
            router_hidden_state: Optional hidden state from previous call
            output_router_logits: Whether to output router logits
            
        Returns:
            Tuple containing:
            - output: Output tensor of shape (batch_size, seq_len, output_dim)
            - router_logits: Router logits if output_router_logits is True, else None
            - router_hidden_state: Updated router hidden state
            - aux_loss: Auxiliary load balancing loss
        """
        batch_size, seq_len, input_dim = hidden_states.shape
        device = hidden_states.device
        
        # Get router outputs
        router_logits, router_probs, new_router_hidden = self.router(hidden_states, router_hidden_state)
        
        # Compute auxiliary load balancing loss
        aux_loss = self.router.compute_load_balancing_loss(router_probs)
        self.aux_loss = aux_loss.item()
        
        # Get top-k experts and their probabilities
        router_probs_top_k, indices = torch.topk(router_probs, k=self.k, dim=-1)
        
        # Normalize the routing probabilities
        router_probs_top_k = router_probs_top_k / router_probs_top_k.sum(dim=-1, keepdim=True)
        
        # Initialize output tensor
        moe_output = torch.zeros_like(hidden_states)
        
        # Process each token through its assigned experts
        for batch_idx in range(batch_size):
            for seq_idx in range(seq_len):
                token_input = hidden_states[batch_idx, seq_idx].unsqueeze(0)  # (1, input_dim)
                
                # Get the top-k experts for this token
                token_experts = indices[batch_idx, seq_idx]  # (k,)
                token_probs = router_probs_top_k[batch_idx, seq_idx]  # (k,)
                
                # Process token through each assigned expert
                token_output = torch.zeros(1, self.output_dim, device=device)
                for k_idx in range(self.k):
                    expert_idx = token_experts[k_idx].item()
                    expert_prob = token_probs[k_idx].item()
                    
                    # Get expert output and weight by probability
                    expert_output = self.experts[expert_idx](token_input)
                    token_output += expert_output * expert_prob
                
                # Store the result
                moe_output[batch_idx, seq_idx] = token_output
        
        # Add shared expert contribution if needed
        if self.add_shared_expert:
            shared_output = self.shared_expert(hidden_states)
            final_output = (moe_output * self.moe_weight) + (shared_output * self.shared_expert_weight)
        else:
            final_output = moe_output
        
        # Add bias if needed
        if self.add_bias:
            final_output = final_output + self.bias
        
        # Return appropriate outputs
        if output_router_logits:
            return final_output, router_logits, new_router_hidden, aux_loss
        else:
            return final_output, None, new_router_hidden, aux_loss


class RecurrentMoEBlock(nn.Module):
    """
    A complete Recurrent MoE block including normalization and residual connection.
    
    This block can be used as a drop-in replacement for a feed-forward network
    in a transformer architecture.
    """
    
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        router_hidden_dim: int,
        k: int = 2,
        router_type: str = "gru",
        dropout: float = 0.1,
        add_shared_expert: bool = False,
        shared_expert_weight: float = 0.5,
        moe_weight: float = 0.5,
        add_bias: bool = False,
        router_aux_loss_coef: float = 0.001,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.router_aux_loss_coef = router_aux_loss_coef
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dim, eps=norm_eps)
        
        # Recurrent MoE layer
        self.moe = RecurrentMoE(
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,  # Output dim matches input for residual connection
            router_hidden_dim=router_hidden_dim,
            k=k,
            router_type=router_type,
            dropout=dropout,
            add_shared_expert=add_shared_expert,
            shared_expert_weight=shared_expert_weight,
            moe_weight=moe_weight,
            add_bias=add_bias,
            router_aux_loss_coef=router_aux_loss_coef,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_hidden_state: Optional[torch.Tensor] = None,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the Recurrent MoE block.
        
        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, input_dim)
            router_hidden_state: Optional hidden state from previous call
            output_router_logits: Whether to output router logits
            
        Returns:
            Tuple containing:
            - output: Output tensor of shape (batch_size, seq_len, input_dim)
            - router_logits: Router logits if output_router_logits is True, else None
            - router_hidden_state: Updated router hidden state
            - aux_loss: Auxiliary load balancing loss
        """
        # Apply layer normalization
        normalized_states = self.norm(hidden_states)
        
        # Apply MoE layer
        moe_output, router_logits, new_router_hidden, aux_loss = self.moe(
            normalized_states,
            router_hidden_state=router_hidden_state,
            output_router_logits=output_router_logits
        )
        
        # Apply residual connection
        output = hidden_states + moe_output
        
        return output, router_logits, new_router_hidden, aux_loss 