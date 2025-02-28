"""
File: prototype/recurrent_router.py
Description: Implements a recurrent router for Mixture-of-Experts (MoE) models.
             This router uses GRU or RNN cells to maintain state across tokens,
             enabling more context-aware routing decisions.
             
             The GRU-based router is the primary innovation in this implementation,
             allowing the router to consider the history of tokens when making
             routing decisions. This leads to more coherent expert selection and
             improved specialization, as experts can focus on specific sequence
             patterns rather than individual tokens.
             
             The router produces a probability distribution over experts for each
             token, which is used to select the top-k experts for processing.
             The hidden state is updated after each token, maintaining context
             information throughout the sequence.

Author: Sonny Mir
Email: sonnym@hotmail.se
Date: Feb 28, 2024
Last Modified: Feb 28, 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RecurrentRouterCell(nn.Module):
    """Base class for recurrent router cells."""
    
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, x, h=None):
        raise NotImplementedError("Subclasses must implement forward method")


class GRURouterCell(RecurrentRouterCell):
    """GRU cell implementation for recurrent routing."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__(hidden_size)
        self.input_size = input_size
        
        # Reset gate parameters
        self.weight_ir = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hr = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ir = nn.Parameter(torch.empty(hidden_size))
        self.bias_hr = nn.Parameter(torch.empty(hidden_size))
        
        # Update gate parameters
        self.weight_iz = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hz = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_iz = nn.Parameter(torch.empty(hidden_size))
        self.bias_hz = nn.Parameter(torch.empty(hidden_size))
        
        # New gate parameters
        self.weight_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_in = nn.Parameter(torch.empty(hidden_size))
        self.bias_hn = nn.Parameter(torch.empty(hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        for weight in [self.weight_ir, self.weight_hr, self.weight_iz, 
                      self.weight_hz, self.weight_in, self.weight_hn]:
            nn.init.xavier_uniform_(weight)
            
        for bias in [self.bias_ir, self.bias_hr, self.bias_iz, 
                    self.bias_hz, self.bias_in, self.bias_hn]:
            nn.init.zeros_(bias)
    
    def forward(self, x, h=None):
        """
        Forward pass of the GRU cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h: Hidden state tensor of shape (batch_size, hidden_size), optional
            
        Returns:
            h_new: New hidden state tensor of shape (batch_size, hidden_size)
        """
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        
        # Reset gate
        r = torch.sigmoid(F.linear(x, self.weight_ir, self.bias_ir) + 
                         F.linear(h, self.weight_hr, self.bias_hr))
        
        # Update gate
        z = torch.sigmoid(F.linear(x, self.weight_iz, self.bias_iz) + 
                         F.linear(h, self.weight_hz, self.bias_hz))
        
        # New gate
        n = torch.tanh(F.linear(x, self.weight_in, self.bias_in) + 
                      r * F.linear(h, self.weight_hn, self.bias_hn))
        
        # Final output
        h_new = (1 - z) * n + z * h
        
        return h_new


class RNNRouterCell(RecurrentRouterCell):
    """Simple RNN cell implementation for recurrent routing."""
    
    def __init__(self, input_size, hidden_size):
        super().__init__(hidden_size)
        self.input_size = input_size
        
        self.weight_in = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hn = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_in = nn.Parameter(torch.empty(hidden_size))
        self.bias_hn = nn.Parameter(torch.empty(hidden_size))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight_in)
        nn.init.xavier_uniform_(self.weight_hn)
        nn.init.zeros_(self.bias_in)
        nn.init.zeros_(self.bias_hn)
    
    def forward(self, x, h=None):
        """
        Forward pass of the RNN cell.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            h: Hidden state tensor of shape (batch_size, hidden_size), optional
            
        Returns:
            h_new: New hidden state tensor of shape (batch_size, hidden_size)
        """
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device, dtype=x.dtype)
        
        h_new = torch.tanh(F.linear(x, self.weight_in, self.bias_in) + 
                          F.linear(h, self.weight_hn, self.bias_hn))
        
        return h_new


class RecurrentRouter(nn.Module):
    """
    Recurrent router for Mixture-of-Experts models.
    
    This router maintains state across tokens in a sequence, enabling more
    context-aware routing decisions. It can use either GRU or RNN cells.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_experts: int,
        cell_type: str = "gru",
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.cell_type = cell_type.lower()
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Recurrent cell
        if self.cell_type == "gru":
            self.rnn_cell = GRURouterCell(hidden_dim, hidden_dim)
        elif self.cell_type == "rnn":
            self.rnn_cell = RNNRouterCell(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}. Use 'gru' or 'rnn'.")
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, num_experts)
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the router."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=1.0 / math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass of the recurrent router.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            hidden_state: Optional hidden state from previous call
            
        Returns:
            Tuple containing:
            - router_logits: Logits for expert selection
            - router_probs: Probabilities for expert selection
            - new_hidden_state: Updated hidden state
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Project input
        projected_input = self.input_projection(x)
        
        # Process sequence through recurrent cell
        hidden_states = []
        current_hidden = hidden_state
        
        for t in range(seq_len):
            token_input = projected_input[:, t]
            current_hidden = self.rnn_cell(token_input, current_hidden)
            hidden_states.append(current_hidden)
        
        # Stack hidden states
        hidden_states = torch.stack(hidden_states, dim=1)
        
        # Generate router logits
        router_logits = self.output_projection(hidden_states)
        
        # Apply temperature scaling
        scaled_logits = router_logits / (self.temperature.abs() + 1e-6)
        
        # Calculate router probabilities
        router_probs = F.softmax(scaled_logits, dim=-1)
        
        return router_logits, router_probs, current_hidden
    
    def compute_load_balancing_loss(self, router_probs):
        """
        Compute the load balancing loss for the router.
        
        Args:
            router_probs: Router probabilities of shape (batch_size, seq_len, num_experts)
            
        Returns:
            Load balancing loss
        """
        # Calculate mean routing probability per expert
        mean_prob_per_expert = router_probs.mean(dim=[0, 1])
        
        # Calculate coefficient of variation (ideal is 1/num_experts for each expert)
        ideal_prob = 1.0 / self.num_experts
        expert_imbalance = torch.sum(mean_prob_per_expert * torch.log(mean_prob_per_expert + 1e-10) + 
                                    torch.log(ideal_prob + 1e-10) * ideal_prob)
        
        return expert_imbalance 