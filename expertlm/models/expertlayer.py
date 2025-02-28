"""
File: models/expertlayer.py
Author: Sonny Mir
Email: sonnym@hotmail.se
Last Modified: 02/28/2025

Description: Implements a Routing Mixture-of-Experts (RMoE) architecture.
             Contains the ExpertLayer class which implements a single expert
             neural network with residual connections, layer normalization,
             and dropout for robust training. Each expert processes inputs
             independently based on the router's decisions.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertLayer(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            residual = x
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            x = self.layer_norm(x + residual)
            return x
        except RuntimeError as e:
            logger.error(f"Error in ExpertLayer forward pass: {str(e)}")
            raise

