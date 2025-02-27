"""
File: models/domainspecificattention.py
Author: Jeffrey Rivero
Email: jeff@check-ai.com
Created: 02/20/2025
Last Modified: 02/24/2025
Description: Implements a Mixture-of-Experts (MoE) language model architecture.
             Includes core components like ExpertLayer, PositionalEncoding,
             MixtureOfExperts, and the main MoELanguageModel class with
             generation capabilities.
"""

import logging

import torch
import torch.nn as nn
import torch.utils.checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DomainSpecificAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Add saved attention weights
        self.register_buffer("last_attention_weights", None, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            residual = x
            attn_out, attn_weights = self.mha(x, x, x, need_weights=True)

            # Save attention weights for domain-specific metrics
            self.last_attention_weights = attn_weights.detach()

            return self.layer_norm(residual + self.dropout(attn_out))
        except Exception as e:
            logger.error(f"Error in DomainSpecificAttention forward pass: {str(e)}")
            raise

    def get_attention_weights(self):
        """Return the last saved attention weights"""
        if self.last_attention_weights is None:
            # Return a default tensor if no weights saved yet
            return torch.ones(1, device=self.layer_norm.weight.device)
        return self.last_attention_weights
