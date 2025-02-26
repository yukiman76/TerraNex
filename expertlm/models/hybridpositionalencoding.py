"""
File: models/hybridpositionalencoding.py
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
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, learned: bool = False):
        super().__init__()
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.learned = learned

        if learned:
            # Learned positional embeddings
            self.learned_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.learned_pe, mean=0.0, std=0.02)
        else:
            # Fixed sinusoidal embeddings
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )

            pe = torch.zeros((max_len, d_model), dtype=torch.float)
            pe[:, 0::2] = torch.sin(position * div_term)
            if d_model % 2 == 0:
                pe[:, 1::2] = torch.cos(position * div_term)
            else:
                pe[:, 1::2] = torch.cos(position * div_term[:-1])

            self.register_buffer("fixed_pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            seq_len = x.size(1)
            if self.learned:
                return x + self.learned_pe[:, :seq_len]
            return x + self.fixed_pe[:seq_len].unsqueeze(0)
        except Exception as e:
            logger.error(f"Error in HybridPositionalEncoding forward pass: {str(e)}")
            raise
