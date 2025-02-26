"""
File: models/positionalencoding.py
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        logger.info(f"Creating PE with d_model={d_model}, max_len={max_len}")

        # Ensure integer dimensions
        d_model = int(d_model)
        max_len = int(max_len)

        # Create position indices
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Initialize pe with explicit size and dtype
        pe = torch.zeros((max_len, d_model), dtype=torch.float)

        # Calculate pe values
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        # Register buffer with explicit name
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor"""
        seq_len = x.size(1)
        positional_encoding = self.pe[:seq_len].unsqueeze(0)
        return x + positional_encoding
