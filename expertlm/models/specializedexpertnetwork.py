"""
File: models/specializedexpertnetwork.py
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
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from expertlm.models.domainspecificattention import DomainSpecificAttention
from expertlm.models.hybridpositionalencoding import  HybridPositionalEncoding
from expertlm.models.expertlayer import ExpertLayer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpecializedExpertNetwork(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_sub_experts: int = 4,
        max_seq_len: int = 1024,
        expert_type: str = "general",
        dropout: float = 0.1,
    ):
        super().__init__()

        # Enhanced input validation
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")
        if not isinstance(num_sub_experts, int) or num_sub_experts <= 0:
            raise ValueError(
                f"num_sub_experts must be a positive integer, got {num_sub_experts}"
            )
        if expert_type not in [
            "medical",
            "nursing",
            "general",
            "research",
            "emergency",
        ]:
            raise ValueError(f"Invalid expert_type: {expert_type}")

        try:
            # Initialize components with proper error handling
            self.domain_attention = DomainSpecificAttention(
                d_model=d_model,
                num_heads=max(1, d_model // 64),  # Dynamic head calculation
                dropout=dropout,
            )

            self.pos_encoding = HybridPositionalEncoding(
                d_model=d_model, max_len=max_seq_len, learned=True
            )

            # Enhanced input projection with skip connection
            self.input_proj = nn.Sequential(
                nn.LayerNorm(d_model),  # Normalize first
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
            )

            # Initialize router with proper scaling
            self.sub_router = nn.Linear(d_model, num_sub_experts)
            self.router_temperature = nn.Parameter(torch.ones(1) / math.sqrt(d_model))

            # Initialize sub-experts with domain-specific dropout
            dropout_scale = 0.8 if expert_type != "general" else 1.0
            self.sub_experts = nn.ModuleList(
                [
                    ExpertLayer(
                        d_model, d_model * 4, d_model, dropout=dropout * dropout_scale
                    )
                    for _ in range(num_sub_experts)
                ]
            )

            self.expert_type = expert_type
            self.layer_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            # Enhanced gating mechanism
            self.gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )

            # Initialize all weights properly
            self.apply(self._init_weights)

            # Add specialization metrics
            self.register_buffer("input_statistics", torch.zeros(d_model))
            self.register_buffer("activation_patterns", torch.zeros(num_sub_experts))
            self.specialization_threshold = 0.7

            # Add domain-specific loss tracking
            self.domain_loss_weight = 0.1
            self.domain_statistics = {
                "medical": {"keyword_weight": 1.2, "attention_weight": 1.1},
                "nursing": {"keyword_weight": 1.1, "attention_weight": 1.2},
                "general": {"keyword_weight": 1.0, "attention_weight": 1.0},
                "research": {"keyword_weight": 1.3, "attention_weight": 0.9},
                "emergency": {"keyword_weight": 1.4, "attention_weight": 1.3},
            }

        except Exception as e:
            logger.error(f"Error initializing SpecializedExpertNetwork: {str(e)}")
            raise

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _update_specialization_metrics(
        self, x: torch.Tensor, expert_weights: torch.Tensor
    ):
        """Track expert specialization patterns"""
        with torch.no_grad():
            # Update input statistics
            self.input_statistics = 0.9 * self.input_statistics + 0.1 * x.mean(
                dim=(0, 1)
            )

            # Update activation patterns
            expert_activations = (
                expert_weights > self.specialization_threshold
            ).float()
            self.activation_patterns = (
                0.9 * self.activation_patterns
                + 0.1 * expert_activations.mean(dim=(0, 1))
            )

    def _compute_domain_statistics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute domain-specific statistics for input"""
        with torch.no_grad():
            stats = {
                "mean_activation": x.mean(dim=1),
                "pattern_strength": self.activation_patterns.clone(),
                "attention_distribution": self.domain_attention.get_attention_weights(),
                "expert_utilization": torch.zeros(
                    len(self.sub_experts), device=x.device
                ),
            }
            return stats

    def _calculate_domain_specific_loss(
        self, stats: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Calculate domain-specific loss based on expert type"""
        domain_config = self.domain_statistics[self.expert_type]

        # Calculate weighted loss components
        keyword_loss = stats["mean_activation"].std() * domain_config["keyword_weight"]
        attention_loss = (
            self.entropy(stats["attention_distribution"]).mean()
            * domain_config["attention_weight"]
        )

        # Combine losses
        domain_loss = (keyword_loss + attention_loss) * self.domain_loss_weight
        return domain_loss

    def entropy(self, distribution: torch.Tensor) -> torch.Tensor:
        """Calculate the entropy of a probability distribution"""
        # Avoid log(0) by adding a small epsilon
        eps = 1e-10
        distribution = distribution + eps
        distribution = distribution / distribution.sum(dim=-1, keepdim=True)
        return -torch.sum(distribution * torch.log(distribution), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            # Input validation
            if not isinstance(x, torch.Tensor):
                raise TypeError("Input must be a torch.Tensor")
            if x.dim() != 3:
                raise ValueError(f"Expected 3D tensor, got {x.dim()}D")
            if x.size(-1) != self.input_proj[1].in_features:
                raise ValueError(
                    f"Expected input dimension {self.input_proj[1].in_features}, got {x.size(-1)}"
                )

            # Process input with residual connections
            residual = x
            x = self.input_proj(x)
            x = x + residual

            # Apply positional encoding and attention
            x = self.pos_encoding(x)
            x = self.domain_attention(x)

            # Route to sub-experts with temperature scaling
            router_logits = self.sub_router(x) / self.router_temperature
            scores = F.softmax(router_logits, dim=-1)

            # Process through sub-experts with gating
            output = torch.zeros_like(x)
            for i, expert in enumerate(self.sub_experts):
                expert_weight = scores[..., i].unsqueeze(-1)
                expert_output = expert(x)
                gate_value = self.gate(x)
                output += gate_value * expert_weight * expert_output

            # Final normalization and residual connection
            output = self.layer_norm(output + self.dropout(x))

            # Add domain-specific loss tracking
            self.domain_loss = None  # Reset attribute
            if self.training:
                domain_stats = self._compute_domain_statistics(x)
                self.domain_loss = self._calculate_domain_specific_loss(domain_stats)

            # Update specialization metrics
            self._update_specialization_metrics(x, scores)

            return output

        except Exception as e:
            logger.error(f"Error in SpecializedExpertNetwork forward pass: {str(e)}")
            raise
