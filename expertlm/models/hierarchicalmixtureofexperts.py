"""
File: models/hierarchicalmixtureofexperts.py
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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from expertlm.models.hybridpositionalencoding import HybridPositionalEncoding
from expertlm.models.specializedexpertnetwork import SpecializedExpertNetwork

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HierarchicalMixtureOfExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_seq_len: int = 1024,
        k: int = 2,
        dropout: float = 0.1,
        load_balance: bool = True,
    ):
        super().__init__()

        # Enhanced input validation
        if not all(
            isinstance(x, int)
            for x in [num_experts, input_dim, hidden_dim, output_dim, max_seq_len]
        ):
            raise TypeError("Dimensions must be integers")
        if not all(
            x > 0 for x in [num_experts, input_dim, hidden_dim, output_dim, max_seq_len]
        ):
            raise ValueError("All dimensions must be positive")
        if not 0 < k <= num_experts:
            raise ValueError(
                f"k ({k}) must be between 1 and num_experts ({num_experts})"
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        self.load_balance = load_balance
        self.num_experts = num_experts
        self.k = k

        # Initialize components with proper error handling
        try:
            self.main_pos_encoding = HybridPositionalEncoding(
                d_model=input_dim, max_len=max_seq_len, learned=False
            )

            self.router = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),  # Add dropout for regularization
                nn.Linear(hidden_dim, num_experts),
            )

            # Initialize router weights properly
            for layer in self.router:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

            self.router_temperature = nn.Parameter(torch.ones(1))
            self.capacity_factor = 1.25

            expert_types = ["medical", "nursing", "general", "research", "emergency"]
            self.experts = nn.ModuleList(
                [
                    SpecializedExpertNetwork(
                        d_model=input_dim,
                        num_sub_experts=4,
                        max_seq_len=max_seq_len,
                        expert_type=expert_types[i % len(expert_types)],
                        dropout=dropout,
                    )
                    for i in range(num_experts)
                ]
            )

            self.expert_combiner = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, input_dim),
                nn.LayerNorm(input_dim),
            )

            # Add expert pruning
            self.register_buffer("expert_importance", torch.ones(num_experts))

            # Register a dummy buffer to help determine device in forward pass
            self.register_buffer("dummy", torch.zeros(1))

            # Add adaptive capacity factor
            self.min_capacity_factor = 1.0
            self.max_capacity_factor = 2.0
            self.capacity_factor_decay = 0.99

            # Add expert pruning parameters
            self.expert_pruning_threshold = 0.001
            self.pruning_warmup_steps = 1000
            self.steps_since_pruning = 0
            self.pruning_interval = 5000
            self.min_active_experts = num_experts // 2

        except Exception as e:
            logger.error(f"Error initializing HierarchicalMixtureOfExperts: {str(e)}")
            raise

    def _compute_capacity(self, router_probs: torch.Tensor) -> torch.Tensor:
        try:
            expert_load = router_probs.sum(dim=(0, 1)) / router_probs.shape[0]
            capacity_loss = torch.mean(expert_load**2) * self.num_experts
            return capacity_loss
        except Exception as e:
            logger.error(f"Error computing capacity: {str(e)}")
            raise

    def _update_expert_importance(self, router_probs: torch.Tensor):
        """Update expert importance scores based on usage"""
        with torch.no_grad():
            # Calculate expert usage
            expert_usage = router_probs.sum(dim=(0, 1))

            # Update importance scores with exponential moving average
            self.expert_importance = (
                0.9 * self.expert_importance + 0.1 * expert_usage / expert_usage.sum()
            )

    def _adjust_capacity_factor(self, expert_counts: torch.Tensor):
        """Dynamically adjust capacity factor based on expert utilization"""
        max_usage = expert_counts.max().item()
        target_usage = expert_counts.sum().item() / self.num_experts

        if max_usage > target_usage * 1.5:
            self.capacity_factor = min(
                self.max_capacity_factor,
                self.capacity_factor / self.capacity_factor_decay,
            )
        elif max_usage < target_usage:
            self.capacity_factor = max(
                self.min_capacity_factor,
                self.capacity_factor * self.capacity_factor_decay,
            )

    def _prune_inactive_experts(self) -> None:
        """Prune experts that are consistently unused"""
        try:
            with torch.no_grad():
                # Only prune after warmup and at intervals
                if (
                    self.steps_since_pruning < self.pruning_warmup_steps
                    or self.steps_since_pruning % self.pruning_interval != 0
                ):
                    return

                # Calculate expert importance scores
                importance_scores = self.expert_importance.clone()

                # Find inactive experts
                inactive_mask = importance_scores < self.expert_pruning_threshold
                active_experts = (~inactive_mask).sum()

                # Ensure minimum number of active experts
                if active_experts > self.min_active_experts:
                    inactive_indices = torch.where(inactive_mask)[0]

                    if len(inactive_indices) > 0:
                        logger.info(f"Pruning {len(inactive_indices)} inactive experts")

                        # Reinitialize pruned experts
                        for idx in inactive_indices:
                            # Reset expert weights
                            self.experts[idx].apply(self._init_weights)
                            # Reset importance score
                            self.expert_importance[idx] = 1.0

                        # Log pruning event
                        logger.info(f"Experts pruned: {inactive_indices.tolist()}")
                        logger.info(f"Remaining active experts: {active_experts}")

                # Reset steps counter
                self.steps_since_pruning = 0

        except Exception as e:
            logger.error(f"Error in expert pruning: {str(e)}")
            raise

    def forward(
        self, x: torch.Tensor, is_training: bool = True, step: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the hierarchical MoE

        Args:
            x: input tensor of shape [batch_size, seq_len, input_dim]
            is_training: whether in training mode
            step: current training step (used for pruning)

        Returns:
            output tensor, router_logits, capacity_loss
        """
        try:
            batch_size, seq_len, input_dim = x.shape
            device = x.device

            # Ensure expert_importance is on the right device
            if self.expert_importance.device != device:
                self.expert_importance = self.expert_importance.to(device)

            # Apply fixed positional encoding at main router level
            pos_encoded = self.main_pos_encoding(x)

            # Generate router logits
            router_logits = self.router(pos_encoded)

            # Apply temperature scaling to logits
            router_logits = router_logits / (self.router_temperature + 1e-6)

            # Masked router probs for top-k routing
            router_probs = F.softmax(router_logits, dim=-1)

            # Get indices of top-k experts per token
            _, indices = torch.topk(router_probs, k=self.k, dim=-1)

            # Prepare storage for outputs
            combined_output = torch.zeros_like(x)

            # Loop through each expert selected per token, allowing gradient flow
            for sample_idx in range(batch_size):
                for token_idx in range(seq_len):
                    token_experts = indices[sample_idx, token_idx]
                    token_probs = router_probs[sample_idx, token_idx, token_experts]
                    token_input = (
                        pos_encoded[sample_idx, token_idx].unsqueeze(0).unsqueeze(0)
                    )

                    # Process through selected experts with their weights
                    expert_outputs = []
                    for expert_idx, prob in zip(token_experts, token_probs):
                        expert_output = self.experts[expert_idx](token_input).squeeze(0)
                        expert_outputs.append(expert_output * prob)

                        # Update expert importance (during training)
                        if is_training:
                            self.expert_importance[expert_idx] = (
                                0.99 * self.expert_importance[expert_idx]
                                + 0.01 * prob.detach()
                            )

                    # Combine expert outputs
                    token_output = sum(expert_outputs)
                    combined_output[sample_idx, token_idx] = token_output

            # Calculate capacity loss if load balancing
            capacity_loss = (
                self._compute_capacity(router_probs)
                if self.load_balance
                else torch.tensor(0.0, device=device)
            )

            # Apply pruning during training at specified intervals
            if is_training and step is not None:
                self.steps_since_pruning += 1
                if (
                    step > self.pruning_warmup_steps
                    and self.steps_since_pruning >= self.pruning_interval
                ):
                    pruned = self.prune_experts()
                    if pruned:
                        self.steps_since_pruning = 0

            # Final output combination
            output = self.expert_combiner(combined_output)
            return output, router_logits, capacity_loss

        except Exception as e:
            logger.error(
                f"Error in HierarchicalMixtureOfExperts forward pass: {str(e)}"
            )
            raise

    def prune_experts(self):
        """Prune inactive experts based on their importance scores"""
        try:
            # Get device from a tensor to ensure we're on the right device
            device = self.dummy.device

            # Sort experts by importance
            importance = self.expert_importance.clone()
            sorted_indices = torch.argsort(importance)

            # Identify experts to prune (keeping at least min_active_experts)
            num_to_prune = max(0, self.num_experts - self.min_active_experts)
            if num_to_prune == 0:
                return False  # Nothing to prune

            # Get indices of experts to prune (lowest importance)
            prune_indices = sorted_indices[:num_to_prune]

            # Only prune if importance is below threshold
            prune_mask = importance[prune_indices] < self.expert_pruning_threshold
            prune_indices = prune_indices[prune_mask]

            if len(prune_indices) == 0:
                return False  # No experts below threshold

            logger.info(
                f"Pruning {len(prune_indices)} experts: {prune_indices.tolist()}"
            )

            # Re-initialize the experts that need pruning
            expert_types = ["medical", "nursing", "general", "research", "emergency"]
            for idx in prune_indices:
                # Reset expert importance
                self.expert_importance[idx] = 1.0

                # Re-initialize the expert with the same type but new weights
                expert_type = expert_types[idx % len(expert_types)]
                self.experts[idx] = SpecializedExpertNetwork(
                    d_model=self.experts[idx].d_model,
                    num_sub_experts=self.experts[idx].num_sub_experts,
                    max_seq_len=self.experts[idx].max_seq_len,
                    expert_type=expert_type,
                    dropout=0.1,  # Default dropout
                ).to(device)

            return True  # Pruning occurred

        except Exception as e:
            logger.error(f"Error during expert pruning: {str(e)}")
            return False
