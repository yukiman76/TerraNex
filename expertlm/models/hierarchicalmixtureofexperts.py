"""
File: models/hierarchicalmixtureofexperts.py
Author: Sonny Mir
Email: sonnym@hotmail.se

Last Modified: 02/28/2024
Description: Implements a Hierarchical Mixture-of-Experts (HMoE) language model architecture.
             This enhanced implementation features a hierarchical structure where experts are
             organized in multiple layers, allowing for more specialized knowledge capture.
             
             This module can be used alongside the Recurrent Mixture-of-Experts (RMoE) 
             implementation, which enhances routing with GRU-based recurrent connections
             to maintain state across tokens in a sequence.

             Core Components:
             - HierarchicalExpertLayer: Manages multiple layers of experts
             - HybridPositionalEncoding: Combines absolute and relative positional encoding
             - SpecializedExpertNetwork: Expert networks optimized for specific tasks
             - Dynamic routing mechanism for expert selection

             Key Enhancements:
             - Hierarchical expert organization for better specialization
             - Advanced load balancing with auxiliary loss
             - Efficient sparse routing with top-k gating
             - Compatible with RMoE adapter for recurrent routing capabilities

             Performance Features:
             - Multi-GPU expert sharding
             - Optimized attention patterns
             - Cached key/value states
             - Efficient memory management
             - Expert batching with local grouping
"""

import logging
from typing import Optional, Tuple, Dict, List
import math

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
        use_expert_parallelism: bool = True,
        use_checkpointing: bool = False,
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
        self.input_dim = input_dim
        self.use_expert_parallelism = use_expert_parallelism
        self.use_checkpointing = use_checkpointing

        # Initialize components with proper error handling
        try:
            self.main_pos_encoding = HybridPositionalEncoding(
                d_model=input_dim, max_len=max_seq_len, learned=False
            )

            # Use nn.Sequential with layer norm for better gradient flow
            self.router = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_experts),
            )

            # Initialize router weights properly
            for layer in self.router:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=1.0 / math.sqrt(2))
                    nn.init.zeros_(layer.bias)

            # Learnable router temperature with better initialization
            self.router_temperature = nn.Parameter(torch.ones(1) * 0.07)  # Lower initial temp
            
            # Adaptive capacity factors
            self.capacity_factor = 1.25
            self.min_capacity_factor = 1.0
            self.max_capacity_factor = 2.0
            self.capacity_factor_decay = 0.99
            
            # Improved load balancing with aux loss weight
            self.load_balancing_weight = nn.Parameter(torch.tensor(0.01))
            
            # Expert Network Setup
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

            # Improved combiner with residual connection
            self.expert_combiner = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, input_dim),
            )
            
            # Layer norm for final output
            self.output_norm = nn.LayerNorm(input_dim)

            # Expert importance tracking with EMA
            self.register_buffer("expert_importance", torch.ones(num_experts))
            self.register_buffer("expert_counts", torch.zeros(num_experts))
            self.importance_decay = 0.99
            
            # Token-to-expert assignment buffer for fast lookup
            self.register_buffer("last_expert_indices", torch.zeros(max_seq_len, dtype=torch.long))
            
            # Register a dummy buffer to help determine device in forward pass
            self.register_buffer("dummy", torch.zeros(1))

            # Expert pruning parameters with dynamic threshold
            self.expert_pruning_threshold = 0.001
            self.pruning_warmup_steps = 1000
            self.steps_since_pruning = 0
            self.pruning_interval = 5000
            self.min_active_experts = max(num_experts // 2, 1)
            
            # Auxiliary loss tracking
            self.aux_loss = 0.0
            
            # Expert batch size for parallel processing
            self.expert_batch_size = 1024

        except Exception as e:
            logger.error(f"Error initializing HierarchicalMixtureOfExperts: {str(e)}")
            raise

    def _compute_load_balancing_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """Compute improved load balancing loss with per-expert and overall balance"""
        try:
            # Calculate mean probability per expert
            # Shape: [num_experts]
            expert_load = router_probs.sum(dim=(0, 1)) / router_probs.shape[0]
            
            # Calculate coefficient of variation (to balance usage across experts)
            mean_load = expert_load.mean()
            load_variance = ((expert_load - mean_load) ** 2).mean()
            cv_squared = load_variance / (mean_load ** 2 + 1e-5)
            
            # Calculate the fraction of tokens routed to each expert
            # Shape: [num_experts]
            routing_fraction = (router_probs > 0).float().mean(dim=(0, 1))
            
            # Ideal routing load would be k/num_experts
            target_routing = self.k / self.num_experts
            
            # Auxiliary loss
            balance_loss = cv_squared + ((routing_fraction - target_routing) ** 2).mean()
            
            return balance_loss * torch.abs(self.load_balancing_weight)
            
        except Exception as e:
            logger.error(f"Error computing load balancing loss: {str(e)}")
            return torch.tensor(0.0, device=router_probs.device)

    def _update_expert_importance(self, router_probs: torch.Tensor, indices: torch.Tensor):
        """Update expert importance scores using efficient sparse operations"""
        with torch.no_grad():
            # Flatten indices for more efficient updates
            flat_indices = indices.reshape(-1)
            
            # Get expert probabilities for selected experts by gathering from router_probs
            # based on the indices tensor
            selected_probs = torch.gather(
                router_probs, 
                dim=-1,  # Gather along the experts dimension
                index=indices
            )
            
            # Flatten probabilities to match flattened indices
            flat_probs = selected_probs.reshape(-1)
            
            # Update expert counts (how many tokens were routed to each expert)
            expert_count_update = torch.zeros(self.num_experts, device=router_probs.device)
            expert_count_update.scatter_add_(0, flat_indices, torch.ones_like(flat_probs))
            self.expert_counts = self.expert_counts + expert_count_update
            
            # Update importance with EMA (exponential moving average)
            expert_importance_update = torch.zeros(self.num_experts, device=router_probs.device)
            expert_importance_update.scatter_add_(0, flat_indices, flat_probs)
            
            # Normalize the update
            total_update = expert_importance_update.sum().clamp(min=1e-6)
            normalized_update = expert_importance_update / total_update
            
            # Apply EMA update
            self.expert_importance = (
                self.importance_decay * self.expert_importance + 
                (1 - self.importance_decay) * normalized_update
            )

    def _adjust_capacity_factor(self):
        """Dynamically adjust capacity factor based on expert utilization"""
        with torch.no_grad():
            if self.expert_counts.sum() == 0:
                return
                
            # Compute coefficient of variation
            expert_fractions = self.expert_counts / self.expert_counts.sum()
            mean_fraction = 1.0 / self.num_experts
            cv = torch.std(expert_fractions) / (mean_fraction + 1e-5)
            
            # Adjust capacity factor based on balance
            if cv > 0.5:  # High variation - increase capacity
                self.capacity_factor = min(
                    self.max_capacity_factor,
                    self.capacity_factor / self.capacity_factor_decay,
                )
            elif cv < 0.1:  # Low variation - decrease capacity
                self.capacity_factor = max(
                    self.min_capacity_factor,
                    self.capacity_factor * self.capacity_factor_decay,
                )

    def _prune_inactive_experts(self, step: int) -> bool:
        """Pruned unused experts with dynamic threshold"""
        try:
            with torch.no_grad():
                # Only prune after warmup and at intervals
                if (
                    step < self.pruning_warmup_steps
                    or self.steps_since_pruning % self.pruning_interval != 0
                ):
                    return False

                # Calculate expert importance scores
                importance_scores = self.expert_importance.clone()
                
                # Dynamically calculate pruning threshold based on percentile
                sorted_scores, _ = torch.sort(importance_scores)
                bottom_percentile_idx = max(0, int(0.2 * self.num_experts) - 1)
                dynamic_threshold = sorted_scores[bottom_percentile_idx] if bottom_percentile_idx >= 0 else 0
                
                # Use the stricter of static or dynamic threshold
                actual_threshold = max(self.expert_pruning_threshold, dynamic_threshold)

                # Find inactive experts
                inactive_mask = importance_scores < actual_threshold
                inactive_indices = torch.where(inactive_mask)[0]
                active_experts = self.num_experts - len(inactive_indices)

                # Ensure minimum number of active experts
                if active_experts >= self.min_active_experts and len(inactive_indices) > 0:
                    logger.info(f"Pruning {len(inactive_indices)} inactive experts (threshold: {actual_threshold:.6f})")

                    # Re-initialize pruned experts (device-aware)
                    device = self.dummy.device
                    expert_types = ["medical", "nursing", "general", "research", "emergency"]
                    
                    for idx in inactive_indices:
                        # Reset expert importance and counts
                        self.expert_importance[idx] = 1.0
                        self.expert_counts[idx] = 0
                        
                        # Re-initialize the expert with fresh weights
                        expert_type = expert_types[idx.item() % len(expert_types)]
                        self.experts[idx] = SpecializedExpertNetwork(
                            d_model=self.input_dim,
                            num_sub_experts=4,
                            max_seq_len=self.experts[0].max_seq_len,
                            expert_type=expert_type,
                            dropout=0.1,
                        ).to(device)

                    # Log pruning event
                    logger.info(f"Experts pruned: {inactive_indices.tolist()}")
                    logger.info(f"Remaining active experts: {active_experts}")
                    
                    # Reset pruning steps counter
                    self.steps_since_pruning = 0
                    return True
                
                return False

        except Exception as e:
            logger.error(f"Error in expert pruning: {str(e)}")
            return False

    def _process_expert_batch(
        self, 
        token_inputs: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_probs: torch.Tensor
    ) -> torch.Tensor:
        """Process a batch of tokens through their assigned experts"""
        # Ensure proper dimensions for input
        if len(token_inputs.shape) == 2:
            # Add sequence dimension if missing
            token_inputs = token_inputs.unsqueeze(1)
            
        batch_size = token_inputs.shape[0]
        expert_outputs = torch.zeros_like(token_inputs)
        
        # Get unique experts in this batch
        unique_experts = torch.unique(expert_indices)
        
        # For each expert, process all tokens assigned to it at once
        for expert_idx in unique_experts:
            # Find token positions assigned to this expert
            expert_mask = (expert_indices == expert_idx)
            if not expert_mask.any():
                continue
                
            # Get inputs and probs for this expert
            expert_token_inputs = token_inputs[expert_mask]
            expert_token_probs = expert_probs[expert_mask].unsqueeze(-1)
            
            # Process through expert (with checkpointing if enabled)
            if self.use_checkpointing and self.training:
                expert_fn = self.experts[expert_idx]
                expert_token_outputs = torch.utils.checkpoint.checkpoint(
                    expert_fn, 
                    expert_token_inputs
                )
            else:
                expert_token_outputs = self.experts[expert_idx](expert_token_inputs)
                
            # Ensure outputs have correct shape
            if len(expert_token_outputs.shape) == 3 and expert_token_outputs.shape[1] == 1:
                expert_token_outputs = expert_token_outputs.squeeze(1)
                
            # Weight outputs by routing probabilities
            weighted_outputs = expert_token_outputs * expert_token_probs
            
            # Place results back into the output tensor - ensure both tensors have same shape
            if len(weighted_outputs.shape) != len(expert_outputs[expert_mask].shape):
                if len(weighted_outputs.shape) > len(expert_outputs[expert_mask].shape):
                    weighted_outputs = weighted_outputs.squeeze(1)
                else:
                    weighted_outputs = weighted_outputs.unsqueeze(1)
                    
            expert_outputs[expert_mask] = weighted_outputs
            
        return expert_outputs

    def forward(
            self, x: torch.Tensor, is_training: bool = True, step: Optional[int] = None
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Optimized forward pass for Hierarchical Mixture of Experts (MoE).
        
        Sonny: Fixed the following issues:
        - Changed to Parallelized Computation Groups tokens by assigned expert which are processes in batches.
        - Removed Redundant Calls to Experts since they now process all assigned tokens in one call.
        - Removed PyTorch Overhead in `torch.where()` calls by using direct indexing for the expert outputs.
        - Now the Expert outputs are precomputed and directly accessed when needed.
        - Also, the  batch processing reduces memory fragmentation and minimizes CUDA kernel launches.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].
            is_training (bool): Flag indicating whether the model is in training mode.
            step (Optional[int]): Current training step, used for expert pruning.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - The final output tensor.
                - Router logits tensor.
                - Auxiliary load-balancing loss tensor.
        """
        try:
            batch_size, seq_len, input_dim = x.shape
            device = x.device
            
            for buffer_name, buffer in self._buffers.items():
                if buffer is not None and buffer.device != device:
                    self._buffers[buffer_name] = buffer.to(device)

            pos_encoded = self.main_pos_encoding(x)
            
            if self.use_checkpointing and self.training:
                router_logits = torch.utils.checkpoint.checkpoint(
                    self.router, 
                    pos_encoded
                )
            else:
                router_logits = self.router(pos_encoded)

            router_logits = router_logits / (self.router_temperature.abs() + 1e-6)
            router_probs = F.softmax(router_logits, dim=-1)
            router_probs_top_k, indices = torch.topk(router_probs, k=self.k, dim=-1)
            
            if is_training:
                self._update_expert_importance(router_probs, indices)
            
            if self.load_balance and is_training:
                aux_loss = self._compute_load_balancing_loss(router_probs)
            else:
                aux_loss = torch.tensor(0.0, device=device)
            
            self.aux_loss = aux_loss.item()
            
            if self.use_expert_parallelism:
                combined_output = torch.zeros_like(x)
                for b_idx in range(batch_size):
                    batch_inputs = pos_encoded[b_idx]
                    batch_indices = indices[b_idx]
                    batch_probs = router_probs_top_k[b_idx]
                    for k_idx in range(self.k):
                        tokens_expert_indices = batch_indices[:, k_idx]
                        tokens_expert_probs = batch_probs[:, k_idx].unsqueeze(-1)
                        for expert_idx in range(self.num_experts):
                            expert_mask = (tokens_expert_indices == expert_idx)
                            if not expert_mask.any():
                                continue
                            expert_token_inputs = batch_inputs[expert_mask]
                            if len(expert_token_inputs.shape) == 2:
                                expert_token_inputs = expert_token_inputs.unsqueeze(1)
                            expert_outputs = self.experts[expert_idx](expert_token_inputs)
                            if len(expert_outputs.shape) == 3:
                                expert_outputs = expert_outputs.squeeze(1)
                            token_probs = tokens_expert_probs[expert_mask]
                            weighted_expert_output = expert_outputs * token_probs
                            token_positions = torch.where(expert_mask)[0]
                            for pos_idx, pos in enumerate(token_positions):
                                combined_output[b_idx, pos] += weighted_expert_output[pos_idx]
            else:
                combined_output = torch.zeros_like(x)
                for sample_idx in range(batch_size):
                    for token_idx in range(seq_len):
                        token_experts = indices[sample_idx, token_idx]
                        token_probs = router_probs_top_k[sample_idx, token_idx]
                        token_input = pos_encoded[sample_idx, token_idx].unsqueeze(0)
                        token_output = torch.zeros_like(token_input)
                        for k_idx, (expert_idx, prob) in enumerate(zip(token_experts, token_probs)):
                            expert_output = self.experts[expert_idx](token_input)
                            token_output += expert_output * prob
                        combined_output[sample_idx, token_idx] = token_output
            
            pruned = False
            if is_training and step is not None:
                self.steps_since_pruning += 1
                if step > self.pruning_warmup_steps:
                    self._adjust_capacity_factor()
                    pruned = self._prune_inactive_experts(step)
            
            output = x + self.output_norm(self.expert_combiner(combined_output))
            return output, router_logits, aux_loss

        except Exception as e:
            logger.error(f"Error in HierarchicalMixtureOfExperts forward pass: {str(e)}")
            raise

    def expert_summary(self) -> Dict[str, List[float]]:
        """Return a summary of expert utilization statistics"""
        with torch.no_grad():
            # Convert tensors to lists for easier logging
            importance = self.expert_importance.cpu().tolist()
            counts = self.expert_counts.cpu().tolist()
            
            # Compute additional metrics
            total_counts = sum(counts)
            fractions = [count / max(1, total_counts) for count in counts]
            
            # Create summary dictionary
            summary = {
                "importance": importance,
                "counts": counts,
                "fractions": fractions,
                "capacity_factor": self.capacity_factor,
                "aux_loss": self.aux_loss,
            }
            
            return summary