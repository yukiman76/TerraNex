"""
File: expert_model.py
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
from typing import Optional, Tuple, List, Union, Dict, Any, Callable
from collections import namedtuple
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            "forward_pass_times": [],
            "router_times": [],
            "expert_processing_times": [],
            "memory_usage": [],
            "expert_utilization": []
        }
        
    def update(self, metric_name: str, value: Any):
        self.metrics[metric_name].append(value)
        
    def get_summary(self) -> Dict[str, float]:
        return {
            name: sum(values) / len(values) 
            for name, values in self.metrics.items()
            if values
        }


class ExpertLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
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
        load_balance: bool = True
    ):
        super().__init__()
        
        # Enhanced input validation
        if not all(isinstance(x, int) for x in [num_experts, input_dim, hidden_dim, output_dim, max_seq_len]):
            raise TypeError("Dimensions must be integers")
        if not all(x > 0 for x in [num_experts, input_dim, hidden_dim, output_dim, max_seq_len]):
            raise ValueError("All dimensions must be positive")
        if not 0 < k <= num_experts:
            raise ValueError(f"k ({k}) must be between 1 and num_experts ({num_experts})")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")
            
        self.load_balance = load_balance
        self.num_experts = num_experts
        self.k = k
        
        # Initialize components with proper error handling
        try:
            self.main_pos_encoding = HybridPositionalEncoding(
                d_model=input_dim,
                max_len=max_seq_len,
                learned=False
            )
            
            self.router = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),  # Add dropout for regularization
                nn.Linear(hidden_dim, num_experts)
            )
            
            # Initialize router weights properly
            for layer in self.router:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
                    
            self.router_temperature = nn.Parameter(torch.ones(1))
            self.capacity_factor = 1.25
            
            expert_types = ["medical", "nursing", "general", "research", "emergency"]
            self.experts = nn.ModuleList([
                SpecializedExpertNetwork(
                    d_model=input_dim,
                    num_sub_experts=4,
                    max_seq_len=max_seq_len,
                    expert_type=expert_types[i % len(expert_types)],
                    dropout=dropout
                )
                for i in range(num_experts)
            ])
            
            self.expert_combiner = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, input_dim),
                nn.LayerNorm(input_dim)
            )
            
            # Add expert pruning
            self.register_buffer(
                "expert_importance", 
                torch.ones(num_experts)
            )
            
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
            capacity_loss = torch.mean(expert_load ** 2) * self.num_experts
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
                0.9 * self.expert_importance + 
                0.1 * expert_usage / expert_usage.sum()
            )
            
    def _adjust_capacity_factor(self, expert_counts: torch.Tensor):
        """Dynamically adjust capacity factor based on expert utilization"""
        max_usage = expert_counts.max().item()
        target_usage = expert_counts.sum().item() / self.num_experts
        
        if max_usage > target_usage * 1.5:
            self.capacity_factor = min(
                self.max_capacity_factor,
                self.capacity_factor / self.capacity_factor_decay
            )
        elif max_usage < target_usage:
            self.capacity_factor = max(
                self.min_capacity_factor,
                self.capacity_factor * self.capacity_factor_decay
            )
        
    def _prune_inactive_experts(self) -> None:
        """Prune experts that are consistently unused"""
        try:
            with torch.no_grad():
                # Only prune after warmup and at intervals
                if (self.steps_since_pruning < self.pruning_warmup_steps or
                    self.steps_since_pruning % self.pruning_interval != 0):
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
        self, 
        x: torch.Tensor, 
        is_training: bool = True,
        step: Optional[int] = None
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
                    token_input = pos_encoded[sample_idx, token_idx].unsqueeze(0).unsqueeze(0)
                    
                    # Process through selected experts with their weights
                    expert_outputs = []
                    for expert_idx, prob in zip(token_experts, token_probs):
                        expert_output = self.experts[expert_idx](token_input).squeeze(0)
                        expert_outputs.append(expert_output * prob)
                        
                        # Update expert importance (during training)
                        if is_training:
                            self.expert_importance[expert_idx] = (
                                0.99 * self.expert_importance[expert_idx] + 
                                0.01 * prob.detach()
                            )
                    
                    # Combine expert outputs
                    token_output = sum(expert_outputs)
                    combined_output[sample_idx, token_idx] = token_output
            
            # Calculate capacity loss if load balancing
            capacity_loss = self._compute_capacity(router_probs) if self.load_balance else torch.tensor(0.0, device=device)
            
            # Apply pruning during training at specified intervals
            if is_training and step is not None:
                self.steps_since_pruning += 1
                if (
                    step > self.pruning_warmup_steps and 
                    self.steps_since_pruning >= self.pruning_interval
                ):
                    pruned = self.prune_experts()
                    if pruned:
                        self.steps_since_pruning = 0
            
            # Final output combination
            output = self.expert_combiner(combined_output)
            return output, router_logits, capacity_loss
            
        except Exception as e:
            logger.error(f"Error in HierarchicalMixtureOfExperts forward pass: {str(e)}")
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
                
            logger.info(f"Pruning {len(prune_indices)} experts: {prune_indices.tolist()}")
            
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
                    dropout=0.1  # Default dropout
                ).to(device)
                
            return True  # Pruning occurred
            
        except Exception as e:
            logger.error(f"Error during expert pruning: {str(e)}")
            return False


class SpecializedExpertNetwork(nn.Module):
    def __init__(
        self, 
        d_model: int,
        num_sub_experts: int = 4,
        max_seq_len: int = 1024,
        expert_type: str = "general",
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Enhanced input validation
        if not isinstance(d_model, int) or d_model <= 0:
            raise ValueError(f"d_model must be a positive integer, got {d_model}")
        if not isinstance(num_sub_experts, int) or num_sub_experts <= 0:
            raise ValueError(f"num_sub_experts must be a positive integer, got {num_sub_experts}")
        if expert_type not in ["medical", "nursing", "general", "research", "emergency"]:
            raise ValueError(f"Invalid expert_type: {expert_type}")
            
        try:
            # Initialize components with proper error handling
            self.domain_attention = DomainSpecificAttention(
                d_model=d_model,
                num_heads=max(1, d_model // 64),  # Dynamic head calculation
                dropout=dropout
            )
            
            self.pos_encoding = HybridPositionalEncoding(
                d_model=d_model,
                max_len=max_seq_len,
                learned=True
            )
            
            # Enhanced input projection with skip connection
            self.input_proj = nn.Sequential(
                nn.LayerNorm(d_model),  # Normalize first
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model)
            )
            
            # Initialize router with proper scaling
            self.sub_router = nn.Linear(d_model, num_sub_experts)
            self.router_temperature = nn.Parameter(torch.ones(1) / math.sqrt(d_model))
            
            # Initialize sub-experts with domain-specific dropout
            dropout_scale = 0.8 if expert_type != "general" else 1.0
            self.sub_experts = nn.ModuleList([
                ExpertLayer(
                    d_model, 
                    d_model * 4,
                    d_model,
                    dropout=dropout * dropout_scale
                )
                for _ in range(num_sub_experts)
            ])
            
            self.expert_type = expert_type
            self.layer_norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            
            # Enhanced gating mechanism
            self.gate = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid()
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
                "emergency": {"keyword_weight": 1.4, "attention_weight": 1.3}
            }
            
        except Exception as e:
            logger.error(f"Error initializing SpecializedExpertNetwork: {str(e)}")
            raise
            
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
                
    def _update_specialization_metrics(self, x: torch.Tensor, expert_weights: torch.Tensor):
        """Track expert specialization patterns"""
        with torch.no_grad():
            # Update input statistics
            self.input_statistics = (
                0.9 * self.input_statistics + 
                0.1 * x.mean(dim=(0, 1))
            )
            
            # Update activation patterns
            expert_activations = (expert_weights > self.specialization_threshold).float()
            self.activation_patterns = (
                0.9 * self.activation_patterns + 
                0.1 * expert_activations.mean(dim=(0, 1))
            )
            
    def _compute_domain_statistics(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute domain-specific statistics for input"""
        with torch.no_grad():
            stats = {
                "mean_activation": x.mean(dim=1),
                "pattern_strength": self.activation_patterns.clone(),
                "attention_distribution": self.domain_attention.get_attention_weights(),
                "expert_utilization": torch.zeros(len(self.sub_experts), device=x.device)
            }
            return stats
            
    def _calculate_domain_specific_loss(self, stats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculate domain-specific loss based on expert type"""
        domain_config = self.domain_statistics[self.expert_type]
        
        # Calculate weighted loss components
        keyword_loss = (
            stats["mean_activation"].std() * domain_config["keyword_weight"]
        )
        attention_loss = (
            self.entropy(stats["attention_distribution"]).mean() * 
            domain_config["attention_weight"]
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
                raise ValueError(f"Expected input dimension {self.input_proj[1].in_features}, got {x.size(-1)}")
                
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


class DomainSpecificAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
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


class MoELanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_layers: int = 4,
        num_experts: int = 8,
        ffn_hidden_dim: int = 2048,
        num_heads: int = 8,
        max_seq_len: int = 1024,
        k_experts: int = 2,
        dropout: float = 0.1,
        use_gradient_checkpointing: bool = False  # Add gradient checkpointing option
    ):
        super().__init__()

        logger.info("Starting MoELanguageModel initialization")

        # Convert all dimensions to integers explicitly
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.num_heads = int(num_heads)
        max_seq_len = int(max_seq_len)
        n_layers = int(n_layers)
        num_experts = int(num_experts)
        ffn_hidden_dim = int(ffn_hidden_dim)
        k_experts = int(k_experts)

        logger.info(
            f"Initialized dimensions: d_model={self.d_model}, vocab_size={self.vocab_size}, "
            f"num_heads={self.num_heads}, max_seq_len={max_seq_len}"
        )

        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = HybridPositionalEncoding(self.d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Build transformer layers with MoE
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            logger.info(f"Creating layer {i}")
            attention = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=self.num_heads,
                dropout=dropout,
                batch_first=True,
            )
            attention_norm = nn.LayerNorm(self.d_model)

            moe = HierarchicalMixtureOfExperts(
                num_experts=num_experts,
                input_dim=self.d_model,
                hidden_dim=ffn_hidden_dim,
                output_dim=self.d_model,
                max_seq_len=max_seq_len,
                k=k_experts,
                dropout=dropout,
                load_balance=True
            )
            moe_norm = nn.LayerNorm(self.d_model)

            self.layers.append(
                nn.ModuleList([attention, attention_norm, moe, moe_norm])
            )

        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)

        # Initialize parameters
        self.apply(self._init_weights)

        # In MoELanguageModel.__init__, add:
        self.expert_dropout = nn.Dropout(dropout * 0.5)
        self.load_balance_weight = 0.01

        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Add memory efficient attention
        self.attention_impl = "flash_attention" if torch.cuda.is_available() else "standard"
        logger.info(f"Using attention implementation: {self.attention_impl}")
        
        # Add expert pruning threshold
        self.expert_pruning_threshold = 0.001
        
        # Add expert cache for inference
        self.expert_cache = {}
        self.max_cache_size = 1000

        # Add performance monitor
        self.performance_monitor = PerformanceMonitor()

        logger.info("MoELanguageModel initialization complete")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_router_logits: bool = False,
        use_cache: bool = False  # Add caching option
    ):
        try:
            batch_size, seq_len = input_ids.shape
            
            # Initialize loss
            loss = torch.tensor(0.0, device=input_ids.device)
            
            # Create embeddings
            x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            x = self.dropout(x)

            # Handle attention mask
            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_len, device=x.device)

            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()

            # Create attention mask that combines padding and causal masks
            padding_mask = attention_mask.float()
            mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
            
            for b in range(batch_size):
                for i in range(seq_len):
                    mask[b, i, i + 1:] = float("-inf")
                    mask[b, i, :] = torch.where(
                        padding_mask[b].bool(),
                        mask[b, i, :],
                        torch.tensor(float("-inf"), device=x.device),
                    )

            router_logits_list = [] if return_router_logits else None
            capacity_loss = torch.tensor(0.0, device=x.device)

            # Use gradient checkpointing if enabled
            if self.use_gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                # Apply gradient checkpointing to attention and MoE layers
                for layer in self.layers:
                    self_attn, attn_norm, moe_ffn, moe_norm = layer
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self_attn),
                        x, x, x,
                        key_padding_mask=key_padding_mask,
                        use_reentrant=False
                    )
            
            # Use expert caching during inference
            if use_cache and not self.training:
                cache_key = input_ids.cpu().numpy().tobytes()
                if cache_key in self.expert_cache:
                    return self.expert_cache[cache_key]
                    
            # Process through transformer layers
            for self_attn, attn_norm, moe_ffn, moe_norm in self.layers:
                # Self-attention block
                residual = x
                key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None

                attn_output, _ = self_attn(
                    x, x, x,
                    key_padding_mask=key_padding_mask,
                    need_weights=False
                )
                x = attn_norm(residual + attn_output)

                # MoE FFN block
                residual = x
                moe_output, layer_router_logits, layer_capacity_loss = moe_ffn(x)
                moe_output = self.expert_dropout(moe_output)
                x = moe_norm(residual + moe_output)

                if return_router_logits:
                    router_logits_list.append(layer_router_logits)

                if layer_capacity_loss is not None:
                    capacity_loss += layer_capacity_loss * self.load_balance_weight

            # Project to vocabulary
            logits = self.output_projection(x)

            # Calculate loss if labels provided
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                loss += capacity_loss

            Output = namedtuple("Output", ["loss", "logits", "router_logits", "capacity_loss"])
            outputs = Output(
                loss=loss if labels is not None else None,
                logits=logits,
                router_logits=router_logits_list,
                capacity_loss=capacity_loss
            )

            # Update expert cache
            if use_cache and not self.training:
                if len(self.expert_cache) > self.max_cache_size:
                    # Remove oldest entry
                    self.expert_cache.pop(next(iter(self.expert_cache)))
                self.expert_cache[cache_key] = outputs

            return outputs

        except Exception as e:
            logger.error(f"Error in MoELanguageModel forward pass: {str(e)}")
            raise

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        try:
            if not isinstance(input_ids, torch.Tensor):
                raise TypeError("input_ids must be a torch.Tensor")
            if max_new_tokens <= 0:
                raise ValueError("max_new_tokens must be positive")
            if temperature <= 0:
                raise ValueError("temperature must be positive")
            if top_k < 0:
                raise ValueError("top_k must be non-negative")

            batch_size = input_ids.shape[0]
            generated_ids = input_ids.clone()

            for _ in range(max_new_tokens):
                # Use only the last 1024 tokens if sequence is too long
                if generated_ids.shape[1] > 1024:
                    context_ids = generated_ids[:, -1024:]
                else:
                    context_ids = generated_ids

                # Get next token probabilities
                with torch.no_grad():
                    try:
                        outputs = self(context_ids, return_router_logits=False)
                        
                        if isinstance(outputs, tuple):
                            logits = outputs[1]  # Get logits from the Output namedtuple
                        else:
                            logits = outputs
                            
                        next_token_logits = logits[:, -1, :] / temperature

                        # Apply top-k sampling
                        if top_k > 0:
                            indices_to_remove = (
                                next_token_logits
                                < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                            )
                            next_token_logits[indices_to_remove] = float("-inf")

                        # Sample from the filtered distribution
                        probs = F.softmax(next_token_logits, dim=-1)
                        next_tokens = torch.multinomial(probs, num_samples=1)

                    except RuntimeError as e:
                        logger.error(f"Error during token generation: {str(e)}")
                        raise

                # Append new tokens
                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

            return generated_ids
            
        except Exception as e:
            logger.error(f"Error in generate method: {str(e)}")
            raise