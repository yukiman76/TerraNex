"""
File: prototype/config.py
Description: Configuration classes for Recurrent Mixture-of-Experts (RMoE) models.
             
             This module provides structured configuration classes for the RMoE
             architecture, including:
             - RecurrentRouterConfig: Configuration for the GRU-based recurrent router
             - RecurrentMoEConfig: Configuration for the RMoE layer
             - RecurrentMoEModelConfig: Configuration for the complete language model
             
             These classes use Python's dataclasses for clean, type-hinted configuration
             management. They include methods for loading configurations from pretrained
             models and converting configurations to and from dictionaries, facilitating
             integration with the expertlm codebase and other frameworks.
             
             The configuration system supports all key parameters of the RMoE architecture,
             including router type (GRU or RNN), expert dimensions, shared expert options,
             and auxiliary loss coefficients.

Author: Sonny Mir
Email: sonnym@hotmail.se
Date: Feb 28, 2024
Last Modified: Feb 28, 2024
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union


@dataclass
class RecurrentRouterConfig:
    """Configuration for the recurrent router."""
    
    input_dim: int = 768
    """Dimension of the input embeddings."""
    
    hidden_dim: int = 256
    """Hidden dimension of the router."""
    
    cell_type: str = "gru"
    """Type of recurrent cell to use. Options: 'gru', 'rnn'."""
    
    dropout: float = 0.1
    """Dropout probability."""


@dataclass
class RecurrentMoEConfig:
    """Configuration for the Recurrent Mixture-of-Experts layer."""
    
    num_experts: int = 8
    """Number of experts in the mixture."""
    
    expert_hidden_dim: int = 3072
    """Hidden dimension of each expert."""
    
    k_experts: int = 2
    """Number of experts to route each token to."""
    
    router_config: RecurrentRouterConfig = field(default_factory=RecurrentRouterConfig)
    """Configuration for the recurrent router."""
    
    add_shared_expert: bool = False
    """Whether to add a shared expert that processes all tokens."""
    
    shared_expert_weight: float = 0.5
    """Weight for the shared expert output."""
    
    moe_weight: float = 0.5
    """Weight for the mixture of experts output."""
    
    add_bias: bool = False
    """Whether to add a bias to the output."""
    
    router_aux_loss_coef: float = 0.001
    """Coefficient for the router auxiliary loss."""


@dataclass
class RecurrentMoEModelConfig:
    """Configuration for the Recurrent Mixture-of-Experts language model."""
    
    vocab_size: int = 32000
    """Size of the vocabulary."""
    
    d_model: int = 768
    """Dimension of the model embeddings."""
    
    n_layers: int = 12
    """Number of transformer layers."""
    
    num_heads: int = 12
    """Number of attention heads."""
    
    max_seq_len: int = 2048
    """Maximum sequence length."""
    
    dropout: float = 0.1
    """Dropout probability."""
    
    moe_config: RecurrentMoEConfig = field(default_factory=RecurrentMoEConfig)
    """Configuration for the MoE layers."""
    
    use_gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing to save memory."""
    
    pad_token_id: int = 0
    """ID of the padding token."""
    
    bos_token_id: int = 1
    """ID of the beginning of sequence token."""
    
    eos_token_id: int = 2
    """ID of the end of sequence token."""
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "RecurrentMoEModelConfig":
        """
        Load a configuration from a pretrained model.
        
        Args:
            model_name_or_path: Path to the pretrained model or its name.
            
        Returns:
            Model configuration.
        """
        # This is a placeholder for loading configurations from pretrained models
        # In a real implementation, this would load the config from a file
        if "small" in model_name_or_path.lower():
            return cls(
                d_model=768,
                n_layers=12,
                num_heads=12,
                moe_config=RecurrentMoEConfig(
                    num_experts=8,
                    expert_hidden_dim=2048,
                    k_experts=2,
                    router_config=RecurrentRouterConfig(
                        input_dim=768,
                        hidden_dim=256,
                        cell_type="gru"
                    )
                )
            )
        elif "base" in model_name_or_path.lower():
            return cls(
                d_model=1024,
                n_layers=24,
                num_heads=16,
                moe_config=RecurrentMoEConfig(
                    num_experts=16,
                    expert_hidden_dim=4096,
                    k_experts=2,
                    router_config=RecurrentRouterConfig(
                        input_dim=1024,
                        hidden_dim=256,
                        cell_type="gru"
                    )
                )
            )
        elif "large" in model_name_or_path.lower():
            return cls(
                d_model=1536,
                n_layers=32,
                num_heads=24,
                moe_config=RecurrentMoEConfig(
                    num_experts=32,
                    expert_hidden_dim=6144,
                    k_experts=4,
                    router_config=RecurrentRouterConfig(
                        input_dim=1536,
                        hidden_dim=512,
                        cell_type="gru"
                    )
                )
            )
        else:
            # Default configuration
            return cls()
    
    def to_dict(self) -> Dict[str, Union[int, float, bool, Dict]]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration.
        """
        router_config = {
            "input_dim": self.moe_config.router_config.input_dim,
            "hidden_dim": self.moe_config.router_config.hidden_dim,
            "cell_type": self.moe_config.router_config.cell_type,
            "dropout": self.moe_config.router_config.dropout
        }
        
        moe_config = {
            "num_experts": self.moe_config.num_experts,
            "expert_hidden_dim": self.moe_config.expert_hidden_dim,
            "k_experts": self.moe_config.k_experts,
            "router_config": router_config,
            "add_shared_expert": self.moe_config.add_shared_expert,
            "shared_expert_weight": self.moe_config.shared_expert_weight,
            "moe_weight": self.moe_config.moe_weight,
            "add_bias": self.moe_config.add_bias,
            "router_aux_loss_coef": self.moe_config.router_aux_loss_coef
        }
        
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "num_heads": self.num_heads,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "moe_config": moe_config,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Union[int, float, bool, Dict]]) -> "RecurrentMoEModelConfig":
        """
        Create a configuration from a dictionary.
        
        Args:
            config_dict: Dictionary containing the configuration.
            
        Returns:
            Model configuration.
        """
        moe_config_dict = config_dict.pop("moe_config", {})
        router_config_dict = moe_config_dict.pop("router_config", {})
        
        router_config = RecurrentRouterConfig(
            input_dim=router_config_dict.get("input_dim", 768),
            hidden_dim=router_config_dict.get("hidden_dim", 256),
            cell_type=router_config_dict.get("cell_type", "gru"),
            dropout=router_config_dict.get("dropout", 0.1)
        )
        
        moe_config = RecurrentMoEConfig(
            num_experts=moe_config_dict.get("num_experts", 8),
            expert_hidden_dim=moe_config_dict.get("expert_hidden_dim", 3072),
            k_experts=moe_config_dict.get("k_experts", 2),
            router_config=router_config,
            add_shared_expert=moe_config_dict.get("add_shared_expert", False),
            shared_expert_weight=moe_config_dict.get("shared_expert_weight", 0.5),
            moe_weight=moe_config_dict.get("moe_weight", 0.5),
            add_bias=moe_config_dict.get("add_bias", False),
            router_aux_loss_coef=moe_config_dict.get("router_aux_loss_coef", 0.001)
        )
        
        return cls(
            vocab_size=config_dict.get("vocab_size", 32000),
            d_model=config_dict.get("d_model", 768),
            n_layers=config_dict.get("n_layers", 12),
            num_heads=config_dict.get("num_heads", 12),
            max_seq_len=config_dict.get("max_seq_len", 2048),
            dropout=config_dict.get("dropout", 0.1),
            moe_config=moe_config,
            use_gradient_checkpointing=config_dict.get("use_gradient_checkpointing", False),
            pad_token_id=config_dict.get("pad_token_id", 0),
            bos_token_id=config_dict.get("bos_token_id", 1),
            eos_token_id=config_dict.get("eos_token_id", 2)
        ) 