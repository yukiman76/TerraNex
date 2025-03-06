"""
File: models/recurrent_moe_integration.py
Description: Integration module for the Recurrent Mixture-of-Experts (RMoE) architecture
             with the expertlm codebase. Provides adapter classes and utility functions
             to use the RMoE implementation from the prototype folder.

             This module implements a GRU-based recurrent router that maintains state
             across tokens in a sequence, enabling context-aware routing decisions.
             The recurrent nature of the router allows it to consider the history of
             tokens when making routing decisions, leading to more coherent expert
             selection and improved specialization.

Author: Sonny Mir
Email: sonnym@hotmail.se
Date: Feb 28, 2024
Last Modified: Feb 28, 2024

Reference: "Recurrent Mixture-of-Experts: Adaptive Memory-Augmented Language Models"
           https://arxiv.org/abs/2402.xxxxx
"""

import logging
import sys
import os
from typing import Optional, Dict

import torch
import torch.nn as nn


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "prototype"
    )
)


from expertlm.models.recurrent_moe import RecurrentMoEBlock
from expertlm.models.recurrent_moe_language_model import RecurrentMoELanguageModel
from expertlm.utils.rmoe_config import (
    RecurrentMoEModelConfig,
    RecurrentMoEConfig,
    RecurrentRouterConfig,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RMoEAdapter(nn.Module):
    """
    Adapter class that wraps the RecurrentMoEBlock to make it compatible with
    the expertlm codebase as a drop-in replacement for HierarchicalMixtureOfExperts.
    """

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
        router_type: str = "gru",
        router_hidden_dim: int = 256,
        add_shared_expert: bool = False,
        shared_expert_weight: float = 0.5,
        moe_weight: float = 0.5,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.k = k
        self.load_balance = load_balance

        # Create RecurrentMoEBlock
        self.rmoe_block = RecurrentMoEBlock(
            num_experts=num_experts,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            router_hidden_dim=router_hidden_dim,
            k=k,
            router_type=router_type,
            dropout=dropout,
            add_shared_expert=add_shared_expert,
            shared_expert_weight=shared_expert_weight,
            moe_weight=moe_weight,
            add_bias=False,
            router_aux_loss_coef=0.01 if load_balance else 0.0,
        )

        # Initialize router hidden state
        self.router_hidden_state = None

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward pass of the RMoE adapter.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, input_dim)

        Returns:
            Tuple containing:
            - output: Output tensor of shape (batch_size, seq_len, output_dim)
            - router_logits: Router logits
            - aux_loss: Auxiliary load balancing loss
        """
        # Forward pass through RecurrentMoEBlock
        output, router_logits, self.router_hidden_state, aux_loss = self.rmoe_block(
            hidden_states,
            router_hidden_state=self.router_hidden_state,
            output_router_logits=True,
        )

        return output, router_logits, aux_loss

    def reset_router_state(self):
        """Reset the router hidden state."""
        self.router_hidden_state = None


class RecurrentMoELanguageModelAdapter(nn.Module):
    """
    Adapter class that wraps the RecurrentMoELanguageModel to make it compatible
    with the expertlm codebase as a drop-in replacement for MoELanguageModel.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        num_experts: int = 8,
        ffn_hidden_dim: int = 3072,
        num_heads: int = 12,
        max_seq_len: int = 2048,
        k_experts: int = 2,
        dropout: float = 0.1,
        router_type: str = "gru",
        router_hidden_dim: int = 256,
        add_shared_expert: bool = False,
        shared_expert_weight: float = 0.5,
        moe_weight: float = 0.5,
        use_gradient_checkpointing: bool = False,
        pad_token_id: int = 0,
    ):
        super().__init__()

        logger.info("Initializing RecurrentMoELanguageModelAdapter")

        # store for latter use
        self.k_experts = k_experts
        self.num_experts = num_experts
        # Create RecurrentMoELanguageModel
        self.model = RecurrentMoELanguageModel(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            num_heads=num_heads,
            num_experts=num_experts,
            ffn_hidden_dim=ffn_hidden_dim,
            router_hidden_dim=router_hidden_dim,
            max_seq_len=max_seq_len,
            k_experts=k_experts,
            dropout=dropout,
            router_type=router_type,
            add_shared_expert=add_shared_expert,
            shared_expert_weight=shared_expert_weight,
            moe_weight=moe_weight,
            add_bias=False,
            router_aux_loss_coef=0.01,
            use_gradient_checkpointing=use_gradient_checkpointing,
            pad_token_id=pad_token_id,
        )

        logger.info(
            f"Model initialized with {sum(p.numel() for p in self.parameters())} parameters"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_router_logits: bool = False,
        use_cache: bool = False,
    ):
        """
        Forward pass of the RecurrentMoELanguageModelAdapter.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Optional labels for computing loss
            return_router_logits: Whether to return router logits
            use_cache: Whether to use caching for generation

        Returns:
            Dictionary containing model outputs
        """
        # print(f"Adapter received input_ids shape: {input_ids.shape}")
        # print(f"Adapter received attention_mask shape: {attention_mask.shape}")
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_router_logits=return_router_logits,
            use_cache=use_cache,
        )

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ):
        """
        Generate text using the language model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Cumulative probability for nucleus sampling
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End of sequence token ID

        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.model.device

    def get_input_embeddings(self):
        """Get the input embeddings."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings):
        """Set the input embeddings."""
        self.model.set_input_embeddings(embeddings)


def create_rmoe_config_from_dict(config_dict: Dict) -> RecurrentMoEModelConfig:
    """
    Create a RecurrentMoEModelConfig from a dictionary.

    Args:
        config_dict: Dictionary containing configuration parameters

    Returns:
        RecurrentMoEModelConfig instance
    """
    # Extract router config parameters
    router_config = RecurrentRouterConfig(
        input_dim=config_dict.get("d_model", 768),
        hidden_dim=config_dict.get("router_hidden_dim", 256),
        cell_type=config_dict.get("router_type", "gru"),
        dropout=config_dict.get("dropout", 0.1),
    )

    # Extract MoE config parameters
    moe_config = RecurrentMoEConfig(
        num_experts=config_dict.get("num_experts", 8),
        expert_hidden_dim=config_dict.get("ffn_hidden_dim", 3072),
        k_experts=config_dict.get("k_experts", 2),
        router_config=router_config,
        add_shared_expert=config_dict.get("add_shared_expert", False),
        shared_expert_weight=config_dict.get("shared_expert_weight", 0.5),
        moe_weight=config_dict.get("moe_weight", 0.5),
        add_bias=config_dict.get("add_bias", False),
        router_aux_loss_coef=config_dict.get("router_aux_loss_coef", 0.01),
    )

    # Create model config
    model_config = RecurrentMoEModelConfig(
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
        eos_token_id=config_dict.get("eos_token_id", 2),
    )

    return model_config
