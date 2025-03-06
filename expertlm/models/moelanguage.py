"""
File: models/moelanguage.py
Author: Sonny Mir
Email: sonnym@hotmail.se
Created: 02/20/2024
Last Modified: 02/28/2024
Description: Implements a Mixture-of-Experts (MoE) language model architecture.
             Includes core components like ExpertLayer, PositionalEncoding,
             MixtureOfExperts, and the main MoELanguageModel class with
             generation capabilities.
             
             This module works alongside the Recurrent Mixture-of-Experts (RMoE)
             implementation, which enhances the standard MoE with GRU-based
             recurrent routing. The RecurrentMoELanguageModelAdapter class
             provides a compatible interface to use the RMoE implementation
             as a drop-in replacement for this MoELanguageModel.
             
             Key differences in the RMoE implementation:
             - Uses a GRU-based recurrent router that maintains state across tokens
             - Provides context-aware routing decisions based on sequence history
             - Includes options for shared experts alongside specialized experts
             - Implements auxiliary load balancing loss for better expert utilization
"""

import logging
import math
from typing import Optional
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from expertlm.models.hybridpositionalencoding import HybridPositionalEncoding
from expertlm.models.hierarchicalmixtureofexperts import HierarchicalMixtureOfExperts
from expertlm.utils.preformance import PerformanceMonitor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        use_gradient_checkpointing: bool = False,  # Add gradient checkpointing option
    ):
        super().__init__()

        logger.info("Starting MoELanguageModel initialization")

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

        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_encoding = HybridPositionalEncoding(self.d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)


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
                load_balance=True,
            )
            moe_norm = nn.LayerNorm(self.d_model)

            self.layers.append(
                nn.ModuleList([attention, attention_norm, moe, moe_norm])
            )

        self.output_projection = nn.Linear(self.d_model, self.vocab_size)

        self.apply(self._init_weights)

        # In MoELanguageModel.__init__, add:
        self.expert_dropout = nn.Dropout(dropout * 0.5)
        self.load_balance_weight = 0.01

        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Add memory efficient attention
        self.attention_impl = (
            "flash_attention" if torch.cuda.is_available() else "standard"
        )
        logger.info(f"Using attention implementation: {self.attention_impl}")


        self.expert_pruning_threshold = 0.001

        self.expert_cache = {}
        self.max_cache_size = 1000

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
        use_cache: bool = False,  # Add caching option
    ):
        try:
            batch_size, seq_len = input_ids.shape


            loss = torch.tensor(0.0, device=input_ids.device)

            x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)
            x = self.dropout(x)

            if attention_mask is None:
                attention_mask = torch.ones(batch_size, seq_len, device=x.device)

            # causal_mask = torch.triu(
            #     torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            # ).bool()

            padding_mask = attention_mask.float()
            key_padding_mask = (
                ~attention_mask.bool() if attention_mask is not None else None
            )
            mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device)

            for b in range(batch_size):
                for i in range(seq_len):
                    mask[b, i, i + 1 :] = float("-inf")
                    mask[b, i, :] = torch.where(
                        padding_mask[b].bool(),
                        mask[b, i, :],
                        torch.tensor(float("-inf"), device=x.device),
                    )

            router_logits_list = [] if return_router_logits else None
            capacity_loss = torch.tensor(0.0, device=x.device)


            if self.use_gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward


                for layer in self.layers:
                    self_attn, attn_norm, moe_ffn, moe_norm = layer
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self_attn),
                        x,
                        x,
                        x,
                        key_padding_mask=key_padding_mask,
                        use_reentrant=False,
                    )


            if use_cache and not self.training:
                cache_key = input_ids.cpu().numpy().tobytes()
                if cache_key in self.expert_cache:
                    return self.expert_cache[cache_key]


            for self_attn, attn_norm, moe_ffn, moe_norm in self.layers:
                # Self-attention block
                residual = x
                # key_padding_mask = (
                #     ~attention_mask.bool() if attention_mask is not None else None
                # )

                attn_output, _ = self_attn(
                    x, x, x, key_padding_mask=key_padding_mask, need_weights=False
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


            logits = self.output_projection(x)

            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                loss += capacity_loss

            Output = namedtuple(
                "Output", ["loss", "logits", "router_logits", "capacity_loss"]
            )
            outputs = Output(
                loss=loss if labels is not None else None,
                logits=logits,
                router_logits=router_logits_list,
                capacity_loss=capacity_loss,
            )

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

                generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

            return generated_ids

        except Exception as e:
            logger.error(f"Error in generate method: {str(e)}")
            raise
