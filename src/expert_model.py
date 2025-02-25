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
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from collections import namedtuple

logger = logging.getLogger(__name__)


class ExpertLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


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


class MixtureOfExperts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        k: int = 2,
    ):
        super().__init__()

        # Ensure integer dimensions
        self.num_experts = int(num_experts)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.k = min(int(k), self.num_experts)

        # Create router
        self.router = nn.Linear(self.input_dim, self.num_experts)

        # Create expert networks
        self.experts = nn.ModuleList(
            [
                ExpertLayer(self.input_dim, self.hidden_dim, self.output_dim)
                for _ in range(self.num_experts)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape

        # Get router logits and probabilities
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.k, dim=-1)

        # Normalize probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Initialize output tensor
        output = torch.zeros_like(x)

        # we might want to Parallelize expert processing
        # Process experts
        for i in range(self.k):
            expert_indices = top_k_indices[:, :, i]
            expert_probs = top_k_probs[:, :, i].unsqueeze(-1)

            for expert_idx in range(self.num_experts):
                tokens_mask = (expert_indices == expert_idx).unsqueeze(-1)
                if not tokens_mask.any():
                    continue

                expert_input = x * tokens_mask.float()
                expert_output = self.experts[expert_idx](expert_input)
                output += expert_output * expert_probs * tokens_mask.float()
        
        return output, router_logits


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
        self.pos_encoding = PositionalEncoding(self.d_model, max_seq_len)
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

            moe = MixtureOfExperts(
                num_experts=num_experts,
                input_dim=self.d_model,
                hidden_dim=ffn_hidden_dim,
                output_dim=self.d_model,
                k=k_experts,
            )
            moe_norm = nn.LayerNorm(self.d_model)

            self.layers.append(
                nn.ModuleList([attention, attention_norm, moe, moe_norm])
            )

        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)

        # Initialize parameters
        self.apply(self._init_weights)

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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:

        batch_size, seq_len = input_ids.shape

        # Create embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_len, device=x.device)

        # Create causal mask for self-attention
        # Shape: [seq_len, seq_len]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        # Create attention mask that combines padding mask and causal mask
        # Shape: [batch_size, seq_len]
        padding_mask = attention_mask.float()

        # Create the attention mask that combines causal and padding masks
        # First create a seq_len x seq_len mask for each item in the batch
        mask = torch.zeros(batch_size, seq_len, seq_len, device=x.device)
        for b in range(batch_size):
            for i in range(seq_len):
                # Apply causal masking (can't look at future tokens)
                mask[b, i, i + 1 :] = float("-inf")
                # Apply padding mask (can't look at padding tokens)
                mask[b, i, :] = torch.where(
                    padding_mask[b].bool(),
                    mask[b, i, :],
                    torch.tensor(float("-inf"), device=x.device),
                )

        # Store router logits if requested
        router_logits_list = [] if return_router_logits else None

        # Process through transformer layers
        for self_attn, attn_norm, moe_ffn, moe_norm in self.layers:
            # Self-attention block
            residual = x
            # Handle the key padding mask separately
            key_padding_mask = (
                ~attention_mask.bool() if attention_mask is not None else None
            )

            attn_output, _ = self_attn(
                x, x, x, key_padding_mask=key_padding_mask, need_weights=False
            )
            x = attn_norm(residual + attn_output)

            # MoE FFN block
            residual = x
            moe_output, layer_router_logits = moe_ffn(x)
            x = moe_norm(residual + moe_output)

            if return_router_logits:
                router_logits_list.append(layer_router_logits)

        # Project to vocabulary
        logits = self.output_projection(x)

        # import IPython
        # IPython.embed()
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        # print(f"loss {loss}")
        # print(f"logits {logits}")
        # if return_router_logits:
        Output = namedtuple("Output", ["loss", "logits", "router_logits"])
        return Output(loss=loss, logits=logits, router_logits=router_logits_list)

        # print(f"loss {loss}")
        # print(f"logits {logits}")
        # return loss if loss is not None else logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        ) -> torch.Tensor:

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
                # Forward pass returns a tuple during training but we only need logits here
                outputs = self(context_ids, return_router_logits=False)
                
                # Check if outputs is a tuple (happened during training) or just logits
                if isinstance(outputs, tuple):
                    logits = outputs[0] if outputs[0] is not None else outputs[1]
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

            # Append new tokens
            generated_ids = torch.cat([generated_ids, next_tokens], dim=1)

        return generated_ids