"""
File: prototype/recurrent_moe_language_model.py
Description: Implements a language model that uses Recurrent Mixture-of-Experts (RMoE)
             layers with a recurrent router for more context-aware expert routing.
             
             This language model integrates the RMoE architecture into a transformer-based
             framework, combining the benefits of attention mechanisms with context-aware
             expert routing. The GRU-based recurrent router maintains state across tokens,
             enabling more coherent expert selection throughout the sequence.
             
             Key features:
             - Transformer architecture with multi-head self-attention
             - RMoE layers with GRU-based recurrent routing
             - Text generation capabilities with various sampling strategies
             - Gradient checkpointing for memory efficiency
             - Support for auxiliary load balancing loss
             - Compatible with the expertlm codebase through adapter classes

Author: Sonny Mir
Email: sonnym@hotmail.se
Date: Feb 28, 2024
Last Modified: Feb 28, 2024
"""

import logging
import math
from typing import Optional, Tuple, Dict, List, Union
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from prototype.recurrent_router import RecurrentRouter
from prototype.recurrent_moe import RecurrentMoEBlock, ExpertLayer


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    This implementation supports both learned and sinusoidal positional encodings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
        learned: bool = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.learned = learned
        
        if learned:
            # Learned positional embeddings
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pe, mean=0, std=0.02)
        else:
            # Sinusoidal positional encodings
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)
    
    def forward(self, x):
        """
        Add positional encoding to the input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor with positional encoding added
        """
        if self.learned:
            x = x + self.pe[:, :x.size(1), :]
        else:
            x = x + self.pe[:, :x.size(1), :].to(x.device)
        
        return self.dropout(x)


class RecurrentMoELanguageModel(nn.Module):
    """
    Language model that uses Recurrent Mixture-of-Experts (RMoE) layers.
    
    This model integrates recurrent routing into a transformer architecture,
    enabling more context-aware expert selection.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        num_heads: int = 12,
        num_experts: int = 8,
        ffn_hidden_dim: int = 3072,
        router_hidden_dim: int = 256,
        max_seq_len: int = 2048,
        k_experts: int = 2,
        dropout: float = 0.1,
        router_type: str = "gru",
        add_shared_expert: bool = False,
        shared_expert_weight: float = 0.5,
        moe_weight: float = 0.5,
        add_bias: bool = False,
        router_aux_loss_coef: float = 0.001,
        use_gradient_checkpointing: bool = False,
        pad_token_id: int = 0,
    ):
        super().__init__()
        
        logger.info("Initializing RecurrentMoELanguageModel")
        
        # Model dimensions
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.pad_token_id = pad_token_id
        self.router_aux_loss_coef = router_aux_loss_coef
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        self.dropout = nn.Dropout(dropout)
        
        # Build transformer layers with RMoE
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            logger.info(f"Creating layer {i}")
            
            # Multi-head attention
            attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            attention_norm = nn.LayerNorm(d_model)
            
            # Recurrent MoE block
            moe_block = RecurrentMoEBlock(
                num_experts=num_experts,
                input_dim=d_model,
                hidden_dim=ffn_hidden_dim,
                router_hidden_dim=router_hidden_dim,
                k=k_experts,
                router_type=router_type,
                dropout=dropout,
                add_shared_expert=add_shared_expert,
                shared_expert_weight=shared_expert_weight,
                moe_weight=moe_weight,
                add_bias=add_bias,
                router_aux_loss_coef=router_aux_loss_coef,
            )
            
            self.layers.append(nn.ModuleList([attention, attention_norm, moe_block]))
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_weights()
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _init_weights(self, module=None):
        """Initialize the weights of the model."""
        if module is None:
            modules = self.modules()
        else:
            modules = [module]
            
        for module in modules:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return next(self.parameters()).device
    
    def get_input_embeddings(self):
        """Get the input embeddings."""
        return self.token_embedding
    
    def set_input_embeddings(self, embeddings):
        """Set the input embeddings."""
        self.token_embedding = embeddings
    
    def calculate_loss(self, logits, labels):
        """
        Calculate the loss with robust handling of out-of-range labels.
        
        Args:
            logits: Model logits of shape (batch_size, seq_len, vocab_size)
            labels: Label tensor of shape (batch_size, seq_len)
            
        Returns:
            Loss tensor
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Ensure all labels are within the valid range [0, vocab_size-1]
        # Create a mask for invalid labels
        invalid_mask = (shift_labels < 0) | (shift_labels >= self.vocab_size)
        
        if invalid_mask.any():
            # Replace invalid labels with padding token
            shift_labels = shift_labels.clone()
            shift_labels[invalid_mask] = self.pad_token_id
        
        # Create a mask to ignore padding tokens
        padding_mask = (shift_labels != self.pad_token_id)
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id, reduction='none')
        flat_logits = shift_logits.view(-1, self.vocab_size)
        flat_labels = shift_labels.view(-1)
        
        # Calculate per-token loss
        per_token_loss = loss_fct(flat_logits, flat_labels)
        
        # Reshape per-token loss to match the input shape and apply padding mask
        per_token_loss = per_token_loss.view(shift_labels.size())
        masked_loss = per_token_loss * padding_mask.float()
        
        # Average the loss over non-padding tokens
        loss = masked_loss.sum() / max(padding_mask.sum().float(), 1.0)
        
        return loss

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_router_logits: bool = False,
        use_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the language model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            labels: Optional labels for computing loss
            return_router_logits: Whether to return router logits
            use_cache: Whether to use caching for generation
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size, seq_length = input_ids.shape
        device = input_ids.device
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Create causal mask for self-attention (with same dtype as attention_mask)
        mask_dtype = attention_mask.dtype
        causal_mask = torch.triu(
            torch.ones((seq_length, seq_length), device=device, dtype=mask_dtype) * float("-inf"),
            diagonal=1,
        )
        
        # Ensure key_padding_mask has the right format
        # MultiheadAttention expects key_padding_mask where True means to mask
        key_padding_mask = (1 - attention_mask).to(torch.bool)
        
        # Embed tokens and add positional encoding
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.pos_encoding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Initialize router hidden states
        router_hidden_states = [None] * len(self.layers)
        
        # Store router logits if needed
        all_router_logits = [] if return_router_logits else None
        
        # Initialize auxiliary loss
        aux_loss = torch.tensor(0.0, device=device)
        
        # Process through transformer layers
        for i, (attention, attention_norm, moe_block) in enumerate(self.layers):
            # Apply attention with residual connection
            residual = hidden_states
            hidden_states = attention_norm(hidden_states)
            
            if self.use_gradient_checkpointing and self.training:
                # Custom forward with explicit use_reentrant=False
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # Explicitly unpack the inputs
                        query, key, value, attn_mask, key_padding_mask = inputs
                        return module(
                            query, key, value, 
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask
                        )
                    return custom_forward
                
                # Use checkpoint with use_reentrant=False
                attn_output, _ = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attention),
                    hidden_states,  # query
                    hidden_states,  # key
                    hidden_states,  # value
                    causal_mask,    # attn_mask
                    key_padding_mask,  # key_padding_mask
                    use_reentrant=False,
                )
            else:
                attn_output, _ = attention(
                    hidden_states,
                    hidden_states,
                    hidden_states,
                    attn_mask=causal_mask,
                    key_padding_mask=key_padding_mask,
                )
            
            hidden_states = residual + attn_output
            
            # Apply MoE block
            if self.use_gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        hidden_state, router_state, output_logits = inputs
                        return module(
                            hidden_state,
                            router_hidden_state=router_state,
                            output_router_logits=output_logits,
                        )
                    return custom_forward
                
                # Use checkpoint with use_reentrant=False
                hidden_states, router_logits, router_hidden_states[i], layer_aux_loss = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(moe_block),
                    hidden_states,
                    router_hidden_states[i],
                    return_router_logits,
                    use_reentrant=False,
                )
            else:
                hidden_states, router_logits, router_hidden_states[i], layer_aux_loss = moe_block(
                    hidden_states,
                    router_hidden_state=router_hidden_states[i],
                    output_router_logits=return_router_logits,
                )
            
            # Accumulate auxiliary loss
            aux_loss = aux_loss + layer_aux_loss
            
            # Store router logits if needed
            if return_router_logits and router_logits is not None:
                all_router_logits.append(router_logits)
        
        # Apply final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Project to vocabulary
        logits = self.output_projection(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            print("calculate_loss")
            loss = self.calculate_loss(logits, labels)
            print(f"calculate_loss {loss}")
            
            # Add auxiliary loss
            if self.router_aux_loss_coef > 0:
                loss = loss + self.router_aux_loss_coef * aux_loss
        

        Output = namedtuple(
            "Output", ["loss", "logits", "router_logits", "capacity_loss"]
        )

        outputs = Output(
            loss=loss if labels is not None else None,
            logits=logits,
            router_logits=all_router_logits if return_router_logits else None,
            capacity_loss=aux_loss if return_router_logits else None,
        )
        
        return outputs

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
    ) -> torch.Tensor:
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
        if pad_token_id is None:
            pad_token_id = self.pad_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize router hidden states
        router_hidden_states = [None] * len(self.layers)
        
        # Initialize attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Start generation
        for _ in range(max_new_tokens):
            # Get model outputs for the current input_ids
            with torch.no_grad():
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_router_logits=False,
                )
                
                # Get the next token logits
                next_token_logits = outputs["logits"][:, -1, :]
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, k=top_k, dim=-1)
                    indices_to_remove = next_token_logits < top_k_values[:, -1].unsqueeze(-1)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Scatter sorted tensors to original indexing
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        dim=1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("inf"))
                
                # Sample or greedy decode
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1)
                
                # Append new tokens to input_ids
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((batch_size, 1), device=device)], dim=1
                )
                
                # Check for EOS
                if eos_token_id is not None and (next_tokens == eos_token_id).all():
                    break
        
        return input_ids