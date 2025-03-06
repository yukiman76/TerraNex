"""
File: prototype/example.py
Description: Example script demonstrating how to use the Recurrent Mixture-of-Experts (RMoE) implementation.
"""

import torch
import argparse
import logging

from prototype.config import RecurrentMoEModelConfig
from prototype.recurrent_moe_language_model import RecurrentMoELanguageModel


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model(config_name="small", device="cuda"):
    """
    Create a Recurrent MoE model based on the specified configuration.
    
    Args:
        config_name: Configuration name ("small", "base", or "large")
        device: Device to place the model on
        
    Returns:
        Configured RecurrentMoELanguageModel
    """
    logger.info(f"Creating model with {config_name} configuration")
    
    # Load configuration
    config = RecurrentMoEModelConfig.from_pretrained(config_name)
    
    # Create model
    model = RecurrentMoELanguageModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        num_heads=config.num_heads,
        num_experts=config.moe_config.num_experts,
        ffn_hidden_dim=config.moe_config.expert_hidden_dim,
        router_hidden_dim=config.moe_config.router_config.hidden_dim,
        max_seq_len=config.max_seq_len,
        k_experts=config.moe_config.k_experts,
        dropout=config.dropout,
        router_type=config.moe_config.router_config.cell_type,
        add_shared_expert=config.moe_config.add_shared_expert,
        shared_expert_weight=config.moe_config.shared_expert_weight,
        moe_weight=config.moe_config.moe_weight,
        add_bias=config.moe_config.add_bias,
        router_aux_loss_coef=config.moe_config.router_aux_loss_coef,
        use_gradient_checkpointing=config.use_gradient_checkpointing,
        pad_token_id=config.pad_token_id,
    )
    
    # Move model to device
    model = model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    return model, config


def generate_text(model, input_ids, max_new_tokens=50, temperature=0.7, top_k=50, top_p=0.9):
    """
    Generate text using the model.
    
    Args:
        model: The language model
        input_ids: Input token IDs
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature
        top_k: Number of highest probability tokens to keep for top-k sampling
        top_p: Cumulative probability for nucleus sampling
        
    Returns:
        Generated token IDs
    """
    logger.info(f"Generating text with {max_new_tokens} new tokens")
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
        )
    
    return generated_ids


def forward_pass_example(model, batch_size=2, seq_len=16):
    """
    Demonstrate a forward pass through the model.
    
    Args:
        model: The language model
        batch_size: Batch size for the example
        seq_len: Sequence length for the example
        
    Returns:
        Model outputs
    """
    logger.info(f"Running forward pass with batch_size={batch_size}, seq_len={seq_len}")
    
    # Create random input IDs and labels
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=model.device)
    labels = torch.randint(0, model.vocab_size, (batch_size, seq_len), device=model.device)
    
    # Create attention mask (all tokens attended to)
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        return_router_logits=True,
    )
    
    # Log output shapes
    logger.info(f"Logits shape: {outputs['logits'].shape}")
    logger.info(f"Loss: {outputs['loss'].item()}")
    logger.info(f"Auxiliary loss: {outputs['aux_loss'].item()}")
    logger.info(f"Number of router logits: {len(outputs['router_logits'])}")
    
    return outputs


def main():
    """Main function to demonstrate the RMoE implementation."""
    parser = argparse.ArgumentParser(description="Recurrent MoE Example")
    parser.add_argument("--config", type=str, default="small", choices=["small", "base", "large"],
                        help="Model configuration (small, base, large)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for forward pass example")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length for forward pass example")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Maximum number of new tokens to generate")
    
    args = parser.parse_args()
    
    # Create model
    model, config = create_model(args.config, args.device)
    
    # Run forward pass example
    outputs = forward_pass_example(model, args.batch_size, args.seq_len)
    
    # Generate text example
    input_ids = torch.randint(0, config.vocab_size, (1, 10), device=args.device)
    generated_ids = generate_text(model, input_ids, args.max_new_tokens)
    
    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Generated shape: {generated_ids.shape}")
    
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    main() 