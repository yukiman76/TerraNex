"""
File: inference_rmoe.py
Description: Script for loading and running inference with trained Recurrent MoE models.
             This script is a modified version of inference.py that uses the RMoE implementation
             from the prototype folder.
             
             Key features:
             - Supports loading and inference with GRU-based recurrent router models
             - Provides text generation capabilities with various sampling strategies
             - Includes expert usage analysis to visualize routing decisions
             - Offers interactive mode for real-time text generation
             - Calculates generation speed and performance metrics

Author: Sonny Mir
Email: sonnym@hotmail.se
Date: Feb 28, 2024
Last Modified: Feb 28, 2024
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from expertlm.models import RecurrentMoELanguageModelAdapter, create_rmoe_config_from_dict
from expertlm.utils.device import setup_device
from safetensors.torch import load_file
import json
import logging
import os
import time


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(model_path):
    """Load the configuration file saved during training"""
    config_path = f"{model_path}/config.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logger.info(f"Loaded config: {json.dumps(config, indent=2)}")
            return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at {config_path}")


def load_model(model_path, device):
    """Load the saved RMoE model and its configuration"""
    try:
        # Load configuration
        config = load_config(model_path)
        
        # Create model instance
        logger.info("Creating RecurrentMoELanguageModelAdapter")
        model = RecurrentMoELanguageModelAdapter(
            vocab_size=config.get("vocab_size", 32000),
            d_model=config.get("d_model", 768),
            n_layers=config.get("n_layers", 12),
            num_experts=config.get("num_experts", 8),
            ffn_hidden_dim=config.get("ffn_hidden_dim", 3072),
            num_heads=config.get("num_heads", 12),
            max_seq_len=config.get("max_seq_len", 2048),
            k_experts=config.get("k_experts", 2),
            dropout=config.get("dropout", 0.1),
            router_type=config.get("router_type", "gru"),
            router_hidden_dim=config.get("router_hidden_dim", 256),
            add_shared_expert=config.get("add_shared_expert", False),
            shared_expert_weight=config.get("shared_expert_weight", 0.5),
            moe_weight=config.get("moe_weight", 0.5),
            use_gradient_checkpointing=False,
            pad_token_id=config.get("pad_token_id", 0),
        )
        
        # Move model to CPU first for loading
        model = model.to("cpu")
        
        # Check if model weights exist in safetensors format
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            logger.info(f"Loading model weights from {safetensors_path}")
            state_dict = load_file(safetensors_path)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        elif os.path.exists(pytorch_path):
            logger.info(f"Loading model weights from {pytorch_path}")
            state_dict = torch.load(pytorch_path, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        else:
            logger.warning(f"No model weights found at {safetensors_path} or {pytorch_path}")
            logger.warning("Using randomly initialized weights")
        
        # Log loading results
        if 'missing_keys' in locals() and 'unexpected_keys' in locals():
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        # Move model to target device
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()
        
        return model, config
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise


def generate_text(
    model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50, top_p=0.9
):
    """
    Generate text using the RMoE model
    
    Args:
        model: The RMoE language model
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Text prompt to start generation
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of highest probability tokens to keep for top-k sampling
        top_p: Cumulative probability for nucleus sampling
        
    Returns:
        Generated text
    """
    logger.info(f"Generating text with prompt: {prompt}")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Record start time
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Record end time
    end_time = time.time()
    
    # Calculate generation speed
    num_new_tokens = output_ids.shape[1] - input_ids.shape[1]
    generation_time = end_time - start_time
    tokens_per_second = num_new_tokens / generation_time if generation_time > 0 else 0
    
    logger.info(f"Generated {num_new_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
    
    # Decode generated text
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return generated_text


def analyze_expert_usage(model, tokenizer, prompt):
    """
    Analyze expert usage for a given prompt
    
    Args:
        model: The RMoE language model
        tokenizer: Tokenizer for encoding/decoding text
        prompt: Text prompt to analyze
        
    Returns:
        Dictionary with expert usage statistics
    """
    logger.info(f"Analyzing expert usage for prompt: {prompt}")
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    
    # Forward pass with router logits
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            return_router_logits=True,
        )
    
    # Get router logits
    router_logits = outputs["router_logits"]
    
    # Analyze expert usage
    expert_usage = {}
    
    for layer_idx, layer_logits in enumerate(router_logits):
        # Get top-k experts for each token
        probs = F.softmax(layer_logits, dim=-1)
        values, indices = torch.topk(probs, k=2, dim=-1)
        
        # Count usage per expert
        expert_counts = {}
        for expert_idx in range(layer_logits.shape[-1]):
            mask = (indices == expert_idx)
            count = mask.sum().item()
            expert_counts[f"expert_{expert_idx}"] = count
        
        # Calculate entropy
        mean_probs = probs.mean(dim=[0, 1])
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10)).item()
        
        # Store layer statistics
        expert_usage[f"layer_{layer_idx}"] = {
            "counts": expert_counts,
            "entropy": entropy,
        }
    
    return expert_usage


def main():
    """Main function for inference with RMoE model"""
    parser = argparse.ArgumentParser(description="Run inference with a trained RMoE model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt to start generation")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling parameter")
    parser.add_argument("--analyze", action="store_true", help="Analyze expert usage")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cpu, cuda)")
    args = parser.parse_args()
    
    try:
        # Setup device
        device = args.device
        if device is None:
            device, _ = setup_device(-1)
        
        # Load tokenizer
        tokenizer_path = args.model_path
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model, config = load_model(args.model_path, device)
        
        # Log model size
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model size: {num_params / 1e6:.2f}M parameters")
        
        if args.analyze:
            # Analyze expert usage
            expert_usage = analyze_expert_usage(model, tokenizer, args.prompt)
            logger.info(f"Expert usage analysis: {json.dumps(expert_usage, indent=2)}")
        
        if args.interactive:
            # Interactive mode
            logger.info("Starting interactive mode. Type 'exit' to quit.")
            while True:
                prompt = input("\nPrompt: ")
                if prompt.lower() in ["exit", "quit", "q"]:
                    break
                
                max_tokens = input("Max tokens (default: 100): ")
                max_tokens = int(max_tokens) if max_tokens.strip() else 100
                
                temp = input("Temperature (default: 0.7): ")
                temp = float(temp) if temp.strip() else 0.7
                
                # Generate text
                generated_text = generate_text(
                    model, tokenizer, prompt, max_tokens, temp, args.top_k, args.top_p
                )
                
                print("\nGenerated text:")
                print(generated_text)
        else:
            # Single generation
            generated_text = generate_text(
                model, tokenizer, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.top_p
            )
            
            print("\nGenerated text:")
            print(generated_text)
        
        logger.info("Inference completed successfully")
    
    except Exception as e:
        logger.exception(f"Error during inference: {str(e)}")
        raise


if __name__ == "__main__":
    main() 