"""
File: inference.py
Author: Jeffrey Rivero
Email: jeff@check-ai.com
Created: 02/20/2025
Last Modified: 02/24/2025
Description: Script for loading and running inference with trained MoE models. Handles model 
              loading from safetensors format, interactive text generation, and provides debug 
              logging. Includes optimized token generation using temperature and top-k sampling.
"""

import argparse
import torch
from transformers import AutoTokenizer
from expert_model import MoELanguageModel
from utils.device import setup_device
from safetensors.torch import load_file
import json
import logging
import os


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


def check_saved_model_old(model_path):
    """Check if model files exist and print debug info"""
    model_file = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")

    logger.info(f"Found model file at {model_file}")
    state_dict = load_file(model_file)
    logger.info(f"State dict keys: {state_dict.keys()}")

    # Get positional encoding dimensions
    if "pos_encoding.pe" in state_dict:
        pe_shape = tuple(state_dict["pos_encoding.pe"].shape)
        logger.info(f"Positional encoding shape: {pe_shape}")
        return state_dict, pe_shape
    else:
        raise ValueError("Could not find positional encoding in state dict")


def create_model_with_config(config, pe_shape):
    """Create model instance with proper initialization"""
    # Get the max sequence length from PE shape
    max_seq_len = pe_shape[0]
    d_model = pe_shape[1]

    logger.info(f"Creating model with max_seq_len={max_seq_len}, d_model={d_model}")

    # Create model instance with correct dimensions
    model = MoELanguageModel(
        vocab_size=config.get("vocab_size", 50257),  # default to GPT2 vocab size
        d_model=d_model,
        n_layers=config.get("n_layers", 4),
        num_experts=config.get("num_experts", 8),
        ffn_hidden_dim=config.get("ffn_hidden_dim", 2048),
        num_heads=config.get("num_heads", 8),
        max_seq_len=max_seq_len,
        k_experts=config.get("k_experts", 2),
        dropout=config.get("dropout", 0.1),
    )

    return model


def load_model_old(model_path, device):
    """Load the saved MoE model and its configuration"""
    try:
        # Load and check config
        config = load_config(model_path)

        # Check model file and get state dict with PE shape
        state_dict, pe_shape = check_saved_model(model_path)

        # Create model instance with proper dimensions
        model = create_model_with_config(config, pe_shape)

        # Move model to CPU first
        model = model.to("cpu")

        # Load state dict
        logger.info("Loading state dict...")
        try:
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
        except Exception as e:
            logger.error(f"Error during state dict loading: {str(e)}")
            raise

        # Move to target device
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()

        return model, config

    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        raise


def load_model_old2(model_path, device):
    """Load the saved MoE model and its configuration"""
    try:
        # Load and check config
        config = load_config(model_path)

        # Check model file and get state dict with PE shape
        state_dict, pe_shape = check_saved_model(model_path)

        # Create model instance with proper dimensions
        logger.info(
            f"Creating model with max_seq_len={pe_shape[0]}, d_model={pe_shape[1]}"
        )
        model = MoELanguageModel(
            vocab_size=config.get("vocab_size", 50257),
            d_model=pe_shape[1],  # Use d_model from PE shape
            n_layers=config.get("n_layers", 4),
            num_experts=config.get("num_experts", 8),
            ffn_hidden_dim=config.get("ffn_hidden_dim", 2048),
            num_heads=config.get("num_heads", 8),
            max_seq_len=pe_shape[0],  # Use max_seq_len from PE shape
            k_experts=config.get("k_experts", 2),
            dropout=config.get("dropout", 0.1),
        )

        # Load state dict
        logger.info("Loading state dict...")
        try:
            # Move state dict to CPU first
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
            missing_keys, unexpected_keys = model.load_state_dict(
                state_dict, strict=False
            )
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")
        except Exception as e:
            logger.error(f"Error during state dict loading: {str(e)}")
            raise

        # Move to target device
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()

        return model, config

    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        raise


def generate_text_old(
    model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_k=50
):
    """Generate text from a prompt"""
    logger.info(f"Generating text for prompt: {prompt}")

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    logger.info(f"Input shape: {input_ids.shape}")

    # Generate
    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            # Decode and clean up the generated text
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Remove the original prompt from the output
            new_text = generated_text[len(prompt) :]
            logger.info("Text generation successful")

            return new_text

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            return None


def check_saved_model(model_path):
    """Check if model files exist and print debug info"""
    model_file = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found at {model_file}")

    logger.info(f"Found model file at {model_file}")
    state_dict = load_file(model_file)
    logger.info(f"State dict keys: {state_dict.keys()}")

    # Get positional encoding dimensions
    if "pos_encoding.pe" in state_dict:
        pe_tensor = state_dict["pos_encoding.pe"]
        max_seq_len = int(pe_tensor.size(0))
        d_model = int(pe_tensor.size(1))
        logger.info(f"Positional encoding shape: ({max_seq_len}, {d_model})")
        return state_dict, (max_seq_len, d_model)
    else:
        raise ValueError("Could not find positional encoding in state dict")


def load_model_2(model_path, device):
    """Load the saved MoE model and its configuration"""
    try:
        # Load config
        config = load_config(model_path)

        # Get state dict and shapes
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))

        # Get positional encoding dimensions
        if "pos_encoding.pe" not in state_dict:
            raise ValueError("Could not find positional encoding in state dict")

        pe_tensor = state_dict["pos_encoding.pe"]

        # Add debug logging
        logger.info(f"PE tensor type: {type(pe_tensor)}")
        logger.info(f"PE tensor shape: {pe_tensor.shape}")
        logger.info(f"PE tensor dtype: {pe_tensor.dtype}")

        # Extract dimensions safely
        max_seq_len = int(pe_tensor.shape[0])
        d_model = int(pe_tensor.shape[1])

        logger.info(f"Creating model with max_seq_len={max_seq_len}, d_model={d_model}")

        # Create model with explicit int dimensions
        model = MoELanguageModel(
            vocab_size=config.get("vocab_size", 50257),
            d_model=d_model,
            n_layers=config.get("n_layers", 4),
            num_experts=config.get("num_experts", 8),
            ffn_hidden_dim=config.get("ffn_hidden_dim", 2048),
            num_heads=config.get("num_heads", 8),
            max_seq_len=max_seq_len,
            k_experts=config.get("k_experts", 2),
            dropout=config.get("dropout", 0.1),
        )

        # Debug state dict loading
        try:
            logger.info("Beginning state dict load")
            # Move to CPU and ensure float tensors
            processed_state_dict = {}
            for k, v in state_dict.items():
                logger.info(f"Processing key: {k}, shape: {v.shape}, dtype: {v.dtype}")
                processed_state_dict[k] = v.to("cpu").float()

            # Load processed state dict
            missing_keys, unexpected_keys = model.load_state_dict(
                processed_state_dict, strict=False
            )

            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

        except Exception as e:
            logger.error(f"Error during state dict loading: {str(e)}")
            raise

        # Move model to device
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()

        return model, config

    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        raise


def load_model(model_path, device):
    """Load the saved MoE model and its configuration"""
    try:
        # Load config
        config = load_config(model_path)

        # Load tokenizer to get vocab size
        tokenizer_name = config.get("tokenizer_name", "gpt2")
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        vocab_size = len(tokenizer)
        logger.info(f"Determined vocab_size: {vocab_size}")

        # Get state dict and shapes
        state_dict = load_file(os.path.join(model_path, "model.safetensors"))

        # Get positional encoding dimensions
        if "pos_encoding.pe" not in state_dict:
            raise ValueError("Could not find positional encoding in state dict")

        pe_tensor = state_dict["pos_encoding.pe"]

        # Debug logging
        logger.info(f"PE tensor type: {type(pe_tensor)}")
        logger.info(f"PE tensor shape: {pe_tensor.shape}")
        logger.info(f"PE tensor dtype: {pe_tensor.dtype}")

        # Extract dimensions safely
        max_seq_len = int(pe_tensor.shape[0])
        d_model = int(pe_tensor.shape[1])

        logger.info(
            f"Creating model with max_seq_len={max_seq_len}, d_model={d_model}, vocab_size={vocab_size}"
        )

        # Set model creation parameters with explicit types
        model_kwargs = {
            "vocab_size": vocab_size,  # Use determined vocab size
            "d_model": d_model,
            "n_layers": int(config.get("n_layers", 4)),
            "num_experts": int(config.get("num_experts", 8)),
            "ffn_hidden_dim": int(config.get("ffn_hidden_dim", 2048)),
            "num_heads": int(config.get("num_heads", 8)),
            "max_seq_len": max_seq_len,
            "k_experts": int(config.get("k_experts", 2)),
            "dropout": float(config.get("dropout", 0.1)),
        }

        logger.info(f"Model creation parameters: {model_kwargs}")

        # Create model
        try:
            logger.info("Creating model instance")
            model = MoELanguageModel(**model_kwargs)
        except Exception as e:
            logger.error(f"Error creating model instance: {str(e)}")
            raise

        # Load state dict
        try:
            logger.info("Processing state dict")
            processed_state_dict = {}
            for k, v in state_dict.items():
                logger.info(f"Processing key: {k}, shape: {v.shape}, dtype: {v.dtype}")
                # Ensure tensor is on CPU and has correct dtype
                processed_state_dict[k] = v.to("cpu", dtype=torch.float32)

            logger.info("Loading state dict into model")
            missing_keys, unexpected_keys = model.load_state_dict(
                processed_state_dict, strict=False
            )

            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

        except Exception as e:
            logger.error(f"Error during state dict loading: {str(e)}")
            raise

        # Move model to device
        logger.info(f"Moving model to device: {device}")
        model = model.to(device)
        model.eval()

        return model, config

    except Exception as e:
        logger.error(f"Error in load_model: {str(e)}")
        raise


def generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 50,
) -> torch.Tensor:
    """
    Generate text tokens autoregressively

    Args:
        input_ids: Starting token ids [batch_size, seq_len]
        max_new_tokens: Number of new tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: Number of highest probability tokens to keep for sampling
    """
    logger.info(f"Starting generation with temp={temperature}, top_k={top_k}")

    # Move to model device if needed
    if input_ids.device != self.device:
        input_ids = input_ids.to(self.device)

    # Initialize with input sequence
    generated_ids = input_ids.clone()
    min_tokens = 5  # Generate at least this many tokens before allowing EOS

    # Generate tokens
    for i in range(max_new_tokens):
        # Get context (handle long sequences)
        context_ids = (
            generated_ids[:, -1024:] if generated_ids.shape[1] > 1024 else generated_ids
        )

        # Get model output
        with torch.no_grad():
            try:
                # Forward pass
                outputs = self(context_ids)
                next_token_logits = outputs[:, -1, :].clone()

                # Log logits statistics
                logger.info(
                    f"Step {i} logits - mean: {next_token_logits.mean().item():.4f}, "
                    f"std: {next_token_logits.std().item():.4f}"
                )

                # Block EOS token for first few tokens
                if i < min_tokens:
                    next_token_logits[:, 50256] = float("-inf")

                # Apply temperature
                next_token_logits = next_token_logits / temperature

                # Apply top-k sampling
                if top_k > 0:
                    # Get top k logits and indices
                    values, indices = torch.topk(
                        next_token_logits, min(top_k, next_token_logits.size(-1))
                    )
                    # Create mask for non-top-k tokens
                    mask = torch.zeros_like(next_token_logits)
                    mask.scatter_(1, indices, 1)
                    # Apply mask
                    next_token_logits = torch.where(
                        mask.bool(),
                        next_token_logits,
                        torch.full_like(next_token_logits, float("-inf")),
                    )

                # Get probabilities
                probs = F.softmax(next_token_logits, dim=-1)

                # Log top 5 most likely tokens
                top_probs, top_tokens = torch.topk(probs, min(5, probs.size(-1)))
                for j, (token, prob) in enumerate(zip(top_tokens[0], top_probs[0])):
                    logger.info(
                        f"Top {j+1} token {token.item()}: prob={prob.item():.4f}"
                    )

                # Sample next token
                next_token = torch.multinomial(probs, num_samples=1)
                logger.info(f"Selected token {next_token.item()}")

                # Append new token
                generated_ids = torch.cat([generated_ids, next_token], dim=1)

                # Break if we generate an EOS token after min_tokens
                if next_token.item() == 50256 and i >= min_tokens:
                    logger.info("Generated EOS token, stopping generation")
                    break

            except Exception as e:
                logger.error(f"Error during generation step {i}: {str(e)}")
                raise

    return generated_ids


def generate_text(
    model, tokenizer, prompt, max_new_tokens=100, temperature=0.9, top_k=50
):
    """Generate text from a prompt"""
    logger.info(f"Generating text for prompt: {prompt}")

    # Add debug settings
    logger.info(
        f"Generation parameters: max_new_tokens={max_new_tokens}, "
        f"temperature={temperature}, top_k={top_k}"
    )

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)

    logger.info(f"Input shape: {input_ids.shape}")
    logger.info(f"Input tokens: {input_ids.tolist()}")
    logger.info(f"Input text: {tokenizer.decode(input_ids[0])}")

    # Generate
    with torch.no_grad():
        try:
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            logger.info(f"Output shape: {outputs.shape}")
            logger.info(f"Output tokens: {outputs.tolist()}")

            # Get only the new tokens
            new_tokens = outputs[:, input_ids.shape[1] :]
            logger.info(f"New tokens: {new_tokens.tolist()}")

            # Decode only the new tokens
            generated_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            logger.info(f"Generated text: '{generated_text}'")

            # Return message if no meaningful text generated
            if not generated_text.strip():
                return "I need to be trained further to provide meaningful responses."

            return generated_text

        except Exception as e:
            logger.error(f"Error during text generation: {str(e)}")
            logger.exception(e)
            return None


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained MoE model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the saved model directory",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)",
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling parameter (default: 50)"
    )
    args = parser.parse_args()

    # Setup device
    device_info = setup_device()
    device = device_info.device
    logger.info(f"Using device: {device}")

    try:
        # Load the model and config
        model, config = load_model(args.model_path, device)
        logger.info("Model loaded successfully")

        # Load the tokenizer
        tokenizer_name = config.get("tokenizer_name", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        logger.info(f"Loaded tokenizer: {tokenizer_name}")

        # Ensure the tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Interactive loop
        print("\nEnter your prompts (type 'quit' to exit):")
        while True:
            try:
                prompt = input("\nPrompt> ")

                if prompt.lower() in ["quit", "exit", "q"]:
                    break

                if not prompt.strip():
                    continue

                generated_text = generate_text(
                    model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                )

                if generated_text:
                    print("\nGenerated text:")
                    print("-" * 40)
                    print(generated_text)
                    print("-" * 40)
                else:
                    print("No text was generated.")

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing prompt: {str(e)}")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return


if __name__ == "__main__":
    main()
