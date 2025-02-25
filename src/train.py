"""
File: train_moe.py
Author: Jeffrey Rivero
Email: jeff@check-ai.com
Created: 02/20/2025
Last Modified: 02/24/2025
Description: Main training script for the Mixture-of-Experts language model.
             Handles configuration loading, model creation, dataset preparation,
             and training execution with support for MLflow tracking and
             performance optimization.
"""

import os
import json
import argparse
import math
import logging
from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    set_seed,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import yaml
import mlflow
from torch.utils.data import DataLoader
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# Import our MoE model
from expert_model import MoELanguageModel

from utils.device import (
    setup_device,
    optimize_cuda_performance,
    get_device_map,
    log_device_info,
)
from utils.data import load_and_prepare_datasets, DataConfig, CustomDataCollator
from utils.moe_trainer import ExpertBalancingTrainer

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def setup_mlflow(config):
    """Setup MLflow tracking based on config"""
    from mlflow.system_metrics import enable_system_metrics_logging

    mlflow_config = config.get("logging", {}).get("mlflow", {})
    if not mlflow_config.get("enabled", False):
        return None

    # Set tracking URI if provided
    if tracking_uri := mlflow_config.get("tracking_uri"):
        mlflow.set_tracking_uri(tracking_uri)

    # Set experiment
    if experiment_name := mlflow_config.get("experiment_name"):
        mlflow.set_experiment(experiment_name)

    # Start a new run with optional tags
    tags = mlflow_config.get("tags", {})
    tags.update(
        {
            "model_type": "MoELanguageModel",
            "dataset": config.get("data", {}).get("dataset_name", "custom"),
        }
    )

    if run_name := config.get("run_name"):
        tags["run_name"] = run_name

    enable_system_metrics_logging()
    mlflow.start_run(run_name=run_name, tags=tags)

    return mlflow.active_run()


def load_config(config_path):
    """
    Load configuration from YAML or JSON file
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Determine file type from extension
    _, ext = os.path.splitext(config_path)

    if ext.lower() in [".yaml", ".yml"]:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    elif ext.lower() == ".json":
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        raise ValueError(
            f"Unsupported config file format: {ext}. Use .yaml, .yml, or .json"
        )

    return config


def config_to_args(config):
    """
    Convert config dictionary to an argparse Namespace
    """
    args = argparse.Namespace()

    # Set default values
    defaults = {
        # Data defaults
        "dataset_name": None,
        "dataset_config_name": None,
        "train_file": None,
        "validation_file": None,
        "max_seq_length": 1024,
        "tokenizer_name": "gpt2",
        # Model defaults
        "vocab_size": None,  # Will be determined from tokenizer
        "d_model": 512,
        "n_layers": 4,
        "num_experts": 8,
        "ffn_hidden_dim": 2048,
        "num_heads": 8,
        "k_experts": 2,
        "dropout": 0.1,
        # Training defaults
        "dataloader_num_workers": 1, 
        "output_dir": "./moe-model",
        "overwrite_output_dir": False,
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "learning_rate": 5e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 3.0,
        "max_steps": -1,
        "warmup_steps": 0,
        "logging_dir": None,
        "logging_steps": 100,
        "eval_steps": 500,
        "save_steps": 500,
        "save_total_limit": 2,
        "expert_balance_importance": 0.01,
        "seed": 42,
        "report_to": "mlflow",
        "fp16": False,
        "bf16": False,
        # New data loading options
        "preprocessing_num_workers": 4,
        "overwrite_cache": False,
        "validation_split_percentage": 5,
        "streaming": False,  # Enable for very large datasets
        "max_train_samples": None,  # Limit number of training samples
        "max_eval_samples": None,  # Limit number of validation samples
        "max_test_samples": None,  # Limit number of test samples
        # setup logging
        "use_mlflow": False,
        "use_tensorboard": True,
        "tensorboard_log_dir": "runs/",
    }

    # Set defaults
    for key, value in defaults.items():
        setattr(args, key, value)

    # Override with config values
    for key, value in config.items():
        if key == "logging":
            # Handle logging configuration
            if "mlflow" in value:
                setattr(args, "use_mlflow", value["mlflow"].get("enabled", False))
            if "tensorboard" in value:
                setattr(
                    args, "use_tensorboard", value["tensorboard"].get("enabled", True)
                )
                setattr(
                    args,
                    "tensorboard_log_dir",
                    value["tensorboard"].get("log_dir", "runs/"),
                )
        elif key in ["training", "model", "data"]:
            for sub_key, sub_value in value.items():
                setattr(args, sub_key, sub_value)
        else:
            setattr(args, key, value)

    return args


def validate_config(config: dict) -> None:
    """Validate configuration parameters"""
    required_sections = ["model", "data", "training"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
            
    # Validate model config
    model_config = config["model"]
    required_model_params = ["d_model", "n_layers", "num_experts", "num_heads"]
    for param in required_model_params:
        if param not in model_config:
            raise ValueError(f"Missing required model parameter: {param}")
            
    # Validate data config
    data_config = config["data"]
    if not data_config.get("dataset_name") and not data_config.get("train_file"):
        raise ValueError("Either dataset_name or train_file must be specified")
        
    # Validate training config
    training_config = config["training"]
    required_training_params = [
        "output_dir", "per_device_train_batch_size", 
        "learning_rate", "num_train_epochs"
    ]
    for param in required_training_params:
        if param not in training_config:
            raise ValueError(f"Missing required training parameter: {param}")


class ExpertMonitor:
    def __init__(self):
        self.expert_usage = defaultdict(int)
        self.expert_loss = defaultdict(float)
        self.domain_metrics = defaultdict(lambda: defaultdict(float))
        self.total_tokens = 0
        
    def update(self, router_logits: torch.Tensor, loss: torch.Tensor, model: nn.Module):
        with torch.no_grad():
            # Existing metrics update
            expert_assignments = router_logits.argmax(dim=-1)
            for expert_idx in range(router_logits.shape[-1]):
                self.expert_usage[expert_idx] += (expert_assignments == expert_idx).sum().item()
            self.total_tokens += router_logits.numel()
            
            # Track domain-specific metrics - fixed to access experts properly
            if hasattr(model, 'layers') and model.layers:
                for layer_idx, layer in enumerate(model.layers):
                    if len(layer) >= 3 and hasattr(layer[2], 'experts'):
                        moe_layer = layer[2]
                        for expert_idx, expert in enumerate(moe_layer.experts):
                            if hasattr(expert, 'expert_type'):
                                domain_type = expert.expert_type
                                self.domain_metrics[domain_type]["usage"] += self.expert_usage[expert_idx]
                                if hasattr(expert, "activation_patterns"):
                                    self.domain_metrics[domain_type]["activation"] = (
                                        expert.activation_patterns.mean().item()
                                    )
                                if hasattr(expert, "domain_loss"):
                                    self.domain_metrics[domain_type]["loss"] = expert.domain_loss.item()
                    
    def get_metrics(self):
        metrics = {}
        
        # Expert usage metrics
        for expert_idx, count in self.expert_usage.items():
            metrics[f"expert_{expert_idx}_usage"] = count / self.total_tokens
            
        # Domain-specific metrics
        for domain, domain_stats in self.domain_metrics.items():
            for metric_name, value in domain_stats.items():
                metrics[f"domain_{domain}_{metric_name}"] = (
                    value / self.total_tokens if metric_name == "usage" else value
                )
                
        return metrics


def cleanup_resources():
    """Clean up resources properly"""
    try:
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # End MLflow run if active
        if mlflow.active_run():
            mlflow.end_run()
            
        # Close all file handles
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)
            
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Train MoE Language Model with config file"
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to YAML or JSON configuration file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Optional run name for tracking"
    )

    cmd_args = parser.parse_args()

    # Load config
    config = load_config(cmd_args.config_file)

    # Convert config to args
    args = config_to_args(config)

    # Override with command line arguments if provided
    if cmd_args.output_dir:
        args.output_dir = cmd_args.output_dir
    if cmd_args.run_name:
        args.run_name = cmd_args.run_name

    # Save the final config for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Set random seed
    set_seed(args.seed)

    # Setup device and optimizations
    device_info = setup_device(
        no_cuda=getattr(args, "no_cuda", False),
        mixed_precision=(
            "fp16"
            if getattr(args, "fp16", False)
            else ("bf16" if getattr(args, "bf16", False) else None)
        ),
    )

    # Log device information
    log_device_info(device_info)

    # Apply CUDA optimizations if available
    if device_info.device_type == "cuda":
        optimize_cuda_performance()

    mlflow_run = None
    my_report_to = None
    if args.use_mlflow:
        mlflow_run = setup_mlflow(config)
        if mlflow_run:
            logger.info(f"MLflow tracking enabled. Run ID: {mlflow_run.info.run_id}")
            my_report_to = "mlflow"
    # if args.tensorboard:
    try:
        # Prepare training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=args.overwrite_output_dir,
            do_train=args.do_train,
            do_eval=args.do_eval,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            logging_dir=args.logging_dir,
            logging_steps=args.logging_steps,
            evaluation_strategy="steps" if args.do_eval else "no",
            eval_steps=args.eval_steps if args.do_eval else None,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            report_to=my_report_to,
            seed=args.seed,
            group_by_length=True,
            load_best_model_at_end=args.do_eval,
            metric_for_best_model="loss" if args.do_eval else None,
            # Disable mixed precision for MPS and CPU
            fp16=args.fp16 and device_info.device_type == "cuda",
            bf16=args.bf16 and device_info.device_type == "cuda",
            run_name=getattr(args, "run_name", None),
            max_grad_norm=1.0,
            logging_strategy="steps",
            # Add explicit saving strategy
            save_strategy="steps",
            # make dataloader faster
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=True,
            dataloader_prefetch_factor=2,
            dataloader_persistent_workers=True,
            # Enable torch.compile for PyTorch 2.0+
            # torch_compile=True,
            # torch_compile_backend="inductor"  # or "nvfuser" for older GPUs
        )

        # Log the config
        logger.info(f"Training with config:\n{json.dumps(vars(args), indent=2)}")

        # Prepare tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        # Ensure the tokenizer has padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Adjust vocab size based on tokenizer
        vocab_size = len(tokenizer) if args.vocab_size is None else args.vocab_size

        # Load dataset
        data_files = {}
        if args.train_file:
            data_files["train"] = args.train_file
        if args.validation_file:
            data_files["validation"] = args.validation_file

        if args.dataset_name:
            # Load from Hugging Face hub
            datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        else:
            # Load from local files
            extension = args.train_file.split(".")[-1] if args.train_file else "txt"
            datasets = load_dataset(extension, data_files=data_files)

        # Create data config
        data_config = DataConfig(
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            train_file=args.train_file,
            validation_file=args.validation_file,
            max_seq_length=args.max_seq_length,
            preprocessing_num_workers=args.preprocessing_num_workers,
            overwrite_cache=args.overwrite_cache,
            validation_split_percentage=args.validation_split_percentage,
            streaming=args.streaming,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            max_test_samples=args.max_test_samples,
        )

        # Load and prepare datasets
        datasets = load_and_prepare_datasets(tokenizer, data_config)

        # Create model
        model = MoELanguageModel(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            num_experts=args.num_experts,
            ffn_hidden_dim=args.ffn_hidden_dim,
            num_heads=args.num_heads,
            max_seq_len=args.max_seq_length,
            k_experts=args.k_experts,
            dropout=args.dropout,
        )

        # Print model size
        param_count = sum(p.numel() for p in model.parameters())
        logger.info(f"Model has {param_count:,} parameters")

        # Check if we need model parallelism
        device_map = get_device_map(param_count, device_info)

        # Move model to appropriate device
        if device_map is not None:
            logger.info("Using model parallelism across devices")
            model.parallelize(**device_map)
        else:
            model = model.to(device_info.device)

        # Enable gradient checkpointing if specified
        if getattr(args, "gradient_checkpointing", False):
            model.gradient_checkpointing_enable()

        # Update training arguments based on device info
        if device_info.device_type == "mps":
            # Disable features not supported by MPS
            training_args.fp16 = False
            training_args.bf16 = False
            training_args.tf32 = False
        elif device_info.device_type == "cpu":
            # Disable GPU-specific features
            training_args.fp16 = False
            training_args.bf16 = False
            training_args.tf32 = False
            training_args.gradient_checkpointing = False

        logger.info("Setting up CustomDataCollator")
        data_collator = CustomDataCollator(tokenizer)

        logger.info("Creating Trainer")
        trainer = ExpertBalancingTrainer(
            model=model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets.get("validation", None),
            tokenizer=tokenizer,  # Use tokenizer instead of processing_class
            expert_balance_importance=args.expert_balance_importance,
            data_collator=data_collator,
            use_mlflow=args.use_mlflow,
            expert_monitor=ExpertMonitor()
        )

        # Train model
        if args.do_train:
            logger.info("Starting train")
            train_result = trainer.train()
            logger.info("Saving Model")
            trainer.save_model()

            logger.info("Logging Metrics")
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            logger.info("Saving Sate")
            trainer.save_state()
            logger.info("Saving Tokenizer")

        # Evaluate
        if args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()

            perplexity = math.exp(metrics["eval_loss"])
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

            logger.info(f"Perplexity: {perplexity:.2f}")

        # Final analysis of expert utilization
        if args.do_eval:
            trainer.on_evaluate(args=training_args, state=None, control=None)
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()
