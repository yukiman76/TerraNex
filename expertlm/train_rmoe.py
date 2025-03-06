"""
File: train_rmoe.py
Description: Training script for the Recurrent Mixture-of-Experts (RMoE) language model.
             This script is a modified version of train.py that uses the RMoE implementation
             from the prototype folder.

             Key features:
             - Implements training for models with GRU-based recurrent routers
             - Tracks expert usage and load balancing metrics during training
             - Supports auxiliary load balancing loss for better expert utilization
             - Provides profiling capabilities for performance optimization
             - Integrates with MLflow for experiment tracking

Author: Sonny Mir
Email: sonnym@hotmail.se
Date: Feb 28, 2024
Last Modified: Feb 28, 2024
"""

import os
import json
import argparse
import math
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from transformers import (
    AutoTokenizer,
    set_seed,
    TrainingArguments,
)
import yaml
import mlflow
from torch.profiler import ProfilerActivity, tensorboard_trace_handler

# Import our RMoE model
from expertlm.models import RecurrentMoELanguageModelAdapter

from expertlm.utils.device import (
    setup_device,
    optimize_cuda_performance,
    log_device_info,
)
from expertlm.utils.data import (
    load_and_prepare_datasets,
    DataConfig,
    CustomDataCollator,
)
from expertlm.utils.moe_trainer import ExpertBalancingTrainer

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
            "model_type": "RecurrentMoELanguageModel",
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
        raise ValueError(f"Unsupported config file format: {ext}")

    # Validate config
    validate_config(config)

    return config


def config_to_args(config):
    """
    Convert configuration dictionary to command line arguments
    """
    # Training parameters
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./output"),
        overwrite_output_dir=config.get("overwrite_output_dir", True),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=8,  # config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 8),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 500),
        logging_dir=config.get("logging_dir", "./logs"),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 2),
        eval_strategy=config.get("evaluation_strategy", "steps"),
        eval_steps=config.get("eval_steps", 500),
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=config.get("greater_is_better", False),
        fp16=config.get("fp16", False),
        fp16_opt_level=config.get("fp16_opt_level", "O1"),
        dataloader_num_workers=0,  # config.get("dataloader_num_workers", 4),
        dataloader_pin_memory=False,
        report_to=config.get("report_to", ["tensorboard"]),
        run_name=config.get("run_name", "rmoe-run"),
    )

    return training_args


def validate_config(config: dict) -> None:
    """
    Validate configuration dictionary
    """
    required_keys = [
        "vocab_size",
        "d_model",
        "n_layers",
        "num_experts",
        "ffn_hidden_dim",
        "num_heads",
        "max_seq_len",
        "k_experts",
    ]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate model dimensions
    if config.get("d_model", 0) % config.get("num_heads", 1) != 0:
        raise ValueError(
            f"d_model ({config['d_model']}) must be divisible by num_heads ({config['num_heads']})"
        )

    # Validate k_experts
    if not 0 < config.get("k_experts", 0) <= config.get("num_experts", 0):
        raise ValueError(
            f"k_experts ({config['k_experts']}) must be between 1 and num_experts ({config['num_experts']})"
        )


class ExpertMonitor:
    def __init__(self, k_experts=8, num_experts=12):
        self.expert_usage = defaultdict(int)
        self.total_tokens = 0
        self.aux_losses = []
        self.k_experts = k_experts
        self.num_experts = num_experts

    def update(self, router_logits: torch.Tensor, loss: torch.Tensor, model: nn.Module):
        """
        Update expert usage statistics
        """
        if router_logits is None:
            return

        # Track auxiliary losses
        if hasattr(model, "aux_loss"):
            self.aux_losses.append(model.aux_loss)

        # Count expert usage
        for layer_logits in router_logits:
            # Get top-k experts for each token
            _, indices = torch.topk(layer_logits, k=self.k_experts, dim=-1)

            # Count usage
            for expert_idx in range(self.num_experts):
                mask = indices == expert_idx
                self.expert_usage[expert_idx] += mask.sum().item()

            # Count total tokens
            self.total_tokens += layer_logits.shape[0] * layer_logits.shape[1]

    def get_metrics(self):
        """
        Get expert usage metrics
        """
        metrics = {}

        # Expert usage percentages
        if self.total_tokens > 0:
            for expert_idx, count in self.expert_usage.items():
                metrics[f"expert_{expert_idx}_usage"] = count / self.total_tokens

        # Expert usage entropy
        if self.expert_usage:
            total = sum(self.expert_usage.values())
            if total > 0:
                probs = [count / total for count in self.expert_usage.values()]
                entropy = -sum(p * math.log(p) if p > 0 else 0 for p in probs)
                metrics["expert_entropy"] = entropy

        # Average auxiliary loss
        if self.aux_losses:
            metrics["avg_aux_loss"] = sum(self.aux_losses) / len(self.aux_losses)

        return metrics


def cleanup_resources():
    """
    Clean up resources before exiting
    """
    # Clean up CUDA resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Close MLflow run if active
    if mlflow.active_run():
        mlflow.end_run()


def main():
    """
    Main function for training the RMoE language model
    """
    parser = argparse.ArgumentParser(description="Train a Recurrent MoE Language Model")
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    args = parser.parse_args()

    try:
        # Set random seed
        set_seed(args.seed)

        # Load configuration
        config = load_config(args.config_file)

        # Add RMoE-specific parameters if not present
        if "router_type" not in config:
            config["router_type"] = "gru"
        if "router_hidden_dim" not in config:
            config["router_hidden_dim"] = 256
        if "add_shared_expert" not in config:
            config["add_shared_expert"] = False
        if "shared_expert_weight" not in config:
            config["shared_expert_weight"] = 0.5
        if "moe_weight" not in config:
            config["moe_weight"] = 0.5

        # Setup device
        device_info = setup_device(args.local_rank)

        # Log device information
        log_device_info(device_info)

        # Optimize CUDA performance if available
        if torch.cuda.is_available():
            optimize_cuda_performance()

        # Setup MLflow tracking
        mlflow_run = setup_mlflow(config)

        # Log configuration
        logger.info(f"Configuration: {json.dumps(config, indent=2)}")
        if mlflow_run:
            mlflow.log_params(config)

        # Create tokenizer
        tokenizer_name = config.get("tokenizer", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Add padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Update vocab size in config if needed
        if config.get("vocab_size", 0) != len(tokenizer):
            logger.warning(
                f"Updating vocab_size in config from {config.get('vocab_size')} to {len(tokenizer)}"
            )
            config["vocab_size"] = len(tokenizer)

        # Create data configuration
        # TODO: fixme
        data_config = DataConfig(
            dataset_name=config.get("data", {}).get("dataset_name", "wikitext"),
            dataset_config_name=config.get("data", {}).get(
                "dataset_config_name", "wikitext-103-raw-v1"
            ),
            # text_column_name=config.get("data", {}).get("text_column_name", "text"),
            max_seq_length=config.get("max_seq_len", 1024),
            preprocessing_num_workers=config.get("data", {}).get(
                "preprocessing_num_workers", 4
            ),
            overwrite_cache=config.get("data", {}).get("overwrite_cache", False),
            validation_split_percentage=config.get("data", {}).get(
                "validation_split_percentage", 5
            ),
            # block_size=config.get("data", {}).get("block_size", None),
        )

        # Load and prepare datasets
        processed_datasets = load_and_prepare_datasets(
            tokenizer=tokenizer,
            data_config=data_config,
        )

        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets.get("test", None)

        # Create model
        logger.info("Creating RecurrentMoELanguageModelAdapter")
        model = RecurrentMoELanguageModelAdapter(
            vocab_size=config["vocab_size"],
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            num_experts=config["num_experts"],
            ffn_hidden_dim=config["ffn_hidden_dim"],
            num_heads=config["num_heads"],
            max_seq_len=config["max_seq_len"],
            k_experts=config["k_experts"],
            dropout=config.get("dropout", 0.1),
            router_type=config["router_type"],
            router_hidden_dim=config["router_hidden_dim"],
            add_shared_expert=config["add_shared_expert"],
            shared_expert_weight=config["shared_expert_weight"],
            moe_weight=config["moe_weight"],
            use_gradient_checkpointing=config.get("use_gradient_checkpointing", False),
            pad_token_id=tokenizer.pad_token_id,
        )

        # Move model to device
        model = model.to(device_info.device)

        # Log model size
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model size: {num_params / 1e6:.2f}M parameters")
        if mlflow_run:
            mlflow.log_param("num_parameters", num_params)

        # Create data collator
        # TODO: fixme
        data_collator = CustomDataCollator(
            tokenizer=tokenizer,
            # mlm=False,
            # mlm_probability=0.15,
        )

        # Create training arguments
        training_args = config_to_args(config)

        # Create expert monitor
        expert_monitor = ExpertMonitor(k_experts=config["k_experts"],
                                       num_experts=config["num_experts"])

        # Create trainer
        trainer = ExpertBalancingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            expert_monitor=expert_monitor,
        )

        # Enable profiling if requested
        if args.profile:
            logger.info("Enabling profiling")

            # Create profiler
            profiler_schedule = torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1,
            )

            # TODO: fixme add MPS
            profiler_activities = [
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA if torch.cuda.is_available() else None,
            ]
            profiler_activities = [a for a in profiler_activities if a is not None]

            profiler = torch.profiler.profile(
                activities=profiler_activities,
                schedule=profiler_schedule,
                on_trace_ready=tensorboard_trace_handler(
                    os.path.join(training_args.output_dir, "profiler")
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )

            # Define profiling function
            def profile_training():
                with profiler:
                    trainer.train()

            # Create profiler callback
            from transformers.trainer_callback import TrainerCallback

            class ProfilerCallback(TrainerCallback):
                def __init__(self, profiler):
                    self.profiler = profiler

                def on_step_end(self, args, state, control, **kwargs):
                    self.profiler.step()

            # Add profiler callback
            trainer.add_callback(ProfilerCallback(profiler))

            # Train with profiling
            profile_training()
        else:
            # Train without profiling
            trainer.train()

        # Save model
        trainer.save_model(training_args.output_dir)

        # Save tokenizer
        tokenizer.save_pretrained(training_args.output_dir)

        # Save configuration
        with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        # Log metrics
        metrics = trainer.evaluate()
        logger.info(f"Evaluation metrics: {metrics}")
        if mlflow_run:
            mlflow.log_metrics(metrics)

        # Log expert usage metrics
        expert_metrics = expert_monitor.get_metrics()
        logger.info(f"Expert usage metrics: {expert_metrics}")
        if mlflow_run:
            mlflow.log_metrics(expert_metrics)

        logger.info("Training completed successfully")

    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        raise

    finally:
        # Clean up resources
        cleanup_resources()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
