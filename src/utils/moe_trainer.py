"""
File: utils/moe_trainer.py
Author: Jeffrey Rivero
Email: jeff@check-ai.com
Created: 02/20/2025
Last Modified: 02/24/2025
Description: Extension of HuggingFace Trainer class that implements expert balancing loss and 
             utilization metrics for Mixture of Experts (MoE) models. Tracks expert usage across 
             layers and provides monitoring capabilities through MLflow integration.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import Trainer
import logging
from scipy.stats import entropy

logger = logging.getLogger(__name__)


class ExpertBalancingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        # Extract custom parameters
        self.expert_balance_importance = kwargs.pop("expert_balance_importance", 0.01)
        self.use_mlflow = kwargs.pop("use_mlflow", False)

        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Initialize step tracking
        self.last_logged_step = -1

    def _calculate_balance_loss(self, router_logits):
        """Calculate expert balancing loss across all layers"""
        total_loss = 0.0
        num_experts = router_logits[0].shape[-1]

        for layer_router_logits in router_logits:
            # Get router probabilities
            router_probs = F.softmax(layer_router_logits, dim=-1)
            # Average over batch and sequence length
            expert_assignment = router_probs.mean(dim=[0, 1])
            uniform_dist = torch.ones_like(expert_assignment) / num_experts

            # KL divergence to uniform distribution
            kl_loss = F.kl_div(
                expert_assignment.log(), uniform_dist, reduction="batchmean"
            )

            # Add variance penalty
            importance_loss = torch.var(expert_assignment) * num_experts
            layer_loss = kl_loss + importance_loss
            total_loss += layer_loss

        return total_loss / len(router_logits)

    def _calculate_expert_usage(self, router_logits):
        """Calculate expert usage statistics for each layer"""
        expert_usages = []
        for layer_router_logits in router_logits:
            router_probs = F.softmax(layer_router_logits, dim=-1)
            # Average over batch and sequence length
            expert_usage = router_probs.mean(dim=[0, 1]).detach().cpu().numpy()
            expert_usages.append(expert_usage)
        return expert_usages

    def _calculate_entropy(self, probabilities):
        """Calculate Shannon entropy"""
        probabilities = np.clip(probabilities, 1e-10, 1.0)
        probabilities = probabilities / probabilities.sum()
        return entropy(probabilities)

    def _calculate_gini(self, values):
        """Calculate Gini coefficient"""
        values = np.abs(values)
        values = np.sort(values)
        n = len(values)
        if n == 0 or values.sum() == 0:
            return 0
        index = np.arange(1, n + 1)
        return ((2 * index - n - 1) * values).sum() / (n * values.sum())

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Override compute_loss to add expert balancing loss"""
        # Forward pass with router logits
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask", None),
            labels=inputs.get("labels", None),
            return_router_logits=True,
        )

        # Get main loss and router logits
        loss = outputs.loss
        router_logits = outputs.router_logits

        # Calculate expert balancing loss
        balance_loss = self._calculate_balance_loss(router_logits)
        total_loss = loss + balance_loss * self.expert_balance_importance

        # Log metrics exactly once per step
        current_step = self.state.global_step
        if current_step is not None and current_step != self.last_logged_step:
            self.last_logged_step = current_step

            # Calculate expert usage statistics
            expert_usages = self._calculate_expert_usage(router_logits)

            # Prepare metrics dictionary
            metrics = {
                "loss/cross_entropy": float(loss.detach().cpu()),
                "loss/balance": float(balance_loss.detach().cpu()),
                "loss/total": float(total_loss.detach().cpu()),
            }

            # Add expert usage metrics
            for layer_idx, layer_usage in enumerate(expert_usages):
                for expert_idx, usage in enumerate(layer_usage):
                    metrics[
                        f"train/expert_usage/layer_{layer_idx}/expert_{expert_idx}"
                    ] = float(usage)

            # Add overall balance metrics
            all_usages = np.array([usage for usage in expert_usages]).mean(axis=0)
            metrics.update(
                {
                    "train/expert_balance/overall/variance": float(np.var(all_usages)),
                    "train/expert_balance/overall/entropy": float(
                        self._calculate_entropy(all_usages)
                    ),
                    "train/expert_balance/overall/gini": float(
                        self._calculate_gini(all_usages)
                    ),
                }
            )

            # Log to both trainer and MLflow
            self._log_metrics(metrics, current_step)

        return (total_loss, outputs) if return_outputs else total_loss

    def _log_metrics(self, metrics, step):
        """Log metrics to both trainer and MLflow"""
        # Log to trainer
        for name, value in metrics.items():
            self.log({name: value})

        # Log to MLflow if enabled
        if self.use_mlflow:
            try:
                import mlflow

                if mlflow.active_run():
                    mlflow.log_metrics(metrics, step=step)
            except Exception as e:
                logger.warning(f"Error logging to MLflow: {str(e)}")

    def on_evaluate(self, args=None, state=None, control=None, **kwargs):
        """Perform final expert utilization analysis during evaluation"""
        logger.info("Performing final expert utilization analysis...")

        # Get evaluation dataset
        eval_dataset = self.eval_dataset
        if eval_dataset is None:
            logger.warning("No evaluation dataset available for expert analysis")
            return

        # Setup evaluation
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model.eval()

        # Collect expert usage statistics
        all_expert_usages = []

        with torch.no_grad():
            for batch in eval_dataloader:
                # Move batch to device
                batch = {k: v.to(model.device) for k, v in batch.items()}

                # Forward pass with router logits
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                    labels=batch.get("labels", None),
                    return_router_logits=True,
                )

                # Calculate expert usage
                expert_usages = self._calculate_expert_usage(outputs.router_logits)
                all_expert_usages.append(expert_usages)

        # Aggregate statistics
        avg_expert_usages = np.mean(
            [usage for usages in all_expert_usages for usage in usages], axis=0
        )

        # Calculate final metrics
        final_metrics = {
            "eval/expert_balance/final/variance": float(np.var(avg_expert_usages)),
            "eval/expert_balance/final/entropy": float(
                self._calculate_entropy(avg_expert_usages)
            ),
            "eval/expert_balance/final/gini": float(
                self._calculate_gini(avg_expert_usages)
            ),
        }

        # Add per-expert final usage
        for expert_idx, usage in enumerate(avg_expert_usages):
            final_metrics[f"eval/expert_usage/final/expert_{expert_idx}"] = float(usage)

        # Log final metrics
        logger.info("Final Expert Utilization Metrics:")
        for name, value in final_metrics.items():
            logger.info(f"{name}: {value:.4f}")

        # Log to MLflow if enabled
        if self.use_mlflow:
            try:
                import mlflow

                if mlflow.active_run():
                    mlflow.log_metrics(final_metrics, step=self.state.global_step)
            except Exception as e:
                logger.warning(f"Error logging final metrics to MLflow: {str(e)}")
