"""
File: device.py
Author: Jeffrey Rivero
Email: jeff@check-ai.com
Created: 02/20/2025
Last Modified: 02/24/2025
Description: Manages device setup and optimization for model training.
             Handles CUDA, MPS (Apple Silicon), and CPU configurations,
             supports mixed precision training, and provides device-specific
             performance optimizations.
"""

import torch
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    device_type: str
    device: torch.device
    n_gpu: int
    device_name: str
    mixed_precision: Optional[str] = None


def setup_device(
    no_cuda: bool = False, mixed_precision: Optional[str] = None
) -> DeviceInfo:
    """
    Setup the device for training, handling CUDA, MPS, and CPU options.

    Args:
        no_cuda: If True, disable CUDA even if available
        mixed_precision: Mixed precision training type ('fp16', 'bf16', or None)

    Returns:
        DeviceInfo object containing device configuration
    """
    # First check if CUDA is available and not disabled
    if torch.cuda.is_available() and not no_cuda:
        device_type = "cuda"
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {device_name} ({n_gpu} devices available)")

        # Validate mixed precision options for CUDA
        if mixed_precision == "bf16" and not torch.cuda.is_bf16_supported():
            logger.warning("BF16 not supported on current GPU, falling back to FP32")
            mixed_precision = None

    # Then check for MPS (Apple Silicon)
    elif torch.backends.mps.is_available() and not no_cuda:
        device_type = "mps"
        device = torch.device("mps")
        n_gpu = 1  # MPS is a single device
        device_name = "Apple Silicon"
        logger.info("Using MPS (Apple Silicon) device")

        # MPS doesn't support mixed precision currently
        if mixed_precision is not None:
            logger.warning("Mixed precision not supported on MPS, falling back to FP32")
            mixed_precision = None

    # Fall back to CPU
    else:
        device_type = "cpu"
        device = torch.device("cpu")
        n_gpu = 0
        device_name = "CPU"
        logger.info("Using CPU")

        # CPU doesn't support mixed precision
        if mixed_precision is not None:
            logger.warning("Mixed precision not supported on CPU, falling back to FP32")
            mixed_precision = None

    return DeviceInfo(
        device_type=device_type,
        device=device,
        n_gpu=n_gpu,
        device_name=device_name,
        mixed_precision=mixed_precision,
    )


def get_device_map(
    model_size: int, device_info: DeviceInfo, max_memory: Optional[dict] = None
) -> Optional[dict]:
    """
    Get device map for model parallelism if needed.

    Args:
        model_size: Number of parameters in the model
        device_info: DeviceInfo object from setup_device
        max_memory: Optional dict of max memory per device

    Returns:
        Device map dict or None if not needed
    """
    if device_info.device_type == "cuda" and device_info.n_gpu > 1:
        # Only use device map for large models or when explicitly specified
        if model_size > 1e9 or max_memory is not None:  # > 1B parameters
            if max_memory is None:
                max_memory = {i: "8GiB" for i in range(device_info.n_gpu)}
            max_memory["cpu"] = "16GiB"  # Allow CPU offloading

            return {"device_map": "auto", "max_memory": max_memory}

    return None


def optimize_cuda_performance():
    """
    Apply CUDA-specific optimizations
    """
    if torch.cuda.is_available():
        # Enable TF32 for better performance on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Set benchmark mode for potentially faster training
        torch.backends.cudnn.benchmark = True

        # Disable debug mode
        torch.backends.cudnn.enabled = True


def get_world_size(device_info: DeviceInfo) -> int:
    """
    Get the world size for distributed training
    """
    if device_info.device_type == "cuda":
        return torch.cuda.device_count()
    return 1


def log_device_info(device_info: DeviceInfo):
    """
    Log detailed device information
    """
    logger.info("=" * 50)
    logger.info("Device Configuration:")
    logger.info(f"  Device Type: {device_info.device_type}")
    logger.info(f"  Device Name: {device_info.device_name}")
    logger.info(f"  Number of GPUs: {device_info.n_gpu}")
    logger.info(f"  Mixed Precision: {device_info.mixed_precision or 'None'}")

    if device_info.device_type == "cuda":
        logger.info("\nCUDA Information:")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
        logger.info(f"  cuDNN Version: {torch.backends.cudnn.version()}")
        for i in range(device_info.n_gpu):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"\n  GPU {i}: {props.name}")
            logger.info(f"    Memory Total: {props.total_memory / 1024**3:.2f} GB")
            logger.info(f"    Compute Capability: {props.major}.{props.minor}")
    logger.info("=" * 50)
