# RMoE Integration with expertlm

Author: Sonny Mir  
Email: sonnym@hotmail.se  
Last Modified: Feb 28, 2024

This document summarizes the integration of the Recurrent Mixture-of-Experts (RMoE) architecture from the `prototype` folder with the `expertlm` codebase.

## Integration Overview

I have created a set of adapter classes and utility functions that allow the RMoE implementation to be used as a drop-in replacement for the existing MoE implementation in the expertlm codebase. This integration enables the use of a GRU-based recurrent router for more context-aware expert routing.

## Files Created

1. **Integration Module**:
   - `expertlm/models/recurrent_moe_integration.py`: Provides adapter classes to integrate the RMoE implementation with the expertlm codebase.

2. **Training and Inference Scripts**:
   - `expertlm/train_rmoe.py`: Modified version of the training script that uses the RMoE implementation.
   - `expertlm/inference_rmoe.py`: Modified version of the inference script that uses the RMoE implementation.

3. **Configuration**:
   - `expertlm/configs/rmoe_config.yaml`: Sample configuration file for training the RMoE model.

4. **Documentation**:
   - `expertlm/README_RMOE.md`: Documentation explaining how to use the RMoE implementation with the expertlm codebase.

## Integration Details

### Adapter Classes

1. **RMoEAdapter**: Wraps the `RecurrentMoEBlock` to make it compatible with the expertlm codebase as a drop-in replacement for `HierarchicalMixtureOfExperts`.

2. **RecurrentMoELanguageModelAdapter**: Wraps the `RecurrentMoELanguageModel` to make it compatible with the expertlm codebase as a drop-in replacement for `MoELanguageModel`.

### Configuration Utilities

1. **create_rmoe_config_from_dict**: Creates a `RecurrentMoEModelConfig` from a dictionary, making it easy to convert between expertlm's configuration format and the RMoE configuration format.

### Model Registration

The adapter classes are registered in `expertlm/models/__init__.py` to make them available through the expertlm package.

## Usage

### Training

To train a model using the RMoE architecture:

```bash
python -m expertlm.train_rmoe --config expertlm/configs/rmoe_config.yaml
```

### Inference

To run inference with a trained RMoE model:

```bash
python -m expertlm.inference_rmoe --model_path ./output/rmoe --prompt "Once upon a time" --max_new_tokens 100
```

## Key Differences from Original MoE

1. **Recurrent Router**: The RMoE architecture uses a GRU-based recurrent router that maintains state across tokens, enabling more context-aware routing decisions.

2. **Expert Implementation**: The experts in the RMoE architecture use a SwiGLU-like activation function, similar to modern transformer architectures.

3. **Shared Expert Option**: The RMoE architecture includes an option to add a shared expert that processes all tokens, providing a balance between specialization and generalization.

4. **Auxiliary Loss**: The RMoE implementation includes an auxiliary load balancing loss that encourages uniform utilization of experts.

## Next Steps

1. **Testing**: Thoroughly test the integration to ensure that the RMoE implementation works correctly with the expertlm codebase.

2. **Performance Optimization**: Optimize the performance of the RMoE implementation, particularly for large models and long sequences.

3. **Hyperparameter Tuning**: Experiment with different hyperparameters to find the optimal configuration for the RMoE architecture.

4. **Evaluation**: Compare the performance of the RMoE architecture with the original MoE implementation on various tasks and datasets. 