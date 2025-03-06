"""
File: test_moelanguagemodel_performance.py
Description: Performance testing script for MoELanguageModel module
             Uses PyTorch profiler to measure detailed performance metrics
             with configuration loaded from YAML
"""

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
import numpy as np
import argparse
import gc
import yaml
import json
import os
from collections import defaultdict

# Import the MoE model
from expertlm.models.moelanguage import MoELanguageModel

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

def profile_model(model, input_ids, attention_mask, num_runs=10, warmup_runs=5, use_cuda=False):
    """Profile model using PyTorch profiler"""
    # Warmup runs to eliminate initialization overhead
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Configure profiler with appropriate activities
    activities = [ProfilerActivity.CPU]
    if use_cuda:
        activities.append(ProfilerActivity.CUDA)
    
    # Run profiler
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for _ in range(num_runs):
            with record_function("model_inference"):
                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)
                if use_cuda:
                    torch.cuda.synchronize()
    
    return prof

def create_model_from_config(config, vocab_size=50257):
    """Create MoELanguageModel from config dict"""
    model_config = config.get("model", {})
    
    model = MoELanguageModel(
        vocab_size=vocab_size,
        d_model=model_config.get("d_model", 384),
        n_layers=model_config.get("n_layers", 4),
        num_experts=model_config.get("num_experts", 8),
        ffn_hidden_dim=model_config.get("ffn_hidden_dim", 1536),
        num_heads=model_config.get("num_heads", 6),
        max_seq_len=config.get("data", {}).get("max_seq_length", 1024),
        k_experts=model_config.get("k_experts", 2),
        dropout=model_config.get("dropout", 0.1)
    )
    
    return model

def benchmark_model_with_config(config, device, batch_sizes=None, seq_lengths=None, save_results=True):
    """Benchmark MoELanguageModel with configuration"""
    if batch_sizes is None:
        batch_sizes = [1, 4, 16]
    
    if seq_lengths is None:
        seq_lengths = [128, 512, 1024]
    
    results = defaultdict(list)
    configs = []
    
    # Create the model based on the config
    model = create_model_from_config(config)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    for batch_size in batch_sizes:
        for seq_len in seq_lengths:
            configs.append((batch_size, seq_len))
            
            print(f"\nTesting configuration: batch_size={batch_size}, seq_len={seq_len}")
            
            # Create random input tensors (simulating tokenized input)
            input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
            attention_mask = torch.ones(batch_size, seq_len, device=device)
            
            # Clear cache before profiling
            torch.cuda.empty_cache() if device.type == 'cuda' else None
            gc.collect()
            
            # Profile model
            prof = profile_model(
                model, 
                input_ids, 
                attention_mask,
                use_cuda=(device.type == 'cuda')
            )
            
            # Extract metrics
            avg_time = prof.key_averages().total_average().cpu_time_total / 1000  # ms
            memory = prof.key_averages().total_average().self_cpu_memory_usage / (1024 * 1024)  # MB
            
            if device.type == 'cuda':
                cuda_time = prof.key_averages().total_average().cuda_time_total / 1000  # ms
                cuda_memory = prof.key_averages().total_average().self_cuda_memory_usage / (1024 * 1024)  # MB
                avg_time = cuda_time
                memory = cuda_memory
            
            # Store results
            results['avg_time'].append(avg_time)
            results['memory'].append(memory)
            
            # Print profiler summary
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            
            # Export chrome trace if requested
            if save_results:
                prof.export_chrome_trace(f"moe_bs{batch_size}_seq{seq_len}.json")
    
    if save_results:
        # Plot results
        plot_performance(configs, results, 'avg_time', 'Execution Time (ms)', 'moe_execution_time.png')
        plot_performance(configs, results, 'memory', 'Memory Usage (MB)', 'moe_memory_usage.png')
    
    return results, configs

def plot_performance(configs, results, metric, ylabel, filename):
    """Plot performance results for a given metric"""
    plt.figure(figsize=(12, 6))
    
    x = range(len(configs))
    
    plt.bar(x, results[metric], width=0.6)
    
    plt.xlabel('Configuration (batch_size, seq_len)')
    plt.ylabel(ylabel)
    plt.title(f'{ylabel} for MoELanguageModel')
    plt.xticks(x, [f'({b},{s})' for b, s in configs], rotation=45)
    plt.tight_layout()
    
    plt.savefig(filename)
    plt.close()

def analyze_router_performance(model, input_ids, attention_mask, device, top_n=10):
    """Analyze the performance of router components specifically"""
    print("\nAnalyzing router performance:")
    
    # Configure profiler
    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
        profile_memory=True
    ) as prof:
        with record_function("model_inference"):
            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            if device.type == 'cuda':
                torch.cuda.synchronize()
    
    # Print table of most expensive operators
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total" if device.type == 'cpu' else "self_cuda_time_total", 
        row_limit=top_n
    ))
    
    # Try to find router-related operations in the profiler results
    print("\nRouter-related operations:")
    for event in prof.key_averages():
        if "router" in event.key.lower() or "expert" in event.key.lower() or "gate" in event.key.lower():
            print(f"{event.key}: {event.cpu_time_total/1000:.2f}ms")
    
    return prof

def scaling_test(config, device, save_results=True):
    """Test how model performance scales with increasing sequence length and experts"""
    # Fixed batch size
    batch_size = 4
    
    # Vary sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048]
    # Vary number of experts (must modify config)
    expert_counts = [2, 4, 8, 16]
    
    results = {
        'seq_length': {
            'time': [],
            'memory': []
        },
        'num_experts': {
            'time': [],
            'memory': []
        }
    }
    
    # Test sequence length scaling
    print("\nTesting sequence length scaling:")
    base_model = create_model_from_config(config)
    base_model.to(device)
    base_model.eval()
    
    for seq_len in seq_lengths:
        print(f"Testing sequence length: {seq_len}")
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Clear cache
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        gc.collect()
        
        # Profile
        prof = profile_model(
            base_model, 
            input_ids, 
            attention_mask,
            num_runs=5,  # Reduce for longer sequences
            use_cuda=(device.type == 'cuda')
        )
        
        # Extract metrics
        avg_time = prof.key_averages().total_average().cpu_time_total / 1000
        memory = prof.key_averages().total_average().self_cpu_memory_usage / (1024 * 1024)
        
        if device.type == 'cuda':
            avg_time = prof.key_averages().total_average().cuda_time_total / 1000
            memory = prof.key_averages().total_average().self_cuda_memory_usage / (1024 * 1024)
        
        results['seq_length']['time'].append(avg_time)
        results['seq_length']['memory'].append(memory)
        
        print(f"Time: {avg_time:.4f} ms, Memory: {memory:.2f} MB")
    
    # Free up memory
    del base_model
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    gc.collect()
    
    # Test expert count scaling
    print("\nTesting expert count scaling:")
    for num_experts in expert_counts:
        print(f"Testing with {num_experts} experts")
        
        # Modify config
        expert_config = config.copy()
        expert_config['model'] = config.get('model', {}).copy()
        expert_config['model']['num_experts'] = num_experts
        
        # Create model with the number of experts
        model = create_model_from_config(expert_config)
        model.to(device)
        model.eval()
        
        # Fixed sequence length for expert scaling test
        seq_len = 512
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        # Clear cache
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        gc.collect()
        
        # Profile
        prof = profile_model(
            model, 
            input_ids, 
            attention_mask,
            use_cuda=(device.type == 'cuda')
        )
        
        # Extract metrics
        avg_time = prof.key_averages().total_average().cpu_time_total / 1000
        memory = prof.key_averages().total_average().self_cpu_memory_usage / (1024 * 1024)
        
        if device.type == 'cuda':
            avg_time = prof.key_averages().total_average().cuda_time_total / 1000
            memory = prof.key_averages().total_average().self_cuda_memory_usage / (1024 * 1024)
        
        results['num_experts']['time'].append(avg_time)
        results['num_experts']['memory'].append(memory)
        
        print(f"Time: {avg_time:.4f} ms, Memory: {memory:.2f} MB")
        
        # Free up memory
        del model
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        gc.collect()
    
    if save_results:
        # Plot scaling results
        plot_scaling(seq_lengths, results['seq_length'], 'Sequence Length', 'seq_length_scaling.png')
        plot_scaling(expert_counts, results['num_experts'], 'Number of Experts', 'expert_count_scaling.png')
    
    return results

def plot_scaling(x_values, results, x_label, filename):
    """Plot scaling results"""
    plt.figure(figsize=(12, 8))
    
    # Plot time scaling
    plt.subplot(2, 1, 1)
    plt.plot(x_values, results['time'], 'o-')
    plt.xlabel(x_label)
    plt.ylabel('Execution Time (ms)')
    plt.title(f'Time Scaling with {x_label}')
    plt.grid(True)
    
    # Plot memory scaling
    plt.subplot(2, 1, 2)
    plt.plot(x_values, results['memory'], 'o-')
    plt.xlabel(x_label)
    plt.ylabel('Memory Usage (MB)')
    plt.title(f'Memory Scaling with {x_label}')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Profile MoELanguageModel using PyTorch profiler')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run profiling on (cuda, mps, or cpu)')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 16], 
                        help='Batch sizes to test')
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[128, 512],
                        help='Sequence lengths to test')
    parser.add_argument('--scaling_test', action='store_true', help='Run scaling test with sequence length and expert count')
    parser.add_argument('--router_analysis', action='store_true', help='Analyze router component performance')
    parser.add_argument('--no_save', action='store_true', help='Disable saving results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine device
    if args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    print(f"Running profiling on {device}")
    
    if device.type == 'cuda':
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create output directory for results
    os.makedirs("performance_results", exist_ok=True)
    
    # Run benchmark tests
    print("\nRunning benchmark tests...")
    results, configs = benchmark_model_with_config(
        config,
        device,
        batch_sizes=args.batch_sizes,
        seq_lengths=args.seq_lengths,
        save_results=not args.no_save
    )
    
    # Print summary
    print("\nResults Summary:")
    print("================")
    
    for i, (batch_size, seq_len) in enumerate(configs):
        print(f"\nConfiguration: batch_size={batch_size}, seq_len={seq_len}")
        print(f"  Execution Time: {results['avg_time'][i]:.4f} ms")
        print(f"  Memory Usage: {results['memory'][i]:.2f} MB")
    
    # Run scaling test if requested
    if args.scaling_test:
        print("\nRunning Scaling Test:")
        print("=====================")
        
        scaling_results = scaling_test(config, device, save_results=not args.no_save)
        
        print("\nSequence Length Scaling Results:")
        for i, seq_len in enumerate(scaling_results['seq_length']['time']):
            print(f"  Sequence Length {i}: Time={scaling_results['seq_length']['time'][i]:.4f}ms, Memory={scaling_results['seq_length']['memory'][i]:.2f}MB")
        
        print("\nExpert Count Scaling Results:")
        for i, expert_count in enumerate(scaling_results['num_experts']['time']):
            print(f"  Expert Count {i}: Time={scaling_results['num_experts']['time'][i]:.4f}ms, Memory={scaling_results['num_experts']['memory'][i]:.2f}MB")
    
    # Run router analysis if requested
    if args.router_analysis:
        print("\nRouter Analysis:")
        print("=================")
        
        # Use a moderate size for analysis
        batch_size = 8
        seq_len = 512
        
        # Create the model
        model = create_model_from_config(config)
        model.to(device)
        model.eval()
        
        # Create inputs
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        
        analyze_router_performance(model, input_ids, attention_mask, device)

if __name__ == "__main__":
    main()