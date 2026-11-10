import torch
import numpy as np
import time
import psutil
import os
from typing import Dict, List, Tuple, Optional

def get_memory_usage():
    """Get current CPU and GPU memory usage"""
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    cpu_usage = cpu_memory.percent
    cpu_available = cpu_memory.available / (1024**3)  # GB
    cpu_total = cpu_memory.total / (1024**3)  # GB
    
    # GPU memory
    gpu_info = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.memory_stats(i)
            allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024**3)  # GB
            max_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            
            gpu_info[f"GPU_{i}"] = {
                "allocated": allocated,
                "reserved": reserved,
                "total": max_memory,
                "usage_percent": (allocated / max_memory) * 100
            }
    
    return {
        "cpu": {
            "usage_percent": cpu_usage,
            "available_gb": cpu_available,
            "total_gb": cpu_total
        },
        "gpu": gpu_info
    }

def print_memory_stats(label: str = ""):
    """Print formatted memory statistics"""
    stats = get_memory_usage()
    
    print(f"\n{'='*50}")
    print(f"Memory Usage {label}")
    print(f"{'='*50}")
    
    # CPU stats
    cpu = stats["cpu"]
    print(f"CPU Memory: {cpu['usage_percent']:.1f}% used")
    print(f"Available: {cpu['available_gb']:.2f} GB / Total: {cpu['total_gb']:.2f} GB")
    
    # GPU stats
    if stats["gpu"]:
        for gpu_name, gpu_stats in stats["gpu"].items():
            print(f"\n{gpu_name}:")
            print(f"  Allocated: {gpu_stats['allocated']:.2f} GB")
            print(f"  Reserved: {gpu_stats['reserved']:.2f} GB")
            print(f"  Total: {gpu_stats['total']:.2f} GB")
            print(f"  Usage: {gpu_stats['usage_percent']:.1f}%")
    else:
        print("\nNo GPU available")
    
    print(f"{'='*50}\n")

def monitor_function_memory(func, *args, **kwargs):
    """Decorator/wrapper to monitor memory usage of a function"""
    print_memory_stats("Before function call")
    
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    print_memory_stats("After function call")
    print(f"Function execution time: {end_time - start_time:.2f} seconds")
    
    return result

def test_torch_operations():
    """Test function to demonstrate memory monitoring with PyTorch operations"""
    print("Testing PyTorch operations...")
    
    # Create some tensors
    print("Creating large tensors...")
    x = torch.randn(1000, 1000, device='cuda' if torch.cuda.is_available() else 'cpu')
    y = torch.randn(1000, 1000, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    print_memory_stats("After creating tensors")
    
    # Matrix multiplication
    print("Performing matrix multiplication...")
    z = torch.matmul(x, y)
    
    print_memory_stats("After matrix multiplication")
    
    # Clean up
    del x, y, z
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print_memory_stats("After cleanup")

def benchmark_memory_operations():
    """Benchmark different memory-intensive operations"""
    operations = {
        "CPU Large Array": lambda: np.random.randn(5000, 5000),
        "GPU Tensor": lambda: torch.randn(5000, 5000, device='cuda') if torch.cuda.is_available() else None,
        "CPU to GPU Transfer": lambda: torch.randn(5000, 5000).cuda() if torch.cuda.is_available() else None,
    }
    
    results = {}
    
    for op_name, operation in operations.items():
        if operation() is None and "GPU" in op_name:
            print(f"Skipping {op_name} - GPU not available")
            continue
            
        print(f"\nBenchmarking: {op_name}")
        print_memory_stats("Before")
        
        start_time = time.time()
        data = operation()
        end_time = time.time()
        
        print_memory_stats("After")
        
        results[op_name] = {
            "execution_time": end_time - start_time,
            "memory_after": get_memory_usage()
        }
        
        # Cleanup
        del data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results

def main():
    """Main function to demonstrate memory monitoring"""
    print("Memory Monitoring Script for COLMAP_SLAM")
    print_memory_stats("Initial state")
    
    # Test basic operations
    test_torch_operations()
    
    # Benchmark operations
    print("\n" + "="*60)
    print("BENCHMARKING MEMORY OPERATIONS")
    print("="*60)
    
    results = benchmark_memory_operations()
    
    # Print benchmark results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    
    for op_name, result in results.items():
        print(f"\n{op_name}:")
        print(f"  Execution time: {result['execution_time']:.4f} seconds")
        if result['memory_after']['gpu']:
            for gpu_name, gpu_stats in result['memory_after']['gpu'].items():
                print(f"  {gpu_name} usage: {gpu_stats['usage_percent']:.1f}%")

if __name__ == "__main__":
    main()