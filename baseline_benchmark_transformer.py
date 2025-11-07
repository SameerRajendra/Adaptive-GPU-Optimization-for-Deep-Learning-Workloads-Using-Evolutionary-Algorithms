"""
Baseline benchmark for GPT-2 model
Measures inference performance without optimizations
"""

import torch
import time
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List
import json
from py3nvml import py3nvml


class BaselineBenchmark:
    def __init__(self, model_name: str = 'gpt2'):
        """Initialize model and tokenizer"""
        print(f"Loading {model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = self.model.cuda()
        self.model.eval()
        py3nvml.nvmlInit()
        self.nvml_handle = py3nvml.nvmlDeviceGetHandleByIndex(0)  # Use device 0

        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
    
    def log_power(self):
        power = py3nvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0  # Returns mW, convert to W
        return power



    def benchmark_inference(self, batch_sizes: List[int], seq_lengths: List[int], 
                          num_runs: int = 100, warmup: int = 10) -> Dict:
        """Benchmark inference across different configurations"""
        results = []
        
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                print(f"\nBenchmarking batch_size={batch_size}, seq_length={seq_length}")
                
                # Create dummy input
                input_ids = torch.randint(0, self.tokenizer.vocab_size, 
                                        (batch_size, seq_length)).cuda()
                
                # Warmup
                print("  Warmup...", end=" ")
                for _ in range(warmup):
                    with torch.no_grad():
                        _ = self.model(input_ids)
                torch.cuda.synchronize()
                print("Done")
                
                # Benchmark
                print("  Benchmarking...", end=" ")
                latencies = []
                powers=[]
                
                for _ in range(num_runs):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    power_before = self.log_power()
                    
                    with torch.no_grad():
                        output = self.model(input_ids)
                    
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    power_after =self.log_power()
                    latencies.append((end - start) * 1000)  # Convert to ms
                    powers.append((power_before+power_after)/2)
                
                avg_power = float(np.mean(powers))
                std_power = float(np.std(powers))
                min_power = float(np.min(powers))
                max_power = float(np.max(powers))
                
                print("Done")
                
                # Calculate statistics
                latencies = np.array(latencies)
                throughput = (batch_size * seq_length) / (latencies.mean() / 1000)  # tokens/sec
                
                # Memory usage
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                
                result = {
                    'batch_size': batch_size,
                    'seq_length': seq_length,
                    'latency_mean_ms': float(latencies.mean()),
                    'latency_std_ms': float(latencies.std()),
                    'latency_min_ms': float(latencies.min()),
                    'latency_max_ms': float(latencies.max()),
                    'throughput_tokens_per_sec': float(throughput),
                    'memory_allocated_gb': float(memory_allocated),
                    'memory_reserved_gb': float(memory_reserved),
                    'power_mean_W': avg_power,
                    'power_min_W': min_power,
                    'power_max_W': max_power
                }
                
                results.append(result)
                
                # Print summary
                print(f"  Latency: {result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f} ms")
                print(f"  Throughput: {result['throughput_tokens_per_sec']:.0f} tokens/sec")
                print(f"  Memory: {result['memory_allocated_gb']:.2f} GB")
                print(f"  Power: {result['power_mean_W']:.2f} W")
        
        
        return results
    
    def save_results(self, results: List[Dict], filename: str = 'results/baseline_results.json'):
        """Save results to JSON file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filename}")

def main():
    # Configuration
    batch_sizes = [1, 4, 8, 16]
    seq_lengths = [64, 128, 256, 512]
    
    # Run benchmark
    benchmark = BaselineBenchmark(model_name='gpt2')
    results = benchmark.benchmark_inference(batch_sizes, seq_lengths, num_runs=100)
    benchmark.save_results(results)
    
    # Print summary table
    print("\n" + "="*80)
    print("BASELINE BENCHMARK SUMMARY")
    print("="*80)
    print(f"{'Batch':>6} {'SeqLen':>7} {'Latency (ms)':>15} {'Throughput (tok/s)':>20} {'Memory (GB)':>12}")
    print("-"*80)
    
    for r in results:
        print(f"{r['batch_size']:>6} {r['seq_length']:>7} "
              f"{r['latency_mean_ms']:>10.2f} ± {r['latency_std_ms']:>4.2f} "
              f"{r['throughput_tokens_per_sec']:>20.0f} "
              f"{r['memory_allocated_gb']:>12.2f}")

if __name__ == '__main__':
    main()

    py3nvml.nvmlShutdown()
