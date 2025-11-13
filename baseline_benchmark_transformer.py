import torch
import time
import numpy as np
import argparse
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List
import py3nvml.py3nvml as nvml

class BaselineBenchmark:
    def __init__(self, model_name: str = 'gpt2'):
        """Initialize model and tokenizer"""
        print(f"Loading {model_name}...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = self.model.cuda()
        self.model.eval()
        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        
        # Initialize NVML for power monitoring
        try:
            nvml.nvmlInit()
            self.nvml_handle = nvml.nvmlDeviceGetHandleByIndex(0)
            self.power_monitoring = True
        except:
            print("Warning: NVML initialization failed. Power monitoring disabled.")
            self.power_monitoring = False
    
    def benchmark_single_config(self, batch_size: int, seq_length: int, 
                                precision: str = 'fp32', num_runs: int = 100, 
                                warmup: int = 10) -> Dict:
        """Benchmark a single configuration"""
        print(f"Benchmarking: batch_size={batch_size}, seq_length={seq_length}, precision={precision}")
        
        # Set precision
        if precision == 'fp16':
            self.model = self.model.half()
            dtype = torch.float16
        else:
            self.model = self.model.float()
            dtype = torch.float32
        
        # Create dummy input
        input_ids = torch.randint(0, self.tokenizer.vocab_size, (batch_size, seq_length)).cuda()
        
        # Warmup
        print("Warmup...", end='', flush=True)
        for _ in range(warmup):
            with torch.no_grad():
                _ = self.model(input_ids)
        torch.cuda.synchronize()
        print(" Done")
        
        # Benchmark
        print(f"Running {num_runs} iterations...", end='', flush=True)
        latencies = []
        power_readings = []
        
        for _ in range(num_runs):
            # Record power before inference
            if self.power_monitoring:
                try:
                    power_before = nvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0  # Convert to W
                except:
                    power_before = 0
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(input_ids)
            
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            # Record power after inference
            if self.power_monitoring:
                try:
                    power_after = nvml.nvmlDeviceGetPowerUsage(self.nvml_handle) / 1000.0
                    power_readings.append((power_before + power_after) / 2)
                except:
                    pass
            
            latencies.append((end - start) * 1000)  # Convert to ms
        
        print(" Done")
        
        # Calculate statistics
        latencies = np.array(latencies)
        throughput = (batch_size * seq_length) / (np.mean(latencies) / 1000)  # tokens per second
        
        # Memory usage
        memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB
        
        results = {
            'batch_size': batch_size,
            'seq_length': seq_length,
            'precision': precision,
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_std_ms': float(np.std(latencies)),
            'latency_min_ms': float(np.min(latencies)),
            'latency_max_ms': float(np.max(latencies)),
            'throughput_tokens_per_sec': float(throughput),
            'memory_allocated_gb': float(memory_allocated),
            'memory_reserved_gb': float(memory_reserved),
        }
        
        if power_readings:
            results['power_mean_W'] = float(np.mean(power_readings))
            results['power_min_W'] = float(np.min(power_readings))
            results['power_max_W'] = float(np.max(power_readings))
        
        return results
    
    def save_results(self, results: List[Dict], output_file: str):
        """Save results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
    
    def __del__(self):
        """Cleanup NVML"""
        if self.power_monitoring:
            try:
                nvml.nvmlShutdown()
            except:
                pass


def main():
    # Configuration
    parser = argparse.ArgumentParser(description='Benchmark GPT-2 transformer model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
    parser.add_argument('--seq_length', type=int, default=128, help='Sequence length')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'],
                        help='Precision for inference')
    parser.add_argument('--output', type=str, default='results.json', help='Output JSON file')
    parser.add_argument('--num_runs', type=int, default=100, help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=10, help='Number of warmup runs')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Model name from HuggingFace')
    parser.add_argument('--sweep', action='store_true', 
                        help='Run sweep across multiple configurations (ignores single config args)')
    args = parser.parse_args()
    
    # Initialize benchmark
    benchmark = BaselineBenchmark(model_name=args.model_name)
    
    if args.sweep:
        # Run sweep across multiple configurations
        batch_sizes = [1, 4, 8, 16]
        seq_lengths = [128, 256, 512]
        precisions = ['fp32', 'fp16']
        
        results = []
        for batch_size in batch_sizes:
            for seq_length in seq_lengths:
                for precision in precisions:
                    try:
                        result = benchmark.benchmark_single_config(
                            batch_size=batch_size,
                            seq_length=seq_length,
                            precision=precision,
                            num_runs=args.num_runs,
                            warmup=args.warmup
                        )
                        results.append(result)
                    except RuntimeError as e:
                        print(f"Skipping config (batch={batch_size}, seq={seq_length}, precision={precision}): {e}")
                        continue
        
        # Save results
        benchmark.save_results(results, args.output)
        
        # Print summary table
        print("\n" + "="*100)
        print("BASELINE BENCHMARK SUMMARY")
        print("="*100)
        print(f"{'Batch':>6} {'SeqLen':>7} {'Precision':>9} {'Latency (ms)':>20} "
              f"{'Throughput (tok/s)':>20} {'Memory (GB)':>12} {'Power (W)':>12}")
        print("-"*100)
        
        for r in results:
            power_str = f"{r.get('power_mean_W', 0):>12.1f}" if 'power_mean_W' in r else "N/A".rjust(12)
            print(f"{r['batch_size']:>6} {r['seq_length']:>7} {r['precision']:>9} "
                  f"{r['latency_mean_ms']:>10.2f} ± {r['latency_std_ms']:>4.2f} "
                  f"{r['throughput_tokens_per_sec']:>20.0f} "
                  f"{r['memory_allocated_gb']:>12.2f} {power_str}")
    
    else:
        # Run single configuration (for EA optimizer)
        result = benchmark.benchmark_single_config(
            batch_size=args.batch_size,
            seq_length=args.seq_length,
            precision=args.precision,
            num_runs=args.num_runs,
            warmup=args.warmup
        )
        
        # Save as list with single element (for consistency with EA)
        benchmark.save_results([result], args.output)
        
        # Print result
        print("\n" + "="*80)
        print("BENCHMARK RESULT")
        print("="*80)
        print(f"Batch Size:       {result['batch_size']}")
        print(f"Sequence Length:  {result['seq_length']}")
        print(f"Precision:        {result['precision']}")
        print(f"Latency:          {result['latency_mean_ms']:.2f} ± {result['latency_std_ms']:.2f} ms")
        print(f"Throughput:       {result['throughput_tokens_per_sec']:.0f} tokens/s")
        print(f"Memory:           {result['memory_allocated_gb']:.2f} GB")
        if 'power_mean_W' in result:
            print(f"Power:            {result['power_mean_W']:.1f} W")
        print("="*80)


if __name__ == '__main__':
    main()
