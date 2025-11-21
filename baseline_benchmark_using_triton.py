import argparse
import torch
import time
import json
import sys
from transformers import GPT2LMHeadModel, GPT2Config
from profiling_utils import nvtx_range

# H100 specific: Allow TF32 for FP32 operations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def run_whitebox_benchmark(batch_size, seq_len, precision, power_limit):
    # 1. PRECISION MAPPING
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"--> [WhiteBox] Running: BS={batch_size}, SEQ={seq_len}, PREC={precision}, PWR={power_limit}")

    try:
        # 2. MODEL SETUP
        config = GPT2Config.from_pretrained("gpt2")
        # Force deterministic execution (removes randomness from the equation)
        model = GPT2LMHeadModel(config).to(device="cuda", dtype=dtype)
        model.eval()

        # 3. COMPILE (The Fix: Let Inductor handle the Graphs)
        # 'reduce-overhead' automatically uses CUDA Graphs where possible
        print("--> [WhiteBox] Compiling...")
        model = torch.compile(model, mode="reduce-overhead")

        # 4. DATA SETUP
        input_ids = torch.randint(0, 50257, (batch_size, seq_len), device="cuda")

        # 5. WARMUP (Essential for Compilation)
        # The compiler needs to see the data flow a few times to build the optimized kernel
        print("--> [WhiteBox] Warming up...")
        with torch.no_grad():
            # Force Flash Attention context
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                for _ in range(10):
                    _ = model(input_ids)

        # 6. BENCHMARK LOOP
        num_steps = 50
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        print("--> [WhiteBox] Benchmarking...")
        start_event.record()
        
        with torch.no_grad():
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                for _ in range(num_steps):
                    _ = model(input_ids)
                    
        end_event.record()
        torch.cuda.synchronize()

        # 7. METRICS
        total_time_ms = start_event.elapsed_time(end_event)
        avg_latency = total_time_ms / num_steps
        throughput = (batch_size * seq_len * num_steps) / (total_time_ms / 1000)
        
        # MFU Calculation
        flops_per_token = 2 * 124e6 
        h100_peak = 1979e12 if (precision in ["fp16", "bf16"]) else 60e12 
        mfu = (throughput * flops_per_token) / h100_peak

        results = {
            "batch_size": batch_size,
            "seq_length": seq_len,
            "precision": precision,
            "throughput": throughput,
            "latency": avg_latency,
            "mfu": mfu * 100,
            "status": "success"
        }
        
        print(f"__EA_RESULT__:{json.dumps(results)}")

    except Exception as e:
        err_msg = str(e)
        print(f"[BENCHMARK FAIL] {err_msg}")
        # Keep the EA alive by returning a failure JSON
        print(f"__EA_RESULT__:{json.dumps({'status': 'failed', 'error': err_msg})}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--power_limit", type=int)
    args = parser.parse_args()
    
    run_whitebox_benchmark(args.batch_size, args.seq_length, args.precision, args.power_limit)