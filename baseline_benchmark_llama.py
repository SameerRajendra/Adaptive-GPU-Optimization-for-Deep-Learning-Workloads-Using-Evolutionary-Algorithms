import argparse
import torch
import time
import json
import sys
from transformers import AutoModelForCausalLM, AutoConfig
from profiling_utils import nvtx_range

# H100 Optimization Flags
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def run_whitebox_benchmark(model_name, batch_size, seq_len, precision, power_limit):
    # 1. PRECISION SETUP
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"--> [WhiteBox] Loading {model_name} | BS={batch_size} | SEQ={seq_len} | {precision}")

    try:
        # 2. MODEL LOADING (Generic AutoModel)
        # We use 'trust_remote_code=True' to support modern models like Phi/Mpt if needed
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # HARDWARE CHECK: GPT-2 has a hard limit of 1024 tokens. 
        # We clamp the sequence length to avoid crashing.
        max_pos = getattr(config, "n_positions", getattr(config, "max_position_embeddings", 2048))
        if seq_len > max_pos:
            print(f"[WARN] reducing seq_len {seq_len} -> {max_pos} to fit model limit")
            seq_len = max_pos

        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            config=config, 
            torch_dtype=dtype, 
            trust_remote_code=True
        ).to("cuda")
        model.eval()

        # 3. COMPILE
        print("--> [WhiteBox] Compiling (Heavy Lift)...")
        # 'max-autotune' is better for large models if you have patience (60s+ compile time)
        # We stick to 'reduce-overhead' for speed/stability balance
        model = torch.compile(model, mode="reduce-overhead")

        # 4. DATA SETUP
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device="cuda")

        # 5. WARMUP
        print("--> [WhiteBox] Warming up...")
        with torch.no_grad():
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                for _ in range(3): # Fewer warmup steps for big models to save time
                    _ = model(input_ids)

        # 6. BENCHMARK
        print("--> [WhiteBox] Benchmarking...")
        num_steps = 20 # Reduce steps for large models (they are slower)
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

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
        
        # MFU CALCULATION (Generic)
        # Estimate FLOPs: 2 * Num_Params * Tokens
        num_params = sum(p.numel() for p in model.parameters())
        flops_per_token = 2 * num_params
        h100_peak = 989e12 if (precision in ["fp16", "bf16"]) else 60e12 # Dense Peak
        mfu = (throughput * flops_per_token) / h100_peak

        results = {
            "model": model_name,
            "batch_size": batch_size,
            "seq_length": seq_len,
            "precision": precision,
            "throughput": throughput,
            "latency": avg_latency,
            "mfu": mfu * 100,
            "params": num_params,
            "status": "success"
        }
        
        print(f"__EA_RESULT__:{json.dumps(results)}")

    except Exception as e:
        err_msg = str(e)
        # print(f"[BENCHMARK FAIL] {err_msg}") 
        # Reduce noise in logs, the EA script will catch this
        print(f"__EA_RESULT__:{json.dumps({'status': 'failed', 'error': err_msg})}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2-xl") # Default to Big Model
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--seq_length", type=int)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--power_limit", type=int)
    args = parser.parse_args()
    
    run_whitebox_benchmark(args.model_name, args.batch_size, args.seq_length, args.precision, args.power_limit)