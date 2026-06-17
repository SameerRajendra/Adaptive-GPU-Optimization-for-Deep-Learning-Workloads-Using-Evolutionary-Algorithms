#!/usr/bin/env python3
import argparse
import os
import sys
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from math import isfinite

# H100 Optimization Flags (allow TF32 for matmuls where appropriate)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def apply_autotune_envs(block_size, tile_cfg, fused_kernel, attn_kernel):
    """
    Export autotuning knobs via environment variables and 
    return the PyTorch SDPA context dictionary.
    """
    if block_size is not None:
        os.environ["BLOCK_SIZE"] = str(block_size)
    if tile_cfg is not None:
        os.environ["TILE_CFG"] = str(tile_cfg)
    if fused_kernel is not None:
        os.environ["FUSED_KERNEL"] = str(fused_kernel)
    if attn_kernel is not None:
        os.environ["FLASH_ATTN_KERNEL"] = str(attn_kernel)

    # Map optimizer choices to torch.backends.cuda.sdp_kernel
    if attn_kernel == "flash":
        sdpk = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)
    elif attn_kernel == "math":
        sdpk = dict(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    elif attn_kernel == "mem_efficient":
        sdpk = dict(enable_flash=False, enable_math=False, enable_mem_efficient=True)
    else:
        # Default/Fallback (e.g., for "cutlass" or "none")
        # We default to Flash if available, otherwise PyTorch decides
        sdpk = dict(enable_flash=True, enable_math=False, enable_mem_efficient=False)

    return sdpk

def estimate_flops_per_token(num_params, flops_per_param=53.0):
    return float(num_params) * float(flops_per_param)

def run_whitebox_benchmark(model_name, batch_size, seq_len, precision, power_limit,
                           block_size=None, tile_cfg=None, fused_kernel=None, attn_kernel=None,
                           flops_per_param=53.0, num_steps=20, warmup_steps=3):
    
    # 1. PRECISION SETUP
    if precision == "bf16":
        dtype = torch.bfloat16
    elif precision == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    # 2. APPLY KNOBS
    sdpk = apply_autotune_envs(block_size, tile_cfg, fused_kernel, attn_kernel)

    # (Optional) Print to stderr so it doesn't mess up stdout JSON parsing if strict
    # but your EA script ignores non-JSON lines, so print is fine.
    # print(f"--> [WhiteBox] Autotune: block={block_size} tile={tile_cfg} fused={fused_kernel} attn={attn_kernel}")

    try:
        # 3. MODEL LOADING
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # Cap sequence length to model max to prevent crash
        max_pos = getattr(config, "n_positions", getattr(config, "max_position_embeddings", 2048))
        if seq_len > max_pos:
            seq_len = max_pos

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto" # Handle multi-GPU or just auto-place on CUDA
        )
        model.eval()
        
        # 4. COMPILATION (Optional but good for H100)
        # Note: This might timeout the first run in the EA if it takes > 300s
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass # Fallback if compile fails

        num_params = sum(p.numel() for p in model.parameters())

        # 5. INPUT GENERATION
        input_ids = torch.randint(
            0, config.vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.long
        )

        # 6. WARMUP
        with torch.no_grad(), torch.backends.cuda.sdp_kernel(**sdpk):
            for _ in range(warmup_steps):
                _ = model(input_ids)
        
        torch.cuda.synchronize()

        # 7. BENCHMARK LOOP
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad(), torch.backends.cuda.sdp_kernel(**sdpk):
            for _ in range(num_steps):
                _ = model(input_ids)
        end.record()
        torch.cuda.synchronize()

        # 8. METRICS CALCULATION
        total_time_ms = start.elapsed_time(end)
        
        if total_time_ms <= 0:
            raise ValueError(f"Invalid time measured: {total_time_ms}ms")

        avg_latency_sec = (total_time_ms / 1000.0) / num_steps
        throughput = (batch_size * seq_len) / avg_latency_sec

        # MFU Calculation
        flops_per_token = estimate_flops_per_token(num_params, flops_per_param)
        # Peak FLOPs approximation for H100 (SXM5) or A100
        # H100 FP16 TC ~989 TFLOPS. A100 ~312 TFLOPS.
        # We'll use H100 number as default since you mentioned H100 flags.
        gpu_peak_flops = 989e12 if precision in ["bf16", "fp16"] else 60e12
        mfu_pct = ((throughput * flops_per_token) / gpu_peak_flops) * 100

        # 9. JSON OUTPUT FOR OPTIMIZER
        result = {
            "status": "success",
            "model": model_name,
            "batch_size": batch_size,
            "seq_length": seq_len,
            "precision": precision,
            "block_size": block_size,
            "tile_cfg": tile_cfg,
            "fused_kernel": fused_kernel,
            "attn_kernel": attn_kernel,
            "throughput": throughput,
            "latency": total_time_ms / num_steps, # ms
            "mfu": mfu_pct,
            "params": num_params
        }
        print("__EA_RESULT__:" + json.dumps(result))

    except torch.cuda.OutOfMemoryError:
        print("__EA_RESULT__:" + json.dumps({"status": "failed", "error": "OOM"}))
    except Exception as e:
        print("__EA_RESULT__:" + json.dumps({"status": "failed", "error": str(e)}))

def parse_args():
    parser = argparse.ArgumentParser(description="Baseline LLaMA prefill benchmark with autotune knobs")
    parser.add_argument("--model_name", type=str, default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--seq_length", type=int, required=True)
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], required=True)
    parser.add_argument("--power_limit", type=int, default=700)
    
    # Autotune knobs
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--tile_cfg", type=str, default=None)
    parser.add_argument("--fused_kernel", type=str, default=None)
    parser.add_argument("--attn_kernel", type=str, default="flash")
    
    # Eval control
    parser.add_argument("--flops_per_param", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=20)
    parser.add_argument("--warmup_steps", type=int, default=3)
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_whitebox_benchmark(
        model_name=args.model_name,
        batch_size=args.batch_size,
        seq_len=args.seq_length,
        precision=args.precision,
        power_limit=args.power_limit,
        block_size=args.block_size,
        tile_cfg=args.tile_cfg,
        fused_kernel=args.fused_kernel,
        attn_kernel=args.attn_kernel,
        flops_per_param=args.flops_per_param,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps
    )