# LLM-GPU-Autotuner: Evolutionary Parameter Optimization

This repository contains an automated framework for optimizing the hardware and software execution parameters of Large Language Models (specifically LLaMA-2-7B) on high-performance accelerators like NVIDIA H100 GPUs. 

By leveraging an Evolutionary Algorithm (EA), the system intelligently navigates a massive, complex search space of GPU configurations—including batch sizes, precision formats, power limits, and low-level CUDA kernel choices—to find the configurations that yield the highest throughput (tokens/second) and optimal power efficiency.

## 📁 Repository Structure

* **`baseline_benchmark_llama_v2.py`**: The core evaluation engine. It runs a whitebox benchmark of the `NousResearch/Llama-2-7b-hf` model. It accepts dynamic arguments for batch size, sequence length, precision (`bf16`, `fp16`), power limits, and autotuning knobs (block sizes, tile configurations, and attention kernels like FlashAttention or CUTLASS).
* **`ea_optimizer_v4.py`**: The evolutionary search script built using the `deap` framework. It initializes a population of configurations, evaluates their fitness via the benchmark script, and applies crossover and mutation to evolve highly optimized parameter sets over multiple generations.
* **`llama_champions_v2.json`**: A Hall of Fame output file containing the absolute best configurations discovered during the evolutionary runs, ranked by throughput.
* **`llama_trials_v3.csv`**: A comprehensive log of all trial executions, recording timestamps, parameters, resulting throughput, Model Flops Utilization (MFU), and error logs for failed runs.
* **`transformer_result.xlsx - Sheet1.csv`**: Generation-over-generation statistical logs tracking the average and maximum fitness improvements of the H100 transformer benchmark.

## 🧠 How It Works

The optimizer treats the GPU execution parameters as "genes" in an individual configuration. 



1.  **Initialization:** The script generates a random initial population of 16 distinct configurations.
2.  **Evaluation:** Each configuration is passed to the benchmark script. The benchmark measures the throughput (tokens/s) and records the power efficiency. 
3.  **Selection & Evolution:** Using a $\mu + \lambda$ evolutionary strategy, the best-performing configurations are selected to "breed." 
4.  **Crossover & Mutation:** The algorithm swaps parameters between successful runs and randomly mutates others to explore new optimization strategies (e.g., swapping `tile_16x16` for `tile_32x32` or changing the thread block size).
5.  **Hall of Fame:** After 5 generations, the absolute best configurations are saved to `llama_champions_v2.json`.

### 🔍 The Search Space

| Category | Parameters | Options |
| :--- | :--- | :--- |
| **Workload** | Batch Size, Seq Length | `[1, 2, 4, 8, 16, 32]`, `[128, 256, 512, 1024]` |
| **Hardware** | Power Limit (W), Precision | `[300, 500, 600, 700]`, `[bf16, fp16]` |
| **Compute** | Block Size, Tile Config | `[64, 128, 256]`, `[tile_16x16, tile_32x32]` |
| **Kernels** | Fused Kernel, Attention | `[standard, fused_mha_ffn]`, `[flash, math, mem_efficient, cutlass]` |

## 🚀 Getting Started

### Prerequisites

Ensure you have a CUDA-compatible environment (ideally with H100 GPUs) and the following Python packages installed:

```bash
pip install torch transformers deap
