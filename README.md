# AEGOF — Adaptive Evolutionary GPU Optimization Framework

> An automated multi-objective autotuning pipeline for **LLaMA-2-7B on NVIDIA H100**, jointly optimizing 8 hardware/software knobs using an NSGA-II evolutionary algorithm. Achieves up to **8.09× efficiency multiplier** over standard PyTorch defaults.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [How It Works](#how-it-works)
- [Search Space](#search-space)
- [Repository Structure](#repository-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Requirements](#requirements)

---

## Overview

Manually tuning GPU execution parameters for LLMs is a combinatorial nightmare — batch sizes, power limits, precision formats, kernel strategies, and tile configurations interact non-linearly. AEGOF automates this via **evolutionary search**:

- **NSGA-II multi-objective optimizer** (via DEAP) balancing throughput (tokens/s) vs. power consumption (Watts)
- **FlashAttention-2** and **CUTLASS** backends integrated via a white-box benchmarking harness
- **Robust fault tolerance**: handles CUDA OOM and subprocess timeouts, enabling 238+ unique configurations evaluated without manual intervention
- **Hall of Fame** output: top configurations saved to JSON for reproducible deployment

---

## Key Results

| Metric | PyTorch Default | AEGOF Best Config | Improvement |
|--------|----------------|-------------------|-------------|
| Efficiency multiplier | 1.0× | **8.09×** | +709% |
| Configurations evaluated | — | 238+ | — |
| Generations run | — | 5 | — |
| Initial population size | — | 16 | — |

---

## How It Works

```
  ┌────────────────────────────┐
  │   Random Initial Population    │  (16 configurations)
  └───────────┬───────────────┘
                 │
                 ▼
  ┌────────────────────────────┐
  │   Benchmark Evaluation          │  (baseline_benchmark_llama_v2.py)
  │   • Throughput (tokens/s)       │
  │   • Power draw (Watts)          │
  │   • MFU measurement             │
  └───────────┬───────────────┘
                 │
                 ▼
  ┌────────────────────────────┐
  │   NSGA-II Selection             │  (μ+λ strategy)
  │   Crossover & Mutation          │
  └───────────┬───────────────┘
                 │
          5 Generations
                 │
                 ▼
  ┌────────────────────────────┐
  │   Hall of Fame (JSON)           │  (top configs saved)
  └────────────────────────────┘
```

---

## Search Space

| Category | Parameter | Options |
|----------|-----------|--------|
| **Workload** | Batch Size | `[1, 2, 4, 8, 16, 32]` |
| **Workload** | Sequence Length | `[128, 256, 512, 1024]` |
| **Hardware** | Power Limit (W) | `[300, 500, 600, 700]` |
| **Hardware** | Precision | `[bf16, fp16]` |
| **Compute** | Block Size | `[64, 128, 256]` |
| **Compute** | Tile Config | `[tile_16x16, tile_32x32]` |
| **Kernels** | Fused Kernel | `[standard, fused_mha_ffn]` |
| **Kernels** | Attention Backend | `[flash, math, mem_efficient, cutlass]` |

---

## Repository Structure

```
AEGOF/
├── baseline_benchmark_llama_v2.py   # White-box benchmark harness
├── ea_optimizer_v4.py               # NSGA-II evolutionary optimizer (DEAP)
├── llama_champions_v2.json          # Hall of Fame — best discovered configs
├── llama_trials_v3.csv              # Full trial log (params, throughput, MFU, errors)
├── transformer_result.xlsx          # Generation-over-generation fitness logs
├── Sameer_AAI_800_Final_Report.pdf  # Full project report
└── README.md
```

---

## Setup & Installation

### Prerequisites
- NVIDIA H100 GPU (or any CUDA-compatible GPU)
- CUDA Toolkit ≥ 12.0
- Python 3.10+

```bash
git clone https://github.com/SameerRajendra/Adaptive-GPU-Optimization-for-Deep-Learning-Workloads-Using-Evolutionary-Algorithms.git
cd Adaptive-GPU-Optimization-for-Deep-Learning-Workloads-Using-Evolutionary-Algorithms
pip install torch transformers deap flash-attn
```

---

## Usage

```bash
# Run a single benchmark config manually
python baseline_benchmark_llama_v2.py \
    --batch_size 8 \
    --seq_len 512 \
    --precision bf16 \
    --power_limit 600 \
    --attention flash

# Run the full evolutionary optimizer
python ea_optimizer_v4.py
# Outputs: llama_champions_v2.json (best configs)
#          llama_trials_v3.csv     (full trial log)
```

---

## Requirements

```
torch>=2.2.0
transformers>=4.40.0
deap>=1.4.1
flash-attn>=2.5.0
nvidia-ml-py>=12.0.0
```

---

*Course: AAI 800 — Stevens Institute of Technology • Sept–Dec 2025*  
*Author: [Sameer Rajendra](https://github.com/SameerRajendra)*  
*[📄 Read the Full Report](https://raw.githubusercontent.com/SameerRajendra/Adaptive-GPU-Optimization-for-Deep-Learning-Workloads-Using-Evolutionary-Algorithms/transformer-with-custom-kernal/Sameer_AAI_800_Final_Report.pdf)*
