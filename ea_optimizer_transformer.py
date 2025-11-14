import os
import random
import subprocess
import csv
import json
import sys
from deap import base, creator, tools, algorithms

# ==== H100 80GB SEARCH SPACE ====
BATCH_SIZES = [8, 16, 32, 64]
SEQ_LENGTHS = [128, 256, 512, 1024]
POWER_LIMITS = [300, 400, 500, 600, 700]
MEM_CLOCKS = [1593, 2619]
GRAPHICS_CLOCKS = [1200, 1410, 1620, 1800, 1980]
PRECISIONS = ['fp32', 'fp16']

GPU_INDEX = 0
RESULTS_FILE = "ea_h100_transformer_results.csv"

if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gpu_index", "batch_size", "seq_length", "power_limit", "mem_clock", 
            "graphics_clock", "precision", "throughput_tokens_per_sec", "latency_ms", 
            "memory_gb", "power_w", "status"
        ])

creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()

# Create individual as a list with FIXED order
def create_individual():
    return creator.Individual([
        random.choice(BATCH_SIZES),      # 0: batch_size
        random.choice(SEQ_LENGTHS),      # 1: seq_length
        random.choice(POWER_LIMITS),     # 2: power_limit
        random.choice(MEM_CLOCKS),       # 3: mem_clock
        random.choice(GRAPHICS_CLOCKS),  # 4: graphics_clock
        random.choice(PRECISIONS)        # 5: precision
    ])

toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# FIXED GENETIC OPERATORS - maintain type constraints
def cx_constrained(ind1, ind2):
    """Crossover that respects parameter types"""
    # Only swap complete genes, don't shuffle positions
    if random.random() < 0.5:
        for i in range(len(ind1)):
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

def mut_constrained(ind, indpb=0.3):
    """Mutation that respects parameter types"""
    if random.random() < indpb:
        ind[0] = random.choice(BATCH_SIZES)
    if random.random() < indpb:
        ind[1] = random.choice(SEQ_LENGTHS)
    if random.random() < indpb:
        ind[2] = random.choice(POWER_LIMITS)
    if random.random() < indpb:
        ind[3] = random.choice(MEM_CLOCKS)
    if random.random() < indpb:
        ind[4] = random.choice(GRAPHICS_CLOCKS)
    if random.random() < indpb:
        ind[5] = random.choice(PRECISIONS)
    return ind,

def evaluate(ind):
    """Evaluate with strict type checking"""
    batch_size, seq_length, power_limit, mem_clock, graphics_clock, precision = ind
    status = "success"
    
    # Validate types
    if not isinstance(batch_size, int) or batch_size not in BATCH_SIZES:
        print(f"[ERROR] Invalid batch_size: {batch_size}")
        return 1.0, 9999.0, 9999.0
    if not isinstance(seq_length, int) or seq_length not in SEQ_LENGTHS:
        print(f"[ERROR] Invalid seq_length: {seq_length}")
        return 1.0, 9999.0, 9999.0
    if not isinstance(power_limit, int) or power_limit not in POWER_LIMITS:
        print(f"[ERROR] Invalid power_limit: {power_limit}")
        return 1.0, 9999.0, 9999.0
    if not isinstance(mem_clock, int) or mem_clock not in MEM_CLOCKS:
        print(f"[ERROR] Invalid mem_clock: {mem_clock}")
        return 1.0, 9999.0, 9999.0
    if not isinstance(graphics_clock, int) or graphics_clock not in GRAPHICS_CLOCKS:
        print(f"[ERROR] Invalid graphics_clock: {graphics_clock}")
        return 1.0, 9999.0, 9999.0
    if not isinstance(precision, str) or precision not in PRECISIONS:
        print(f"[ERROR] Invalid precision: {precision}")
        return 1.0, 9999.0, 9999.0

    # Set power limit
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "-i", str(GPU_INDEX), "-pl", str(power_limit)],
            check=True, capture_output=True, text=True, timeout=10
        )
    except Exception:
        status = "power_fail"

    # Set clocks
    try:
        subprocess.run(
            ["sudo", "nvidia-smi", "-i", str(GPU_INDEX), "-ac", f"{mem_clock},{graphics_clock}"],
            check=True, capture_output=True, text=True, timeout=10
        )
    except Exception:
        status = "clock_fail"

    # Run benchmark
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    out_fn = f"temp_result_{random.randint(1000,9999)}.json"

    try:
        subprocess.run([
            "python", "baseline_benchmark_transformer.py",
            f"--batch_size={batch_size}",
            f"--seq_length={seq_length}",
            f"--precision={precision}",
            f"--output={out_fn}",
            "--num_runs=50",
            "--warmup=5"
        ],
        check=True, env=env, capture_output=True, text=True, timeout=300)
        
        with open(out_fn) as f:
            results = json.load(f)
        result = results[0]
        os.remove(out_fn)
        
        throughput = result.get("throughput_tokens_per_sec", 1.0)
        latency = result.get("latency_mean_ms", 9999.0)
        memory = result.get("memory_allocated_gb", 0.0)
        power_w = result.get("power_mean_W", float(power_limit))
        
    except Exception as e:
        print(f"[ERROR] Benchmark failed: {batch_size},{seq_length},{power_limit},{mem_clock},{graphics_clock},{precision}")
        throughput, latency, memory, power_w = 1.0, 9999.0, 0.0, float(power_limit)
        status = "benchmark_fail"
    finally:
        if os.path.exists(out_fn):
            try:
                os.remove(out_fn)
            except:
                pass

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([GPU_INDEX, batch_size, seq_length, power_limit,
                         mem_clock, graphics_clock, precision, throughput, 
                         latency, memory, power_w, status])
    
    return throughput, latency, power_w

toolbox.register("evaluate", evaluate)
toolbox.register("mate", cx_constrained)
toolbox.register("mutate", mut_constrained, indpb=0.3)
toolbox.register("select", tools.selNSGA2)

def main():
    print("="*80)
    print("EVOLUTIONARY OPTIMIZER FOR H100 TRANSFORMER")
    print("="*80)
    
    pop = toolbox.population(n=12)
    hof = tools.HallOfFame(5)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: tuple(sum(x) / len(x) for x in zip(*fits)))
    stats.register("max", lambda fits: (max(f[0] for f in fits), min(f[1] for f in fits), min(f[2] for f in fits)))
    
    try:
        pop, logbook = algorithms.eaMuPlusLambda(
            pop, toolbox, mu=12, lambda_=12, cxpb=0.6, mutpb=0.4,
            ngen=20, stats=stats, halloffame=hof, verbose=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] EA failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS:")
    print("="*80)
    for i, ind in enumerate(hof, 1):
        batch_size, seq_length, power_limit, mem_clock, graphics_clock, precision = ind
        throughput, latency, power = ind.fitness.values
        print(f"\n{i}. Batch={batch_size}, Seq={seq_length}, Power={power_limit}W")
        print(f"   Mem={mem_clock}MHz, Gfx={graphics_clock}MHz, Prec={precision}")
        print(f"   → {throughput:,.0f} tok/s | {latency:.2f}ms | {power:.1f}W")
    
    print(f"\n{'='*80}")
    print(f"Results: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
