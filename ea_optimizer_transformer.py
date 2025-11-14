import os
import random
import subprocess
import csv
import json
from deap import base, creator, tools, algorithms
import pandas as pd

# Define search space for transformer optimization
BATCH_SIZES = [8, 16, 32, 64, 128]
SEQ_LENGTHS = [128, 256, 512]
POWER_LIMITS = [200, 300, 400, 500, 600, 700]  # Watts
MEM_CLOCKS = [5001, 5201, 5401]  # MHz (adjust for your GPU)
PRECISIONS = ["fp64","fp32", "fp16", "fp8"]

GPU_INDEX = 0
RESULTS_FILE = "ea_transformer_results.csv"
GENERATION_LOG_FILE = "ea_transformer_generation_log.csv"

# Initialize results file
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "gpu_index", "batch_size", "seq_length", "power_limit", 
            "mem_clock", "precision", "latency_ms", "throughput_tokens_per_sec", 
            "memory_gb", "power_w"
        ])

# Multi-objective: maximize throughput, minimize latency and power
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0, -1.0))  # (throughput+, latency-, power-)
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("batch", random.choice, BATCH_SIZES)
toolbox.register("seq", random.choice, SEQ_LENGTHS)
toolbox.register("power", random.choice, POWER_LIMITS)
toolbox.register("mem", random.choice, MEM_CLOCKS)
toolbox.register("precision", random.choice, PRECISIONS)

toolbox.register(
    "individual",
    tools.initCycle,
    creator.Individual,
    (toolbox.batch, toolbox.seq, toolbox.power, toolbox.mem, toolbox.precision),
    n=1,
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("map", map)

def evaluate(ind):
    """Evaluate a transformer configuration"""
    batch_size, seq_length, power_limit, mem_clock, precision = ind
    
    # Set hardware parameters
    try:
        subprocess.run(
            ["nvidia-smi", "-i", str(GPU_INDEX), "-pl", str(power_limit)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.run(
            ["nvidia-smi", "-i", str(GPU_INDEX), "-ac", f"{mem_clock},{mem_clock}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(f"Hardware setting failed for config {ind}")
    
    # Run benchmark
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    
    result_file = f"temp_result_{random.randint(1000,9999)}.json"
    
    subprocess.run([
        "python", "baseline_benchmark_transformer.py",
        f"--batch_size={batch_size}",
        f"--seq_length={seq_length}",
        f"--precision={precision}",
        f"--output={result_file}"
    ], check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Parse results
    with open(result_file, "r") as f:
        results = json.load(f)
    os.remove(result_file)
    
    # Extract metrics (assuming single run)
    latency_ms = results[0]["latency_mean_ms"]
    throughput = results[0]["throughput_tokens_per_sec"]
    memory_gb = results[0]["memory_allocated_gb"]
    power_w = results[0].get("power_mean_W", 0)
    
    # Save to CSV
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            GPU_INDEX, batch_size, seq_length, power_limit, 
            mem_clock, precision, latency_ms, throughput, memory_gb, power_w
        ])
    
    return throughput, latency_ms, power_w

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
toolbox.register("select", tools.selNSGA2)

def main():
    """Run evolutionary optimization"""
    pop = toolbox.population(n=12)
    hof = tools.HallOfFame(5)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", lambda fits: (
        sum(f[0] for f in fits) / len(fits),
        sum(f[1] for f in fits) / len(fits),
        sum(f[2] for f in fits) / len(fits)
    ))
    stats.register("max", lambda fits: (
        max(f[0] for f in fits),
        min(f[1] for f in fits),
        min(f[2] for f in fits)
    ))
    
    print("Starting EA optimization for Transformer workloads...")
    pop, logbook = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=12, lambda_=12, cxpb=0.6, mutpb=0.3,
        ngen=10, stats=stats, halloffame=hof, verbose=True
    )
    
    # Save generation log
    log_df = pd.DataFrame(logbook)
    log_df.to_csv(GENERATION_LOG_FILE, index=False)
    print(f"\nGeneration log saved to {GENERATION_LOG_FILE}")
    
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS:")
    print("="*80)
    for i, ind in enumerate(hof, 1):
        batch_size, seq_length, power_limit, mem_clock, precision = ind
        throughput, latency, power = ind.fitness.values
        print(f"\n{i}. Batch: {batch_size}, SeqLen: {seq_length}, Power: {power_limit}W, "
              f"MemClock: {mem_clock}MHz, Precision: {precision}")
        print(f"   Throughput: {throughput:.0f} tokens/s, Latency: {latency:.2f}ms, Power: {power:.1f}W")
    
    print(f"\nAll results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
