import os
import random
import subprocess
import json
import sys
import numpy as np
from deap import base, creator, tools, algorithms

# ==== H100 HYBRID SEARCH SPACE ====
# We shifted the window: Small batches are waste on H100.
# FlashAttention allows much longer sequences.
BATCH_SIZES = [32, 64, 128, 256]  
SEQ_LENGTHS = [512, 1024, 2048, 4096] # H100 loves long sequences
POWER_LIMITS = [500, 600, 700] # Ignore <500W, it's just throttling
PRECISIONS = ['fp16'] # Force FP16, FP32 is obsolete for this

# Setup DEAP
if "FitnessMulti" not in creator.__dict__:
    # Maximize Throughput, Minimize Latency, Minimize Power (Optional)
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0, -0.1))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def get_random_params():
    return [
        random.choice(BATCH_SIZES),
        random.choice(SEQ_LENGTHS),
        random.choice(POWER_LIMITS),
        "fp16"
    ]

toolbox.register("attr_params", get_random_params)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_params)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    bs, seq, pwr, prec = individual
    
    # Construct command to call the White-Box Engine
    cmd = [
        "python", "whitebox_benchmark.py",
        "--batch_size", str(bs),
        "--seq_length", str(seq),
        "--power_limit", str(pwr),
        "--precision", prec
    ]
    
    try:
        # Run the Ferrari engine
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # Parse the specific output tag we added in Step 2
        for line in result.stdout.split('\n'):
            if line.startswith("__EA_RESULT__:"):
                data = json.loads(line.replace("__EA_RESULT__:", ""))
                
                if data['status'] == 'failed':
                    return 0, 10000, 700 # Penalize failure
                    
                # Return: Throughput (Max), Latency (Min), Power (Min - placeholder)
                # Note: We prioritize Throughput heavily now
                return data['throughput'], data['latency'], pwr
                
        return 0, 10000, 700 # Parse failure penalty

    except subprocess.TimeoutExpired:
        return 0, 10000, 700
    except Exception as e:
        print(f"Eval failed: {e}")
        return 0, 10000, 700

# Register Genetic Operators
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=10, indpb=0.2) # Simplified mutation
toolbox.register("select", tools.selNSGA2)

def run_hybrid_optimization():
    print("Starting Hybrid White-Box Evolutionary Optimization...")
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(5)
    
    # Run for fewer generations because each step is high-quality engineering
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max_tput", lambda fits: max(f[0] for f in fits))
    
    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=10, lambda_=10, 
        cxpb=0.5, mutpb=0.2, ngen=5, 
        stats=stats, halloffame=hof, verbose=True
    )
    
    print("\nTop Optimized Configurations:")
    for ind in hof:
        print(f"BS: {ind[0]}, Seq: {ind[1]}, Pwr: {ind[2]}W -> Fitness: {ind.fitness.values}")

if __name__ == "__main__":
    run_hybrid_optimization()