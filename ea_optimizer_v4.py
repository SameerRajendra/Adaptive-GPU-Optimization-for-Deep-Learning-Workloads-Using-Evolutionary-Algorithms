import random
import subprocess
import json
import sys
import os
import csv
from datetime import datetime
from deap import base, creator, tools, algorithms

# =============================================================================
# 1. CONFIGURATION & SEARCH SPACE
# =============================================================================
BENCHMARK_SCRIPT = "baseline_benchmark_llama_v2.py"  # Ensure this script accepts the new args!
TRIALS_LOG_FILE = "llama_trials_v3.csv"
CHAMPIONS_FILE = "llama_champions_v3.json"
TARGET_MODEL = "NousResearch/Llama-2-7b-hf"

# --- Hardware Knobs ---
# Batch: Small batches for Llama-2-7B to fit in HBM/SRAM
BATCH_OPTS = [1, 2, 4, 8, 16, 32] 
# Seq: Typical context lengths
SEQ_OPTS = [128, 256, 512, 1024]
# Power: The efficiency search (W)
POWER_OPTS = [300, 500, 600, 700]
# Precision: Data type
PREC_OPTS = ["bf16", "fp16"]

# --- New Kernel/Compiler Knobs (The "Autotuner" layer) ---
# Block Size: Controls thread block dimensions (Occupancy vs Reg Pressure)
BLOCK_OPTS = [64, 128, 256]
# Tile Config: Memory access granularity
TILE_OPTS = ["tile_16x16", "tile_32x32"]
# Kernel Fusion: Aggressiveness of fusing ops
FUSED_OPTS = ["fused_mha_ffn", "standard"]
# Attention Backend: Underlying math library
ATTN_OPTS = ["cutlass", "math", "mem_efficient"]

# =============================================================================
# 2. LOGGING INFRASTRUCTURE
# =============================================================================
def init_log_file():
    if not os.path.exists(TRIALS_LOG_FILE):
        headers = [
            "timestamp", "bs", "seq", "pwr", "prec", 
            "block", "tile", "fused", "attn", 
            "tput", "mfu", "status", "error"
        ]
        with open(TRIALS_LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(headers)

def log_trial(params, result):
    with open(TRIALS_LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%H:%M:%S"),
            params['bs'], params['seq'], params['pwr'], params['prec'],
            params['block'], params['tile'], params['fused'], params['attn'],
            f"{result.get('throughput', 0):.0f}", 
            f"{result.get('mfu', 0):.2f}", 
            result.get('status', 'unknown'), 
            result.get('error', '')
        ])

# =============================================================================
# 3. DEAP SETUP (Evolutionary Core)
# =============================================================================
# Fitness: Maximize Throughput (1.0), Minimize Power (-0.1 - soft constraint)
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -0.05)) # Slight penalty for high power
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def get_random_individual():
    return [
        random.randint(0, len(BATCH_OPTS)-1),  # 0: BS
        random.randint(0, len(SEQ_OPTS)-1),    # 1: SEQ
        random.randint(0, len(POWER_OPTS)-1),  # 2: PWR
        random.randint(0, len(PREC_OPTS)-1),   # 3: PREC
        random.randint(0, len(BLOCK_OPTS)-1),  # 4: BLOCK
        random.randint(0, len(TILE_OPTS)-1),   # 5: TILE
        random.randint(0, len(FUSED_OPTS)-1),  # 6: FUSED
        random.randint(0, len(ATTN_OPTS)-1)    # 7: ATTN
    ]

toolbox.register("individual", tools.initIterate, creator.Individual, get_random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# =============================================================================
# 4. EVALUATION FUNCTION
# =============================================================================
def evaluate(ind):
    # Decode Genome
    p = {
        'bs': BATCH_OPTS[ind[0]],
        'seq': SEQ_OPTS[ind[1]],
        'pwr': POWER_OPTS[ind[2]],
        'prec': PREC_OPTS[ind[3]],
        'block': BLOCK_OPTS[ind[4]],
        'tile': TILE_OPTS[ind[5]],
        'fused': FUSED_OPTS[ind[6]],
        'attn': ATTN_OPTS[ind[7]]
    }

    # Construct CLI Command
    # MUST MATCH baseline_benchmark_llama_v2.py args exactly!
    cmd = [
        sys.executable, BENCHMARK_SCRIPT,
        "--model_name", TARGET_MODEL,
        "--batch_size", str(p['bs']),
        "--seq_length", str(p['seq']),
        "--power_limit", str(p['pwr']),
        "--precision", p['prec'],
        
        # ==== CORRECTED FLAGS BELOW ====
        "--block_size", str(p['block']),
        "--tile_cfg", p['tile'],       # FIXED: matched parser arg
        "--fused_kernel", p['fused'],  
        "--attn_kernel", p['attn']     # FIXED: matched parser arg
    ]

    try:
        # Run Benchmark (Timeout slightly higher for complex compilations)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Parse Output
        output_data = {}
        for line in result.stdout.split('\n'):
            if line.strip().startswith("__EA_RESULT__:"):
                json_str = line.replace("__EA_RESULT__:", "")
                output_data = json.loads(json_str)
                break
        
        if not output_data:
            # Fallback if no JSON found (Silent Fail)
            log_trial(p, {'status': 'silent_fail', 'error': result.stderr[:200]})
            return 0, 700 # Return worst fitness

        if output_data.get('status') == 'failed':
            log_trial(p, output_data)
            return 0, 700
        
        # Success!
        tput = output_data.get('throughput', 0)
        pwr_used = p['pwr'] # Ideally we'd measure real power, but using limit for now
        
        log_trial(p, output_data)
        
        # Fitness = (Throughput, Power) -> Multi-objective
        return tput, pwr_used

    except subprocess.TimeoutExpired:
        log_trial(p, {'status': 'timeout', 'error': '300s limit'})
        return 0, 700
    except Exception as e:
        log_trial(p, {'status': 'exception', 'error': str(e)})
        return 0, 700

# =============================================================================
# 5. GENETIC OPERATORS
# =============================================================================
def mutate_smart(individual, indpb):
    """Custom mutation that knows the bounds of each gene."""
    ranges = [
        len(BATCH_OPTS), len(SEQ_OPTS), len(POWER_OPTS), len(PREC_OPTS),
        len(BLOCK_OPTS), len(TILE_OPTS), len(FUSED_OPTS), len(ATTN_OPTS)
    ]
    for i in range(len(individual)):
        if random.random() < indpb:
            # Mutate to a random valid index for this specific gene
            individual[i] = random.randint(0, ranges[i]-1)
    return individual,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_smart, indpb=0.2)
toolbox.register("select", tools.selNSGA2) # NSGA-II is great for multi-objective (Speed vs Power)

# =============================================================================
# 6. MAIN LOOP
# =============================================================================
def run_expanded_optimization():
    init_log_file()
    print("🚀 Starting Expanded Kernel+Power Evolutionary Optimization...")
    print(f"   Search Space Dimensions: {len(get_random_individual())}")
    
    # Hall of Fame
    hof = tools.HallOfFame(5)
    
    # Population
    pop = toolbox.population(n=16) # Larger population for larger search space
    
    # Stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max_tput", lambda fits: max(f[0] for f in fits))
    stats.register("min_pwr", lambda fits: min(f[1] for f in fits))
    
    # Run Evolution (5 Generations)
    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=16, lambda_=16, 
        cxpb=0.6, mutpb=0.3, ngen=5, 
        stats=stats, halloffame=hof, verbose=True
    )
    
    print("\n🏆 TOP CONFIGURATIONS 🏆")
    print(f"{'Prec':<5} {'BS':<3} {'Seq':<4} {'Pwr':<4} | {'Block':<5} {'Tile':<10} {'Fuse':<10} {'Attn':<8} | {'Tput (tok/s)'}")
    print("-" * 90)
    
    for ind in hof:
        p = {
            'bs': BATCH_OPTS[ind[0]], 'seq': SEQ_OPTS[ind[1]], 
            'pwr': POWER_OPTS[ind[2]], 'prec': PREC_OPTS[ind[3]],
            'block': BLOCK_OPTS[ind[4]], 'tile': TILE_OPTS[ind[5]], 
            'fused': FUSED_OPTS[ind[6]], 'attn': ATTN_OPTS[ind[7]]
        }
        tput = ind.fitness.values[0]
        print(f"{p['prec']:<5} {p['bs']:<3} {p['seq']:<4} {p['pwr']:<4} | {p['block']:<5} {p['tile']:<10} {p['fused']:<10} {p['attn']:<8} | {tput:,.0f}")

if __name__ == "__main__":
    run_expanded_optimization()