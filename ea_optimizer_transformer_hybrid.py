import random
import subprocess
import json
import sys
import os
import csv
import time
from datetime import datetime
from deap import base, creator, tools, algorithms

# ==== CONFIGURATION ====
BENCHMARK_SCRIPT = "baseline_benchmark_using_triton.py" 
TRIALS_LOG_FILE = "whitebox_trials.csv"
CHAMPIONS_FILE = "whitebox_champions.json"

# ==== H100 SEARCH SPACE ====
BATCH_OPTS = [32, 64, 128] 
SEQ_OPTS = [512, 1024, 2048] 
POWER_OPTS = [500, 600, 700]
PREC_OPTS = ["fp16", "bf16", "fp32"] 

# ==== LOGGING SETUP ====
def init_log_file():
    """Creates the CSV file with headers if it doesn't exist."""
    if not os.path.exists(TRIALS_LOG_FILE):
        with open(TRIALS_LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "batch_size", "seq_length", "power_limit", 
                "precision", "throughput", "latency", "mfu", "status", "error_msg"
            ])
        print(f"[INFO] Created log file: {TRIALS_LOG_FILE}")

def log_trial(bs, seq, pwr, prec, tput, lat, mfu, status, err=""):
    """Appends a single experiment result to the CSV."""
    with open(TRIALS_LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            bs, seq, pwr, prec, 
            f"{tput:.2f}", f"{lat:.4f}", f"{mfu:.2f}", 
            status, err
        ])

# ==== EVOLUTIONARY SETUP ====
if "FitnessMulti" not in creator.__dict__:
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMulti)

toolbox = base.Toolbox()

def get_random_individual():
    return [
        random.randint(0, len(BATCH_OPTS)-1), 
        random.randint(0, len(SEQ_OPTS)-1),   
        random.randint(0, len(POWER_OPTS)-1), 
        random.randint(0, len(PREC_OPTS)-1)   
    ]

toolbox.register("individual", tools.initIterate, creator.Individual, get_random_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    # 1. DECODE
    bs = BATCH_OPTS[ind[0]]
    seq = SEQ_OPTS[ind[1]]
    pwr = POWER_OPTS[ind[2]]
    prec = PREC_OPTS[ind[3]]
    
    # 2. CHECK SCRIPT
    if not os.path.exists(BENCHMARK_SCRIPT):
        msg = f"Script {BENCHMARK_SCRIPT} not found"
        print(f"[ERROR] {msg}")
        log_trial(bs, seq, pwr, prec, 0, 0, 0, "failed", msg)
        return 0, 10000

    # 3. BUILD COMMAND
    cmd = [
        sys.executable, BENCHMARK_SCRIPT,
        "--batch_size", str(bs),
        "--seq_length", str(seq),
        "--power_limit", str(pwr),
        "--precision", prec
    ]
    
    try:
        # Run with timeout
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        
        # 4. CHECK FOR CRASHES
        if result.returncode != 0:
            err_sample = result.stderr[:200].replace('\n', ' ')
            print(f"\n[CRASH] BS={bs} Seq={seq} Prec={prec} -> {err_sample}")
            log_trial(bs, seq, pwr, prec, 0, 0, 0, "crashed", err_sample)
            return 0, 10000

        # 5. PARSE OUTPUT
        for line in result.stdout.split('\n'):
            if line.strip().startswith("__EA_RESULT__:"):
                try:
                    clean_line = line.strip().replace("__EA_RESULT__:", "")
                    data = json.loads(clean_line)
                    
                    if data.get('status') == 'failed':
                        log_trial(bs, seq, pwr, prec, 0, 0, 0, "failed", data.get('error', 'unknown'))
                        return 0, 10000 
                    
                    # SUCCESS LOG
                    tput = data.get('throughput', 0)
                    lat = data.get('latency', 0)
                    mfu = data.get('mfu', 0)
                    
                    log_trial(bs, seq, pwr, prec, tput, lat, mfu, "success")
                    return tput, lat

                except json.JSONDecodeError:
                    log_trial(bs, seq, pwr, prec, 0, 0, 0, "json_error", line)
                    return 0, 10000

        # Fallback
        log_trial(bs, seq, pwr, prec, 0, 0, 0, "silent_fail", "No output tag found")
        return 0, 10000

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] BS={bs} SEQ={seq}")
        log_trial(bs, seq, pwr, prec, 0, 0, 0, "timeout", "90s limit reached")
        return 0, 10000
    except Exception as e:
        print(f"[EXCEPTION] {e}")
        log_trial(bs, seq, pwr, prec, 0, 0, 0, "exception", str(e))
        return 0, 10000

# CUSTOM MUTATION
def mutate_indices(individual, indpb):
    if random.random() < indpb: individual[0] = random.randint(0, len(BATCH_OPTS)-1)
    if random.random() < indpb: individual[1] = random.randint(0, len(SEQ_OPTS)-1)
    if random.random() < indpb: individual[2] = random.randint(0, len(POWER_OPTS)-1)
    if random.random() < indpb: individual[3] = random.randint(0, len(PREC_OPTS)-1)
    return individual,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_indices, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def run_optimization():
    init_log_file()
    print(f"Starting Logging Optimization calling '{BENCHMARK_SCRIPT}'...")
    
    pop = toolbox.population(n=12)
    hof = tools.HallOfFame(5)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max_tput", lambda fits: max(f[0] for f in fits))
    
    pop, log = algorithms.eaMuPlusLambda(
        pop, toolbox, mu=12, lambda_=12, 
        cxpb=0.5, mutpb=0.2, ngen=5, 
        stats=stats, halloffame=hof, verbose=True
    )
    
    print("\n" + "="*50)
    print("FINAL CHAMPIONS (Saved to JSON)")
    print("="*50)
    
    champions = []
    for i, ind in enumerate(hof):
        bs = BATCH_OPTS[ind[0]]
        seq = SEQ_OPTS[ind[1]]
        pwr = POWER_OPTS[ind[2]]
        prec = PREC_OPTS[ind[3]]
        tput = ind.fitness.values[0]
        
        champ_data = {
            "rank": i+1,
            "precision": prec,
            "batch_size": bs,
            "seq_length": seq,
            "power_limit": pwr,
            "throughput": tput
        }
        champions.append(champ_data)
        print(f"[{i+1}] {prec.upper()} | BS:{bs:<3} | Seq:{seq:<4} | Pwr:{pwr}W -> {tput:,.0f} tok/s")

    # Save Champions
    with open(CHAMPIONS_FILE, 'w') as f:
        json.dump(champions, f, indent=4)
    print(f"\n[INFO] Full logs saved to: {TRIALS_LOG_FILE}")
    print(f"[INFO] Champions saved to: {CHAMPIONS_FILE}")

if __name__ == "__main__":
    run_optimization()