import random
import subprocess
import json
import sys
import os
import csv
from datetime import datetime
from deap import base, creator, tools, algorithms

# ==== CONFIGURATION ====
BENCHMARK_SCRIPT = "baseline_benchmark_llama.py" 
TRIALS_LOG_FILE = "llama_trials.csv"
CHAMPIONS_FILE = "llama_champions.json"
# We use the NousResearch mirror to avoid HF Token requirements
TARGET_MODEL = "NousResearch/Llama-2-7b-hf" 

# ==== SEARCH SPACE (Llama-2-7B Specific) ====
# 7B Params = 14GB VRAM just for weights.
# We must use smaller batches than GPT-2.
BATCH_OPTS = [1, 2, 4, 8, 16, 32] 
SEQ_OPTS = [128, 256, 512, 1024]
POWER_OPTS = [500, 600, 700]
PREC_OPTS = ["bf16", "fp16"] 

# ==== LOGGING ====
def init_log_file():
    if not os.path.exists(TRIALS_LOG_FILE):
        with open(TRIALS_LOG_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["timestamp", "model", "bs", "seq", "pwr", "prec", "tput", "mfu", "status", "error"])

def log_trial(bs, seq, pwr, prec, tput, mfu, status, err=""):
    with open(TRIALS_LOG_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            datetime.now().strftime("%H:%M:%S"), "llama-2-7b",
            bs, seq, pwr, prec, f"{tput:.0f}", f"{mfu:.2f}", status, err
        ])

# ==== EVOLUTION ====
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
    bs = BATCH_OPTS[ind[0]]
    seq = SEQ_OPTS[ind[1]]
    pwr = POWER_OPTS[ind[2]]
    prec = PREC_OPTS[ind[3]]
    
    cmd = [
        sys.executable, BENCHMARK_SCRIPT,
        "--model_name", TARGET_MODEL,
        "--batch_size", str(bs),
        "--seq_length", str(seq),
        "--power_limit", str(pwr),
        "--precision", prec
    ]
    
    try:
        # Llama-2 takes longer to compile. Timeout increased to 300s (5 mins).
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        for line in result.stdout.split('\n'):
            if line.strip().startswith("__EA_RESULT__:"):
                data = json.loads(line.replace("__EA_RESULT__:", ""))
                if data.get('status') == 'failed':
                    log_trial(bs, seq, pwr, prec, 0, 0, "failed", data.get('error'))
                    return 0, 10000
                
                log_trial(bs, seq, pwr, prec, data['throughput'], data['mfu'], "success")
                return data['throughput'], data['latency']
        
        log_trial(bs, seq, pwr, prec, 0, 0, "silent", result.stderr[:200])
        return 0, 10000

    except subprocess.TimeoutExpired:
        log_trial(bs, seq, pwr, prec, 0, 0, "timeout", "300s limit")
        return 0, 10000
    except Exception as e:
        return 0, 10000

def mutate_indices(individual, indpb):
    ranges = [len(BATCH_OPTS), len(SEQ_OPTS), len(POWER_OPTS), len(PREC_OPTS)]
    for i in range(4):
        if random.random() < indpb:
            individual[i] = random.randint(0, ranges[i]-1)
    return individual,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_indices, indpb=0.2)
toolbox.register("select", tools.selNSGA2)

def run_optimization():
    init_log_file()
    print(f"Starting Llama-2-7B White-Box Optimization...")
    
    # Instantiate the HallOfFame object into a variable FIRST
    hof = tools.HallOfFame(3) 
    
    pop, log = algorithms.eaMuPlusLambda(
        toolbox.population(n=8), toolbox, mu=8, lambda_=8, 
        cxpb=0.5, mutpb=0.2, ngen=3, 
        stats=tools.Statistics(lambda ind: ind.fitness.values), 
        halloffame=hof, # Pass the variable
        verbose=True
    )
    
    print("\n" + "="*50)
    print("FINAL LLAMA-2 CHAMPIONS")
    print("="*50)
    
    # Iterate over the populated variable 'hof'
    for i, ind in enumerate(hof):
        bs = BATCH_OPTS[ind[0]]
        seq = SEQ_OPTS[ind[1]]
        pwr = POWER_OPTS[ind[2]]
        prec = PREC_OPTS[ind[3]]
        tput = ind.fitness.values[0]
        print(f"[{i+1}] {prec} | BS:{bs} | Seq:{seq} | Pwr:{pwr} -> {tput:,.0f} tok/s")

if __name__ == "__main__":
    run_optimization()