import os
import random
import subprocess
import csv
from deap import base, creator, tools, algorithms

# Define search space
BATCH_SIZES = [32, 64, 128]
POWER_LIMITS = [200, 300, 400, 500, 600, 700]
MEM_CLOCKS = [0, 100, 200, 300, 400]  # Memory MHz offset
PRECISIONS = ["fp32", "fp16"]

GPU_INDEX = 0  # GPU to target

RESULTS_FILE = "ea_optimizer_results.csv"
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        ## FIXED 4: Added new parameters to CSV header
        writer.writerow(["gpu_index", "batch_size", "power_limit", "mem_clock", "precision", "test_accuracy", "train_time"])

# Multi-objective: maximize accuracy, minimize time
creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("batch", random.choice, BATCH_SIZES)
toolbox.register("power", random.choice, POWER_LIMITS)
toolbox.register("mem", random.choice, MEM_CLOCKS)
toolbox.register("precision", random.choice, PRECISIONS)

toolbox.register("individual", tools.initCycle, creator.Individual,
    (toolbox.batch, toolbox.power, toolbox.mem, toolbox.precision), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("map", map)

def mutate_configuration(individual):
   
    # Choose which gene to mutate: 0, 1, 2, or 3
    gene_to_mutate = random.randint(0, len(individual) - 1)

    if gene_to_mutate == 0:
        # Mutate batch size
        individual[0] = random.choice(BATCH_SIZES)
    elif gene_to_mutate == 1:
        # Mutate power limit
        individual[1] = random.choice(POWER_LIMITS)
    elif gene_to_mutate == 2:
        # Mutate memory clock
        individual[2] = random.choice(MEM_CLOCKS)
    else: # gene_to_mutate == 3
        # Mutate precision
        individual[3] = random.choice(PRECISIONS)

    return individual,

def evaluate(ind):
    #Unpack all four values from the individual
    batch_size, power_limit, mem_clock, precision = ind

    ## Use the new parameters
    # Set GPU power limit
    subprocess.run(["nvidia-smi", "-i", str(GPU_INDEX), "-pl", str(power_limit)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Set GPU memory clock offset
    subprocess.run(["nvidia-smi", "-i", str(GPU_INDEX), "-lmc", str(mem_clock)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Run training, passing the new arguments
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_INDEX)
    command = [
        "python", "workload_runner.py",
        f"--batch_size={batch_size}",
        f"--precision={precision}"
    ]
    subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL)

    # Parse results
    test_acc = train_time = None
    with open("baseline_results.txt") as f:
        for line in f:
            if "Test Accuracy" in line:
                test_acc = float(line.split(":")[1].replace("%","").strip())
            elif "Training Time" in line:
                train_time = float(line.split(":")[1].replace("seconds","").strip())
    
    # Save all data to CSV
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([GPU_INDEX, batch_size, power_limit, mem_clock, precision, test_acc, train_time])
        
    return test_acc, train_time

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_configuration)
toolbox.register("select", tools.selNSGA2)


def main():
    pop = toolbox.population(n=10)
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    
    stats.register("avg", lambda fits: (sum(f[0] for f in fits)/len(fits), sum(f[1] for f in fits)/len(fits)))
    stats.register("max_acc_min_time", lambda fits: (max(f[0] for f in fits), min(f[1] for f in fits)))
    
    # Create a Logbook to store the statistics
    logbook = tools.Logbook()
    logbook.header = "gen", "nevals", "avg", "max_acc_min_time"

    #Capture the output of the algorithm
    pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=10, lambda_=10, cxpb=0.6, mutpb=0.3,
                                             ngen=5, stats=stats, halloffame=hof, verbose=True)

    # Save the logbook to a CSV file
    log_df = pd.DataFrame(logbook)
    log_df.to_csv("ea_generation_log.csv", index=False)
    print("\nEvolution log saved to ea_generation_log.csv")

    print("\nTop Configurations:")
    for ind in hof:
        print(
            f"Batch: {ind[0]}, Power: {ind[1]}W, MemClock: +{ind[2]}MHz, Precision: {ind[3]} -> "
            f"TestAcc={ind.fitness.values[0]:.2f}%, TrainTime={ind.fitness.values[1]:.2f}s"
        )

if __name__ == "__main__":
    main()