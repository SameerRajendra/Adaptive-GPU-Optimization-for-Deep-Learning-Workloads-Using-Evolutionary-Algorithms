import subprocess
import itertools
import csv
import os

gpu_index = 0  # Set this to 0 or 1 depending on which GPU you want to sweep
batch_sizes = [32, 64, 128]
power_limits = [200, 350, 500, 650, 800]
results_file = "sweep_results.csv"

if os.path.exists(results_file):
    os.remove(results_file)

with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["gpu_index", "batch_size", "power_limit", "epoch_accuracy", "test_accuracy", "train_time"])

for batch_size, power_limit in itertools.product(batch_sizes, power_limits):
    print(f"Setting power limit to {power_limit}W for GPU {gpu_index}")
    try:
        subprocess.run([
            "nvidia-smi", "-i", str(gpu_index), "-pl", str(power_limit)
        ], check=True)
    except subprocess.CalledProcessError:
        print(f"Power limit set failed for GPU {gpu_index}, limit {power_limit}W (skip).")
        continue

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    print(f"Running baseline_resnet.py with batch_size={batch_size}, power_limit={power_limit}")
    subprocess.run([
        "python", "workload_runner.py",
        f"--batch_size={batch_size}"
    ], check=True, env=env)

    with open("baseline_results.txt") as f:
        epoch_acc, test_acc, train_time = None, None, None
        for line in f:
            if "Epoch Accuracy" in line:
                epoch_acc = float(line.split(":")[1].replace("%", "").strip())
            elif "Test Accuracy" in line:
                test_acc = float(line.split(":")[1].replace("%", "").strip())
            elif "Training Time" in line:
                train_time = float(line.split(":")[1].replace("seconds", "").strip())

    with open(results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([gpu_index, batch_size, power_limit, epoch_acc, test_acc, train_time])

    print(f"Sweeped: GPU={gpu_index}, batch_size={batch_size}, power_limit={power_limit}, epoch_acc={epoch_acc}, test_acc={test_acc}, train_time={train_time}")

print("All sweeps complete. See sweep_results.csv for results.")