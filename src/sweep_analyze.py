import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('sweep_results.csv')

# Save full CSV (already done by sweep_runner) and show sample
print("Sweep Data:\n", df.head())

# Plot 1: Test Accuracy vs Batch Size (color by power limit)
plt.figure(figsize=(8,6))
for pl in sorted(df['power_limit'].unique()):
    subset = df[df['power_limit'] == pl]
    plt.plot(subset['batch_size'], subset['test_accuracy'], marker='o', label=f'Power {pl}W')
plt.title("Test Accuracy vs Batch Size (by Power Limit)")
plt.xlabel("Batch Size")
plt.ylabel("Test Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("test_accuracy_vs_batch_size.png")
plt.close()

# Plot 2: Training Time vs Batch Size (color by power limit)
plt.figure(figsize=(8,6))
for pl in sorted(df['power_limit'].unique()):
    subset = df[df['power_limit'] == pl]
    plt.plot(subset['batch_size'], subset['train_time'], marker='o', label=f'Power {pl}W')
plt.title("Training Time vs Batch Size (by Power Limit)")
plt.xlabel("Batch Size")
plt.ylabel("Training Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("train_time_vs_batch_size.png")
plt.close()

# Plot 3: Test Accuracy vs Power Limit (color by batch size)
plt.figure(figsize=(8,6))
for bs in sorted(df['batch_size'].unique()):
    subset = df[df['batch_size'] == bs]
    plt.plot(subset['power_limit'], subset['test_accuracy'], marker='o', label=f'Batch Size {bs}')
plt.title("Test Accuracy vs Power Limit (by Batch Size)")
plt.xlabel("Power Limit (W)")
plt.ylabel("Test Accuracy (%)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("test_accuracy_vs_power_limit.png")
plt.close()

# Plot 4: Training time vs Power Limit (color by batch size)
plt.figure(figsize=(8,6))
for bs in sorted(df['batch_size'].unique()):
    subset = df[df['batch_size'] == bs]
    plt.plot(subset['power_limit'], subset['train_time'], marker='o', label=f'Batch Size {bs}')
plt.title("Training Time vs Power Limit (by Batch Size)")
plt.xlabel("Power Limit (W)")
plt.ylabel("Training Time (s)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Training Time_vs_power_limit.png")
plt.close()

print("Saved plots as PNG images. You can view them for analysis and reporting.")
