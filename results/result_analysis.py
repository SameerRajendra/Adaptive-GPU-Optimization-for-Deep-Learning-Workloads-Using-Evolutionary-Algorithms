import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

RESULTS_FILE = "ea_optimizer_results.csv"
BASELINE_FILE = "baseline_results.txt"

def load_baseline_results():
    """Loads the baseline results from the text file."""
    try:
        with open(BASELINE_FILE, 'r') as f:
            content = f.read()
            # Use regex to find floating point numbers in the specific lines
            accuracy_match = re.search(r"Test Accuracy: ([\d.]+)", content)
            time_match = re.search(r"Training Time: ([\d.]+)", content)

            if accuracy_match and time_match:
                accuracy = float(accuracy_match.group(1))
                time = float(time_match.group(1))
                return {"test_accuracy": accuracy, "train_time": time}
            else:
                print(f"Warning: Could not parse baseline results from '{BASELINE_FILE}'.")
                return None
    except FileNotFoundError:
        print(f"Warning: Baseline results file '{BASELINE_FILE}' not found.")
        return None

def plot_parameter_trends(df):
    """
    Creates line graphs to show the impact of each parameter on accuracy and time.
    """
    params = ['batch_size', 'power_limit', 'mem_clock', 'precision']
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Impact of Parameters on Performance', fontsize=20, weight='bold')
    axes = axes.flatten()

    for i, param in enumerate(params):
        ax = axes[i]
        
        # Use a pointplot for the categorical 'precision' parameter
        if param == 'precision':
            # Melt the dataframe to plot two different metrics with pointplot
            df_melted = df.melt(id_vars=[param], value_vars=['test_accuracy', 'train_time'], var_name='metric', value_name='value')
            sns.pointplot(data=df_melted, x=param, y='value', hue='metric', ax=ax, dodge=True, palette=['tab:blue', 'tab:red'])
            ax.set_title(f'Impact of {param.replace("_", " ").title()}')
            ax.set_xlabel(param.replace("_", " ").title())
            ax.set_ylabel('Average Value')
            ax.legend(title='Metric')
            ax.grid(True)
            continue

        # Group by the current parameter and calculate the mean for the metrics
        grouped = df.groupby(param)[['test_accuracy', 'train_time']].mean().reset_index()

        # Plot Test Accuracy
        color = 'tab:blue'
        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel('Avg Test Accuracy (%)', color=color)
        ax.plot(grouped[param], grouped['test_accuracy'], color=color, marker='o', label='Avg Accuracy')
        ax.tick_params(axis='y', labelcolor=color)
        
        # Create a second y-axis for Training Time
        ax2 = ax.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Avg Training Time (s)', color=color)
        ax2.plot(grouped[param], grouped['train_time'], color=color, marker='s', linestyle='--', label='Avg Time')
        ax2.tick_params(axis='y', labelcolor=color)
        
        ax.set_title(f'Impact of {param.replace("_", " ").title()}')
        ax.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    param_plot_filename = "parameter_trends_plot.png"
    plt.savefig(param_plot_filename)
    print(f"Parameter trends plot saved to '{param_plot_filename}'")


def analyze_results():
    """
    Loads optimization and baseline results, compares them, and creates plots.
    """
    # Load baseline and optimization data
    baseline = load_baseline_results()
    try:
        df = pd.read_csv(RESULTS_FILE)
    except FileNotFoundError:
        print(f"Error: The results file '{RESULTS_FILE}' was not found.")
        print("Please run the optimizer script first to generate the results.")
        return

    if df.empty:
        print("The results file is empty. No data to analyze.")
        return

    # --- Create the new parameter trends plot ---
    plot_parameter_trends(df.copy())


    # --- Find the Pareto Front ---
    pareto_front = []
    for index, row in df.iterrows():
        is_dominated = False
        for _, other_row in df.iterrows():
            if (other_row['test_accuracy'] >= row['test_accuracy'] and other_row['train_time'] <= row['train_time']) and \
               (other_row['test_accuracy'] > row['test_accuracy'] or other_row['train_time'] < row['train_time']):
                is_dominated = True
                break
        if not is_dominated:
            pareto_front.append(row)

    if not pareto_front:
        print("Could not determine the Pareto front.")
        return
        
    pareto_df = pd.DataFrame(pareto_front).sort_values(by='test_accuracy', ascending=False)

    # --- Print Comparison with Baseline ---
    print("\n--- Comparison of Top 3 Configurations vs. Baseline ---")
    if baseline:
        print(f"\nBaseline: Test Accuracy = {baseline['test_accuracy']:.2f}%, Training Time = {baseline['train_time']:.2f}s\n")
        
        top_3 = pareto_df.head(3)
        for i, (_, row) in enumerate(top_3.iterrows()):
            acc_improvement = ((row['test_accuracy'] - baseline['test_accuracy']) / baseline['test_accuracy']) * 100
            time_improvement = ((baseline['train_time'] - row['train_time']) / baseline['train_time']) * 100
            
            print(f"--- Top Configuration #{i+1} ---")
            print(f"  Config: Batch={int(row['batch_size'])}, Power={int(row['power_limit'])}W, MemClock=+{int(row['mem_clock'])}, Precision={row['precision']}")
            print(f"  Result: Test Accuracy = {row['test_accuracy']:.2f}%, Training Time = {row['train_time']:.2f}s")
            print(f"  Improvement: Accuracy {acc_improvement:+.2f}%, Time Reduction {time_improvement:+.2f}%")
            print("-" * 25)
    else:
        print("\nCould not load baseline results for comparison. Printing Pareto front only.")
        print(pareto_df.to_string(index=False))


    # --- Create the Visualization ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 9))

    sns.scatterplot(
        data=df, x='train_time', y='test_accuracy', hue='precision',
        size='batch_size', style='power_limit', palette={'fp32': 'skyblue', 'fp16': 'coral'},
        sizes=(50, 250), ax=ax, alpha=0.7
    )
    
    ax.plot(pareto_df['train_time'], pareto_df['test_accuracy'],
            marker='o', linestyle='-', color='black', alpha=0.8,
            label='Pareto Front', markersize=8, markerfacecolor='lime')
    
    if baseline:
        ax.scatter(baseline['train_time'], baseline['test_accuracy'],
                   marker='*', s=500, color='gold', edgecolor='black', zorder=5, label='Baseline')
        ax.text(baseline['train_time'], baseline['test_accuracy'] - 0.2, 'Baseline',
                ha='center', va='top', fontsize=12, weight='bold')

    # --- Formatting ---
    ax.set_title('Accuracy vs. Training Time Trade-offs', fontsize=18, weight='bold')
    ax.set_xlabel('Training Time (seconds)', fontsize=14)
    ax.set_ylabel('Test Accuracy (%)', fontsize=14)
    ax.legend(title='Parameters', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    
    output_filename = "optimization_results_plot.png"
    plt.savefig(output_filename)
    print(f"\nAnalysis complete. Scatter plot saved to '{output_filename}'")
    
    plt.show()


if __name__ == "__main__":
    analyze_results()

