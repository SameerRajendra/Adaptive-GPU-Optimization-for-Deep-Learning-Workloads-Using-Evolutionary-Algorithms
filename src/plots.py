import pandas as pd, matplotlib.pyplot as plt
df = pd.read_excel('results/ea_optimizer_results_four_parameters.xlsx')
for p in df['precision'].unique():
    subset = df[df['precision']==p]
    plt.scatter(subset['train_time'], subset['test_accuracy'], label=p)
plt.xlabel('Training time (s)')
plt.ylabel('Test accuracy (%)')
plt.legend()
plt.grid(True)
plt.savefig('pareto_frontier.png', dpi=300)
