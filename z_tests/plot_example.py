import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Llama-70B', 'GPT-OSS-120B', 'Nemotron-8B']
methods = ['DP', 'TP', 'Shift-Parallel', 'Ours']

# Median TPOT data
median_tpot = {
    'Llama-70B': [51, 20.8, 21.8, 22],
    'GPT-OSS-120B': [23, 16, 0, 18],
    'Nemotron-8B': [18, 13, 13.6, 13.8]
}

# Peak generation throughput data
peak_throughput = {
    'Llama-70B': [3169, 1509, 2505, 3059],
    'GPT-OSS-120B': [9994, 3880, 0, 9581],
    'Nemotron-8B': [24820, 9380, 20084, 23654]
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Set up bar positions
x = np.arange(len(models))
width = 0.2  # Width of bars
positions = [x - 1.5*width, x - 0.5*width, x + 0.5*width, x + 1.5*width]

# Colors for each method
colors = {'Ours': '#1f77b4', 'TP': '#ff7f0e', 'DP': '#2ca02c', 'Shift-Parallel': '#9467bd'}
font_size = 22

# Plot Median TPOT
for i, method in enumerate(methods):
    values = [median_tpot[model][i] for model in models]
    # Replace 0 with NaN to skip plotting (method not supported)
    values_plot = [v if v != 0 else np.nan for v in values]
    ax1.bar(positions[i], values_plot, width, label=method, color=colors[method], alpha=0.8)
    # Mark unsupported methods with red X
    for j, val in enumerate(values):
        if val == 0:
            ax1.plot(positions[i][j], 3, 'rx', markersize=15, markeredgewidth=3, zorder=10)

# ax1.set_xlabel('Model', fontsize=font_size)
# ax1.set_ylabel('Median TPOT (ms)', fontsize=font_size)
ax1.set_title('Median TPOT(ms)', fontsize=font_size, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=font_size, ha='center')
ax1.legend(loc='best', fontsize=font_size)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_yticklabels(np.arange(0, 55, 10), fontsize=font_size)

# Plot Peak Generation Throughput
for i, method in enumerate(methods):
    values = [peak_throughput[model][i] for model in models]
    # Replace 0 with NaN to skip plotting (method not supported)
    values_plot = [v if v != 0 else np.nan for v in values]
    ax2.bar(positions[i], values_plot, width, label=method, color=colors[method], alpha=0.8)
    # Mark unsupported methods with red X
    for j, val in enumerate(values):
        if val == 0:
            ax2.plot(positions[i][j], 1300, 'rx', markersize=15, markeredgewidth=3, zorder=10)

# ax2.set_xlabel('Model', fontsize=12)
# ax2.set_ylabel('Peak Generation Throughput (tokens/s)', fontsize=font_size)
ax2.set_title('Peak Generation Throughput(tokens/s)', fontsize=font_size, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=font_size, ha='center')
ax2.legend(loc='best', fontsize=font_size)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_yticklabels(np.arange(0, 25000, 5000), fontsize=font_size)

plt.tight_layout()
plt.savefig('z_tests/comparison_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()