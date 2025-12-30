import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Llama-70B', 'GPT-OSS-120B', 'Nemotron-8B']
methods = ['DP', 'TP', 'Ours', 'Shift-Parallel']

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

# Additional data for Peak Prompt throughput, TTFT, and ILT
models_new = ['Llama-70B-8K', 'GPT-OSS-120B-128K', 'Nemotron-8B-1M']
methods_new = ['Ours', 'TP', 'DP']
peak_prompt = {
    'Llama-70B-8K': [32000, 24727, 32000],
    'GPT-OSS-120B-128K': [33644, 24436, 33744],
    'Nemotron-8B-1M': [74473, 74473, 74472],
}
ttft = {
    'Llama-70B-8K': [361, 357.25, 1061.12],
    'GPT-OSS-120B-128K': [561, 557.25, 1561.12],
    'Nemotron-8B-1M': [1970.74, 1969.74, 5999.86],
}
ilt = {
    'Llama-70B-8K': [190, 181, 352],
    'GPT-OSS-120B-128K': [263.21, 267.06, 469.21],
    'Nemotron-8B-1M': [463.21, 467.06, 869.21],
}

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Set up bar positions
x = np.arange(len(models))
width = 0.2  # Width of bars
positions = [x - 1.5*width, x - 0.5*width, x + 0.5*width, x + 1.5*width]

# Colors for each method
colors = {'Ours': '#fee255', 'TP': '#d697f9', 'DP': '#4887f0', 'Shift-Parallel': '#38c1a8'}
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

ax1.set_xlabel('(a) Median TPOT(ms)', fontsize=font_size, fontweight='bold')
# ax1.set_ylabel('Median TPOT (ms)', fontsize=font_size)
# ax1.set_title('Median TPOT(ms)', fontsize=font_size, fontweight='bold')
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

ax2.set_xlabel('(b) Peak Generation Throughput(tokens/s)', fontsize=font_size, fontweight='bold')
# ax2.set_ylabel('Peak Generation Throughput (tokens/s)', fontsize=font_size)
# ax2.set_title('Peak Generation Throughput(tokens/s)', fontsize=font_size, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=font_size, ha='center')
ax2.legend(loc='best', fontsize=font_size)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_yticklabels([0, '5k', '10k', '15k', '20k', '25k'], fontsize=font_size)

fig.tight_layout()
fig.savefig('z_tests/comparison_plot.pdf', dpi=300, bbox_inches='tight')

# ------------------------------------------------------------

models_new_labels = ['Llama-70B\n8K', 'GPT-OSS\n128K', 'Nemotron\n1M']
font_size = font_size+4

# New figure with three subplots for additional metrics
fig2, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(20, 6))
x_new = np.arange(len(models_new))
width_new = 0.25
positions_new = [x_new - width_new, x_new, x_new + width_new]
colors_new = colors

# Peak Prompt throughput
for i, method in enumerate(methods_new):
    values = [peak_prompt[model][i] for model in models_new]
    ax3.bar(positions_new[i], values, width_new, label=method, color=colors_new[method], alpha=0.8)
# ax3.set_title('Peak Prompt Throughput(tokens/s)', fontsize=font_size, fontweight='bold')
ax3.set_xlabel('(a) Peak Prompt Throughput(tokens/s)', fontsize=font_size, fontweight='bold')
ax3.set_xticks(x_new)
ax3.set_xticklabels(models_new_labels, fontsize=font_size, ha='center')
ax3.legend(loc='best', fontsize=font_size)
ax3.set_yticklabels([0, '10k', '20k', '30k', '40k', '50k', '60k', '70k', '80k'], fontsize=font_size)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# TTFT
for i, method in enumerate(methods_new):
    values = [ttft[model][i] for model in models_new]
    ax4.bar(positions_new[i], values, width_new, label=method, color=colors_new[method], alpha=0.8)
# ax4.set_title('TTFT(ms)', fontsize=font_size, fontweight='bold')
ax4.set_xlabel('(b) TTFT(ms)', fontsize=font_size, fontweight='bold')
ax4.set_xticks(x_new)
ax4.set_xticklabels(models_new_labels, fontsize=font_size, ha='center')
ax4.set_yticklabels([0, '1k', '2k', '3k', '4k', '5k', '6k'], fontsize=font_size)
ax4.legend(loc='best', fontsize=font_size)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# ILT
for i, method in enumerate(methods_new):
    values = [ilt[model][i] for model in models_new]
    ax5.bar(positions_new[i], values, width_new, label=method, color=colors_new[method], alpha=0.8)
# ax5.set_title('ILT(ms)', fontsize=font_size, fontweight='bold')
ax5.set_xlabel('(c) ILT(ms)', fontsize=font_size, fontweight='bold')
ax5.set_xticks(x_new)
ax5.set_xticklabels(models_new_labels, fontsize=font_size, ha='center')
ax5.set_yticklabels([0, '200', '400', '600', '800'], fontsize=font_size)
ax5.legend(loc='best', fontsize=font_size)
ax5.grid(axis='y', alpha=0.3, linestyle='--')

fig2.tight_layout()
fig2.savefig('z_tests/additional_comparison_plot.pdf', dpi=300, bbox_inches='tight')
plt.show()


