import matplotlib.pyplot as plt
import numpy as np

# Data for the plot
methods = ['Logistic Regression', 'Random Forest', 'SVM', 'Proposed Method']
metrics = ['Accuracy', 'NMI', 'Precision', 'Recall', 'R2']
data = np.array([
    [94.44, 96.06, 96.36, 95.00, 63.13],  # Logistic Regression
    [95.83, 94.38, 96.50, 96.08, 87.59],  # Random Forest
    [91.67, 92.55, 92.61, 92.57, 62.19],  # SVM
    [97.22, 96.44, 97.57, 97.22, 90.17]   # Proposed Method
])

# Bar width and positions
bar_width = 0.15
x = np.arange(len(metrics))

# Create figure and axis for plotting
fig, ax = plt.subplots(figsize=(12, 6))

# Grayscale color mapping (light to dark gray)
grayscale_colors = ['0.2', '0.4', '0.6', '0.8']

# Define hatch patterns for each method
patterns = ['/', '\\', '|', '-', '+']

# Plot bars with grayscale colors and hatch patterns
for i, method in enumerate(methods):
    bars = ax.bar(
        x + i * bar_width, data[i], bar_width,
        color=grayscale_colors[i % len(grayscale_colors)],
        edgecolor='black', label=method
    )
    for bar in bars:
        bar.set_hatch(patterns[i % len(patterns)])

# Set axis labels, title, and ticks
ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Scores (%)', fontsize=12)
ax.set_title('Comparison of Methods Across Different Metrics (Grayscale)', pad=20)
ax.set_xticks(x + bar_width * 1.5)
ax.set_xticklabels(metrics, fontsize=10)

# Display legend outside the plot
ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Methods", fontsize=10)

# Adjust layout for better visualization
plt.tight_layout()

# Show the plot
plt.show()