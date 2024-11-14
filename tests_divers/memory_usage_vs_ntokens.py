import matplotlib.pyplot as plt
import numpy as np
from math import nan

x = [
    # 2 ** np.arange(0, 8)*64,
     2 ** np.arange(0, 9)*64,
     ]

# Without checkpointing
# Sliding Window Attention
y1 = [
    # [0.79,0.79,0.80,0.90,1.25,2.56,7.75,28.51],
    [0.78,0.78,0.79,0.85,0.99,1.64,3.64,12.16,46.08]
               ]
# Wrapped Window Attention
y2 = [
    # [1.12,1.49,2.22,3.72,6.69,12.63,24.45,48.17],
    [0.91,1.08,1.29,2.05,3.36,5.97,11.04,21.55,42.30]
               ]

# Wrapped Window Attention
y3 = [
    # [1,1.27,1.67,2.83,4.89,9.04,17.27,33.76],
    [1,1.27,1.67,2.83,4.89,9.04,17.27,33.76, nan]
               ]

# Create a figure and axis
fig, axes = plt.subplots(1)

axes = [axes]

for i, ax in enumerate(axes):
    # Plot the curves with logarithmic axes
    ax.plot(x[i], y1[i], 'o-', color="darkblue", label='Full Attention')
    ax.plot(x[i], y2[i], 'o-', color="darkred", label='Sliding Window Attention')
    ax.plot(x[i], y3[i], 'o-', color="darkgreen", label='CAM Sliding Window Attention')

    # Set x-axis and y-axis to logarithmic scale with base 2
    # ax.set_xscale('log', base=2)
    # ax.set_yscale('log', base=2)

    ax.set_xticks(x[-1])

    # Set axis labels as placeholders
    ax.set_xlabel('# tokens')
    ax.set_ylabel('Memory usage (Gb x 4)')

    # Add legend
    ax.legend()

axes[0].set_title('Batch size = 4, Depth = 4, D = 128, n_heads = 8')
# Show the plot
plt.savefig('plots/memory_usage_vs_ntokens.png', format='png')
# plt.show()
