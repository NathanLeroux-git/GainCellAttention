import matplotlib.pyplot as plt
import numpy as np

# # Generating sample data for two tensors (replace these with your actual tensors)
# n_channels = 10
# n_time_steps = 20
# preds = np.random.rand(n_channels, n_time_steps)  # Example data for tensor 1
# target_trans_probs = np.random.rand(n_channels, n_time_steps)  # Example data for tensor 2

def plot_sequences(preds, target_trans_probs, n_channels=10, out_file_name='./transition_probs_sequences.png'):

    # Determine common color limits for both tensors
    common_vmin = 0
    common_vmax = target_trans_probs[target_trans_probs!=0].max()
    cmap='magma'
    
    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plotting the first tensor in the first subplot
    cax1 = axs[0].imshow(preds, cmap=cmap, aspect='auto', vmin=common_vmin, vmax=common_vmax)
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n_channels]
    axs[0].set_yticks(np.arange(n_channels))
    axs[0].set_yticklabels(list(alphabet))
    axs[0].set_xlabel('Time Steps')
    axs[0].set_ylabel('Channels')
    axs[0].set_title('Predictied Transition Probability')
    cbar1 = fig.colorbar(cax1, ax=axs[0])
    cbar1.set_label('Intensity')

    # Plotting the second tensor in the second subplot
    cax2 = axs[1].imshow(target_trans_probs, cmap=cmap, aspect='auto', vmin=common_vmin, vmax=common_vmax)
    axs[1].set_yticks(np.arange(n_channels))
    axs[1].set_yticklabels(list(alphabet))
    axs[1].set_xlabel('Time Steps')
    axs[1].set_ylabel('Channels')
    axs[1].set_title('Target Transition Probabilities')
    cbar2 = fig.colorbar(cax2, ax=axs[1])
    cbar2.set_label('Intensity')

    # Adjust layout
    plt.tight_layout()

    # Save the figure as a PNG file
    plt.savefig(out_file_name, dpi=300)  # Change the file name as needed

    # # Show the plot
    # plt.show()