import torch
from experiments.utils import load_interaction_matrices, binarize_top_k_percent
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    int1, int2, int3, int4, int_out = load_interaction_matrices('dataset/interaction_matrix', 'InterGAT-15min')

    plot_multi_attention_masks([int1, int2, int3, int4, int_out], cmap='Reds')

def plot_multi_attention_masks(attn_list, titles=None, cmap="Blues", save_path=None):
    """
    Plot multiple attention/adjacency matrices side-by-side.

    Parameters:
        attn_list (list of np.ndarray or torch.Tensor): List of (N x N) attention matrices.
        titles (list of str): List of titles for each subplot. If None, defaults to ["Head 1", ..., "Output Attention"].
        cmap (str): Colormap for heatmap.
        save_path (str): Path to save the plot (optional).
    """
    num_heads = len(attn_list)
    if titles is None:
        titles = [f"Interaction Head {i+1}" for i in range(num_heads - 1)] + ["Interaction Aggregate"]

    plt.figure(figsize=(4 * num_heads, 4))

    for i, attn in enumerate(attn_list):
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()

        A = attn  # shape: (N, N)
        A = binarize_top_k_percent(A, 3.0)

        plt.subplot(1, num_heads, i + 1)
        sns.heatmap(A, cmap=cmap, square=True, cbar=True, xticklabels=30, yticklabels=30)
        plt.title(titles[i], fontsize=15)
        plt.xlabel("Node", fontsize=15, labelpad=10)
        plt.ylabel("Node", fontsize=15, labelpad=10)
        plt.subplots_adjust(wspace=1, hspace=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()



if __name__ == "__main__":
    main()
