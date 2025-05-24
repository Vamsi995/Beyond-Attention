import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch


def load_interaction_matrices(path, prefix):

    att1_intermatrix = torch.load(f"{path}/att1_intermatrix_{prefix}.pt",
                                  map_location=torch.device('cpu'))
    att2_intermatrix = torch.load(f"{path}/att2_intermatrix_{prefix}.pt",
                                  map_location=torch.device('cpu'))
    att3_intermatrix = torch.load(f"{path}/att3_intermatrix_{prefix}.pt",
                                  map_location=torch.device('cpu'))
    att4_intermatrix = torch.load(f"{path}/att4_intermatrix_{prefix}.pt",
                                  map_location=torch.device('cpu'))
    attout_intermatrix = torch.load(f"{path}/attout_intermatrix_{prefix}.pt",
                                    map_location=torch.device('cpu'))

    return att1_intermatrix, att2_intermatrix, att3_intermatrix, att4_intermatrix, attout_intermatrix



def plot_adjacency(adj, title="Adjacency Matrix", cmap="Blues", save_path=None):

    if isinstance(adj, torch.Tensor):
        adj = adj.detach().cpu().numpy()
    plt.figure(figsize=(6, 5))
    sns.heatmap(adj, cmap=cmap,
                square=True, cbar=True)
    plt.title(title, fontsize=14)
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def binarize_top_k_percent(I, topk_percent=5.0):
    """
    Keeps top-k% of entries as 1, rest as 0.
    I: torch.Tensor (N, N)
    topk_percent: float (e.g., 5.0 means top 5% retained)
    Returns: torch.Tensor (N, N) binary
    """
    I = torch.tensor(I)
    I_flat = I.flatten()
    k = int(len(I_flat) * (topk_percent / 100))

    # Get the threshold value at k-th largest
    threshold = torch.topk(I_flat, k).values.min()

    # Binarize
    I_bin = (I >= threshold).float()
    return I_bin.detach().numpy()
