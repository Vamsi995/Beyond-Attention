import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import SpectralClustering
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import pandas as pd
from experiments.utils import load_interaction_matrices

def spectral_clustering_communities(adj_matrix, k, normalized=True):
    """
    Custom spectral clustering using eigen decomposition and KMeans.

    Parameters:
        adj_matrix (np.ndarray): Symmetric NxN adjacency or similarity matrix.
        k (int): Number of communities.
        normalized (bool): Whether to use normalized Laplacian (recommended).

    Returns:
        labels (np.ndarray): Cluster assignment for each node.
    """
    # Degree matrix
    degrees = np.sum(adj_matrix, axis=1)

    # Construct Laplacian
    D = np.diag(degrees)
    L = D - adj_matrix

    if normalized:
        # Normalized Laplacian: L_sym = D^(-1/2) L D^(-1/2)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees + 1e-10))
        L = D_inv_sqrt @ L @ D_inv_sqrt

    # Compute the bottom-k eigenvectors (excluding zero eigenvalues)
    # 'which="SM"' selects smallest magnitude eigenvalues
    eigvals, eigvecs = eigsh(L, k=k, which='SM')

    # Normalize rows (for better k-means stability)
    X = eigvecs / (np.linalg.norm(eigvecs, axis=1, keepdims=True) + 1e-10)

    # Cluster using KMeans on eigenvector space
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)

    return labels


def extract_intra_inter_values(I, communities):
    N = I.shape[0]
    intra_vals, inter_vals = [], []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if communities[i] == communities[j]:
                intra_vals.append(I[i, j])
            else:
                inter_vals.append(I[i, j])
    return np.array(intra_vals), np.array(inter_vals)

def analyze_and_plot_individually(interaction_matrices, adj_matrix, k=4):
    # Compute communities once
    communities = spectral_clustering_communities(adj_matrix, k)

    for name, I in interaction_matrices.items():
        I = I.detach().cpu().numpy() if hasattr(I, 'detach') else I
        intra_vals, inter_vals = extract_intra_inter_values(I, communities)

        # Create boxplot for this matrix
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=[intra_vals, inter_vals])
        plt.xticks([0, 1], ['Intra-Community', 'Inter-Community'])
        plt.title(f"{name} — Interaction Score Distribution")
        plt.ylabel("I(i,j) value")
        plt.tight_layout()
        plt.show()

        # Optional: Print stats
        print(f"--- {name} ---")
        print(f"Intra Mean: {intra_vals.mean():.4f}, Var: {intra_vals.var():.4f}")
        print(f"Inter Mean: {inter_vals.mean():.4f}, Var: {inter_vals.var():.4f}\n")

    return communities


def extract_intra_inter_stats(I, communities):
    N = I.shape[0]
    intra_vals, inter_vals = [], []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # skip diagonal
            if communities[i] == communities[j]:
                intra_vals.append(I[i, j])
            else:
                inter_vals.append(I[i, j])

    intra_vals_lis = np.array(intra_vals)
    inter_vals_lis = np.array(inter_vals)

    return {
        "Intra Mean": intra_vals_lis.mean(),
        # "Intra Std": intra_vals_lis.std(),
        "Inter Mean": inter_vals_lis.mean(),
        # "Inter Std": inter_vals_lis.std()
    }


def plot_multiple_stats_vs_k(interaction_matrices, adj_matrix, k_range):
    """
    For each interaction matrix, plot how intra/inter mean and variance change with number of clusters (k),
    and show all plots in a grid figure.
    """
    num_matrices = len(interaction_matrices)
    fig, axes = plt.subplots(num_matrices, 1, figsize=(10, 5 * num_matrices), sharex=True)

    if num_matrices == 1:
        axes = [axes]

    for ax, (name, I) in zip(axes, interaction_matrices.items()):
        stats_per_k = {"k": [], "Metric": [], "Value": []}
        for k in k_range:
            communities = spectral_clustering_communities(adj_matrix, k)
            I = I.detach().cpu().numpy() if hasattr(I, 'detach') else I
            result = extract_intra_inter_stats(I, communities)
            for metric, value in result.items():
                stats_per_k["k"].append(k)
                stats_per_k["Metric"].append(metric)
                stats_per_k["Value"].append(value)

        df = pd.DataFrame(stats_per_k)
        sns.lineplot(data=df, x="k", y="Value", hue="Metric", marker="o", ax=ax)
        ax.set_title(f"{name} — Stats vs Number of Communities (k)")
        ax.set_xlabel("Number of Clusters (k)")
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_each_stat_separately(interaction_matrices, adj_matrix, k_range):
    """
    For each interaction matrix, generate and display a separate plot for intra/inter mean and variance
    over different values of k using spectral clustering.
    """
    for name, I in interaction_matrices.items():
        stats_per_k = {"k": [], "Metric": [], "Value": []}
        for k in k_range:
            communities = spectral_clustering_communities(adj_matrix, k)
            I_np = I.detach().cpu().numpy() if hasattr(I, 'detach') else I
            result = extract_intra_inter_stats(I_np, communities)
            for metric, value in result.items():
                stats_per_k["k"].append(k)
                stats_per_k["Metric"].append(metric)
                stats_per_k["Value"].append(value)

        df = pd.DataFrame(stats_per_k)


        # Plot each head individually
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=df, x="k", y="Value", hue="Metric", marker="o")
        plt.title(f"{name} — Stats vs Number of Communities (k)")
        plt.xlabel("Number of Clusters (k)")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend(loc="best", fontsize='large')
        plt.tight_layout()
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.show()


def main():

    int1, int2, int3, int4, int_out = load_interaction_matrices('dataset/interaction_matrix', 'InterGAT-15min')

    interaction_matrices = {
        "Interaction 1": int1,
        "Interaction 2": int2,
        "Interaction 3": int3,
        "Interaction 4": int4,
        "Interaction Aggregate": int_out
    }

    plot_each_stat_separately(interaction_matrices, adj, range(2, 32))


if __name__ == "__main__":
    main()