

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from experiments.utils import load_interaction_matrices


def frequency_analysis(I):
    eigenvalues, eigenvectors = np.linalg.eigh(I)

    k = np.sort(eigenvalues)
    plt.plot(k, marker='o', color='red')
    # plt.title("Eigenvalue Spectrum of I")
    plt.xlabel("Index", fontsize=16, labelpad=4)
    plt.ylabel("Eigenvalue", fontsize=16, labelpad=4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.show()

    # Step 3: Visualize Graph
    # G = nx.from_numpy_array(I)
    # nx.draw_spring(G, node_size=100, node_color='lightblue', edge_color='gray')
    # plt.title("Graph from I")
    # plt.show()

    # Step 4: Plot Top Eigenvectors (Frequency Modes)
    top_k = 3
    fig, axs = plt.subplots(1, top_k, figsize=(18, 4))
    for i in range(top_k):
        axs[i].bar(range(len(I)), eigenvectors[:, -1 - i],edgecolor='black', linewidth=1.0)
        axs[i].set_title(f"Eigenvector {i + 1}", fontsize=16)
    plt.tight_layout()
    plt.show()


def main():
    int1, int2, int3, int4, int_out = load_interaction_matrices('dataset/interaction_matrix', 'InterGAT-15min')
    frequency_analysis(int_out)



if __name__ == "__main__":
    main()