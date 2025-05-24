import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

def main(prefix):
    norm = os.path.join('dataset/interaction_matrix_norm', f"{prefix}_frob_norm.pt")
    sparsity = os.path.join('dataset/interaction_matrix_norm', f"{prefix}_sparsity.pt")

    with open(f'{norm}', 'rb') as f:
        loaded_dict = pickle.load(f)

    with open(f'{sparsity}', 'rb') as f:
        loaded_dict2 = pickle.load(f)


    epochs = 100
    epochs_range = range(1, epochs + 1)
    sns.set(style="whitegrid", font_scale=1.0)

    # Frobenius Norm
    plt.figure(figsize=(8, 5))
    for head_id, norms in loaded_dict.items():
        if head_id == 'out_att':
            plt.plot(epochs_range, norms, label=f'Interaction Aggregate')
            continue

        plt.plot(epochs_range, norms, label=f'Interaction {int(head_id) + 1}')
    plt.xlabel("Epoch", fontsize=15, labelpad=10)
    plt.ylabel("Frobenius Norm", fontsize=15, labelpad=10)
    plt.title("Interaction Matrix Norm (Per Head)", fontsize=15, pad=10)
    plt.legend(fontsize='large')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Sparsity
    plt.figure(figsize=(8, 5))
    for head_id, sparsities in loaded_dict2.items():
        if head_id == 'out_att':
            plt.plot(epochs_range, sparsities, label=f'Interaction Aggregate')
            continue
        plt.plot(epochs_range, sparsities, label=f'Interaction {int(head_id) + 1}')
    plt.xlabel("Epoch", fontsize=15, labelpad=10)
    plt.ylabel("Sparsity (Fraction Near Zero)", fontsize=15, labelpad=10)
    plt.title("Interaction Matrix Sparsity (Per Head)", fontsize=15, pad=10)
    plt.legend(fontsize='large')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main('InterGAT-15min')