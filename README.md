
# InterGAT-GRU: Interaction-Based Spatial Modeling for Spatio-Temporal Forecasting

This repository contains the official implementation of **InterGAT-GRU**, a novel GAT-based spatio-temporal forecasting architecture. Unlike conventional GATs that rely on masked attention over predefined neighbors, InterGAT uses a fully learnable symmetric interaction matrix to capture dense, long-range spatial dependencies — significantly improving performance and training efficiency.

> 📄 _This work introduces InterGAT-GRU, showing that eliminating adjacency masking and replacing dynamic attention with a static interaction matrix enables more expressive and interpretable spatio-temporal models._

---

## 🚀 Highlights

- **Adjacency-Free GAT Layer**: Replaces hard graph priors with a learnable interaction matrix.
- **GRU-Based Temporal Module**: Handles sequential node-level forecasting.
- **Efficiency**: Achieves **2× reduction** in per-epoch training time vs. GAT-GRU (see [Results](#results)).
- **Interpretability**: Learns sparse, symmetric matrices revealing community structures.
- **Reproducible**: Deterministic data loaders and fixed seeds (optional).

---

## 📦 Repo Structure

```
.
├── dataset/
│   ├── data/                  # Input data (speed + adjacency)
│   ├── checkpoints/           # Periodic checkpoints
│   ├── best_model/            # Best model by validation loss
│   ├── meta_data/             # Training time logs
│   ├── interaction_matrix/    # Raw interaction matrices
│   └── preprocessing.py       # Dataset loader (CSV)
├── models/
│   └── gat_gru.py             # InterGAT-GRU model definition
├── train.py                   # Main training script
└── README.md
```

---

## 🧠 Model Architecture

**InterGAT-GRU** is a spatio-temporal forecasting model that replaces adjacency-masked GAT attention with a dense, fully learnable interaction matrix. The architecture consists of two core components:

### 🔹 Spatial Encoder: InterGAT Layer

Instead of computing attention scores based on neighbor features and adjacency masks, we use a **symmetric interaction matrix** $\( \mathbf{I} \in \mathbb{R}^{N \times N} \)$, learned end-to-end, to encode latent structural dependencies among all node pairs.

The encoding process follows:

- **Feature Projection**:
  $\mathbf{h}_i = \mathbf{W} \mathbf{x}_i$

- **Interaction Matrix Processing**:
  $\mathbf{I} \leftarrow \text{softmax}\left( \text{LayerNorm}\left( \frac{1}{2}(\mathbf{I} + \mathbf{I}^\top) \right) \right)$


- **Spatial Aggregation**:
  $\mathbf{Z}_i = \text{ELU} \left( \sum_{j=1}^{N} \mathbf{I}_{ij} \cdot \mathbf{h}_j \right)$

Where:
- $\( \mathbf{I} \)$: trainable matrix, row-softmax normalized after symmetrization.
- $\( \mathbf{h}_i \)$: transformed input node features.
- $\( \mathbf{Z}_i \)$: spatial output embedding for node \( i \).

### 🔹 Temporal Decoder: GRU

The GRU module processes a sequence of spatial representations to capture temporal dynamics over time:


The GRU output is passed through a linear layer to predict the target feature at future steps:

$\hat{\mathbf{x}}_{t+\tau} = \text{Linear}(\mathbf{h}_t)$

### 🔹 Regularization

To encourage sparsity and symmetry in $\( \mathbf{I} \)$, the model includes:

- **L1 Regularization**:
  $\mathcal{L}_{\text{sparse}} = \lambda_{\text{sparse}} \cdot \|\mathbf{I}\|_1$

- **Symmetry Constraint**:
  $\mathbf{I} \leftarrow \frac{1}{2}(\mathbf{I} + \mathbf{I}^\top)$

### 🔹 Complexity Analysis

- Both GAT and InterGAT scale as \( \mathcal{O}(N^2) \), but InterGAT avoids repeated softmax and pairwise projections.
- InterGAT computes and reuses a static \( \mathbf{I} \), which is more efficient for small-to-medium graphs.


---

## 📊 Results

| Model         | Epoch Time (s) | Epochs to Convergence | Total Train Time (min) | Forward Pass (s) | Backward Pass (s) |
|---------------|----------------|------------------------|-------------------------|------------------|-------------------|
| **BaseGAT-GRU** | 5.47           | 100                    | 9.11                    | 2.67             | 2.66              |
| **InterGAT-GRU**| 2.23           | 100                    | 3.71                    | 0.86             | 1.27              |

- Training performed on **SZ-Taxi** dataset for 15-min horizon.
- Both models were trained with identical hyperparameters, optimizer, and hardware setup.
- See `meta_data/` for full logs and convergence curves.

---

## 📈 Community Analysis

We compute intra/inter-community interaction contrast to measure alignment of learned structure with spectral clusters. The aggregated head shows the **highest structural fidelity**, capturing modular behavior.

| Head           | Mean Contrast (\( \frac{\mu_\text{intra} - \mu_\text{inter}}{\mu_\text{inter}} \)) | Std Dev |
|----------------|---------------------|----------|
| Interaction 1  | 1.49                | 1.08     |
| Interaction 2  | 0.76                | 0.79     |
| Interaction 3  | 0.64                | 1.05     |
| Interaction 4  | 3.27                | 4.16     |
| Aggregated     | 1.47                | 1.01     |

---

## 📦 Installation

```bash
git clone https://github.com/Vamsi995/Rethinking-Graph-Attention.git
cd InterGAT-GRU
pip install -r requirements.txt
```

Make sure to place `sz_speed.csv` and `sz_adj.csv` in `dataset/data/`.

---

## 🏃‍♂️ Running the Model

```bash
python train.py \
  --learning_rate 0.001 \
  --epochs 100 \
  --batch_size 32 \
  --n_feat 1 \
  --n_hidden 32 \
  --n_heads 4 \
  --dropout 0.6 \
  --alpha 0.2 \
  --output_dim 1 \
  --seq_len 12 \
  --n_nodes 156
```

---

## 📂 Outputs

- `meta_data/`: Contains `.txt` logs of epoch-wise training time and loss.
- `interaction_matrix/`: Saved interaction matrix weights per head.
- `checkpoints/`: Model checkpoints at every 10 epochs.
- `best_model/`: Best model by validation loss.

---

## 📚 Citation

```bibtex
@inproceedings{your2024intergat,
  title     = {InterGAT-GRU: Interaction-Based Spatial Modeling for Spatio-Temporal Forecasting},
  author    = {Your Name and Collaborator},
  booktitle = {NeurIPS},
  year      = {2024}
}
```

---

## 📌 Acknowledgements

This repo is part of a broader research study on attention-free GNNs and structural priors in spatio-temporal learning. If you use this code or ideas, please consider citing and referencing our work.

---

## 🔍 Future Work

- Extend interaction matrices to **time-dependent variants** $\( \mathbf{I}_t \)$
- Generalize to **dynamic graphs** with evolving node sets
- Integrate **contextual factors** like weather, events, or anomalies
