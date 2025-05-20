import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from dataset.preprocessing import SpatioTemporalCSVDataModule
from models.gat_gru import GAT_GRU
from collections import defaultdict
import time
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Model Initialization
def initialize_model(args, adj):
    model = GAT_GRU(args.n_nodes, args.n_feat, args.n_hidden, args.n_heads, args.dropout, args.alpha, args.output_dim, adj).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    return model, optimizer



def train(args, model, optimizer, train_loader, val_loader, prefix_str):

    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    train_loss_per_epoch = []
    val_loss_per_epoch = []
    interaction_norm_per_epoch = defaultdict(list)     # key = head index or name
    interaction_sparsity_per_epoch = defaultdict(list)

    epoch_times = []

    # Training Loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        model.train()  # Set model to training mode
        epoch_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} - Training"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Forward pass
            b, t, n = inputs.shape
            inputs = inputs.reshape((b, t, n, 1))  # Add a channel dimension
            outputs = model(inputs)

            outputs = outputs.transpose(1, 2)

            # Compute the loss (Mean Squared Error)
            loss = criterion(outputs, targets)

            # Add L1 regularization on interaction matrix
            lambda_sparse = 0.1  # Adjust based on sparsity you want
            l1_penalty = 0.0

            for name, param in model.named_parameters():
                if "interaction_matrix" in name:
                    l1_penalty += param.abs().mean()

            loss += lambda_sparse * l1_penalty

            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        # Print the training loss for this epoch
        avg_train_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Training Loss: {avg_train_loss:.4f}")

        train_loss_per_epoch.append(avg_train_loss)
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        print(f"Epoch {epoch + 1} time: {epoch_time:.2f} seconds")


        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                b, t, n = x.shape
                x = x.reshape((b, t, n, 1))  # Add a channel dimension
                outputs = model(x, adj)
                outputs = outputs.transpose(1, 2)

                loss = criterion(outputs, y)
                val_loss += loss.item()

        val_loss /= len(val_loader)



        # Inside the training loop, at the end of each epoch:
        for name, param in model.named_parameters():
          if "interaction_matrix" in name:
              head_id = name.split('.')[-2]
              I = param.detach().cpu()
              norm_I = torch.norm(I, p='fro').item()
              sparsity_I = (I.abs() < 1e-4).float().mean().item()

              interaction_norm_per_epoch[head_id].append(norm_I)
              interaction_sparsity_per_epoch[head_id].append(sparsity_I)

              print(f"Epoch {epoch+1} | {head_id} ||I||_F = {norm_I:.4f}, Sparsity = {sparsity_I:.4f}")

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")

        val_loss_per_epoch.append(val_loss)

        # Checkpoint saving
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{prefix_str}_checkpoint_epoch_{epoch + 1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(best_model_dir, f'{prefix_str}_best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, best_model_path)
            print(f"Saved best model: {best_model_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Hyperparameters for traffic forecasting model")

    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--n_feat', type=int, default=1, help='Number of input features per node')
    parser.add_argument('--n_hidden', type=int, default=32, help='Number of hidden units in GAT')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate')
    parser.add_argument('--alpha', type=float, default=0.2, help='LeakyReLU negative slope')
    parser.add_argument('--output_dim', type=int, default=1, help='Dimensionality of model output')
    parser.add_argument('--seq_len', type=int, default=12, help='Length of input time sequence')
    parser.add_argument('--n_nodes', type=int, default=156, help='Number of graph nodes')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')

    args = parser.parse_args()

    # Dataset Prep
    data_module = SpatioTemporalCSVDataModule('./dataset/data/sz_speed.csv', './dataset/data/sz_adj.csv')
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    adj = data_module._adj
    adj = torch.from_numpy(adj).to(device)

    checkpoint_dir = './dataset/checkpoints'
    best_model_dir = './dataset/best_model'
    meta_data_dir = './dataset/meta_data'
    interaction_matrix_dir = './dataset/interaction_matrix'
    interaction_matrix_norm_dir = './dataset/interaction_matrix_norm'
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(meta_data_dir, exist_ok=True)
    os.makedirs(interaction_matrix_dir, exist_ok=True)
    os.makedirs(interaction_matrix_norm_dir, exist_ok=True)

    model, optimizer = initialize_model(args, adj)

    train(args, model, optimizer, train_loader, val_loader,'experiment1')