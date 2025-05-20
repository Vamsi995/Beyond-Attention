import torch
from tqdm import tqdm
from torch import nn, optim
import torch.nn.functional as F
from dataset.preprocessing import SpatioTemporalCSVDataModule
from models.gat_gru import GAT_GRU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def accuracy_f(pred, y):
    """
    :param pred: predictions
    :param y: ground truth
    :return: accuracy, defined as 1 - (norm(y - pred) / norm(y))
    """
    return 1 - torch.linalg.norm(y - pred, "fro") / torch.linalg.norm(y, "fro")


def r2_f(pred, y):
    """
    :param y: ground truth
    :param pred: predictions
    :return: R square (coefficient of determination)
    """
    return 1 - torch.sum((y - pred) ** 2) / torch.sum((y - torch.mean(pred)) ** 2)


def explained_variance_f(pred, y):
    return 1 - torch.var(y - pred) / torch.var(y)


def validate(model, val_loader, adj):
        model.eval()
        val_loss, rmse, mae, r2, explained_var = 0.0, 0.0, 0.0, 0.0, 0.0
        acc = 0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                x, y = batch
                x, y = x.to(device), y.to(device)

                b, t, n = x.shape
                x = x.reshape((b, t, n, 1))  # Add a channel dimension
                predictions = model(x, adj)
                predictions = predictions * data_module.feat_max_val

                predictions = predictions.transpose(1, 2)  # Reshape predictions
                predictions = predictions.reshape((-1, predictions.size(2)))  # Reshape predictions
                y = y.reshape((-1, y.size(2)))  # Reshape targets
                y = y * data_module.feat_max_val



                # Compute loss
                loss = criterion(predictions, y)
                val_loss += loss.item()

                # Metrics
                rmse += torch.sqrt(F.mse_loss(predictions, y)).item()
                mae += F.l1_loss(predictions, y).item()
                r2 += r2_f(predictions, y)
                explained_var += explained_variance_f(predictions, y)
                acc += accuracy_f(predictions, y)

        num_batches = len(val_loader)
        print(
            f"Validation Loss: {val_loss / num_batches:.4f}, RMSE: {rmse / num_batches:.4f}, "
            f"MAE: {mae / num_batches:.4f}, R2: {r2 / num_batches:.4f}, Explained Variance: {explained_var / num_batches:.4f}", f"Accuracy: {acc / num_batches:.4f}"
        )

def initialize_model(args, adj):
    model = GAT_GRU(args.n_nodes, args.n_feat, args.n_hidden, args.n_heads, args.dropout, args.alpha, args.output_dim, adj).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

if __name__ == '__main__':
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

    data_module = SpatioTemporalCSVDataModule('./dataset/data/sz_speed.csv', './dataset/data/sz_adj.csv')
    val_loader = data_module.val_dataloader()
    adj = data_module._adj
    adj = torch.from_numpy(adj).to(device)

    model = initialize_model(args, adj)

    state = torch.load('checkpoint_path_here')
    model.load_state_dict(state, strict=False)
    validate(model, val_loader, adj)