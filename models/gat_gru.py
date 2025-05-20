from torch import nn
import torch
from interactive_gat import InteractiveGAT
from base_gat import GAT

class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.Wr = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wz = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.Wh = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        r = torch.sigmoid(self.Wr(combined))  # Reset gate
        z = torch.sigmoid(self.Wz(combined))  # Update gate
        h_tilde = torch.tanh(self.Wh(torch.cat([x, r * h], dim=-1)))  # Candidate hidden state
        h_next = (1 - z) * h + z * h_tilde  # Compute next hidden state
        return h_next

class GAT_GRU(nn.Module):
    def __init__(self, n_nodes, n_feat, n_hidden, n_heads, dropout, alpha, output_dim, adj, interactive=False):
        super().__init__()
        if interactive:
            self.gat = InteractiveGAT(n_feat, n_hidden, adj, n_heads, dropout)
        else:
            self.gat = GAT(n_feat, n_hidden, n_heads, dropout, alpha)

        self.gru = GRUCell(n_hidden, n_hidden)
        self.n_nodes = n_nodes
        self.n_hidden = n_hidden
        self.fc = nn.Linear(n_hidden, output_dim)


    def forward(self, x, adj, C, L):
        batch_size, seq_len, num_nodes, _ = x.shape
        h = torch.zeros(batch_size, num_nodes, self.n_hidden).to(x.device)  # Init hidden state

        for t in range(2, seq_len):  # Process each time step
            gat_out = self.gat(x[:, t, :, :], adj, C, L)  # GAT extracts spatial features
            # gat_out = self.gat(x, t, adj, C, L)  # GAT extracts spatial features
            h = self.gru(gat_out, h)  # GRU processes temporal dependencies

        out = self.fc(h)  # Fc layer for mapping to output prediction length

        return out  # Final hidden state as prediction