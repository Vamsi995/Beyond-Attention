import torch
import torch.nn as nn
import torch.nn.functional as F

class InteractiveGraphAttentionLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_features, dropout, adj, concat=True):
        super(InteractiveGraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # F
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # W: F x F'
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        N = adj.shape[0]
        self.interaction_matrix = nn.Parameter(torch.ones(N, N))
        nn.init.xavier_uniform_(self.interaction_matrix.data, gain=1.414)

    def forward(self, x):
        """
        h: input features with shape (B, N, F) -> B x N x F
        adj: adjacency matrix with shape (B, N, N), where B is the batch size.
        """

        B, N, _ = x.size()

        # Apply W to each graph in the batch
        Wh = torch.matmul(x, self.W)  # Wh: B x N x F'

        e = self.interaction_matrix  # NxN

        # Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, N, F')
        # Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, N, F')

        # # GATv2 attention: apply shared transformation on sum
        # e = self.leakyrelu(Wh_i + Wh_j)  # (B, N, N, F')
        # e = self.attn_fc(e).squeeze(-1)  # (B, N, N)

        # Symmetric Constraint
        e = (e + e.T) / 2

        # L2 Norm and Layer Norm
        e = e / (torch.linalg.norm(e, dim=-1, keepdim=True) + 1e-8)
        e = torch.nn.functional.layer_norm(e, e.shape[-1:])


        # Apply adjacency matrix mask -> Injecting graph structure -> Need to inject covariance/bottleneck here
        # e = e / torch.linalg.norm(e)
        # e = e.masked_fill(adj == 0, float('-inf'))

        # Softmax Attention Scores
        attention = F.softmax(e, dim=-1)

        # Apply attention weights to the node features
        h_prime = torch.matmul(attention, Wh)  # h_prime: B x N x F'

        if self.concat:
            return F.elu(h_prime)  # Apply activation function
        else:
            return h_prime


class InteractiveGAT(nn.Module):
    def __init__(self, n_features, n_hidden, adj, n_heads=4, dropout=0.6):
        super(InteractiveGAT, self).__init__()
        self.attentions = nn.ModuleList([InteractiveGraphAttentionLayer(n_features, n_hidden, dropout, adj, concat=True) for _ in range(n_heads)])
        self.out_att = InteractiveGraphAttentionLayer(n_hidden * n_heads, n_hidden, dropout, adj, concat=False)

    def forward(self, x):
        """
        x: input features with shape (B, N, F) -> B x N x F
        adj: adjacency matrix with shape (B, N, N), where B is the batch size.
        """
        # Apply multi-head attention
        x = torch.cat([att(x) for att in self.attentions], dim=2)  # B x N x (n_hidden * n_heads)

        # Apply the output layer
        x = self.out_att(x)  # B x N x n_hidden

        return x