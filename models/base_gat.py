import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer
    """
    def __init__(self, in_features, out_features, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features  # F
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))  # W: F x F'
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))  # a: 2F' x 1
        self.attn_fc = nn.Linear(out_features, 1, bias=False)


    def forward(self, x, adj):
        """
        h: input features with shape (B, N, F) -> B x N x F
        adj: adjacency matrix with shape (B, N, N), where B is the batch size.
        """

        B, N, _ = x.size()

        # Apply W to each graph in the batch
        Wh = torch.matmul(x, self.W)  # Wh: B x N x F'

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, N, 1)  # (B, N, N, F')
        Wh_j = Wh.unsqueeze(1).repeat(1, N, 1, 1)  # (B, N, N, F')

        # # GATv2 attention: apply shared transformation on sum
        e = self.leakyrelu(Wh_i + Wh_j)  # (B, N, N, F')
        e = self.attn_fc(e).squeeze(-1)  # (B, N, N)

        # Apply adjacency matrix mask -> Injecting graph structure
        e = e / torch.linalg.norm(e)
        e = e.masked_fill(adj == 0, float('-inf'))

        # Softmax Attention Scores
        attention = F.softmax(e, dim=-1)

        # Apply attention weights to the node features
        h_prime = torch.matmul(attention, Wh)  # h_prime: B x N x F'

        if self.concat:
            return F.elu(h_prime)  # Apply activation function
        else:
            return h_prime



class GAT(nn.Module):
    def __init__(self, n_features, n_hidden, n_heads=4, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.attentions = nn.ModuleList([GraphAttentionLayer(n_features, n_hidden, alpha, concat=True) for _ in range(n_heads)])
        self.out_att = GraphAttentionLayer(n_hidden * n_heads, n_hidden, alpha, concat=False)

    def forward(self, x, adj):
        """
        x: input features with shape (B, N, F) -> B x N x F
        adj: adjacency matrix with shape (B, N, N), where B is the batch size.
        """
        # Apply multi-head attention
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)  # B x N x (n_hidden * n_heads)

        # Apply the output layer
        x = self.out_att(x, adj)  # B x N x n_hidden

        return x

