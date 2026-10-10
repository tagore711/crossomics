import torch
import torch.nn as nn
import torch.nn.functional as F

class PatientGraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for Patient-Level Similarity Graph.
    
    Computes masked attention coefficients between patient nodes based on feature similarity:
        alpha_uv = Softmax_v( LeakyReLU( a^T [W h_u || W h_v] ) )
    """
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2, 
                 alpha: float = 0.2, concat: bool = True):
        super(PatientGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Node feature matrix of shape (N, in_features) where N is number of patients.
            adj: Adjacency/affinity matrix of shape (N, N) where adj[u, v] > 0 if edge exists.
        Returns:
            h_prime: Updated node embeddings (N, out_features)
        """
        Wh = torch.mm(h, self.W) # (N, out_features)
        N = Wh.size(0)
        
        # Self-attention mechanism
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :]) # (N, 1)
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :]) # (N, 1)
        
        # Broadcast addition for all pairs (u, v): (N, N)
        e = self.leakyrelu(Wh1 + Wh2.T)
        
        # Masked attention: set attention to -1e9 for non-edges
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_prime = torch.matmul(attention, Wh) # (N, out_features)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class PatientGAT(nn.Module):
    """
    Multi-Head Patient-Level Graph Attention Network.
    """
    def __init__(self, in_features: int, hidden_dim: int = 64, num_classes: int = 3, 
                 nheads: int = 4, dropout: float = 0.2, alpha: float = 0.2):
        super(PatientGAT, self).__init__()
        self.dropout = dropout
        
        # Multi-head attention
        self.attentions = nn.ModuleList([
            PatientGraphAttentionLayer(in_features, hidden_dim, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(nheads)
        ])
        
        # Classifier head
        total_gat_dim = hidden_dim * nheads
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(total_gat_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(total_gat_dim, num_classes)
        )
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor):
        """
        Args:
            h: Patient embeddings from feature fusion (N, D)
            adj: Patient similarity graph adjacency matrix (N, N)
        Returns:
            gat_logits: Subtype logits (N, C)
            gat_embedding: Aggregated patient graph embeddings (N, hidden_dim * nheads)
        """
        h_drop = F.dropout(h, self.dropout, training=self.training)
        head_outs = [att(h_drop, adj) for att in self.attentions]
        gat_embedding = torch.cat(head_outs, dim=1) # (N, total_gat_dim)
        
        gat_drop = F.dropout(gat_embedding, self.dropout, training=self.training)
        gat_logits = self.classifier(gat_drop)
        return gat_logits, gat_embedding
