import torch
import torch.nn as nn
import torch.nn.functional as F

class VCDN(nn.Module):
    """
    Vector Clinical Decision Network (VCDN) for Prediction-Level Multi-Omics Fusion.
    
    Given per-omics class probability vectors P_1, P_2, ..., P_M (each in R^C),
    VCDN computes the tensor outer product:
        P_tensor = P_1 (x) P_2 (x) ... (x) P_M  in R^(C^M)
    and maps the flattened joint probability vector through a multi-layer classifier:
        Logits_VCDN = MLP(vec(P_tensor))
    """
    def __init__(self, num_modalities: int, num_classes: int, hidden_dim: int = None, dropout: float = 0.2):
        super(VCDN, self).__init__()
        self.num_modalities = num_modalities
        self.num_classes = num_classes
        self.in_dim = pow(num_classes, num_modalities)
        if hidden_dim is None:
            hidden_dim = min(256, max(32, self.in_dim))
            
        self.mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, prob_list: list) -> torch.Tensor:
        """
        Args:
            prob_list: List of per-omics probability distributions [P_1, P_2, ..., P_M], 
                       each of shape (B, C) with sum(P_m, dim=-1) = 1 (or softmax probabilities).
        Returns:
            vcdn_logits: Tensor of shape (B, C) representing integrated prediction logits.
        """
        batch_size = prob_list[0].size(0)
        num_views = len(prob_list)
        
        # Ensure probabilities are normalized (using softmax if raw logits passed)
        probs = [F.softmax(p, dim=-1) if p.min() < 0 or p.max() > 1 else p for p in prob_list]
        
        if num_views == 1:
            return prob_list[0]
            
        # First 2 views outer product: (B, C, 1) x (B, 1, C) -> (B, C, C) -> (B, C^2, 1)
        x = torch.bmm(probs[0].unsqueeze(2), probs[1].unsqueeze(1)).view(batch_size, -1, 1)
        
        # Iteratively compute Kronecker/outer product for remaining modalities
        for i in range(2, num_views):
            x = torch.bmm(x, probs[i].unsqueeze(1)).view(batch_size, -1, 1)
            
        # Flatten to (B, C^M)
        vcdn_feature = x.view(batch_size, self.in_dim)
        vcdn_logits = self.mlp(vcdn_feature)
        return vcdn_logits
