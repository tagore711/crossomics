import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OmicsTransformerEncoder(nn.Module):
    """
    Omics-Specific Transformer Encoder.
    
    Transforms input genomic features (e.g. selected mRNA, miRNA, or Methylation features)
    into a dense latent vector Z_m in R^D.
    
    Architecture options:
    1. Token-level feature embedding: Projects feature vector to token embeddings followed by multi-head self-attention.
    2. Sequence of feature sub-groups / CLS token pooling or mean pooling.
    """
    def __init__(self, in_features: int, d_model: int = 128, nhead: int = 4, 
                 dim_feedforward: int = 256, num_layers: int = 2, dropout: float = 0.2):
        super(OmicsTransformerEncoder, self).__init__()
        self.in_features = in_features
        self.d_model = d_model
        
        # Initial projection from raw selected feature dimension to d_model tokens
        # We project input features to a sequence of feature tokens or use linear projection + Transformer block
        self.input_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Latent projection head
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)
        Returns:
            z: Latent representation of shape (batch_size, d_model)
        """
        # Linear projection: (B, D) -> treat as single token or expand
        h = self.input_proj(x) # (B, D)
        h_seq = h.unsqueeze(1) # (B, 1, D)
        
        # Self-attention over feature token sequence
        h_trans = self.transformer_encoder(h_seq) # (B, 1, D)
        h_out = h_trans.squeeze(1) # (B, D)
        
        # Latent projection head with residual connection
        z = self.latent_proj(h_out) + h
        return z
