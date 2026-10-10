import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_encoder import OmicsTransformerEncoder
from models.interaction_module import CrossOmicsInteractionModule
from models.vcdn import VCDN
from models.patient_gat import PatientGAT

class InteractionAwareMultiOmicsModel(nn.Module):
    """
    Interaction-Aware Multi-Omics Transformer–VCDN–GAT Architecture.
    
    Stages:
      Stage 1: Omics-Specific Transformer Encoders -> Z_mRNA, Z_miRNA, Z_methylation in R^D
      Stage 2: Explicit Learnable/Gated Cross-Omics Interaction Module -> I_ij = W_p(g_ij ⊙ (Z_i ⊙ Z_j))
      Stage 3: Multi-View Representation & Interaction Fusion -> h_patient in R^D
      Stage 4: Prediction-Level Fusion via Vector Clinical Decision Network (VCDN)
      Stage 5: Patient-Level Graph Attention Network (GAT) on patient similarity graph
    """
    def __init__(self, 
                 in_feature_dims: list, 
                 num_classes: int,
                 d_model: int = 128,
                 nhead_transformer: int = 4,
                 num_transformer_layers: int = 2,
                 dim_feedforward: int = 256,
                 gat_hidden_dim: int = 64,
                 gat_nheads: int = 4,
                 vcdn_hidden_dim: int = None,
                 dropout: float = 0.2,
                 use_interactions: bool = True,
                 use_gating: bool = True,
                 active_pairs: list = None,
                 include_triple: bool = False,
                 ablation_mode: str = "full"):
        super(InteractionAwareMultiOmicsModel, self).__init__()
        
        self.num_modalities = len(in_feature_dims)
        self.num_classes = num_classes
        self.d_model = d_model
        self.use_interactions = use_interactions
        self.use_gating = use_gating
        self.active_pairs = active_pairs
        self.ablation_mode = ablation_mode # 'full', 'no_interaction', 'no_vcdn', 'no_gat', 'concat_only', 'single_omics'
        
        # Stage 1: Omics-Specific Transformers
        self.encoders = nn.ModuleList([
            OmicsTransformerEncoder(
                in_features=in_dim,
                d_model=d_model,
                nhead=nhead_transformer,
                dim_feedforward=dim_feedforward,
                num_layers=num_transformer_layers,
                dropout=dropout
            ) for in_dim in in_feature_dims
        ])
        
        # Per-omics classifier heads (for VCDN inputs)
        self.per_omics_classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_classes)
            ) for _ in range(self.num_modalities)
        ])
        
        # Stage 2 & 3: Cross-Omics Interaction Module
        if self.use_interactions:
            self.interaction_module = CrossOmicsInteractionModule(
                d_model=d_model,
                num_modalities=self.num_modalities,
                use_gating=use_gating,
                use_projection=True,
                dropout=dropout,
                include_triple=include_triple
            )
        else:
            self.linear_concat_proj = nn.Sequential(
                nn.Linear(self.num_modalities * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        # Stage 4: Prediction-Level VCDN
        self.vcdn = VCDN(
            num_modalities=self.num_modalities,
            num_classes=num_classes,
            hidden_dim=vcdn_hidden_dim,
            dropout=dropout
        )
        
        # Stage 5: Patient-Level GAT
        self.patient_gat = PatientGAT(
            in_features=d_model,
            hidden_dim=gat_hidden_dim,
            num_classes=num_classes,
            nheads=gat_nheads,
            dropout=dropout
        )
        
        # Multi-modal fusion integration head (combines GAT logits, VCDN logits, and per-omics signals)
        self.ensemble_gate = nn.Parameter(torch.tensor([0.5, 0.5])) # Learnable weighting between GAT and VCDN

    def forward(self, omics_data_list: list, adj_matrix: torch.Tensor = None):
        """
        Args:
            omics_data_list: List of tensors [X_1, X_2, ..., X_M], each of shape (N, in_features_m)
            adj_matrix: Patient graph adjacency matrix (N, N)
        Returns:
            outputs: Dictionary containing:
                - 'final_logits': Integrated classification logits (N, C)
                - 'per_omics_logits': List of logits from individual omics (N, C)
                - 'vcdn_logits': Logits from prediction-level VCDN (N, C)
                - 'gat_logits': Logits from patient-level GAT (N, C)
                - 'patient_embeddings': Fused patient latent embeddings (N, D)
                - 'gate_weights': Cross-omics gating attention weights for interpretability
        """
        N = omics_data_list[0].size(0)
        
        # Stage 1: Omics-Specific Transformers
        z_list = [self.encoders[i](omics_data_list[i]) for i in range(self.num_modalities)] # List of (N, D)
        
        # Per-omics classification predictions for VCDN
        per_omics_logits = [self.per_omics_classifiers[i](z_list[i]) for i in range(self.num_modalities)] # List of (N, C)
        per_omics_probs = [F.softmax(logit, dim=-1) for logit in per_omics_logits]
        
        # Stage 2 & 3: Feature-level Interaction Modeling & Fusion
        gate_weights = {}
        interaction_dict = {}
        if self.use_interactions:
            h_fused, interaction_dict, gate_weights, token_tensor = self.interaction_module(
                z_list, active_pairs=self.active_pairs
            )
        else:
            # Baseline concatenation
            concat_z = torch.cat(z_list, dim=-1)
            h_fused = self.linear_concat_proj(concat_z)
            token_tensor = None
            
        # Stage 4: VCDN Prediction-Level Fusion
        vcdn_logits = self.vcdn(per_omics_probs) # (N, C)
        
        # Stage 5: Patient-Level GAT
        if adj_matrix is None:
            # Fallback: identity adjacency (self-loops only)
            adj_matrix = torch.eye(N, device=h_fused.device)
            
        gat_logits, gat_embedding = self.patient_gat(h_fused, adj_matrix) # (N, C), (N, GAT_dim)
        
        # Multi-level Ensemble Prediction:
        # Final logits combine feature-interaction GAT pathway and prediction-level VCDN pathway
        gate_weights_norm = F.softmax(self.ensemble_gate, dim=0)
        final_logits = gate_weights_norm[0] * gat_logits + gate_weights_norm[1] * vcdn_logits
        
        # Specific ablation bypasses
        if self.ablation_mode == "no_vcdn":
            final_logits = gat_logits
        elif self.ablation_mode == "no_gat":
            final_logits = vcdn_logits
        elif self.ablation_mode == "single_omics":
            final_logits = per_omics_logits[0]
            
        return {
            "final_logits": final_logits,
            "per_omics_logits": per_omics_logits,
            "vcdn_logits": vcdn_logits,
            "gat_logits": gat_logits,
            "patient_embeddings": h_fused,
            "gate_weights": gate_weights,
            "interaction_tensors": interaction_dict
        }
