import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossOmicsInteractionModule(nn.Module):
    """
    Explicit Learnable/Gated Cross-Omics Interaction Module.
    
    This is the core feature-level cross-omics interaction module.
    For pairs of omics latent vectors (Z_i, Z_j) in R^D:
      1. Raw Hadamard interaction: H_ij = Z_i ⊙ Z_j
      2. Learnable gating mechanism: g_ij = sigmoid(W_g [Z_i || Z_j] + b_g)
      3. Gated interaction: I_ij = g_ij ⊙ H_ij
      4. Learnable projection & LayerNorm: I_proj_ij = LayerNorm(W_p(I_ij))
    
    Supports:
      - Pairwise interactions (mRNA x miRNA, mRNA x Methylation, miRNA x Methylation)
      - Ablation modes: 'none', 'ungated', 'gated', 'single_pair', 'triple'
    """
    def __init__(self, d_model: int = 128, num_modalities: int = 3, 
                 use_gating: bool = True, use_projection: bool = True,
                 dropout: float = 0.2, include_triple: bool = False):
        super(CrossOmicsInteractionModule, self).__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        self.use_gating = use_gating
        self.use_projection = use_projection
        self.include_triple = include_triple
        
        # Calculate number of pairs: C(M, 2)
        self.num_pairs = (num_modalities * (num_modalities - 1)) // 2
        
        if use_gating:
            # Gating network per pair: input dimension is 2 * d_model -> output is d_model
            self.gate_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * d_model, d_model),
                    nn.Dropout(dropout),
                    nn.Sigmoid()
                ) for _ in range(self.num_pairs)
            ])
            
            if include_triple:
                self.triple_gate = nn.Sequential(
                    nn.Linear(num_modalities * d_model, d_model),
                    nn.Dropout(dropout),
                    nn.Sigmoid()
                )
        
        if use_projection:
            # Dimension-controlled projection per interaction
            self.proj_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(self.num_pairs)
            ])
            
            if include_triple:
                self.triple_proj = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.LayerNorm(d_model),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )

        # Modality and interaction type embeddings
        total_tokens = num_modalities + self.num_pairs + (1 if include_triple else 0)
        self.token_type_embeddings = nn.Parameter(torch.randn(total_tokens, d_model) * 0.02)
        
        self.fusion_norm = nn.LayerNorm(d_model)

    def forward(self, omics_latents: list, active_pairs: list = None):
        """
        Args:
            omics_latents: List of tensors [Z_1, Z_2, ..., Z_M], each of shape (B, D)
            active_pairs: Optional list of pair indices to compute (for ablation studies)
        Returns:
            fused_patient_repr: Fused patient representation in R^D (or concatenated/pooled)
            interaction_dict: Dictionary containing computed interaction tensors and gate weights
            all_tokens: Tensor of shape (B, num_total_tokens, D)
        """
        batch_size = omics_latents[0].size(0)
        num_m = len(omics_latents)
        
        tokens = list(omics_latents) # [Z_1, Z_2, ..., Z_M]
        interaction_dict = {}
        gate_weights_dict = {}
        
        pair_idx = 0
        for i in range(num_m):
            for j in range(i + 1, num_m):
                if active_pairs is not None and pair_idx not in active_pairs:
                    pair_idx += 1
                    continue
                
                z_i = omics_latents[i]
                z_j = omics_latents[j]
                
                # 1. Hadamard product
                hadamard = z_i * z_j # (B, D)
                
                # 2. Learnable gating
                if self.use_gating:
                    concat_ij = torch.cat([z_i, z_j], dim=-1) # (B, 2D)
                    g_ij = self.gate_nets[pair_idx](concat_ij) # (B, D)
                    i_ij = g_ij * hadamard
                    gate_weights_dict[f"gate_{i}_{j}"] = g_ij
                else:
                    i_ij = hadamard
                
                # 3. Projection
                if self.use_projection:
                    i_ij = self.proj_nets[pair_idx](i_ij)
                
                interaction_dict[f"interaction_{i}_{j}"] = i_ij
                tokens.append(i_ij)
                pair_idx += 1
                
        # Optional Triple interaction
        if self.include_triple and num_m == 3:
            triple_hadamard = omics_latents[0] * omics_latents[1] * omics_latents[2]
            if self.use_gating:
                concat_all = torch.cat(omics_latents, dim=-1)
                g_triple = self.triple_gate(concat_all)
                i_triple = g_triple * triple_hadamard
                gate_weights_dict["gate_triple"] = g_triple
            else:
                i_triple = triple_hadamard
            
            if self.use_projection:
                i_triple = self.triple_proj(i_triple)
            
            interaction_dict["interaction_triple"] = i_triple
            tokens.append(i_triple)
            
        # Stack tokens: (B, num_tokens, D)
        token_tensor = torch.stack(tokens, dim=1)
        num_tokens = token_tensor.size(1)
        
        # Add token type embeddings
        token_tensor = token_tensor + self.token_type_embeddings[:num_tokens].unsqueeze(0)
        token_tensor = self.fusion_norm(token_tensor)
        
        # Patient-level pooled feature: mean pooling across modality and interaction tokens
        fused_patient_repr = token_tensor.mean(dim=1) # (B, D)
        
        return fused_patient_repr, interaction_dict, gate_weights_dict, token_tensor
