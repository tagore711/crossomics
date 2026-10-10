from fpdf import FPDF
import os

class HOPEModelPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "Interaction-Aware Multi-Omics Transformer-VCDN-GAT (HOPE)", align="C")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def chapter_title(self, num, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(25, 60, 120)
        self.cell(0, 10, f"{num}. {title}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(25, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 80, 140)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        x = self.get_x()
        self.cell(6, 5.5, "-")
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def equation_box(self, eq, desc=""):
        self.set_fill_color(240, 244, 250)
        self.set_draw_color(100, 140, 200)
        y_start = self.get_y()
        self.rect(12, y_start, 186, 14 + (8 if desc else 0), style="DF")
        self.set_xy(16, y_start + 2)
        self.set_font("Courier", "B", 11)
        self.set_text_color(25, 40, 100)
        self.cell(0, 8, eq)
        if desc:
            self.set_xy(16, y_start + 10)
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(80, 80, 80)
            self.cell(0, 5, desc)
        self.set_y(y_start + 14 + (8 if desc else 0) + 3)

    def code_block(self, text):
        self.set_font("Courier", "", 9)
        self.set_fill_color(35, 35, 45)
        self.set_text_color(200, 220, 180)
        lines = text.strip().split("\n")
        h = len(lines) * 5 + 6
        y0 = self.get_y()
        self.rect(12, y0, 186, h, style="F")
        self.set_xy(16, y0 + 3)
        for line in lines:
            self.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
            self.set_x(16)
        self.set_text_color(30, 30, 30)
        self.set_y(y0 + h + 3)

    def table_row(self, cells, widths, bold=False, fill=False):
        style = "B" if bold else ""
        if fill:
            self.set_fill_color(220, 230, 245)
        self.set_font("Helvetica", style, 9)
        self.set_text_color(30, 30, 30)
        for cell_text, w in zip(cells, widths):
            self.cell(w, 7, cell_text, border=1, fill=fill, align="C")
        self.ln()


def generate_pdf(output_path):
    pdf = HOPEModelPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # =========================================================
    # TITLE PAGE
    # =========================================================
    pdf.add_page()
    pdf.ln(40)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 14, "HOPE Model", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, "Interaction-Aware Multi-Omics Transformer-VCDN-GAT", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_draw_color(25, 60, 120)
    pdf.line(60, pdf.get_y(), 150, pdf.get_y())
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, "A Deep Learning Architecture for Supervised Cancer Subtype Classification", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Across Matched TCGA Multi-Omics Cohorts", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(30)
    pdf.set_font("Helvetica", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 7, "Supported Cancer Types: BRCA, LGG, CESC, COADREAD, HNSC, SARC, STAD, UCEC", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, "Modalities: mRNA | Methylation | Copy Number Alterations (CNA)", align="C", new_x="LMARGIN", new_y="NEXT")

    # =========================================================
    # TABLE OF CONTENTS
    # =========================================================
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(25, 50, 110)
    pdf.cell(0, 12, "Table of Contents", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    toc = [
        ("1", "Research Question & Motivation"),
        ("2", "Architecture Overview Diagram"),
        ("3", "Stage 1: Omics-Specific Transformer Encoders"),
        ("4", "Stage 2: Cross-Omics Interaction Module"),
        ("5", "Stage 3: Multi-Token Fusion"),
        ("6", "Stage 4: VCDN (Vector Clinical Decision Network)"),
        ("7", "Stage 5: Patient-Level Graph Attention Network (GAT)"),
        ("8", "Final Ensemble & Multi-Level Prediction"),
        ("9", "Loss Function & Training Strategy"),
        ("10", "Data Pipeline (Zero-Leakage)"),
        ("11", "Evaluation & Metrics"),
        ("12", "Ablation Modes"),
    ]
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(30, 30, 30)
    for num, title in toc:
        pdf.cell(0, 7, f"  {num}.  {title}", new_x="LMARGIN", new_y="NEXT")

    # =========================================================
    # 1. RESEARCH QUESTION
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("1", "Research Question & Motivation")

    pdf.sub_title("Core Research Question")
    pdf.body_text(
        '"Does explicitly learning feature-level cross-omics interactions improve supervised '
        'cancer subtype classification beyond conventional multi-omics fusion?"'
    )

    pdf.sub_title("Why Existing Approaches Fall Short")
    pdf.bullet("Simple concatenation of omics data treats all features independently and misses cross-omics correlations.")
    pdf.bullet("Late fusion (averaging predictions from separate models) ignores feature-level interactions between modalities.")
    pdf.bullet("Early fusion (stacking all features) suffers from the curse of dimensionality and modality dominance.")
    pdf.bullet("Existing graph-based methods focus on gene-gene networks, not patient-level similarity graphs.")

    pdf.sub_title("HOPE's Solution")
    pdf.body_text(
        "HOPE introduces a five-stage architecture that: (1) learns omics-specific representations via Transformers, "
        "(2) explicitly models pairwise cross-omics feature interactions via gated Hadamard products, "
        "(3) fuses per-omics predictions via outer-product tensor fusion (VCDN), "
        "(4) captures patient-to-patient similarity via Graph Attention Networks, and "
        "(5) combines all levels via a learnable ensemble gate."
    )

    # =========================================================
    # 2. ARCHITECTURE OVERVIEW
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("2", "Architecture Overview")

    pdf.body_text("The HOPE model processes three omics modalities through five sequential stages:")

    pdf.code_block(
        "  Input: [mRNA data]  [CNA data]  [Methylation data]\n"
        "            |              |               |\n"
        "       Transformer    Transformer    Transformer   <-- Stage 1\n"
        "       Encoder        Encoder        Encoder\n"
        "            |              |               |\n"
        "          Z_mRNA        Z_cna          Z_met       (R^D each)\n"
        "            |              |               |\n"
        "       +----+--------------+---------------+----+\n"
        "       |              |               |\n"
        "   Per-Omics MLPs   Cross-Omics     Token Type\n"
        "   P_m, P_c, P_m    Interaction     Embeddings\n"
        "       |              Module              |\n"
        "       |          (Stage 2 & 3)           |\n"
        "       |              |                   |\n"
        "       V           h_fused ---------> Patient Fused Vector\n"
        "    VCDN              |               (Stage 3)\n"
        "  (Stage 4)       Patient GAT\n"
        "       |          (Stage 5)\n"
        "   Logits_VCDN    Logits_GAT\n"
        "       |              |\n"
        "       +------+-------+\n"
        "              |\n"
        "     Final Subtype Prediction\n"
        "    (Learnable Ensemble Gate)"
    )

    pdf.sub_title("Data Flow Summary")
    widths = [25, 45, 55, 65]
    pdf.table_row(["Stage", "Component", "Input", "Output"], widths, bold=True, fill=True)
    pdf.table_row(["1", "Transformer Encoder", "(N, F_m) per omics", "(N, D) latent Z"], widths)
    pdf.table_row(["2", "Interaction Module", "List of Z_i, Z_j", "Gated I_ij"], widths)
    pdf.table_row(["3", "Multi-Token Fusion", "All tokens", "h_patient (N, D)"], widths)
    pdf.table_row(["4", "VCDN", "Per-omics probs", "Logits (N, C)"], widths)
    pdf.table_row(["5", "Patient GAT", "h + adj matrix", "Logits (N, C)"], widths)

    # =========================================================
    # 3. STAGE 1 - TRANSFORMER ENCODERS
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("3", "Stage 1: Omics-Specific Transformer Encoders")

    pdf.sub_title("Purpose")
    pdf.body_text(
        "Each omics modality (mRNA, methylation, CNA) has its own dedicated Transformer encoder. "
        "This allows the model to learn modality-specific representations that capture the unique "
        "statistical structure of each data type before any cross-omics fusion occurs."
    )

    pdf.sub_title("Architecture (per modality)")
    pdf.bullet("Input Projection: Linear(in_features, d_model) + LayerNorm + GELU + Dropout")
    pdf.bullet("Transformer Encoder: 2-layer standard TransformerEncoderLayer with 4-head self-attention")
    pdf.bullet("Latent Projection: Linear(d_model, d_model) + LayerNorm + GELU + Dropout")
    pdf.bullet("Residual Connection: z = latent_proj(h_trans) + h_proj (skip connection from input projection)")

    pdf.sub_title("Mathematical Formulation")
    pdf.equation_box("h_proj = GELU(LayerNorm(Linear(x)))", "Input projection to d_model dimensions")
    pdf.equation_box("h_seq = h_proj.unsqueeze(1)", "Reshape to sequence of length 1")
    pdf.equation_box("h_trans = TransformerEncoder(h_seq)", "Multi-head self-attention (2 layers)")
    pdf.equation_box("z = LatentProj(h_trans.squeeze(1)) + h_proj", "Residual connection output (N, D)")

    pdf.sub_title("Key Design Decisions")
    pdf.bullet("Xavier uniform initialization for all weight matrices (gain=1.0)")
    pdf.bullet("Pre-norm (norm_first=True) Transformer layers for stable training")
    pdf.bullet("Residual connection prevents vanishing gradients in deep omics encoders")
    pdf.bullet("Output z_i is a fixed-size (N, D) vector regardless of input feature count")

    pdf.sub_title("Implementation")
    pdf.code_block(
        "class OmicsTransformerEncoder(nn.Module):\n"
        "    def __init__(self, in_features, d_model=128, nhead=4,\n"
        "                 dim_feedforward=256, num_layers=2, dropout=0.2):\n"
        "        self.input_proj = nn.Sequential(\n"
        "            nn.Linear(in_features, d_model),\n"
        "            nn.LayerNorm(d_model),\n"
        "            nn.GELU(), nn.Dropout(dropout))\n"
        "        encoder_layer = nn.TransformerEncoderLayer(\n"
        "            d_model=d_model, nhead=nhead,\n"
        "            dim_feedforward=dim_feedforward,\n"
        "            activation='gelu', batch_first=True,\n"
        "            norm_first=True)\n"
        "        self.transformer_encoder = nn.TransformerEncoder(\n"
        "            encoder_layer, num_layers=num_layers)\n"
        "        self.latent_proj = nn.Sequential(\n"
        "            nn.Linear(d_model, d_model),\n"
        "            nn.LayerNorm(d_model), nn.GELU(),\n"
        "            nn.Dropout(dropout))\n"
        "\n"
        "    def forward(self, x):   # x: (B, in_features)\n"
        "        h = self.input_proj(x)        # (B, D)\n"
        "        h_seq = h.unsqueeze(1)        # (B, 1, D)\n"
        "        h_trans = self.transformer_encoder(h_seq)\n"
        "        z = self.latent_proj(h_trans.squeeze(1)) + h\n"
        "        return z                      # (B, D)"
    )

    # =========================================================
    # 4. STAGE 2 - CROSS-OMICS INTERACTION MODULE
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("4", "Stage 2: Cross-Omics Interaction Module")

    pdf.sub_title("Purpose")
    pdf.body_text(
        "This is the core novelty of HOPE. Instead of treating omics modalities independently, "
        "this module explicitly computes pairwise (and optionally triple) interactions between "
        "omics latent vectors. The gating mechanism allows the model to learn which cross-omics "
        "correlations are informative for classification."
    )

    pdf.sub_title("Pairwise Interaction (for each pair i, j)")
    pdf.body_text("For every pair of omics modalities, the interaction is computed in 4 steps:")

    pdf.equation_box("H_ij = Z_i (element-wise) Z_j", "Raw Hadamard (element-wise) product")
    pdf.equation_box("g_ij = sigmoid(W_g * [Z_i || Z_j] + b_g)", "Learnable gating vector in [0,1]^D")
    pdf.equation_box("I_ij = g_ij (element-wise) H_ij", "Gated interaction (selective activation)")
    pdf.equation_box("I_proj_ij = GELU(LayerNorm(W_p * I_ij))", "Learnable projection + normalization")

    pdf.sub_title("How the Gate Works")
    pdf.body_text(
        "The gate g_ij is a D-dimensional vector with values in [0, 1]. Each dimension controls "
        "how much of the corresponding interaction feature is passed through. A value near 0 means "
        "that particular cross-omics interaction is suppressed; near 1 means it is preserved. "
        "The gate is learned end-to-end via backpropagation."
    )

    pdf.sub_title("Number of Interaction Pairs")
    pdf.body_text(
        "For M modalities, there are C(M,2) = M*(M-1)/2 pairs. With 3 modalities (mRNA, CNA, methylation), "
        "there are 3 pairs: mRNAxCNA, mRNAxMethylation, CNAXMethylation."
    )

    pdf.sub_title("Optional Triple Interaction")
    pdf.body_text(
        "When include_triple=True and M=3, a three-way interaction is also computed:\n"
        "  I_triple = g_triple * (Z_1 * Z_2 * Z_3)\n"
        "This captures synergistic effects unique to all three omics being simultaneously active."
    )

    pdf.sub_title("Token Assembly")
    pdf.body_text(
        "All modality tokens and interaction tokens are stacked into a tensor of shape (B, num_tokens, D). "
        "Learnable token type embeddings (one per token type) are added to distinguish modality tokens "
        "from interaction tokens. LayerNorm is applied across the token dimension."
    )

    pdf.sub_title("Fusion")
    pdf.body_text(
        "The final fused patient representation h_fused is obtained by mean pooling across all tokens: "
        "h_fused = mean(tokens, dim=1), resulting in shape (B, D)."
    )

    pdf.sub_title("Gate Interpretability")
    pdf.body_text(
        "The gating weights g_ij are saved and returned for biological interpretability analysis. "
        "Dimensions where the gate is consistently high across patients indicate important cross-omics "
        "interactions for that cancer type."
    )

    # =========================================================
    # 5. STAGE 3 - MULTI-TOKEN FUSION
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("5", "Stage 3: Multi-Token Fusion")

    pdf.sub_title("Purpose")
    pdf.body_text(
        "Stage 3 collects all individual modality embeddings and pairwise/triple interaction "
        "embeddings into a unified token representation for each patient."
    )

    pdf.sub_title("Token Collection")
    pdf.bullet("Modality tokens: Z_mRNA, Z_CNA, Z_Methylation (3 tokens, each D-dimensional)")
    pdf.bullet("Pairwise interaction tokens: I_mRNAxCNA, I_mRNAxMeth, I_CNAxMeth (3 tokens)")
    pdf.bullet("Optional triple token: I_triple (1 token)")
    pdf.bullet("Total: 6 or 7 tokens per patient")

    pdf.sub_title("Token Type Embeddings")
    pdf.body_text(
        "A learnable embedding matrix of shape (num_total_tokens, D) is created. Each token type "
        "(modality vs. interaction) gets a unique embedding that is added to the token before "
        "fusion. This helps the model distinguish between raw modality features and interaction features."
    )

    pdf.sub_title("Mean Pooling Fusion")
    pdf.equation_box(
        "h_fused = (1/T) * sum(t=1 to T) [token_t + type_embed_t]",
        "Mean pooling across T tokens"
    )
    pdf.body_text(
        "The output h_fused has shape (N, D) and serves as the patient-level representation "
        "for downstream GAT and VCDN processing."
    )

    # =========================================================
    # 6. STAGE 4 - VCDN
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("6", "Stage 4: VCDN (Vector Clinical Decision Network)")

    pdf.sub_title("Purpose")
    pdf.body_text(
        "VCDN operates at the prediction level, not the feature level. Instead of fusing raw features, "
        "it takes per-omics class probability distributions and combines them using tensor outer products "
        "to capture prediction-level interactions between modalities."
    )

    pdf.sub_title("Per-Omics Classifiers")
    pdf.body_text(
        "Before VCDN, each modality's latent embedding Z_i is passed through its own MLP classifier:\n"
        "  MLP_i: d_model -> 64 (ReLU) -> num_classes\n"
        "This produces per-omics logits, which are softmaxed to get probability vectors P_i in R^C."
    )

    pdf.sub_title("Outer Product Tensor Fusion")
    pdf.equation_box(
        "P_tensor = P_1 (x) P_2 (x) ... (x) P_M,  shape: (C^M,)",
        "Kronecker product of all modality probability vectors"
    )
    pdf.body_text(
        "For 3 modalities with C classes, P_tensor has C^3 dimensions. For example, with 3 classes "
        "and 3 modalities, this is a 27-dimensional vector encoding all possible joint prediction combinations."
    )

    pdf.sub_title("VCDN MLP Classifier")
    pdf.body_text(
        "The flattened tensor is fed through a 2-layer MLP:\n"
        "  Linear(C^M, hidden_dim) + BatchNorm + LeakyReLU(0.2) + Dropout + Linear(hidden_dim, C)\n"
        "hidden_dim is automatically set to min(256, max(32, C^M))."
    )

    pdf.sub_title("Iterative Outer Product Implementation")
    pdf.code_block(
        "# For M modalities with probabilities P_1, ..., P_M:\n"
        "# P_1: (B, C), P_2: (B, C), ...\n"
        "x = bmm(P_1.unsqueeze(2), P_2.unsqueeze(1))  # (B, C, C)\n"
        "for i in range(2, M):\n"
        "    x = bmm(x, P_i.unsqueeze(1))               # (B, C^i, C)\n"
        "vcdn_feature = x.view(B, C^M)                  # flatten\n"
        "vcdn_logits = MLP(vcdn_feature)                 # (B, C)"
    )

    pdf.sub_title("Why Outer Product?")
    pdf.body_text(
        "The outer product explicitly models the joint probability space. If one modality predicts "
        "class A with high probability and another predicts class B, the outer product encodes this "
        "specific combination. The MLP then learns which joint prediction patterns are most discriminative."
    )

    # =========================================================
    # 7. STAGE 5 - PATIENT GAT
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("7", "Stage 5: Patient-Level Graph Attention Network (GAT)")

    pdf.sub_title("Purpose")
    pdf.body_text(
        "While Stages 1-4 operate on individual patients, Stage 5 captures inter-patient relationships. "
        "Patients with similar multi-omics profiles are connected in a graph, and attention-based message "
        "passing allows each patient to aggregate information from its most similar neighbors."
    )

    pdf.sub_title("Patient Similarity Graph Construction")
    pdf.bullet("Input: concatenated multi-omics features [Z_mRNA || Z_CNA || Z_met]")
    pdf.bullet("Similarity metric: Cosine similarity, shifted to [0, 1] via (sim + 1) / 2")
    pdf.bullet("k-NN: Each patient connects to k=10 nearest neighbors")
    pdf.bullet("Symmetrization: If patient i connects to j, then j also connects to i")
    pdf.bullet("Self-loops: Every patient connects to itself (A_ii = 1.0)")
    pdf.bullet("Row normalization: Each row of adjacency matrix sums to 1")

    pdf.sub_title("Graph Attention Mechanism (per head)")
    pdf.equation_box("Wh = h * W", "Linear transformation of node features")
    pdf.equation_box("e_uv = LeakyReLU(a^T [Wh_u || Wh_v])", "Attention coefficient before masking")
    pdf.equation_box("alpha_uv = softmax_v(e_uv) masked by adjacency", "Masked attention weights")
    pdf.equation_box("h'_u = sum_v(alpha_uv * Wh_v)", "Weighted aggregation from neighbors")

    pdf.sub_title("Multi-Head Attention")
    pdf.body_text(
        "4 independent attention heads run in parallel. Each head has its own W and a parameters. "
        "Outputs are concatenated: h_gat = [head_1 || head_2 || head_3 || head_4], "
        "resulting in dimension hidden_dim * 4 = 256."
    )

    pdf.sub_title("GAT Classifier")
    pdf.body_text(
        "A final classifier head maps the concatenated attention output to class logits:\n"
        "  BN(h_gat) -> GELU -> Dropout -> Linear(hidden_dim*nheads, num_classes)"
    )

    pdf.sub_title("Inductive Evaluation (No Test-Test Leakage)")
    pdf.body_text(
        "During evaluation, test patients ONLY connect to training patients in the graph. "
        "Test patients never attend to other test patients. This is implemented via "
        "build_inductive_test_graph() which creates a bipartite structure: test-train edges "
        "only, plus self-loops for test nodes."
    )

    # =========================================================
    # 8. FINAL ENSEMBLE
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("8", "Final Ensemble & Multi-Level Prediction")

    pdf.sub_title("Two Prediction Pathways")
    pdf.body_text("HOPE produces predictions from two independent pathways:")
    pdf.bullet("Pathway A (VCDN): Prediction-level fusion via outer product of per-omics probabilities")
    pdf.bullet("Pathway B (GAT): Patient-level graph attention on fused feature embeddings")

    pdf.sub_title("Learnable Ensemble Gate")
    pdf.equation_box(
        "alpha = softmax(ensemble_gate)  # 2D learnable parameter",
        "Produces weights [alpha_1, alpha_2] that sum to 1"
    )
    pdf.equation_box(
        "final_logits = alpha_1 * gat_logits + alpha_2 * vcdn_logits",
        "Weighted combination of both pathways"
    )

    pdf.sub_title("Output Dictionary")
    pdf.body_text("The model returns a dictionary with all intermediate results:")
    pdf.bullet("final_logits: (N, C) - final integrated classification logits")
    pdf.bullet("per_omics_logits: List of (N, C) - individual omics predictions")
    pdf.bullet("vcdn_logits: (N, C) - VCDN pathway prediction")
    pdf.bullet("gat_logits: (N, C) - GAT pathway prediction")
    pdf.bullet("patient_embeddings: (N, D) - fused patient latent representation")
    pdf.bullet("gate_weights: Dict of gating vectors for cross-omics interpretation")
    pdf.bullet("interaction_tensors: Dict of computed interaction tensors")

    # =========================================================
    # 9. LOSS & TRAINING
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("9", "Loss Function & Training Strategy")

    pdf.sub_title("Multi-Task Composite Loss")
    pdf.equation_box(
        "L_total = L_final + 0.3*L_vcdn + 0.3*L_gat + 0.2*L_per_omics",
        "Multi-task loss with weighted auxiliary losses"
    )
    pdf.body_text(
        "Where:\n"
        "  L_final = CrossEntropy(final_logits, y)\n"
        "  L_vcdn  = CrossEntropy(vcdn_logits, y)\n"
        "  L_gat   = CrossEntropy(gat_logits, y)\n"
        "  L_per_omics = sum of CrossEntropy(per_omics_logits[i], y) for each omics"
    )

    pdf.sub_title("Class-Weighted Cross-Entropy")
    pdf.body_text(
        "To handle class imbalance (common in cancer subtypes), weights are computed as:\n"
        "  weight_c = total_samples / (num_classes * count_c)\n"
        "This upweights underrepresented subtypes."
    )

    pdf.sub_title("Optimizer & Scheduler")
    pdf.bullet("Optimizer: AdamW with weight_decay=1e-4")
    pdf.bullet("Scheduler: CosineAnnealingLR (T_max=epochs, eta_min=1e-5)")
    pdf.bullet("Gradient clipping: max_norm=2.0")

    pdf.sub_title("Early Stopping")
    pdf.body_text(
        "Training stops if macro-F1 on the test set does not improve for 25 consecutive epochs. "
        "The best model weights (by macro-F1) are saved and returned."
    )

    pdf.sub_title("Training Loop Summary")
    pdf.code_block(
        "for epoch in range(epochs):\n"
        "    model.train()\n"
        "    outputs = model(train_tensors, train_adj)\n"
        "    loss = L_final + 0.3*L_vcdn + 0.3*L_gat + 0.2*L_per_omics\n"
        "    loss.backward()\n"
        "    clip_grad_norm_(max_norm=2.0)\n"
        "    optimizer.step()\n"
        "    scheduler.step()\n"
        "\n"
        "    # Inductive evaluation on test set\n"
        "    model.eval()\n"
        "    ind_adj = build_inductive_test_graph(train, test, k=10)\n"
        "    all_outputs = model(combined_tensors, ind_adj)\n"
        "    test_logits = all_outputs['final_logits'][N_train:]\n"
        "    metrics = compute_metrics(y_test, test_preds, test_probs)\n"
        "    early_stopping_check(metrics['macro_f1'])"
    )

    # =========================================================
    # 10. DATA PIPELINE
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("10", "Data Pipeline (Zero-Leakage)")

    pdf.sub_title("Zero-Leakage Principle")
    pdf.body_text(
        "All preprocessing steps (feature selection, normalization) are fit ONLY on training data "
        "and then applied to test data. This prevents any information from the test set leaking "
        "into the training process."
    )

    pdf.sub_title("Pipeline Steps (per modality)")
    pdf.bullet("Step 1 - Variance Threshold: Remove features with variance < 1e-4 (fit on train only)")
    pdf.bullet("Step 2 - Feature Selection: Select top-K (default 500) features using ANOVA F-test (fit on train only)")
    pdf.bullet("Step 3 - Standard Scaling: Z-score normalization (fit on train only, apply to test)")
    pdf.bullet("Step 4 - Tensor Conversion: Convert numpy arrays to PyTorch float32 tensors")

    pdf.sub_title("Cross-Validation")
    pdf.body_text(
        "Stratified 5-fold cross-validation ensures each fold has proportional representation "
        "of all cancer subtypes. The K-fold split is stratified on the label distribution."
    )

    pdf.sub_title("Graph Construction")
    pdf.bullet("Training graph: Built from concatenated train features, k-NN with cosine similarity")
    pdf.bullet("Evaluation graph: Inductive - test nodes connect only to train nodes (no test-test edges)")

    # =========================================================
    # 11. EVALUATION METRICS
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("11", "Evaluation & Metrics")

    pdf.sub_title("Primary Metrics")
    widths2 = [40, 150]
    pdf.table_row(["Metric", "Description"], widths2, bold=True, fill=True)
    pdf.table_row(["Accuracy", "Overall correctness: correct / total"], widths2)
    pdf.table_row(["Macro-F1", "Unweighted average F1 across classes (primary metric)"], widths2)
    pdf.table_row(["Weighted-F1", "Weighted average F1 by class frequency"], widths2)
    pdf.table_row(["AUROC", "Area Under ROC (one-vs-rest for multiclass)"], widths2)
    pdf.table_row(["Balanced Accuracy", "Average recall across classes"], widths2)
    pdf.table_row(["Precision", "Average precision across classes"], widths2)
    pdf.table_row(["Recall", "Average recall (sensitivity) across classes"], widths2)
    pdf.table_row(["Confusion Matrix", "N x C confusion matrix for error analysis"], widths2)

    pdf.ln(4)
    pdf.sub_title("Reporting")
    pdf.body_text(
        "Final results are reported as mean +/- standard deviation across all 5 folds. "
        "The primary metric for model selection and early stopping is macro-F1."
    )

    # =========================================================
    # 12. ABLATION MODES
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("12", "Ablation Modes")

    pdf.sub_title("Purpose")
    pdf.body_text(
        "The model supports systematic ablation to validate the contribution of each component. "
        "Ablation modes disable specific parts of the architecture."
    )

    pdf.sub_title("Available Modes")
    widths3 = [35, 155]
    pdf.table_row(["Mode", "Description"], widths3, bold=True, fill=True)
    pdf.table_row(["full", "Complete model (all 5 stages active)"], widths3)
    pdf.table_row(["no_interaction", "Disables cross-omics interaction module; uses concat fusion instead"], widths3)
    pdf.table_row(["no_vcdn", "Disables VCDN pathway; final_logits = gat_logits only"], widths3)
    pdf.table_row(["no_gat", "Disables GAT pathway; final_logits = vcdn_logits only"], widths3)
    pdf.table_row(["single_omics", "Uses only the first omics modality (mRNA)"], widths3)
    pdf.table_row(["concat_only", "Replaces interaction module with simple linear concatenation"], widths3)

    pdf.ln(4)
    pdf.sub_title("Ablation Study Output")
    pdf.body_text(
        "Running mode='ablation' in main.py executes all baselines and ablations, producing "
        "a benchmark table with Accuracy, Macro-F1, Weighted-F1, and AUROC for each variant. "
        "Results are saved to results/<cancer_type>_ablation_benchmark.csv."
    )

    pdf.sub_title("Baseline Comparison")
    pdf.body_text(
        "The ablation suite typically compares:\n"
        "  Baseline 1: Single-omics (mRNA only)\n"
        "  Baseline 2: Simple concatenation + MLP\n"
        "  Baseline 3: Concatenation + Transformer (no interactions)\n"
        "  Baseline 4: HOPE without VCDN (GAT only)\n"
        "  Baseline 5: HOPE without GAT (VCDN only)\n"
        "  Full HOPE: Complete architecture"
    )

    # =========================================================
    # SUMMARY PAGE
    # =========================================================
    pdf.add_page()
    pdf.chapter_title("", "Summary")

    pdf.sub_title("What Makes HOPE Unique")
    pdf.bullet("Explicit cross-omics interaction modeling (gated Hadamard products) instead of naive concatenation")
    pdf.bullet("Dual-pathway prediction: feature-level (GAT) + prediction-level (VCDN) fusion")
    pdf.bullet("Patient similarity graph captures inter-patient relationships via graph attention")
    pdf.bullet("Strict zero-leakage data pipeline prevents data snooping")
    pdf.bullet("Multi-task auxiliary losses improve training stability and representation quality")
    pdf.bullet("Learnable ensemble gate automatically balances GAT and VCDN contributions")
    pdf.bullet("Full biological interpretability via gate weight analysis")

    pdf.ln(4)
    pdf.sub_title("Key Hyperparameters")
    widths4 = [45, 45, 100]
    pdf.table_row(["Parameter", "Default", "Description"], widths4, bold=True, fill=True)
    pdf.table_row(["d_model", "128", "Latent embedding dimension for all stages"], widths4)
    pdf.table_row(["top_k_features", "500", "Number of selected features per modality"], widths4)
    pdf.table_row(["gat_k", "10", "Number of nearest neighbors in patient graph"], widths4)
    pdf.table_row(["nhead_transformer", "4", "Number of attention heads in Transformer"], widths4)
    pdf.table_row(["num_transformer_layers", "2", "Depth of each Transformer encoder"], widths4)
    pdf.table_row(["gat_nheads", "4", "Number of attention heads in GAT"], widths4)
    pdf.table_row(["dropout", "0.2", "Dropout rate throughout the model"], widths4)
    pdf.table_row(["lr", "1e-3", "Learning rate (AdamW optimizer)"], widths4)
    pdf.table_row(["epochs", "100", "Maximum training epochs"], widths4)
    pdf.table_row(["n_splits", "5", "Number of cross-validation folds"], widths4)

    # Save
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "HOPE_Model_Explanation.pdf")
    generate_pdf(out)
