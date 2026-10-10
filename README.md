# Interaction-Aware Multi-Omics Transformer–VCDN–GAT

A research-grade PyTorch implementation of the **Interaction-Aware Multi-Omics Transformer–VCDN–GAT** architecture for supervised cancer subtype classification across matched TCGA cohorts.

## Core Research Question
> *"Does explicitly learning feature-level cross-omics interactions improve supervised cancer subtype classification beyond conventional multi-omics fusion?"*

---

## Architecture Overview

```
                        [ Matched TCGA Multi-Omics ]
                 mRNA Data       miRNA/CNA Data    Methylation Data
                     │                 │                  │
               (Train-only)      (Train-only)       (Train-only)
             Feature Selection Feature Selection  Feature Selection
                     │                 │                  │
             ┌───────────────┐ ┌───────────────┐  ┌───────────────┐
             │  Transformer  │ │  Transformer  │  │  Transformer  │
             │    Encoder    │ │    Encoder    │  │    Encoder    │
             └───────┬───────┘ └───────┬───────┘  └───────┬───────┘
                     │                 │                  │
                   Z_mRNA            Z_cna              Z_met
                   (R^D)             (R^D)              (R^D)
                     │                 │                  │
        ┌────────────┴─────────────────┼──────────────────┴────────────┐
        │                              │                               │
        ▼                              ▼                               ▼
[ Per-Omics MLPs ]             [ Cross-Omics Gated ]           [ Modality Tokens ]
  P_mRNA, P_cna, P_met            Interaction Module             + Token Types
        │                      I_ij = W_p(g_ij ⊙ (Z_i ⊙ Z_j))          │
        ▼                              │                               ▼
┌───────────────┐                      └───────────────┬───────────────┘
│     VCDN      │                                      │
│  Prediction-  │                                      ▼
│  Level Fusion │                           [ Patient Fused Vector ]
└───────┬───────┘                                      │
        │                                              ▼
        │                              ┌───────────────────────────────┐
        │                              │   Patient-Level GAT Module    │
        │                              │  (k-NN Patient Similarity G)  │
        │                              └───────────────┬───────────────┘
        │                                              │
        │ (Prediction Pathway)                         │ (Interaction Graph Pathway)
        ▼                                              ▼
    Logits_VCDN                                    Logits_GAT
        └──────────────────────┬───────────────────────┘
                               ▼
            Final Subtype Prediction & Biomarkers
```

---

## Project Structure

```
HOPE/
├── data/                       # TCGA datasets (brca, lgg, cesc, coadread, hnsc, sarc, stad, ucec)
├── data_utils/
│   ├── dataset_loader.py       # Zero-leakage data loader, train-only feature selection & scaler
│   └── graph_builder.py        # Inductive and transductive patient similarity graph constructor
├── models/
│   ├── transformer_encoder.py  # Omics-specific Transformer encoder
│   ├── interaction_module.py   # Explicit learnable/gated pairwise cross-omics interaction module
│   ├── vcdn.py                 # Vector Clinical Decision Network (prediction tensor fusion)
│   ├── patient_gat.py          # Patient-level Graph Attention Network (multi-head)
│   └── interaction_aware_model.py # End-to-end unified model with multi-pathway integration
├── training/
│   ├── train.py                # Stratified 5-Fold training loop with multi-task loss & early stopping
│   ├── ablation.py             # Automated systematic baseline and ablation benchmark suite
│   └── interpretability.py     # Cross-omics gating analysis & biomarker saliency
├── main.py                     # Main CLI entrypoint
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation
```

---

## Execution Guide

### 1. 5-Fold Stratified Cross Validation Training
```bash
python main.py --cancer_type lgg --epochs 100 --top_k_features 500 --d_model 128 --mode train
```

### 2. Systematic Ablation Suite (Baselines 1–5 & Ablations A–G)
```bash
python main.py --cancer_type lgg --epochs 100 --mode ablation
```

### 3. Biological Interpretability & Gating Analysis
```bash
python main.py --cancer_type lgg --mode interpret
```
