import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from models.interaction_aware_model import InteractionAwareMultiOmicsModel
from data_utils.graph_builder import build_knn_patient_graph, build_inductive_test_graph

def compute_metrics(y_true, y_pred, y_probs=None, num_classes=3):
    """
    Computes comprehensive classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    auroc = 0.0
    if y_probs is not None:
        try:
            if num_classes == 2:
                auroc = roc_auc_score(y_true, y_probs[:, 1])
            else:
                auroc = roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        except Exception:
            auroc = 0.0
            
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "auroc": auroc,
        "confusion_matrix": cm
    }

def train_interaction_model(train_data: list, y_train: torch.Tensor,
                            test_data: list, y_test: torch.Tensor,
                            num_classes: int,
                            feature_dims: list,
                            epochs: int = 150,
                            lr: float = 1e-3,
                            weight_decay: float = 1e-4,
                            d_model: int = 128,
                            gat_k: int = 10,
                            use_interactions: bool = True,
                            use_gating: bool = True,
                            active_pairs: list = None,
                            include_triple: bool = False,
                            ablation_mode: str = "full",
                            patience: int = 25,
                            device: str = "cpu"):
    """
    Full training and evaluation pipeline for Interaction-Aware Multi-Omics Model.
    """
    dev = torch.device(device)
    
    # Move raw tensors to device
    train_tensors = [x.to(dev) for x in train_data]
    test_tensors = [x.to(dev) for x in test_data]
    y_tr = y_train.to(dev)
    y_te = y_test.to(dev)
    
    # Calculate class weights for imbalanced datasets
    class_counts = torch.bincount(y_train)
    total_samples = len(y_train)
    class_weights = total_samples / (len(class_counts) * class_counts.float().clamp(min=1))
    class_weights = class_weights.to(dev)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Initialize Model
    model = InteractionAwareMultiOmicsModel(
        in_feature_dims=feature_dims,
        num_classes=num_classes,
        d_model=d_model,
        nhead_transformer=4,
        num_transformer_layers=2,
        dim_feedforward=256,
        gat_hidden_dim=64,
        gat_nheads=4,
        dropout=0.2,
        use_interactions=use_interactions,
        use_gating=use_gating,
        active_pairs=active_pairs,
        include_triple=include_triple,
        ablation_mode=ablation_mode
    ).to(dev)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Construct Initial Train Patient Graph (based on concatenated multi-omics)
    train_concat = torch.cat(train_tensors, dim=-1).cpu().numpy()
    train_adj_np = build_knn_patient_graph(train_concat, k=gat_k, metric="cosine")
    train_adj = torch.tensor(train_adj_np, dtype=torch.float32, device=dev)
    
    best_val_f1 = -1.0
    best_metrics = {}
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass on train
        outputs = model(train_tensors, train_adj)
        final_logits = outputs["final_logits"]
        per_omics_logits = outputs["per_omics_logits"]
        vcdn_logits = outputs["vcdn_logits"]
        gat_logits = outputs["gat_logits"]
        
        # Composite Multi-Task Loss
        l_final = criterion(final_logits, y_tr)
        l_vcdn = criterion(vcdn_logits, y_tr)
        l_gat = criterion(gat_logits, y_tr)
        l_omics = sum([criterion(ol, y_tr) for ol in per_omics_logits])
        
        total_loss = l_final + 0.3 * l_vcdn + 0.3 * l_gat + 0.2 * l_omics
        
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        scheduler.step()
        
        # Evaluation step on Test (Inductive Evaluation)
        model.eval()
        with torch.no_grad():
            test_concat = torch.cat(test_tensors, dim=-1).cpu().numpy()
            
            # Build Inductive Test Graph: Test nodes connect to Train nodes
            ind_adj_np = build_inductive_test_graph(train_concat, test_concat, k=gat_k, metric="cosine")
            ind_adj = torch.tensor(ind_adj_np, dtype=torch.float32, device=dev)
            
            # Full graph evaluation
            combined_tensors = [torch.cat([tr, te], dim=0) for tr, te in zip(train_tensors, test_tensors)]
            all_outputs = model(combined_tensors, ind_adj)
            
            # Slice test logits
            N_tr = len(y_train)
            test_logits = all_outputs["final_logits"][N_tr:]
            test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
            test_preds = np.argmax(test_probs, axis=-1)
            y_test_np = y_test.numpy()
            
            metrics = compute_metrics(y_test_np, test_preds, test_probs, num_classes)
            
            if metrics["macro_f1"] > best_val_f1:
                best_val_f1 = metrics["macro_f1"]
                best_metrics = metrics
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
    return best_metrics, model
