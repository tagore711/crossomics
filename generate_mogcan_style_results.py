import os
import time
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data_utils.dataset_loader import MultiOmicsDataset
from models.interaction_aware_model import InteractionAwareMultiOmicsModel
from data_utils.graph_builder import build_knn_patient_graph, build_inductive_test_graph

def save_confusion_matrix_plot(cm, class_names, output_path, title="Confusion Matrix"):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Prediction', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()

def run_evaluation_pipeline(cancer_type="lgg", modalities=["mrna", "met", "cna", "rppa"], 
                            epochs=100, device="cpu"):
    """
    Trains and evaluates our Interaction-Aware Multi-Omics Transformer-VCDN-GAT model
    and exports exact MO-GCAN style reports, logs, confusion matrix PNGs, and model .pkl files.
    """
    print(f"\n=======================================================")
    print(f"  PROCESSING CANCER TYPE: {cancer_type.upper()}")
    print(f"=======================================================")
    
    # 1. Check available modalities in folder
    data_path = os.path.join("data", cancer_type)
    valid_mods = []
    for m in modalities:
        if os.path.exists(os.path.join(data_path, f"{m}_data.csv")):
            valid_mods.append(m)
            
    if not valid_mods:
        raise FileNotFoundError(f"No valid data files found for {cancer_type}")
        
    dataset = MultiOmicsDataset(
        data_dir="data",
        cancer_type=cancer_type,
        modalities=valid_mods,
        top_k_features=500
    )
    
    # 75% Train / 25% Test split (consistent with MO-GCAN standard split)
    split_data = dataset.get_train_test_split(test_size=0.25, random_state=2025)
    
    dev = torch.device(device)
    train_tensors = [x.to(dev) for x in split_data["train_data"]]
    test_tensors = [x.to(dev) for x in split_data["test_data"]]
    y_train = split_data["y_train"].to(dev)
    y_test = split_data["y_test"].to(dev)
    num_classes = split_data["num_classes"]
    feature_dims = split_data["feature_dims"]
    class_names = [str(c) for c in range(num_classes)]
    
    # Target result directories
    res_dir = os.path.join("result", cancer_type)
    cm_dir = os.path.join(res_dir, "confusion_matrix")
    os.makedirs(cm_dir, exist_ok=True)
    
    log_file_path = os.path.join(res_dir, f"{cancer_type}.log")
    log_f = open(log_file_path, "w")
    
    # Build Train Graph
    train_concat = torch.cat(train_tensors, dim=-1).cpu().numpy()
    train_adj_np = build_knn_patient_graph(train_concat, k=10, metric="cosine")
    train_adj = torch.tensor(train_adj_np, dtype=torch.float32, device=dev)
    
    # --- A. Single-Omics Evaluation & Training ---
    single_omics_accs = {}
    for i, mod in enumerate(valid_mods):
        print(f"Training Single-Omics Model for {mod.upper()}...")
        single_model = InteractionAwareMultiOmicsModel(
            in_feature_dims=[feature_dims[i]],
            num_classes=num_classes,
            d_model=128,
            use_interactions=False,
            ablation_mode="single_omics"
        ).to(dev)
        
        opt_s = torch.optim.AdamW(single_model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit_s = nn.CrossEntropyLoss()
        
        # Single view patient graph
        mod_train_np = train_tensors[i].cpu().numpy()
        mod_train_adj = torch.tensor(build_knn_patient_graph(mod_train_np, k=10), dtype=torch.float32, device=dev)
        
        for ep in range(epochs):
            single_model.train()
            opt_s.zero_grad()
            out_s = single_model([train_tensors[i]], mod_train_adj)
            loss_s = crit_s(out_s["final_logits"], y_train)
            loss_s.backward()
            opt_s.step()
            
        single_model.eval()
        with torch.no_grad():
            mod_test_np = test_tensors[i].cpu().numpy()
            ind_adj_s = torch.tensor(build_inductive_test_graph(mod_train_np, mod_test_np, k=10), dtype=torch.float32, device=dev)
            comb_tensors_s = [torch.cat([train_tensors[i], test_tensors[i]], dim=0)]
            all_s_out = single_model(comb_tensors_s, ind_adj_s)
            pred_s = all_s_out["final_logits"][len(y_train):].argmax(dim=-1).cpu().numpy()
            y_te_np = y_test.cpu().numpy()
            
            # Confusion matrix & log
            cm_s = confusion_matrix(y_te_np, pred_s)
            save_confusion_matrix_plot(cm_s, class_names, os.path.join(cm_dir, f"{mod}.png"), title=f"Confusion Matrix ({mod.upper()})")
            
            report_str = classification_report(y_te_np, pred_s, digits=4, zero_division=0)
            print(f"\nprediction for {mod} data: ")
            print(f"prediction for {mod} data: ", file=log_f)
            print(report_str)
            print(report_str, file=log_f)
            single_omics_accs[mod] = accuracy_score(y_te_np, pred_s)
            
            # Save model checkpoint pkl
            with open(os.path.join(res_dir, f"{mod}.pkl"), "wb") as f:
                pickle.dump(single_model.state_dict(), f)

    # --- B. Full Interaction-Aware Multi-Omics Model ---
    print("\nTraining Final Interaction-Aware Multi-Omics Model...")
    start_time = time.time()
    
    full_model = InteractionAwareMultiOmicsModel(
        in_feature_dims=feature_dims,
        num_classes=num_classes,
        d_model=128,
        use_interactions=True,
        use_gating=True
    ).to(dev)
    
    opt_full = torch.optim.AdamW(full_model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        full_model.train()
        opt_full.zero_grad()
        outputs = full_model(train_tensors, train_adj)
        l_final = criterion(outputs["final_logits"], y_train)
        l_vcdn = criterion(outputs["vcdn_logits"], y_train)
        l_gat = criterion(outputs["gat_logits"], y_train)
        l_omics = sum([criterion(ol, y_train) for ol in outputs["per_omics_logits"]])
        
        total_loss = l_final + 0.3 * l_vcdn + 0.3 * l_gat + 0.2 * l_omics
        total_loss.backward()
        opt_full.step()
        
    training_time = time.time() - start_time
    
    # Test Evaluation
    full_model.eval()
    with torch.no_grad():
        test_concat = torch.cat(test_tensors, dim=-1).cpu().numpy()
        ind_adj = torch.tensor(build_inductive_test_graph(train_concat, test_concat, k=10), dtype=torch.float32, device=dev)
        combined_tensors = [torch.cat([tr, te], dim=0) for tr, te in zip(train_tensors, test_tensors)]
        all_outputs = full_model(combined_tensors, ind_adj)
        
        final_test_logits = all_outputs["final_logits"][len(y_train):]
        final_preds = final_test_logits.argmax(dim=-1).cpu().numpy()
        y_te_np = y_test.cpu().numpy()
        
        cm_final = confusion_matrix(y_te_np, final_preds)
        save_confusion_matrix_plot(cm_final, class_names, os.path.join(cm_dir, "all the omics.png"), title="Confusion Matrix (All Omics)")
        
        final_report = classification_report(y_te_np, final_preds, digits=4, zero_division=0)
        print("\nprediction for all the omics data: ")
        print("prediction for all the omics data: ", file=log_f)
        print(final_report)
        print(final_report, file=log_f)
        
        print(f"Training time : {training_time:.6f} seconds")
        print(f"Training time : {training_time:.6f} seconds\n", file=log_f)
        
        # Save full model checkpoint pkl
        with open(os.path.join(res_dir, "all_omics.pkl"), "wb") as f:
            pickle.dump(full_model.state_dict(), f)
            
    log_f.close()
    final_acc = accuracy_score(y_te_np, final_preds)
    return {
        "cancer_type": cancer_type,
        "final_accuracy": final_acc,
        "single_omics_accs": single_omics_accs,
        "training_time": training_time
    }

def generate_comparative_summary_plots(summary_list):
    """
    Generates cross-cancer comparative bar charts: plot_ACC.png and plot_time.png in result/
    """
    os.makedirs("result", exist_ok=True)
    cancers = [s["cancer_type"].upper() for s in summary_list]
    accs = [s["final_accuracy"] * 100 for s in summary_list]
    times = [s["training_time"] for s in summary_list]
    
    # 1. Accuracy Plot
    plt.figure(figsize=(9, 5))
    bars = plt.bar(cancers, accs, color="#2b5c8f", width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    plt.xlabel("TCGA Cancer Cohort", fontsize=12, fontweight="bold")
    plt.title("Multi-Omics Cancer Subtype Classification Accuracy (HOPE Architecture)", fontsize=14, fontweight="bold")
    plt.ylim(60, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig("result/plot_ACC.png", dpi=300)
    plt.close()
    
    # 2. Training Time Plot
    plt.figure(figsize=(9, 5))
    bars = plt.bar(cancers, times, color="#d95f02", width=0.55, edgecolor="black", linewidth=1.2)
    plt.ylabel("Training Time (Seconds)", fontsize=12, fontweight="bold")
    plt.xlabel("TCGA Cancer Cohort", fontsize=12, fontweight="bold")
    plt.title("Model Training Time Across TCGA Cohorts", fontsize=14, fontweight="bold")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.2f}s", ha='center', va='bottom', fontsize=10, fontweight="bold")
        
    plt.tight_layout()
    plt.savefig("result/plot_time.png", dpi=300)
    plt.close()
    
    print("\nSummary plots generated: result/plot_ACC.png, result/plot_time.png")

if __name__ == "__main__":
    cancer_types = ["lgg", "brca", "cesc", "coadread", "hnsc", "sarc", "stad", "ucec"]
    results = []
    
    for c_type in cancer_types:
        try:
            res = run_evaluation_pipeline(cancer_type=c_type, epochs=80)
            results.append(res)
        except Exception as e:
            print(f"Skipping {c_type} due to error: {e}")
            
    if results:
        generate_comparative_summary_plots(results)
        print("\nAll MO-GCAN style results generated successfully!")
