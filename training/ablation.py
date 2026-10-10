import numpy as np
import pandas as pd
import torch
from training.train import train_interaction_model

def run_comprehensive_ablation_study(dataset_splits: list, num_classes: int, feature_dims: list, 
                                     cancer_type: str = "LGG", epochs: int = 100, device: str = "cpu"):
    """
    Executes the full suite of baselines and ablation configurations across folds/splits.
    """
    configs = [
        # Baselines
        {"name": "Baseline 1 (Single-Omics mRNA)", "mode": "single_omics", "use_inter": False, "gating": False, "pairs": None, "triple": False},
        {"name": "Baseline 2 (Transformer + Concat)", "mode": "concat_only", "use_inter": False, "gating": False, "pairs": None, "triple": False},
        {"name": "Baseline 3 (Transformer + VCDN)", "mode": "no_gat", "use_inter": False, "gating": False, "pairs": None, "triple": False},
        {"name": "Baseline 4 (Transformer + GAT)", "mode": "no_vcdn", "use_inter": False, "gating": False, "pairs": None, "triple": False},
        {"name": "Baseline 5 (Trans + VCDN + GAT w/o Inter)", "mode": "full", "use_inter": False, "gating": False, "pairs": None, "triple": False},
        
        # Interaction Ablations
        {"name": "Ablation B (mRNA x Modality2 Only)", "mode": "full", "use_inter": True, "gating": True, "pairs": [0], "triple": False},
        {"name": "Ablation C (mRNA x Modality3 Only)", "mode": "full", "use_inter": True, "gating": True, "pairs": [1], "triple": False},
        {"name": "Ablation D (Modality2 x Modality3 Only)", "mode": "full", "use_inter": True, "gating": True, "pairs": [2], "triple": False},
        {"name": "Ablation F (Ungated Hadamard Inter)", "mode": "full", "use_inter": True, "gating": False, "pairs": None, "triple": False},
        {"name": "Ablation G (Triple Interaction)", "mode": "full", "use_inter": True, "gating": True, "pairs": None, "triple": True},
        
        # Proposed Full Model
        {"name": "PROPOSED (Gated Inter + VCDN + GAT)", "mode": "full", "use_inter": True, "gating": True, "pairs": None, "triple": False},
    ]
    
    results = []
    
    print(f"\n=======================================================")
    print(f"  RUNNING COMPREHENSIVE EXPERIMENTAL STUDY FOR {cancer_type.upper()}")
    print(f"=======================================================\n")
    
    for cfg in configs:
        fold_metrics = []
        for fold, split in enumerate(dataset_splits):
            train_data = split["train_data"]
            test_data = split["test_data"]
            y_train = split["y_train"]
            y_test = split["y_test"]
            dims = split["feature_dims"]
            
            # For single-omics baseline, isolate first modality (mRNA)
            if cfg["mode"] == "single_omics":
                tr_d = [train_data[0]]
                te_d = [test_data[0]]
                f_dims = [dims[0]]
            else:
                tr_d = train_data
                te_d = test_data
                f_dims = dims
                
            metrics, _ = train_interaction_model(
                train_data=tr_d,
                y_train=y_train,
                test_data=te_d,
                y_test=y_test,
                num_classes=num_classes,
                feature_dims=f_dims,
                epochs=epochs,
                use_interactions=cfg["use_inter"],
                use_gating=cfg["gating"],
                active_pairs=cfg["pairs"],
                include_triple=cfg["triple"],
                ablation_mode=cfg["mode"],
                device=device
            )
            fold_metrics.append(metrics)
            
        acc_vals = [m["accuracy"] for m in fold_metrics]
        f1_vals = [m["macro_f1"] for m in fold_metrics]
        wf1_vals = [m["weighted_f1"] for m in fold_metrics]
        bacc_vals = [m["balanced_accuracy"] for m in fold_metrics]
        auc_vals = [m["auroc"] for m in fold_metrics]
        
        summary = {
            "Model Architecture": cfg["name"],
            "Accuracy": f"{np.mean(acc_vals):.4f} ± {np.std(acc_vals):.4f}",
            "Macro-F1": f"{np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}",
            "Weighted-F1": f"{np.mean(wf1_vals):.4f} ± {np.std(wf1_vals):.4f}",
            "Balanced Acc": f"{np.mean(bacc_vals):.4f} ± {np.std(bacc_vals):.4f}",
            "AUROC": f"{np.mean(auc_vals):.4f} ± {np.std(auc_vals):.4f}"
        }
        results.append(summary)
        print(f"-> {cfg['name']}: Accuracy = {summary['Accuracy']} | Macro-F1 = {summary['Macro-F1']}")
        
    df_results = pd.DataFrame(results)
    return df_results
