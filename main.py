import os
import argparse
import torch
import numpy as np

from data_utils.dataset_loader import MultiOmicsDataset
from training.train import train_interaction_model
from training.ablation import run_comprehensive_ablation_study
from training.interpretability import analyze_cross_omics_interpretability

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Interaction-Aware Multi-Omics Transformer–VCDN–GAT")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to HOPE multi-omics data directory")
    parser.add_argument("--cancer_type", "-t", type=str, default="lgg", 
                        choices=["lgg", "brca", "cesc", "coadread", "hnsc", "sarc", "stad", "ucec"],
                        help="Target TCGA cancer type")
    parser.add_argument("--modalities", nargs="+", default=["mrna", "met", "cna"], 
                        help="List of omics modalities to use (e.g. mrna met cna rppa)")
    parser.add_argument("--top_k_features", type=int, default=500, help="Number of selected features per modality")
    parser.add_argument("--d_model", type=int, default=128, help="Latent embedding dimension")
    parser.add_argument("--gat_k", type=int, default=10, help="Number of nearest neighbors for patient similarity graph")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--mode", type=str, choices=["train", "ablation", "interpret"], default="train",
                        help="Execution mode: train, ablation, or interpret")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation splits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    print(f"\n=======================================================")
    print(f"  INTERACTION-AWARE MULTI-OMICS TRANSFORMER–VCDN–GAT")
    print(f"  Cancer Type: {args.cancer_type.upper()}")
    print(f"  Modalities: {args.modalities}")
    print(f"  Top-K Features per Modality: {args.top_k_features}")
    print(f"  Device: {args.device}")
    print(f"=======================================================\n")
    
    # Load multi-omics dataset with zero-leakage pipeline
    dataset = MultiOmicsDataset(
        data_dir=args.data_dir,
        cancer_type=args.cancer_type,
        modalities=args.modalities,
        top_k_features=args.top_k_features
    )
    
    print(f"Total Samples: {len(dataset.labels)} | Subtype Classes: {dataset.num_classes} ({dataset.class_names})")
    
    if args.mode == "train":
        splits = list(dataset.get_kfold_splits(n_splits=args.n_splits, random_state=args.seed))
        fold_results = []
        
        for fold_idx, split in splits:
            print(f"\n--- Running Fold {fold_idx + 1}/{args.n_splits} ---")
            metrics, model = train_interaction_model(
                train_data=split["train_data"],
                y_train=split["y_train"],
                test_data=split["test_data"],
                y_test=split["y_test"],
                num_classes=split["num_classes"],
                feature_dims=split["feature_dims"],
                epochs=args.epochs,
                lr=args.lr,
                d_model=args.d_model,
                gat_k=args.gat_k,
                device=args.device
            )
            fold_results.append(metrics)
            print(f"Fold {fold_idx + 1} - Accuracy: {metrics['accuracy']:.4f}, Macro-F1: {metrics['macro_f1']:.4f}, Weighted-F1: {metrics['weighted_f1']:.4f}, AUROC: {metrics['auroc']:.4f}")
            
        accs = [m["accuracy"] for m in fold_results]
        f1s = [m["macro_f1"] for m in fold_results]
        wf1s = [m["weighted_f1"] for m in fold_results]
        aurocs = [m["auroc"] for m in fold_results]
        
        print("\n================ FINAL 5-FOLD CV RESULTS ================")
        print(f"Accuracy:    {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"Macro-F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"Weighted-F1: {np.mean(wf1s):.4f} ± {np.std(wf1s):.4f}")
        print(f"AUROC:       {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
        print("=========================================================\n")
        
    elif args.mode == "ablation":
        splits = [s[1] for s in dataset.get_kfold_splits(n_splits=args.n_splits, random_state=args.seed)]
        ablation_df = run_comprehensive_ablation_study(
            dataset_splits=splits,
            num_classes=dataset.num_classes,
            feature_dims=[args.top_k_features] * len(args.modalities),
            cancer_type=args.cancer_type,
            epochs=args.epochs,
            device=args.device
        )
        print("\n=== COMPLETE ABLATION BENCHMARK TABLE ===")
        try:
            print(ablation_df.to_markdown(index=False))
        except Exception:
            print(ablation_df.to_string(index=False))
        os.makedirs("results", exist_ok=True)
        ablation_df.to_csv(f"results/{args.cancer_type}_ablation_benchmark.csv", index=False)
        print(f"\nResults saved to results/{args.cancer_type}_ablation_benchmark.csv")
        
    elif args.mode == "interpret":
        split = dataset.get_train_test_split(test_size=0.2, random_state=args.seed)
        metrics, model = train_interaction_model(
            train_data=split["train_data"],
            y_train=split["y_train"],
            test_data=split["test_data"],
            y_test=split["y_test"],
            num_classes=split["num_classes"],
            feature_dims=split["feature_dims"],
            epochs=args.epochs,
            lr=args.lr,
            d_model=args.d_model,
            gat_k=args.gat_k,
            device=args.device
        )
        # Run interpretability on test set
        test_tensors = [x.to(args.device) for x in split["test_data"]]
        from data_utils.graph_builder import build_knn_patient_graph
        test_concat = torch.cat(test_tensors, dim=-1).cpu().numpy()
        test_adj = torch.tensor(build_knn_patient_graph(test_concat, k=args.gat_k), dtype=torch.float32, device=args.device)
        analyze_cross_omics_interpretability(model, test_tensors, test_adj, modality_names=args.modalities)

if __name__ == "__main__":
    main()
