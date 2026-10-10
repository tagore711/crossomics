import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold

class MultiOmicsDataset:
    """
    Leakage-free TCGA Multi-Omics Dataset Loader & Preprocessor.
    
    Supports:
      - Loading datasets from HOPE/data/<cancer_type>/
      - Modalities: mRNA, Methylation (MET), Copy Number Alterations (CNA), RPPA / miRNA
      - Strict Train-Only Feature Selection & Normalization
    """
    def __init__(self, data_dir: str, cancer_type: str = "lgg", 
                 modalities: list = ["mrna", "met", "cna"], 
                 top_k_features: int = 500):
        self.data_dir = data_dir
        self.cancer_type = cancer_type.lower()
        self.modalities = modalities
        self.top_k_features = top_k_features
        
        self.raw_data = {}
        self.patient_ids = []
        self.labels = None
        self.num_classes = 0
        self.class_names = []
        
        self._load_data()
        
    def _load_data(self):
        cancer_path = os.path.join(self.data_dir, self.cancer_type)
        if not os.path.exists(cancer_path):
            raise FileNotFoundError(f"Cancer directory not found: {cancer_path}")
            
        # Load Subtype labels
        subtype_file = os.path.join(cancer_path, "subtype_data.csv")
        subtype_df = pd.read_csv(subtype_file)
        
        # Determine label column
        if 'class' in subtype_df.columns:
            y_raw = subtype_df['class'].values
        elif 'SUBTYPE' in subtype_df.columns:
            y_raw = subtype_df['SUBTYPE'].values
        else:
            y_raw = subtype_df.iloc[:, 1].values
            
        # Ensure integer encoding [0, ..., C-1]
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(y_raw)
        self.num_classes = len(np.unique(self.labels))
        self.class_names = [str(c) for c in self.label_encoder.classes_]
        
        # Load each omics modality
        for mod in self.modalities:
            file_candidates = [
                f"{mod}_data.csv",
                f"{mod}.csv",
                f"TCGA_{self.cancer_type}_{mod}.csv"
            ]
            loaded = False
            for fc in file_candidates:
                fp = os.path.join(cancer_path, fc)
                if os.path.exists(fp):
                    df = pd.read_csv(fp)
                    # Exclude sample ID / non-numeric columns if first col is ID
                    if df.iloc[:, 0].dtype == object:
                        feat_df = df.iloc[:, 1:]
                    else:
                        feat_df = df
                    self.raw_data[mod] = feat_df.values.astype(np.float32)
                    loaded = True
                    break
            if not loaded:
                raise FileNotFoundError(f"Could not find data for modality '{mod}' in {cancer_path}")

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42):
        """
        Creates a train-test split with strictly leakage-free normalization and feature selection.
        """
        num_samples = len(self.labels)
        indices = np.arange(num_samples)
        
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=self.labels
        )
        
        return self._process_splits(train_idx, test_idx)

    def get_kfold_splits(self, n_splits: int = 5, random_state: int = 42):
        """
        Generator yielding train/test folds with strict train-only feature selection.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(self.labels)), self.labels)):
            processed = self._process_splits(train_idx, test_idx)
            yield fold, processed

    def _process_splits(self, train_idx, test_idx):
        processed_train = []
        processed_test = []
        feature_selectors = []
        selected_dims = []

        for mod in self.modalities:
            X_mod = self.raw_data[mod]
            X_tr = X_mod[train_idx]
            X_te = X_mod[test_idx]

            # 1. Variance Threshold on Train only
            var_thresh = VarianceThreshold(threshold=1e-4)
            X_tr_v = var_thresh.fit_transform(X_tr)
            X_te_v = var_thresh.transform(X_te)

            # 2. Train-only Feature Selection via ANOVA F-value
            k = min(self.top_k_features, X_tr_v.shape[1])
            selector = SelectKBest(score_func=f_classif, k=k)
            X_tr_sel = selector.fit_transform(X_tr_v, self.labels[train_idx])
            X_te_sel = selector.transform(X_te_v)

            # 3. Train-only Standard Scaling
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_sel)
            X_te_scaled = scaler.transform(X_te_sel)

            processed_train.append(torch.tensor(X_tr_scaled, dtype=torch.float32))
            processed_test.append(torch.tensor(X_te_scaled, dtype=torch.float32))
            selected_dims.append(k)

        y_train = torch.tensor(self.labels[train_idx], dtype=torch.long)
        y_test = torch.tensor(self.labels[test_idx], dtype=torch.long)

        return {
            "train_data": processed_train,
            "test_data": processed_test,
            "y_train": y_train,
            "y_test": y_test,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "feature_dims": selected_dims,
            "num_classes": self.num_classes
        }
