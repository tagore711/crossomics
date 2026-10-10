import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

def build_knn_patient_graph(features: np.ndarray, k: int = 10, metric: str = "cosine", 
                            sigma: float = 1.0, include_self_loop: bool = True) -> np.ndarray:
    """
    Constructs a k-Nearest Neighbors patient similarity adjacency matrix.
    
    Args:
        features: (N, D) patient feature matrix (or concatenated multi-omics)
        k: Number of nearest neighbors
        metric: 'cosine' or 'euclidean'
        sigma: Scaling factor for Gaussian RBF kernel when using euclidean
        include_self_loop: If True, adds self-loops A_ii = 1.0
    Returns:
        adj: (N, N) symmetric adjacency matrix with edge weights in [0, 1]
    """
    N = features.shape[0]
    k = min(k, N - 1)
    
    if metric == "cosine":
        # Cosine similarity matrix in [-1, 1] -> shifted to [0, 1]
        sim_matrix = (cosine_similarity(features) + 1.0) / 2.0
    elif metric == "euclidean":
        dist_matrix = euclidean_distances(features)
        gamma = 1.0 / (2 * (sigma ** 2) * np.median(dist_matrix) ** 2 + 1e-8)
        sim_matrix = np.exp(-gamma * (dist_matrix ** 2))
    else:
        raise ValueError(f"Unsupported metric: {metric}")
        
    adj = np.zeros((N, N), dtype=np.float32)
    
    # Keep top-k neighbors per patient
    for i in range(N):
        top_k_idx = np.argsort(sim_matrix[i, :])[-(k + 1):] # includes self
        for j in top_k_idx:
            if i != j:
                adj[i, j] = sim_matrix[i, j]
                adj[j, i] = sim_matrix[i, j] # Symmetrize
                
    if include_self_loop:
        np.fill_diagonal(adj, 1.0)
        
    # Row normalization
    row_sum = adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    adj_norm = adj / row_sum
    
    return adj_norm

def build_inductive_test_graph(train_features: np.ndarray, test_features: np.ndarray, 
                               k: int = 10, metric: str = "cosine") -> np.ndarray:
    """
    Constructs an inductive graph for testing:
    Test nodes only connect to training nodes (and themselves), preventing test-test data leakage.
    
    Args:
        train_features: (N_train, D)
        test_features: (N_test, D)
    Returns:
        full_adj: (N_train + N_test, N_train + N_test)
    """
    N_tr = train_features.shape[0]
    N_te = test_features.shape[0]
    N_total = N_tr + N_te
    
    full_adj = np.zeros((N_total, N_total), dtype=np.float32)
    
    # 1. Train-Train subgraph
    train_adj = build_knn_patient_graph(train_features, k=k, metric=metric)
    full_adj[:N_tr, :N_tr] = train_adj
    
    # 2. Test-Train bipartite connections (top k training neighbors for each test patient)
    if metric == "cosine":
        cross_sim = (cosine_similarity(test_features, train_features) + 1.0) / 2.0
    else:
        dist = euclidean_distances(test_features, train_features)
        gamma = 1.0 / (2 * np.median(dist) ** 2 + 1e-8)
        cross_sim = np.exp(-gamma * (dist ** 2))
        
    for i in range(N_te):
        top_k_tr = np.argsort(cross_sim[i, :])[-k:]
        test_idx = N_tr + i
        for tr_idx in top_k_tr:
            w = cross_sim[i, tr_idx]
            full_adj[test_idx, tr_idx] = w
            full_adj[tr_idx, test_idx] = w
        full_adj[test_idx, test_idx] = 1.0 # self loop
        
    return full_adj
