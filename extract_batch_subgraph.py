import torch
import numpy as np
import sys

def extract_subgraph_for_batch(adj_matrix, graph_features, batch_indices):
    """
    Extracts a subgraph for a batch using the adjacency matrix.
    
    Args:
        adj_matrix (Tensor): Full adjacency matrix [N, N]
        graph_features (Tensor): Feature matrix [N, D]
        batch_indices (Tensor): Indices of the batch nodes [B]

    Returns:
        Tensor: Subgraph adjacency matrix [B, B]
    """
    # Ensure indices are in list format
    if isinstance(batch_indices, torch.Tensor):
        batch_indices = batch_indices.tolist()

    # Slice adjacency matrix
    adj_sub = adj_matrix[batch_indices][:, batch_indices]  # [B, B]
    return adj_sub