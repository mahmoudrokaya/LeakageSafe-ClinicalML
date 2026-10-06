import torch
import numpy as np

# === Configuration ===
full_adj_path = 'E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project/data/preprocessed/adj_matrix_knn.npy'
full_features_path = 'E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project/data/preprocessed/graph_features.npy'
batch_indices = [0, 3, 7, 12, 20]  # Replace with dynamic batch from DataLoader

# === Load full data ===
adj_full = np.load(full_adj_path)
features_full = np.load(full_features_path)

# === Convert to tensors ===
adj_full_tensor = torch.tensor(adj_full, dtype=torch.float32)
features_tensor = torch.tensor(features_full, dtype=torch.float32)
batch_indices_tensor = torch.tensor(batch_indices)

# === Extract batch subgraph ===
adj_batch = adj_full_tensor[batch_indices_tensor][:, batch_indices_tensor]
features_batch = features_tensor[batch_indices_tensor]

# === Save (optional) ===
np.save('adj_matrix_batch.npy', adj_batch.numpy())
np.save('graph_features_batch.npy', features_batch.numpy())

print("Subgraph extracted successfully.")
print(f"Adjacency shape: {adj_batch.shape}")
print(f"Features shape: {features_batch.shape}")
