import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import networkx as nx

# === Paths ===
input_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed"
output_path = input_path  # You can change if needed

# === Load Embeddings ===
embedding_file = os.path.join(input_path, "contrastive_embeddings.npy")
features = np.load(embedding_file)

# === Save Features as graph node features ===
np.save(os.path.join(output_path, "graph_features.npy"), features)
print("Saved: graph_features.npy")

# === Option 1: Fully connected graph with top 10% cosine similarities ===
cos_sim_matrix = cosine_similarity(features)
np.fill_diagonal(cos_sim_matrix, 0)  # Remove self-loops

# Threshold top 10% similarity values
threshold = np.percentile(cos_sim_matrix, 90)
adj_matrix_fc = (cos_sim_matrix >= threshold).astype(int)

np.save(os.path.join(output_path, "adj_matrix_fully_connected.npy"), adj_matrix_fc)
print("Saved: adj_matrix_fully_connected.npy")

# Print stats
G_fc = nx.from_numpy_array(adj_matrix_fc)
print("\n Fully Connected (Top 10%) Graph:")
print(f"  Nodes: {G_fc.number_of_nodes()}")
print(f"  Edges: {G_fc.number_of_edges()}")
print(f"  Average Degree: {np.mean([deg for _, deg in G_fc.degree()]):.2f}")

# === Option 2: K-Nearest Neighbors Graph (k = 10) ===
k = 10
nbrs = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(features)
distances, indices = nbrs.kneighbors(features)

adj_matrix_knn = np.zeros((features.shape[0], features.shape[0]), dtype=int)
for i, neighbors in enumerate(indices):
    for j in neighbors[1:]:  # Skip self (first entry)
        adj_matrix_knn[i, j] = 1
        adj_matrix_knn[j, i] = 1  # undirected

np.save(os.path.join(output_path, "adj_matrix_knn.npy"), adj_matrix_knn)
print("Saved: adj_matrix_knn.npy")

# Print stats
G_knn = nx.from_numpy_array(adj_matrix_knn)
print("\n KNN Graph (k=10):")
print(f"  Nodes: {G_knn.number_of_nodes()}")
print(f"  Edges: {G_knn.number_of_edges()}")
print(f"  Average Degree: {np.mean([deg for _, deg in G_knn.degree()]):.2f}")
