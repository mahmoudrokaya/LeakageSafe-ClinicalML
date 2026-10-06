import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Set path to import from models/layers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from models.classifiers.hfagm_model import HFAGM

# Load data
features_path     = "E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project/data/preprocessed/graph_features.npy"
contrastive_path  = "E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project/data/preprocessed/contrastive_embeddings.npy"
adj_path          = "E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project/data/preprocessed/adj_matrix_fully_connected.npy"
labels_path       = "E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project/data/preprocessed/y_train.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Load tensors
graph_features  = torch.tensor(np.load(features_path), dtype=torch.float32).to(device)
contrastive_emb = torch.tensor(np.load(contrastive_path), dtype=torch.float32).to(device)
adj_matrix      = torch.tensor(np.load(adj_path), dtype=torch.float32).to(device)
labels          = torch.tensor(pd.read_csv(labels_path).values.squeeze(), dtype=torch.long).to(device)

# Model config
input_dim   = contrastive_emb.shape[1]
graph_dim   = graph_features.shape[1]
hidden_dim  = 64
num_classes = len(torch.unique(labels))
batch_size  = 16
epochs      = 20

# Init model
model = HFAGM(input_dim, graph_dim, hidden_dim, num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Logging
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train()
    permutation = torch.randperm(labels.size(0), device=device)
    epoch_loss = 0.0
    epoch_preds = []
    epoch_targets = []

    for i in range(0, labels.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        x_attr = contrastive_emb[indices]
        x_graph = graph_features[indices]
        adj_sub = adj_matrix[indices][:, indices]
        target = labels[indices]

        outputs, attn = model(x_attr, x_graph, adj_sub)
        loss = criterion(outputs, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        epoch_preds.extend(predicted.cpu().numpy())
        epoch_targets.extend(target.cpu().numpy())

    acc = accuracy_score(epoch_targets, epoch_preds)
    train_losses.append(epoch_loss)
    train_accuracies.append(acc)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {acc:.4f}")

# === Plotting Results ===
plt.figure()
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("training_loss.png")

plt.figure()
plt.plot(range(1, epochs+1), train_accuracies, marker='x')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("training_accuracy.png")

print("Training complete. Plots saved as 'training_loss.png' and 'training_accuracy.png'")
