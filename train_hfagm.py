import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib
from models.classifiers.hfagm import HFAGM

# ==== Paths ====
base_path = r"E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project"
data_path = os.path.join(base_path, "data", "preprocessed")
saved_model_path = os.path.join(base_path, "saved_models", "classifiers", "hfagm_model.pt")
results_path = os.path.join(base_path, "outputs", "evaluation", "hfagm_results.csv")

# ==== Load Data ====
X = np.load(os.path.join(data_path, "graph_features.npy"))
adj = np.load(os.path.join(data_path, "adj_matrix.npy"))
contrastive = np.load(os.path.join(data_path, "contrastive_embeddings.npy"))
y = pd.read_csv(os.path.join(data_path, "y_train.csv"))['status'].values

# ==== Convert to torch tensors ====
X_tensor = torch.tensor(X, dtype=torch.float32)
adj_tensor = torch.tensor(adj, dtype=torch.float32)
contrastive_tensor = torch.tensor(contrastive, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

dataset = TensorDataset(X_tensor, adj_tensor, contrastive_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# ==== Initialize Model ====
model = HFAGM(input_dim=X.shape[1], contrastive_dim=contrastive.shape[1], hidden_dim=64)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ==== Training Loop ====
model.train()
epochs = 20
for epoch in range(epochs):
    epoch_loss = 0.0
    for X_batch, adj_batch, contrastive_batch, y_batch in dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch, adj_batch, contrastive_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")

# ==== Save Trained Model ====
os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
torch.save(model.state_dict(), saved_model_path)
print(f"Model saved to: {saved_model_path}")

# ==== Evaluation ====
model.eval()
with torch.no_grad():
    logits = model(X_tensor, adj_tensor, contrastive_tensor)
    probs = torch.sigmoid(logits).squeeze().numpy()
    preds = (probs >= 0.5).astype(int)

y_true = y_tensor.squeeze().numpy()

acc = accuracy_score(y_true, preds)
f1 = f1_score(y_true, preds)
auc = roc_auc_score(y_true, probs)

print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")

# ==== Save Evaluation Results ====
pd.DataFrame({
    'Accuracy': [acc],
    'F1-score': [f1],
    'AUC': [auc]
}).to_csv(results_path, index=False)
print(f"Results saved to: {results_path}")
