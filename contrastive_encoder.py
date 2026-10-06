import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os

# Paths
base_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed"
X_train = pd.read_csv(os.path.join(base_path, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))['status']

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_tensor = torch.tensor(y_train.values, dtype=torch.long)
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Encoder Model
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
    def forward(self, x):
        return self.model(x)

# Supervised Contrastive Loss
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits = torch.div(torch.matmul(features, features.T), self.temperature)
        logits -= torch.max(logits, dim=1, keepdim=True)[0].detach()
        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0)).to(device)
        mask *= logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return -mean_log_prob.mean()

# Initialize and train
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder(input_dim=X_tensor.shape[1]).to(device)
loss_fn = SupConLoss()
optimizer = optim.Adam(encoder.parameters(), lr=1e-3)

print("Training contrastive encoder...")
for epoch in range(50):
    total_loss = 0
    encoder.train()
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        z = nn.functional.normalize(encoder(x_batch), dim=1)
        loss = loss_fn(z, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# Save embeddings
encoder.eval()
with torch.no_grad():
    embeddings = encoder(X_tensor.to(device)).cpu().numpy()
    np.save(os.path.join(base_path, "contrastive_embeddings.npy"), embeddings)

print("Saved embeddings to contrastive_embeddings.npy")
