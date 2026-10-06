import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.classifiers.hfagm_model import HFAGM  # Ensure this path matches your project

# === Load Data ===
base_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project"
data_path = os.path.join(base_path, "data", "preprocessed")

X = np.load(os.path.join(data_path, "contrastive_embeddings.npy"))
graph_features = np.load(os.path.join(data_path, "graph_features.npy"))
adj_matrix = np.load(os.path.join(data_path, "adj_matrix_knn.npy"))
y = np.loadtxt(os.path.join(data_path, "y_train.csv"), delimiter=",", skiprows=1)  # ensure no header row

# === Train/Val Split ===
split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
graph_train, graph_val = graph_features[:split], graph_features[split:]
y_train, y_val = y[:split], y[split:]

# === Convert to Torch ===
X_train = torch.tensor(X_train, dtype=torch.float32)
graph_train = torch.tensor(graph_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
graph_val = torch.tensor(graph_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

# === Model Setup ===
input_dim = X_train.shape[1]
graph_dim = graph_train.shape[1]
hidden_dim = 64
num_classes = len(np.unique(y))

model = HFAGM(input_dim, graph_dim, hidden_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# === Train ===
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    logits, attn_weights = model(X_train, graph_train, adj_matrix)
    loss = criterion(logits, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Eval ===
model.eval()
with torch.no_grad():
    logits_val, attn_val = model(X_val, graph_val, adj_matrix)
    preds = torch.argmax(logits_val, dim=1)
    acc = (preds == y_val).float().mean().item()
    print(f"\nValidation Accuracy: {acc:.4f}")

# === Save Attention Plot ===
attn_dir = os.path.join(base_path, "outputs", "attention_weights")
plt.figure(figsize=(10, 6))
sns.heatmap(attn_val.numpy(), cmap='viridis')
plt.title("Attention Weights")
plt.xlabel("Head")
plt.ylabel("Node Index")
attn_path = os.path.join(attn_dir, "attention_weights.png")
plt.tight_layout()
plt.savefig(attn_path)
plt.close()
print(f"Saved attention weights to {attn_path}")

# === Save Confusion Matrix and Report ===
conf_mat = confusion_matrix(y_val.numpy(), preds.numpy())
report = classification_report(y_val.numpy(), preds.numpy(), output_dict=True)

conf_dir = os.path.join(base_path, "outputs", "evaluation", "confusion_matrix")
report_dir = os.path.join(base_path, "outputs", "evaluation", "classification_report")

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Confusion Matrix")
plt.ylabel("True")
plt.xlabel("Predicted")
plt.tight_layout()
conf_path = os.path.join(conf_dir, "confusion_matrix.png")
plt.savefig(conf_path)
plt.close()
print(f"Saved confusion matrix to {conf_path}")

# Classification Report
report_path = os.path.join(report_dir, "classification_report.csv")
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(report_path)
print(f"Saved classification report to {report_path}")

# === Save Model ===
model_path = os.path.join(base_path, "saved_models", "classifiers", "hfagm_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
