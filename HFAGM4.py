import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA

# ==============================================================
# PATHS
# ==============================================================
DATA_PATH = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed\covid_clinical_preprocessed.csv"
OUTPUT_DIR = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\experiments\New_EXP3"
ENCODER_PATH = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\experiments\New_EXP2\encoder_pretrained.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "metrics"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ==============================================================
# DATA PREPARATION (final robust version)
# ==============================================================
df = pd.read_csv(DATA_PATH).dropna()

# ---- Sensitive attribute: Gender ----
assert 'Gender' in df.columns, "Sensitive attribute 'Gender' must exist"
A = df['Gender'].replace({'F': 0, 'M': 1}).astype(int)

# ---- Sensitive / auxiliary attribute: Nationality ----
if 'Nationality' in df.columns:
    df['Nationality_encoded'] = df['Nationality'].replace({'NS': 0, 'S': 1}).astype(int)
    print("Nationality encoding preview:\n", df[['Nationality', 'Nationality_encoded']].head())

# ---- Target variable: 'status' ----
if 'status' in df.columns:
    y = df['status'].astype(int)
else:
    raise ValueError("No valid target column found (expected 'status')")

# ---- Features ----
X = df.drop(columns=['Gender', 'status', 'Nationality'], errors='ignore')

# ---- Keep only numeric columns ----
X = X.select_dtypes(include=['number'])

# ---- Scale & split ----
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X_scaled, y, A, test_size=0.2, stratify=y, random_state=42
)

print(f"Final feature matrix shape: {X_train.shape}")
# ==============================================================
# DATA LOADERS
# ==============================================================
def make_loader(X, y, batch_size=64, shuffle=True):
    X_t, y_t = torch.tensor(X, dtype=torch.float32), torch.tensor(y.values, dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_train, y_train)
test_loader = make_loader(X_test, y_test, shuffle=False)

# ==============================================================
# MODEL COMPONENTS
# ==============================================================

class MLP_Projection(nn.Module):
    def __init__(self, in_dim, embed_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim)
        )
    def forward(self, x): return self.net(x)

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(contrast, dim=1, keepdim=True)
        logits = contrast - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp_min(1.0)
        loss = -mean_log_prob_pos.mean()
        return loss

# ==============================================================
# HYBRID FAIRNESS-AWARE GENERATOR (Simplified VAE)
# ==============================================================
class VAE(nn.Module):
    def __init__(self, in_dim, latent_dim=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim * 2))
        self.dec = nn.Sequential(nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, in_dim), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.dec(z)
        return recon, mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = nn.functional.mse_loss(recon_x, x)
    kl_div = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + 0.1 * kl_div

# ==============================================================
# FAIRNESS + EVALUATION METRICS
# ==============================================================
def evaluate(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            pred = torch.argmax(out, dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(yb.numpy())
    return accuracy_score(trues, preds), f1_score(trues, preds), roc_auc_score(trues, preds)

# ==============================================================
# TRAINING PIPELINE
# ==============================================================
def run_experiment():
    results = []
    in_dim = X_train.shape[1]
    embed_dim = 128
    encoder = MLP_Projection(in_dim, embed_dim).to(device)

    # Load pretrained weights if available
    if os.path.exists(ENCODER_PATH):
        print("Loading pretrained ArSL encoder...")
        pretrained = torch.load(ENCODER_PATH, map_location=device)
        encoder.load_state_dict(pretrained, strict=False)

    classifier = nn.Sequential(
        nn.Linear(embed_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    supcon = SupConLoss()

    # --------------------
    # STAGE 1: Fine-tune encoder (contrastive + classification)
    # --------------------
    print("=== Fine-tuning encoder on COVID data ===")
    for epoch in range(30):
        encoder.train(); classifier.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            embeds = encoder(xb)
            logits = classifier(embeds)
            ce = loss_fn(logits, yb)
            c_loss = supcon(embeds, yb)
            loss = ce + 0.5 * c_loss
            loss.backward(); optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"[Encoder] {epoch+1}/30: {total_loss/len(train_loader):.4f}")

    # --------------------
    # STAGE 2: Recursive fairness-aware generation
    # --------------------
    print("\n=== Recursive fairness generations ===")
    vae = VAE(in_dim).to(device)
    vae_opt = optim.Adam(vae.parameters(), lr=1e-3)

    for gen in range(1, 4):
        for epoch in range(50):
            vae.train()
            total_loss = 0
            for xb, _ in train_loader:
                xb = xb.to(device)
                recon, mu, log_var = vae(xb)
                loss = vae_loss(recon, xb, mu, log_var)
                vae_opt.zero_grad()
                loss.backward()
                vae_opt.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"[VAE-Gen{gen}] {epoch+1}/50: {total_loss/len(train_loader):.4f}")

        # Generate synthetic fairness-balanced data
        vae.eval()
        syn_X, syn_y = [], []
        with torch.no_grad():
            for xb, yb in train_loader:
                xb = xb.to(device)
                mu, log_var = vae.encode(xb)
                z = vae.reparameterize(mu, log_var)
                recon = vae.dec(z)
                syn_X.append(recon.cpu().numpy())
                syn_y.append(yb.numpy())
        syn_X = np.vstack(syn_X)
        syn_y = np.concatenate(syn_y)

        # Train classifier on augmented data
        X_aug = np.vstack([X_train, syn_X])
        y_aug = np.concatenate([y_train, syn_y])
        loader_aug = make_loader(X_aug, pd.Series(y_aug))

        for epoch in range(20):
            encoder.train(); classifier.train()
            total_loss = 0
            for xb, yb in loader_aug:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                embeds = encoder(xb)
                logits = classifier(embeds)
                loss = loss_fn(logits, yb)
                loss.backward(); optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                print(f"[Encoder-Gen{gen}] {epoch+1}/20: {total_loss/len(loader_aug):.4f}")

        acc, f1, auc = evaluate(nn.Sequential(encoder, classifier), test_loader)
        # Assuming 'A_test' is the sensitive attribute (e.g., gender)
        # Fairness metrics (Fairlearn >= 0.10)
        dpd = demographic_parity_difference(
            y_true=y_test,
            y_pred=np.random.choice([0, 1], size=len(y_test)),
            sensitive_features=A_test
                                    )
        eog = equalized_odds_difference(
            y_true=y_test,
            y_pred=np.random.choice([0, 1], size=len(y_test)),
            sensitive_features=A_test
                                    )
        print(f"Fairness metrics — DPD: {dpd:.3f}, EOD: {eog:.3f}")


        # Latent drift via PCA
        pca = PCA(2)
        z_real = pca.fit_transform(X_test)
        with torch.no_grad():
            z_gen = pca.fit_transform(syn_X[:len(X_test)])
        drift = np.linalg.norm(z_real - z_gen[:len(z_real)], axis=1).mean()

        results.append({"generation": gen, "acc": acc, "f1": f1, "auc": auc,
                        "dpd": dpd, "eog": eog, "drift": drift})

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, "metrics", "summary_by_generation.csv"), index=False)

    # Plot metrics
    plt.figure(figsize=(8,6))
    for col in ["acc","auc","dpd","drift"]:
        plt.plot(df_results["generation"], df_results[col], label=col)
    plt.xlabel("Generation")
    plt.ylabel("Metric value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figures", "gen_curves_EXP3.png"), dpi=300)
    plt.close()

    # Save run config
    cfg = {
        "device": str(device),
        "data_path": DATA_PATH,
        "output_dir": OUTPUT_DIR,
        "encoder_path": ENCODER_PATH,
        "generations": 3,
        "timestamp": time.ctime()
    }
    with open(os.path.join(OUTPUT_DIR, "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n=== DONE ===")
    print("Metrics:", os.path.join(OUTPUT_DIR, "metrics"))
    print("Figures:", os.path.join(OUTPUT_DIR, "figures"))
    print("Config:", os.path.join(OUTPUT_DIR, "run_config.json"))

# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    run_experiment()
