# -*- coding: utf-8 -*-
"""
HFAGM++ Clinical Experiment — Robust Single Runner
Fixes:
- Avoid empty splits; adaptive stratification with fallbacks
- No drop-all-rows: impute numeric NaNs with median
- Supervised contrastive loss rewritten to avoid shape mismatches
- No division-by-zero when 0 batches
Paths kept identical to user's request.
"""

import os, json, random, warnings
warnings.filterwarnings("ignore")

# ========= USER PATHS (unchanged) =========
INPUT_DIR = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed"
INPUT_FILE = "covid_clinical_preprocessed.csv"
OUT_DIR = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\experiments\New_EXPs"
# ==========================================

# ======= CONFIG =======
LABEL_COL     = "status"
PROTECTED_COL = "Gender"
DROP_COLS     = ["patient_id", "Nationality", "ID", "id"]

SEED_LIST  = [42, 47, 53]
TEST_SIZE  = 0.2           # will adapt down if needed
VAL_SIZE   = 0.2           # relative to train+val; will adapt down if needed
EMBED_DIM  = 64
BATCH_SIZE = 256
ENC_EPOCHS = 40
ENC_LR     = 1e-3
TAU        = 0.07
VAE_EPOCHS = 60
VAE_LATENT = 16
GEN_SIZES  = [0.3, 0.3, 0.3]
MIX_COEFFS = (0.6, 0.2, 0.2)
FAIR_TARGET_PARITY = True

# ======= Imports =======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from itertools import combinations

# ======= Utils =======
def ensure_dirs():
    for sub in ["metrics", "figures", "synthetic", "artifacts"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_data():
    path = os.path.join(INPUT_DIR, INPUT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}")
    df = pd.read_csv(path)

    # drop unneeded IDs
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # enforce presence
    if LABEL_COL not in df.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found.")
    if PROTECTED_COL not in df.columns:
        raise ValueError(f"Protected column '{PROTECTED_COL}' not found.")

    # encode protected if categorical
    if df[PROTECTED_COL].dtype == object:
        df[PROTECTED_COL] = df[PROTECTED_COL].astype("category").cat.codes

    # coerce numerics for all non label/protected
    for c in df.columns:
        if c not in [LABEL_COL, PROTECTED_COL]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # keep rows that have label/protected
    df = df.dropna(subset=[LABEL_COL, PROTECTED_COL])

    # impute numeric NaNs (don't drop all rows!)
    df = df.fillna(df.median(numeric_only=True))

    # ensure ints
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df[PROTECTED_COL] = df[PROTECTED_COL].astype(int)

    if len(df) < 20:
        print(f" Warning: dataset is small ({len(df)} rows). Splits will adapt.")
    return df

def adaptive_splits(X, y, A, test_size, val_size, seed):
    """Try stratified split; if fails (few samples / single class), reduce sizes and/or drop stratify."""
    # ensure y has at least 2 classes
    if len(np.unique(y)) < 2:
        raise ValueError("Dataset has a single class in label; cannot run classification.")

    # progressively reduce test/val if necessary
    ts_list = [test_size, 0.15, 0.1, 0.05]
    vs_list = [val_size, 0.15, 0.1, 0.05]

    last_err = None
    for ts in ts_list:
        try:
            X_tv, X_test, y_tv, y_test, A_tv, A_test = train_test_split(
                X, y, A, test_size=ts, random_state=seed, stratify=y
            )
            val_ratio = vs_list[0] / (1.0 - ts)
            try:
                X_tr, X_val, y_tr, y_val, A_tr, A_val = train_test_split(
                    X_tv, y_tv, A_tv, test_size=val_ratio, random_state=seed, stratify=y_tv
                )
                return X_tr, X_val, X_test, y_tr, y_val, y_test, A_tr, A_val, A_test
            except Exception as e2:
                last_err = e2
                # try smaller val
                for vs in vs_list[1:]:
                    val_ratio = vs / (1.0 - ts)
                    try:
                        X_tr, X_val, y_tr, y_val, A_tr, A_val = train_test_split(
                            X_tv, y_tv, A_tv, test_size=val_ratio, random_state=seed, stratify=y_tv
                        )
                        return X_tr, X_val, X_test, y_tr, y_val, y_test, A_tr, A_val, A_test
                    except Exception as e3:
                        last_err = e3
        except Exception as e1:
            last_err = e1
            continue

    # fallback: no stratify
    print("Falling back to non-stratified split (data too small/imbalanced).")
    X_tv, X_test, y_tv, y_test, A_tv, A_test = train_test_split(
        X, y, A, test_size=ts_list[-1], random_state=seed, stratify=None
    )
    val_ratio = vs_list[-1] / (1.0 - ts_list[-1])
    X_tr, X_val, y_tr, y_val, A_tr, A_val = train_test_split(
        X_tv, y_tv, A_tv, test_size=val_ratio, random_state=seed, stratify=None
    )
    if len(np.unique(y_tr)) < 2:
        print("Train set ended up single-class; training may be unstable.")
    return X_tr, X_val, X_test, y_tr, y_val, y_test, A_tr, A_val, A_test

def stratified_splits(df, seed=42):
    X = df.drop(columns=[LABEL_COL])
    y = df[LABEL_COL].values
    A = df[PROTECTED_COL].values

    X_tr, X_val, X_te, y_tr, y_val, y_te, A_tr, A_val, A_te = adaptive_splits(
        X, y, A, TEST_SIZE, VAL_SIZE, seed
    )

    scaler = StandardScaler()
    feature_cols = [c for c in X.columns if c != PROTECTED_COL]
    X_tr_s = scaler.fit_transform(X_tr[feature_cols])
    X_val_s = scaler.transform(X_val[feature_cols])
    X_te_s = scaler.transform(X_te[feature_cols])

    return (X_tr_s, y_tr, A_tr,
            X_val_s, y_val, A_val,
            X_te_s, y_te, A_te,
            scaler, feature_cols)

# ======= Contrastive Encoder =======
class TabularDataset(Dataset):
    def __init__(self, X, y, k=2, noise_std=0.05, train=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.k = k
        self.noise = noise_std
        self.train = train
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx]; y = self.y[idx]
        if self.train:
            views = [x + torch.randn_like(x)*self.noise for _ in range(self.k)]
            return torch.stack(views, dim=0), y
        return x.unsqueeze(0), y

class EncoderMLP(nn.Module):
    def __init__(self, in_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )
    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=-1)

def supcon_loss(z_flat, y, K=2, tau=0.07):
    """
    z_flat: [B*K, d], y: [B], K: views
    Implements supervised contrastive loss (Khosla et al. 2020) robustly.
    """
    device = z_flat.device
    B = y.shape[0]
    y = y.view(B, 1)
    mask = torch.eq(y, y.T).float().to(device)  # [B, B]

    # Expand mask to [B*K, B*K]
    mask = mask.repeat_interleave(K, dim=0).repeat_interleave(K, dim=1)

    # Compute cosine similarity
    sim = torch.matmul(z_flat, z_flat.T) / tau
    # Remove self-contrast
    logits_mask = torch.ones_like(sim) - torch.eye(B*K, device=device)
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    # Positive mask (exclude self)
    positive_mask = mask * logits_mask

    # For samples with no positives, avoid NaN by denominator clamp
    pos_count = positive_mask.sum(dim=1)
    mean_log_pos = (positive_mask * log_prob).sum(dim=1) / torch.clamp(pos_count, min=1.0)
    loss = -mean_log_pos.mean()
    return loss

def train_encoder(Xtr, ytr, Xv, yv, emb_dim, epochs, lr, tau, device):
    if len(Xtr) < 2:
        raise ValueError("Training set too small.")
    model = EncoderMLP(Xtr.shape[1], emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    bs = min(BATCH_SIZE, max(2, len(Xtr)))     # ensure >=2
    train_loader = DataLoader(TabularDataset(Xtr, ytr, k=2, train=True),
                              batch_size=bs, shuffle=True, drop_last=False)

    for ep in range(epochs):
        model.train(); tot, nb = 0.0, 0
        for x_aug, y in train_loader:
            # x_aug: [B, K, d] → flatten to [B*K, d]
            Bbatch, K, d = x_aug.shape
            x = x_aug.view(Bbatch*K, d).to(device)
            y = y.to(device)

            z = model(x)
            loss = supcon_loss(z, y, K=K, tau=tau)

            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1

        if (ep+1) % 10 == 0 and nb > 0:
            print(f"[Encoder] {ep+1}/{epochs}: {tot/nb:.4f}")
    return model

def get_embeddings(model, X, device):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        Z = model(Xt).cpu().numpy()
    return Z

# ======= Ensemble =======
def fit_ensemble(Ztr, ytr, Zval, yval):
    rf  = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    knn = KNeighborsClassifier(n_neighbors=5)
    mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=42)
    for clf in (rf, knn, mlp):
        clf.fit(Ztr, ytr)
    accs = [accuracy_score(yval, clf.predict(Zval)) for clf in (rf, knn, mlp)]
    w = np.exp(10*np.array(accs)); w = w / w.sum()
    return (rf, knn, mlp), w

def ensemble_predict(models, w, Z):
    P = np.zeros((Z.shape[0], 2))
    for m, wi in zip(models, w):
        # Some sklearn models may lack predict_proba if trained on single-class batch: guard it.
        if hasattr(m, "predict_proba"):
            P += wi * m.predict_proba(Z)
        else:
            preds = m.predict(Z)
            P += wi * np.stack([1-preds, preds], axis=1)
    return P / np.clip(w.sum(), 1e-9, None)

def ensemble_disagreement(models, Z):
    preds = np.array([m.predict(Z) for m in models])  # [M, N]
    pairs = list(combinations(range(preds.shape[0]), 2))
    if not pairs: return 0.0
    return float(np.mean([np.mean(preds[i] != preds[j]) for i, j in pairs]))

# ======= Metrics =======
def compute_metrics(y_true, y_prob, A):
    y_pred = (y_prob[:,1] >= 0.5).astype(int)
    out = dict(
        accuracy = accuracy_score(y_true, y_pred),
        f1       = f1_score(y_true, y_pred, zero_division=0),
        auc      = roc_auc_score(y_true, y_prob[:,1]) if len(np.unique(y_true))>1 else np.nan,
        dpd      = demographic_parity_difference(y_true, y_pred, sensitive_features=A),
        eod      = equalized_odds_difference(y_true, y_pred, sensitive_features=A),
    )
    return out

# ======= Simple VAE for tabular synthesis =======
class VAE(nn.Module):
    def __init__(self, in_dim, z_dim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(),
                                 nn.Linear(128,64), nn.ReLU())
        self.mu, self.logv = nn.Linear(64,z_dim), nn.Linear(64,z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim,64), nn.ReLU(),
                                 nn.Linear(64,128), nn.ReLU(),
                                 nn.Linear(128,in_dim))
    def forward(self, x):
        h = self.enc(x); mu, logv = self.mu(h), self.logv(h)
        std = torch.exp(0.5*logv); z = mu + std*torch.randn_like(std)
        xr = self.dec(z); return xr, mu, logv

def vae_loss(x, xr, mu, logv):
    recon = F.mse_loss(xr, x)
    kld   = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return recon + 1e-3*kld

def train_vae(X, epochs, zdim, device):
    vae = VAE(X.shape[1], z_dim=zdim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32).to(device)
    for ep in range(epochs):
        xr, mu, logv = vae(Xt)
        loss = vae_loss(Xt, xr, mu, logv)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 15 == 0:
            print(f"[VAE] {ep+1}/{epochs}: {loss.item():.4f}")
    return vae

def sample_vae(vae, n, device):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n, vae.mu.out_features, device=device)
        Xs = vae.dec(z).cpu().numpy()
    return Xs

def parity_adjust_thresholds(y_prob, A):
    groups = np.unique(A)
    rates = [np.mean((y_prob[A==g,1] >= 0.5).astype(int)) for g in groups if np.sum(A==g)>0]
    target = np.mean(rates) if len(rates)>0 else 0.5
    thr = {}
    for g in groups:
        scores = y_prob[A==g,1]
        thr[g] = float(np.quantile(scores, 1-target)) if len(scores)>0 else 0.5
    return thr

def apply_group_thresholds(y_prob, A, thresholds):
    y_pred = np.zeros(y_prob.shape[0], dtype=int)
    for g, t in thresholds.items():
        mask = (A==g)
        y_pred[mask] = (y_prob[mask,1] >= t).astype(int)
    return y_pred

def make_synthetic_round(X_tr, y_tr, A_tr, size_frac, device):
    n_syn = int(size_frac * len(X_tr))
    if n_syn < 10:
        return np.empty((0, X_tr.shape[1])), np.empty(0,int), np.empty(0,int)
    vae = train_vae(X_tr, VAE_EPOCHS, VAE_LATENT, device)
    X_syn = sample_vae(vae, n_syn, device)

    # pseudo labels
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_tr, y_tr)
    y_prob = lr.predict_proba(X_syn)

    # synthetic sensitive attributes following train distribution
    A_vals, A_cnts = np.unique(A_tr, return_counts=True)
    A_syn = np.random.choice(A_vals, size=n_syn, p=(A_cnts/A_cnts.sum()))

    if FAIR_TARGET_PARITY:
        thr = parity_adjust_thresholds(y_prob, A_syn)
        y_syn = apply_group_thresholds(y_prob, A_syn, thr)
    else:
        y_syn = (y_prob[:,1] >= 0.5).astype(int)

    return X_syn, y_syn.astype(int), A_syn.astype(int)

def mix_pools(X_real, y_real, A_real, prev_syn, cur_syn):
    alpha, beta, gamma = MIX_COEFFS
    Xp, yp, Ap = prev_syn
    Xc, yc, Ac = cur_syn

    parts = []
    # sample alpha fraction of real
    n_real = max(1, int(alpha * len(X_real)))
    idx = np.random.choice(np.arange(len(X_real)), size=min(n_real, len(X_real)), replace=False)
    parts.append((X_real[idx], y_real[idx], A_real[idx]))

    if len(Xp) > 0 and beta > 0:
        n_prev = int(beta * len(X_real))
        pidx = np.random.choice(np.arange(len(Xp)), size=min(n_prev, len(Xp)), replace=False)
        parts.append((Xp[pidx], yp[pidx], Ap[pidx]))

    if len(Xc) > 0 and gamma > 0:
        n_cur = int(gamma * len(X_real))
        cidx = np.random.choice(np.arange(len(Xc)), size=min(n_cur, len(Xc)), replace=False)
        parts.append((Xc[cidx], yc[cidx], Ac[cidx]))

    Xmix = np.vstack([p[0] for p in parts])
    ymix = np.concatenate([p[1] for p in parts])
    Amix = np.concatenate([p[2] for p in parts])
    return Xmix, ymix, Amix

# ======= Plotting =======
def plot_gen_curves(df, outpath):
    plt.figure(figsize=(8,5))
    for m in ["accuracy", "auc", "dpd", "drift_z"]:
        plt.plot(df["generation"], df[m], marker='o', label=m)
    plt.xlabel("Generation (0=Real-only)")
    plt.title("HFAGM++ Generations: Accuracy/AUC/DPD/Δz")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

# ======= Main run for one seed =======
def run_one_seed(seed, device):
    set_seed(seed)
    df = load_data()
    (X_tr, y_tr, A_tr,
     X_val, y_val, A_val,
     X_te, y_te, A_te,
     scaler, feat_cols) = stratified_splits(df, seed)

    # Train encoder (robust batch size, no drop_last)
    enc = train_encoder(X_tr, y_tr, X_val, y_val, EMBED_DIM, ENC_EPOCHS, ENC_LR, TAU, device)
    Z_tr = get_embeddings(enc, X_tr, device)
    Z_val = get_embeddings(enc, X_val, device)
    Z_te = get_embeddings(enc, X_te, device)

    models, w = fit_ensemble(Z_tr, y_tr, Z_val, y_val)
    P0 = ensemble_predict(models, w, Z_te)
    m0 = compute_metrics(y_te, P0, A_te)
    rho0 = ensemble_disagreement(models, Z_te)
    mu_real = Z_te.mean(axis=0)

    rows = [{"generation":0, **m0, "drift_z":0.0, "rho":rho0}]
    prev_syn = (np.empty((0, X_tr.shape[1])), np.empty(0,int), np.empty(0,int))

    for gi, frac in enumerate(GEN_SIZES, start=1):
        print(f"\n=== Generation {gi} ===")
        X_syn, y_syn, A_syn = make_synthetic_round(X_tr, y_tr, A_tr, frac, device)
        X_mix, y_mix, A_mix = mix_pools(X_tr, y_tr, A_tr, prev_syn, (X_syn, y_syn, A_syn))

        # retrain encoder on mixed pool (shorter epochs)
        enc_g = train_encoder(X_mix, y_mix, X_val, y_val, EMBED_DIM, max(20, ENC_EPOCHS//2), ENC_LR, TAU, device)
        Z_mix = get_embeddings(enc_g, X_mix, device)
        Z_val_g = get_embeddings(enc_g, X_val, device)
        Z_te_g = get_embeddings(enc_g, X_te, device)

        models_g, w_g = fit_ensemble(Z_mix, y_mix, Z_val_g, y_val)
        P = ensemble_predict(models_g, w_g, Z_te_g)
        m = compute_metrics(y_te, P, A_te)
        drift = float(np.linalg.norm(mu_real - Z_te_g.mean(axis=0), ord=2))
        rho = ensemble_disagreement(models_g, Z_te_g)

        rows.append({"generation":gi, **m, "drift_z":drift, "rho":rho})
        prev_syn = (X_syn, y_syn, A_syn)

    return pd.DataFrame(rows)

def main():
    os.makedirs(OUT_DIR, exist_ok=True); ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    all_runs = []
    for s in SEED_LIST:
        df_s = run_one_seed(s, device)
        df_s["seed"] = s
        df_s.to_csv(os.path.join(OUT_DIR, "metrics", f"per_seed_results_seed{s}.csv"), index=False)
        all_runs.append(df_s)

    df_all = pd.concat(all_runs, ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR, "metrics", "all_seeds_generations.csv"), index=False)

    # aggregate mean/std
    summary = df_all.groupby("generation").agg(["mean","std"])
    summary.columns = [f"{a}_{b}" for a,b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT_DIR, "metrics", "summary_by_generation.csv"), index=False)

    # plot mean only
    plot_df = df_all.groupby("generation").mean(numeric_only=True).reset_index()
    plot_gen_curves(plot_df, os.path.join(OUT_DIR, "figures", "gen_curves.png"))

    # save config
    cfg = dict(input_dir=INPUT_DIR, input_file=INPUT_FILE, out_dir=OUT_DIR,
               label=LABEL_COL, protected=PROTECTED_COL, seeds=SEED_LIST,
               test_size=TEST_SIZE, val_size=VAL_SIZE, embed_dim=EMBED_DIM,
               enc_epochs=ENC_EPOCHS, enc_lr=ENC_LR, tau=TAU, vae_epochs=VAE_EPOCHS,
               vae_latent=VAE_LATENT, gen_sizes=GEN_SIZES, mix_coeffs=MIX_COEFFS,
               fair_target_parity=FAIR_TARGET_PARITY)
    with open(os.path.join(OUT_DIR, "artifacts", "run_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n DONE")
    print(f"- Metrics: {os.path.join(OUT_DIR, 'metrics')}")
    print(f"- Figures: {os.path.join(OUT_DIR, 'figures')}")
    print(f"- Config:  {os.path.join(OUT_DIR, 'artifacts', 'run_config.json')}")

if __name__ == "__main__":
    main()
