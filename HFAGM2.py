# -*- coding: utf-8 -*-
"""
HFAGM++ — Experiment 2 (Arabic Sign Language, YOLO-style image dataset)
- Parses data.yaml (train/val/test), reads images + YOLO labels
- Converts detection to classification by cropping the first bbox (or using full image)
- Extracts signer_id (protected attribute) from filename by regex (editable)
- Trains a supervised-contrastive encoder (small CNN + projection head)
- Trains ensemble (RF, KNN, MLP) on embeddings
- Runs 3 fairness-aware synthetic generations in embedding space via VAE
- Tracks Accuracy, macro AUC, macro F1, DPD (on correctness), Δz (drift), and ρ̄ (diversity)
- Saves results/plots under New_EXP2 (paths fixed as requested)
"""

import os, re, json, random, warnings
warnings.filterwarnings("ignore")

# ======= PATHS (fixed per user request) =======
ROOT = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\raw\ArSL"
YAML_PATH = os.path.join(ROOT, "data.yaml")
OUT_DIR = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\experiments\New_EXP2"

# ======= CONFIG =======
SEEDS = [42, 47, 53]
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS_ENC = 30
LR_ENC = 1e-3
TAU = 0.07              # temperature for SupCon
EMBED_DIM = 64
VAE_EPOCHS = 50
VAE_Z = 16
GEN_SIZES = [0.3, 0.3, 0.3]    # fraction of real-train embeddings per generation
MIX_COEFFS = (0.6, 0.2, 0.2)   # (alpha real, beta prev-gen, gamma cur-gen)
FAIR_BALANCE_SIGNERS = True    # balance synthetic counts across signer groups

# Signer extraction: edit/add patterns as needed to fit your filenames.
SIGNER_REGEXES = [
    r"signer[_-]?(\d+)", r"[^\w]S(\d+)[^\w]", r"Signer[_-]?(\d+)", r"_s(\d+)_", r"-s(\d+)-"
]

# ======= IMPORTS =======
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import combinations

from fairlearn.metrics import demographic_parity_difference

# ================== UTIL ==================
def ensure_dirs():
    for sub in ["metrics", "figures", "artifacts"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def signer_from_name(name: str) -> int:
    """Extract signer id via regex; return 0 if not found."""
    for rx in SIGNER_REGEXES:
        m = re.search(rx, f"_{name}_", re.IGNORECASE)  # pad to catch boundaries
        if m:
            try: return int(m.group(1))
            except: continue
    return 0

def yolo_read_label(txt_path: str):
    """
    YOLO txt: lines of 'cls x y w h' normalized. We use the FIRST bbox as classification target.
    Returns (cls_id, bbox_tuple or None)
    """
    if not os.path.exists(txt_path):
        return None, None
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if len(lines) == 0:
        return None, None
    parts = lines[0].split()
    if len(parts) < 5:
        return None, None
    cls = int(float(parts[0]))
    x, y, w, h = map(float, parts[1:5])
    return cls, (x, y, w, h)

def crop_from_yolo(img: Image.Image, bbox):
    """bbox in normalized coords (x_center,y_center,w,h), returns cropped PIL.Image."""
    if bbox is None:
        return img
    W, H = img.size
    xc, yc, w, h = bbox
    x1 = max(0, int((xc - w/2) * W))
    y1 = max(0, int((yc - h/2) * H))
    x2 = min(W, int((xc + w/2) * W))
    y2 = min(H, int((yc + h/2) * H))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))

# ================== DATASET ==================
class ArSLDataset(Dataset):
    def __init__(self, items, transform=None, k_views=2, train=True):
        """
        items: list of dicts with keys: image_path, label(int), signer(int)
        transform: torchvision transforms
        k_views: number of contrastive views for SupCon when train=True
        """
        self.items = items
        self.transform = transform
        self.k = k_views
        self.train = train

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["image_path"]).convert("RGB")
        # Read YOLO label and crop (if missing, use full image)
        lbl_path = it["label_path"]
        cls_id, bbox = yolo_read_label(lbl_path)
        if cls_id is None:  # fall back to provided label
            cls_id = it["label"]
        img = crop_from_yolo(img, bbox)

        if self.train:
            views = []
            for _ in range(self.k):
                views.append(self.transform(img) if self.transform else T.ToTensor()(img))
            x = torch.stack(views, dim=0)  # [K, C, H, W]
        else:
            x = (self.transform(img) if self.transform else T.ToTensor()(img)).unsqueeze(0)  # [1, C, H, W]

        return x, cls_id, it["signer"]

def build_manifest(root, split_key, names):
    """
    root/train|val|test/images/*.jpg and labels/*.txt
    'names' from data.yaml maps class ids to names; we only need num_classes.
    """
    split_dir = os.path.join(root, split_key)
    img_dir = os.path.join(split_dir, "images")
    lbl_dir = os.path.join(split_dir, "labels")
    images = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")) + glob(os.path.join(img_dir, "*.jpeg")))
    items = []
    for ip in images:
        base = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, base + ".txt")
        cls_id, _ = yolo_read_label(lp)
        if cls_id is None:
            # If no label file or malformed, skip this sample
            continue
        signer = signer_from_name(base)
        items.append({"image_path": ip, "label_path": lp, "label": int(cls_id), "signer": int(signer)})
    return items, len(names)

# ================== MODEL (Encoder + Head) ==================
class SmallCNN(nn.Module):
    def __init__(self, embed_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128,3, stride=2, padding=1), nn.BatchNorm2d(128),nn.ReLU(),
            nn.Conv2d(128,256,3, stride=2, padding=1), nn.BatchNorm2d(256),nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.proj = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, x):
        h = self.features(x)           # [B,256,1,1]
        h = h.view(h.size(0), -1)      # [B,256]
        z = self.proj(h)               # [B,embed_dim]
        z = F.normalize(z, p=2, dim=-1)
        return z

def supcon_loss(z_flat, y, K=2, tau=0.07):
    """
    z_flat: [B*K, d]   y: [B]
    """
    device = z_flat.device
    B = y.shape[0]
    y = y.view(B,1)
    mask = torch.eq(y, y.T).float().to(device)  # [B,B]
    mask = mask.repeat_interleave(K,0).repeat_interleave(K,1)  # [B*K, B*K]

    sim = torch.matmul(z_flat, z_flat.T) / tau
    logits_mask = torch.ones_like(sim) - torch.eye(B*K, device=device)
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

    positive_mask = mask * logits_mask
    pos_count = positive_mask.sum(dim=1)
    mean_log_pos = (positive_mask * log_prob).sum(dim=1) / torch.clamp(pos_count, min=1.0)
    return -mean_log_pos.mean()

# ================== TRAIN / EMBED ==================
def train_encoder(train_items, val_items, num_classes, device):
    aug_train = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.9,1.1)),
        T.ToTensor()
    ])
    aug_eval = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])

    ds_tr = ArSLDataset(train_items, transform=aug_train, k_views=2, train=True)
    ds_vl = ArSLDataset(val_items,   transform=aug_eval,  k_views=1, train=False)
    bs = min(BATCH_SIZE, max(2, len(ds_tr)))
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, drop_last=False, num_workers=0)
    dl_vl = DataLoader(ds_vl, batch_size=bs, shuffle=False, drop_last=False, num_workers=0)

    model = SmallCNN(embed_dim=EMBED_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR_ENC)

    for ep in range(EPOCHS_ENC):
        model.train(); tot, nb = 0.0, 0
        for xk, y, _ in dl_tr:
            # xk: [B, K, C, H, W] → flatten
            Bk, K, C, H, W = xk.shape
            x = xk.view(Bk*K, C, H, W).to(device)
            y = y.to(device)
            z = model(x)
            loss = supcon_loss(z, y, K=K, tau=TAU)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if (ep+1) % 10 == 0 and nb > 0:
            print(f"[Encoder] {ep+1}/{EPOCHS_ENC}: {tot/nb:.4f}")
    # Return model and eval transforms for embedding
    return model, aug_eval

@torch.no_grad()
def embed_items(model, items, transform, device):
    model.eval()
    feats, labels, signers = [], [], []
    bs = min(BATCH_SIZE, max(2, len(items)))
    for i in range(0, len(items), bs):
        batch = items[i:i+bs]
        imgs = []
        yb, sb = [], []
        for it in batch:
            img = Image.open(it["image_path"]).convert("RGB")
            cls, bbox = yolo_read_label(it["label_path"])
            if cls is None: cls = it["label"]
            img = crop_from_yolo(img, bbox)
            xt = transform(img)
            imgs.append(xt); yb.append(cls); sb.append(it["signer"])
        X = torch.stack(imgs, dim=0).to(device)  # [B,3,H,W]
        z = model(X).cpu().numpy()
        feats.append(z); labels.extend(yb); signers.extend(sb)
    Z = np.vstack(feats)
    y = np.array(labels, dtype=int)
    A = np.array(signers, dtype=int)
    return Z, y, A

# ================== ENSEMBLE ==================
def fit_ensemble(Ztr, ytr, Zvl, yvl):
    rf  = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
    knn = KNeighborsClassifier(n_neighbors=5)
    mlp = MLPClassifier(hidden_layer_sizes=(256,128), max_iter=300, random_state=42)
    for clf in (rf,knn,mlp):
        clf.fit(Ztr, ytr)
    accs = [accuracy_score(yvl, c.predict(Zvl)) for c in (rf,knn,mlp)]
    w = np.exp(10*np.array(accs)); w = w / w.sum()
    return (rf,knn,mlp), w

def ensemble_predict(models, w, Z):
    P = None
    for m, wi in zip(models, w):
        p = m.predict_proba(Z)
        P = p*wi if P is None else P + p*wi
    return P / np.clip(w.sum(), 1e-9, None)

def ensemble_disagreement(models, Z):
    preds = np.array([m.predict(Z) for m in models])
    return float(np.mean([np.mean(preds[i]!=preds[j]) for i,j in combinations(range(preds.shape[0]),2)]))

# ================== METRICS ==================
def metrics_multiclass(y_true, P, A):
    """Utility: Acc, macro-F1, macro-AUC; Fairness: DPD over 'prediction correctness' (binary)."""
    y_pred = np.argmax(P, axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # macro-AUC (ovr)
    classes = np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)
    auc = np.nan
    try:
        auc = roc_auc_score(y_bin, P[:, classes], average="macro", multi_class="ovr")
    except Exception:
        try:
            auc = roc_auc_score(y_bin, P, average="macro", multi_class="ovr")
        except Exception:
            pass

    # Fairness proxy: correctness as binary outcome per group (works with Fairlearn DPD)
    correct = (y_pred == y_true).astype(int)
    try:
        dpd = demographic_parity_difference(correct, correct, sensitive_features=A)
    except Exception:
        dpd = np.nan

    return {"accuracy": acc, "macro_f1": f1m, "macro_auc": auc, "dpd_correctness": dpd}

# ================== VAE over Embeddings ==================
class VAE(nn.Module):
    def __init__(self, in_dim, z_dim=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU())
        self.mu, self.logv = nn.Linear(64,z_dim), nn.Linear(64,z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim,64), nn.ReLU(), nn.Linear(64,128), nn.ReLU(), nn.Linear(128,in_dim))
    def forward(self, x):
        h = self.enc(x); mu, logv = self.mu(h), self.logv(h)
        std = torch.exp(0.5*logv); z = mu + std*torch.randn_like(std)
        xr = self.dec(z); return xr, mu, logv

def vae_loss(x, xr, mu, logv):
    recon = F.mse_loss(xr, x)
    kld   = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
    return recon + 1e-3 * kld

def train_vae_embed(Z, epochs=50, zdim=16, device="cpu"):
    vae = VAE(Z.shape[1], z_dim=zdim).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    Zt = torch.tensor(Z, dtype=torch.float32).to(device)
    for ep in range(epochs):
        xr, mu, logv = vae(Zt)
        loss = vae_loss(Zt, xr, mu, logv)
        opt.zero_grad(); loss.backward(); opt.step()
        if (ep+1) % 10 == 0:
            print(f"[VAE-emb] {ep+1}/{epochs}: {loss.item():.4f}")
    return vae

@torch.no_grad()
def sample_vae_embed(vae, n, device="cpu"):
    z = torch.randn(n, vae.mu.out_features, device=device)
    Xs = vae.dec(z).cpu().numpy()
    return Xs

def synthesize_embeddings(Z_tr, y_tr, A_tr, size_frac, device):
    n_syn = int(size_frac * len(Z_tr))
    if n_syn < 10:
        return np.empty((0, Z_tr.shape[1])), np.empty(0,int), np.empty(0,int)
    vae = train_vae_embed(Z_tr, epochs=VAE_EPOCHS, zdim=VAE_Z, device=device)
    Z_syn = sample_vae_embed(vae, n_syn, device=device)

    # Assign synthetic labels by nearest neighbor in class centroids (simple & stable)
    classes = np.unique(y_tr)
    centroids = {c: Z_tr[y_tr==c].mean(axis=0) for c in classes}
    # Map each synthetic vector to nearest class centroid
    y_syn = []
    for z in Z_syn:
        dists = [(c, np.linalg.norm(z - centroids[c])) for c in classes]
        y_syn.append(min(dists, key=lambda t: t[1])[0])
    y_syn = np.array(y_syn, dtype=int)

    # Assign signer groups (protected) to balance across groups
    if FAIR_BALANCE_SIGNERS:
        groups, counts = np.unique(A_tr, return_counts=True)
        probs = np.ones_like(counts, dtype=float) / len(counts)  # uniform target across groups
        A_syn = np.random.choice(groups, size=n_syn, p=probs/probs.sum())
    else:
        groups, counts = np.unique(A_tr, return_counts=True)
        A_syn = np.random.choice(groups, size=n_syn, p=(counts/counts.sum()))
    return Z_syn, y_syn, A_syn

def mix_pools(Z_real, y_real, A_real, prev_syn, cur_syn):
    alpha, beta, gamma = MIX_COEFFS
    Zp, yp, Ap = prev_syn
    Zc, yc, Ac = cur_syn
    parts = []
    n_real = max(1, int(alpha * len(Z_real)))
    ridx = np.random.choice(np.arange(len(Z_real)), size=min(n_real, len(Z_real)), replace=False)
    parts.append((Z_real[ridx], y_real[ridx], A_real[ridx]))
    if len(Zp) > 0 and beta > 0:
        n_prev = int(beta * len(Z_real))
        pidx = np.random.choice(np.arange(len(Zp)), size=min(n_prev, len(Zp)), replace=False)
        parts.append((Zp[pidx], yp[pidx], Ap[pidx]))
    if len(Zc) > 0 and gamma > 0:
        n_cur = int(gamma * len(Z_real))
        cidx = np.random.choice(np.arange(len(Zc)), size=min(n_cur, len(Zc)), replace=False)
        parts.append((Zc[cidx], yc[cidx], Ac[cidx]))
    Zm = np.vstack([p[0] for p in parts]); ym = np.concatenate([p[1] for p in parts]); Am = np.concatenate([p[2] for p in parts])
    return Zm, ym, Am

# ================== MAIN EXPERIMENT ==================
def run_one_seed(seed, device):
    set_seed(seed)

    # YAML layout: expect relative subpaths for train/val/test under root
    dct = load_yaml(YAML_PATH)
    # We rely on folder structure root/train, root/valid, root/test
    names = dct.get("names", [])
    if not isinstance(names, list) or len(names)==0:
        # fallback to nc
        nc = int(dct.get("nc", 0)) if "nc" in dct else 0
        names = [str(i) for i in range(nc)]
    print(f"Classes detected: {len(names)}")

    train_items, num_classes = build_manifest(ROOT, "train", names)
    val_items, _ = build_manifest(ROOT, "valid", names)
    test_items, _ = build_manifest(ROOT, "test", names)

    if len(train_items) < 10 or len(val_items) < 5 or len(test_items) < 5:
        raise ValueError("Dataset splits too small. Please check ArSL paths / labels.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Train encoder (one time). (For simplicity, recursive gens operate in embedding space.)
    enc, eval_tf = train_encoder(train_items, val_items, num_classes, device)
    Z_tr, y_tr, A_tr = embed_items(enc, train_items, eval_tf, device)
    Z_vl, y_vl, A_vl = embed_items(enc, val_items,  eval_tf, device)
    Z_te, y_te, A_te = embed_items(enc, test_items,  eval_tf, device)

     # ==========================================================
    # SAVE PRETRAINED ENCODER (for cross-dataset transfer in Exp 3)
    # ==========================================================
    encoder_save_path = os.path.join(OUT_DIR, "encoder_pretrained.pth")
    try:
        torch.save(enc.state_dict(), encoder_save_path)
        print(f"\nEncoder weights saved successfully to:\n{encoder_save_path}")
    except Exception as e:
        print(f"\n Warning: Encoder could not be saved due to error: {e}")

    models, w = fit_ensemble(Z_tr, y_tr, Z_vl, y_vl)
    P0 = ensemble_predict(models, w, Z_te)
    m0 = metrics_multiclass(y_te, P0, A_te)
    rho0 = ensemble_disagreement(models, Z_te)
    mu0 = Z_te.mean(axis=0)

    rows = [{"generation":0, **m0, "drift_z":0.0, "rho":rho0}]

    prev = (np.empty((0, Z_tr.shape[1])), np.empty(0,int), np.empty(0,int))
    Z_base, y_base, A_base = Z_tr, y_tr, A_tr

    for gi, frac in enumerate(GEN_SIZES, start=1):
        print(f"\n=== Generation {gi} ===")
        Z_syn, y_syn, A_syn = synthesize_embeddings(Z_base, y_base, A_base, frac, device)
        Z_mix, y_mix, A_mix = mix_pools(Z_base, y_base, A_base, prev, (Z_syn, y_syn, A_syn))
        # retrain only the ensemble on mixed embeddings (encoder fixed)
        models_g, w_g = fit_ensemble(Z_mix, y_mix, Z_vl, y_vl)
        Pg = ensemble_predict(models_g, w_g, Z_te)
        mg = metrics_multiclass(y_te, Pg, A_te)
        drift = float(np.linalg.norm(mu0 - Z_te.mean(axis=0), ord=2))  # test centroid unchanged (fixed encoder)
        rho = ensemble_disagreement(models_g, Z_te)
        rows.append({"generation":gi, **mg, "drift_z":drift, "rho":rho})
        prev = (Z_syn, y_syn, A_syn)
        # Update base pool for next round (optional): keep original real base to anchor fairness; we keep Z_base=y_base as original.

    return pd.DataFrame(rows)

def plot_curves(df, out_file):
    plt.figure(figsize=(8,5))
    for m in ["accuracy", "macro_auc", "macro_f1", "dpd_correctness", "drift_z"]:
        plt.plot(df["generation"], df[m], marker='o', label=m)
    plt.xlabel("Generation (0=Real-only)")
    plt.title("HFAGM++ ArSL: Accuracy/AUC/F1/DPD/Δz across Generations")
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(out_file, dpi=200); plt.close()

def main():
    os.makedirs(OUT_DIR, exist_ok=True); ensure_dirs()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    all_ = []
    for s in SEEDS:
        df = run_one_seed(s, device)
        df["seed"] = s
        df.to_csv(os.path.join(OUT_DIR, "metrics", f"per_seed_ArSL_seed{s}.csv"), index=False)
        all_.append(df)

    df_all = pd.concat(all_, ignore_index=True)
    df_all.to_csv(os.path.join(OUT_DIR, "metrics", "all_seeds_generations_ArSL.csv"), index=False)

    mean = df_all.groupby("generation").mean(numeric_only=True).reset_index()
    mean.to_csv(os.path.join(OUT_DIR, "metrics", "summary_by_generation_ArSL.csv"), index=False)

    plot_curves(mean, os.path.join(OUT_DIR, "figures", "gen_curves_ArSL.png"))

    # Save run config
    cfg = dict(root=ROOT, yaml=YAML_PATH, out_dir=OUT_DIR, seeds=SEEDS,
               img_size=IMG_SIZE, batch_size=BATCH_SIZE, epochs_enc=EPOCHS_ENC,
               lr_enc=LR_ENC, tau=TAU, embed_dim=EMBED_DIM,
               vae_epochs=VAE_EPOCHS, vae_z=VAE_Z, gen_sizes=GEN_SIZES,
               mix_coeffs=MIX_COEFFS, fair_balance_signers=FAIR_BALANCE_SIGNERS,
               signer_regexes=SIGNER_REGEXES)
    with open(os.path.join(OUT_DIR, "artifacts", "run_config_ArSL.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("\nDONE. Results saved to:", OUT_DIR)
    
if __name__ == "__main__":
    main()
