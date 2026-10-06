# -*- coding: utf-8 -*-
"""
HFAGM++ Experiment 2 — Arabic Sign Language (ArSL)
Full self-contained implementation with encoder saving for cross-dataset transfer (Exp 3).
Author: Dr. Mahmoud B. M. Rokaya
"""

# =============================================================
# 1. Imports & Configuration
# =============================================================
import os, re, json, random, warnings
warnings.filterwarnings("ignore")

import yaml
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from fairlearn.metrics import demographic_parity_difference

# =============================================================
# 2. Paths & Hyper-parameters
# =============================================================
DATA_ROOT = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\raw\ArSL"
YAML_PATH = os.path.join(DATA_ROOT, "data.yaml")
OUT_DIR   = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\experiments\New_EXP2"

SEEDS         = [42, 47, 53]
IMG_SIZE      = 224
BATCH_SIZE    = 64
EPOCHS_ENC    = 30
LR_ENC        = 1e-3
TAU           = 0.07
EMBED_DIM     = 64
VAE_EPOCHS    = 50
Z_DIM         = 16
FAIR_COEFF    = 0.3          # weighting for fairness loss
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================
# 3. Utilities
# =============================================================
def ensure_dirs():
    for sub in ["metrics", "figures", "artifacts"]:
        os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def signer_from_name(name):
    m = re.search(r"signer[_-]?(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else 0

def read_yolo(txt_path):
    if not os.path.exists(txt_path): return None, None
    with open(txt_path, "r") as f:
        line = f.readline().strip().split()
    if len(line) < 5: return None, None
    cls = int(line[0]); x, y, w, h = map(float, line[1:5])
    return cls, (x, y, w, h)

def crop(img, bbox):
    if bbox is None: return img
    W, H = img.size; x, y, w, h = bbox
    x1, y1 = int((x-w/2)*W), int((y-h/2)*H)
    x2, y2 = int((x+w/2)*W), int((y+h/2)*H)
    x1, y1 = max(0,x1), max(0,y1); x2, y2 = min(W,x2), min(H,y2)
    return img.crop((x1,y1,x2,y2))

# =============================================================
# 4. Dataset
# =============================================================
class ArSLDataset(Dataset):
    def __init__(self, items, transform, k=2, train=True):
        self.items, self.transform, self.k, self.train = items, transform, k, train
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["image"]).convert("RGB")
        cls, bbox = read_yolo(it["label"])
        if cls is None: cls = it["cls"]
        img = crop(img, bbox)
        if self.train:
            x = torch.stack([self.transform(img) for _ in range(self.k)])
        else:
            x = self.transform(img).unsqueeze(0)
        return x, cls, it["signer"]

def make_manifest(split, names):
    img_dir = os.path.join(DATA_ROOT, split, "images")
    lbl_dir = os.path.join(DATA_ROOT, split, "labels")
    images = sorted(glob(os.path.join(img_dir, "*.jpg")) + glob(os.path.join(img_dir, "*.png")))
    items = []
    for ip in images:
        base = os.path.splitext(os.path.basename(ip))[0]
        lbl = os.path.join(lbl_dir, base + ".txt")
        cls, _ = read_yolo(lbl)
        if cls is None: continue
        items.append({"image": ip, "label": lbl, "cls": int(cls), "signer": signer_from_name(base)})
    return items, len(names)

# =============================================================
# 5. Encoder & Contrastive Loss
# =============================================================
class SmallCNN(nn.Module):
    def __init__(self, d=EMBED_DIM):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3,32,3,2,1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,256,3,2,1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.g = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Linear(128,d))
    def forward(self,x):
        h = self.f(x).view(x.size(0),-1)
        z = self.g(h)
        return F.normalize(z,p=2,dim=-1)

def supcon_loss(z,y,K=2,tau=TAU):
    B = y.size(0)
    mask = (y.unsqueeze(1)==y.unsqueeze(0)).float().repeat_interleave(K,0).repeat_interleave(K,1)
    sim = torch.matmul(z,z.T)/tau
    logits_mask = torch.ones_like(sim)-torch.eye(B*K,device=z.device)
    exp_sim = torch.exp(sim)*logits_mask
    log_prob = sim - torch.log(exp_sim.sum(1,keepdim=True)+1e-12)
    pos = mask*logits_mask
    denom = pos.sum(1).clamp(min=1.)
    mean = (pos*log_prob).sum(1)/denom
    return -mean.mean()

def train_encoder(train_items,val_items,n_classes,device):
    aug_t = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),T.RandomHorizontalFlip(),
                       T.ColorJitter(0.2,0.2,0.2),T.RandomAffine(10,translate=(0.05,0.05)),T.ToTensor()])
    aug_v = T.Compose([T.Resize((IMG_SIZE,IMG_SIZE)),T.ToTensor()])
    ds = ArSLDataset(train_items,aug_t,2,True)
    dl = DataLoader(ds,batch_size=min(BATCH_SIZE,len(ds)),shuffle=True)
    enc = SmallCNN().to(device)
    opt = torch.optim.Adam(enc.parameters(),lr=LR_ENC)
    for ep in range(EPOCHS_ENC):
        enc.train(); tot=0
        for x,y,_ in dl:
            Bk,K,C,H,W = x.shape
            x = x.view(Bk*K,C,H,W).to(device); y=y.to(device)
            z = enc(x)
            loss = supcon_loss(z,y,K)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if (ep+1)%10==0:
            print(f"[Encoder] {ep+1}/{EPOCHS_ENC}: {tot/len(dl):.4f}")
    return enc, aug_v

@torch.no_grad()
def embed(enc, items, tf, device):
    enc.eval(); feats=[]; labels=[]; signers=[]
    bs = min(BATCH_SIZE, max(2,len(items)))
    for i in range(0,len(items),bs):
        b = items[i:i+bs]
        ims,ys,ss=[],[],[]
        for it in b:
            img = Image.open(it["image"]).convert("RGB")
            cls,bbox = read_yolo(it["label"])
            if cls is None: cls=it["cls"]
            img=crop(img,bbox)
            ims.append(tf(img)); ys.append(cls); ss.append(it["signer"])
        X = torch.stack(ims).to(device)
        z = enc(X).cpu().numpy()
        feats.append(z); labels.extend(ys); signers.extend(ss)
    return np.vstack(feats),np.array(labels),np.array(signers)

# =============================================================
# 6. Fairness-Aware VAE Generator
# =============================================================
class VAE(nn.Module):
    def __init__(self, dim_in=EMBED_DIM, z_dim=Z_DIM):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim_in,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU())
        self.mu = nn.Linear(64,z_dim); self.logvar = nn.Linear(64,z_dim)
        self.dec = nn.Sequential(nn.Linear(z_dim,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,dim_in))
    def forward(self,x):
        h = self.enc(x); mu=self.mu(h); logv=self.logvar(h)
        std=torch.exp(0.5*logv); eps=torch.randn_like(std)
        z = mu+eps*std; out=self.dec(z)
        return out,mu,logv
    def kl(self,mu,logv): return -0.5*torch.sum(1+logv-mu.pow(2)-logv.exp())

def synthesize(Z,y,A,vae,device):
    vae.train(); opt=torch.optim.Adam(vae.parameters(),lr=1e-3)
    Zt=torch.tensor(Z,dtype=torch.float32).to(device)
    for ep in range(VAE_EPOCHS):
        opt.zero_grad(); out,mu,lv=vae(Zt)
        rec = F.mse_loss(out,Zt); kl=vae.kl(mu,lv)/len(Z)
        loss=rec+0.001*kl
        loss.backward(); opt.step()
        if (ep+1)%10==0: print(f"[VAE-emb] {ep+1}/{VAE_EPOCHS}: {loss.item():.4f}")
    vae.eval()
    with torch.no_grad():
        Zs,_mu,_lv=vae(Zt)
        Zs=Zs.cpu().numpy()
    idx=np.random.choice(len(Zs),int(0.3*len(Zs)),replace=False)
    return Zs[idx],y[idx],A[idx]

# =============================================================
# 7. Ensemble & Metrics
# =============================================================
def ensemble_predict(Ztr,ytr,Zte):
    clf1=MLPClassifier(hidden_layer_sizes=(128,),max_iter=300)
    clf2=RandomForestClassifier(n_estimators=100)
    clf3=KNeighborsClassifier(n_neighbors=5)
    for c in [clf1,clf2,clf3]: c.fit(Ztr,ytr)
    P=np.mean([c.predict_proba(Zte) for c in [clf1,clf2,clf3]],axis=0)
    return np.argmax(P,1),P

def evaluate(y_true,y_pred,P,A_true,Zr,Zg):
    acc=accuracy_score(y_true,y_pred)
    f1=f1_score(y_true,y_pred,average='macro')
    auc=roc_auc_score(label_binarize(y_true,classes=np.unique(y_true)),
                      P,average='macro')
    dpd=demographic_parity_difference(y_true,y_pred,sensitive_features=A_true)
    drift=np.linalg.norm(Zr.mean(0)-Zg.mean(0))
    return dict(accuracy=acc,macro_f1=f1,macro_auc=auc,dpd_correctness=dpd,drift_z=drift)

# =============================================================
# 8. Experiment Core
# =============================================================
def run_one_seed(seed,device):
    set_seed(seed)
    cfg=load_yaml(YAML_PATH)
    names=cfg.get("names",[str(i) for i in range(int(cfg.get("nc",0)))])
    print(f"Classes detected: {len(names)}")

    tr,_=make_manifest("train",names)
    vl,_=make_manifest("valid",names)
    te,_=make_manifest("test",names)

    enc,tfv=train_encoder(tr,vl,len(names),device)

    # ---------- Encoder saving fix ----------
    enc_path=os.path.join(OUT_DIR,"encoder_pretrained.pth")
    try:
        torch.save(enc.state_dict(),enc_path)
        print(f"\nEncoder weights saved to: {enc_path}")
    except Exception as e:
        print(f"Warning: encoder save failed: {e}")

    # embeddings
    Zt,yt,At=embed(enc,tr,tfv,device)
    Zv,yv,Av=embed(enc,vl,tfv,device)
    Ze,ye,Ae=embed(enc,te,tfv,device)

    vae=VAE().to(device)
    results=[]
    for gen in range(4):
        if gen==0:
            Zg,yA,Ag=Zt,yt,At
        else:
            Zs,ys,As=synthesize(Zg,yA,Ag,vae,device)
            Zg=np.vstack([Zg,Zs]); yA=np.concatenate([yA,ys]); Ag=np.concatenate([Ag,As])
        y_pred,P=ensemble_predict(Zg,yA,Ze)
        m=evaluate(ye,y_pred,P,Ae,Zt,Zg)
        m["generation"]=gen; results.append(m)
    return pd.DataFrame(results)

# =============================================================
# 9. Main Driver
# =============================================================
def main():
    os.makedirs(OUT_DIR,exist_ok=True); ensure_dirs()
    print("Device:",DEVICE)
    dfs=[]
    for s in SEEDS:
        df=run_one_seed(s,DEVICE)
        df["seed"]=s
        df.to_csv(os.path.join(OUT_DIR,"metrics",f"per_seed_results_seed{s}.csv"),index=False)
        dfs.append(df)
    all=pd.concat(dfs,ignore_index=True)
    all.to_csv(os.path.join(OUT_DIR,"metrics","all_seeds_generations.csv"),index=False)
    mean=all.groupby("generation").mean(numeric_only=True).reset_index()
    mean.to_csv(os.path.join(OUT_DIR,"metrics","summary_by_generation.csv"),index=False)

    plt.figure(figsize=(8,5))
    for m in ["accuracy","macro_auc","macro_f1","dpd_correctness","drift_z"]:
        if m in mean.columns: plt.plot(mean["generation"],mean[m],marker='o',label=m)
    plt.legend(); plt.title("HFAGM++ ArSL Generations"); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"figures","gen_curves_ArSL.png"),dpi=250)

    cfg={"data_root":DATA_ROOT,"yaml":YAML_PATH,"out_dir":OUT_DIR,"seeds":SEEDS}
    with open(os.path.join(OUT_DIR,"artifacts","run_config.json"),"w") as f: json.dump(cfg,f,indent=2)
    print("\nDONE. Results saved to:",OUT_DIR)

# =============================================================
if __name__=="__main__":
    main()
