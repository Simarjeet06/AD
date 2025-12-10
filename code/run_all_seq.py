# E:\adni_python\code\run_all_seq.py
import os, sys, math, json, random
from pathlib import Path
import numpy as np
import pandas as pd

print("USING run_all_seq.py FROM:", __file__)

# -------------------- dependencies --------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError as e:
    print("Missing package:", e)
    print("Install with: pip install torch scikit-learn pandas numpy")
    sys.exit(1)

# -------------------- config --------------------
CSV_PATH = Path(r"E:\adni_python\outputs\master_with_imaging_match.csv")
OUT_DIR  = CSV_PATH.parent
BEST_MODEL_PATH = OUT_DIR / "best_seq_model.pt"
SEED = 42
EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------- utils --------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

VIS_ORDER = [
    "sc","bl","m03","m06","m12","m18","m24","m36","m48","m60","m72",
    "m84","m96","m108","m120","m132","m144","m156","m168","m180",
    "m192","m204","m216","m228"
]
APOE_MAP = {"2/2":0, "2/3":1, "3/2":1, "3/3":2, "3/4":3, "4/3":3, "4/4":4, "2/4":5, "4/2":5}
TARGET_NAMES = ["MMSE", "CDR_GLOBAL", "CDR_SOB", "ADAS_TOTSCORE"]

def norm_visit(v):
    if pd.isna(v): return None
    return str(v).strip().lower().replace(" ","")

def apoe_onehot(g):
    idx = APOE_MAP.get(str(g).strip(), None)
    vec = np.zeros(6, dtype=np.float32)
    if idx is not None:
        vec[idx] = 1.0
    return vec

def gender_bin(g):
    if pd.isna(g): return np.nan
    s = str(g).strip().upper()
    if s.startswith("M"): return 1.0
    if s.startswith("F"): return 0.0
    return np.nan

# -------------------- data building --------------------
def assemble_features(df):
    print("assemble_features: using fixed version")
    feats = []
    # Imaging flags (safe defaults if columns missing)
    feats.append(df.get("has_t1_match", pd.Series([0]*len(df))).fillna(0).astype(float).values[:,None])
    feats.append(df.get("has_pet_match", pd.Series([0]*len(df))).fillna(0).astype(float).values[:,None])
    # Clinical inputs (used alongside targets)
    for col in ["ADAS_TOTSCORE","ADAS_TOTAL13","CDR_GLOBAL","CDR_SOB","MMSE_SCORE","PTEDUCAT"]:
        x = df.get(col, pd.Series([np.nan]*len(df))).astype(float)
        feats.append(x.values[:,None])
    # Gender
    g = df.get("PTGENDER", pd.Series([np.nan]*len(df))).apply(gender_bin)
    feats.append(g.values[:,None])
    # APOE one-hot (6 dims)
    ap = np.stack([apoe_onehot(a) for a in df.get("APOE_GENOTYPE", pd.Series([""]*len(df)))], axis=0)
    feats.append(ap.astype(np.float32))
    X = np.concatenate(feats, axis=1).astype(np.float32)
    obs_mask = ~np.isnan(X)
    X = np.nan_to_num(X, nan=0.0)
    return X, obs_mask.astype(np.float32)

def assemble_targets(df):
    cols = ["MMSE_SCORE","CDR_GLOBAL","CDR_SOB","ADAS_TOTSCORE"]
    Y, M = [], []
    for c in cols:
        v = df.get(c, pd.Series([np.nan]*len(df))).astype(float)
        Y.append(v.values[:,None])
        M.append((~v.isna()).values[:,None])
    Y = np.concatenate(Y, axis=1).astype(np.float32)
    M = np.concatenate(M, axis=1).astype(np.float32)
    Y = np.nan_to_num(Y, nan=0.0)
    return Y, M

def build_sequences(csv_path, min_visits=1):
    df = pd.read_csv(csv_path, low_memory=False)
    df["visit"] = df["visit"].apply(norm_visit)
    df = df[df["visit"].isin(VIS_ORDER)]
    order_map = {v:i for i,v in enumerate(VIS_ORDER)}
    df["visit_idx"] = df["visit"].map(order_map)

    seqs = []
    for pid, g in df.groupby("subject_id"):
        g = g.sort_values("visit_idx")
        X, Xmask = assemble_features(g)
        Y, Ymask = assemble_targets(g)
        if len(g) >= min_visits:
            seqs.append({
                "pid": pid,
                "visits": g["visit"].tolist(),
                "X": X, "Xmask": Xmask,
                "Y": Y, "Ymask": Ymask
            })
    return seqs

def pad_batch(batch):
    T = [b["X"].shape[0] for b in batch]
    Tmax = max(T)
    Din = batch[0]["X"].shape[1]
    Dy  = batch[0]["Y"].shape[1]
    B   = len(batch)
    Xp = np.zeros((B, Tmax, Din), np.float32)
    Xmask = np.zeros((B, Tmax, Din), np.float32)
    Yp = np.zeros((B, Tmax, Dy), np.float32)
    Ymask = np.zeros((B, Tmax, Dy), np.float32)
    seq_mask = np.zeros((B, Tmax), np.float32)
    for i,b in enumerate(batch):
        t = b["X"].shape[0]
        Xp[i,:t,:] = b["X"]
        Xmask[i,:t,:] = b["Xmask"]
        Yp[i,:t,:] = b["Y"]
        Ymask[i,:t,:] = b["Ymask"]
        seq_mask[i,:t] = 1.0
    return Xp, Xmask, Yp, Ymask, seq_mask

# -------------------- model --------------------
class FusionDegradation(nn.Module):
    def __init__(self, d_in, d_latent, out_slices):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 128), nn.ReLU(),
            nn.Linear(128, d_latent)
        )
        self.decoders = nn.ModuleList([
            nn.Sequential(nn.Linear(d_latent, 64), nn.ReLU(), nn.Linear(64, dm))
            for dm in out_slices
        ])
    def forward(self, X):
        H = self.enc(X)
        recons = [dec(H) for dec in self.decoders]
        Xrec = torch.cat(recons, dim=-1)
        return H, Xrec
    @staticmethod
    def recon_loss(X, Xmask, Xrec):
        diff = (Xrec - X) * Xmask
        denom = Xmask.sum() + 1e-6
        return diff.pow(2).sum() / denom

class ModelFillingLSTM(nn.Module):
    def __init__(self, d_in, d_latent, d_targets, d_hidden=128, num_layers=1):
        super().__init__()
        self.fusion = FusionDegradation(d_in, d_latent, [d_in])
        self.lstm  = nn.LSTM(input_size=d_latent + d_targets, hidden_size=d_hidden, num_layers=num_layers, batch_first=True)
        self.pred_targets = nn.Linear(d_hidden, d_targets)
    def forward(self, X, Xmask, Y, Ymask, seq_mask):
        Henc, Xrec = self.fusion(X)
        lstm_in = torch.cat([Henc, Y], dim=-1)  # teacher forcing
        out, _ = self.lstm(lstm_in)
        Yhat = self.pred_targets(out)
        Lrec = self.fusion.recon_loss(X, Xmask, Xrec)
        Ltar = (torch.abs(Yhat - Y) * Ymask).sum() / (Ymask.sum() + 1e-6)
        return {"loss": Lrec + Ltar, "Lrec": Lrec, "Ltar": Ltar, "Yhat": Yhat}

# -------------------- training & eval --------------------
def collate(batch):
    X, Xmask, Y, Ymask, S = pad_batch(batch)
    to_t = lambda a: torch.from_numpy(a)
    return to_t(X), to_t(Xmask), to_t(Y), to_t(Ymask), to_t(S)

def train_and_eval():
    set_seed(SEED)
    print("CSV:", CSV_PATH)
    if not CSV_PATH.exists():
        print("CSV not found. Expected:", CSV_PATH)
        sys.exit(1)

    print("Loading sequences from:", CSV_PATH)
    seqs = build_sequences(CSV_PATH, min_visits=1)
    if len(seqs) == 0:
        print("No sequences built. Check CSV content.")
        return

    pids = [s["pid"] for s in seqs]
    train_ids, val_ids = train_test_split(pids, test_size=0.2, random_state=SEED)
    train = [s for s in seqs if s["pid"] in train_ids]
    val   = [s for s in seqs if s["pid"] in val_ids]

    Din = train[0]["X"].shape[1]
    Dy  = train[0]["Y"].shape[1]
    print(f"Dims -> Din={Din}, Dy={Dy}, train_n={len(train)}, val_n={len(val)}")

    model = ModelFillingLSTM(d_in=Din, d_latent=64, d_targets=Dy, d_hidden=128).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    dl_train = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    dl_val   = DataLoader(val,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    best_val = float("inf")
    for ep in range(1, EPOCHS+1):
        model.train(); tr_loss=tr_rec=tr_tar=0.0; ntr=0
        for X,Xm,Y,Ym,S in dl_train:
            X=X.to(DEVICE); Xm=Xm.to(DEVICE); Y=Y.to(DEVICE); Ym=Ym.to(DEVICE); S=S.to(DEVICE)
            out = model(X,Xm,Y,Ym,S); loss=out["loss"]
            opt.zero_grad(); loss.backward(); opt.step()
            bs=X.size(0); tr_loss+=loss.item()*bs; tr_rec+=out["Lrec"].item()*bs; tr_tar+=out["Ltar"].item()*bs; ntr+=bs
        tr_loss/=max(ntr,1); tr_rec/=max(ntr,1); tr_tar/=max(ntr,1)

        model.eval(); vl_loss=vl_rec=vl_tar=0.0; nvl=0
        with torch.no_grad():
            for X,Xm,Y,Ym,S in dl_val:
                X=X.to(DEVICE); Xm=Xm.to(DEVICE); Y=Y.to(DEVICE); Ym=Ym.to(DEVICE); S=S.to(DEVICE)
                out = model(X,Xm,Y,Ym,S)
                bs=X.size(0); vl_loss+=out["loss"].item()*bs; vl_rec+=out["Lrec"].item()*bs; vl_tar+=out["Ltar"].item()*bs; nvl+=bs
        vl_loss/=max(nvl,1); vl_rec/=max(nvl,1); vl_tar/=max(nvl,1)
        print(f"Epoch {ep:03d} | train {tr_loss:.4f} (rec {tr_rec:.4f}, tar {tr_tar:.4f}) | val {vl_loss:.4f} (rec {vl_rec:.4f}, tar {vl_tar:.4f})")

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), str(BEST_MODEL_PATH))
            print("  saved best:", BEST_MODEL_PATH)

    # Evaluation on validation (observed entries only)
    model.eval()
    y_true = [[] for _ in range(Dy)]
    y_pred = [[] for _ in range(Dy)]
    with torch.no_grad():
        for X,Xm,Y,Ym,S in dl_val:
            X=X.to(DEVICE); Xm=Xm.to(DEVICE); Y=Y.to(DEVICE); Ym=Ym.to(DEVICE)
            out = model(X,Xm,Y,Ym,S.to(DEVICE))
            Yhat = out["Yhat"].cpu().numpy()
            Ynp = Y.cpu().numpy()
            Mnp = Ym.cpu().numpy()
            B,T,_ = Ynp.shape
            for d in range(Dy):
                mask = Mnp[:,:,d] > 0.5
                y_true[d].extend(list(Ynp[:,:,d][mask]))
                y_pred[d].extend(list(Yhat[:,:,d][mask]))

    print("\nValidation metrics (current-visit prediction):")
    for d, name in enumerate(TARGET_NAMES[:Dy]):
        yt = np.array(y_true[d]); yp = np.array(y_pred[d])
        if yt.size == 0:
            print(f"- {name}: no observed targets")
            continue
        mae = mean_absolute_error(yt, yp)
        mse = mean_squared_error(yt, yp)
        rmse = math.sqrt(mse)
        try:
            r2 = r2_score(yt, yp)
        except Exception:
            r2 = float("nan")
        print(f"- {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f} ({r2*100:.1f}%)")

# -------------------- run --------------------
if __name__ == "__main__":
    if not CSV_PATH.exists():
        print("CSV not found. Expected:", CSV_PATH)
        sys.exit(1)
    train_and_eval()
