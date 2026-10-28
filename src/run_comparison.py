"""
Few-Shot vs Full Learning Comparison Experiment.

Compares two approaches under 1-shot and 3-shot settings:
  1. Full Learning:  Train DualStream model from scratch with limited data
  2. Few-Shot (Proto): Use pre-trained DualStream features + Prototypical Network

Datasets:
  - CAPPIMU: sweeping (label 7) + mopping (label 8)  → 2 classes
  - Dunqi:   蹲起 (squat) from for_prototypical dir   → 1 class
  Total: 3 classes

Since CAPPIMU has many samples per class and dunqi has fewer,
we use ALL dunqi samples (squat) as query set,
and ALL dunqi samples for support (shot=1 means 1 per class in CAPPIMU style).

Actually, the dunqi data has 3 sessions of squat data.
We split: session 0 → support, session 1-2 → query.
For CAPPIMU: take first N for support, rest for query.

Usage:
  python run_comparison.py
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# ── Local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from src.models.dual_stream import DualStreamHAR
from src.models.prototypical import PrototypicalClassifier
from src.models.gaf import GAFEncoder


# ════════════════════════════════════════════════════════════════════════════
#  Config
# ════════════════════════════════════════════════════════════════════════════
CAPPIMU_LABELS = {7: "sweeping", 8: "mopping"}
DUNQI_LABEL = 99  # sentinel label for squat (will be remapped)

SEQ_LEN = 128
NUM_CHANNELS = 9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# ════════════════════════════════════════════════════════════════════════════
#  Data Loading
# ════════════════════════════════════════════════════════════════════════════

def load_phyphox_csv(acc_path: str, gyro_path: str):
    acc = np.loadtxt(acc_path, delimiter=',', skiprows=1)
    gyro = np.loadtxt(gyro_path, delimiter=',', skiprows=1)
    return acc, gyro


def estimate_gravity(acc_data: np.ndarray, n: int = 100) -> np.ndarray:
    return acc_data[:n, 1:].mean(axis=0)


def align_signals(acc: np.ndarray, gyro: np.ndarray):
    t_start = max(acc[0, 0], gyro[0, 0])
    t_end = min(acc[-1, 0], gyro[-1, 0])
    t_common = np.linspace(t_start, t_end, min(len(acc), len(gyro)))

    ax = interp1d(acc[:, 0], acc[:, 1], kind='linear', fill_value='extrapolate')
    ay = interp1d(acc[:, 0], acc[:, 2], kind='linear', fill_value='extrapolate')
    az = interp1d(acc[:, 0], acc[:, 3], kind='linear', fill_value='extrapolate')
    gx = interp1d(gyro[:, 0], gyro[:, 1], kind='linear', fill_value='extrapolate')
    gy = interp1d(gyro[:, 0], gyro[:, 2], kind='linear', fill_value='extrapolate')
    gz = interp1d(gyro[:, 0], gyro[:, 3], kind='linear', fill_value='extrapolate')

    acc_aligned = np.stack([t_common, ax(t_common), ay(t_common), az(t_common)], axis=1)
    gyro_aligned = np.stack([t_common, gx(t_common), gy(t_common), gz(t_common)], axis=1)
    return acc_aligned, gyro_aligned


def process_session(acc: np.ndarray, gyro: np.ndarray,
                   stride: int = 64, seq_len: int = SEQ_LEN) -> np.ndarray:
    acc_a, gyro_a = align_signals(acc, gyro)
    gravity = estimate_gravity(acc_a)
    body_acc = acc_a[:, 1:] - gravity

    signal = np.stack([
        body_acc[:, 0], body_acc[:, 1], body_acc[:, 2],
        gyro_a[:, 1], gyro_a[:, 2], gyro_a[:, 3],
        acc_a[:, 1], acc_a[:, 2], acc_a[:, 3],
    ], axis=1)  # (N, 9)

    windows = []
    for start in range(0, len(signal) - seq_len + 1, stride):
        win = signal[start:start + seq_len]
        windows.append(win.T)  # (9, seq_len)
    return np.array(windows)


def load_dunqi(data_root: str):
    """Load all dunqi sessions. Returns (windows, labels, session_ids)."""
    data_root = Path(data_root)
    sessions = sorted(data_root.glob("*-*-*"))
    all_windows, all_labels, all_sids = [], [], []

    for sid, session_dir in enumerate(sessions):
        acc_path = session_dir / "Accelerometer.csv"
        gyro_path = session_dir / "Gyroscope.csv"
        if not (acc_path.exists() and gyro_path.exists()):
            continue
        acc, gyro = load_phyphox_csv(str(acc_path), str(gyro_path))
        wins = process_session(acc, gyro)
        all_windows.append(wins)
        all_labels.append(np.full(len(wins), DUNQI_LABEL, dtype=np.int64))
        all_sids.append(np.full(len(wins), sid, dtype=np.int64))
        print(f"  [Dunqi] Session {sid}: {len(wins)} windows")

    return np.concatenate(all_windows, axis=0), np.concatenate(all_labels), np.concatenate(all_sids)


def resample_to(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a single window (C, T) to target_len timesteps via linear interpolation.
    """
    from scipy.interpolate import interp1d
    C, T = x.shape
    t_old = np.linspace(0, 1, T)
    t_new = np.linspace(0, 1, target_len)
    out = np.empty((C, target_len), dtype=x.dtype)
    for c in range(C):
        f = interp1d(t_old, x[c], kind='linear')
        out[c] = f(t_new)
    return out


def load_cappimu_sweeping_mopping(cappimu_data_root: str, target_len: int = SEQ_LEN):
    """
    Load CAPPIMU sweeping (label 7) and mopping (label 8) data.
    The CAPPIMU data lives under data/CAPPIMU-main/data/data/<subject>/<position>.csv
    We need chest position (chest.csv) files.

    Raw window is 300 timesteps at 30Hz (~10s). We resample to target_len (128).
    """
    data_root = Path(cappimu_data_root)

    # Label name → remapped integer
    label_map = {"sweep": 7, "mop": 8}
    X_list, y_list = [], []

    for subject_dir in sorted(data_root.glob("subject_*")):
        chest_file = subject_dir / "chest.csv"
        if not chest_file.exists():
            continue

        try:
            raw = np.loadtxt(chest_file, delimiter=',', skiprows=1, dtype=str)
        except Exception:
            continue

        # Column layout: time,ax,ay,az,wx,wy,wz,ex,ey,ez,label
        if raw.ndim == 1:
            continue

        # Parse timestamps and data
        try:
            times = raw[:, 0].astype(float)
            data_cols = raw[:, 1:10].astype(float)
            labels_str = raw[:, 10]
        except (ValueError, IndexError):
            continue

        # Resample each continuous segment to SEQ_LEN windows
        WINDOW_SIZE = 300  # CAPPIMU window = 300 samples
        STRIDE = 150        # 50% overlap

        for start in range(0, len(data_cols) - WINDOW_SIZE + 1, STRIDE):
            segment = data_cols[start:start + WINDOW_SIZE]
            label_str = labels_str[start + WINDOW_SIZE // 2].strip()

            if label_str not in label_map:
                continue

            # Resample to target_len
            seg_t = np.linspace(0, 1, WINDOW_SIZE)
            seg_out = np.empty((9, target_len), dtype=np.float32)
            for c in range(9):
                f = interp1d(seg_t, segment[:, c], kind='linear', fill_value='extrapolate')
                seg_out[c] = f(np.linspace(0, 1, target_len))

            X_list.append(seg_out)
            y_list.append(label_map[label_str])

    X = np.stack(X_list, axis=0)  # (N, 9, target_len)
    y = np.array(y_list)
    print(f"  [CAPPIMU] Loaded {len(y)} windows: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


def normalize_all(X: np.ndarray) -> np.ndarray:
    """Z-score normalize all windows together."""
    X_t = torch.from_numpy(X).float()
    mean = X_t.mean(dim=(0, 2), keepdim=True)
    std = X_t.std(dim=(0, 2), keepdim=True)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)
    return ((X_t - mean) / std).numpy()


def split_by_shot(X: np.ndarray, y: np.ndarray,
                  shot: int, dunqi_sids: np.ndarray = None,
                  seed: int = 42) -> tuple:
    """
    Split data into support (few samples) and query (rest) sets.

    For CAPPIMU classes: take first `shot` windows per class as support.
    For dunqi (DUNQI_LABEL): use session 0 as support, sessions 1+ as query.

    Returns: support_X, support_y, query_X, query_y  (numpy arrays)
    """
    np.random.seed(seed)

    # Remap labels: sweeping=0, mopping=1, dunqi=2
    label_map = {}
    cur = 0
    for lbl in sorted(set(CAPPIMU_LABELS.keys())):
        label_map[lbl] = cur; cur += 1
    label_map[DUNQI_LABEL] = cur  # 2

    y_remapped = np.array([label_map[l] for l in y])

    support_X_list, support_y_list = [], []
    query_X_list, query_y_list = [], []

    # CAPPIMU classes
    for orig_lbl, new_lbl in [(7, 0), (8, 1)]:
        mask = (y == orig_lbl)
        class_X = X[mask]
        class_y = y_remapped[mask]
        indices = np.random.permutation(len(class_X))
        support_X_list.append(class_X[indices[:shot]])
        support_y_list.append(class_y[indices[:shot]])
        query_X_list.append(class_X[indices[shot:]])
        query_y_list.append(class_y[indices[shot:]])

    # Dunqi class
    if dunqi_sids is not None:
        support_mask = (dunqi_sids == 0)
        query_mask = (dunqi_sids > 0)
        support_X_list.append(X[support_mask])
        support_y_list.append(y_remapped[support_mask])
        query_X_list.append(X[query_mask])
        query_y_list.append(y_remapped[query_mask])

    support_X = np.concatenate(support_X_list, axis=0)
    support_y = np.concatenate(support_y_list)
    query_X = np.concatenate(query_X_list, axis=0)
    query_y = np.concatenate(query_y_list)

    return support_X, support_y, query_X, query_y


# ════════════════════════════════════════════════════════════════════════════
#  Full Learning Model
# ════════════════════════════════════════════════════════════════════════════

def build_full_model(num_classes=3):
    model = DualStreamHAR(
        num_classes=num_classes,
        num_channels=NUM_CHANNELS,
        seq_len=SEQ_LEN,
        d_model=128,
        cnn_feat=256,
        dropout=0.4,
    )
    return model.to(DEVICE)


def train_full(model, support_X, support_y, query_X, query_y,
               epochs=30, lr=1e-3, wd=1e-4, batch_size=16):
    """Train the full model on support set, evaluate on query set."""
    model.train()

    # Build dataloaders
    support_ds = TensorDataset(
        torch.from_numpy(support_X).float(),
        torch.from_numpy(support_y).long()
    )
    query_ds = TensorDataset(
        torch.from_numpy(query_X).float(),
        torch.from_numpy(query_y).long()
    )
    support_loader = DataLoader(support_ds, batch_size=batch_size, shuffle=True)
    query_loader = DataLoader(query_ds, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    pbar = tqdm(range(1, epochs + 1), desc="  Full Learning", unit="epoch", leave=False)
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in support_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluate
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for X_b, y_b in query_loader:
                X_b = X_b.to(DEVICE)
                logits = model(X_b)
                preds = logits.argmax(dim=1)
                preds_all.append(preds.cpu().numpy())
                labels_all.append(y_b.numpy())

        preds_all = np.concatenate(preds_all)
        labels_all = np.concatenate(labels_all)
        acc = accuracy_score(labels_all, preds_all)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix(loss=f"{epoch_loss/len(support_loader):.4f}", acc=f"{acc:.4f}", best=f"{best_acc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return best_acc


# ════════════════════════════════════════════════════════════════════════════
#  Few-Shot (Prototypical) Model
# ════════════════════════════════════════════════════════════════════════════

class FeatureExtractorWrapper(nn.Module):
    """Wrapper that returns only features, not logits."""
    def __init__(self, dual_stream: DualStreamHAR):
        super().__init__()
        self.stream1 = dual_stream.stream1
        self.stream2 = dual_stream.stream2
        self.attention_fusion = dual_stream.attention_fusion
        self.fusion_dim = dual_stream.fusion_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        f1 = self.stream1(x_t)
        f2 = self.stream2(x)
        fused = torch.cat([f1, f2], dim=-1)
        fused = self.attention_fusion(fused)
        return fused


class SimpleProtoClassifier(nn.Module):
    """
    Simple prototypical classifier (no trainable parameters).
    Classification by nearest prototype (negative squared Euclidean distance).
    """
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim

    def forward(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # (N, feat_dim), (C, feat_dim) → (N, C) negative squared Euclidean distance
        return -torch.cdist(query_features, prototypes, p=2)


def extract_features(model, X: np.ndarray, batch_size=32) -> np.ndarray:
    """Extract features using a feature extractor model."""
    model.eval()
    ds = TensorDataset(torch.from_numpy(X).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(DEVICE)
            f = model(batch_x)
            feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0)


def train_fewshot(feat_extractor, support_X, support_y, query_X, query_y,
                  epochs=50, lr=5e-4, wd=1e-5, num_classes=3):
    """
    Few-shot learning (no fine-tuning needed):
      - Freeze feature extractor
      - Compute class prototypes from support features
      - Classify query by nearest prototype
    """
    feat_extractor.eval()

    # Extract features (frozen)
    support_feat = extract_features(feat_extractor, support_X)
    query_feat = extract_features(feat_extractor, query_X)

    support_feat_t = torch.from_numpy(support_feat).float()
    support_y_t = torch.from_numpy(support_y).long()
    query_feat_t = torch.from_numpy(query_feat).float()
    query_y_np = query_y

    # Compute prototypes from support set
    prototypes = []
    for c in range(num_classes):
        mask = (support_y_t == c)
        if mask.sum() == 0:
            prototypes.append(torch.zeros(feat_extractor.fusion_dim))
        else:
            prototypes.append(support_feat_t[mask].mean(dim=0))
    prototypes = torch.stack(prototypes)  # (C, feat_dim)

    # Evaluate across epochs (for progress bar), but no parameter updates
    pbar = tqdm(range(1, epochs + 1), desc="  Few-Shot Proto", unit="epoch", leave=False)
    best_acc = 0.0
    for epoch in pbar:
        with torch.no_grad():
            dist = torch.cdist(query_feat_t, prototypes, p=2)  # (N, C)
            preds = dist.argmin(dim=1).numpy()
            acc = accuracy_score(query_y_np, preds)

        if acc > best_acc:
            best_acc = acc
            best_prototypes = prototypes.clone()

        pbar.set_postfix(acc=f"{acc:.4f}", best=f"{best_acc:.4f}")

    # Use best prototypes
    proto_model = SimpleProtoClassifier(feat_extractor.fusion_dim)
    return best_acc, proto_model, best_prototypes


def eval_fewshot(feat_extractor, query_X, query_y, prototypes):
    """Evaluate few-shot model on query set."""
    feat_extractor.eval()
    query_feat = extract_features(feat_extractor, query_X)
    query_feat_t = torch.from_numpy(query_feat).float()
    with torch.no_grad():
        dist = torch.cdist(query_feat_t, prototypes, p=2)
        preds = dist.argmin(dim=1).numpy()
    return accuracy_score(query_y, preds)


# ════════════════════════════════════════════════════════════════════════════
#  Pre-training (required for few-shot)
# ════════════════════════════════════════════════════════════════════════════

def pretrain_feature_extractor(data_root: str, epochs=30, lr=1e-3, wd=1e-4,
                               batch_size=32, log_dir="logs/comparison"):
    """
    Pre-train DualStream on full CAPPIMU sweeping+mopping data (with dummy dunqi).
    The feature extractor is trained with full data so it learns good representations.
    """
    print("\n=== Pre-training Feature Extractor (Full CAPPIMU Data) ===")

    cappimu_root = os.path.join(data_root, "CAPPIMU-main", "data", "data")
    X_c, y_c = load_cappimu_sweeping_mopping(cappimu_root)
    X_c = normalize_all(X_c)

    # Add a dummy third class for dunqi (label 2 → all zeros placeholder)
    # We don't use dunqi in pre-training, just the two CAPPIMU classes
    # Actually: train on 2 CAPPIMU classes, the 3rd class (dunqi) is unseen
    # For pre-training, use only the 2 CAPPIMU classes
    mask_c = np.isin(y_c, [7, 8])
    X_c2 = X_c[mask_c]
    y_c2 = np.array([0 if l == 7 else 1 for l in y_c[mask_c]])

    # Full split for pre-training
    n = len(X_c2)
    perm = np.random.RandomState(RANDOM_SEED).permutation(n)
    train_n = int(0.8 * n)
    X_tr, y_tr = X_c2[perm[:train_n]], y_c2[perm[:train_n]]
    X_te, y_te = X_c2[perm[train_n:]], y_c2[perm[train_n:]]

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    test_ds = TensorDataset(torch.from_numpy(X_te).float(), torch.from_numpy(y_te).long())
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_full_model(num_classes=2).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    pbar = tqdm(range(1, epochs + 1), desc="  Pre-train", unit="epoch", leave=False)
    for epoch in pbar:
        model.train()
        for X_b, y_b in train_ld:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for X_b, y_b in test_ld:
                X_b = X_b.to(DEVICE)
                preds = model(X_b).argmax(dim=1)
                preds_all.append(preds.cpu().numpy())
                labels_all.append(y_b.numpy())
        acc = accuracy_score(np.concatenate(labels_all), np.concatenate(preds_all))

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pbar.set_postfix(acc=f"{acc:.4f}", best=f"{best_acc:.4f}")

    os.makedirs(log_dir, exist_ok=True)
    pretrained_path = os.path.join(log_dir, "pretrained_dualstream.pt")
    torch.save(best_state, pretrained_path)
    print(f"  Pre-trained model saved: {pretrained_path} (acc={best_acc:.4f})")
    return best_state


# ════════════════════════════════════════════════════════════════════════════
#  Main Experiment
# ════════════════════════════════════════════════════════════════════════════

def run_experiment(shots=[1, 3], n_runs=5, data_root=None, log_dir="logs/comparison"):
    """
    Run few-shot vs full learning comparison.

    Returns dict of results:
      results[shot]["full"] = {acc, std}
      results[shot]["proto"] = {acc, std}
    """
    if data_root is None:
        data_root = str(Path(__file__).parent / "data")

    cappimu_root = os.path.join(data_root, "CAPPIMU-main", "data", "data")
    dunqi_root = os.path.join(data_root, "for_prototypical")

    # Load raw data
    print("\n=== Loading Data ===")
    X_c, y_c = load_cappimu_sweeping_mopping(cappimu_root)
    X_d, y_d, sids_d = load_dunqi(dunqi_root)

    print(f"  CAPPIMU: {X_c.shape}, labels: {np.unique(y_c)}")
    print(f"  Dunqi:   {X_d.shape}, labels: {np.unique(y_d)}, sessions: {np.unique(sids_d)}")

    # Normalize all data together
    all_X = np.concatenate([X_c, X_d], axis=0)
    all_X = normalize_all(all_X)
    X_c_norm = all_X[:len(X_c)]
    X_d_norm = all_X[len(X_c):]
    y_c_arr = y_c
    y_d_arr = y_d

    # ── Pre-train feature extractor (once) ──────────────────────────────────
    pretrained_state = pretrain_feature_extractor(data_root, log_dir=log_dir)

    results = {}

    for shot in shots:
        print(f"\n{'='*60}")
        print(f"  Shot = {shot}")
        print(f"{'='*60}")

        full_accs, proto_accs = [], []

        run_pbar = tqdm(range(n_runs), desc=f"  Run (shot={shot})", unit="run", leave=False)
        for run in run_pbar:
            seed = RANDOM_SEED + run * 100

            # Split data: support / query
            sup_X_c, sup_y_c, qry_X_c, qry_y_c = split_by_shot(
                X_c_norm, y_c_arr, shot=shot, seed=seed
            )
            # Dunqi: session 0 → support, others → query
            d_sid_mask_sup = (sids_d == 0)
            d_sid_mask_qry = (sids_d > 0)
            sup_X_d, sup_y_d = X_d_norm[d_sid_mask_sup], y_d_arr[d_sid_mask_sup]
            qry_X_d, qry_y_d = X_d_norm[d_sid_mask_qry], y_d_arr[d_sid_mask_qry]

            # Combine
            sup_X = np.concatenate([sup_X_c, sup_X_d], axis=0)
            sup_y = np.concatenate([sup_y_c, sup_y_d], axis=0)
            qry_X = np.concatenate([qry_X_c, qry_X_d], axis=0)
            qry_y = np.concatenate([qry_y_c, qry_y_d], axis=0)

            # Remap dunqi label from 99 → 2
            sup_y = np.where(sup_y == DUNQI_LABEL, 2, sup_y).astype(np.int64)
            qry_y = np.where(qry_y == DUNQI_LABEL, 2, qry_y).astype(np.int64)

            tqdm.write(f"    Support: {sup_X.shape}, Query: {qry_X.shape}")
            tqdm.write(f"    Support label dist: {dict(zip(*np.unique(sup_y, return_counts=True)))}")

            num_classes = 3

            # ── Method 1: Full Learning ──────────────────────────────────────
            tqdm.write(f"    [Full Learning] Training...")
            model = build_full_model(num_classes=num_classes).to(DEVICE)
            full_acc = train_full(
                model, sup_X, sup_y, qry_X, qry_y,
                epochs=30, lr=1e-3, wd=1e-4
            )
            full_accs.append(full_acc)
            tqdm.write(f"    [Full Learning] Best query accuracy: {full_acc:.4f}")

            # ── Method 2: Few-Shot (Prototypical) ───────────────────────────
            tqdm.write(f"    [Few-Shot Proto] Loading pretrained features...")
            feat_model = build_full_model(num_classes=2).to(DEVICE)
            feat_model.load_state_dict(pretrained_state)
            feat_model.to(DEVICE)
            feat_extractor = FeatureExtractorWrapper(feat_model)

            proto_acc, proto_model, prototypes = train_fewshot(
                feat_extractor, sup_X, sup_y, qry_X, qry_y,
                epochs=50, lr=5e-4, wd=1e-5, num_classes=num_classes
            )
            proto_accs.append(proto_acc)
            tqdm.write(f"    [Few-Shot Proto] Best query accuracy: {proto_acc:.4f}")

        results[shot] = {
            "full": {
                "acc": float(np.mean(full_accs)),
                "std": float(np.std(full_accs)),
                "runs": [float(a) for a in full_accs],
            },
            "proto": {
                "acc": float(np.mean(proto_accs)),
                "std": float(np.std(proto_accs)),
                "runs": [float(a) for a in proto_accs],
            },
        }

        print(f"\n  Shot={shot} Summary:")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 3],
                        help="Shot values to test")
    parser.add_argument("--runs", type=int, default=5,
                        help="Number of random runs per setting")
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs/comparison")
    args = parser.parse_args()

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("  Few-Shot vs Full Learning Comparison")
    print("  Datasets: CAPPIMU (sweeping+mopping) + Real-world (蹲起)")
    print(f"  Shots: {args.shots}, Runs: {args.runs}")
    print("=" * 60)

    start = time.time()
    results = run_experiment(
        shots=args.shots,
        n_runs=args.runs,
        data_root=args.data_root,
        log_dir=args.log_dir,
    )
    elapsed = time.time() - start

    # Save results
    os.makedirs(args.log_dir, exist_ok=True)
    out_path = os.path.join(args.log_dir, "comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "elapsed_seconds": elapsed}, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")
    print(f"Total time: {elapsed:.1f}s")

    # Print summary table
    print("\n" + "=" * 60)
    print("  Final Summary")
    print("=" * 60)
    print(f"{'Shot':>6}  {'Full Learning':>20}  {'Few-Shot Proto':>20}")
    print("-" * 60)
    for shot, data in sorted(results.items()):
        fs = f"{data['full']['acc']:.4f} ± {data['full']['std']:.4f}"
        ps = f"{data['proto']['acc']:.4f} ± {data['proto']['std']:.4f}"
        print(f"{shot:>6}  {fs:>20}  {ps:>20}")


if __name__ == "__main__":
    main()
