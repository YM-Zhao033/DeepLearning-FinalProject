"""
Dual-Stream HAR Training on Phyphox 6-class Data.

Data processing pipeline:
  1. Load raw Accelerometer + Gyroscope CSVs from data/phyphox
  2. Resample to 50 Hz (UCI HAR format)
  3. Remove first and last 4 seconds (transient)
  4. Window into 2.56 s (128 samples) non-overlapping segments
  5. Train / test split (80 / 20)

Usage:
  python train_dual_phyphox.py --epochs 80 --batch_size 32
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from src.models.dual_stream import DualStreamHAR

# ── Label mapping (matches UCI HAR) ────────────────────────────────────────
LABEL_NAMES = {
    0: "WALKING",
    1: "WALKING_UPSTAIRS",
    2: "WALKING_DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING",
}

# ── Fixed signal configuration ───────────────────────────────────────────────
SAMPLE_RATE   = 50          # Hz
WINDOW_SIZE   = 128         # 2.56 s × 50 Hz
NUM_CHANNELS  = 6           # 3 accel + 3 gyro (phyphox provides no separate gravity)


# ════════════════════════════════════════════════════════════════════════════
#  Data loading & preprocessing
# ════════════════════════════════════════════════════════════════════════════

def load_raw_activity(folder: Path):
    """
    Load one activity folder's Accelerometer and Gyroscope CSV files.

    Returns:
        accel: (N, 3) raw accelerometer data  [m/s^2]
        gyro:  (N, 3) raw gyroscope data       [rad/s]
        time:  (N,)   timestamps in seconds
    """
    accel = np.genfromtxt(folder / "Accelerometer.csv", delimiter=",",
                          skip_header=1, filling_values=np.nan)
    gyro  = np.genfromtxt(folder / "Gyroscope.csv",  delimiter=",",
                          skip_header=1, filling_values=np.nan)

    time  = accel[:, 0]
    accel = accel[:, 1:4]
    gyro  = gyro[:,  1:4]
    return accel, gyro, time


def resample_to_50hz(signal: np.ndarray, time: np.ndarray,
                      target_rate: float = 50.0) -> np.ndarray:
    """
    Linear interpolation to uniform 50 Hz sampling.
    Returns (N, 3) resampled signal.
    """
    from scipy.interpolate import interp1d
    target_time = np.arange(time[0], time[-1], 1.0 / target_rate)
    interp = interp1d(time, signal, kind="linear", axis=0, fill_value="extrapolate")
    return interp(target_time)


def trim_and_window(signal: np.ndarray, time: np.ndarray,
                    trim_seconds: float = 4.0,
                    window_size: int = 128) -> list[np.ndarray]:
    """
    1. Remove first & last `trim_seconds` seconds.
    2. Split into non-overlapping windows of `window_size` samples.

    Returns list of windows, each (window_size, 3).
    """
    start_t = time[0] + trim_seconds
    end_t   = time[-1] - trim_seconds

    mask = (time >= start_t) & (time <= end_t)
    signal = signal[mask]
    time   = time[mask]

    n = len(signal)
    n_windows = n // window_size
    if n_windows == 0:
        return []
    signal = signal[: n_windows * window_size]
    windows = np.stack(np.split(signal, n_windows, axis=0), axis=0)
    return windows  # (n_windows, window_size, 3)


def process_activity(folder: Path, label: int, min_trim: float = 4.0,
                     window_samples: int = 128) -> np.ndarray:
    """
    Load, resample, trim, and window a single activity folder.
    Returns (n_windows, window_samples, 9) merged signal array.

    min_trim: seconds to remove from each end.  If remaining duration after
              trimming is < window duration, trim is reduced automatically.
    """
    accel, gyro, time = load_raw_activity(folder)

    # Sync accelerometer and gyroscope to the shorter length
    n = min(len(accel), len(gyro))
    accel = accel[:n]
    gyro  = gyro[:n]
    time  = time[:n]

    # Resample both to 50 Hz on shared time grid
    accel_rs = resample_to_50hz(accel, time)
    gyro_rs  = resample_to_50hz(gyro,  time)

    shared_time = np.arange(accel_rs.shape[0]) / SAMPLE_RATE + time[0]

    # Adaptive trim: if recording is too short, reduce trim
    total_dur = shared_time[-1] - shared_time[0]
    trim = min_trim
    window_dur = window_samples / SAMPLE_RATE
    if total_dur - 2 * trim < window_dur:      # less than window duration left after standard trim
        trim = total_dur / 4.0          # use 1/4 of duration each side

    accel_wins = trim_and_window(accel_rs, shared_time, trim_seconds=trim,
                                  window_size=window_samples)
    gyro_wins  = trim_and_window(gyro_rs,  shared_time, trim_seconds=trim,
                                  window_size=window_samples)

    if len(accel_wins) == 0:
        return np.empty((0, window_samples, 9))

    merged = np.concatenate([accel_wins, gyro_wins], axis=2)  # (n_w, window_samples, 6)
    return merged


def load_phyphox_data(data_root: Path, window_samples: int = 128):
    """
    Load all 6 activity classes from the phyphox directory.

    Class mapping (matching LABEL_NAMES):
      0 → walking
      1 → upstair  (combine upstair + upstair2)
      2 → downstair
      3 → sitting
      4 → standing
      5 → laying

    Returns:
      X: (N, window_samples, 9)
      y: (N,)  labels 0–5
    """
    folders = {
        0: ["walking"],
        1: ["upstair", "upstair2"],
        2: ["downstair"],
        3: ["sitting"],
        4: ["standing"],
        5: ["laying"],
    }

    all_X, all_y = [], []
    for label, names in folders.items():
        for name in names:
            folder = data_root / name
            if not folder.exists():
                print(f"  [WARN] Folder not found: {folder}")
                continue
            X = process_activity(folder, label, window_samples=window_samples)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(np.full(len(X), label, dtype=np.int64))
                print(f"  {name:15s} → label {label}: {len(X):4d} windows")
            else:
                print(f"  {name:15s} → label {label}: no windows after trimming")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


# ════════════════════════════════════════════════════════════════════════════
#  Dataset & DataLoader
# ════════════════════════════════════════════════════════════════════════════

class PhyphoxDataset(Dataset):
    """
    X shape: (N, 128, 9)  → transposed to (N, 9, 128) for the model.
    Normalization: per-channel z-score over train set.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 mean: np.ndarray = None, std: np.ndarray = None):
        # store (N, T, C) then transpose to (N, C, T) at forward pass
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.n_samples, self.seq_len, self.n_channels = self.X.shape

        if mean is None:
            mean = self.X.mean(dim=(0, 2), keepdim=True)   # (1, T, 1)
            std  = self.X.std(dim=(0, 2), keepdim=True)
            std  = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.mean = mean
        self.std  = std
        self.X = (self.X - self.mean) / self.std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # model expects (C, T)
        x = self.X[idx].transpose(0, 1)
        return x, self.y[idx]


def get_phyphox_dataloaders(data_root: Path, batch_size: int = 32,
                              num_workers: int = 0, test_size: float = 0.2,
                              seed: int = 42, window_samples: int = 128):
    """
    Build train / val / test DataLoaders from phyphox data.
    Train is further split 85/15 for train/val.
    """
    X, y = load_phyphox_data(data_root, window_samples=window_samples)
    print(f"\n  Total: {len(y)} samples | Shape: {X.shape}")

    # First split: train_val / test
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y, shuffle=True
    )
    # Second split: train / val (use 30% of train_val for val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.3, random_state=seed,
        stratify=y_tv, shuffle=True
    )

    print(f"  Train: {len(y_train):4d} | Val: {len(y_val):4d} | Test: {len(y_test):4d}")

    train_ds = PhyphoxDataset(X_train, y_train)
    val_ds   = PhyphoxDataset(X_val,   y_val,   train_ds.mean, train_ds.std)
    test_ds  = PhyphoxDataset(X_test,  y_test,  train_ds.mean, train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


# ════════════════════════════════════════════════════════════════════════════
#  Training & evaluation helpers
# ════════════════════════════════════════════════════════════════════════════

def build_model(num_classes: int = 6, num_channels: int = 9, seq_len: int = 128):
    model = DualStreamHAR(
        num_classes=num_classes,
        num_channels=num_channels,
        seq_len=seq_len,
        d_model=128,
        cnn_feat=256,
        dropout=0.3,
    )
    return model


def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler):
    model.train()
    total_loss, preds, targets = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            out  = model(X)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * X.size(0)
        preds.append(out.argmax(dim=-1).cpu())
        targets.append(y.cpu())

    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    acc     = accuracy_score(targets, preds)
    return total_loss / len(targets), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, targets = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            out  = model(X)
            loss = criterion(out, y)

        total_loss += loss.item() * X.size(0)
        preds.append(out.argmax(dim=-1).cpu())
        targets.append(y.cpu())

    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    acc     = accuracy_score(targets, preds)
    return total_loss / len(targets), acc, preds.numpy(), targets.numpy()


def print_report(y_true, y_pred, class_names):
    present = sorted(set(y_true) | set(y_pred))
    if len(present) < len(class_names):
        # Print which classes are missing
        missing = set(range(len(class_names))) - set(present)
        print(f"\n[Note] Classes with no samples: {sorted(missing)} → excluded from report")
    active_names = [class_names[i] for i in present]

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred, target_names=active_names, digits=4,
        labels=present, zero_division=0
    )
    print(report)

    cm = confusion_matrix(y_true, y_pred, labels=present)
    print("CONFUSION MATRIX")
    print("-" * 60)
    header = "".join(f"{n:>18}" for n in active_names)
    print(f"{'':>18}{header}")
    for i, row in enumerate(cm):
        print(f"{active_names[i]:>18}" + "".join(f"{v:>18}" for v in row))

    class_acc = {}
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        correct = cm[i, i]
        class_acc[name] = {"correct": int(correct), "total": int(total), "accuracy": float(correct / total) if total > 0 else 0.0}
    return cm, class_acc


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Dual-Stream HAR on Phyphox Data")
    parser.add_argument("--data_root",     type=str,  default=None)
    parser.add_argument("--epochs",        type=int,  default=80)
    parser.add_argument("--batch_size",    type=int,  default=32)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--num_workers",   type=int,  default=0)
    parser.add_argument("--seed",          type=int,  default=42)
    parser.add_argument("--log_dir",       type=str,  default="logs/dual_stream_phyphox")
    parser.add_argument("--dropout",       type=float, default=0.3)
    parser.add_argument("--test_size",     type=float, default=0.5)
    parser.add_argument("--window_seconds", type=float, default=2.56,
                        help="Window length in seconds (default: 2.56)")
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = str(SCRIPT_DIR / "data" / "phyphox")
    data_root = Path(args.data_root)

    window_samples = int(args.window_seconds * SAMPLE_RATE)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"train_log_{args.window_seconds}s.json")

    print(f"\n{'='*60}")
    print(f"  Dual-Stream HAR on Phyphox Data (6-class)")
    print(f"{'='*60}")
    print(f"  Device          : {device}")
    print(f"  Data dir       : {data_root}")
    print(f"  Window         : {args.window_seconds} s ({window_samples} samples)")
    print(f"  Epochs         : {args.epochs}")
    print(f"  LR             : {args.lr}")
    print(f"  Batch          : {args.batch_size}")
    print(f"  Dropout        : {args.dropout}")
    print(f"  Test %         : {args.test_size * 100:.0f}%")
    print(f"{'='*60}\n")

    print("[1/4] Loading & preprocessing phyphox data...")
    train_loader, val_loader, test_loader = get_phyphox_dataloaders(
        data_root, batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_size=args.test_size,
        seed=args.seed,
        window_samples=window_samples,
    )

    print(f"\n[2/4] Building dual-stream model...")
    model = build_model(
        num_classes=6, num_channels=NUM_CHANNELS, seq_len=window_samples
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Stream 1 (PatchTST): d_model=128, layers=3, heads=8")
    print(f"  Stream 2 (GAF-CNN):  18ch GAF → 32→64→128→256 conv")
    print(f"  Fusion:              384 → Multi-Head Attention (4 heads) → FC")
    print(f"  Total params: {total_params:,}  |  Trainable: {trainable:,}")

    criterion  = nn.CrossEntropyLoss()
    optimizer  = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler  = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    print(f"\n[3/4] Training for {args.epochs} epochs...")
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [],   "val_acc": [],
    }
    best_val_acc = 0.0
    best_state   = None
    patience     = 20
    no_improve   = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler
        )
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        marker  = " *" if va_acc > best_val_acc else ""
        stop   = " [EARLY STOP]" if no_improve >= patience else ""
        print(
            f"  Epoch {epoch:>3}/{args.epochs} | "
            f"Tr loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
            f"Val loss: {va_loss:.4f} acc: {va_acc:.4f} | "
            f"{elapsed:.1f}s{marker}{stop}",
            flush=True
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    print(f"\n[4/4] Test evaluation (best val acc: {best_val_acc:.4f})...")
    model.load_state_dict(best_state)
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    cm, class_acc = print_report(y_true, y_pred, list(LABEL_NAMES.values()))

    model_path = os.path.join(args.log_dir, "best_model.pt")
    torch.save(best_state, model_path)
    with open(log_path, "w") as f:
        json.dump({
            "test_accuracy":    float(test_acc),
            "best_val_accuracy": float(best_val_acc),
            "history":          history,
            "n_train": len(train_loader.dataset),
            "n_val":   len(val_loader.dataset),
            "n_test":  len(test_loader.dataset),
            "cm":      cm.tolist(),
            "class_accuracy": class_acc,
        }, f, indent=2)

    print(f"\n  Model saved : {model_path}")
    print(f"  Log saved   : {log_path}")
    print(f"\n  Test Accuracy: {test_acc*100:.2f}%")

    return test_acc, best_val_acc


if __name__ == "__main__":
    main()
