"""
UCI HAR Dataset loader for PatchTST.
Loads raw Inertial Signals (9 channels, 128 time steps per sample).
"""

import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


SIGNAL_FILES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]

LABEL_NAMES = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


def load_signals(split: str, data_root: str):
    """
    Load all 9 inertial signal files for a given split (train/test).
    Returns:
        X: np.ndarray of shape (n_samples, 9, 128)
    """
    signals = []
    for name in SIGNAL_FILES:
        fname = f"{name}_{split}.txt"
        fpath = os.path.join(data_root, split, "Inertial Signals", fname)
        arr = np.loadtxt(fpath)          # (n_samples, 128)
        signals.append(arr)
    X = np.stack(signals, axis=1)         # (n_samples, 9, 128)
    return X


def load_labels(split: str, data_root: str):
    """
    Load labels for a given split.
    Returns:
        y: np.ndarray of shape (n_samples,), values 0-5
    """
    fpath = os.path.join(data_root, split, f"y_{split}.txt")
    y = np.loadtxt(fpath, dtype=int)     # values 1-6
    y = y - 1                             # convert to 0-5
    return y


class UCIDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.n_samples, self.n_channels, self.seq_len = self.X.shape

        if normalize:
            self._normalize()

    def _normalize(self):
        mean = self.X.mean(dim=(0, 2), keepdim=True)
        std = self.X.std(dim=(0, 2), keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.X = (self.X - mean) / std

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(data_root: str, batch_size: int = 64, num_workers: int = 0):
    """
    Build train and test DataLoaders.
    Train set is split 80/20 for train/val.
    """
    X_train = load_signals("train", data_root)
    y_train = load_labels("train", data_root)
    X_test  = load_signals("test",  data_root)
    y_test  = load_labels("test",  data_root)

    n_train = int(0.85 * len(y_train))
    indices = np.random.default_rng(42).permutation(len(y_train))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    X_tr, y_tr = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    train_ds = UCIDataset(X_tr, y_tr, normalize=True)
    val_ds   = UCIDataset(X_val, y_val, normalize=True)
    test_ds  = UCIDataset(X_test, y_test, normalize=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    print(f"  Shape: ({train_ds.n_channels} channels, {train_ds.seq_len} time steps)")
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "UCI HAR Dataset" / "UCI HAR Dataset"
    train_loader, val_loader, test_loader = get_dataloaders(DATA_ROOT, batch_size=64)
    x, y = next(iter(train_loader))
    print(f"X shape: {x.shape}  y shape: {y.shape}")
    print(f"Label distribution (train): {torch.bincount(y)}")
