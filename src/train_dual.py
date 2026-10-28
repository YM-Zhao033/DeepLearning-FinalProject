"""
Dual-Stream (PatchTST + GAF-CNN) Training for UCI HAR.
Usage: python train_dual.py --epochs 50 --batch_size 32 --lr 2e-4
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.dual_stream import DualStreamHAR
from src.data.dataset import get_dataloaders, LABEL_NAMES


def build_model(num_channels: int, num_classes: int, seq_len: int):
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
            out = model(X)
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

    preds  = torch.cat(preds)
    targets = torch.cat(targets)
    acc = accuracy_score(targets, preds)
    return total_loss / len(targets), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, targets = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            out = model(X)
            loss = criterion(out, y)

        total_loss += loss.item() * X.size(0)
        preds.append(out.argmax(dim=-1).cpu())
        targets.append(y.cpu())

    preds   = torch.cat(preds)
    targets = torch.cat(targets)
    acc     = accuracy_score(targets, preds)
    return total_loss / len(targets), acc, preds.numpy(), targets.numpy()


def print_report(y_true, y_pred, class_names):
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    print("CONFUSION MATRIX")
    print("-" * 60)
    header = "".join(f"{n:>18}" for n in class_names)
    print(f"{'':>18}{header}")
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>18}" + "".join(f"{v:>18}" for v in row))
    return cm


def main():
    parser = argparse.ArgumentParser(description="Dual-Stream HAR Classification")
    parser.add_argument("--data_root",    type=str,  default=None)
    parser.add_argument("--epochs",      type=int,  default=50)
    parser.add_argument("--batch_size",  type=int,  default=32)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,  default=0)
    parser.add_argument("--seed",        type=int,  default=42)
    parser.add_argument("--log_dir",     type=str,  default="logs/dual_stream")
    parser.add_argument("--tag",         type=str,  default="")
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = str(Path(__file__).resolve().parent / "UCI HAR Dataset" / "UCI HAR Dataset")

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.json")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  Dual-Stream HAR (PatchTST + GAF-CNN)")
    print(f"{'='*60}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Batch   : {args.batch_size}")
    print(f"  Log dir : {args.log_dir}")
    print(f"{'='*60}\n", flush=True)

    print("[1/4] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    num_channels = 9
    num_classes  = 6
    seq_len      = 128

    print(f"\n[2/4] Building dual-stream model...")
    model = build_model(num_channels, num_classes, seq_len).to(device)
    total_params  = sum(p.numel() for p in model.parameters())
    trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    scaler = torch.cuda.amp.GradScaler()

    print(f"\n[3/4] Training for {args.epochs} epochs...")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state   = None
    patience     = 15
    no_improve   = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        marker = " *" if va_acc > best_val_acc else ""
        stopper = " [EARLY STOP]" if no_improve >= patience else ""
        print(
            f"  Epoch {epoch:>3}/{args.epochs} | "
            f"Tr loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
            f"Val loss: {va_loss:.4f} acc: {va_acc:.4f} | "
            f"{elapsed:.1f}s{marker}{stopper}",
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

    cm = print_report(y_true, y_pred, list(LABEL_NAMES.values()))

    model_path = os.path.join(args.log_dir, "best_model.pt")
    torch.save(best_state, model_path)
    with open(log_path, "w") as f:
        json.dump({
            "test_accuracy": test_acc,
            "best_val_accuracy": best_val_acc,
            "history": history,
        }, f, indent=2)

    print(f"\n  Model saved to {model_path}")
    print(f"  Log saved to {log_path}")
    print(f"\n  Baseline (PatchTST-only) Test Acc: 88.36%")
    print(f"  Dual-Stream Test Acc:              {test_acc*100:.2f}%")
    diff = (test_acc - 0.8836) * 100
    print(f"  Improvement:                        {diff:+.2f}%")

    return test_acc, best_val_acc


if __name__ == "__main__":
    main()
