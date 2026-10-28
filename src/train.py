"""
PatchTST Classification Training Script for UCI HAR Dataset.
Uses HuggingFace Transformers' PatchTSTForClassification.
"""

import os
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

from transformers import PatchTSTConfig, PatchTSTForClassification
from src.data.dataset import get_dataloaders, LABEL_NAMES


def build_model(num_channels: int, num_classes: int, seq_len: int):
    config = PatchTSTConfig(
        num_input_channels=num_channels,
        num_targets=num_classes,
        context_length=seq_len,
        patch_length=16,
        stride=8,
        num_hidden_layers=3,
        d_model=128,
        num_attention_heads=8,
        ffn_dim=256,
        norm_type="batchnorm",
        pre_norm=True,
        positional_encoding_type="sincos",
        use_cls_token=True,
        head_dropout=0.1,
        attention_dropout=0.1,
        ff_dropout=0.1,
    )
    model = PatchTSTForClassification(config)
    return model, config


def train_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss, preds, targets = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X = X.transpose(1, 2)            # (B, 9, 128) -> (B, 128, 9)
        optimizer.zero_grad()
        out = model(past_values=X)
        loss = criterion(out.prediction_logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * X.size(0)
        preds.append(out.prediction_logits.argmax(dim=-1).cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    acc = accuracy_score(targets, preds)
    return total_loss / len(targets), acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, targets = 0, [], []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        X = X.transpose(1, 2)            # (B, 9, 128) -> (B, 128, 9)
        out = model(past_values=X)
        loss = criterion(out.prediction_logits, y)

        total_loss += loss.item() * X.size(0)
        preds.append(out.prediction_logits.argmax(dim=-1).cpu())
        targets.append(y.cpu())

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    acc = accuracy_score(targets, preds)
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


def main():
    parser = argparse.ArgumentParser(description="PatchTST UCI HAR Classification")
    parser.add_argument("--data_root",   type=str,  default=None)
    parser.add_argument("--epochs",      type=int,  default=50)
    parser.add_argument("--batch_size",  type=int,  default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,  default=0)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--log_dir",    type=str,  default="logs")
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = str(Path(__file__).resolve().parent / "UCI HAR Dataset" / "UCI HAR Dataset")

    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, "train_log.json")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    print("\n[1/4] Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    num_channels = 9
    num_classes   = 6
    seq_len       = 128

    print(f"\n[2/4] Building PatchTST model...")
    model, config = build_model(num_channels, num_classes, seq_len)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} | Trainable: {trainable:,}")
    print(f"  Config: layers={config.num_hidden_layers}, d_model={config.d_model}, "
          f"heads={config.num_attention_heads}, patch_len={config.patch_length}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    print(f"\n[3/4] Training for {args.epochs} epochs...")
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_state    = None

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc  = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        va_loss, va_acc, _, _ = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        marker = " *" if va_acc > best_val_acc else ""
        print(
            f"  Epoch {epoch:>3}/{args.epochs} | "
            f"Train loss: {tr_loss:.4f} acc: {tr_acc:.4f} | "
            f"Val loss: {va_loss:.4f} acc: {va_acc:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[4/4] Test evaluation (best val acc: {best_val_acc:.4f})...")
    model.load_state_dict(best_state)
    test_loss, test_acc, y_pred, y_true = evaluate(model, test_loader, criterion, device)
    print(f"\n  Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")

    print_report(y_true, y_pred, list(LABEL_NAMES.values()))

    torch.save(best_state, os.path.join(args.log_dir, "best_model.pt"))
    with open(log_path, "w") as f:
        json.dump({"test_accuracy": test_acc, "best_val_accuracy": best_val_acc, "history": history}, f, indent=2)
    print(f"\n  Model saved to {args.log_dir}/best_model.pt")
    print(f"  Log saved to {log_path}")


if __name__ == "__main__":
    main()
