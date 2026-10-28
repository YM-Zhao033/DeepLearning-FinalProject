"""
Prototypical Network 训练脚本。
用于训练少样本学习分类器。

Usage:
    # 预训练阶段（使用已有双分支模型）
    python train_prototypical.py --mode pretrain --epochs 100
    
    # Episodic 训练阶段（学习度量空间）
    python train_prototypical.py --mode episodic --epochs 500 --num_ways 5 --num_support 5
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import accuracy_score

from src.models.dual_stream import DualStreamHAR
from src.models.prototypical import (
    PrototypicalClassifier, 
    EpisodicTrainer,
    euclidean_distance,
    cosine_distance
)
from src.data.dataset import get_dataloaders, LABEL_NAMES


def extract_features(model: nn.Module, dataloader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用预训练模型提取所有数据的特征。
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            # 获取特征（双分支模型的融合特征）
            # 修改：直接使用模型输出作为特征
            features = model(X)
            # 或者返回中间层特征
            all_features.append(features.cpu())
            all_labels.append(y)
    
    return torch.cat(all_features), torch.cat(all_labels)


class FeatureExtractor(nn.Module):
    """
    包装双分支模型，输出特征而非分类 logits。
    """
    def __init__(self, dual_stream_model: DualStreamHAR):
        super().__init__()
        self.stream1 = dual_stream_model.stream1
        self.stream2 = dual_stream_model.stream2
        self.attention_fusion = dual_stream_model.attention_fusion
        self.feat_dim = dual_stream_model.fusion_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回融合后的特征向量。
        """
        x_t = x.transpose(1, 2)
        f1 = self.stream1(x_t)
        f2 = self.stream2(x)
        fused = torch.cat([f1, f2], dim=-1)
        fused = self.attention_fusion(fused)
        return fused


def parse_args():
    parser = argparse.ArgumentParser(description="Prototypical Network Training")
    
    # 模式选择
    parser.add_argument("--mode", type=str, default="episodic",
                       choices=["pretrain", "episodic", "finetune"],
                       help="训练模式: pretrain(微调特征提取器), episodic(Episodic训练), finetune(微调整个模型)")
    
    # 数据参数
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    
    # Episodic 参数
    parser.add_argument("--num_ways", type=int, default=5,
                       help="每个episode的类别数")
    parser.add_argument("--num_support", type=int, default=5,
                       help="每个类别的支持集样本数")
    parser.add_argument("--num_query", type=int, default=15,
                       help="每个类别的查询集样本数")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--episodes_per_epoch", type=int, default=100,
                       help="每个epoch的episode数")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    
    # 距离度量
    parser.add_argument("--distance", type=str, default="euclidean",
                       choices=["euclidean", "cosine"],
                       help="距离度量方式")
    
    # 模型参数
    parser.add_argument("--pretrained_path", type=str, default="logs/dual_stream/best_model.pt",
                       help="预训练模型路径")
    
    # 其他
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_dir", type=str, default="logs/prototypical")
    parser.add_argument("--save_interval", type=int, default=50,
                       help="保存模型的间隔(epoch)")
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_backbone(pretrained_path: str, num_classes: int, 
                             num_channels: int, seq_len: int, device: str):
    """
    加载预训练的双分支模型。
    """
    print(f"加载预训练模型: {pretrained_path}")
    
    # 构建模型
    model = DualStreamHAR(
        num_classes=num_classes,
        num_channels=num_channels,
        seq_len=seq_len,
        d_model=128,
        cnn_feat=256,
        dropout=0.3,
    )
    
    # 加载权重
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"  成功加载预训练权重")
    else:
        print(f"  警告: 预训练模型不存在，将从头训练")
    
    return model.to(device)


def extract_all_features(model: nn.Module, dataloader, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    提取所有数据的特征。
    """
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            # 获取特征
            features = model.extract_features(X) if hasattr(model, 'extract_features') else model(X)
            all_features.append(features.cpu())
            all_labels.append(y)
    
    return torch.cat(all_features), torch.cat(all_labels)


def train_episodic(args):
    """
    Episodic 训练模式。
    
    特点：
    1. 冻结预训练特征提取器
    2. 使用 Episodic 采样训练 Prototypical Layer
    3. 学习好的度量空间
    """
    print(f"\n{'='*60}")
    print(f"  Prototypical Network - Episodic Training")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    if args.data_root is None:
        args.data_root = str(Path(__file__).resolve().parent / "UCI HAR Dataset" / "UCI HAR Dataset")
    
    print("\n[1/5] 加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    num_channels = 9
    num_classes = 6
    seq_len = 128
    
    # 加载预训练模型
    print("\n[2/5] 加载预训练特征提取器...")
    backbone = load_pretrained_backbone(
        args.pretrained_path, num_classes, num_channels, seq_len, device
    )
    backbone.eval()
    
    # 冻结 backbone
    for param in backbone.parameters():
        param.requires_grad = False
    
    # 提取特征（用于 Episodic 训练）
    print("\n[3/5] 提取训练集特征...")
    train_features, train_labels = [], []
    with torch.no_grad():
        for X, y in train_loader:
            X = X.to(device)
            x_t = X.transpose(1, 2)
            f1 = backbone.stream1(x_t)
            f2 = backbone.stream2(X)
            fused = torch.cat([f1, f2], dim=-1)
            fused = backbone.attention_fusion(fused)
            train_features.append(fused.cpu())
            train_labels.append(y)
    
    train_features = torch.cat(train_features)
    train_labels = torch.cat(train_labels)
    
    # 同样提取验证集和测试集特征
    val_features, val_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            x_t = X.transpose(1, 2)
            f1 = backbone.stream1(x_t)
            f2 = backbone.stream2(X)
            fused = torch.cat([f1, f2], dim=-1)
            fused = backbone.attention_fusion(fused)
            val_features.append(fused.cpu())
            val_labels.append(y)
    val_features = torch.cat(val_features)
    val_labels = torch.cat(val_labels)
    
    print(f"  训练特征形状: {train_features.shape}")
    print(f"  验证特征形状: {val_features.shape}")
    
    # 创建 Prototypical 分类器
    print("\n[4/5] 创建 Prototypical 分类器...")
    feat_dim = backbone.fusion_dim  # 384
    proto_classifier = PrototypicalClassifier(
        backbone=backbone,
        feat_dim=feat_dim,
        num_classes=num_classes,
        device=str(device)
    )
    
    # 创建 Episodic 训练器
    trainer = EpisodicTrainer(
        model=proto_classifier,
        num_ways=args.num_ways,
        num_support=args.num_support,
        num_query=args.num_query,
        distance=args.distance
    )
    
    # 优化 Prototypical Layer
    optimizer = AdamW(
        proto_classifier.prototypical_layer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 训练循环
    print(f"\n[5/5] 开始 Episodic 训练...")
    print(f"  每 epoch {args.episodes_per_epoch} episodes")
    print(f"  {args.num_ways}-way {args.num_support}-shot 任务")
    print(f"{'='*60}")
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "best_val_acc": 0.0,
    }
    
    best_val_acc = 0.0
    best_state = None
    
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        epoch_accs = []
        
        # 训练 episodes
        for _ in range(args.episodes_per_epoch):
            # 创建 episode
            support_feat, support_lab, query_feat, query_lab = trainer.create_episode(
                train_features, train_labels
            )
            
            # 训练
            loss, acc = trainer.train_episode(
                support_feat, support_lab, query_feat, query_lab,
                optimizer
            )
            epoch_losses.append(loss)
            epoch_accs.append(acc)
        
        # 验证
        val_losses = []
        val_accs = []
        for _ in range(min(20, args.episodes_per_epoch)):  # 少量验证 episodes
            support_feat, support_lab, query_feat, query_lab = trainer.create_episode(
                val_features, val_labels
            )
            loss, acc = trainer.evaluate_episode(
                support_feat, support_lab, query_feat, query_lab
            )
            val_losses.append(loss)
            val_accs.append(acc)
        
        train_loss = np.mean(epoch_losses)
        train_acc = np.mean(epoch_accs)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        marker = " *" if val_acc > best_val_acc else ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.cpu().clone() 
                for k, v in proto_classifier.prototypical_layer.state_dict().items()
            }
        
        if epoch % 10 == 0 or marker:
            print(f"  Epoch {epoch:>4}/{args.epochs} | "
                  f"Tr Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{marker}")
        
        # 保存检查点
        if epoch % args.save_interval == 0:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "prototypical_layer_state": proto_classifier.prototypical_layer.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "history": history,
            }, checkpoint_path)
    
    print(f"\n  最佳验证准确率: {best_val_acc:.4f}")
    
    # 保存最终模型
    os.makedirs(args.log_dir, exist_ok=True)
    final_path = os.path.join(args.log_dir, "prototypical_final.pt")
    torch.save(best_state, final_path)
    print(f"  模型保存至: {final_path}")
    
    # 保存训练历史
    log_path = os.path.join(args.log_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  训练历史保存至: {log_path}")
    
    return best_val_acc, history


def evaluate_fewshot(args):
    """
    少样本评估：在少量样本上计算原型，然后评估。
    """
    print(f"\n{'='*60}")
    print(f"  Few-Shot Evaluation")
    print(f"{'='*60}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    if args.data_root is None:
        args.data_root = str(Path(__file__).resolve().parent / "UCI HAR Dataset" / "UCI HAR Dataset")
    
    print("\n[1/4] 加载数据...")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers
    )
    
    # 加载预训练模型
    print("\n[2/4] 加载预训练特征提取器...")
    backbone = load_pretrained_backbone(
        args.pretrained_path, 6, 9, 128, device
    )
    backbone.eval()
    
    # 提取特征
    print("\n[3/4] 提取特征...")
    all_features, all_labels = [], []
    with torch.no_grad():
        for X, y in train_loader:
            X = X.to(device)
            x_t = X.transpose(1, 2)
            f1 = backbone.stream1(x_t)
            f2 = backbone.stream2(X)
            fused = torch.cat([f1, f2], dim=-1)
            fused = backbone.attention_fusion(fused)
            all_features.append(fused.cpu())
            all_labels.append(y)
    
    features = torch.cat(all_features)
    labels = torch.cat(all_labels)
    
    # 创建分类器
    proto_classifier = PrototypicalClassifier(
        backbone=backbone,
        feat_dim=384,
        num_classes=6,
        device=str(device)
    )
    
    # 加载训练好的 Prototypical Layer
    proto_path = os.path.join(args.log_dir, "prototypical_final.pt")
    if os.path.exists(proto_path):
        state_dict = torch.load(proto_path, map_location=device)
        proto_classifier.prototypical_layer.load_state_dict(state_dict)
        print(f"  加载 Prototypical Layer: {proto_path}")
    
    print("\n[4/4] Few-Shot 评估...")
    
    # 测试不同 shot 数
    for shot in [1, 5, 10, 20]:
        # 模拟：取前 shot 个样本作为支持集
        support_features_list = []
        support_labels_list = []
        
        for c in range(6):
            mask = (labels == c)
            class_features = features[mask][:shot]
            support_features_list.append(class_features)
            support_labels_list.append(torch.full((len(class_features),), c, dtype=torch.long))
        
        support_features = torch.cat(support_features_list)
        support_labels = torch.cat(support_labels_list)
        
        # 计算原型
        proto_classifier.update_prototypes(support_features, support_labels, num_classes=6)
        
        # 在测试集上评估
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                preds, dists = proto_classifier.predict_with_distance(X)
                correct += (preds.cpu() == y).sum().item()
                total += len(y)
        
        acc = correct / total
        print(f"  {shot}-shot 准确率: {acc*100:.2f}%")
    
    return {}


def main():
    args = parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    if args.mode == "episodic":
        train_episodic(args)
    elif args.mode == "evaluate":
        evaluate_fewshot(args)
    else:
        print(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
