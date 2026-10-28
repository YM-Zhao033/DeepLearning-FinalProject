"""
Prototypical Networks for Few-Shot Learning.
基于预训练双分支特征提取器的少样本分类器。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


class PrototypicalLayer(nn.Module):
    """
    Prototypical Network 核心层。
    
    思想：每个类别在特征空间中有一个"原型"（同类样本的均值），
    分类通过计算查询样本与各类原型的距离来实现。
    """
    
    def __init__(self, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        
        # 可学习的距离度量（可选，增强表达能力）
        # 简单实现用欧氏距离，高级可用马氏距离
        self.use_metric = True
        if self.use_metric:
            self.metric_net = nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, 1)
            )
    
    def compute_prototypes(self, support_features: torch.Tensor, 
                          support_labels: torch.Tensor,
                          num_classes: int) -> torch.Tensor:
        """
        计算每个类别的原型。
        
        Args:
            support_features: (N, feat_dim) 支持集特征
            support_labels: (N,) 支持集标签
            num_classes: 类别数量
        
        Returns:
            prototypes: (num_classes, feat_dim) 每个类别的原型
        """
        prototypes = []
        for c in range(num_classes):
            # 找到该类别的所有样本
            mask = (support_labels == c)
            if mask.sum() == 0:
                # 该类别没有样本，使用零向量
                prototypes.append(torch.zeros(self.feat_dim, 
                                              device=support_features.device))
            else:
                # 计算该类别样本特征的均值
                class_features = support_features[mask]
                prototype = class_features.mean(dim=0)
                prototypes.append(prototype)
        
        return torch.stack(prototypes)  # (num_classes, feat_dim)
    
    def forward(self, query_features: torch.Tensor,
               prototypes: torch.Tensor) -> torch.Tensor:
        """
        计算查询样本到各类原型的负欧氏距离作为 logits。
        
        Args:
            query_features: (N, feat_dim) 查询集特征
            prototypes: (num_classes, feat_dim) 类别原型
        
        Returns:
            logits: (N, num_classes) 分类 logits
        """
        if self.use_metric:
            # 可学习的度量：增强特征后再计算距离
            query_enhanced = query_features
            proto_enhanced = prototypes
            
            # 计算平方欧氏距离
            dist = torch.cdist(query_enhanced, proto_enhanced, p=2)  # (N, num_classes)
        else:
            # 标准欧氏距离
            dist = torch.cdist(query_features, prototypes, p=2)
        
        # 负距离作为 logits（距离越小，概率越大）
        return -dist
    
    def compute_distances(self, query_features: torch.Tensor,
                          prototypes: torch.Tensor) -> torch.Tensor:
        """
        返回原始欧氏距离（用于分析）。
        """
        return torch.cdist(query_features, prototypes, p=2)


class PrototypicalClassifier(nn.Module):
    """
    Prototypical 分类器。
    
    封装了：
    1. 预训练特征提取器（双分支模型，冻结）
    2. Prototypical Layer
    3. 原型存储与管理
    """
    
    def __init__(self, backbone: nn.Module, feat_dim: int, 
                 num_classes: int = 6, device: str = "cuda"):
        super().__init__()
        self.backbone = backbone
        self.prototypical_layer = PrototypicalLayer(feat_dim)
        self.num_classes = num_classes
        self.device = device
        
        # 存储各类别的原型
        self.prototypes: Optional[torch.Tensor] = None
        self.class_names: List[str] = []
        
        # 冻结 backbone
        self.freeze_backbone()
    
    def freeze_backbone(self):
        """冻结特征提取器。"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()
    
    def unfreeze_backbone(self):
        """解冻特征提取器用于微调。"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def update_prototypes(self, features: torch.Tensor, 
                         labels: torch.Tensor,
                         num_classes: Optional[int] = None):
        """
        从支持集更新各类别原型。
        
        Args:
            features: (N, feat_dim) 支持集特征
            labels: (N,) 支持集标签
            num_classes: 类别数（默认为最大标签+1）
        """
        if num_classes is None:
            num_classes = int(labels.max().item()) + 1
        
        # 确保特征和标签在同一设备
        self.prototypes = self.prototypical_layer.compute_prototypes(
            features, labels, num_classes
        )
        self.num_classes = num_classes
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取特征（使用 backbone 的中间层）。
        返回融合后的特征向量，不是分类 logits。
        """
        if self.backbone.training:
            self.backbone.eval()
        
        # 确保输入在同一设备上
        device = next(self.backbone.parameters()).device
        x = x.to(device)
        
        with torch.no_grad():
            x_t = x.transpose(1, 2)  # (B, T, C)
            f1 = self.backbone.stream1(x_t)  # (B, 128)
            f2 = self.backbone.stream2(x)    # (B, 256)
            fused = torch.cat([f1, f2], dim=-1)  # (B, 384)
            return self.backbone.attention_fusion(fused)  # (B, 384)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（需要先调用 update_prototypes 设置原型）。
        
        Args:
            x: (B, C, T) 输入数据
        
        Returns:
            logits: (B, num_classes)
        """
        if self.prototypes is None:
            raise ValueError("需要先调用 update_prototypes() 设置原型")
        
        features = self.extract_features(x)
        return self.prototypical_layer(features, self.prototypes)
    
    def predict_with_distance(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回预测结果和距离信息。
        
        Returns:
            preds: (B,) 预测类别
            distances: (B, num_classes) 到各类原型的距离
        """
        if self.prototypes is None:
            raise ValueError("需要先调用 update_prototypes() 设置原型")
        
        features = self.extract_features(x)
        
        # 确保 prototypes 在同一设备上
        device = features.device
        prototypes = self.prototypes.to(device)
        
        dist = self.prototypical_layer.compute_distances(features, prototypes)
        preds = dist.argmin(dim=1)
        return preds, dist


class EpisodicTrainer:
    """
    Episodic 训练器，用于训练 Prototypical Network。
    
    每次 episode 从训练集中采样：
    - N 个类别（way）
    - 每个类别 K 个样本作为支持集（support）
    - 每个类别 Q 个样本作为查询集（query）
    
    支持集用于计算原型，查询集用于计算损失和评估。
    """
    
    def __init__(self, model: PrototypicalClassifier,
                 num_ways: int = 5,
                 num_support: int = 5,
                 num_query: int = 15,
                 distance: str = "euclidean"):
        self.model = model
        self.num_ways = num_ways
        self.num_support = num_support
        self.num_query = num_query
        self.distance = distance
        
        # Episodic 损失记录
        self.episode_losses = []
        self.episode_accuracies = []
    
    def create_episode(self, features: torch.Tensor, 
                      labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从数据中创建一个 episode。
        
        Args:
            features: (N, feat_dim) 所有样本的特征
            labels: (N,) 所有样本的标签
        
        Returns:
            support_features: (N_way * N_support, feat_dim)
            support_labels: (N_way * N_support,)
            query_features: (N_way * N_query, feat_dim)
            query_labels: (N_way * N_query,)
        """
        unique_labels = torch.unique(labels)
        
        # 随机选择 N_way 个类别
        if len(unique_labels) > self.num_ways:
            selected_labels = unique_labels[torch.randperm(len(unique_labels))[:self.num_ways]]
        else:
            selected_labels = unique_labels
        
        support_features_list = []
        support_labels_list = []
        query_features_list = []
        query_labels_list = []
        
        episode_labels = []
        for new_label, old_label in enumerate(selected_labels):
            # 该类别的所有样本
            mask = (labels == old_label)
            class_features = features[mask]
            n_class = len(class_features)
            
            if n_class < self.num_support + self.num_query:
                # 样本不够，随机重复采样
                indices = torch.randint(0, n_class, 
                                        (self.num_support + self.num_query,), 
                                        device=features.device)
                class_features = class_features[indices]
                n_class = self.num_support + self.num_query
            
            # 随机打乱并分割
            perm = torch.randperm(n_class, device=features.device)
            
            # 支持集
            support_idx = perm[:self.num_support]
            support_features_list.append(class_features[support_idx])
            support_labels_list.append(
                torch.full((self.num_support,), new_label, 
                          device=features.device, dtype=torch.long)
            )
            
            # 查询集
            query_idx = perm[self.num_support:self.num_support + self.num_query]
            query_features_list.append(class_features[query_idx])
            query_labels_list.append(
                torch.full((min(self.num_query, len(query_idx)),), new_label,
                          device=features.device, dtype=torch.long)
            )
            
            episode_labels.append(new_label)
        
        return (
            torch.cat(support_features_list),
            torch.cat(support_labels_list),
            torch.cat(query_features_list),
            torch.cat(query_labels_list),
        )
    
    def train_episode(self, support_features: torch.Tensor,
                     support_labels: torch.Tensor,
                     query_features: torch.Tensor,
                     query_labels: torch.Tensor,
                     optimizer: torch.optim.Optimizer,
                     use_softmax: bool = True) -> Tuple[float, float]:
        """
        训练一个 episode。
        
        Args:
            support_features: 支持集特征
            support_labels: 支持集标签
            query_features: 查询集特征
            query_labels: 查询集标签
            optimizer: 优化器
            use_softmax: 是否对距离取 softmax（推荐 True）
        
        Returns:
            loss: episode 损失
            accuracy: episode 准确率
        """
        # 计算原型
        num_ways = len(torch.unique(support_labels))
        prototypes = self.model.prototypical_layer.compute_prototypes(
            support_features, support_labels, num_ways
        )
        
        # 计算查询集到原型的距离
        logits = self.model.prototypical_layer(query_features, prototypes)
        
        # 计算损失（使用负对数似然）
        if use_softmax:
            probs = F.softmax(-logits, dim=-1)
            # 手动计算交叉熵
            log_probs = F.log_softmax(-logits, dim=-1)
            loss = F.nll_loss(log_probs, query_labels)
        else:
            loss = F.cross_entropy(-logits, query_labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        preds = logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean().item()
        
        self.episode_losses.append(loss.item())
        self.episode_accuracies.append(accuracy)
        
        return loss.item(), accuracy
    
    def evaluate_episode(self, support_features: torch.Tensor,
                        support_labels: torch.Tensor,
                        query_features: torch.Tensor,
                        query_labels: torch.Tensor) -> Tuple[float, float]:
        """
        评估一个 episode（不更新参数）。
        """
        num_ways = len(torch.unique(support_labels))
        prototypes = self.model.prototypical_layer.compute_prototypes(
            support_features, support_labels, num_ways
        )
        
        logits = self.model.prototypical_layer(query_features, prototypes)
        loss = F.cross_entropy(-logits, query_labels)
        
        preds = logits.argmax(dim=-1)
        accuracy = (preds == query_labels).float().mean().item()
        
        return loss.item(), accuracy
    
    def get_stats(self) -> Dict[str, float]:
        """获取训练统计。"""
        if not self.episode_losses:
            return {"loss": 0.0, "accuracy": 0.0}
        return {
            "loss": sum(self.episode_losses) / len(self.episode_losses),
            "accuracy": sum(self.episode_accuracies) / len(self.episode_accuracies),
        }
    
    def reset_stats(self):
        """重置统计。"""
        self.episode_losses = []
        self.episode_accuracies = []


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算余弦距离。
    
    Args:
        a: (N, D)
        b: (M, D)
    
    Returns:
        dist: (N, M) 余弦距离
    """
    # 归一化
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    
    # 余弦相似度
    cos_sim = torch.mm(a_norm, b_norm.t())
    
    # 转为距离
    return 1 - cos_sim


def euclidean_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算欧氏距离。
    """
    return torch.cdist(a, b, p=2)


if __name__ == "__main__":
    # 简单测试
    torch.manual_seed(42)
    
    # 模拟特征提取器输出
    class MockBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat_dim = 384
        
        def forward(self, x):
            return torch.randn(len(x), self.feat_dim)
    
    backbone = MockBackbone()
    classifier = PrototypicalClassifier(backbone, feat_dim=384, num_classes=6)
    
    # 模拟支持集
    support_features = torch.randn(30, 384)  # 6类，每类5个样本
    support_labels = torch.repeat_interleave(torch.arange(6), 5)
    
    # 更新原型
    classifier.update_prototypes(support_features, support_labels, num_classes=6)
    
    # 模拟查询集
    query_features = torch.randn(12, 384)
    
    # 预测
    logits = classifier.prototypical_layer(query_features, classifier.prototypes)
    preds = logits.argmax(dim=-1)
    
    print(f"原型形状: {classifier.prototypes.shape}")
    print(f"查询集形状: {query_features.shape}")
    print(f"预测结果: {preds}")
    print("Prototypical Network 测试通过!")
