"""
CAPPIMU Dataset Loader
与双分支模型和小样本学习框架兼容的 CAPPIMU 数据加载器
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# CAPPIMU 21类活动标签（整数索引到名称）
CAPPIMU_LABELS = {
    0: "falling",
    1: "brushing_teeth",
    2: "washing_face",
    3: "slicing",
    4: "eating",
    5: "washing_dishes",
    6: "folding_clothes",
    7: "sweeping",
    8: "mopping",
    9: "toileting",
    10: "window_cleaning",
    11: "drinking_water",
    12: "hanging_out_clothes",
    13: "ironing",
    14: "using_computer",
    15: "watching_tv",
    16: "jogging",
    17: "walking",
    18: "cycling",
    19: "writing",
    20: "play_phone",
}

# 居家相关活动（适合扩展到扫地/拖地研究）
# 注意：实际数据中只有部分类别，标签5,9,10,16可能在数据中被过滤
HOME_ACTIVITIES = [7, 8, 13, 11, 6, 12]  # sweeping, mopping, ironing, drinking_water, folding_clothes, hanging_out_clothes
HYGIENE_ACTIVITIES = [1, 2]  # brushing_teeth, washing_face
EXERCISE_ACTIVITIES = [17, 18]  # walking, cycling


class CAPPIMUDataset(Dataset):
    """
    CAPPIMU 数据集类
    输出形状: (n_samples, 9 channels, 128 time steps)
    与 UCIDataset 兼容
    """
    
    def __init__(self, X: np.ndarray, y: np.ndarray, normalize: bool = True):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.n_samples, self.n_channels, self.seq_len = self.X.shape
        
        if normalize:
            self._normalize()
    
    def _normalize(self):
        """Z-score 标准化"""
        mean = self.X.mean(dim=(0, 2), keepdim=True)
        std = self.X.std(dim=(0, 2), keepdim=True)
        std = torch.where(std < 1e-6, torch.ones_like(std), std)
        self.X = (self.X - mean) / std
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_cappimu_split(data_root: str, split: str = 'train') -> tuple:
    """
    加载 CAPPIMU 数据集的指定划分
    
    Args:
        data_root: 预处理后的数据根目录 (e.g., 'data/processed/cappimu_chest')
        split: 'train', 'val', 或 'test'
    
    Returns:
        X: np.ndarray (n_samples, 9, 128)
        y: np.ndarray (n_samples,)
    """
    data_root = Path(data_root)
    
    X = np.load(data_root / f'X_{split}.npy')
    y = np.load(data_root / f'y_{split}.npy')
    
    return X, y


def get_cappimu_dataloaders(data_root: str, batch_size: int = 64, 
                            num_workers: int = 0,
                            selected_classes: list = None) -> tuple:
    """
    创建 CAPPIMU DataLoaders
    
    Args:
        data_root: 预处理后的数据根目录
        batch_size: 批次大小
        num_workers: 数据加载线程数
        selected_classes: 如果指定，只加载这些类别
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # 加载所有数据
    X_train, y_train = load_cappimu_split(data_root, 'train')
    X_val, y_val = load_cappimu_split(data_root, 'val')
    X_test, y_test = load_cappimu_split(data_root, 'test')
    
    # 如果指定了类别，进行过滤
    if selected_classes is not None:
        train_mask = np.isin(y_train, selected_classes)
        val_mask = np.isin(y_val, selected_classes)
        test_mask = np.isin(y_test, selected_classes)
        
        X_train, y_train = X_train[train_mask], y_train[train_mask]
        X_val, y_val = X_val[val_mask], y_val[val_mask]
        X_test, y_test = X_test[test_mask], y_test[test_mask]
        
        # 重新映射标签为连续索引
        label_map = {old: new for new, old in enumerate(selected_classes)}
        y_train = np.array([label_map[l] for l in y_train])
        y_val = np.array([label_map[l] for l in y_val])
        y_test = np.array([label_map[l] for l in y_test])
    
    # 创建数据集
    train_ds = CAPPIMUDataset(X_train, y_train, normalize=True)
    val_ds = CAPPIMUDataset(X_val, y_val, normalize=True)
    test_ds = CAPPIMUDataset(X_test, y_test, normalize=True)
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def get_cappimu_for_fewshot(data_root: str, 
                           new_classes: list = HOME_ACTIVITIES,
                           support_shot: int = 5,
                           random_seed: int = 42) -> dict:
    """
    为小样本学习准备 CAPPIMU 数据
    
    Args:
        data_root: 预处理后的数据根目录
        new_classes: 新类别列表（用于小样本测试）
        support_shot: 每个新类别抽取的支持样本数
        random_seed: 随机种子
    
    Returns:
        dict: {
            'support_X': torch.Tensor,
            'support_y': torch.Tensor,
            'query_X': torch.Tensor,
            'query_y': torch.Tensor,
            'class_names': list of str
        }
    """
    np.random.seed(random_seed)
    
    # 加载所有数据
    X_all = []
    y_all = []
    
    for split in ['train', 'val', 'test']:
        X, y = load_cappimu_split(data_root, split)
        X_all.append(X)
        y_all.append(y)
    
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    
    # 只保留新类别
    mask = np.isin(y_all, new_classes)
    X_new = X_all[mask]
    y_new = y_all[mask]
    
    # 重新映射标签
    label_map = {old: new for new, old in enumerate(new_classes)}
    y_mapped = np.array([label_map[l] for l in y_new])
    
    # 为每个新类别抽取支持样本
    support_X_list = []
    support_y_list = []
    query_X_list = []
    query_y_list = []
    
    for new_label, original_label in enumerate(new_classes):
        class_mask = (y_mapped == new_label)
        class_X = X_new[class_mask]
        class_y = y_mapped[class_mask]
        
        # 随机打乱
        indices = np.random.permutation(len(class_X))
        class_X = class_X[indices]
        class_y = class_y[indices]
        
        # 抽取支持样本
        support_X = class_X[:support_shot]
        support_y = class_y[:support_shot]
        
        # 剩余作为查询样本
        query_X = class_X[support_shot:]
        query_y = class_y[support_shot:]
        
        support_X_list.append(support_X)
        support_y_list.append(support_y)
        query_X_list.append(query_X)
        query_y_list.append(query_y)
    
    # 合并
    support_X = torch.from_numpy(np.concatenate(support_X_list, axis=0)).float()
    support_y = torch.from_numpy(np.concatenate(support_y_list, axis=0)).long()
    query_X = torch.from_numpy(np.concatenate(query_X_list, axis=0)).float()
    query_y = torch.from_numpy(np.concatenate(query_y_list, axis=0)).long()
    
    # 获取类别名称
    class_names = [CAPPIMU_LABELS[orig] for orig in new_classes]
    
    return {
        'support_X': support_X,
        'support_y': support_y,
        'query_X': query_X,
        'query_y': query_y,
        'class_names': class_names,
        'n_classes': len(new_classes)
    }


if __name__ == '__main__':
    # 测试加载器
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / 'data' / 'processed' / 'cappimu_chest'
    
    if DATA_ROOT.exists():
        print("测试 CAPPIMU DataLoaders...")
        train_loader, val_loader, test_loader = get_cappimu_dataloaders(DATA_ROOT)
        
        # 获取一个batch
        X, y = next(iter(train_loader))
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        print(f"Label distribution: {torch.bincount(y)}")
        
        # 测试小样本数据准备
        print("\n测试小样本数据准备...")
        fewshot_data = get_cappimu_for_fewshot(DATA_ROOT, new_classes=HOME_ACTIVITIES, support_shot=5)
        print(f"Support X shape: {fewshot_data['support_X'].shape}")
        print(f"Query X shape: {fewshot_data['query_X'].shape}")
        print(f"Classes: {fewshot_data['class_names']}")
    else:
        print(f"数据不存在: {DATA_ROOT}")
        print("请先运行 preprocess_cappimu.py 预处理数据")
