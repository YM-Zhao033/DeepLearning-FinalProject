"""
Dunqi Dataset Loader for Few-Shot Learning.
合并三份实测蹲起数据，按2.56秒(128采样点@50Hz)分片。
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Optional
import glob


# UCI HAR 格式参数
SAMPLING_RATE = 50  # Hz
SEQ_LEN = 128      # 2.56s * 50Hz = 128 samples
NUM_CHANNELS = 9   # body_acc(3) + body_gyro(3) + total_acc(3)


def load_phyphox_csv(acc_path: str, gyro_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 phyphox 导出的 CSV 文件。

    Returns:
        acc: (N, 4) time, x, y, z
        gyro: (N, 4) time, x, y, z
    """
    acc = np.loadtxt(acc_path, delimiter=',', skiprows=1)
    gyro = np.loadtxt(gyro_path, delimiter=',', skiprows=1)
    return acc, gyro


def estimate_gravity(acc_data: np.ndarray, n_samples: int = 100) -> np.ndarray:
    """
    估计重力向量（使用静止时段的数据）。
    取前 n_samples 个样本的均值作为重力估计。
    """
    return acc_data[:n_samples, 1:].mean(axis=0)


def align_signals(acc: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    对齐加速度和陀螺仪数据到共同的时间网格。
    使用线性插值。
    """
    from scipy.interpolate import interp1d

    # 共同时间范围
    t_start = max(acc[0, 0], gyro[0, 0])
    t_end = min(acc[-1, 0], gyro[-1, 0])

    # 创建统一的时间网格
    t_common = np.linspace(t_start, t_end, min(len(acc), len(gyro)))

    # 加速度插值
    acc_interp_x = interp1d(acc[:, 0], acc[:, 1], kind='linear', fill_value='extrapolate')
    acc_interp_y = interp1d(acc[:, 0], acc[:, 2], kind='linear', fill_value='extrapolate')
    acc_interp_z = interp1d(acc[:, 0], acc[:, 3], kind='linear', fill_value='extrapolate')

    acc_aligned = np.stack([
        t_common,
        acc_interp_x(t_common),
        acc_interp_y(t_common),
        acc_interp_z(t_common)
    ], axis=1)

    # 陀螺仪插值
    gyro_interp_x = interp1d(gyro[:, 0], gyro[:, 1], kind='linear', fill_value='extrapolate')
    gyro_interp_y = interp1d(gyro[:, 0], gyro[:, 2], kind='linear', fill_value='extrapolate')
    gyro_interp_z = interp1d(gyro[:, 0], gyro[:, 3], kind='linear', fill_value='extrapolate')

    gyro_aligned = np.stack([
        t_common,
        gyro_interp_x(t_common),
        gyro_interp_y(t_common),
        gyro_interp_z(t_common)
    ], axis=1)

    return acc_aligned, gyro_aligned


def process_session(acc: np.ndarray, gyro: np.ndarray, 
                   stride: int = 64, 
                   seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    处理单次采集数据，生成固定长度的窗口样本。

    Args:
        acc: (N, 4) time, x, y, z
        gyro: (N, 4) time, x, y, z
        stride: 滑动窗口步长
        seq_len: 序列长度（默认128，对应2.56秒）

    Returns:
        windows: (N_windows, 9, seq_len) 9通道信号
    """
    # 对齐数据
    acc_aligned, gyro_aligned = align_signals(acc, gyro)

    # 去除重力（body acceleration）
    gravity = estimate_gravity(acc_aligned)
    body_acc = acc_aligned[:, 1:] - gravity

    # 构建 9 通道信号
    # 通道顺序: body_acc(x,y,z), body_gyro(x,y,z), total_acc(x,y,z)
    signal = np.stack([
        body_acc[:, 0], body_acc[:, 1], body_acc[:, 2],  # body acc
        gyro_aligned[:, 1], gyro_aligned[:, 2], gyro_aligned[:, 3],  # gyro
        acc_aligned[:, 1], acc_aligned[:, 2], acc_aligned[:, 3],  # total acc
    ], axis=1)  # (N, 9)

    # 滑窗分片
    windows = []
    for start in range(0, len(signal) - seq_len + 1, stride):
        window = signal[start:start + seq_len]  # (seq_len, 9)
        windows.append(window.T)  # (9, seq_len)

    return np.array(windows)  # (N_windows, 9, seq_len)


def load_dunqi_sessions(data_root: str) -> np.ndarray:
    """
    加载所有 dunqi 采集会话的数据。

    Args:
        data_root: 数据根目录

    Returns:
        all_windows: (N_total, 9, 128) 所有会话的窗口数据
        session_ids: (N_total,) 每个窗口对应的会话ID
    """
    data_root = Path(data_root)

    all_windows = []
    session_ids = []

    # 查找所有 dunqi* 目录
    session_dirs = sorted(glob.glob(str(data_root / "dunqi*")))

    print(f"找到 {len(session_dirs)} 个 dunqi 会话目录:")
    for i, session_dir in enumerate(session_dirs):
        session_path = Path(session_dir)
        acc_path = session_path / "Accelerometer.csv"
        gyro_path = session_path / "Gyroscope.csv"

        if not acc_path.exists() or not gyro_path.exists():
            print(f"  跳过 {session_dir}（缺少文件）")
            continue

        print(f"  处理: {session_path.name}")
        acc, gyro = load_phyphox_csv(str(acc_path), str(gyro_path))
        windows = process_session(acc, gyro)

        all_windows.append(windows)
        session_ids.extend([i] * len(windows))

        print(f"    - 原始数据点: {len(acc)}")
        print(f"    - 分片窗口数: {len(windows)}")

    if not all_windows:
        raise ValueError("未找到有效的 dunqi 数据")

    return np.concatenate(all_windows, axis=0), np.array(session_ids)


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """
    标准化窗口数据（按通道归一化）。
    """
    # 转为 torch tensor
    windows_tensor = torch.from_numpy(windows).float()

    # 按通道计算均值和标准差
    mean = windows_tensor.mean(dim=(0, 2), keepdim=True)  # (1, 9, 1)
    std = windows_tensor.std(dim=(0, 2), keepdim=True)     # (1, 9, 1)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)

    # 归一化
    normalized = (windows_tensor - mean) / std

    return normalized.numpy()


class DunqiDataset(Dataset):
    """Dunqi 小样本学习数据集。"""

    def __init__(self, data_root: str, normalize: bool = True,
                 train_ratio: float = 0.8, random_seed: int = 42):
        """
        Args:
            data_root: 数据根目录（包含 dunqi, dunqi2, dunqi3 等）
            normalize: 是否归一化
            train_ratio: 训练集比例
            random_seed: 随机种子
        """
        self.data_root = Path(data_root)

        # 加载所有数据
        windows, session_ids = load_dunqi_sessions(str(self.data_root))
        self.windows = windows  # (N, 9, 128)
        self.session_ids = session_ids
        self.n_samples = len(windows)

        print(f"\n总计: {self.n_samples} 个样本")

        # 按会话划分训练/测试集（保持会话独立性）
        np.random.seed(random_seed)
        unique_sessions = np.unique(session_ids)
        np.random.shuffle(unique_sessions)

        n_train_sessions = max(1, int(len(unique_sessions) * train_ratio))
        train_sessions = unique_sessions[:n_train_sessions]
        test_sessions = unique_sessions[n_train_sessions:]

        train_mask = np.isin(session_ids, train_sessions)
        test_mask = np.isin(session_ids, test_sessions)

        self.train_indices = np.where(train_mask)[0]
        self.test_indices = np.where(test_mask)[0]

        print(f"训练会话: {list(train_sessions)} ({len(self.train_indices)} 样本)")
        print(f"测试会话: {list(test_sessions)} ({len(self.test_indices)} 样本)")

        # 归一化
        if normalize:
            self.windows = normalize_windows(self.windows)

        # 标签：全部为0（蹲起是单一类别）
        self.labels = np.zeros(self.n_samples, dtype=np.int64)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = torch.from_numpy(self.windows[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def get_train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取训练集（窗口, 标签）"""
        return self.windows[self.train_indices], self.labels[self.train_indices]

    def get_test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取测试集（窗口, 标签）"""
        return self.windows[self.test_indices], self.labels[self.test_indices]


def create_fewshot_split(windows: np.ndarray, labels: np.ndarray,
                        n_shot: int, n_query: int = 20,
                        n_classes: int = 1,
                        random_seed: int = 42) -> Tuple:
    """
    创建小样本学习的 support/query 分割。

    Args:
        windows: (N, 9, 128)
        labels: (N,)
        n_shot: 每类支持样本数
        n_query: 每类查询样本数
        n_classes: 类别数（默认1，只有蹲起）
        random_seed: 随机种子

    Returns:
        support_X, support_y, query_X, query_y
    """
    np.random.seed(random_seed)

    support_list = []
    support_labels = []
    query_list = []
    query_labels = []

    for c in range(n_classes):
        # 该类别的所有样本
        mask = (labels == c)
        class_windows = windows[mask]

        # 随机打乱
        indices = np.random.permutation(len(class_windows))
        class_windows = class_windows[indices]

        # 分割
        support = class_windows[:n_shot]
        query = class_windows[n_shot:n_shot + n_query]

        support_list.append(support)
        support_labels.extend([c] * len(support))
        query_list.append(query)
        query_labels.extend([c] * len(query))

    support_X = np.concatenate(support_list, axis=0)
    support_y = np.array(support_labels)
    query_X = np.concatenate(query_list, axis=0)
    query_y = np.array(query_labels)

    return support_X, support_y, query_X, query_y


if __name__ == "__main__":
    # 测试数据加载
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"
    dataset = DunqiDataset(str(DATA_ROOT), normalize=True)

    print(f"\n数据集统计:")
    print(f"  总样本数: {len(dataset)}")
    print(f"  训练样本: {len(dataset.train_indices)}")
    print(f"  测试样本: {len(dataset.test_indices)}")
    print(f"  窗口形状: {dataset.windows.shape}")

    # 测试 few-shot split
    train_X, train_y = dataset.get_train_set()
    support_X, support_y, query_X, query_y = create_fewshot_split(
        train_X, train_y, n_shot=5, n_query=20
    )

    print(f"\n小样本分割 (5-shot):")
    print(f"  Support: {support_X.shape}")
    print(f"  Query: {query_X.shape}")
