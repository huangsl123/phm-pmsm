# -*- coding: utf-8 -*-
"""
多模态数据处理器 (stride=128版本)
处理时域信号和频域谱图
默认使用 stride=128 产生更多样本
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy import signal
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_spectrogram(x: np.ndarray, n_fft: int = 256, hop_length: int = 64,
                       win_length: Optional[int] = None) -> np.ndarray:
    """
    计算短时傅里叶变换谱图

    Parameters:
    -----------
    x : np.ndarray
        输入信号 (seq_len, n_channels)
    n_fft : int
        FFT窗口大小
    hop_length : int
        跳跃长度
    win_length : int, optional
        窗口长度（默认=n_fft）

    Returns:
    --------
    spec : np.ndarray
        谱图 (n_channels, freq_bins, time_frames)
    """
    if win_length is None:
        win_length = n_fft

    seq_len, n_channels = x.shape
    specs = []

    for i in range(n_channels):
        # 计算STFT
        f, t, Zxx = signal.stft(
            x[:, i],
            nperseg=win_length,
            noverlap=win_length - hop_length,
            nfft=n_fft
        )

        # 幅度谱
        spec = np.abs(Zxx)
        specs.append(spec)

    return np.stack(specs, axis=0)  # (n_channels, freq_bins, time_frames)


def normalize_signal(x: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    归一化信号

    Parameters:
    -----------
    x : np.ndarray
        输入信号
    method : str
        归一化方法 ('minmax', 'zscore', 'robust')

    Returns:
    --------
    x_norm : np.ndarray
        归一化后的信号
    """
    if method == 'minmax':
        # Min-Max归一化到[-1, 1]
        x_min = np.min(x, axis=-1, keepdims=True)
        x_max = np.max(x, axis=-1, keepdims=True)
        x_norm = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1

    elif method == 'zscore':
        # Z-score标准化
        x_mean = np.mean(x, axis=-1, keepdims=True)
        x_std = np.std(x, axis=-1, keepdims=True)
        x_norm = (x - x_mean) / (x_std + 1e-8)

    elif method == 'robust':
        # 鲁棒归一化（使用中位数和四分位距）
        x_median = np.median(x, axis=-1, keepdims=True)
        x_q75 = np.percentile(x, 75, axis=-1, keepdims=True)
        x_q25 = np.percentile(x, 25, axis=-1, keepdims=True)
        x_norm = (x - x_median) / (x_q75 - x_q25 + 1e-8)

    else:
        raise ValueError(f'Unknown normalization method: {method}')

    return x_norm


def resize_spectrogram(spec: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    调整谱图大小

    Parameters:
    -----------
    spec : np.ndarray
        输入谱图 (n_channels, H, W)
    target_size : Tuple[int, int]
        目标大小 (target_height, target_width)

    Returns:
    --------
    spec_resized : np.ndarray
        调整后的谱图 (n_channels, target_height, target_width)
    """
    from skimage.transform import resize

    n_channels, H, W = spec.shape
    target_h, target_w = target_size

    spec_resized = np.zeros((n_channels, target_h, target_w))

    for i in range(n_channels):
        spec_resized[i] = resize(spec[i], (target_h, target_w), mode='reflect', anti_aliasing=True)

    return spec_resized


def augment_signal(x: np.ndarray, noise_level: float = 0.01,
                   scale_range: Tuple[float, float] = (0.95, 1.05)) -> np.ndarray:
    """
    信号数据增强

    Parameters:
    -----------
    x : np.ndarray
        输入信号
    noise_level : float
        噪声水平
    scale_range : Tuple[float, float]
        幅值缩放范围

    Returns:
    --------
    x_aug : np.ndarray
        增强后的信号
    """
    x_aug = x.copy()

    # 添加高斯噪声
    noise = np.random.randn(*x_aug.shape) * noise_level
    x_aug = x_aug + noise

    # 幅值缩放
    scale = np.random.uniform(scale_range[0], scale_range[1])
    x_aug = x_aug * scale

    return x_aug


class MultiModalFaultDataset(Dataset):
    """
    多模态故障诊断数据集
    同时提供时域信号和频域谱图
    """

    def __init__(self, time_signals: np.ndarray, spectrograms: np.ndarray,
                 labels: np.ndarray, augment: bool = False):
        """
        Parameters:
        -----------
        time_signals : np.ndarray
            时域信号 (n_samples, seq_len, n_channels)
        spectrograms : np.ndarray
            频域谱图 (n_samples, n_channels, H, W)
        labels : np.ndarray
            标签 (n_samples,)
        augment : bool
            是否进行数据增强
        """
        self.time_signals = torch.FloatTensor(time_signals)
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        time_sig = self.time_signals[idx]
        spec = self.spectrograms[idx]
        label = self.labels[idx]

        if self.augment:
            # 时域信号增强
            time_sig = self._augment_time(time_sig)

            # 频域谱图增强
            spec = self._augment_spec(spec)

        return time_sig, spec, label

    def _augment_time(self, time_sig):
        """时域信号数据增强"""
        # 添加高斯噪声
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(time_sig) * 0.02
            time_sig = time_sig + noise

        # 幅值缩放
        if torch.rand(1) < 0.5:
            scale = 0.8 + 0.4 * torch.rand(1).item()
            time_sig = time_sig * scale

        # 时序偏移
        if torch.rand(1) < 0.3:
            shift = torch.randint(-50, 50, (1,)).item()
            time_sig = torch.roll(time_sig, shifts=shift, dims=0)

        # 裁剪到合理范围
        time_sig = torch.clamp(time_sig, -1.0, 1.0)
        return time_sig

    def _augment_spec(self, spec):
        """频域谱图数据增强"""
        # 频率遮蔽
        if torch.rand(1) < 0.3:
            f = torch.randint(0, spec.shape[1], (1,)).item()
            f_end = min(f + torch.randint(5, 20, (1,)).item(), spec.shape[1])
            spec[:, f:f_end, :] = 0

        # 时间遮蔽
        if torch.rand(1) < 0.3:
            t = torch.randint(0, spec.shape[2], (1,)).item()
            t_end = min(t + torch.randint(5, 20, (1,)).item(), spec.shape[2])
            spec[:, :, t:t_end] = 0

        return spec


class UnlabeledMultiModalDataset(Dataset):
    """无标签多模态数据集（用于域适应）"""

    def __init__(self, time_signals: np.ndarray, spectrograms: np.ndarray):
        self.time_signals = torch.FloatTensor(time_signals)
        self.spectrograms = torch.FloatTensor(spectrograms)

    def __len__(self):
        return len(self.time_signals)

    def __getitem__(self, idx):
        return self.time_signals[idx], self.spectrograms[idx]


def load_csv_data(data_path: str, window_size: int = 1024, stride: int = 128,
                   n_fft: int = 256, hop_length: int = 64,
                   spec_size: Tuple[int, int] = (128, 128),
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42,
                   split_mode: str = 'legacy') -> Dict:
    """
    从CSV文件加载数据并创建多模态数据集 (stride=128版本)

    Parameters:
    -----------
    data_path : str
        CSV文件路径
    window_size : int
        滑动窗口大小
    stride : int
        滑动步长 (默认128，产生更多样本)
    n_fft : int
        FFT大小
    hop_length : int
        STFT跳跃长度
    spec_size : Tuple[int, int]
        谱图目标大小
    test_size : float
        测试集比例
    val_size : float
        验证集比例
    random_state : int
        随机种子
    split_mode : str
        分割模式: 'legacy' (默认) 或 '721' (7:2:1分割)
        - 'legacy': 训练集=1-test_size-val_size, 验证集=val_size, 测试集=test_size
        - '721': 训练集=70%, 测试集=20%, 验证集=10%

    Returns:
    --------
    data : Dict
        数据字典
    """
    # 读取CSV
    df = pd.read_csv(data_path)

    # 获取故障类别
    fault_codes = sorted(df['fault_code'].unique())
    n_classes = len(fault_codes)
    fault_to_label = {code: i for i, code in enumerate(fault_codes)}

    # 选择信号通道
    channel_cols = ['channel1', 'channel2', 'channel3']
    n_channels = len(channel_cols)

    # 对每个故障类别创建滑动窗口
    all_time_signals = []
    all_spectrograms = []
    all_labels = []

    print(f'Processing {data_path}...')

    for fault_code in fault_codes:
        fault_df = df[df['fault_code'] == fault_code]
        signals = fault_df[channel_cols].values

        # 检查并清理NaN/Inf
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        # 创建滑动窗口
        for i in range(0, len(signals) - window_size + 1, stride):
            window = signals[i:i + window_size]

            if len(window) == window_size:
                # 检查窗口是否有效
                if np.isnan(window).any() or np.isinf(window).any():
                    continue

                # 检查窗口是否有变化（避免常数信号）
                if np.std(window) < 1e-6:
                    continue

                # 归一化
                window_norm = normalize_signal(window, method='minmax')

                # 再次检查归一化后的数据
                if np.isnan(window_norm).any() or np.isinf(window_norm).any():
                    continue

                # 计算谱图
                spec = compute_spectrogram(window_norm, n_fft=n_fft, hop_length=hop_length)

                # 检查谱图
                if np.isnan(spec).any() or np.isinf(spec).any():
                    continue

                spec_resized = resize_spectrogram(spec, spec_size)

                # 检查调整后的谱图
                if np.isnan(spec_resized).any() or np.isinf(spec_resized).any():
                    continue

                # 谱图归一化
                spec_min = spec_resized.min(axis=(1, 2), keepdims=True)
                spec_max = spec_resized.max(axis=(1, 2), keepdims=True)
                spec_norm = (spec_resized - spec_min) / (spec_max - spec_min + 1e-8)

                # 最终检查
                if np.isnan(spec_norm).any() or np.isinf(spec_norm).any():
                    continue

                all_time_signals.append(window_norm)
                all_spectrograms.append(spec_norm)
                all_labels.append(fault_to_label[fault_code])

    # 转换为numpy数组
    time_signals = np.array(all_time_signals)
    spectrograms = np.array(all_spectrograms)
    labels = np.array(all_labels)

    print(f'  Loaded {len(time_signals)} samples')
    print(f'  Time signals shape: {time_signals.shape}')
    print(f'  Spectrograms shape: {spectrograms.shape}')

    # 划分数据集
    if split_mode == '721':
        # 7:2:1 分割 - 训练:测试:验证 = 70%:20%:10%
        # 先划分测试集 (20%)
        X_temp, X_test_idx, y_temp, y_test = train_test_split(
            np.arange(len(time_signals)), labels,
            test_size=0.20, random_state=random_state, stratify=labels
        )
        # 再从剩余中划分验证集 (10% / 80% = 12.5%)
        X_train_idx, X_val_idx, _, _ = train_test_split(
            X_temp, y_temp,
            test_size=0.125, random_state=random_state, stratify=y_temp
        )
    else:
        # legacy 模式 - 使用原始的 test_size 和 val_size
        X_temp, X_test_idx, y_temp, y_test = train_test_split(
            np.arange(len(time_signals)), labels,
            test_size=test_size, random_state=random_state, stratify=labels
        )

        val_ratio = val_size / (1 - test_size)
        X_train_idx, X_val_idx, _, _ = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio, random_state=random_state, stratify=y_temp
        )

    # 创建数据字典
    data = {
        'X_train_time': time_signals[X_train_idx],
        'X_train_spec': spectrograms[X_train_idx],
        'y_train': labels[X_train_idx],
        'X_val_time': time_signals[X_val_idx],
        'X_val_spec': spectrograms[X_val_idx],
        'y_val': labels[X_val_idx],
        'X_test_time': time_signals[X_test_idx],
        'X_test_spec': spectrograms[X_test_idx],
        'y_test': y_test,
        'n_classes': n_classes,
        'n_channels': n_channels,
        'window_size': window_size,
        'spec_size': spec_size,
        'fault_codes': fault_codes
    }

    return data


def load_cross_domain_data(source_power: str, target_power: str, base_path: str,
                            **kwargs) -> Dict:
    """
    加载跨域数据

    Parameters:
    -----------
    source_power : str
        源域功率
    target_power : str
        目标域功率
    base_path : str
        数据基础路径
    **kwargs : 其他参数传递给load_csv_data

    Returns:
    --------
    cross_data : Dict
        跨域数据字典
    """
    source_data_path = os.path.join(base_path, f'dataset2_{source_power}.csv')
    target_data_path = os.path.join(base_path, f'dataset2_{target_power}.csv')

    source_data = load_csv_data(source_data_path, **kwargs)
    target_data = load_csv_data(target_data_path, **kwargs)

    return {
        'source': source_data,
        'target': target_data
    }


def create_domain_adapt_loaders(cross_data: Dict, batch_size: int = 32,
                                 num_workers: int = 0,
                                 augment: bool = True) -> Dict:
    """
    创建域适应数据加载器

    Parameters:
    -----------
    cross_data : Dict
        跨域数据字典
    batch_size : int
        批次大小
    num_workers : int
        工作进程数
    augment : bool
        是否数据增强

    Returns:
    --------
    loaders : Dict
        数据加载器字典
    """
    source_data = cross_data['source']
    target_data = cross_data['target']

    # 源域：有标签
    source_train_dataset = MultiModalFaultDataset(
        source_data['X_train_time'],
        source_data['X_train_spec'],
        source_data['y_train'],
        augment=augment
    )
    source_val_dataset = MultiModalFaultDataset(
        source_data['X_val_time'],
        source_data['X_val_spec'],
        source_data['y_val'],
        augment=False
    )
    source_test_dataset = MultiModalFaultDataset(
        source_data['X_test_time'],
        source_data['X_test_spec'],
        source_data['y_test'],
        augment=False
    )

    # 目标域：训练时无标签
    target_train_dataset = UnlabeledMultiModalDataset(
        target_data['X_train_time'],
        target_data['X_train_spec']
    )
    target_test_dataset = MultiModalFaultDataset(
        target_data['X_test_time'],
        target_data['X_test_spec'],
        target_data['y_test'],
        augment=False
    )

    loaders = {
        'source_train': DataLoader(source_train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers),
        'source_val': DataLoader(source_val_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers),
        'source_test': DataLoader(source_test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers),
        'target_train': DataLoader(target_train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers),
        'target_test': DataLoader(target_test_dataset, batch_size=batch_size,
                                  shuffle=False, num_workers=num_workers)
    }

    return loaders


if __name__ == '__main__':
    # 测试数据处理器
    print('Testing Data Processor...')

    base_path = r'D:\liyanfu\EI paper\数据集2\data'

    # 加载单域数据
    data = load_csv_data(
        os.path.join(base_path, 'dataset2_1.0kW.csv'),
        window_size=1024,
        stride=512,
        spec_size=(128, 128)
    )

    print(f'\nData loaded:')
    print(f'  Train samples: {len(data["y_train"])}')
    print(f'  Val samples: {len(data["y_val"])}')
    print(f'  Test samples: {len(data["y_test"])}')
    print(f'  Time signals shape: {data["X_train_time"].shape}')
    print(f'  Spectrograms shape: {data["X_train_spec"].shape}')

    # 创建数据集
    dataset = MultiModalFaultDataset(
        data['X_train_time'],
        data['X_train_spec'],
        data['y_train']
    )

    time_sig, spec, label = dataset[0]
    print(f'\nSample:')
    print(f'  Time signal shape: {time_sig.shape}')
    print(f'  Spectrogram shape: {spec.shape}')
    print(f'  Label: {label}')
