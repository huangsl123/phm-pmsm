# -*- coding: utf-8 -*-
"""
增强的数据处理器
包含更多数据增强策略：
1. 时域增强：噪声注入、幅值缩放、时序偏移、时间扭曲、幅值抖动
2. 频域增强：SpecAugment (频率/时间遮蔽)、MixUp、CutMix
3. 跨模态增强：时频混合增强
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from scipy import signal
from scipy.interpolate import interp1d
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAugmentation:
    """时域信号增强"""

    def __init__(
        self,
        noise_level: float = 0.02,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shift_range: int = 50,
        jitter_strength: float = 0.01,
        time_warp_ratio: float = 0.2,
        dropout_prob: float = 0.1,
        dropout_size: int = 10
    ):
        self.noise_level = noise_level
        self.scale_range = scale_range
        self.shift_range = shift_range
        self.jitter_strength = jitter_strength
        self.time_warp_ratio = time_warp_ratio
        self.dropout_prob = dropout_prob
        self.dropout_size = dropout_size

    def add_gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        if self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        return x

    def amplitude_scaling(self, x: torch.Tensor) -> torch.Tensor:
        """幅值缩放"""
        scale = torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]).item()
        return x * scale

    def time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """时序偏移"""
        if self.shift_range > 0:
            shift = torch.randint(-self.shift_range, self.shift_range, (1,)).item()
            return torch.roll(x, shifts=shift, dims=0)
        return x

    def amplitude_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """幅值抖动（逐点微调）"""
        if self.jitter_strength > 0:
            jitter = torch.randn_like(x) * self.jitter_strength
            return x + jitter
        return x

    def time_warping(self, x: np.ndarray) -> np.ndarray:
        """时间扭曲（通过插值实现）"""
        seq_len = x.shape[0]
        n_channels = x.shape[1] if len(x.shape) > 1 else 1

        # 生成扭曲点
        num_warp_points = max(1, int(seq_len * self.time_warp_ratio))
        warp_points = np.sort(np.random.choice(seq_len, num_warp_points, replace=False))

        # 生成扭曲位置
        warp_positions = warp_points + np.random.randint(
            -int(seq_len * 0.1), int(seq_len * 0.1), size=num_warp_points
        )
        warp_positions = np.clip(warp_positions, 0, seq_len - 1)

        # 创建原始索引
        original_indices = np.arange(seq_len)

        # 应用扭曲
        for wp, wpos in zip(warp_points, warp_positions):
            # 在扭曲点附近平滑过渡
            mask = np.abs(original_indices - wp) < (seq_len * 0.05)
            original_indices[mask] = original_indices[mask] + (wpos - wp) * 0.5

        original_indices = np.clip(original_indices, 0, seq_len - 1)

        # 插值
        if n_channels > 1:
            warped = np.zeros_like(x)
            for ch in range(n_channels):
                f = interp1d(np.arange(seq_len), x[:, ch], kind='linear', fill_value='extrapolate')
                warped[:, ch] = f(original_indices)
            return warped
        else:
            f = interp1d(np.arange(seq_len), x, kind='linear', fill_value='extrapolate')
            return f(original_indices)

    def time_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """时序Dropout（随机归零部分时间段）"""
        if torch.rand(1) < self.dropout_prob:
            seq_len = x.shape[0]
            dropout_start = torch.randint(0, seq_len - self.dropout_size, (1,)).item()
            dropout_end = min(dropout_start + self.dropout_size, seq_len)
            x[dropout_start:dropout_end] = 0
        return x

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """应用所有增强（随机选择）"""
        # 随机选择应用的增强方法
        if torch.rand(1) < 0.5:
            x = self.add_gaussian_noise(x)
        if torch.rand(1) < 0.3:
            x = self.amplitude_scaling(x)
        if torch.rand(1) < 0.3:
            x = self.time_shift(x)
        if torch.rand(1) < 0.2:
            x = self.amplitude_jitter(x)
        if torch.rand(1) < 0.1:
            x = self.time_dropout(x)

        return x


class SpectrogramAugmentation:
    """频域谱图增强 (SpecAugment变体)"""

    def __init__(
        self,
        freq_mask_param: int = 16,
        time_mask_param: int = 16,
        freq_mask_num: int = 2,
        time_mask_num: int = 2,
        mixup_alpha: float = 0.2,
        cutout_size: int = 8
    ):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.freq_mask_num = freq_mask_num
        self.time_mask_num = time_mask_num
        self.mixup_alpha = mixup_alpha
        self.cutout_size = cutout_size

    def frequency_masking(self, spec: torch.Tensor) -> torch.Tensor:
        """频率遮蔽"""
        for _ in range(self.freq_mask_num):
            f = torch.randint(0, max(1, spec.shape[1] - self.freq_mask_param), (1,)).item()
            f_width = torch.randint(1, self.freq_mask_param + 1, (1,)).item()
            f_end = min(f + f_width, spec.shape[1])
            spec[:, f:f_end, :] = 0
        return spec

    def time_masking(self, spec: torch.Tensor) -> torch.Tensor:
        """时间遮蔽"""
        for _ in range(self.time_mask_num):
            t = torch.randint(0, max(1, spec.shape[2] - self.time_mask_param), (1,)).item()
            t_width = torch.randint(1, self.time_mask_param + 1, (1,)).item()
            t_end = min(t + t_width, spec.shape[2])
            spec[:, :, t:t_end] = 0
        return spec

    def cutout(self, spec: torch.Tensor) -> torch.Tensor:
        """随机方块遮挡"""
        if torch.rand(1) < 0.3:
            c = spec.shape[0]
            h = spec.shape[1]
            w = spec.shape[2]

            y = torch.randint(0, h - self.cutout_size, (1,)).item()
            x = torch.randint(0, w - self.cutout_size, (1,)).item()

            spec[:, y:y+self.cutout_size, x:x+self.cutout_size] = 0
        return spec

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        """应用谱图增强"""
        if torch.rand(1) < 0.5:
            spec = self.frequency_masking(spec)
        if torch.rand(1) < 0.5:
            spec = self.time_masking(spec)
        if torch.rand(1) < 0.3:
            spec = self.cutout(spec)

        return spec


class CrossModalAugmentation:
    """跨模态增强（时域和频域联合增强）"""

    def __init__(self, mixup_alpha: float = 0.2, mixup_prob: float = 0.3):
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

    def mixup(self, x1: torch.Tensor, x2: torch.Tensor, y1: torch.Tensor, y2: torch.Tensor) -> Tuple:
        """MixUp增强（混合两个样本）"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1 - lam)

            mixed_x1 = lam * x1 + (1 - lam) * x2
            mixed_x2 = lam * x2 + (1 - lam) * x2

            # 对于标签，也进行混合
            mixed_y = lam * y1 + (1 - lam) * y2

            return mixed_x1, mixed_x2, mixed_y, lam
        return x1, x2, y1, 1.0

    def __call__(self, time_x: torch.Tensor, spec_x: torch.Tensor,
                label: torch.Tensor, batch_indices: List[int]) -> Tuple:
        """
        应用跨模态增强

        Parameters:
        -----------
        time_x : Tensor
            时域信号批次
        spec_x : Tensor
            频域谱图批次
        label : Tensor
            标签批次
        batch_indices : List[int]
            当前批次的索引

        Returns:
        --------
        augmented data
        """
        if torch.rand(1) < self.mixup_prob and len(batch_indices) > 1:
            # 随机选择另一个样本进行混合
            idx2 = torch.randint(0, len(batch_indices), (1,)).item()
            if idx2 != batch_indices[0]:
                time_x2 = time_x[idx2:idx2+1]
                spec_x2 = spec_x[idx2:idx2+1]
                label2 = label[idx2:idx2+1]

                time_x, spec_x, label, lam = self.mixup(
                    time_x, time_x2, spec_x, spec_x2, label, label2
                )

        return time_x, spec_x, label


class EnhancedMultiModalDataset(Dataset):
    """
    增强的多模态数据集
    支持多种增强策略
    """

    def __init__(
        self,
        time_signals: np.ndarray,
        spectrograms: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        aug_config: Optional[Dict] = None
    ):
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
        aug_config : Dict, optional
            增强配置
        """
        self.time_signals = torch.FloatTensor(time_signals)
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

        # 初始化增强器
        if aug_config is None:
            aug_config = {
                'time': {
                    'noise_level': 0.02,
                    'scale_range': (0.9, 1.1),
                    'shift_range': 50,
                },
                'spec': {
                    'freq_mask_param': 16,
                    'time_mask_param': 16,
                    'freq_mask_num': 2,
                    'time_mask_num': 2,
                }
            }

        self.time_aug = TimeSeriesAugmentation(**aug_config.get('time', {}))
        self.spec_aug = SpectrogramAugmentation(**aug_config.get('spec', {}))
        self.cross_aug = CrossModalAugmentation(
            mixup_alpha=aug_config.get('mixup_alpha', 0.2),
            mixup_prob=aug_config.get('mixup_prob', 0.3)
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        time_sig = self.time_signals[idx].clone()
        spec = self.spectrograms[idx].clone()
        label = self.labels[idx]

        if self.augment:
            # 时域增强
            time_sig = self.time_aug(time_sig)

            # 频域增强
            spec = self.spec_aug(spec)

            # 裁剪到合理范围
            time_sig = torch.clamp(time_sig, -1.0, 1.0)
            spec = torch.clamp(spec, 0.0, 1.0)

        return time_sig, spec, label


class StrongAugmentationDataset(Dataset):
    """
    强增强数据集
    用于更激进的数据增强策略
    """

    def __init__(
        self,
        time_signals: np.ndarray,
        spectrograms: np.ndarray,
        labels: np.ndarray,
        strong_augment: bool = False
    ):
        self.time_signals = torch.FloatTensor(time_signals)
        self.spectrograms = torch.FloatTensor(spectrograms)
        self.labels = torch.LongTensor(labels)
        self.strong_augment = strong_augment

        # 强增强配置
        self.strong_time_aug = TimeSeriesAugmentation(
            noise_level=0.05,
            scale_range=(0.8, 1.2),
            shift_range=100,
            jitter_strength=0.02,
            time_warp_ratio=0.3,
            dropout_prob=0.2,
            dropout_size=20
        )

        self.strong_spec_aug = SpectrogramAugmentation(
            freq_mask_param=24,
            time_mask_param=24,
            freq_mask_num=3,
            time_mask_num=3,
            cutout_size=16
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        time_sig = self.time_signals[idx].clone()
        spec = self.spectrograms[idx].clone()
        label = self.labels[idx]

        if self.strong_augment:
            # 应用强增强
            time_sig = self.strong_time_aug(time_sig)
            spec = self.strong_spec_aug(spec)

            # 随机通道dropout
            if torch.rand(1) < 0.1:
                ch_idx = torch.randint(0, time_sig.shape[-1], (1,)).item()
                time_sig[:, ch_idx] = 0
                spec[ch_idx, :, :] = 0

            time_sig = torch.clamp(time_sig, -1.0, 1.0)
            spec = torch.clamp(spec, 0.0, 1.0)

        return time_sig, spec, label


class MixupCollator:
    """MixUp数据整理器"""

    def __init__(self, alpha: float = 0.2, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        """
        应用MixUp

        Parameters:
        -----------
        batch : List[Tuple]
            (time_sig, spec, label) 元组列表

        Returns:
        --------
        mixed_batch : Tuple
            混合后的批次
        """
        time_sigs = torch.stack([item[0] for item in batch])
        specs = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])

        if torch.rand(1) < self.prob:
            batch_size = labels.size(0)
            lam = np.random.beta(self.alpha, self.alpha)

            # 随机排列
            index = torch.randperm(batch_size)

            # 混合
            mixed_time = lam * time_sigs + (1 - lam) * time_sigs[index]
            mixed_spec = lam * specs + (1 - lam) * specs[index]

            # 混合标签（用于训练）
            labels_a, labels_b = labels, labels[index]
            mixed_labels = (labels_a, labels_b, lam)

            return mixed_time, mixed_spec, mixed_labels

        return time_sigs, specs, labels


class CutMixCollator:
    """CutMix数据整理器"""

    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        time_sigs = torch.stack([item[0] for item in batch])
        specs = torch.stack([item[1] for item in batch])
        labels = torch.stack([item[2] for item in batch])

        if torch.rand(1) < self.prob:
            batch_size = labels.size(0)
            lam = np.random.beta(self.alpha, self.alpha)

            # 随机排列
            index = torch.randperm(batch_size)

            # 对谱图应用CutMix（随机替换矩形区域）
            specs_a, specs_b = specs, specs[index]
            _, _, H, W = specs.shape

            # 随机选择裁剪区域
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            # 随机位置
            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            # 应用CutMix
            mixed_spec = specs_a.clone()
            mixed_spec[:, :, bby1:bby2, bbx1:bbx2] = specs_b[:, :, bby1:bby2, bbx1:bbx2]

            # 调整lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

            labels_a, labels_b = labels, labels[index]
            mixed_labels = (labels_a, labels_b, lam)

            # 时域信号也应用相同混合比例
            mixed_time = lam * time_sigs + (1 - lam) * time_sigs[index]

            return mixed_time, mixed_spec, mixed_labels

        return time_sigs, specs, labels


# 导出接口
__all__ = [
    'TimeSeriesAugmentation',
    'SpectrogramAugmentation',
    'CrossModalAugmentation',
    'EnhancedMultiModalDataset',
    'StrongAugmentationDataset',
    'MixupCollator',
    'CutMixCollator'
]


if __name__ == '__main__':
    # 测试增强模块
    print('Testing Enhanced Data Augmentation...')

    # 创建测试数据
    n_samples = 100
    seq_len = 1024
    n_channels = 3
    H, W = 128, 128

    time_signals = np.random.randn(n_samples, seq_len, n_channels)
    spectrograms = np.random.rand(n_samples, n_channels, H, W)
    labels = np.random.randint(0, 16, n_samples)

    # 测试标准增强
    print('\n1. Testing Standard Augmentation...')
    dataset = EnhancedMultiModalDataset(
        time_signals, spectrograms, labels,
        augment=True
    )

    time_sig, spec, label = dataset[0]
    print(f'  Time signal shape: {time_sig.shape}')
    print(f'  Spectrogram shape: {spec.shape}')
    print(f'  Label: {label}')

    # 测试强增强
    print('\n2. Testing Strong Augmentation...')
    strong_dataset = StrongAugmentationDataset(
        time_signals, spectrograms, labels,
        strong_augment=True
    )

    time_sig, spec, label = strong_dataset[0]
    print(f'  Time signal shape: {time_sig.shape}')
    print(f'  Spectrogram shape: {spec.shape}')
    print(f'  Label: {label}')

    # 测试MixUp
    print('\n3. Testing MixUp Collator...')
    batch = [dataset[i] for i in range(32)]
    mixup_collator = MixupCollator(alpha=0.2, prob=1.0)
    mixed_time, mixed_spec, mixed_labels = mixup_collator(batch)

    if isinstance(mixed_labels, tuple):
        labels_a, labels_b, lam = mixed_labels
        print(f'  Mixed batch time shape: {mixed_time.shape}')
        print(f'  Mixed batch spec shape: {mixed_spec.shape}')
        print(f'  Lambda: {lam:.4f}')
    else:
        print(f'  Batch time shape: {mixed_time.shape}')
        print(f'  Batch spec shape: {mixed_spec.shape}')

    print('\nAll tests passed!')
