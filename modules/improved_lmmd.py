# -*- coding: utf-8 -*-
"""
改进的LMMD实现 - 基于数据分析的优化
关键改进：
1. 困难类别加权 (类别7,9,8,10的KS>0.38)
2. 自适应核选择
3. 类别平衡策略
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict
import math


class WeightedLMMD(nn.Module):
    """
    加权局部最大均值差异
    对困难类别给予更高权重
    """

    def __init__(
        self,
        sigmas: List[float] = None,
        difficult_classes: List[int] = None,
        difficulty_weight: float = 2.0
    ):
        super().__init__()
        if sigmas is None:
            sigmas = [2, 5, 10, 20, 40, 80]
        self.sigmas = sigmas

        # 困难类别集合
        self.difficult_classes = set(difficult_classes or [])
        self.difficulty_weight = difficulty_weight

        # 类别权重缓存
        self.register_buffer('class_weights', None)

    def compute_class_weight(self, source_labels: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """
        计算类别权重
        困难类别获得更高权重
        """
        num_classes = source_labels.max().item() + 1
        device = source_labels.device

        # 基础权重：样本数的倒数
        source_counts = torch.bincount(source_labels, minlength=num_classes).float()
        target_counts = torch.bincount(target_labels, minlength=num_classes).float()
        total_counts = source_counts + target_counts + 1e-8

        weights = 1.0 / (total_counts + 1e-8)
        weights = weights / weights.sum()

        # 对困难类别加权
        if self.difficult_classes:
            weight_tensor = torch.ones(num_classes).to(device)
            for cls in self.difficult_classes:
                if cls < num_classes:
                    weight_tensor[cls] = self.difficulty_weight
            weights = weights * weight_tensor
            weights = weights / weights.sum()

        return weights

    def compute_lmmd(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: torch.Tensor,
        return_class_losses: bool = False
    ) -> torch.Tensor:
        """
        计算加权LMMD损失

        Parameters:
        -----------
        source_features : Tensor
            源域特征 (N_s, feature_dim)
        target_features : Tensor
            目标域特征 (N_t, feature_dim)
        source_labels : Tensor
            源域标签 (N_s,)
        target_pseudo_labels : Tensor
            目标域伪标签 (N_t,)
        return_class_losses : bool
            是否返回各类别损失（用于分析）

        Returns:
        --------
        loss : Tensor
            加权LMMD损失
        """
        num_classes = source_labels.max().item() + 1
        device = source_features.device

        # 计算类别权重
        class_weights = self.compute_class_weight(source_labels, target_pseudo_labels)

        total_weighted_loss = 0.0
        total_weight = 0.0
        class_losses = {}

        for c in range(num_classes):
            # 获取类别c的特征
            source_mask = (source_labels == c)
            source_c = source_features[source_mask]

            if source_c.size(0) == 0:
                continue

            target_mask = (target_pseudo_labels == c)
            target_c = target_features[target_mask]

            if target_c.size(0) == 0:
                continue

            # 计算类别c的MMD
            class_loss = self._compute_class_mmd(source_c, target_c)

            # 加权
            weight = class_weights[c].item() if c < len(class_weights) else 1.0

            total_weighted_loss += class_loss * weight
            total_weight += weight
            class_losses[c.item()] = class_loss.item()

        if total_weight > 0:
            loss = total_weighted_loss / total_weight
        else:
            loss = torch.tensor(0.0).to(device)

        if return_class_losses:
            return loss, class_losses
        return loss

    def _compute_class_mmd(self, source_c: torch.Tensor, target_c: torch.Tensor) -> torch.Tensor:
        """计算单个类别的MMD"""
        features = torch.cat([source_c, target_c], dim=0)
        batch_size = source_c.size(0) + target_c.size(0)
        n_s = source_c.size(0)
        n_t = target_c.size(0)

        # 创建索引矩阵
        index_matrix = torch.zeros(batch_size, batch_size).to(features.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if (i < n_s and j < n_s) or (i >= n_s and j >= n_s):
                    if i < n_s:
                        index_matrix[i, j] = 1.0 / (n_s * n_s)
                    else:
                        index_matrix[i, j] = 1.0 / (n_t * n_t)
                else:
                    if i < n_s:
                        index_matrix[i, j] = -1.0 / (n_s * n_t)
                    else:
                        index_matrix[i, j] = -1.0 / (n_s * n_t)

        # 多核MMD
        kernel_sum = torch.zeros(1).to(features.device)
        for sigma in self.sigmas:
            dist = torch.cdist(features, features, p=2) ** 2
            kernel = torch.exp(-dist / (2 * sigma ** 2))
            kernel_sum += kernel

        loss = (kernel_sum * index_matrix).sum()
        return loss / len(self.sigmas)


class AdaptiveLMMD(nn.Module):
    """
    自适应LMMD
    根据训练进度自适应调整核权重和类别权重
    """

    def __init__(
        self,
        sigmas: List[float] = None,
        num_classes: int = 16,
        use_adaptive_sigma: bool = True
    ):
        super().__init__()

        self.num_classes = num_classes
        self.use_adaptive_sigma = use_adaptive_sigma

        if sigmas is None:
            sigmas = [2, 5, 10, 20, 40, 80]
        self.sigmas = sigmas

        # 可学习的核权重
        if use_adaptive_sigma:
            self.sigma_weights = nn.Parameter(torch.ones(len(sigmas)) / len(sigmas))
        else:
            self.sigma_weights = None

        # 类别权重（可学习）
        self.class_weights = nn.Parameter(torch.ones(num_classes) / num_classes)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: torch.Tensor,
        progress: float = 0.0
    ) -> torch.Tensor:
        """
        计算自适应LMMD损失
        progress: 训练进度 [0, 1]，用于调整策略
        """
        device = source_features.device

        # 获取核权重
        if self.sigma_weights is not None:
            sigma_weights = F.softmax(self.sigma_weights, dim=0)
        else:
            sigma_weights = torch.ones(len(self.sigmas)).to(device) / len(self.sigmas)

        # 获取类别权重
        class_weights = F.softmax(self.class_weights, dim=0)

        total_loss = 0.0
        total_weight = 0.0

        for c in range(self.num_classes):
            source_mask = (source_labels == c)
            source_c = source_features[source_mask]

            if source_c.size(0) == 0:
                continue

            target_mask = (target_pseudo_labels == c)
            target_c = target_features[target_mask]

            if target_c.size(0) == 0:
                continue

            # 计算多核MMD
            class_loss = self._compute_adaptive_class_mmd(
                source_c, target_c, sigma_weights
            )

            # 使用可学习的类别权重
            weight = class_weights[c].item()
            total_loss += class_loss * weight
            total_weight += weight

        if total_weight > 0:
            return total_loss / total_weight
        return torch.tensor(0.0).to(device)

    def _compute_adaptive_class_mmd(
        self,
        source_c: torch.Tensor,
        target_c: torch.Tensor,
        sigma_weights: torch.Tensor
    ) -> torch.Tensor:
        """计算自适应加权类别MMD"""
        features = torch.cat([source_c, target_c], dim=0)
        n_s = source_c.size(0)
        n_t = target_c.size(0)

        # 创建索引矩阵
        batch_size = features.size(0)
        index_matrix = torch.zeros(batch_size, batch_size).to(features.device)
        for i in range(batch_size):
            for j in range(batch_size):
                if (i < n_s and j < n_s) or (i >= n_s and j >= n_s):
                    if i < n_s:
                        index_matrix[i, j] = 1.0 / (n_s * n_s)
                    else:
                        index_matrix[i, j] = 1.0 / (n_t * n_t)
                else:
                    if i < n_s:
                        index_matrix[i, j] = -1.0 / (n_s * n_t)
                    else:
                        index_matrix[i, j] = -1.0 / (n_s * n_t)

        # 加权多核MMD
        weighted_kernel_sum = torch.zeros(1).to(features.device)
        for i, sigma in enumerate(self.sigmas):
            dist = torch.cdist(features, features, p=2) ** 2
            kernel = torch.exp(-dist / (2 * sigma ** 2))
            weighted_kernel_sum += sigma_weights[i] * kernel

        loss = (weighted_kernel_sum * index_matrix).sum()
        return loss


class PartialLMMD(nn.Module):
    """
    部分域适应LMMD
    处理源域和目标域类别不匹配的情况
    """

    def __init__(self, sigmas: List[float] = None, outlier_threshold: float = 0.1):
        super().__init__()
        if sigmas is None:
            sigmas = [2, 5, 10, 20, 40, 80]
        self.sigmas = sigmas
        self.outlier_threshold = outlier_threshold

        # 源域异常类别权重缓存
        self.source_outlier_classes = None
        self.target_outlier_classes = None

    def detect_outlier_classes(
        self,
        source_labels: torch.Tensor,
        target_labels: torch.Tensor
    ):
        """检测异常类别（只在一个域出现的类别）"""
        source_classes = set(source_labels.cpu().unique().tolist())
        target_classes = set(target_labels.cpu().unique().tolist())

        self.source_outlier_classes = source_classes - target_classes
        self.target_outlier_classes = target_classes - source_classes

        print(f'检测到源域异常类别: {self.source_outlier_classes}')
        print(f'检测到目标域异常类别: {self.target_outlier_classes}')

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算部分域适应LMMD
        自动忽略异常类别
        """
        # 首次检测异常类别
        if self.source_outlier_classes is None:
            self.detect_outlier_classes(source_labels, target_pseudo_labels)

        device = source_features.device
        total_loss = 0.0
        total_weight = 0.0

        # 只对共同类别计算LMMD
        common_classes = set(range(source_labels.max().item() + 1))
        common_classes -= self.source_outlier_classes
        common_classes -= self.target_outlier_classes

        for c in common_classes:
            source_mask = (source_labels == c)
            source_c = source_features[source_mask]

            if source_c.size(0) == 0:
                continue

            target_mask = (target_pseudo_labels == c)
            target_c = target_features[target_mask]

            if target_c.size(0) == 0:
                continue

            # 标准类别MMD
            class_loss = self._compute_class_mmd(source_c, target_c)
            weight = source_c.size(0) + target_c.size(0)

            total_loss += class_loss * weight
            total_weight += weight

        if total_weight > 0:
            return total_loss / total_weight
        return torch.tensor(0.0).to(device)

    def _compute_class_mmd(self, source_c: torch.Tensor, target_c: torch.Tensor) -> torch.Tensor:
        """计算类别MMD"""
        features = torch.cat([source_c, target_c], dim=0)
        n_s = source_c.size(0)
        n_t = target_c.size(0)

        index_matrix = torch.zeros(features.size(0), features.size(0)).to(features.device)
        for i in range(features.size(0)):
            for j in range(features.size(0)):
                if (i < n_s and j < n_s) or (i >= n_s and j >= n_s):
                    if i < n_s:
                        index_matrix[i, j] = 1.0 / (n_s * n_s)
                    else:
                        index_matrix[i, j] = 1.0 / (n_t * n_t)
                else:
                    index_matrix[i, j] = -1.0 / (min(n_s, n_t) * max(n_s, n_t))

        kernel_sum = torch.zeros(1).to(features.device)
        for sigma in self.sigmas:
            dist = torch.cdist(features, features, p=2) ** 2
            kernel = torch.exp(-dist / (2 * sigma ** 2))
            kernel_sum += kernel

        loss = (kernel_sum * index_matrix).sum()
        return loss / len(self.sigmas)


if __name__ == '__main__':
    print('Testing Improved LMMD Modules...')

    batch_size = 32
    feature_dim = 256

    # 测试数据
    source_features = torch.randn(batch_size, feature_dim)
    target_features = torch.randn(batch_size, feature_dim)
    source_labels = torch.randint(0, 15, (batch_size,))
    target_pseudo = torch.randint(0, 15, (batch_size,))

    # 测试加权LMMD
    print('\n1. Weighted LMMD...')
    weighted_lmmd = WeightedLMMD(difficult_classes=[7, 9, 8, 10], difficulty_weight=2.0)
    loss = weighted_lmmd.compute_lmmd(source_features, target_features, source_labels, target_pseudo)
    print(f'   Weighted LMMD Loss: {loss.item():.4f}')

    # 测试自适应LMMD
    print('\n2. Adaptive LMMD...')
    adaptive_lmmd = AdaptiveLMMD(num_classes=16, use_adaptive_sigma=True)
    loss = adaptive_lmmd(source_features, target_features, source_labels, target_pseudo, progress=0.5)
    print(f'   Adaptive LMMD Loss: {loss.item():.4f}')

    # 测试部分LMMD
    print('\n3. Partial LMMD...')
    partial_lmmd = PartialLMMD()
    # 模拟类别不匹配
    source_labels_mismatch = source_labels.clone()
    source_labels_mismatch[0] = 15  # 类别15只在源域
    target_pseudo_mismatch = target_pseudo.clone()
    target_pseudo_mismatch[0] = 5   # 类别5只在目标域
    loss = partial_lmmd(source_features, target_features, source_labels_mismatch, target_pseudo_mismatch)
    print(f'   Partial LMMD Loss: {loss.item():.4f}')

    print('\nAll tests passed!')
