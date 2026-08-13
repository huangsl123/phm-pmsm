# -*- coding: utf-8 -*-
"""
域适应模块
包含MK-MMD损失、域判别器、梯度反转层
参考DALIB实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional


class GaussianKernel(nn.Module):
    """高斯核函数"""

    def __init__(self, sigma: float = 1.0):
        super(GaussianKernel, self).__init__()
        self.sigma = sigma

    def forward(self, x, y):
        """
        计算高斯核矩阵

        Parameters:
        -----------
        x : Tensor
            (batch_size, feature_dim)
        y : Tensor
            (batch_size, feature_dim)

        Returns:
        --------
        kernel : Tensor
            (batch_size, batch_size)
        """
        # ||x - y||^2
        dist = torch.cdist(x, y, p=2) ** 2
        # exp(-||x - y||^2 / (2 * sigma^2))
        kernel = torch.exp(-dist / (2 * self.sigma ** 2))
        return kernel


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """
    多核最大均值差异 (MK-MMD)
    用于对齐源域和目标域的特征分布
    """

    def __init__(self, kernels: List[GaussianKernel], linear: bool = False):
        super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
        self.kernels = nn.ModuleList(kernels)
        self.linear = linear

        # 索引矩阵
        self.index_matrix = None

    def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
        """
        计算MK-MMD损失

        Parameters:
        -----------
        z_s : Tensor
            源域特征 (batch_size, feature_dim)
        z_t : Tensor
            目标域特征 (batch_size, feature_dim)

        Returns:
        --------
        loss : Tensor
            MK-MMD损失值
        """
        features = torch.cat([z_s, z_t], dim=0)
        batch_size = int(z_s.size(0) + z_t.size(0))

        # 创建索引矩阵
        if self.index_matrix is None or self.index_matrix.size(0) != batch_size:
            self.index_matrix = _build_index_matrix(batch_size).to(z_s.device)

        # 计算多核MMD
        # 每个核计算完整的核矩阵
        kernel_matrix = sum([kernel(features, features) for kernel in self.kernels])  # (N, N)

        # 处理 NaN 值
        if kernel_matrix.isnan().any():
            kernel_matrix = kernel_matrix[~torch.isnan(kernel_matrix)].view(
                kernel_matrix.size(0), -1
            )

        loss = (kernel_matrix * self.index_matrix).view(-1).sum()
        return loss


def _build_index_matrix(batch_size: int) -> torch.Tensor:
    """
    构建MMD索引矩阵

    Parameters:
    -----------
    batch_size : int
        总批次大小（源域+目标域）

    Returns:
    --------
    index_matrix : Tensor
        索引矩阵 (batch_size, batch_size)
    """
    index_matrix = torch.zeros(batch_size, batch_size)

    for i in range(batch_size):
        for j in range(batch_size):
            if (i < batch_size // 2 and j < batch_size // 2) or \
               (i >= batch_size // 2 and j >= batch_size // 2):
                index_matrix[i, j] = 1.0 / (batch_size // 2) ** 2
            else:
                index_matrix[i, j] = -1.0 / (batch_size // 2) ** 2

    return index_matrix


class GradientReverseFunction(torch.autograd.Function):
    """
    梯度反转函数
    前向传播：输出 = 输入
    反向传播：梯度 = -lambda * 输入梯度
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        lambda_val = ctx.lambda_val
        return -lambda_val * grad_output, None


class GradientReverseLayer(nn.Module):
    """
    梯度反转层模块
    支持动态调整lambda值
    """

    def __init__(self, lambda_val: float = 1.0):
        super(GradientReverseLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReverseFunction.apply(x, self.lambda_val)

    def set_lambda(self, lambda_val: float):
        """动态调整lambda值"""
        self.lambda_val = lambda_val


class DomainDiscriminator(nn.Module):
    """
    域判别器
    用于对抗训练，判断特征来自源域还是目标域
    """

    def __init__(self, in_dim: int, hidden_dim: int = 1024):
        super(DomainDiscriminator, self).__init__()

        self.discriinator = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Parameters:
        -----------
        x : Tensor
            特征 (batch_size, in_dim)

        Returns:
        --------
        output : Tensor
            域标签概率 (batch_size, 1)
        """
        return self.discriinator(x)

    def get_parameters(self) -> List[nn.Parameter]:
        """获取可学习参数"""
        return [
            {"params": self.discriinator.parameters(), "lr_mult": 1}
        ]


class DomainAdaptationModule(nn.Module):
    """
    完整的域适应模块
    包含MK-MMD损失和对抗域适应
    """

    def __init__(self, feature_dim: int, sigmas: List[float] = None,
                 lambda_mmd: float = 0.1, lambda_adv: float = 0.1):
        super(DomainAdaptationModule, self).__init__()

        self.lambda_mmd = lambda_mmd
        self.lambda_adv = lambda_adv

        # 多核高斯核
        if sigmas is None:
            sigmas = [2, 5, 10, 20, 40, 80]
        kernels = [GaussianKernel(sigma=s) for s in sigmas]
        self.mk_mmd = MultipleKernelMaximumMeanDiscrepancy(kernels)

        # 梯度反转层
        self.grl = GradientReverseLayer(lambda_val=lambda_adv)

        # 域判别器
        self.domain_discriminator = DomainDiscriminator(
            in_dim=feature_dim,
            hidden_dim=1024
        )

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor,
                progress: float = 0.0) -> dict:
        """
        计算域适应损失

        Parameters:
        -----------
        source_features : Tensor
            源域特征 (batch_size, feature_dim)
        target_features : Tensor
            目标域特征 (batch_size, feature_dim)
        progress : float
            训练进度 [0, 1]，用于渐进式调整lambda

        Returns:
        --------
        losses : dict
            包含各项损失的字典
        """
        batch_size = min(source_features.size(0), target_features.size(0))
        source_features = source_features[:batch_size]
        target_features = target_features[:batch_size]

        losses = {}

        # ==================
        # 1. MK-MMD损失
        # ==================
        loss_mmd = self.mk_mmd(source_features, target_features)
        losses['mmd'] = loss_mmd

        # ==================
        # 2. 对抗域适应损失
        # ==================
        # 渐进式调整lambda
        lambda_adv_current = self.lambda_adv * (2. / (1. + math.exp(-10 * progress)) - 1)
        self.grl.set_lambda(lambda_adv_current)

        # 域标签：源域=0，目标域=1
        domain_labels_source = torch.zeros(batch_size, 1).to(source_features.device)
        domain_labels_target = torch.ones(batch_size, 1).to(target_features.device)

        # 梯度反转
        source_features_grl = self.grl(source_features)
        target_features_grl = self.grl(target_features)

        # 域判别
        domain_output_source = self.domain_discriminator(source_features_grl)
        domain_output_target = self.domain_discriminator(target_features_grl)

        # 对抗损失（迷惑域判别器）
        loss_adv_source = F.binary_cross_entropy(
            domain_output_source, domain_labels_target
        )
        loss_adv_target = F.binary_cross_entropy(
            domain_output_target, domain_labels_source
        )
        loss_adv = (loss_adv_source + loss_adv_target) / 2
        losses['adversarial'] = loss_adv

        # 总损失
        losses['total'] = self.lambda_mmd * loss_mmd + loss_adv

        return losses

    def get_discriminator_acc(self, source_features: torch.Tensor,
                              target_features: torch.Tensor) -> tuple:
        """
        计算域判别器准确率

        Returns:
        --------
        acc : float
            域判别准确率
        """
        self.eval()
        with torch.no_grad():
            batch_size = min(source_features.size(0), target_features.size(0))
            source_features = source_features[:batch_size]
            target_features = target_features[:batch_size]

            # 域判别
            domain_output_source = self.domain_discriminator(source_features)
            domain_output_target = self.domain_discriminator(target_features)

            # 计算准确率
            correct_source = (domain_output_source < 0.5).sum().item()
            correct_target = (domain_output_target >= 0.5).sum().item()
            acc = (correct_source + correct_target) / (2 * batch_size)

        return acc


class CoralLoss(nn.Module):
    """
    CORAL (Correlation Alignment) 损失
    对齐源域和目标域的二阶统计特性
    """

    def __init__(self):
        super(CoralLoss, self).__init__()

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        计算CORAL损失

        Parameters:
        -----------
        source_features : Tensor
            源域特征 (batch_size, feature_dim)
        target_features : Tensor
            目标域特征 (batch_size, feature_dim)

        Returns:
        --------
        loss : Tensor
            CORAL损失值
        """
        batch_size = min(source_features.size(0), target_features.size(0))
        source_features = source_features[:batch_size]
        target_features = target_features[:batch_size]

        feature_dim = source_features.size(1)

        # 计算协方差矩阵
        source_cov = self._compute_covariance(source_features)
        target_cov = self._compute_covariance(target_features)

        # Frobenius范数
        loss = torch.norm(source_cov - target_cov, p='fro') ** 2
        loss = loss / (4 * feature_dim * feature_dim)

        return loss

    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """计算协方差矩阵"""
        batch_size, feature_dim = features.size()
        features = features - features.mean(dim=0, keepdim=True)
        cov = (features.t() @ features) / (batch_size - 1)
        return cov


def get_default_domain_adaptation_module(feature_dim: int) -> DomainAdaptationModule:
    """
    获取默认的域适应模块

    Parameters:
    -----------
    feature_dim : int
        特征维度

    Returns:
    --------
    module : DomainAdaptationModule
        域适应模块
    """
    return DomainAdaptationModule(
        feature_dim=feature_dim,
        sigmas=[2, 5, 10, 20, 40, 80],
        lambda_mmd=0.1,
        lambda_adv=0.1
    )


if __name__ == '__main__':
    # 测试域适应模块
    print('Testing Domain Adaptation Module...')

    batch_size = 32
    feature_dim = 256

    # 创建测试数据
    source_features = torch.randn(batch_size, feature_dim)
    target_features = torch.randn(batch_size, feature_dim) + 0.5

    # 创建域适应模块
    da_module = get_default_domain_adaptation_module(feature_dim)

    # 前向传播
    losses = da_module(source_features, target_features, progress=0.5)

    print(f'Losses:')
    for key, value in losses.items():
        print(f'  {key}: {value.item():.4f}')

    # 测试域判别器准确率
    disc_acc = da_module.get_discriminator_acc(source_features, target_features)
    print(f'Domain Discriminator Accuracy: {disc_acc:.4f}')
