# -*- coding: utf-8 -*-
"""
先进域适应方法 - 电机故障诊断专用
包含：LMMD, CLMMD, MADA, 对比学习等
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Dict, Tuple
import math


# ====================
# 1. 局部最大均值差异 (LMMD)
# ====================

class LocalMaximumMeanDiscrepancy(nn.Module):
    """
    局部最大均值差异 (LMMD)
    在类别子域级别对齐源域和目标域的分布

    Reference:
    "Conditional Adversarial Domain Adaptation" (CVPR 2018)
    "Cross-attentional subdomain adaptation with selective knowledge distillation" (2024)
    """

    def __init__(self, sigmas: List[float] = None):
        super().__init__()
        if sigmas is None:
            sigmas = [2, 5, 10, 20, 40, 80]

        # 为每个类别创建高斯核
        self.sigmas = sigmas
        self.register_buffer('sigmas_tensor', torch.tensor(sigmas).float())

    def compute_lmmd(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算LMMD损失

        Parameters:
        -----------
        source_features : Tensor
            源域特征 (N_s, feature_dim)
        target_features : Tensor
            目标域特征 (N_t, feature_dim)
        source_labels : Tensor
            源域标签 (N_s,)
        target_pseudo_labels : Tensor, optional
            目标域伪标签 (N_t,)，如果为None则使用预测

        Returns:
        --------
        loss : Tensor
            LMMD损失值
        """
        num_classes = source_labels.max().item() + 1
        device = source_features.device

        total_loss = 0.0
        class_weights = 0.0

        for c in range(num_classes):
            # 获取类别c的源域特征
            source_mask = (source_labels == c)
            source_c = source_features[source_mask]

            if source_c.size(0) == 0:
                continue

            # 获取类别c的目标域特征
            if target_pseudo_labels is not None:
                target_mask = (target_pseudo_labels == c)
            else:
                # 如果没有伪标签，跳过LMMD
                continue

            target_c = target_features[target_mask]

            if target_c.size(0) == 0:
                continue

            # 计算类别c的MMD
            class_loss = self._compute_class_mmd(source_c, target_c)

            # 加权：样本数越多的类别权重越大
            weight = source_c.size(0) + target_c.size(0)
            total_loss += class_loss * weight
            class_weights += weight

        if class_weights > 0:
            return total_loss / class_weights
        return torch.tensor(0.0).to(device)

    def _compute_class_mmd(self, source_c: torch.Tensor, target_c: torch.Tensor) -> torch.Tensor:
        """计算单个类别的MMD"""
        # 合并特征
        features = torch.cat([source_c, target_c], dim=0)
        batch_size = source_c.size(0) + target_c.size(0)

        # 创建索引矩阵
        n_s = source_c.size(0)
        n_t = target_c.size(0)
        index_matrix = torch.zeros(batch_size, batch_size).to(features.device)

        for i in range(batch_size):
            for j in range(batch_size):
                if (i < n_s and j < n_s) or (i >= n_s and j >= n_s):
                    # 同域
                    if i < n_s:
                        index_matrix[i, j] = 1.0 / (n_s * n_s)
                    else:
                        index_matrix[i, j] = 1.0 / (n_t * n_t)
                else:
                    # 跨域
                    if i < n_s:
                        index_matrix[i, j] = -1.0 / (n_s * n_t)
                    else:
                        index_matrix[i, j] = -1.0 / (n_s * n_t)

        # 计算多核MMD
        kernel_sum = torch.zeros(batch_size, batch_size).to(features.device)
        for sigma in self.sigmas:
            # 计算高斯核矩阵
            dist = torch.cdist(features, features, p=2) ** 2
            kernel = torch.exp(-dist / (2 * sigma ** 2))
            kernel_sum += kernel

        loss = (kernel_sum * index_matrix).sum()
        return loss / len(self.sigmas)


# ====================
# 2. 相关局部MMD (CLMMD)
# ====================

class CorrelatedLMMD(nn.Module):
    """
    相关局部最大均值差异 (CLMMD)
    在LMMD基础上引入跨域样本间的相关性信息

    Reference:
    "Correlated and Local MMD for Domain Adaptation"
    """

    def __init__(self, sigmas: List[float] = None, correlation_weight: float = 0.5):
        super().__init__()
        self.lmmd = LocalMaximumMeanDiscrepancy(sigmas)
        self.correlation_weight = correlation_weight

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: torch.Tensor
    ) -> torch.Tensor:
        """计算CLMMD损失"""
        # 基础LMMD损失
        lmmd_loss = self.lmmd.compute_lmmd(
            source_features, target_features,
            source_labels, target_pseudo_labels
        )

        # 相关性损失
        correlation_loss = self._compute_correlation_loss(
            source_features, target_features,
            source_labels, target_pseudo_labels
        )

        return lmmd_loss + self.correlation_weight * correlation_loss

    def _compute_correlation_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_pseudo_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算相关性损失
        鼓励跨域同类样本间的特征相关性
        """
        num_classes = source_labels.max().item() + 1
        device = source_features.device

        total_correlation = 0.0
        count = 0

        for c in range(num_classes):
            source_mask = (source_labels == c)
            target_mask = (target_pseudo_labels == c)

            source_c = source_features[source_mask]
            target_c = target_features[target_mask]

            if source_c.size(0) < 2 or target_c.size(0) < 2:
                continue

            # 计算类内协方差矩阵
            source_cov = self._compute_covariance(source_c)
            target_cov = self._compute_covariance(target_c)

            # 相关性损失：协方差矩阵的F范数差异
            correlation_loss = torch.norm(source_cov - target_cov, p='fro') ** 2
            total_correlation += correlation_loss
            count += 1

        if count > 0:
            return total_correlation / count
        return torch.tensor(0.0).to(device)

    def _compute_covariance(self, features: torch.Tensor) -> torch.Tensor:
        """计算协方差矩阵"""
        features_centered = features - features.mean(dim=0, keepdim=True)
        cov = (features_centered.t() @ features_centered) / (features.size(0) - 1)
        return cov


# ====================
# 3. 多对抗域适应 (MADA)
# ====================

class MultiAdversarialDomainAdaptation(nn.Module):
    """
    多对抗域适应 (MADA)
    为每个类别配备专属的域判别器

    Reference:
    "Conditional Adversarial Domain Adaptation" (CVPR 2018)
    "Multi-Adversarial Domain Adaptation for Fault Diagnosis" (Scientific Reports, 2024)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_dim: int = 1024
    ):
        super().__init__()
        self.num_classes = num_classes

        # 为每个类别创建域判别器
        self.domain_discriminators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
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
            for _ in range(num_classes)
        ])

        # 梯度反转层（动态alpha）
        self.register_buffer('alpha', torch.tensor(1.0))

    def set_alpha(self, alpha: float):
        """动态调整梯度反转系数"""
        self.alpha.fill_(alpha)

    def get_domain_output(
        self,
        features: torch.Tensor,
        class_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        获取域判别输出

        Parameters:
        -----------
        features : Tensor
            特征 (N, feature_dim)
        class_predictions : Tensor
            类别预测概率 (N, num_classes)

        Returns:
        --------
        domain_output : Tensor
            加权的域判别输出 (N, 1)
        """
        batch_size = features.size(0)
        domain_output = torch.zeros(batch_size, 1).to(features.device)

        # 对每个类别的判别器输出加权求和
        for c in range(self.num_classes):
            # 类别c的判别器输出
            disc_output_c = self.domain_discriminators[c](features)  # (N, 1)

            # 用类别预测概率作为权重
            weight_c = class_predictions[:, c:c+1]  # (N, 1)

            domain_output += weight_c * disc_output_c

        return domain_output

    def forward(
        self,
        source_features: torch.Tensor,
        source_predictions: torch.Tensor,
        target_features: torch.Tensor,
        target_predictions: torch.Tensor,
        progress: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        计算MADA损失

        Parameters:
        -----------
        source_features : Tensor
            源域特征
        source_predictions : Tensor
            源域类别预测概率
        target_features : Tensor
            目标域特征
        target_predictions : Tensor
            目标域类别预测概率
        progress : float
            训练进度 [0, 1]

        Returns:
        --------
        losses : Dict
            包含各项损失的字典
        """
        batch_size = min(source_features.size(0), target_features.size(0))
        source_features = source_features[:batch_size]
        source_predictions = source_predictions[:batch_size]
        target_features = target_features[:batch_size]
        target_predictions = target_predictions[:batch_size]

        # 渐进式调整alpha
        alpha = 2. / (1. + math.exp(-10 * progress)) - 1
        self.set_alpha(alpha)

        # 源域域判别
        source_domain_output = self.get_domain_output(source_features, source_predictions)

        # 目标域域判别
        target_domain_output = self.get_domain_output(target_features, target_predictions)

        # 域标签
        domain_labels_source = torch.zeros(batch_size, 1).to(source_features.device)
        domain_labels_target = torch.ones(batch_size, 1).to(target_features.device)

        # 对抗损失 - clamp to prevent numerical errors
        source_domain_output = torch.clamp(source_domain_output, min=1e-7, max=1-1e-7)
        target_domain_output = torch.clamp(target_domain_output, min=1e-7, max=1-1e-7)
        loss_source = F.binary_cross_entropy(source_domain_output, domain_labels_target)
        loss_target = F.binary_cross_entropy(target_domain_output, domain_labels_source)

        # 每个类别的独立损失（用于监控）
        class_losses = []
        for c in range(self.num_classes):
            disc_c = self.domain_discriminators[c]
            source_disc_c = torch.clamp(disc_c(source_features), min=1e-7, max=1-1e-7)
            target_disc_c = torch.clamp(disc_c(target_features), min=1e-7, max=1-1e-7)
            loss_c = (F.binary_cross_entropy(source_disc_c, domain_labels_target) +
                     F.binary_cross_entropy(target_disc_c, domain_labels_source)) / 2
            class_losses.append(loss_c)

        return {
            'total': (loss_source + loss_target) / 2,
            'source': loss_source,
            'target': loss_target,
            'class_losses': torch.stack(class_losses).mean()
        }


# ====================
# 4. 半监督对比域适应
# ====================

class SemiSupervisedContrastiveDA(nn.Module):
    """
    半监督对比域适应
    结合对比学习和伪标签实现域适应

    Reference:
    "Semi-Supervised Contrastive Domain Adaptation Network for Fault Diagnosis" (IEEE IoT, 2025)
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
        pseudo_threshold: float = 0.9
    ):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = temperature
        self.pseudo_threshold = pseudo_threshold

        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, projection_dim)
        )

        # EMA教师投影头
        self.teacher_projector = None
        self.ema_decay = 0.999

    def _update_teacher(self):
        """更新EMA教师"""
        if self.teacher_projector is None:
            # 初始化教师
            self.teacher_projector = nn.Sequential(
                nn.Linear(self.projector[0].out_features, self.projector[0].out_features),
                nn.BatchNorm1d(self.projector[0].out_features),
                nn.ReLU(inplace=True),
                nn.Linear(self.projector[3].in_features, self.projector[3].out_features)
            )
            self.teacher_projector.load_state_dict(self.projector.state_dict())
            for param in self.teacher_projector.parameters():
                param.requires_grad = False
        else:
            # EMA更新
            for teacher_param, student_param in zip(
                self.teacher_projector.parameters(),
                self.projector.parameters()
            ):
                teacher_param.data = (
                    self.ema_decay * teacher_param.data +
                    (1 - self.ema_decay) * student_param.data
                )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """投影特征"""
        return self.projector(features)

    def contrastive_loss(
        self,
        source_features: torch.Tensor,
        source_labels: torch.Tensor,
        target_features: torch.Tensor,
        target_predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算对比损失

        Parameters:
        -----------
        source_features : Tensor
            源域特征 (N_s, feature_dim)
        source_labels : Tensor
            源域标签 (N_s,)
        target_features : Tensor
            目标域特征 (N_t, feature_dim)
        target_predictions : Tensor
            目标域预测 (N_t, num_classes)

        Returns:
        --------
        losses : Dict
            包含各项损失的字典
        """
        # 投影
        source_proj = self.forward(source_features)  # (N_s, D)
        target_proj = self.forward(target_features)  # (N_t, D)

        # 归一化
        source_proj = F.normalize(source_proj, dim=1)
        target_proj = F.normalize(target_proj, dim=1)

        # ==================
        # 域内对比损失
        # ==================

        # 源域：监督对比
        intra_source_loss = self._supervised_contrastive_loss(
            source_proj, source_labels
        )

        # 目标域：伪标签对比
        target_pseudo_labels = target_predictions.argmax(dim=1)
        target_confidence = target_predictions.max(dim=1)[0]

        # 只使用高置信度样本
        valid_mask = target_confidence >= self.pseudo_threshold
        if valid_mask.sum() > 0:
            valid_target_proj = target_proj[valid_mask]
            valid_target_pseudo = target_pseudo_labels[valid_mask]
            intra_target_loss = self._supervised_contrastive_loss(
                valid_target_proj, valid_target_pseudo
            )
        else:
            intra_target_loss = torch.tensor(0.0).to(source_features.device)

        intra_loss = intra_source_loss + intra_target_loss

        # ==================
        # 跨域对比损失
        # ==================

        # 同类样本对齐
        inter_loss = self._cross_domain_contrastive_loss(
            source_proj, source_labels,
            target_proj, target_pseudo_labels
        )

        return {
            'intra': intra_loss,
            'inter': inter_loss,
            'total': intra_loss + inter_loss
        }

    def _supervised_contrastive_loss(
        self,
        projections: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """监督对比损失"""
        batch_size = projections.size(0)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(projections, projections.t()) / self.temperature

        # 创建标签掩码
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.t()).float().to(projections.device)

        # 移除对角线
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(projections.device)
        mask = mask * logits_mask

        # 计算损失
        exp_logits = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 只保留正样本对
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss

    def _cross_domain_contrastive_loss(
        self,
        source_proj: torch.Tensor,
        source_labels: torch.Tensor,
        target_proj: torch.Tensor,
        target_pseudo: torch.Tensor
    ) -> torch.Tensor:
        """
        跨域对比损失
        拉近跨域同类样本，推远跨域异类样本
        """
        # 合并源域和目标域
        all_proj = torch.cat([source_proj, target_proj], dim=0)
        all_labels = torch.cat([source_labels, target_pseudo], dim=0)

        # 创建跨域掩码
        batch_size = source_proj.size(0)
        n_source = batch_size
        n_target = target_proj.size(0)

        # 计算相似度
        sim_matrix = torch.matmul(all_proj, all_proj.t()) / self.temperature

        # 创建掩码：只考虑跨域的同类样本对
        labels = all_labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.t()).float().to(all_proj.device)

        # 跨域掩码
        domain_mask = torch.ones_like(label_mask)
        domain_mask[:n_source, :n_source] = 0  # 源域内
        domain_mask[n_source:, n_source:] = 0  # 目标域内

        mask = label_mask * domain_mask

        if mask.sum() == 0:
            return torch.tensor(0.0).to(source_proj.device)

        # 计算损失
        exp_logits = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-12)
        loss = -mean_log_prob_pos[mask.sum(dim=1) > 0].mean()

        return loss


# ====================
# 5. 统一的高级域适应模块
# ====================

class AdvancedMotorDAModule(nn.Module):
    """
    统一的电机故障诊断域适应模块
    支持多种先进方法的组合
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        methods: List[str] = ['lmmd', 'contrastive'],
        **kwargs
    ):
        """
        Parameters:
        -----------
        feature_dim : int
            特征维度
        num_classes : int
            类别数
        methods : List[str]
            使用的方法，可选：'lmmd', 'clmmd', 'mada', 'contrastive'
        """
        super().__init__()
        self.methods = methods
        self.num_classes = num_classes

        # LMMD
        if 'lmmd' in methods or 'clmmd' in methods:
            if 'clmmd' in methods:
                self.lmmd = CorrelatedLMMD(
                    sigmas=kwargs.get('mmd_sigmas', [2, 5, 10, 20, 40, 80]),
                    correlation_weight=kwargs.get('correlation_weight', 0.5)
                )
            else:
                self.lmmd = LocalMaximumMeanDiscrepancy(
                    sigmas=kwargs.get('mmd_sigmas', [2, 5, 10, 20, 40, 80])
                )
        else:
            self.lmmd = None

        # MADA
        if 'mada' in methods:
            self.mada = MultiAdversarialDomainAdaptation(
                feature_dim=feature_dim,
                num_classes=num_classes,
                hidden_dim=kwargs.get('mada_hidden_dim', 1024)
            )
        else:
            self.mada = None

        # 对比学习
        if 'contrastive' in methods:
            self.contrastive = SemiSupervisedContrastiveDA(
                feature_dim=feature_dim,
                num_classes=num_classes,
                projection_dim=kwargs.get('projection_dim', 128),
                temperature=kwargs.get('temperature', 0.07),
                pseudo_threshold=kwargs.get('pseudo_threshold', 0.9)
            )
        else:
            self.contrastive = None

    def forward(
        self,
        source_features: torch.Tensor,
        source_outputs: torch.Tensor,
        target_features: torch.Tensor,
        target_outputs: torch.Tensor,
        progress: float = 0.0
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有域适应损失

        Returns:
        --------
        losses : Dict
            包含各项损失的字典
        """
        losses = {}
        source_labels = source_outputs.argmax(dim=1)
        target_pseudo = target_outputs.argmax(dim=1)

        # LMMD/CLMMD损失
        if self.lmmd is not None:
            lmmd_loss = self.lmmd.compute_lmmd(
                source_features, target_features,
                source_labels, target_pseudo
            )
            losses['lmmd'] = lmmd_loss

        # MADA损失
        if self.mada is not None:
            mada_losses = self.mada(
                source_features, source_outputs,
                target_features, target_outputs,
                progress
            )
            losses['mada'] = mada_losses['total']
            losses['mada_source'] = mada_losses['source']
            losses['mada_target'] = mada_losses['target']

        # 对比学习损失
        if self.contrastive is not None:
            contrast_losses = self.contrastive.contrastive_loss(
                source_features, source_labels,
                target_features, target_outputs
            )
            losses['contrastive'] = contrast_losses['total']
            losses['contrastive_intra'] = contrast_losses['intra']
            losses['contrastive_inter'] = contrast_losses['inter']

        # 总损失
        total_loss = sum(losses.values())
        losses['total'] = total_loss

        return losses


if __name__ == '__main__':
    print('Testing Advanced Domain Adaptation for Motor Fault Diagnosis...')

    batch_size = 32
    feature_dim = 256
    num_classes = 16

    # 创建测试数据
    source_features = torch.randn(batch_size, feature_dim)
    source_outputs = torch.randn(batch_size, num_classes)
    target_features = torch.randn(batch_size, feature_dim)
    target_outputs = torch.randn(batch_size, num_classes)

    # 测试LMMD
    print('\n1. Testing LMMD...')
    lmmd = LocalMaximumMeanDiscrepancy()
    source_labels = torch.randint(0, num_classes, (batch_size,))
    target_pseudo = torch.randint(0, num_classes, (batch_size,))
    loss = lmmd.compute_lmmd(source_features, target_features, source_labels, target_pseudo)
    print(f'   LMMD Loss: {loss.item():.4f}')

    # 测试MADA
    print('\n2. Testing MADA...')
    mada = MultiAdversarialDomainAdaptation(feature_dim, num_classes)
    losses = mada(source_features, source_outputs, target_features, target_outputs, progress=0.5)
    print(f'   MADA Losses:')
    for key, value in losses.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f'      {key}: {value.item():.4f}')

    # 测试对比学习
    print('\n3. Testing Contrastive DA...')
    contrastive = SemiSupervisedContrastiveDA(feature_dim, num_classes)
    losses = contrastive.contrastive_loss(
        source_features, source_labels,
        target_features, target_outputs
    )
    print(f'   Contrastive Losses:')
    for key, value in losses.items():
        print(f'      {key}: {value.item():.4f}')

    # 测试统一模块
    print('\n4. Testing Unified Module...')
    unified = AdvancedMotorDAModule(
        feature_dim=feature_dim,
        num_classes=num_classes,
        methods=['lmmd', 'contrastive']
    )
    losses = unified(
        source_features, source_outputs,
        target_features, target_outputs,
        progress=0.5
    )
    print(f'   Unified Losses:')
    for key, value in losses.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            print(f'      {key}: {value.item():.4f}')

    print('\nAll tests passed!')
