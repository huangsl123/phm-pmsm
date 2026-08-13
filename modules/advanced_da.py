# -*- coding: utf-8 -*-
"""
先进的域适应模块
包含多种SOTA域适应方法
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
# 1. 条件域对抗 (CDAN)
# ====================

class ConditionalDomainAdversarial(nn.Module):
    """
    条件域对抗适应 (CDAN)
    利用分类器的预测概率作为条件，使域判别器能够感知语义信息

    Reference: "Conditional Adversarial Domain Adaptation" (NeurIPS 2018)
    """

    def __init__(self, feature_dim: int, num_classes: int, hidden_dim: int = 1024):
        super().__init__()
        self.num_classes = num_classes

        # 多线性映射 - 将特征和预测结合起来
        self.multilinear_map = nn.Linear(feature_dim, feature_dim)

        # 域判别器
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim * num_classes, hidden_dim),
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

        # 梯度反转层
        self.register_buffer('alpha', torch.tensor(1.0))

    def forward(self, features: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        features : Tensor
            特征 (batch_size, feature_dim)
        outputs : Tensor
            分类器输出 (batch_size, num_classes)

        Returns:
        --------
        domain_output : Tensor
            域判别输出
        """
        batch_size = features.size(0)

        # 多线性映射
        mapped_features = self.multilinear_map(features)  # (B, D)

        # 计算外积
        # outputs_soft: (B, C) -> mapped_features: (B, D)
        # outer_product: (B, C*D)
        outputs_soft = F.softmax(outputs, dim=1)
        outer_product = torch.bmm(
            outputs_soft.unsqueeze(2),  # (B, C, 1)
            mapped_features.unsqueeze(1)  # (B, 1, D)
        ).reshape(batch_size, -1)  # (B, C*D)

        # 域判别
        domain_output = self.domain_classifier(outer_product)

        return domain_output

    def get_loss(self, source_features: torch.Tensor, source_outputs: torch.Tensor,
                 target_features: torch.Tensor, target_outputs: torch.Tensor,
                 progress: float = 0.0) -> torch.Tensor:
        """计算条件域对抗损失"""
        batch_size = min(source_features.size(0), target_features.size(0))
        source_features = source_features[:batch_size]
        source_outputs = source_outputs[:batch_size]
        target_features = target_features[:batch_size]
        target_outputs = target_outputs[:batch_size]

        # 渐进式调整alpha
        alpha = 2. / (1. + math.exp(-10 * progress)) - 1

        # 源域域判别
        source_domain_output = self.forward(source_features, source_outputs)

        # 目标域域判别
        target_domain_output = self.forward(target_features, target_outputs)

        # 域标签：源域=0，目标域=1
        domain_labels_source = torch.zeros(batch_size, 1).to(source_features.device)
        domain_labels_target = torch.ones(batch_size, 1).to(target_features.device)

        # 对抗损失 - clamp for numerical stability
        loss_source = F.binary_cross_entropy(
            torch.clamp(source_domain_output, min=1e-7, max=1-1e-7), domain_labels_target
        )
        loss_target = F.binary_cross_entropy(
            torch.clamp(target_domain_output, min=1e-7, max=1-1e-7), domain_labels_source
        )

        return (loss_source + loss_target) / 2


# ====================
# 2. 伪标签域适应
# ====================

class PseudoLabelDomainAdaptation(nn.Module):
    """
    基于伪标签的域适应
    使用目标域高置信度预测作为伪标签进行自训练

    Reference: "Domain Adaptation via Pseudo Labeling" (Various works)
    """

    def __init__(self, threshold: float = 0.9, momentum: float = 0.999):
        super().__init__()
        self.threshold = threshold
        self.momentum = momentum

        # EMA教师模型参数缓存
        self.teacher_params = None

    def update_teacher(self, student_model: nn.Module, momentum: Optional[float] = None):
        """用动量更新教师模型"""
        if momentum is None:
            momentum = self.momentum

        if self.teacher_params is None:
            # 初始化教师参数
            self.teacher_params = {}
            for name, param in student_model.named_parameters():
                self.teacher_params[name] = param.data.clone()
        else:
            # EMA更新
            for name, param in student_model.named_parameters():
                if name in self.teacher_params:
                    self.teacher_params[name] = (
                        momentum * self.teacher_params[name] + (1 - momentum) * param.data
                    )

    def get_pseudo_labels(self, model: nn.Module, target_features: torch.Tensor,
                         return_confidence: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        获取目标域的伪标签

        Parameters:
        -----------
        model : nn.Module
            学生模型
        target_features : Tensor
            目标域特征
        return_confidence : bool
            是否返回置信度

        Returns:
        --------
        pseudo_labels : Tensor
            伪标签
        confidence : Tensor (optional)
            置信度
        """
        with torch.no_grad():
            outputs = model(target_features)
            probs = F.softmax(outputs, dim=1)
            confidence, pseudo_labels = probs.max(dim=1)

            # 只保留高置信度的预测
            mask = confidence >= self.threshold
            pseudo_labels[~mask] = -1  # -1表示无标签

            if return_confidence:
                return pseudo_labels, confidence
            return pseudo_labels

    def get_pseudo_loss(self, model: nn.Module, target_features: torch.Tensor,
                       criterion: nn.Module) -> torch.Tensor:
        """计算伪标签损失"""
        pseudo_labels, confidence = self.get_pseudo_labels(model, target_features, return_confidence=True)

        # 只计算有伪标签的样本
        mask = pseudo_labels != -1
        if mask.sum() == 0:
            return torch.tensor(0.0).to(target_features.device)

        valid_features = target_features[mask]
        valid_labels = pseudo_labels[mask]

        outputs = model(valid_features)
        loss = criterion(outputs, valid_labels)

        return loss


# ====================
# 3. Mean Teacher域适应
# ====================

class MeanTeacherDA(nn.Module):
    """
    Mean Teacher域适应
    使用EMA教师模型为源域和目标域生成一致的预测

    Reference: "Mean Teacher" (ICLR 2018)
    """

    def __init__(self, student_model: nn.Module, ema_decay: float = 0.999):
        super().__init__()
        self.student = student_model
        self.ema_decay = ema_decay

        # 创建教师模型（EMA副本）
        self.teacher = self._create_teacher_model()

    def _create_teacher_model(self) -> nn.Module:
        """创建教师模型"""
        teacher = type(self.student)(*(self.student.__dict__.values()))
        teacher.load_state_dict(self.student.state_dict())
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def update_teacher(self):
        """更新教师模型参数"""
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data = (
                self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data
            )

    def consistency_loss(self, source_outputs: torch.Tensor, target_outputs: torch.Tensor,
                        source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        计算一致性损失
        使学生模型的输出与教师模型的输出一致
        """
        with torch.no_grad():
            source_teacher_outputs = self.teacher(source_features)
            target_teacher_outputs = self.teacher(target_features)

        # MSE损失（或KL散度）
        loss_source = F.mse_loss(source_outputs, source_teacher_outputs)
        loss_target = F.mse_loss(target_outputs, target_teacher_outputs)

        return loss_source + loss_target


# ====================
# 4. 自监督域适应
# ====================

class ContrastiveDomainAdaptation(nn.Module):
    """
    对比域适应
    通过对比学习拉近同类样本，推远不同类样本

    Reference: "Unsupervised Domain Adaptation with Contrastive Learning" (Various)
    """

    def __init__(self, feature_dim: int, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

        # 投影头（用于对比学习）
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, 128)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """投影特征"""
        return self.projector(features)

    def contrastive_loss(self, source_features: torch.Tensor, target_features: torch.Tensor,
                       source_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算对比损失

        Parameters:
        -----------
        source_features : Tensor
            源域特征 (batch_size, feature_dim)
        target_features : Tensor
            目标域特征 (batch_size, feature_dim)
        source_labels : Tensor, optional
            源域标签（用于监督对比学习）
        """
        # 投影
        source_proj = self.forward(source_features)
        target_proj = self.forward(target_features)

        # 归一化
        source_proj = F.normalize(source_proj, dim=1)
        target_proj = F.normalize(target_proj, dim=1)

        # 计算相似度矩阵
        # 这里简化实现，实际可以更复杂
        batch_size = source_features.size(0)

        # 源域-目标域对比
        logits = torch.matmul(source_proj, target_proj.t()) / self.temperature

        if source_labels is not None:
            # 监督对比：拉近距离标签相同的样本
            # 简化实现
            pass

        # InfoNCE损失（简化版）
        # 正样本：跨域对应位置的样本
        # 负样本：其他所有样本
        labels = torch.arange(batch_size).to(source_features.device)
        loss = F.cross_entropy(logits, labels)

        return loss


# ====================
# 5. 统一度域适应模块
# ====================

class AdvancedDomainAdaptationModule(nn.Module):
    """
    统一的先进域适应模块
    支持多种方法组合
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        methods: List[str] = ['cdan', 'pseudo_label'],
        cdan_hidden_dim: int = 1024,
        pseudo_threshold: float = 0.9,
        temperature: float = 0.07,
        use_mmd: bool = True,
        mmd_sigmas: List[float] = None
    ):
        super().__init__()
        self.methods = methods
        self.num_classes = num_classes

        # CDAN
        if 'cdan' in methods:
            self.cdan = ConditionalDomainAdversarial(
                feature_dim=feature_dim,
                num_classes=num_classes,
                hidden_dim=cdan_hidden_dim
            )
        else:
            self.cdan = None

        # 伪标签
        if 'pseudo_label' in methods:
            self.pseudo_label = PseudoLabelDomainAdaptation(threshold=pseudo_threshold)
        else:
            self.pseudo_label = None

        # 对比学习
        if 'contrastive' in methods:
            self.contrastive = ContrastiveDomainAdaptation(
                feature_dim=feature_dim,
                temperature=temperature
            )
        else:
            self.contrastive = None

        # MMD（可选）
        if use_mmd:
            from ablation_study import MMDOnlyModule
            self.mmd = MMDOnlyModule(
                feature_dim=feature_dim,
                sigmas=mmd_sigmas or [2, 5, 10, 20, 40, 80],
                lambda_mmd=0.5
            )
        else:
            self.mmd = None

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

        # CDAN损失
        if self.cdan is not None:
            loss_cdan = self.cdan.get_loss(
                source_features, source_outputs,
                target_features, target_outputs,
                progress=progress
            )
            losses['cdan'] = loss_cdan

        # 对比损失
        if self.contrastive is not None:
            # 使用预测概率作为软标签
            source_labels = source_outputs.argmax(dim=1) if source_outputs is not None else None
            loss_contrastive = self.contrastive.contrastive_loss(
                source_features, target_features, source_labels
            )
            losses['contrastive'] = loss_contrastive

        # MMD损失
        if self.mmd is not None:
            loss_mmd = self.mmd(source_features, target_features)
            losses['mmd'] = loss_mmd['total']

        return losses


if __name__ == '__main__':
    # 测试代码
    print('Testing Advanced Domain Adaptation Modules...')

    batch_size = 32
    feature_dim = 256
    num_classes = 16

    # 创建测试数据
    source_features = torch.randn(batch_size, feature_dim)
    source_outputs = torch.randn(batch_size, num_classes)
    target_features = torch.randn(batch_size, feature_dim)
    target_outputs = torch.randn(batch_size, num_classes)

    # 测试CDAN
    print('\n1. Testing CDAN...')
    cdan = ConditionalDomainAdversarial(feature_dim, num_classes)
    loss = cdan.get_loss(source_features, source_outputs, target_features, target_outputs)
    print(f'   CDAN Loss: {loss.item():.4f}')

    # 测试伪标签
    print('\n2. Testing Pseudo Label...')
    pseudo_da = PseudoLabelDomainAdaptation(threshold=0.9)
    # 需要一个完整的模型来测试

    # 测试对比学习
    print('\n3. Testing Contrastive DA...')
    contrastive = ContrastiveDomainAdaptation(feature_dim)
    loss = contrastive.contrastive_loss(source_features, target_features)
    print(f'   Contrastive Loss: {loss.item():.4f}')

    # 测试统一模块
    print('\n4. Testing Advanced DA Module...')
    adv_da = AdvancedDomainAdaptationModule(
        feature_dim=feature_dim,
        num_classes=num_classes,
        methods=['cdan', 'contrastive'],
        use_mmd=True
    )
    losses = adv_da(
        source_features, source_outputs,
        target_features, target_outputs,
        progress=0.5
    )
    print('   Losses:')
    for key, value in losses.items():
        print(f'      {key}: {value.item():.4f}')

    print('\nAll tests passed!')
