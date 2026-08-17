# -*- coding: utf-8 -*-
"""
CrossViT故障诊断模型
双分支Transformer：时域分支 + 频域分支
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PatchEmbedding(nn.Module):
    """将输入序列patchify并嵌入"""

    def __init__(self, in_channels=1, embed_dim=128, patch_size=16, stride=8):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.stride = stride

        self.proj = nn.Conv1d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )

    def forward(self, x):
        # x: (batch, channels, seq_len) or (batch, seq_len, channels)
        if x.dim() == 3 and x.size(1) > x.size(2):
            # (batch, seq_len, channels) -> (batch, channels, seq_len)
            x = x.transpose(1, 2)

        x = self.proj(x)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x


class MultiScalePatchEmbedding(nn.Module):
    """多尺度patch嵌入（用于2D谱图）"""

    def __init__(self, in_channels=1, embed_dim=128, patch_sizes=[4, 8, 16]):
        super(MultiScalePatchEmbedding, self).__init__()

        self.patch_sizes = patch_sizes
        self.target_size = 8  # 目标特征图尺寸
        self.num_scales = len(patch_sizes)

        # 计算每个尺度的通道数，确保总和等于embed_dim
        channels_per_scale = embed_dim // self.num_scales
        self.total_conv_channels = channels_per_scale * self.num_scales

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, channels_per_scale,
                         kernel_size=ps, stride=ps, padding=ps // 2),
                nn.BatchNorm2d(channels_per_scale),
                nn.ReLU(inplace=True)
            )
            for ps in patch_sizes
        ])

        # 自适应池化到统一尺寸
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((self.target_size, self.target_size))
            for _ in patch_sizes
        ])

        # fusion层的输入通道数等于实际的总通道数
        self.fusion = nn.Sequential(
            nn.Conv2d(self.total_conv_channels, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: (batch, channels, H, W)
        features = []
        for conv, pool in zip(self.convs, self.adaptive_pools):
            feat = conv(x)  # (batch, channels_per_scale, H', W')
            feat = pool(feat)  # (batch, channels_per_scale, target_size, target_size)
            features.append(feat)

        x = torch.cat(features, dim=1)  # (batch, total_conv_channels, target_size, target_size)
        x = self.fusion(x)

        # Flatten spatial dimensions
        batch, embed_dim, h, w = x.shape
        x = x.reshape(batch, embed_dim, h * w).transpose(1, 2)
        return x  # (batch, num_patches, embed_dim)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model=128, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, embed_dim=128, num_heads=8, num_layers=4, mlp_ratio=4, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: (batch, seq_len, embed_dim)
        x = self.encoder(x)
        return x


class CrossAttentionFusion(nn.Module):
    """交叉注意力融合模块"""

    def __init__(self, embed_dim=128, num_heads=8, num_layers=2, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # Q, K, V投影
        self.q_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(2)
        ])
        self.k_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(2)
        ])
        self.v_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(2)
        ])

        # 输出投影
        self.out_proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(2)
        ])

        # Layer Norm
        self.norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(2)
        ])

        self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v):
        # q, k, v: (batch, seq_len, embed_dim)
        batch_size, seq_len_q, embed_dim = q.shape
        seq_len_k = k.size(1)

        # Multi-head attention
        q = q.view(batch_size, seq_len_q, self.num_heads, -1).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, -1).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, -1).transpose(1, 2)

        # (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # (batch, num_heads, seq_len_q, embed_dim // num_heads)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, embed_dim)

        return out

    def forward(self, feat1, feat2):
        """
        双向交叉注意力

        Parameters:
        -----------
        feat1 : Tensor
            分支1特征 (batch, seq_len1, embed_dim)
        feat2 : Tensor
            分支2特征 (batch, seq_len2, embed_dim)

        Returns:
        --------
        fused1, fused2 : Tensor
            融合后的特征
        """
        # 分支1从分支2获取信息
        q1 = self.q_proj[0](feat1)
        k2 = self.k_proj[0](feat2)
        v2 = self.v_proj[0](feat2)
        cross_feat1 = self.attention(q1, k2, v2)
        cross_feat1 = self.out_proj[0](cross_feat1)
        fused1 = self.norm[0](feat1 + cross_feat1)

        # 分支2从分支1获取信息
        q2 = self.q_proj[1](feat2)
        k1 = self.k_proj[1](feat1)
        v1 = self.v_proj[1](feat1)
        cross_feat2 = self.attention(q2, k1, v1)
        cross_feat2 = self.out_proj[1](cross_feat2)
        fused2 = self.norm[1](feat2 + cross_feat2)

        return fused1, fused2


class CrossViTFaultDiagnosis(nn.Module):
    """
    CrossViT故障诊断模型
    双分支：时域分支 + 频域分支
    """

    def __init__(self, in_channels=3, num_classes=15,
                 time_seq_len=1024, spec_height=128, spec_width=128,
                 embed_dim=128, num_heads=8, num_layers=4,
                 mlp_ratio=4, dropout=0.1):
        super(CrossViTFaultDiagnosis, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # ==================
        # 时域分支
        # ==================
        self.time_patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=16,
            stride=8
        )
        self.time_pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        self.time_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # ==================
        # 频域分支
        # ==================
        self.spec_patch_embed = MultiScalePatchEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_sizes=[4, 8, 16]
        )
        self.spec_pos_encoder = PositionalEncoding(embed_dim, dropout=dropout)
        self.spec_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )

        # ==================
        # 交叉注意力融合
        # ==================
        self.cross_attn = CrossAttentionFusion(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=2,
            dropout=dropout
        )

        # ==================
        # 分类器
        # ==================
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, time_x, spec_x, return_features=False):
        """
        前向传播

        Parameters:
        -----------
        time_x : Tensor
            时域信号 (batch, seq_len, channels)
        spec_x : Tensor
            频域谱图 (batch, channels, H, W)
        return_features : bool
            是否返回中间特征（用于域适应）

        Returns:
        --------
        output : Tensor
            分类输出 (batch, num_classes)
        features : Tensor (optional)
            融合特征 (batch, embed_dim * 2)
        """
        # ==================
        # 时域分支
        # ==================
        time_feat = self.time_patch_embed(time_x)  # (batch, num_patches, embed_dim)
        time_feat = self.time_pos_encoder(time_feat)
        time_feat = self.time_encoder(time_feat)

        # 全局平均池化
        time_feat = time_feat.mean(dim=1)  # (batch, embed_dim)

        # ==================
        # 频域分支
        # ==================
        spec_feat = self.spec_patch_embed(spec_x)  # (batch, num_patches, embed_dim)
        spec_feat = self.spec_pos_encoder(spec_feat)
        spec_feat = self.spec_encoder(spec_feat)

        # 全局平均池化
        spec_feat = spec_feat.mean(dim=1)  # (batch, embed_dim)

        # ==================
        # 交叉注意力融合
        # ==================
        # 扩展维度用于交叉注意力
        time_feat_exp = time_feat.unsqueeze(1)  # (batch, 1, embed_dim)
        spec_feat_exp = spec_feat.unsqueeze(1)  # (batch, 1, embed_dim)

        fused_time, fused_spec = self.cross_attn(time_feat_exp, spec_feat_exp)

        # 去除扩展的维度
        fused_time = fused_time.squeeze(1)  # (batch, embed_dim)
        fused_spec = fused_spec.squeeze(1)  # (batch, embed_dim)

        # 拼接两个分支的特征
        fused_feat = torch.cat([fused_time, fused_spec], dim=1)  # (batch, embed_dim * 2)

        # 分类
        output = self.classifier(fused_feat)

        if return_features:
            return output, fused_feat
        return output

    def get_features(self, time_x, spec_x):
        """提取融合特征（用于域适应和可视化）"""
        with torch.no_grad():
            _, features = self.forward(time_x, spec_x, return_features=True)
        return features


if __name__ == '__main__':
    # 测试模型
    print('Testing CrossViT model...')

    batch_size = 4
    time_seq_len = 1024
    spec_height, spec_width = 128, 128
    in_channels = 3
    num_classes = 15

    # 创建测试数据
    time_x = torch.randn(batch_size, time_seq_len, in_channels)
    spec_x = torch.randn(batch_size, in_channels, spec_height, spec_width)

    # 创建模型
    model = CrossViTFaultDiagnosis(
        in_channels=in_channels,
        num_classes=num_classes,
        time_seq_len=time_seq_len,
        spec_height=spec_height,
        spec_width=spec_width,
        embed_dim=128,
        num_heads=8,
        num_layers=4
    )

    # 前向传播
    output, features = model(time_x, spec_x, return_features=True)

    print(f'Time input shape: {time_x.shape}')
    print(f'Spectrogram input shape: {spec_x.shape}')
    print(f'Output shape: {output.shape}')
    print(f'Features shape: {features.shape}')

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')
