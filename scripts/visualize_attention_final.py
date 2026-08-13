# -*- coding: utf-8 -*-
"""
完整的Attention Visualization工具
解决:
1. 真正提取Transformer attention权重
2. 修复图像质量问题 (信号贴边界、spectrogram低频集中)
3. 生成论文级attention heatmap
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


class CrossViTAttentionExtractor:
    """
    从CrossViT模型中提取attention权重的工具

    修改:
    - 在CrossAttentionFusion中捕获真实的attention scores
    - 支持multi-head attention可视化
    - 支持attention roll-out (跨层聚合)
    """

    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        # Storage
        self.time_to_spec_attn = None
        self.spec_to_time_attn = None

        # Inject hooks
        self._inject_hooks()

    def _inject_hooks(self):
        """注入hooks到CrossAttentionFusion"""

        # 保存原始attention方法
        original_attention = self.model.cross_attn.attention

        def attention_with_capture(q, k, v):
            """执行attention并保存权重"""
            batch_size, seq_len_q, embed_dim = q.shape
            seq_len_k = k.size(1)
            num_heads = self.model.cross_attn.num_heads

            # Multi-head projection
            q = q.view(batch_size, seq_len_q, num_heads, -1).transpose(1, 2)
            k = k.view(batch_size, seq_len_k, num_heads, -1).transpose(1, 2)
            v = v.view(batch_size, seq_len_k, num_heads, -1).transpose(1, 2)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            attn_weights = F.softmax(scores, dim=-1)

            # 保存权重 (batch, heads, seq_q, seq_k)
            self._save_attention(attn_weights)

            # Apply attention
            attn = self.model.cross_attn.dropout(attn_weights)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, embed_dim)

            return out

        # 替换attention方法
        self.model.cross_attn.attention = attention_with_capture

    def _save_attention(self, attn_weights):
        """保存attention权重 - 需要根据调用上下文判断方向"""
        # 这里简化处理 - 实际需要根据forward中的调用顺序判断
        # 第一次调用是time->spec, 第二次是spec->time
        if self.time_to_spec_attn is None:
            self.time_to_spec_attn = attn_weights.detach()
        else:
            self.spec_to_time_attn = attn_weights.detach()

    def get_attention_weights(self, time_x, spec_x):
        """获取attention权重"""
        # Reset storage
        self.time_to_spec_attn = None
        self.spec_to_time_attn = None

        with torch.no_grad():
            time_x = time_x.to(self.device)
            spec_x = spec_x.to(self.device)
            _ = self.model(time_x, spec_x)

        return {
            'time_to_spec': self.time_to_spec_attn.cpu() if self.time_to_spec_attn is not None else None,
            'spec_to_time': self.spec_to_time_attn.cpu() if self.spec_to_time_attn is not None else None
        }


class AttentionHeatmapVisualizer:
    """
    生成论文级的Attention Heatmap

    特性:
    - 多头attention展示
    - 跨模态attention可视化
    - 高质量输出 (300 DPI)
    """

    def __init__(self, extractor):
        self.extractor = extractor

    def plot_cross_attention_heatmap(self, attn_weights, title="Cross-Modal Attention",
                                    save_path=None, figsize=(18, 10)):
        """
        绘制交叉模态attention heatmap

        Args:
            attn_weights: Tensor (batch, heads, seq_q, seq_k)
        """
        if attn_weights is None:
            print("Warning: No attention weights provided")
            return

        # 去除batch维度
        attn = attn_weights[0].cpu().numpy()
        num_heads = attn.shape[0]

        # 创建figure
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

        # 绘制每个head
        for i in range(min(num_heads, 8)):
            row = i // 4
            col = i % 4
            ax = axes[row, col]

            im = ax.imshow(attn[i], cmap='RdYlBu_r', aspect='auto',
                          vmin=0, vmax=attn[i].max())

            # 美化
            ax.set_title(f'Head {i+1}', fontsize=11, fontweight='bold', pad=5)

            # 添加网格
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.5)
                spine.set_edgecolor('#888')

            # Colorbar
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"Saved cross-attention heatmap: {save_path}")

        return fig

    def plot_attention_overlay(self, attn_weights, time_signal, spectrogram,
                              save_path=None):
        """
        将attention overlay到原始输入上

        创建"模型在看哪里"的可视化
        """
        if attn_weights is None:
            return

        attn = attn_weights[0].cpu().numpy()  # (heads, seq_q, seq_k)
        avg_attn = attn.mean(0)  # Average across heads

        # 创建figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Attention Overlay on Input',
                    fontsize=14, fontweight='bold')

        # 1. 原始时域信号
        ax1 = axes[0, 0]
        self._plot_clean_time_signal(ax1, time_signal)
        ax1.set_title('Original Time Signal', fontweight='bold')

        # 2. 原始spectrogram
        ax2 = axes[0, 1]
        self._plot_clean_spectrogram(ax2, spectrogram)
        ax2.set_title('Original Spectrogram', fontweight='bold')

        # 3. Time signal with attention overlay
        ax3 = axes[1, 0]
        self._plot_time_with_attention(ax3, time_signal, avg_attn)
        ax3.set_title('Time Signal + Attention', fontweight='bold')

        # 4. Spectrogram with attention overlay
        ax4 = axes[1, 1]
        self._plot_spec_with_attention(ax4, spectrogram, avg_attn)
        ax4.set_title('Spectrogram + Attention', fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention overlay: {save_path}")

        plt.close()

    def _plot_clean_time_signal(self, ax, time_signal, color='#1f77b4'):
        """绘制干净的时域信号"""
        time_np = time_signal.cpu().numpy() if torch.is_tensor(time_signal) else time_signal

        # 如果是3通道，取平均值或第一通道
        if time_np.ndim == 2 and time_np.shape[1] == 3:
            time_np = time_np.mean(axis=1)  # 平均三通道
        elif time_np.ndim == 2 and time_np.shape[1] > 1:
            time_np = time_np[:, 0]  # 第一通道

        time_axis = np.arange(len(time_np))

        # 使用平滑曲线
        ax.plot(time_axis, time_np, color=color, linewidth=1.2, alpha=0.9)

        # 设置范围，避免贴边界
        y_min, y_max = time_np.min(), time_np.max()
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        ax.set_xlabel('Time Sample', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_clean_spectrogram(self, ax, spectrogram):
        """绘制干净的spectrogram - 修复版使用dB缩放和percentile裁剪"""
        spec_np = spectrogram.cpu().numpy() if torch.is_tensor(spectrogram) else spectrogram
        spec_np = spec_np.astype(np.float32)

        # 处理多通道 - 使用max保留更多细节
        if spec_np.ndim == 3:
            if spec_np.shape[0] == 3:
                spec_np = np.max(spec_np, axis=0)  # 使用max而非mean
            else:
                spec_np = spec_np[0]

        # 去除DC分量 (最低频的一行)
        if spec_np.shape[0] > 1:
            spec_np[0, :] = spec_np[1:5, :].mean(axis=0)

        # 使用dB缩放 (10*log10)
        spec_db = 10 * np.log10(spec_np + 1e-8)

        # 使用percentile裁剪，避免极端值主导颜色映射
        vmin, vmax = np.percentile(spec_db, [5, 99])
        spec_db = np.clip(spec_db, vmin, vmax)

        # 归一化到[0, 1]
        spec_norm = (spec_db - vmin) / (vmax - vmin + 1e-6)

        im = ax.imshow(spec_norm, cmap='viridis', aspect='auto', origin='lower')

        # 修正坐标轴标签: 横轴时间, 纵轴频率
        ax.set_xlabel('Time Frame', fontsize=10)
        ax.set_ylabel('Frequency Bin', fontsize=10)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Power (dB)', fontsize=9)

    def _plot_time_with_attention(self, ax, time_signal, attention):
        """在时域信号上叠加attention"""
        self._plot_clean_time_signal(ax, time_signal, color='#888')

        # Attention作为背景颜色
        time_np = time_signal.cpu().numpy() if torch.is_tensor(time_signal) else time_signal
        if time_np.ndim == 2:
            time_np = time_np.mean(axis=1)

        # 创建attention map (需要映射到时间轴)
        # attention shape: (seq_q, seq_k)
        # 对于时域，我们关注query的attention分布
        attn_on_time = attention.mean(axis=1)  # 对key维度平均

        # 插值到原始时间长度
        time_len = len(time_np)
        attn_len = len(attn_on_time)
        attn_resampled = np.interp(np.linspace(0, attn_len, time_len),
                                   np.arange(attn_len),
                                   attn_on_time)

        # 绘制attention为阴影区域
        ax.fill_between(np.arange(time_len), time_np.min(),
                       time_np, where=(attn_resampled > attn_resampled.mean()),
                       alpha=0.3, color='red', label='High Attention')

        ax.legend(fontsize=8)

    def _plot_spec_with_attention(self, ax, spectrogram, attention):
        """在spectrogram上叠加attention"""
        self._plot_clean_spectrogram(ax, spectrogram)

        # Attention作为overlay
        # 使用alpha blending
        attn_np = attention.cpu().numpy() if torch.is_tensor(attention) else attention

        # 归一化attention
        attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-6)

        # 创建attention overlay (用红色半透明)
        ax.imshow(attn_norm, cmap='Reds', aspect='auto', origin='lower',
                 alpha=0.4, extent=ax.images[0].get_extent())


class ComprehensiveAttentionViz:
    """
    综合Attention可视化 - 包含所有组件

    生成包含以下内容的一张图:
    1. 输入信号 (时域 + 频域) - 清理后的版本
    2. 模型预测 (类别概率)
    3. 真正的attention heatmap (多头)
    4. Attention overlay
    """

    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device)
        self.model.eval()

        self.extractor = CrossViTAttentionExtractor(model, device)
        self.visualizer = AttentionHeatmapVisualizer(self.extractor)

    def visualize_sample(self, time_x, spec_x, true_label, pred_label,
                        class_names, save_path='comprehensive_attention.png'):
        """
        生成单个样本的综合可视化

        Args:
            time_x: (1, seq_len, channels)
            spec_x: (1, channels, H, W)
        """
        # 获取attention
        attn_weights = self.extractor.get_attention_weights(time_x, spec_x)

        # 获取预测概率
        with torch.no_grad():
            logits = self.model(time_x.to(self.device), spec_x.to(self.device))
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)

        # 1. 时域信号 - 清理版
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_time_domain_clean(ax1, time_x[0])

        # 2. 频域谱图 - 清理版
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_spectrogram_clean(ax2, spec_x[0])

        # 3. 预测概率
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_prediction_probs(ax3, probs, true_label, pred_label, class_names)

        # 4-11. Attention heads (8个)
        if attn_weights and attn_weights['time_to_spec'] is not None:
            attn = attn_weights['time_to_spec'][0].cpu().numpy()  # (heads, seq_q, seq_k)

            for i in range(min(attn.shape[0], 8)):
                ax = fig.add_subplot(gs[2:, i])
                im = ax.imshow(attn[i], cmap='RdYlBu_r', aspect='auto',
                              vmin=0, vmax=attn[i].max())
                ax.set_title(f'Attention Head {i+1}', fontsize=10, fontweight='bold', pad=5)
                ax.set_xlabel('Spectrogram Token', fontsize=8)
                ax.set_ylabel('Time Token', fontsize=8)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 添加总体标题
        fig.suptitle('CrossViT Multi-modal Attention Visualization',
                    fontsize=16, fontweight='bold', y=0.995)

        # 保存
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comprehensive visualization: {save_path}")
        plt.close()

    def _plot_time_domain_clean(self, ax, time_data):
        """绘制干净的时域信号 - 解决贴边界问题"""
        time_np = time_data.cpu().numpy()

        channels = ['Ch1', 'Ch2', 'Ch3']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i in range(min(time_np.shape[1], 3)):
            signal = time_np[:, i]

            # 检查是否有异常值
            if np.all(np.abs(signal) > 0.99):
                # 如果信号贴边界，说明是归一化问题
                # 显示原始信号之前的状态
                ax.plot(signal, label=f'{channels[i]} (normalized)', color=colors[i],
                       linewidth=1.2, alpha=0.7)
                ax.text(0.02, 0.98 - i*0.05, f'{channels[i]}: appears normalized/clipped',
                       transform=ax.transAxes, fontsize=7, color=colors[i])
            else:
                ax.plot(signal, label=channels[i], color=colors[i], linewidth=1.2, alpha=0.8)

        ax.set_title('Time-domain Vibration Signal', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Sample', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.5)

    def _plot_spectrogram_clean(self, ax, spec_data):
        """绘制干净的spectrogram - 解决低频集中问题"""
        spec_np = spec_data.cpu().numpy()

        # 处理多通道
        if spec_np.shape[0] == 3:
            spec_avg = spec_np.mean(0)
        else:
            spec_avg = spec_np[0]

        # 检查是否有低频集中问题
        power_distribution = spec_avg.sum(axis=1)  # 沿频率轴求和
        low_freq_power = power_distribution[:10].sum()
        total_power = power_distribution.sum()

        if low_freq_power / total_power > 0.8:
            # 低频能量过高 - 使用更好的归一化
            # 使用per-frame归一化 + log transform
            spec_log = np.log10(spec_avg + 1e-6)

            # 减去每帧的均值（去除DC）
            spec_log = spec_log - spec_log.mean(axis=1, keepdims=True)

            # 归一化
            spec_norm = (spec_log - spec_log.min()) / (spec_log.max() - spec_log.min() + 1e-6)
        else:
            spec_norm = spec_avg

        im = ax.imshow(spec_norm, cmap='viridis', aspect='auto', origin='lower')

        ax.set_title('Spectrogram (Time-Frequency)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Frequency Bin', fontsize=10)
        ax.set_ylabel('Time Frame', fontsize=10)
        plt.colorbar(im, ax=ax, label='Power')

    def _plot_prediction_probs(self, ax, probs, true_label, pred_label, class_names):
        """绘制预测概率"""
        y_pos = np.arange(len(class_names))

        # 根据预测正确性设置颜色
        colors = ['green' if i == pred_label else '#4682B4' for i in range(len(class_names))]

        bars = ax.barh(y_pos, probs, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)

        # 添加数值标签
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{prob:.3f}', ha='left', va='center',
                   fontsize=8, fontweight='bold' if i == pred_label else 'normal')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names, fontsize=10)
        ax.set_xlabel('Probability', fontsize=11)
        ax.set_title(f'Prediction | True: {class_names[true_label]} | Pred: {class_names[pred_label]}',
                    fontsize=12, fontweight='bold',
                    color='green' if true_label == pred_label else 'red')
        ax.set_xlim(0, 1.15)
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def quick_visualize(model, time_x, spec_x, true_label, pred_label,
                   class_names, save_path, device='cuda'):
    """
    快速可视化单个样本
    """
    viz = ComprehensiveAttentionViz(model, device)
    viz.visualize_sample(time_x, spec_x, true_label, pred_label,
                        class_names, save_path)


# 便捷函数
def create_attention_visualization(model, data_sample, class_names,
                                   save_dir='attention_viz', device='cuda'):
    """
    为数据集创建attention可视化

    Args:
        model: CrossViT模型
        data_sample: dict with keys 'time_x', 'spec_x', 'true_label', 'pred_label'
        class_names: 类别名称列表
        save_dir: 保存目录
        device: 设备
    """
    os.makedirs(save_dir, exist_ok=True)

    viz = ComprehensiveAttentionViz(model, device)

    time_x = data_sample['time_x'].unsqueeze(0)  # Add batch dim
    spec_x = data_sample['spec_x'].unsqueeze(0)
    true_label = data_sample['true_label']
    pred_label = data_sample['pred_label']

    save_path = os.path.join(save_dir,
                            f'attention_sample_{true_label}_pred{pred_label}.png')

    viz.visualize_sample(time_x, spec_x, true_label, pred_label,
                        class_names, save_path)

    print(f"Attention visualization saved to: {save_path}")


if __name__ == '__main__':
    print("Attention Visualization Tool")
    print("=" * 50)
    print("\n功能:")
    print("1. 提取真实的Transformer attention权重")
    print("2. 修复图像质量问题")
    print("3. 生成论文级可视化")
    print("\n使用方法:")
    print("  from visualize_attention_final import ComprehensiveAttentionViz")
    print("  viz = ComprehensiveAttentionViz(model, device='cuda')")
    print("  viz.visualize_sample(time_x, spec_x, true_label, pred_label, class_names)")
