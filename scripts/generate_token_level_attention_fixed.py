# -*- coding: utf-8 -*-
"""
Generate Encoder Self-Attention Visualization - Fixed Spectrogram Version

修改自 generate_token_level_attention.py
只修改了 spectrogram 绘制部分，使用去DC + dB + percentile 方法
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

from _project_paths import DATASETS_DIR, RESULTS_DIR

from models.crossvit import CrossViTFaultDiagnosis
from data.data_processor import load_csv_data, MultiModalFaultDataset
from torch.utils.data import DataLoader


class EncoderSelfAttentionExtractor(nn.Module):
    """
    提取 Encoder 内部的 Self-Attention

    使用临时修改的 TransformerEncoder 来提取 attention weights
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.attention_storage = {}

        # 临时替换 encoder 为能返回 attention 的版本
        self._patch_encoders()

    def _patch_encoders(self):
        """
        临时修改 encoder 让最后一层返回 attention weights
        通过替换整个 TransformerEncoder
        """
        # 保存原始 encoders
        self._original_time_encoder = self.base_model.time_encoder
        self._original_spec_encoder = self.base_model.spec_encoder

        # 创建新的能返回 attention 的 encoders
        self.base_model.time_encoder = AttentionReturningTransformerEncoder(
            self.base_model.time_encoder, 'time', self.attention_storage)
        self.base_model.spec_encoder = AttentionReturningTransformerEncoder(
            self.base_model.spec_encoder, 'spec', self.attention_storage)

    def forward(self, time_x, spec_x):
        """
        前向传播，使用修改后的 encoder 提取 attention
        """
        # 清空现有的 attention_storage（不要创建新的字典对象）
        self.attention_storage.clear()

        # ==================== Time branch ====================
        time_feat = self.base_model.time_patch_embed(time_x)
        time_feat = self.base_model.time_pos_encoder(time_feat)
        time_feat = self.base_model.time_encoder(time_feat)  # 会存储 attention

        # Pooling
        time_pooled = time_feat.mean(dim=1)

        # ==================== Spectrogram branch ====================
        spec_feat = self.base_model.spec_patch_embed(spec_x)
        spec_feat = self.base_model.spec_pos_encoder(spec_feat)
        spec_feat = self.base_model.spec_encoder(spec_feat)  # 会存储 attention

        # Pooling
        spec_pooled = spec_feat.mean(dim=1)

        # ==================== Cross-attention fusion ====================
        time_feat_exp = time_pooled.unsqueeze(1)
        spec_feat_exp = spec_pooled.unsqueeze(1)

        fused_time, fused_spec = self.base_model.cross_attn(
            time_feat_exp, spec_feat_exp)

        fused_time = fused_time.squeeze(1)
        fused_spec = fused_spec.squeeze(1)

        # ==================== Classification ====================
        fused_feat = torch.cat([fused_time, fused_spec], dim=1)
        output = self.base_model.classifier(fused_feat)

        return output

    def get_attention_weights(self):
        """返回存储的 attention weights"""
        return self.attention_storage


class AttentionReturningTransformerEncoder(nn.Module):
    """
    Wrapper for TransformerEncoder that captures attention from the last layer
    """

    def __init__(self, original_encoder, branch_name, attention_storage):
        super().__init__()
        self.original_encoder = original_encoder
        self.branch_name = branch_name
        self.attention_storage = attention_storage

    def forward(self, x):
        """Forward through encoder, capturing attention from last layer"""
        # Run through all layers except the last one
        layers = list(self.original_encoder.encoder.layers)
        for layer in layers[:-1]:
            x = layer(x)

        # For the last layer, manually call self_attn with need_weights=True
        last_layer = layers[-1]

        # Norm 1
        src = last_layer.norm1(x)

        # Self-attention with weights
        attn_output, attn_weights = last_layer.self_attn(
            src, src, src,
            need_weights=True,
            average_attn_weights=False
        )

        # Store attention weights
        key = f'{self.branch_name}_self_attn'
        self.attention_storage[key] = attn_weights.detach()

        # Rest of the layer
        x = x + last_layer.dropout1(attn_output)
        src = last_layer.norm2(x)
        ffn_output = last_layer.linear2(last_layer.dropout(last_layer.activation(last_layer.linear1(src))))
        x = x + last_layer.dropout2(ffn_output)

        return x


class EncoderSelfAttentionVisualizer:
    """可视化 Encoder 内部的 Self-Attention"""

    def __init__(self, model_with_extractor, device='cuda'):
        self.device = device
        # 不要在这里 to(device)，因为它会创建副本，导致 wrapper 失效
        self.model = model_with_extractor
        self.model.eval()
        # 手动移动到设备
        for param in self.model.parameters():
            param.data = param.data.to(device)
        for buffer in self.model.buffers():
            if buffer.data.device != torch.device(device):
                buffer.data = buffer.data.to(device)

    def visualize_sample(self, time_x, spec_x, true_label, pred_label,
                        class_names, save_path='encoder_self_attention.png'):
        with torch.no_grad():
            time_x = time_x.to(self.device)
            spec_x = spec_x.to(self.device)

            output = self.model(time_x, spec_x)
            attn_weights = self.model.get_attention_weights()

            probs = F.softmax(output, dim=1)[0].cpu().numpy()

        # 创建图形：左侧是 time attention，右侧是 spec attention
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 8, figure=fig, hspace=0.4, wspace=0.3)

        # 第一行：输入数据可视化
        # 1. Time-domain signal
        ax1 = fig.add_subplot(gs[0, :4])
        self._plot_time_domain(ax1, time_x[0])

        # 2. Spectrogram
        ax2 = fig.add_subplot(gs[0, 4:])
        self._plot_spectrogram(ax2, spec_x[0])

        # 第二行：预测概率
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_predictions(ax3, probs, true_label, pred_label, class_names)

        # 第三、四行：Self-attention 可视化
        # 左侧：Time encoder self-attention (4个 head)
        if 'time_self_attn' in attn_weights:
            time_attn = attn_weights['time_self_attn'][0].cpu().numpy()  # (heads, Nt, Nt)
            print(f"  Time self-attention shape: {time_attn.shape}")
            print(f"  Time self-attention range: [{time_attn.min():.4f}, {time_attn.max():.4f}]")

            num_time_heads = min(time_attn.shape[0], 4)
            for i in range(num_time_heads):
                ax = fig.add_subplot(gs[2:, i])
                im = ax.imshow(time_attn[i], cmap='viridis', aspect='auto')
                ax.set_title(f'Time Encoder Head {i+1}', fontsize=9, fontweight='bold', pad=3)
                ax.set_xlabel('Time Tokens', fontsize=7)
                ax.set_ylabel('Time Tokens', fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 右侧：Spectrogram encoder self-attention (4个 head)
        if 'spec_self_attn' in attn_weights:
            spec_attn = attn_weights['spec_self_attn'][0].cpu().numpy()  # (heads, Ns, Ns)
            print(f"  Spec self-attention shape: {spec_attn.shape}")
            print(f"  Spec self-attention range: [{spec_attn.min():.4f}, {spec_attn.max():.4f}]")

            num_spec_heads = min(spec_attn.shape[0], 4)
            for i in range(num_spec_heads):
                ax = fig.add_subplot(gs[2:, 4+i])
                im = ax.imshow(spec_attn[i], cmap='viridis', aspect='auto')
                ax.set_title(f'Spec Encoder Head {i+1}', fontsize=9, fontweight='bold', pad=3)
                ax.set_xlabel('Spec Tokens', fontsize=7)
                ax.set_ylabel('Spec Tokens', fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle('Encoder Self-Attention Visualization\n(Time-Domain & Spectrogram)',
                    fontsize=15, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.close()

    def _plot_time_domain(self, ax, time_data):
        """Plot normalized time-domain signal"""
        time_np = time_data.cpu().numpy()

        channels = ['Ch1 (X)', 'Ch2 (Y)', 'Ch3 (Z)']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i in range(min(time_np.shape[1], 3)):
            signal = time_np[:, i]
            ax.plot(signal, label=channels[i], color=colors[i],
                   linewidth=1.2, alpha=0.8)

            if signal.max() > 0.95 and signal.min() < -0.95:
                ax.text(0.02, 0.95 - i*0.05,
                       f'Note: {channels[i]} appears normalized',
                       transform=ax.transAxes, fontsize=7,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        ax.set_title('Normalized Time-domain Vibration Signal',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Sample', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_spectrogram(self, ax, spec_data):
        """修复版 Spectrogram 绘制 - 去DC + dB + percentile"""
        spec_np = spec_data.cpu().numpy().astype(np.float32)

        if spec_np.shape[0] == 3:
            spec_avg = spec_np.mean(0)
        else:
            spec_avg = spec_np[0]

        # 去除 DC 分量
        if spec_avg.shape[0] > 1:
            spec_avg[0, :] = spec_avg[1:5, :].mean(axis=0)

        # dB 缩放
        spec_db = 10 * np.log10(spec_avg + 1e-8)

        # Percentile 裁剪
        vmin, vmax = np.percentile(spec_db, [5, 99])
        spec_db = np.clip(spec_db, vmin, vmax)

        # 归一化
        spec_norm = (spec_db - vmin) / (vmax - vmin + 1e-6)

        im = ax.imshow(spec_norm, cmap='viridis', aspect='auto', origin='lower')

        ax.set_title('Spectrogram (Time-Frequency)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Frame', fontsize=10)
        ax.set_ylabel('Frequency Bin', fontsize=10)
        plt.colorbar(im, ax=ax, label='Power (dB)')

    def _plot_predictions(self, ax, probs, true_label, pred_label, class_names):
        """Plot prediction probabilities"""
        y_pos = np.arange(len(class_names))

        colors = ['#2ECC71' if i == pred_label else '#4682B4'
                 for i in range(len(class_names))]

        bars = ax.barh(y_pos, probs, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=0.8)

        for i, (bar, prob) in enumerate(zip(bars, probs)):
            width = bar.get_width()
            ax.text(width + 0.015, bar.get_y() + bar.get_height()/2,
                   f'{prob:.3f}', ha='left', va='center',
                   fontsize=8, fontweight='bold' if i == pred_label else 'normal')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel('Probability', fontsize=11)
        ax.set_title(f'Prediction | True: C{true_label} ({class_names[true_label]}) | '
                    f'Pred: C{pred_label} ({class_names[pred_label]})',
                    fontsize=12, fontweight='bold',
                    color='#2ECC71' if true_label == pred_label else '#E74C3C')
        ax.set_xlim(0, 1.2)
        ax.grid(True, axis='x', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate Encoder Self-Attention Visualization - Fixed Spectrogram')
    parser.add_argument('--model', type=str,
                       default=str(RESULTS_DIR / 'models' / 'exp3_v3_20260618_010348' / 'top2_Distill+LWF__384__T_2__alpha_0.3__20260618_010348.pth'))
    parser.add_argument('--output', type=str, default=str(RESULTS_DIR / 'encoder_self_attention_fixed'))
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("\n" + "="*60)
    print("Encoder Self-Attention Visualization (Fixed Spectrogram)")
    print("="*60)
    print("Note: Visualizing self-attention INSIDE the encoders,")
    print("      NOT cross-attention (which operates on pooled features)")
    print("=" * 60 + "\n")

    # Load data
    print("Loading data...")
    BASE_DIR = str(DATASETS_DIR)

    source_data = load_csv_data(
        f"{BASE_DIR}/dataset2_1.0kW.csv",
        window_size=1024, stride=128, spec_size=(128, 128),
        test_size=0.2, val_size=0.1, random_state=192
    )

    target_data = load_csv_data(
        f"{BASE_DIR}/dataset2_3.0kW.csv",
        window_size=1024, stride=128, spec_size=(128, 128),
        test_size=0.2, val_size=0.1, random_state=192
    )

    target_test_dataset = MultiModalFaultDataset(
        target_data['X_test_time'], target_data['X_test_spec'], target_data['y_test']
    )
    test_loader = DataLoader(target_test_dataset, batch_size=1, shuffle=False)

    # Class names - 使用源域的16类故障代码（模型训练时的类别）
    class_names = [str(c) for c in source_data['fault_codes']]

    # Load model
    print(f"Loading model: {args.model}")

    base_model = CrossViTFaultDiagnosis(
        in_channels=source_data['n_channels'],
        num_classes=16,
        time_seq_len=1024,
        spec_height=128,
        spec_width=128,
        embed_dim=384,
        num_heads=8,
        num_layers=2,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(args.model, map_location=device)
    if 'model_state_dict' in checkpoint:
        base_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        base_model.load_state_dict(checkpoint)

    print("Model loaded successfully")

    # Wrap model for self-attention extraction
    model_with_extractor = EncoderSelfAttentionExtractor(base_model)

    # Create visualizer
    visualizer = EncoderSelfAttentionVisualizer(model_with_extractor, device)

    # Generate visualizations
    print(f"\nGenerating {args.num_samples} encoder self-attention visualizations...")

    sample_count = 0
    correct_count = 0
    error_count = 0

    for time_x, spec_x, true_label in test_loader:
        if sample_count >= args.num_samples:
            break

        try:
            with torch.no_grad():
                time_x_dev = time_x.to(device)
                spec_x_dev = spec_x.to(device)
                output = model_with_extractor(time_x_dev, spec_x_dev)
                pred_label = output.argmax(1)[0].cpu().item()

            true_label_val = true_label.item()

            if true_label_val >= len(class_names) or pred_label >= len(class_names):
                print(f"  Warning: label out of range, skipping")
                continue

            save_path = os.path.join(args.output,
                                   f'self_attn_sample_{sample_count+1}_true{true_label_val}_pred{pred_label}.png')

            visualizer.visualize_sample(
                time_x, spec_x, true_label_val, pred_label,
                class_names, save_path
            )

            sample_count += 1
            if true_label_val == pred_label:
                correct_count += 1
            else:
                error_count += 1

            print(f"  [{sample_count}/{args.num_samples}] True: {true_label_val}, Pred: {pred_label} - Saved")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    print("\n" + "=" * 60)
    print("Encoder Self-Attention Visualization Complete!")
    print("=" * 60)
    print(f"Total samples: {sample_count}")
    print(f"Correct: {correct_count}")
    print(f"Errors: {error_count}")
    print(f"Output: {args.output}")
    print("=" * 60)


if __name__ == '__main__':
    main()
