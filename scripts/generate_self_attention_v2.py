# -*- coding: utf-8 -*-
"""
Extract Self-Attention from Transformer Encoders

Uses the built-in need_weights parameter to get trained attention patterns
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
from torch.utils.data import DataLoader

matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from _project_paths import DATASETS_DIR, RESULTS_DIR

from models.crossvit import CrossViTFaultDiagnosis
from data.data_processor import load_csv_data, MultiModalFaultDataset


class AttentionTrackingTransformerEncoder(nn.Module):
    """Wrapper that tracks attention in transformer layers"""

    def __init__(self, original_encoder):
        super().__init__()
        self.original_encoder = original_encoder
        self.attention_weights = []

    def forward(self, x):
        self.attention_weights.clear()

        # Process through each layer manually
        for layer in self.original_encoder.encoder.layers:
            # LayerNorm 1
            src = layer.norm1(x)

            # Self-attention with weight capture using built-in method
            src2, attn_weights = layer.self_attn(
                src, src, src,
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )
            self.attention_weights.append(attn_weights.detach())

            # Residual connection
            x = x + layer.dropout1(src2)

            # Feed-forward network
            x2 = layer.norm2(x)
            x2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(x2))))
            x = x + layer.dropout2(x2)

        return x


class SelfAttentionExtractor(nn.Module):
    """Extract self-attention from trained model"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # Replace encoders with tracking versions
        self.time_encoder_tracker = AttentionTrackingTransformerEncoder(
            base_model.time_encoder)
        self.spec_encoder_tracker = AttentionTrackingTransformerEncoder(
            base_model.spec_encoder)

        # Monkey-patch
        self.base_model.time_encoder = self.time_encoder_tracker
        self.base_model.spec_encoder = self.spec_encoder_tracker

    def forward(self, time_x, spec_x):
        """Forward with attention tracking"""
        return self.base_model(time_x, spec_x)

    def get_attention_weights(self):
        """Get captured attention weights"""
        return {
            'time_self_attn': self.time_encoder_tracker.attention_weights,
            'spec_self_attn': self.spec_encoder_tracker.attention_weights
        }


class SelfAttentionVisualizer:
    """Visualize self-attention patterns"""

    def __init__(self, model_with_extractor, device='cuda'):
        self.device = device
        self.model = model_with_extractor.to(device)
        self.model.eval()

    def visualize_sample(self, time_x, spec_x, true_label, pred_label,
                        class_names, save_path='self_attention.png'):
        with torch.no_grad():
            time_x = time_x.to(self.device)
            spec_x = spec_x.to(self.device)

            output = self.model(time_x, spec_x)
            attn_dict = self.model.get_attention_weights()

            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        # Create figure
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 8, figure=fig, hspace=0.4, wspace=0.3)

        # 1. Time-domain signal
        ax1 = fig.add_subplot(gs[0, :4])
        self._plot_time_domain(ax1, time_x[0])

        # 2. Spectrogram
        ax2 = fig.add_subplot(gs[0, 4:])
        self._plot_spectrogram(ax2, spec_x[0])

        # 3. Prediction probabilities
        ax3 = fig.add_subplot(gs[1, :])
        self._plot_predictions(ax3, probs, true_label, pred_label, class_names)

        # 4-7. Time self-attention (last layer, 4 heads)
        if attn_dict['time_self_attn']:
            time_attn = attn_dict['time_self_attn'][-1][0].cpu().numpy()

            for i in range(min(time_attn.shape[0], 4)):
                ax = fig.add_subplot(gs[2:, i])
                im = ax.imshow(time_attn[i], cmap='RdYlBu_r', aspect='auto',
                              vmin=0, vmax=time_attn[i].max())
                ax.set_title(f'Time Self-Attn H{i+1}', fontsize=9, fontweight='bold', pad=5)
                ax.set_xlabel('Time Token', fontsize=7)
                ax.set_ylabel('Time Token', fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 8-11. Spec self-attention (last layer, 4 heads)
        if attn_dict['spec_self_attn']:
            spec_attn = attn_dict['spec_self_attn'][-1][0].cpu().numpy()

            for i in range(min(spec_attn.shape[0], 4)):
                ax = fig.add_subplot(gs[2:, i+4])
                im = ax.imshow(spec_attn[i], cmap='RdYlBu_r', aspect='auto',
                              vmin=0, vmax=spec_attn[i].max())
                ax.set_title(f'Spec Self-Attn H{i+1}', fontsize=9, fontweight='bold', pad=5)
                ax.set_xlabel('Spec Token', fontsize=7)
                ax.set_ylabel('Spec Token', fontsize=7)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle('Transformer Self-Attention Patterns (Trained Weights)',
                    fontsize=15, fontweight='bold', y=0.995)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
        plt.close()

    def _plot_time_domain(self, ax, time_data):
        time_np = time_data.cpu().numpy()
        channels = ['Ch1', 'Ch2', 'Ch3']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

        for i in range(min(time_np.shape[1], 3)):
            ax.plot(time_np[:, i], label=channels[i], color=colors[i],
                   linewidth=1.2, alpha=0.8)
        ax.set_title('Normalized Time-domain Signal', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time Sample', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def _plot_spectrogram(self, ax, spec_data):
        """修复版 Spectrogram 绘制 - 去DC + dB + percentile"""
        spec_np = spec_data.cpu().numpy().astype(np.float32)

        # 处理多通道
        if spec_np.ndim == 3:
            if spec_np.shape[0] == 3:
                spec_avg = spec_np.mean(0)
            else:
                spec_avg = spec_np[0]
        else:
            spec_avg = spec_np

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
        y_pos = np.arange(len(class_names))
        colors = ['#2ECC71' if i == pred_label else '#4682B4'
                 for i in range(len(class_names))]

        bars = ax.barh(y_pos, probs, color=colors, alpha=0.85,
                      edgecolor='white', linewidth=0.8)

        for i, (bar, prob) in enumerate(zip(bars, probs)):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{prob:.2f}', ha='left', va='center', fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel('Probability', fontsize=10)
        ax.set_title(f'Prediction | True: {true_label} | Pred: {pred_label}',
                    fontsize=11, fontweight='bold',
                    color='#2ECC71' if true_label == pred_label else '#E74C3C')
        ax.set_xlim(0, 1.15)
        ax.grid(True, axis='x', alpha=0.3)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str,
                       default=str(RESULTS_DIR / 'models' / 'exp3_v3_20260618_010348' / 'top2_Distill+LWF__384__T_2__alpha_0.3__20260618_010348.pth'))
    parser.add_argument('--output', type=str, default=str(RESULTS_DIR / 'self_attention_visualization'))
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("Loading data...")
    source_data = load_csv_data(
        str(DATASETS_DIR / "dataset2_1.0kW.csv"),
        window_size=1024, stride=128, spec_size=(128, 128),
        test_size=0.2, val_size=0.1, random_state=192
    )

    target_data = load_csv_data(
        str(DATASETS_DIR / "dataset2_3.0kW.csv"),
        window_size=1024, stride=128, spec_size=(128, 128),
        test_size=0.2, val_size=0.1, random_state=192
    )

    target_test_dataset = MultiModalFaultDataset(
        target_data['X_test_time'], target_data['X_test_spec'], target_data['y_test']
    )
    test_loader = DataLoader(target_test_dataset, batch_size=1, shuffle=False)

    fault_labels = ['Normal', 'IR007', 'IR014', 'IR021',
                   'OR007', 'OR014', 'OR021',
                   'B007', 'B014', 'B021',
                   'IR028', 'OR028', 'B028',
                   'IR040', 'OR040', 'B040']
    class_names = fault_labels

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

    # Wrap model
    model_with_extractor = SelfAttentionExtractor(base_model)

    # Create visualizer
    visualizer = SelfAttentionVisualizer(model_with_extractor, device)

    print(f"Generating {args.num_samples} self-attention visualizations...")

    sample_count = 0
    for time_x, spec_x, true_label in test_loader:
        if sample_count >= args.num_samples:
            break

        try:
            with torch.no_grad():
                time_x = time_x.to(device)
                spec_x = spec_x.to(device)
                output = model_with_extractor(time_x, spec_x)
                pred_label = output.argmax(1)[0].cpu().item()

            true_label_val = true_label.item()

            save_path = os.path.join(args.output,
                                   f'self_attn_sample_{sample_count+1}_true{true_label_val}_pred{pred_label}.png')

            visualizer.visualize_sample(
                time_x, spec_x, true_label_val, pred_label,
                class_names, save_path
            )

            sample_count += 1
            print(f"  [{sample_count}/{args.num_samples}] True: {true_label_val}, Pred: {pred_label}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nComplete! {sample_count} samples saved to {args.output}")


if __name__ == '__main__':
    main()
