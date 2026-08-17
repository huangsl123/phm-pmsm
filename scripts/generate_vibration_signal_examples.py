"""
生成准确的振动信号图用于论文
1. 3通道原始振动信号
2. 固定长度滑窗分段示意图
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

from _project_paths import DATASETS_DIR, PROJECT_ROOT

from data.data_processor_v2 import load_csv_data

# 加载真实数据
print("Loading real vibration data...")
BASE_DIR = str(DATASETS_DIR)
data = load_csv_data(
    f"{BASE_DIR}/dataset2_3.0kW.csv",
    window_size=1024, stride=128, spec_size=None,
    test_size=0.2, val_size=0.1, random_state=192
)

# 获取原始时域信号
X_time = data['X_train_time']  # (N, 1024, 3)
y_train = data['y_train']

print(f"Loaded {X_time.shape[0]} samples")
print(f"Shape: {X_time.shape}")


# ==================== 1. 3通道原始振动信号 ====================
def plot_raw_3channel_signal(save_path=str(PROJECT_ROOT / 'paper_figures' / 'raw_3channel_signal.png')):
    """绘制真实的3通道振动信号"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    channels = ['Channel 1 (X-axis)', 'Channel 2 (Y-axis)', 'Channel 3 (Z-axis)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 选择一个典型样本（例如：故障样本）
    sample_idx = 100
    signal = X_time[sample_idx]  # (1024, 3)

    for i in range(3):
        ax = axes[i]
        ax.plot(signal[:, i], color=colors[i], linewidth=1.2, alpha=0.9)
        ax.set_ylabel('Amplitude', fontsize=11)
        ax.set_title(channels[i], fontsize=12, fontweight='bold', loc='left')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim([signal[:, i].min() - 0.1, signal[:, i].max() + 0.1])

    axes[-1].set_xlabel('Time Sample', fontsize=12)
    plt.suptitle('Raw 3-Channel Vibration Signal', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


# ==================== 2. 固定长度滑窗分段示意图 ====================
def plot_sliding_window(save_path=str(PROJECT_ROOT / 'paper_figures' / 'sliding_window_segmentation.png')):
    """绘制固定长度滑窗分段示意图"""
    fig, ax = plt.subplots(figsize=(14, 5))

    # 使用一段较长信号
    long_signal = X_time[0, :, 0]  # 取第一个样本的第一个通道

    window_size = 200
    stride = 50
    n_windows = 8

    # 绘制原始信号
    ax.plot(long_signal, color='#888', linewidth=1.5, alpha=0.6, label='Original Signal')

    # 绘制滑窗
    colors_seg = plt.cm.viridis(np.linspace(0, 1, n_windows))

    for i in range(n_windows):
        start = i * stride
        end = start + window_size

        if end <= len(long_signal):
            # 绘制窗口区域
            ax.axvspan(start, end, alpha=0.3, color=colors_seg[i], label=f'Window {i+1}' if i < 3 else '')
            # 绘制窗口内的信号（高亮）
            ax.plot(range(start, end), long_signal[start:end],
                   color=colors_seg[i], linewidth=2, alpha=0.8)

    # 标注窗口参数
    ax.annotate('', xy=(window_size, long_signal[window_size]),
                xytext=(0, long_signal[window_size]),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax.text(window_size/2, long_signal[window_size] + 0.5,
           f'Window Size = {window_size}',
           ha='center', fontsize=11, fontweight='bold', color='red',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red'))

    ax.annotate('', xy=(stride, long_signal.min() - 0.3),
                xytext=(0, long_signal.min() - 0.3),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(stride/2, long_signal.min() - 0.5,
           f'Stride = {stride}',
           ha='center', fontsize=11, fontweight='bold', color='blue',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='blue'))

    ax.set_xlabel('Time Sample', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Fixed-Length Sliding Window Segmentation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


# ==================== 3. 单窗口时域信号 ====================
def plot_single_window_signal(save_path=str(PROJECT_ROOT / 'paper_figures' / 'single_window_time_domain.png')):
    """绘制单个窗口的时域信号（3通道）"""
    fig, ax = plt.subplots(figsize=(12, 4))

    channels = ['Ch1 (X)', 'Ch2 (Y)', 'Ch3 (Z)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 选择一个样本
    signal = X_time[50]  # (1024, 3)
    time_axis = np.arange(1024)

    for i in range(3):
        ax.plot(time_axis, signal[:, i], label=channels[i],
               color=colors[i], linewidth=1.5, alpha=0.85)

    ax.set_xlabel('Time Sample', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.set_title('Time-Domain Signal (Single Window)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


# ==================== 4. 故障类型对比图 ====================
def plot_fault_comparison(save_path=str(PROJECT_ROOT / 'paper_figures' / 'fault_type_comparison.png')):
    """绘制不同故障类型的信号对比"""
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))

    fault_types = ['Normal', 'IR', 'OR', 'B']
    fault_labels = {
        0: 'Normal',
        1: 'IR007',
        2: 'IR014',
        4: 'OR007',
        8: 'B007'
    }

    # 找到不同类别的样本索引
    samples_to_plot = {}
    for label, name in fault_labels.items():
        idx = np.where(y_train == label)[0]
        if len(idx) > 0:
            samples_to_plot[name] = X_time[idx[0]]

    # 绘制
    for i, (name, signal) in enumerate(samples_to_plot.items()):
        if i >= 9:
            break
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        # 只画第一通道
        ax.plot(signal[:, 0], color='#1f77b4', linewidth=1.2, alpha=0.85)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Fault Type Comparison (Time-Domain Signals)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    plt.close()


# ==================== 主函数 ====================
if __name__ == '__main__':
    os.makedirs('paper_figures', exist_ok=True)

    print("=" * 60)
    print("Generating Accurate Vibration Signal Figures")
    print("=" * 60)

    print("\n1. Generating raw 3-channel signal...")
    plot_raw_3channel_signal()

    print("\n2. Generating sliding window segmentation...")
    plot_sliding_window()

    print("\n3. Generating single window time-domain signal...")
    plot_single_window_signal()

    print("\n4. Generating fault type comparison...")
    plot_fault_comparison()

    print("\n" + "=" * 60)
    print("All figures saved to: paper_figures/")
    print("=" * 60)
