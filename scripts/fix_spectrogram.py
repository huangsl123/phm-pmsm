"""
修复 Spectrogram 可视化问题
1. 使用 dB 缩放而非简单 log
2. 使用 percentile 裁剪避免极端低频主导
3. 修正坐标轴 (横轴时间, 纵轴频率)
4. 可选去除 DC/低频分量
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_spectrogram_fixed(ax, spectrogram, use_db=True, remove_dc=True,
                           percentile_range=(5, 99), cmap='viridis'):
    """
    修复后的 spectrogram 绘制函数

    Args:
        ax: matplotlib axis
        spectrogram: 2D numpy array 或 torch tensor
        use_db: 是否使用 dB 缩放 (推荐 True)
        remove_dc: 是否去除 DC/极低频分量 (推荐 True)
        percentile_range: 用于裁剪的百分位数范围
        cmap: 颜色映射
    """
    spec_np = spectrogram.cpu().numpy() if hasattr(spectrogram, 'cpu') else spectrogram
    spec_np = spec_np.astype(np.float32)

    # 处理多通道 - 使用 max 而非 mean 以保留更多细节
    if spec_np.ndim == 3:
        if spec_np.shape[0] == 3:
            spec_np = np.max(spec_np, axis=0)  # 使用 max 保留细节
        else:
            spec_np = spec_np[0]

    print(f"  [调试] spec_np shape: {spec_np.shape}")
    print(f"  [调试] 原始 min/max: {spec_np.min():.6f} / {spec_np.max():.6f}")

    # 去除 DC 分量 (最低频的一行/列)
    if remove_dc:
        # 假设是 (freq, time) 格式
        spec_np[0, :] = spec_np[1:5, :].mean(axis=0)  # 用邻近低频的平均值替代 DC
        print(f"  [调试] 已去除 DC 分量")

    # dB 缩放
    if use_db:
        # 添加小常数避免 log(0)
        spec_db = 10 * np.log10(spec_np + 1e-8)
        print(f"  [调试] dB min/max: {spec_db.min():.2f} / {spec_db.max():.2f} dB")

        # 使用 percentile 裁剪，避免极端值主导颜色映射
        vmin, vmax = np.percentile(spec_db, percentile_range)
        spec_plot = np.clip(spec_db, vmin, vmax)
        print(f"  [调试] percentile裁剪: [{vmin:.2f}, {vmax:.2f}] dB")
    else:
        spec_plot = spec_np
        vmin, vmax = np.percentile(spec_plot, percentile_range)
        spec_plot = np.clip(spec_plot, vmin, vmax)

    # 归一化到 [0, 1]
    spec_norm = (spec_plot - vmin) / (vmax - vmin + 1e-8)

    print(f"  [调试] 最终归一化范围: [{spec_norm.min():.3f}, {spec_norm.max():.3f}]")

    # 绘制 - 横轴时间, 纵轴频率
    im = ax.imshow(spec_norm, cmap=cmap, aspect='auto', origin='lower')

    # 修正坐标轴标签
    ax.set_xlabel('Time Frame', fontsize=11)
    ax.set_ylabel('Frequency Bin', fontsize=11)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if use_db:
        cbar.set_label('Power (dB)', fontsize=10)
    else:
        cbar.set_label('Normalized Power', fontsize=10)

    return im


# 测试代码 - 使用模拟数据
if __name__ == '__main__':
    # 创建模拟的 spectrogram 数据 (模拟真实信号特征)
    n_freq, n_time = 64, 100

    # 模拟有结构的频谱图
    spec = np.zeros((n_freq, n_time))

    # 添加一些随时间变化的频率成分
    for t in range(n_time):
        freq_center = 20 + 15 * np.sin(t * 0.1)
        for f in range(n_freq):
            dist = abs(f - freq_center)
            spec[f, t] = 0.8 * np.exp(-dist / 5) + 0.1 * np.random.random()

    # 添加较强的低频成分 (模拟真实情况)
    spec[:10, :] *= 3  # 低频能量更强

    # 添加一些高频瞬态
    spec[30:50, 40:60] += 0.5 * np.random.random((20, 20))

    print("=" * 60)
    print("测试 Spectrogram 可视化修复")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 原始方法 (有问题)
    ax1 = axes[0, 0]
    spec_log = np.log10(spec + 1e-6)
    spec_norm_old = (spec_log - spec_log.min()) / (spec_log.max() - spec_log.min())
    im1 = ax1.imshow(spec_norm_old, cmap='viridis', aspect='auto', origin='lower')
    ax1.set_title('原始方法 (简单log10归一化)', fontweight='bold')
    ax1.set_xlabel('Frequency Bin')  # 错误的标签
    ax1.set_ylabel('Time Frame')
    plt.colorbar(im1, ax=ax1)

    # 2. 修复方法 - dB + percentile
    ax2 = axes[0, 1]
    print("\n[图2] dB + percentile 方法:")
    plot_spectrogram_fixed(ax2, spec, use_db=True, remove_dc=False)
    ax2.set_title('修复方法 1: dB + percentile裁剪', fontweight='bold')

    # 3. 修复方法 - 去DC + dB
    ax3 = axes[1, 0]
    print("\n[图3] 去DC + dB 方法:")
    plot_spectrogram_fixed(ax3, spec, use_db=True, remove_dc=True)
    ax3.set_title('修复方法 2: 去DC + dB', fontweight='bold')

    # 4. 最佳方法
    ax4 = axes[1, 1]
    print("\n[图4] 最佳组合方法:")
    plot_spectrogram_fixed(ax4, spec, use_db=True, remove_dc=True,
                           percentile_range=(2, 98))
    ax4.set_title('推荐方法: 去DC + dB + 更激进裁剪', fontweight='bold')

    plt.tight_layout()
    plt.savefig('spectrogram_fix_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: spectrogram_fix_comparison.png")
    plt.show()
