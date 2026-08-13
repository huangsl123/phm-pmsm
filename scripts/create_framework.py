"""
创建专业总框架图 - 包含四个模块和嵌入式可视化
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Wedge
from matplotlib.patches import ConnectionPatch
from matplotlib import font_manager
import matplotlib.patheffects as path_effects

from _project_paths import PROJECT_ROOT

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'SimHei']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['patch.linewidth'] = 1.5

# 配色方案 - 美化版
COLORS = {
    'module1': '#E8F4F8',      # 浅蓝 - 数据预处理
    'module2': '#FFF4E6',      # 浅橙 - 模型核心
    'module3': '#F0F8E8',      # 浅绿 - 预训练
    'module4': '#F8E8F8',      # 浅紫 - 评估
    'border1': '#4A90A4',      # 模块1边框
    'border2': '#D4A574',      # 模块2边框
    'border3': '#7AA44A',      # 模块3边框
    'border4': '#A44A84',      # 模块4边框
    'arrow': '#666666',        # 箭头颜色
    'text': '#333333',         # 文本颜色
    'highlight': '#FF6B6B',    # 高亮颜色
    'gradient1': '#6CB4EE',     # 渐变色1
    'gradient2': '#87CEEB',     # 渐变色2
}

# 文件路径
BASE_DIR = str(PROJECT_ROOT)
training_curves_path = os.path.join(BASE_DIR, "result_upgrade", "visualizations", "exp3_v3_20260618_010348", "top2_Distill+LWF__384__T_2__alpha_0.3__training_curves.png")
confusion_matrix_path = os.path.join(BASE_DIR, "result_upgrade", "visualizations", "exp3_v3_20260618_010348", "top2_Distill+LWF__384__T_2__alpha_0.3__target_confusion_matrix.png")
output_path = os.path.join(BASE_DIR, "PHM_会议论文_中英文与图", "overall_framework_5_beautified.png")

def add_gradient_background(ax, bbox, colors, alpha=0.3):
    """添加渐变背景"""
    x1, y1, x2, y2 = bbox
    n = 100
    for i in range(n):
        ratio = i / n
        r = int(colors[0][0] * (1-ratio) + colors[1][0] * ratio)
        g = int(colors[0][1] * (1-ratio) + colors[1][1] * ratio)
        b = int(colors[0][2] * (1-ratio) + colors[1][2] * ratio)
        rect = Rectangle((x1, y1 + i*(y2-y1)/n), x2-x1, (y2-y1)/n,
                         facecolor=(r/255, g/255, b/255, alpha),
                         edgecolor='none')
        ax.add_patch(rect)

def add_rounded_box(ax, x, y, width, height, facecolor, edgecolor,
                   text, text_color='black', fontsize=11, bold=False,
                   boxstyle='round,pad=0.1', linewidth=2):
    """添加圆角矩形框"""
    box = FancyBboxPatch((x, y), width, height,
                         boxstyle=boxstyle,
                         facecolor=facecolor,
                         edgecolor=edgecolor,
                         linewidth=linewidth)
    ax.add_patch(box)

    # 添加文本
    fontweight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text,
           ha='center', va='center',
           fontsize=fontsize, fontweight=fontweight,
           color=text_color)

def add_arrow(ax, x1, y1, x2, y2, color=COLORS['arrow'],
              head_width=0.03, head_length=0.05, linewidth=2):
    """添加箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=f'->,head_length={head_length},head_width={head_width}',
                           color=color, linewidth=linewidth, zorder=10)
    ax.add_patch(arrow)

def create_transformer_block(ax, x, y, width, height, title, num_layers=3):
    """创建Transformer块的可视化"""
    # 外框
    box = FancyBboxPatch((x, y), width, height,
                        boxstyle='round,pad=0.02',
                        facecolor='#FFF8DC', edgecolor='#DAA520',
                        linewidth=1.5)
    ax.add_patch(box)

    # 标题
    ax.text(x + width/2, y + height - 0.03, title,
           ha='center', va='center',
           fontsize=8, fontweight='bold', color='#8B4513')

    # Transformer layers
    layer_height = (height - 0.06) / num_layers
    for i in range(num_layers):
        ly = y + 0.02 + i * layer_height
        # LayerNorm + Attention + MLP
        sub_box = Rectangle((x + 0.02, ly), width - 0.04, layer_height - 0.01,
                           facecolor='#F5DEB3', edgecolor='#CD853F',
                           linewidth=0.8)
        ax.add_patch(sub_box)

        # 添加细节
        ax.text(x + width/2, ly + layer_height/2 - 0.01,
               f'Layer {i+1}',
               ha='center', va='center',
               fontsize=6, color='#8B4513')

def create_cross_attention_visual(ax, x, y, size):
    """创建跨模态注意力机制可视化"""
    # 中心融合节点
    center = Circle((x + size/2, y + size/2), size/6,
                   facecolor='#FFD700', edgecolor='#DAA520',
                   linewidth=2, zorder=5)
    ax.add_patch(center)
    ax.text(x + size/2, y + size/2, 'Fusion\nFusion',
           ha='center', va='center', fontsize=6, fontweight='bold',
           color='#8B4513')

    # 时域到频域箭头
    arc1 = Wedge((x + size/2, y + size/2), size/2.5, 150, 210,
                width=0.02, facecolor='#4169E1', edgecolor='#4169E1', alpha=0.7)
    ax.add_patch(arc1)

    # 频域到时域箭头
    arc2 = Wedge((x + size/2, y + size/2), size/2.5, 330, 30,
                width=0.02, facecolor='#DC143C', edgecolor='#DC143C', alpha=0.7)
    ax.add_patch(arc2)

    # 自注意力
    circle2 = Circle((x + size/2, y + size/2), size/3.5,
                    facecolor='none', edgecolor='#32CD32',
                    linewidth=1.5, linestyle='--')
    ax.add_patch(circle2)

def create_framework():
    """创建完整的框架图"""
    # 创建大图
    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor('white')

    # 整体布局: 4个模块横向排列
    module_width = 0.22
    module_gap = 0.02
    start_x = 0.03
    total_width = 4 * module_width + 3 * module_gap + 2 * start_x

    # 创建子图网格 - 更灵活的布局
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1],
                          left=0.02, right=0.98,
                          top=0.95, bottom=0.05,
                          wspace=0.03, hspace=0.05)

    # ==================== 模块1: 数据预处理与多模态构建 ====================
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    ax1.set_title("Module 1:\nData Preprocessing &\nMulti-modal Construction",
                 fontsize=12, fontweight='bold', color=COLORS['border1'],
                 pad=10)

    # 模块1背景
    add_gradient_background(ax1, (0.05, 0.1), ((200, 230, 255), (220, 240, 255)), alpha=0.4)

    # 源域和目标域输入框
    add_rounded_box(ax1, 0.15, 0.75, 0.35, 0.12,
                   '#E6F3FF', COLORS['border1'],
                   'Source Domain\n1.0 kW Data', fontsize=9, bold=True)
    add_rounded_box(ax1, 0.55, 0.75, 0.35, 0.12,
                   '#FFE6F0', COLORS['border1'],
                   'Target Domain\n3.0 kW Data', fontsize=9, bold=True)

    # 数据处理流程
    add_rounded_box(ax1, 0.15, 0.55, 0.75, 0.08,
                   '#F0F8FF', COLORS['border1'],
                   'Multi-channel Waveform Representation', fontsize=8)
    add_rounded_box(ax1, 0.15, 0.42, 0.75, 0.08,
                   '#F0F8FF', COLORS['border1'],
                   'Sliding Window Sampling', fontsize=8)

    # 双模态分支
    add_rounded_box(ax1, 0.15, 0.25, 0.35, 0.12,
                   '#E8FFE8', '#2E8B57',
                   'Time-domain\nSignal', fontsize=9, bold=True)
    add_rounded_box(ax1, 0.55, 0.25, 0.35, 0.12,
                   '#FFE8E8', '#CD5C5C',
                   'Spectrogram\n(STFT)', fontsize=9, bold=True)

    # 连接箭头
    add_arrow(ax1, 0.325, 0.75, 0.325, 0.63, color=COLORS['border1'])
    add_arrow(ax1, 0.725, 0.75, 0.725, 0.63, color=COLORS['border1'])
    add_arrow(ax1, 0.325, 0.55, 0.325, 0.42, color=COLORS['arrow'])
    add_arrow(ax1, 0.725, 0.55, 0.725, 0.42, color=COLORS['arrow'])
    add_arrow(ax1, 0.325, 0.42, 0.325, 0.37, color=COLORS['arrow'])
    add_arrow(ax1, 0.725, 0.42, 0.725, 0.37, color=COLORS['arrow'])

    # ==================== 模块2: 多模态CrossViT诊断模型 ====================
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    ax2.set_title("Module 2:\nCrossViT Fault\nDiagnosis Model",
                 fontsize=12, fontweight='bold', color=COLORS['border2'],
                 pad=10)

    # 模块2背景
    add_gradient_background(ax2, (0.05, 0.1), ((255, 240, 200), (255, 250, 220)), alpha=0.4)

    # 双编码器
    create_transformer_block(ax2, 0.08, 0.55, 0.38, 0.35,
                           'Time-domain Encoder\n(1D Patch Embedding)')
    create_transformer_block(ax2, 0.54, 0.55, 0.38, 0.35,
                           'Spectrogram Encoder\n(2D Patch Embedding)')

    # 跨模态注意力融合
    create_cross_attention_visual(ax2, 0.25, 0.35, 0.5)

    # 分类头
    add_rounded_box(ax2, 0.25, 0.1, 0.5, 0.12,
                   '#FFECB3', '#FFA000',
                   'Classifier Head\n(MLP + Softmax)', fontsize=9, bold=True)

    # 内部箭头
    add_arrow(ax2, 0.27, 0.55, 0.35, 0.45, color='#DAA520')
    add_arrow(ax2, 0.73, 0.55, 0.65, 0.45, color='#DAA520')
    add_arrow(ax2, 0.5, 0.35, 0.5, 0.22, color='#FF6B6B', linewidth=2.5)

    # ==================== 模块3: 源域预训练 ====================
    ax3 = fig.add_subplot(gs[:, 2])
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title("Module 3:\nSource-domain\nPre-training",
                 fontsize=12, fontweight='bold', color=COLORS['border3'],
                 pad=10)

    # 模块3背景
    add_gradient_background(ax3, (0.1, 0.2), ((220, 255, 220), (240, 255, 240)), alpha=0.5)

    # 预训练过程
    add_rounded_box(ax3, 0.15, 0.65, 0.7, 0.1,
                   '#E8F5E9', COLORS['border3'],
                   'Pre-training on Source Domain', fontsize=10, bold=True)

    # 训练过程可视化
    for i in range(3):
        add_rounded_box(ax3, 0.15 + i*0.25, 0.5, 0.2, 0.08,
                       '#C8E6C9', '#66BB6A',
                       f'Epoch {i+1}', fontsize=8)

    # 参数保存
    add_rounded_box(ax3, 0.15, 0.35, 0.7, 0.1,
                   '#DCEDC8', COLORS['border3'],
                   'Save Teacher Model / Initial Parameters', fontsize=9)

    # 知识保持标注
    add_rounded_box(ax3, 0.15, 0.15, 0.7, 0.12,
                   '#F1F8E9', '#8BC34A',
                   'Knowledge Reference\nfor Anti-forgetting', fontsize=9, bold=True)

    # 箭头
    add_arrow(ax3, 0.5, 0.65, 0.5, 0.58, color=COLORS['border3'])
    add_arrow(ax3, 0.27, 0.5, 0.42, 0.5, color='#8BC34A', head_width=0.02)
    add_arrow(ax3, 0.52, 0.5, 0.67, 0.5, color='#8BC34A', head_width=0.02)
    add_arrow(ax3, 0.5, 0.42, 0.5, 0.45, color=COLORS['border3'])

    # ==================== 模块4: 评估与可视化输出 ====================
    ax4_top = fig.add_subplot(gs[0, 3])
    ax4_bottom = fig.add_subplot(gs[1, 3])

    # 移除坐标轴
    for ax in [ax4_top, ax4_bottom]:
        ax.axis('off')

    # 整体标题
    fig.text(0.89, 0.96, "Module 4:\nEvaluation &\nVisualization",
            ha='center', va='top', fontsize=12, fontweight='bold',
            color=COLORS['border4'])

    # 加载并嵌入训练曲线图
    if os.path.exists(training_curves_path):
        img_tc = plt.imread(training_curves_path)
        ax4_top.imshow(img_tc)
        ax4_top.set_title("Training Curves\n(Loss & Accuracy Trends)",
                         fontsize=9, fontweight='bold', color=COLORS['border4'], pad=5)
    else:
        print(f"Warning: 训练曲线图不存在: {training_curves_path}")
        # 占位
        ax4_top.text(0.5, 0.5, 'Training Curves\n(Image not found)',
                    ha='center', va='center', fontsize=10, style='italic')

    # 加载并嵌入混淆矩阵图
    if os.path.exists(confusion_matrix_path):
        img_cm = plt.imread(confusion_matrix_path)
        ax4_bottom.imshow(img_cm)
        ax4_bottom.set_title("Target Domain\nConfusion Matrix",
                             fontsize=9, fontweight='bold', color=COLORS['border4'], pad=5)
    else:
        print(f"Warning: 混淆矩阵图不存在: {confusion_matrix_path}")
        ax4_bottom.text(0.5, 0.5, 'Confusion Matrix\n(Image not found)',
                       ha='center', va='center', fontsize=10, style='italic')

    # ==================== 模块间连接箭头 ====================
    # 模块1到模块2
    ax1.annotate('', xy=(0, 0.5), xytext=(1, 0.5),
                annotation_clip=False,
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                              lw=2.5, mutation_scale=20))

    # 模块2到模块3
    ax2.annotate('', xy=(0, 0.5), xytext=(1, 0.5),
                annotation_clip=False,
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                              lw=2.5, mutation_scale=20))

    # 模块3到模块4
    ax3.annotate('', xy=(0, 0.5), xytext=(1, 0.5),
                annotation_clip=False,
                arrowprops=dict(arrowstyle='->', color=COLORS['arrow'],
                              lw=2.5, mutation_scale=20))

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 框架图已保存到: {output_path}")

    return output_path

def create_enhanced_framework():
    """创建增强版框架图 - 更美观的设计"""
    fig = plt.figure(figsize=(18, 10), dpi=150)
    fig.patch.set_facecolor('#FAFAFA')

    # 使用更复杂的网格布局
    gs = fig.add_gridspec(3, 5, height_ratios=[0.15, 0.7, 0.15],
                          left=0.01, right=0.99,
                          top=0.97, bottom=0.03,
                          wspace=0.02, hspace=0.01)

    # ==================== 标题区域 ====================
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis('off')
    title_ax.text(0.5, 0.7, 'Overall Framework of Multi-modal CrossViT for Cross-power Fault Diagnosis',
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 color='#2C3E50')
    title_ax.text(0.5, 0.3, 'Data Preprocessing → Cross-modal Encoding → Attention Fusion → Evaluation',
                 ha='center', va='center', fontsize=11, color='#7F8C8D', style='italic')

    # ==================== 模块1: 数据预处理 (左下) ====================
    ax_m1 = fig.add_subplot(gs[1, 0])
    ax_m1.set_xlim(0, 1)
    ax_m1.set_ylim(0, 1)
    ax_m1.axis('off')

    # 模块背景卡
    m1_bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle='round,pad=0.01',
                          facecolor='#EBF5FB', edgecolor='#5DADE2',
                          linewidth=2.5)
    ax_m1.add_patch(m1_bg)

    # 模块标题
    ax_m1.text(0.5, 0.95, 'Data Preprocessing\n& Multi-modal',
              ha='center', va='top', fontsize=11, fontweight='bold',
              color='#2874A6')

    # 输入数据框
    add_rounded_box(ax_m1, 0.1, 0.78, 0.35, 0.12,
                   '#D6EAF8', '#3498DB',
                   'Source\n1.0kW', fontsize=9, bold=True,
                   boxstyle='round,pad=0.08')
    add_rounded_box(ax_m1, 0.55, 0.78, 0.35, 0.12,
                   '#FADBD8', '#E74C3C',
                   'Target\n3.0kW', fontsize=9, bold=True,
                   boxstyle='round,pad=0.08')

    # 处理步骤
    steps = ['Waveform\nRep', 'Sliding\nWindow', 'Time\nSignal', 'Spectrogram\n(STFT)']
    step_colors = ['#AED6F1', '#A9DFBF', '#F9E79F', '#F5B7B1']
    for i, (step, color) in enumerate(zip(steps, step_colors)):
        row = i // 2
        col = i % 2
        add_rounded_box(ax_m1, 0.1 + col*0.45, 0.65 - row*0.18, 0.38, 0.14,
                       color, '#566573', step, fontsize=8,
                       boxstyle='round,pad=0.05')

    # ==================== 模块2: CrossViT模型 ====================
    ax_m2 = fig.add_subplot(gs[1, 1])
    ax_m2.set_xlim(0, 1)
    ax_m2.set_ylim(0, 1)
    ax_m2.axis('off')

    m2_bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle='round,pad=0.01',
                          facecolor='#FEF5E7', edgecolor='#F39C12',
                          linewidth=2.5)
    ax_m2.add_patch(m2_bg)

    ax_m2.text(0.5, 0.95, 'CrossViT\nModel',
              ha='center', va='top', fontsize=11, fontweight='bold',
              color='#B7950B')

    # 双编码器
    create_transformer_block(ax_m2, 0.05, 0.55, 0.42, 0.35,
                           'Time Encoder\n1D-Conv', num_layers=2)
    create_transformer_block(ax_m2, 0.53, 0.55, 0.42, 0.35,
                           'Spec Encoder\n2D-Conv', num_layers=2)

    # 跨模态融合 - 美化版
    fusion_y = 0.35
    # 时域特征
    ft_box = FancyBboxPatch((0.05, fusion_y), 0.42, 0.1,
                           boxstyle='round,pad=0.02',
                           facecolor='#A9DFBF', edgecolor='#27AE60',
                           linewidth=1.5)
    ax_m2.add_patch(ft_box)
    ax_m2.text(0.26, fusion_y + 0.05, 'Ft\nFeatures',
             ha='center', va='center', fontsize=7, fontweight='bold')

    # 频域特征
    fs_box = FancyBboxPatch((0.53, fusion_y), 0.42, 0.1,
                           boxstyle='round,pad=0.02',
                           facecolor='#F5B7B1', edgecolor='#C0392B',
                           linewidth=1.5)
    ax_m2.add_patch(fs_box)
    ax_m2.text(0.74, fusion_y + 0.05, 'Fs\nFeatures',
             ha='center', va='center', fontsize=7, fontweight='bold')

    # 融合模块
    fusion_box = FancyBboxPatch((0.3, 0.18), 0.4, 0.12,
                               boxstyle='round,pad=0.02',
                               facecolor='#F9E79F', edgecolor='#F1C40F',
                               linewidth=2)
    ax_m2.add_patch(fusion_box)
    ax_m2.text(0.5, 0.24, 'Cross-modal\nFusion',
             ha='center', va='center', fontsize=8, fontweight='bold', color='#B7950B')

    # 分类头
    cls_box = FancyBboxPatch((0.3, 0.02), 0.4, 0.12,
                             boxstyle='round,pad=0.02',
                             facecolor='#D2B4DE', edgecolor='#8E44AD',
                             linewidth=2)
    ax_m2.add_patch(cls_box)
    ax_m2.text(0.5, 0.08, 'Classifier\nSoftmax',
             ha='center', va='center', fontsize=8, fontweight='bold', color='#6C3483')

    # 连接箭头
    add_arrow(ax_m2, 0.26, fusion_y, 0.4, 0.3, color='#27AE60', head_width=0.02)
    add_arrow(ax_m2, 0.74, fusion_y, 0.6, 0.3, color='#C0392B', head_width=0.02)
    add_arrow(ax_m2, 0.5, 0.18, 0.5, 0.14, color='#8E44AD', head_width=0.025, linewidth=2.5)

    # ==================== 模块3: 源域预训练 ====================
    ax_m3 = fig.add_subplot(gs[1, 2])
    ax_m3.set_xlim(0, 1)
    ax_m3.set_ylim(0, 1)
    ax_m3.axis('off')

    m3_bg = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                          boxstyle='round,pad=0.01',
                          facecolor='#E9F7EF', edgecolor='#27AE60',
                          linewidth=2.5)
    ax_m3.add_patch(m3_bg)

    ax_m3.text(0.5, 0.95, 'Source-domain\nPre-training',
              ha='center', va='top', fontsize=11, fontweight='bold',
              color='#1E8449')

    # 预训练流程
    add_rounded_box(ax_m3, 0.1, 0.75, 0.8, 0.1,
                   '#D5F4E6', '#27AE60',
                   'Source Training', fontsize=9, bold=True,
                   boxstyle='round,pad=0.05')

    # Epoch进度
    for i in range(4):
        alpha = 0.4 + i*0.15
        epoch_box = FancyBboxPatch((0.1 + i*0.21, 0.6), 0.18, 0.1,
                                  boxstyle='round,pad=0.02',
                                  facecolor=(0.4, 0.8, 0.4, alpha),
                                  edgecolor='#27AE60', linewidth=1.2)
        ax_m3.add_patch(epoch_box)
        ax_m3.text(0.19 + i*0.21, 0.65, f'E{i+1}',
                 ha='center', va='center', fontsize=7, fontweight='bold')

    # 保存模型
    add_rounded_box(ax_m3, 0.1, 0.42, 0.8, 0.1,
                   '#A9DFBF', '#2ECC71',
                   'Save Teacher Model', fontsize=9,
                   boxstyle='round,pad=0.05')

    # 知识蒸馏说明
    add_rounded_box(ax_m3, 0.1, 0.25, 0.8, 0.15,
                   '#ABEBC6', '#58D68D',
                   'Knowledge Distillation\n& Source Memory', fontsize=8,
                   boxstyle='round,pad=0.05')

    # 箭头
    add_arrow(ax_m3, 0.5, 0.75, 0.5, 0.7, color='#27AE60', head_width=0.02)
    add_arrow(ax_m3, 0.5, 0.52, 0.5, 0.47, color='#27AE60', head_width=0.02)
    add_arrow(ax_m3, 0.5, 0.35, 0.5, 0.4, color='#27AE60', head_width=0.02)

    # ==================== 模块4: 评估与可视化 (拆分上下两部分) ====================
    # 上半部分 - 训练曲线
    ax_m4_top = fig.add_subplot(gs[1, 3])
    ax_m4_top.axis('off')

    m4_bg_top = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                               boxstyle='round,pad=0.01',
                               facecolor='#F5EEF8', edgecolor='#8E44AD',
                               linewidth=2.5)
    ax_m4_top.add_patch(m4_bg_top)

    ax_m4_top.text(0.5, 0.92, 'Training\nCurves',
                  ha='center', va='top', fontsize=10, fontweight='bold',
                  color='#6C3483')

    # 嵌入训练曲线图
    if os.path.exists(training_curves_path):
        img_tc = plt.imread(training_curves_path)
        ax_m4_top.imshow(img_tc, extent=[0.15, 0.85, 0.1, 0.85], aspect='auto')
    else:
        ax_m4_top.text(0.5, 0.5, 'Image not found',
                     ha='center', va='center', fontsize=9, style='italic', color='red')

    # 下半部分 - 混淆矩阵
    ax_m4_bottom = fig.add_subplot(gs[1, 4])
    ax_m4_bottom.axis('off')

    m4_bg_btm = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                              boxstyle='round,pad=0.01',
                              facecolor='#F5EEF8', edgecolor='#C0392B',
                              linewidth=2.5)
    ax_m4_bottom.add_patch(m4_bg_btm)

    ax_m4_bottom.text(0.5, 0.92, 'Confusion\nMatrix',
                     ha='center', va='top', fontsize=10, fontweight='bold',
                     color='#922B21')

    # 嵌入混淆矩阵图
    if os.path.exists(confusion_matrix_path):
        img_cm = plt.imread(confusion_matrix_path)
        ax_m4_bottom.imshow(img_cm, extent=[0.15, 0.85, 0.1, 0.85], aspect='auto')
    else:
        ax_m4_bottom.text(0.5, 0.5, 'Image not found',
                         ha='center', va='center', fontsize=9, style='italic', color='red')

    # ==================== 底部说明区 ====================
    bottom_ax = fig.add_subplot(gs[2, :])
    bottom_ax.axis('off')

    # 流程说明
    flow_text = 'Data Flow: Source/Target → Multi-modal (Time+Spec) → Dual-branch Encoding → Cross-modal Fusion → Classification → Visualization'
    bottom_ax.text(0.5, 0.6, flow_text,
                 ha='center', va='center', fontsize=10,
                 color='#566573', style='italic')

    # 关键技术标签
    keywords = ['Multi-modal', 'CrossViT', 'Cross-attention', 'Transfer Learning', 'Knowledge Distillation']
    keyword_colors = ['#3498DB', '#E67E22', '#27AE60', '#9B59B6', '#E74C3C']
    x_positions = np.linspace(0.15, 0.85, len(keywords))
    for kw, x, col in zip(keywords, x_positions, keyword_colors):
        bottom_ax.text(x, 0.2, kw, ha='center', va='center',
                      fontsize=9, fontweight='bold',
                      bbox=dict(boxstyle='round,pad=0.3',
                               facecolor=col, edgecolor='none',
                               alpha=0.3),
                      color=col)

    # ==================== 模块间连接箭头 ====================
    # 使用ConnectionPatch创建美观的连接
    # M1 -> M2
    conn1 = ConnectionPatch(xyA=(0, 0.5), xyB=(1, 0.5),
                           coordsA=ax_m1.transData, coordsB=ax_m2.transData,
                           arrowstyle='->,head_width=0.4,head_length=0.4',
                           color='#566573', linewidth=3, zorder=10,
                           shrinkA=5, shrinkB=5)
    fig.add_artist(conn1)

    # M2 -> M3
    conn2 = ConnectionPatch(xyA=(0, 0.5), xyB=(1, 0.5),
                           coordsA=ax_m2.transData, coordsB=ax_m3.transData,
                           arrowstyle='->,head_width=0.4,head_length=0.4',
                           color='#566573', linewidth=3, zorder=10,
                           shrinkA=5, shrinkB=5)
    fig.add_artist(conn2)

    # M3 -> M4
    conn3 = ConnectionPatch(xyA=(0, 0.5), xyB=(1, 0.5),
                           coordsA=ax_m3.transData, coordsB=ax_m4_top.transData,
                           arrowstyle='->,head_width=0.4,head_length=0.4',
                           color='#566573', linewidth=3, zorder=10,
                           shrinkA=5, shrinkB=5)
    fig.add_artist(conn3)

    # 保存
    output_enhanced = output_path.replace('.png', '_enhanced.png')
    plt.savefig(output_enhanced, dpi=300, bbox_inches='tight', facecolor='#FAFAFA')
    print(f"[DONE] 增强版框架图已保存到: {output_enhanced}")

    return output_enhanced

if __name__ == "__main__":
    print("正在创建总框架图...")

    # 检查图片文件
    tc_exists = os.path.exists(training_curves_path)
    cm_exists = os.path.exists(confusion_matrix_path)

    print(f"训练曲线图: {'OK' if tc_exists else 'MISSING'} {training_curves_path}")
    print(f"混淆矩阵图: {'OK' if cm_exists else 'MISSING'} {confusion_matrix_path}")

    if not (tc_exists and cm_exists):
        print("\n警告: 部分图片文件不存在，将显示占位符")

    # 创建增强版框架图
    output = create_enhanced_framework()

    print(f"\n[DONE] 完成! 图片已保存到: {output}")
