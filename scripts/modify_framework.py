"""
修改总框架图 - 替换第四模块的两张图并美化
"""
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from _project_paths import PROJECT_ROOT

# 文件路径
base_dir = str(PROJECT_ROOT)
framework_path = os.path.join(base_dir, "PHM_会议论文_中英文与图", "overall_framework_4.png")
training_curves_path = os.path.join(base_dir, "result_upgrade", "visualizations", "exp3_v3_20260618_010348", "top2_Distill+LWF__384__T_2__alpha_0.3__training_curves.png")
confusion_matrix_path = os.path.join(base_dir, "result_upgrade", "visualizations", "exp3_v3_20260618_010348", "top2_Distill+LWF__384__T_2__alpha_0.3__target_confusion_matrix.png")
output_path = os.path.join(base_dir, "PHM_会议论文_中英文与图", "overall_framework_5_updated.png")

def enhance_image(image, brightness=1.0, contrast=1.0, sharpness=1.0):
    """增强图片质量"""
    from PIL import ImageEnhance
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(brightness)
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast)
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpness)
    return image

def add_gradient_border(draw, bbox, colors, width=3, orientation='vertical'):
    """添加渐变边框"""
    x1, y1, x2, y2 = bbox
    if orientation == 'vertical':
        steps = y2 - y1
        for i in range(steps):
            ratio = i / steps
            r = int(colors[0][0] * (1-ratio) + colors[1][0] * ratio)
            g = int(colors[0][1] * (1-ratio) + colors[1][1] * ratio)
            b = int(colors[0][2] * (1-ratio) + colors[1][2] * ratio)
            draw.rectangle([x1, y1+i, x2, y1+i+1], outline=(r, g, b), width=1)
    else:
        steps = x2 - x1
        for i in range(steps):
            ratio = i / steps
            r = int(colors[0][0] * (1-ratio) + colors[1][0] * ratio)
            g = int(colors[0][1] * (1-ratio) + colors[1][1] * ratio)
            b = int(colors[0][2] * (1-ratio) + colors[1][2] * ratio)
            draw.rectangle([x1+i, y1, x1+i+1, y2], outline=(r, g, b), width=1)

def main():
    print("正在加载图片...")

    # 读取原始框架图
    framework = Image.open(framework_path).convert("RGBA")
    fw_width, fw_height = framework.size

    # 读取新图片
    training_curves = Image.open(training_curves_path).convert("RGBA")
    confusion_matrix = Image.open(confusion_matrix_path).convert("RGBA")

    print(f"框架图尺寸: {fw_width} x {fw_height}")
    print(f"训练曲线图尺寸: {training_curves.size}")
    print(f"混淆矩阵图尺寸: {confusion_matrix.size}")

    # 创建新的画布
    new_framework = framework.copy()
    draw = ImageDraw.Draw(new_framework)

    # 第四模块位置估算（基于4模块布局）
    # 假设是横向4个模块，第四模块在最右侧
    module_width = fw_width // 4
    module4_start = module_width * 3

    # 第四模块内部再分上下两部分
    module_half_height = fw_height // 2

    # 定义插入区域（需要根据实际图片调整）
    top_margin = 100  # 标题区域
    bottom_margin = 50
    side_margin = 20

    # 训练曲线图区域（上方）
    tc_height = module_half_height - top_margin - bottom_margin - 20
    tc_width = module_width - 2 * side_margin

    # 混淆矩阵图区域（下方）
    cm_height = module_half_height - bottom_margin - side_margin - 20
    cm_width = module_width - 2 * side_margin

    # 调整新图片尺寸以适应区域
    training_curves_resized = training_curves.resize((tc_width, tc_height), Image.Resampling.LANCZOS)
    confusion_matrix_resized = confusion_matrix.resize((min(cm_width, cm_height), min(cm_width, cm_height)), Image.Resampling.LANCZOS)

    # 清除原有内容（用白色或背景色填充）
    # 训练曲线区域
    tc_region = [
        module4_start + side_margin,
        top_margin,
        module4_start + side_margin + tc_width,
        top_margin + tc_height
    ]
    draw.rectangle(tc_region, fill=(255, 255, 255, 255))

    # 混淆矩阵区域
    cm_size = min(cm_width, cm_height)
    cm_region = [
        module4_start + side_margin,
        module_half_height + 20,
        module4_start + side_margin + cm_size,
        module_half_height + 20 + cm_size
    ]
    draw.rectangle(cm_region, fill=(255, 255, 255, 255))

    # 粘贴新图片
    new_framework.paste(training_curves_resized,
                       (tc_region[0], tc_region[1]),
                       training_curves_resized)
    new_framework.paste(confusion_matrix_resized,
                       (cm_region[0], cm_region[1]),
                       confusion_matrix_resized)

    # 添加美化元素
    # 1. 为第四模块添加渐变边框
    gradient_colors = [(70, 130, 180), (100, 149, 237)]  # 钢蓝色渐变

    # 2. 添加模块标签装饰
    try:
        # 尝试使用系统字体
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_small = ImageFont.truetype("arial.ttf", 14)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 3. 添加连接线和箭头（美化机制图）
    arrow_color = (100, 100, 100)

    # 在模块间添加流畅的连接线
    for i in range(1, 4):
        x = module_width * i
        # 绘制分隔线
        draw.line([(x, 20), (x, fw_height-20)], fill=(200, 200, 200), width=2)

    # 4. 为每个模块添加圆角矩形边框
    border_radius = 15
    for i in range(4):
        x_start = i * module_width + 10
        y_start = 10
        x_end = (i + 1) * module_width - 10
        y_end = fw_height - 10

        # 绘制圆角矩形（简化版）
        draw.rectangle([x_start, y_start, x_end, y_end],
                       outline=(150, 150, 150), width=2)

    # 5. 添加第四模块标题
    title = "Evaluation & Visualization"
    bbox = draw.textbbox((0, 0), title, font=font_large)
    text_width = bbox[2] - bbox[0]
    title_x = module4_start + (module_width - text_width) // 2
    draw.text((title_x, 30), title, fill=(50, 50, 50), font=font_large)

    # 6. 为嵌入的图添加说明标签
    draw.text((module4_start + side_margin + 10, top_margin - 25),
              "Training Curves", fill=(70, 130, 180), font=font_small)
    draw.text((module4_start + side_margin + 10, module_half_height),
              "Confusion Matrix", fill=(70, 130, 180), font=font_small)

    # 转换为RGB并保存
    final_image = Image.new("RGB", new_framework.size, (255, 255, 255))
    final_image.paste(new_framework, mask=new_framework.split()[3])

    final_image.save(output_path, dpi=(300, 300), quality=95)
    print(f"修改后的图片已保存到: {output_path}")

    return output_path

if __name__ == "__main__":
    main()
