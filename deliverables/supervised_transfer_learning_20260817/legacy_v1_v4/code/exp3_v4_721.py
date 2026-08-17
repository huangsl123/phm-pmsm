# -*- coding: utf-8 -*-
"""
EXP3 改进版 V4：全面域适应策略
1. 分布对齐：MK-MMD、JMMD、Wasserstein距离
2. 对抗训练：DANN、GRL、域判别器
3. 伪标签：目标域无标签数据生成伪标签
4. 知识蒸馏细化：T=1.5-2.5, alpha=0.2-0.4
5. 保留V3最佳方法作为对比
目标：源域40-60%，目标域70-82%
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import json
from datetime import datetime
import random
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from _project_paths import DATASETS_DIR, PROJECT_ROOT, RESULTS_DIR

from models.crossvit import CrossViTFaultDiagnosis
from data.data_processor_v2 import load_csv_data, MultiModalFaultDataset


def get_version_info():
    """获取版本信息"""
    version = {
        'timestamp': datetime.now().isoformat(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        version['cuda_version'] = torch.version.cuda
        version['gpu_name'] = torch.cuda.get_device_name(0)
    return version


def compute_config_hash(config):
    """计算配置的哈希值"""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(source_history, target_history, save_path):
    """Plot training curves for both phases"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Source domain loss
    if source_history and source_history.get('train_loss'):
        axes[0, 0].plot(source_history['train_loss'], label='Train', marker='o', markersize=2, alpha=0.7)
        axes[0, 0].plot(source_history['val_loss'], label='Val', marker='s', markersize=2, alpha=0.7)
        axes[0, 0].set_title('Source Domain - Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

    # Source domain accuracy
    if source_history and source_history.get('train_acc'):
        axes[0, 1].plot(source_history['train_acc'], label='Train', marker='o', markersize=2, alpha=0.7)
        axes[0, 1].plot(source_history['val_acc'], label='Val', marker='s', markersize=2, alpha=0.7)
        axes[0, 1].set_title('Source Domain - Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    # Target domain loss
    if target_history and target_history.get('train_loss'):
        axes[1, 0].plot(target_history['train_loss'], label='Train', marker='o', markersize=2, alpha=0.7)
        axes[1, 0].plot(target_history['val_loss'], label='Val', marker='s', markersize=2, alpha=0.7)
        axes[1, 0].set_title('Target Domain - Loss (Fine-tuning)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    # Target domain accuracy
    if target_history and target_history.get('train_acc'):
        axes[1, 1].plot(target_history['train_acc'], label='Train', marker='o', markersize=2, alpha=0.7)
        axes[1, 1].plot(target_history['val_acc'], label='Val', marker='s', markersize=2, alpha=0.7)
        axes[1, 1].set_title('Target Domain - Accuracy (Fine-tuning)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_attention_heatmaps(model, dataset, output_dir, model_name, device, num_samples=5):
    """
    生成注意力激活热力图

    Parameters:
    -----------
    model : 训练好的模型
    dataset : 测试数据集
    output_dir : 输出目录
    model_name : 模型名称
    device : 计算设备
    num_samples : 生成样本数量
    """
    import matplotlib.gridspec as gridspec

    os.makedirs(output_dir, exist_ok=True)

    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    count = 0
    for time_x, spec_x, label in loader:
        if count >= num_samples:
            break

        time_x = time_x.to(device)
        spec_x = spec_x.to(device)
        label = label.item()

        with torch.no_grad():
            output = model(time_x, spec_x)
            pred = output.argmax(1)[0].item()

        # 创建图形
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # 1. 时域信号
        ax1 = fig.add_subplot(gs[0, 0])
        time_data = time_x[0].cpu().numpy()
        channels = ['Ch1', 'Ch2', 'Ch3']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i in range(3):
            ax1.plot(time_data[:, i], label=channels[i], color=colors[i], alpha=0.7, linewidth=0.8)
        ax1.set_title(f'Time Signal (True: C{label}, Pred: C{pred})', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Time Step', fontsize=10)
        ax1.set_ylabel('Amplitude', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 频域谱图（平均）
        ax2 = fig.add_subplot(gs[0, 1])
        spec_data = spec_x[0].cpu().numpy()
        spec_avg = spec_data.mean(0)
        im2 = ax2.imshow(spec_avg, cmap='viridis', aspect='auto', origin='lower')
        ax2.set_title('Spectrogram (Avg)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Frequency', fontsize=10)
        ax2.set_ylabel('Time', fontsize=10)
        plt.colorbar(im2, ax=ax2)

        # 3. 预测概率
        ax3 = fig.add_subplot(gs[0, 2])
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        ax3.bar(range(len(probs)), probs, color='steelblue', alpha=0.7)
        ax3.set_title('Prediction Probabilities', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Class', fontsize=10)
        ax3.set_ylabel('Probability', fontsize=10)
        ax3.axvline(x=pred, color='red', linestyle='--', label=f'Pred: C{pred}')
        ax3.axvline(x=label, color='green', linestyle='--', label=f'True: C{label}')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 4-6. 三个频段的谱图
        for i in range(3):
            ax = fig.add_subplot(gs[1, i])
            im = ax.imshow(spec_data[i], cmap='viridis', aspect='auto', origin='lower')
            ax.set_title(f'Spectrogram - Ch{i+1}', fontsize=11)
            plt.colorbar(im, ax=ax)

        # 7-9. 时域信号的分段展示
        for i in range(3):
            ax = fig.add_subplot(gs[2, i])
            segment_len = 1024 // 3
            start = i * segment_len
            end = start + segment_len if i < 2 else 1024
            segment = time_data[start:end]

            for ch in range(3):
                ax.plot(segment[:, ch], color=colors[ch], alpha=0.7, linewidth=0.8)

            ax.set_title(f'Time Signal - Seg{i+1}', fontsize=11)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'Attention Heatmap - {model_name}', fontsize=16, fontweight='bold', y=0.995)

        # 保存
        status = 'correct' if label == pred else 'wrong'
        save_path = os.path.join(output_dir, f'sample_{count+1:03d}_trueC{label}_predC{pred}_{status}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f'    Saved: {os.path.basename(save_path)}')
        count += 1


# ============================================================================
# 分布对齐损失函数
# ============================================================================

class MMDLoss(nn.Module):
    """Maximum Mean Discrepancy Loss"""
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMDLoss, self).__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = n_samples

        batch_size = int(source.size()[0])
        kernels = [kernel_mul * (2.0 ** i) for i in range(kernel_num)]

        # 计算每个核的带宽
        if source.dim() == 2:
            source = source.unsqueeze(1)  # (batch, 1, dim)
            target = target.unsqueeze(1)

        # 简化版本：使用线性核
        source_flat = source.view(batch_size, -1)
        target_flat = target.view(target.size(0), -1)

        # 计算核矩阵
        source_source = torch.mm(source_flat, source_flat.t())
        target_target = torch.mm(target_flat, target_flat.t())
        source_target = torch.mm(source_flat, target_flat.t())

        # RBF核
        base = source_source + target_target - 2 * source_target
        base = base.clamp(min=0)  # 确保非负

        # 简化：使用单一RBF核
        gamma = 1.0 / source_flat.size(1)
        kernel = torch.exp(-gamma * base.abs())

        return kernel

    def forward(self, source, target):
        if source.dim() == 3:
            source = source.mean(dim=1)  # 全局平均池化
        if target.dim() == 3:
            target = target.mean(dim=1)

        batch_size = min(source.size(0), target.size(0))
        if source.size(0) != target.size(0):
            if source.size(0) > target.size(0):
                source = source[:target.size(0)]
            else:
                target = target[:source.size(0)]

        # MMD计算
        source_flat = source.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        # 简化MMD
        source_mean = source_flat.mean(0)
        target_mean = target_flat.mean(0)

        mmd = ((source_flat - source_mean).pow(2).mean() +
               (target_flat - target_mean).pow(2).mean())

        return mmd


class WassersteinLoss(nn.Module):
    """Wasserstein距离损失（简化版）"""
    def __init__(self):
        super(WassersteinLoss, self).__init__()

    def forward(self, source, target):
        if source.dim() == 3:
            source = source.mean(dim=1)
        if target.dim() == 3:
            target = target.mean(dim=1)

        batch_size = min(source.size(0), target.size(0))
        if source.size(0) != target.size(0):
            if source.size(0) > target.size(0):
                source = source[:target.size(0)]
            else:
                target = target[:source.size(0)]

        # 简化Wasserstein距离：使用均值差异
        source_flat = source.view(batch_size, -1)
        target_flat = target.view(batch_size, -1)

        # 计算分布中心距离
        wasserstein = torch.norm(source_flat.mean(0) - target_flat.mean(0), p=2)

        return wasserstein


# ============================================================================
# DANN 相关组件
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """梯度反转层"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """梯度反转层"""
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DomainDiscriminator(nn.Module):
    """域判别器"""
    def __init__(self, input_dim=256, hidden_dim=128):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return self.sigmoid(x)


# ============================================================================
# 伪标签生成器
# ============================================================================

class PseudoLabelGenerator:
    """伪标签生成器"""
    def __init__(self, model, device, threshold=0.9):
        self.model = model
        self.device = device
        self.threshold = threshold

    def generate_pseudo_labels(self, target_data):
        """生成目标域伪标签"""
        self.model.eval()

        dataset = MultiModalFaultDataset(
            target_data['X_train_time'], target_data['X_train_spec'],
            target_data['y_train'], augment=False
        )

        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

        pseudo_labels = []
        confidences = []

        with torch.no_grad():
            for time_x, spec_x, _ in loader:
                time_x, spec_x = time_x.to(self.device), spec_x.to(self.device)
                outputs = self.model(time_x, spec_x)
                probs = F.softmax(outputs, dim=1)
                max_probs, preds = probs.max(1)

                pseudo_labels.append(preds.cpu().numpy())
                confidences.append(max_probs.cpu().numpy())

        pseudo_labels = np.concatenate(pseudo_labels)
        confidences = np.concatenate(confidences)

        # 只保留高置信度样本
        high_conf_mask = confidences >= self.threshold

        return pseudo_labels, high_conf_mask, confidences


# ============================================================================
# 知识蒸馏损失（从V3继承）
# ============================================================================

class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失"""
    def __init__(self, temperature=3.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        # 软标签损失
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # 硬标签损失
        hard_loss = self.ce_loss(student_outputs, targets)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


# ============================================================================
# 训练和评估函数
# ============================================================================

def train_on_source(config, source_data, device, seed=42, max_epochs=100):
    """在源域训练模型"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    train_dataset = MultiModalFaultDataset(
        source_data['X_train_time'], source_data['X_train_spec'],
        source_data['y_train'], augment=False
    )
    val_dataset = MultiModalFaultDataset(
        source_data['X_val_time'], source_data['X_val_spec'],
        source_data['y_val'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    model = CrossViTFaultDiagnosis(
        in_channels=source_data['n_channels'],
        num_classes=source_data['n_classes'],
        time_seq_len=source_data['window_size'],
        spec_height=source_data['spec_size'][0],
        spec_width=source_data['spec_size'][1],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for time_x, spec_x, y in train_loader:
            time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(time_x, spec_x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for time_x, spec_x, y in val_loader:
                time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
                outputs = model(time_x, spec_x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 20 == 0:
            print(f'    Epoch {epoch+1}/{max_epochs} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    return model, best_val_acc, history


def fine_tune_with_dann(model, target_data, source_data, device, config,
                        lr=0.0001, max_epochs=100, dann_lambda=0.1):
    """使用DANN进行微调"""
    # 创建域判别器
    feature_dim = config['embed_dim'] * 2  # 融合特征维度
    domain_discriminator = DomainDiscriminator(input_dim=feature_dim).to(device)
    grl = GradientReversalLayer(lambda_=dann_lambda).to(device)

    # 获取目标域数据
    train_dataset = MultiModalFaultDataset(
        target_data['X_train_time'], target_data['X_train_spec'],
        target_data['y_train'], augment=False
    )
    val_dataset = MultiModalFaultDataset(
        target_data['X_val_time'], target_data['X_val_spec'],
        target_data['y_val'], augment=False
    )

    # 获取少量源域数据用于域适应
    source_dataset = MultiModalFaultDataset(
        source_data['X_train_time'], source_data['X_train_spec'],
        source_data['y_train'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    source_loader = DataLoader(source_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    # 优化器
    optimizer = optim.Adam(list(model.parameters()) + list(domain_discriminator.parameters()),
                          lr=lr, weight_decay=config['weight_decay'])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    domain_criterion = nn.BCELoss()

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'  Fine-tuning with DANN (lambda={dann_lambda}, lr={lr})...')

    source_iter = iter(source_loader)

    for epoch in range(max_epochs):
        model.train()
        domain_discriminator.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for time_x, spec_x, y in train_loader:
            time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
            optimizer.zero_grad()

            # 获取源域批次
            try:
                s_time_x, s_spec_x, _ = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                s_time_x, s_spec_x, _ = next(source_iter)
            s_time_x, s_spec_x = s_time_x.to(device), s_spec_x.to(device)

            # Keep the feature graph attached to the backbone.  Detaching here
            # would train only the discriminator and make DANN ineffective.
            outputs, target_feat = model(time_x, spec_x, return_features=True)
            _, source_feat = model(s_time_x, s_spec_x, return_features=True)
            class_loss = criterion(outputs, y)

            # 域对抗损失
            domain_input = torch.cat([source_feat, target_feat], dim=0)
            domain_label = torch.cat([
                torch.ones(source_feat.size(0), 1).to(device),
                torch.zeros(target_feat.size(0), 1).to(device)
            ])

            reversed_feat = grl(domain_input)
            domain_output = domain_discriminator(reversed_feat)
            domain_loss = domain_criterion(domain_output, domain_label)

            # 总损失
            loss = class_loss + 0.1 * domain_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for time_x, spec_x, y in val_loader:
                time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
                outputs = model(time_x, spec_x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}/{max_epochs} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_state)
    print(f'  Target fine-tuning complete: Val={best_val_acc:.2f}%')
    return model, best_val_acc, history


def fine_tune_with_pseudo_labels(model, target_data, source_data, device, config,
                                  lr=0.0001, max_epochs=100, threshold=0.9):
    """使用伪标签进行自训练"""
    # 生成伪标签
    pseudo_gen = PseudoLabelGenerator(model, device, threshold=threshold)
    pseudo_labels, high_conf_mask, confidences = pseudo_gen.generate_pseudo_labels(target_data)

    print(f'  Generated {high_conf_mask.sum()} pseudo labels ({high_conf_mask.mean()*100:.1f}% of data)')
    print(f'  Average confidence: {confidences.mean():.3f}')

    # 创建混合数据集
    high_conf_indices = np.where(high_conf_mask)[0]

    # 使用高置信度伪标签数据
    target_train_dataset = MultiModalFaultDataset(
        target_data['X_train_time'], target_data['X_train_spec'],
        pseudo_labels, augment=False
    )

    # 只使用高置信度样本
    from torch.utils.data import Subset
    pseudo_dataset = Subset(target_train_dataset, high_conf_indices)

    # 混合源域和伪标签数据
    source_dataset = MultiModalFaultDataset(
        source_data['X_train_time'], source_data['X_train_spec'],
        source_data['y_train'], augment=False
    )

    mixed_dataset = ConcatDataset([source_dataset, pseudo_dataset])

    train_loader = DataLoader(mixed_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    val_dataset = MultiModalFaultDataset(
        target_data['X_val_time'], target_data['X_val_spec'],
        target_data['y_val'], augment=False
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'  Fine-tuning with Pseudo Labels (threshold={threshold}, lr={lr})...')

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for time_x, spec_x, y in train_loader:
            time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(time_x, spec_x)
            loss = criterion(outputs, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for time_x, spec_x, y in val_loader:
                time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
                outputs = model(time_x, spec_x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}/{max_epochs} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_state)
    print(f'  Target fine-tuning complete: Val={best_val_acc:.2f}%')
    return model, best_val_acc, history


def fine_tune_with_mmd(model, target_data, source_data, device, config,
                       lr=0.0001, max_epochs=100, mmd_lambda=0.1):
    """使用MMD分布对齐进行微调"""
    mmd_loss = MMDLoss().to(device)

    train_dataset = MultiModalFaultDataset(
        target_data['X_train_time'], target_data['X_train_spec'],
        target_data['y_train'], augment=False
    )
    val_dataset = MultiModalFaultDataset(
        target_data['X_val_time'], target_data['X_val_spec'],
        target_data['y_val'], augment=False
    )

    # 源域数据（用于MMD计算）
    source_dataset = MultiModalFaultDataset(
        source_data['X_train_time'], source_data['X_train_spec'],
        source_data['y_train'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    source_loader = DataLoader(source_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'  Fine-tuning with MMD (lambda={mmd_lambda}, lr={lr})...')

    source_iter = iter(source_loader)

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for time_x, spec_x, y in train_loader:
            time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)

            # 获取源域批次
            try:
                s_time_x, s_spec_x, _ = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                s_time_x, s_spec_x, _ = next(source_iter)
            s_time_x, s_spec_x = s_time_x.to(device), s_spec_x.to(device)

            optimizer.zero_grad()

            # MMD must back-propagate through both feature extractors.
            outputs, target_feat = model(time_x, spec_x, return_features=True)
            _, source_feat = model(s_time_x, s_spec_x, return_features=True)
            class_loss = criterion(outputs, y)

            # MMD损失
            mmd = mmd_loss(target_feat, source_feat)

            # 总损失
            loss = class_loss + mmd_lambda * mmd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for time_x, spec_x, y in val_loader:
                time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
                outputs = model(time_x, spec_x)
                loss = criterion(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}/{max_epochs} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_state)
    print(f'  Target fine-tuning complete: Val={best_val_acc:.2f}%')
    return model, best_val_acc, history


def fine_tune_with_distillation(model, target_data, source_data, device, config,
                                 lr=0.0001, max_epochs=100,
                                 temperature=2.0, distill_alpha=0.3,
                                 mix_source_ratio=0.3):
    """使用知识蒸馏 + LWF进行微调"""
    # 创建教师模型
    teacher_model = CrossViTFaultDiagnosis(
        in_channels=source_data['n_channels'],
        num_classes=source_data['n_classes'],
        time_seq_len=source_data['window_size'],
        spec_height=source_data['spec_size'][0],
        spec_width=source_data['spec_size'][1],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)
    teacher_model.load_state_dict(model.state_dict())
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # 混合数据
    source_train_dataset = MultiModalFaultDataset(
        source_data['X_train_time'], source_data['X_train_spec'],
        source_data['y_train'], augment=False
    )

    target_train_dataset = MultiModalFaultDataset(
        target_data['X_train_time'], target_data['X_train_spec'],
        target_data['y_train'], augment=False
    )

    val_dataset = MultiModalFaultDataset(
        target_data['X_val_time'], target_data['X_val_spec'],
        target_data['y_val'], augment=False
    )

    source_size = int(len(source_train_dataset) * mix_source_ratio)
    target_size = len(target_train_dataset)

    from torch.utils.data import Subset
    mixed_dataset = ConcatDataset([
        Subset(source_train_dataset, np.random.choice(len(source_train_dataset), source_size, replace=False)),
        target_train_dataset
    ])

    train_loader = DataLoader(mixed_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f'  Mixed dataset: {source_size} source + {target_size} target samples')

    distillation_criterion = KnowledgeDistillationLoss(
        temperature=temperature,
        alpha=distill_alpha
    )

    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=config['weight_decay'])

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'  Fine-tuning with Distillation (T={temperature}, alpha={distill_alpha})...')

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for time_x, spec_x, y in train_loader:
            time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
            optimizer.zero_grad()

            student_outputs = model(time_x, spec_x)

            with torch.no_grad():
                teacher_outputs = teacher_model(time_x, spec_x)

            loss = distillation_criterion(student_outputs, teacher_outputs, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = student_outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()

        train_acc = 100. * train_correct / train_total
        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for time_x, spec_x, y in val_loader:
                time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
                outputs = model(time_x, spec_x)
                loss = nn.CrossEntropyLoss()(outputs, y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()

        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 10 == 0:
            print(f'    Epoch {epoch+1}/{max_epochs} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_state)
    print(f'  Target fine-tuning complete: Val={best_val_acc:.2f}%')
    return model, best_val_acc, history


# ============================================================================
# 评估函数
# ============================================================================

def evaluate_with_preds(model, data, device, split='test'):
    """评估模型并返回准确率和预测结果"""
    if split == 'train':
        dataset = MultiModalFaultDataset(
            data['X_train_time'], data['X_train_spec'],
            data['y_train'], augment=False
        )
    elif split == 'val':
        dataset = MultiModalFaultDataset(
            data['X_val_time'], data['X_val_spec'],
            data['y_val'], augment=False
        )
    else:
        dataset = MultiModalFaultDataset(
            data['X_test_time'], data['X_test_spec'],
            data['y_test'], augment=False
        )

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for time_x, spec_x, y in loader:
            time_x, spec_x = time_x.to(device), spec_x.to(device)
            y = y.to(device)

            output = model(time_x, spec_x)
            _, pred = output.max(1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            total += y.size(0)
            correct += pred.eq(y).sum().item()

    accuracy = 100. * correct / total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return accuracy, all_preds, all_labels


def evaluate_all_splits(model, data, device):
    """评估模型在所有数据集上的准确率"""
    train_acc, _, _ = evaluate_with_preds(model, data, device, 'train')
    val_acc, _, _ = evaluate_with_preds(model, data, device, 'val')
    test_acc, test_preds, test_labels = evaluate_with_preds(model, data, device, 'test')

    return {
        'train': train_acc,
        'val': val_acc,
        'test': test_acc,
        'test_preds': test_preds,
        'test_labels': test_labels
    }


# ============================================================================
# 主实验函数
# ============================================================================

def try_improved_models_v4(source_power='1.0kW', target_power='3.0kW', base_path=str(DATASETS_DIR), seed=42):
    """尝试多种改进配置 - V4版本（全面域适应策略）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    version_info = get_version_info()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('\n' + '=' * 80)
    print(f'EXP3 IMPROVED V4 - Comprehensive Domain Adaptation')
    print(f'Source: {source_power} -> Target: {target_power}')
    print(f'Version: {timestamp}')
    print('=' * 80)
    print(f'PyTorch: {version_info["pytorch_version"]}')
    print(f'CUDA: {version_info["cuda_available"]}')
    if version_info["cuda_available"]:
        print(f'GPU: {version_info["gpu_name"]}')
    print('=' * 80)

    # 加载数据
    print('\nLoading data with time-blocked 70:20:10 split (train:test:val), stride=128...')
    # 加载数据 - 使用 按原始时间块 70:20:10 分割（训练:测试:验证）
    source_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{source_power}.csv'),
        window_size=1024,
        stride=128,
        spec_size=None,
        test_size=0.10,
        val_size=0.05,
        random_state=42,
        split_mode='time_blocked'  # time-blocked 70:20:10 split
    )
    target_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{target_power}.csv'),
        window_size=1024,
        stride=128,
        spec_size=None,
        test_size=0.10,
        val_size=0.05,
        random_state=42,
        split_mode='time_blocked'  # time-blocked 70:20:10 split
    )

    # V4 最佳配置 (stride=128版本) - 只保留最好的两个配置
    # 根据实验报告，最佳两个配置是：
    # 1. Distill (T=1.8, alpha=0.3) - 平衡分80.0
    # 2. Distill (T=1.5, alpha=0.25) - 平衡分78.8
    configs = [
        # 最佳配置1: Distill (T=1.8, alpha=0.3)
        {
            'name': 'Distill (T=1.8, alpha=0.3)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'method': 'distill',
            'ft_lr': 0.0001,
            'temperature': 1.8,
            'distill_alpha': 0.3,
            'mix_source_ratio': 0.3
        },
        # 最佳配置2: Distill (T=1.5, alpha=0.25)
        {
            'name': 'Distill (T=1.5, alpha=0.25)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'method': 'distill',
            'ft_lr': 0.0001,
            'temperature': 1.5,
            'distill_alpha': 0.25,
            'mix_source_ratio': 0.3
        },
    ]

    results = []
    best_model = None
    best_config_name = None
    best_target_acc = 0
    best_balance_score = -float('inf')

    # 跟踪前两名模型（基于目标域准确率）
    top2_models = []

    # 统一结果目录（带时间戳）
    base_dir = str(RESULTS_DIR)
    json_dir = os.path.join(base_dir, 'JSON', 'exp3_v4')
    viz_dir = os.path.join(base_dir, 'visualizations', f'exp3_v4_{timestamp}')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    for i, cfg_dict in enumerate(configs):
        print(f'\n{"=" * 80}')
        print(f'CONFIG {i + 1}/{len(configs)}: {cfg_dict["name"]}')
        print(f'{"=" * 80}')

        config_for_hash = {k: v for k, v in cfg_dict.items() if k not in ['name', 'method']}
        config_hash = compute_config_hash(config_for_hash)
        print(f'Config Hash: {config_hash}')

        config = {k: v for k, v in cfg_dict.items()
                  if k not in ['name', 'method', 'ft_lr', 'temperature', 'distill_alpha',
                              'mix_source_ratio', 'dann_lambda', 'mmd_lambda', 'threshold']}

        # 阶段1: 源域训练
        model, source_val_acc, source_history = train_on_source(
            config, source_data, device, seed=seed, max_epochs=100
        )

        source_results_before = evaluate_all_splits(model, source_data, device)
        target_results_before = evaluate_all_splits(model, target_data, device)

        print(f'  After Phase 1 (Source Domain):')
        print(f'    Train: {source_results_before["train"]:.1f}% | Val: {source_results_before["val"]:.1f}% | Test: {source_results_before["test"]:.1f}%')
        print(f'  After Phase 1 (Target Domain - Before FT):')
        print(f'    Train: {target_results_before["train"]:.1f}% | Val: {target_results_before["val"]:.1f}% | Test: {target_results_before["test"]:.1f}%')

        # 阶段2: 目标域微调
        method = cfg_dict.get('method', 'baseline')

        if method == 'distill':
            model, target_val_acc, target_history = fine_tune_with_distillation(
                model, target_data, source_data, device, config,
                lr=cfg_dict['ft_lr'],
                max_epochs=100,
                temperature=cfg_dict.get('temperature', 2.0),
                distill_alpha=cfg_dict.get('distill_alpha', 0.3),
                mix_source_ratio=cfg_dict.get('mix_source_ratio', 0.3)
            )
        elif method == 'dann':
            model, target_val_acc, target_history = fine_tune_with_dann(
                model, target_data, source_data, device, config,
                lr=cfg_dict['ft_lr'],
                max_epochs=100,
                dann_lambda=cfg_dict.get('dann_lambda', 0.1)
            )
        elif method == 'pseudo':
            model, target_val_acc, target_history = fine_tune_with_pseudo_labels(
                model, target_data, source_data, device, config,
                lr=cfg_dict['ft_lr'],
                max_epochs=100,
                threshold=cfg_dict.get('threshold', 0.9)
            )
        elif method == 'mmd':
            model, target_val_acc, target_history = fine_tune_with_mmd(
                model, target_data, source_data, device, config,
                lr=cfg_dict['ft_lr'],
                max_epochs=100,
                mmd_lambda=cfg_dict.get('mmd_lambda', 0.1)
            )
        elif method == 'distill_pseudo':
            # 先蒸馏，后伪标签
            model, target_val_acc, target_history = fine_tune_with_distillation(
                model, target_data, source_data, device, config,
                lr=cfg_dict['ft_lr'],
                max_epochs=50,
                temperature=cfg_dict.get('temperature', 2.0),
                distill_alpha=cfg_dict.get('distill_alpha', 0.3),
                mix_source_ratio=cfg_dict.get('mix_source_ratio', 0.3)
            )
            # 再用伪标签微调
            model, _, _ = fine_tune_with_pseudo_labels(
                model, target_data, source_data, device, config,
                lr=cfg_dict['ft_lr'] * 0.5,
                max_epochs=50,
                threshold=cfg_dict.get('threshold', 0.9)
            )
            target_val_acc = evaluate_all_splits(model, target_data, device)['val']
        else:
            # 默认微调
            train_dataset = MultiModalFaultDataset(
                target_data['X_train_time'], target_data['X_train_spec'],
                target_data['y_train'], augment=False
            )
            val_dataset = MultiModalFaultDataset(
                target_data['X_val_time'], target_data['X_val_spec'],
                target_data['y_val'], augment=False
            )
            train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            print(f'  Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M')

            criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
            optimizer = optim.Adam(trainable_params, lr=cfg_dict['ft_lr'], weight_decay=config['weight_decay'])

            best_val_acc = 0
            best_model_state = None
            patience_counter = 0
            patience = 20
            target_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

            print(f'  Fine-tuning on target (lr={cfg_dict["ft_lr"]}, max_epochs=100)...')

            for epoch in range(100):
                model.train()
                epoch_loss = 0
                train_correct = 0
                train_total = 0

                for time_x, spec_x, y in train_loader:
                    time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
                    optimizer.zero_grad()
                    outputs = model(time_x, spec_x)
                    loss = criterion(outputs, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += y.size(0)
                    train_correct += predicted.eq(y).sum().item()

                train_acc = 100. * train_correct / train_total
                avg_train_loss = epoch_loss / len(train_loader)

                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for time_x, spec_x, y in val_loader:
                        time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
                        outputs = model(time_x, spec_x)
                        loss = criterion(outputs, y)
                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += y.size(0)
                        val_correct += predicted.eq(y).sum().item()

                val_acc = 100. * val_correct / val_total
                avg_val_loss = val_loss / len(val_loader)

                target_history['train_loss'].append(avg_train_loss)
                target_history['train_acc'].append(train_acc)
                target_history['val_loss'].append(avg_val_loss)
                target_history['val_acc'].append(val_acc)

                if (epoch + 1) % 10 == 0:
                    print(f'    Epoch {epoch+1}/100 | Train: {train_acc:.1f}% | Val: {val_acc:.1f}% | Best: {best_val_acc:.1f}%')

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f'    Early stopping at epoch {epoch + 1}')
                    break

            model.load_state_dict(best_model_state)
            print(f'  Target fine-tuning complete: Val={best_val_acc:.2f}%')
            target_val_acc = best_val_acc

        # 详细评估微调后
        source_results_after = evaluate_all_splits(model, source_data, device)
        target_results_after = evaluate_all_splits(model, target_data, device)

        print(f'  After Phase 2 (Source Domain):')
        print(f'    Train: {source_results_after["train"]:.1f}% | Val: {source_results_after["val"]:.1f}% | Test: {source_results_after["test"]:.1f}%')
        print(f'  After Phase 2 (Target Domain):')
        print(f'    Train: {target_results_after["train"]:.1f}% | Val: {target_results_after["val"]:.1f}% | Test: {target_results_after["test"]:.1f}%')
        print(f'    Target Improvement: +{target_results_after["test"] - target_results_before["test"]:.1f}%')

        # 计算平衡分
        source_retention = source_results_after["val"]
        target_acc = target_results_after["val"]
        source_reference = source_results_before["val"]
        balance_score = target_acc - max(0.0, source_reference - source_retention)

        result = {
            'config_name': cfg_dict['name'],
            'config_hash': config_hash,
            'config': config,
            'method': method,
            'source': {
                'train_acc': float(source_results_after['train']),
                'val_acc': float(source_results_after['val']),
                'test_acc': float(source_results_after['test'])
            },
            'target': {
                'train_acc_before': float(target_results_before['train']),
                'val_acc_before': float(target_results_before['val']),
                'test_acc_before': float(target_results_before['test']),
                'train_acc': float(target_results_after['train']),
                'val_acc': float(target_results_after['val']),
                'test_acc': float(target_results_after['test']),
                'improvement': float(target_results_after['test'] - target_results_before['test'])
            },
            'balance_score': float(balance_score)
        }

        if method == 'distill':
            result['temperature'] = cfg_dict.get('temperature', 2.0)
            result['distill_alpha'] = cfg_dict.get('distill_alpha', 0.3)
            result['mix_source_ratio'] = cfg_dict.get('mix_source_ratio', 0.3)
        elif method == 'dann':
            result['dann_lambda'] = cfg_dict.get('dann_lambda', 0.1)
        elif method == 'pseudo':
            result['threshold'] = cfg_dict.get('threshold', 0.9)
        elif method == 'mmd':
            result['mmd_lambda'] = cfg_dict.get('mmd_lambda', 0.1)

        results.append(result)

        if target_acc > best_target_acc:
            best_target_acc = target_acc
            best_model = model
            best_config_name = cfg_dict['name']
            best_source_preds = source_results_after['test_preds']
            best_source_labels = source_results_after['test_labels']
            best_target_preds = target_results_after['test_preds']
            best_target_labels = target_results_after['test_labels']
            best_source_history = source_history
            best_target_history = target_history

        # 跟踪前两名模型
        top2_models.append({
            'target_acc': target_acc,
            'config_name': cfg_dict['name'],
            'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},  # 保存模型状态
            'source_preds': source_results_after['test_preds'].copy(),
            'source_labels': source_results_after['test_labels'].copy(),
            'target_preds': target_results_after['test_preds'].copy(),
            'target_labels': target_results_after['test_labels'].copy(),
            'source_history': source_history,
            'target_history': target_history
        })

        if balance_score > best_balance_score:
            best_balance_score = balance_score
            best_balance_config = cfg_dict['name']
            best_balance_result = result.copy()

    # 最终报告
    print('\n' + '=' * 80)
    print('ALL RESULTS SUMMARY')
    print('=' * 80)

    for r in results:
        print(f'\n{r["config_name"]}:')
        print(f'  Source - Train: {r["source"]["train_acc"]:.1f}% | Val: {r["source"]["val_acc"]:.1f}% | Test: {r["source"]["test_acc"]:.1f}%')
        print(f'  Target - Train: {r["target"]["train_acc"]:.1f}% | Val: {r["target"]["val_acc"]:.1f}% | Test: {r["target"]["test_acc"]:.1f}%')
        print(f'  Balance Score: {r["balance_score"]:.1f}')

    best_result = max(results, key=lambda x: x['target']['val_acc'])
    print(f'\n🏆 Best Target Accuracy: {best_result["config_name"]}')
    print(f'   Source: {best_result["source"]["test_acc"]:.1f}% | Target: {best_result["target"]["test_acc"]:.1f}%')

    print(f'\n⚖️  Best Balance Score: {best_balance_config}')
    print(f'   Source: {best_balance_result["source"]["test_acc"]:.1f}% | Target: {best_balance_result["target"]["test_acc"]:.1f}%')

    # 保存完整结果
    full_results = {
        'version': version_info,
        'timestamp': timestamp,
        'all_results': results,
        'best_target_acc': best_result,
        'best_balance': {
            'config_name': best_balance_config,
            'result': best_balance_result
        },
        'v3_comparison': {
            'v3_best_distill': {
                'source_test': 77.5,
                'target_test': 65.0,
                'balance_score': 75.0
            },
            'v3_best_baseline': {
                'source_test': 17.5,
                'target_test': 77.5,
                'balance_score': 77.5
            }
        },
        'new_methods': [
            'DANN (Domain Adversarial Neural Networks)',
            'MMD (Maximum Mean Discrepancy)',
            'Pseudo Label (Self-Training)',
            'Distill+Pseudo (Combined Strategy)',
            'Fine-tuned Distillation (T=1.5-2.5, alpha=0.25-0.4)'
        ]
    }

    # 保存完整结果到统一目录
    json_path = os.path.join(json_dir, f'exp3_v4_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    # 保存简化版本
    summary = {
        'timestamp': timestamp,
        'version': version_info,
        'best_target': {
            'config': best_result['config_name'],
            'source_test': best_result['source']['test_acc'],
            'target_test': best_result['target']['test_acc']
        },
        'best_balance': {
            'config': best_balance_config,
            'source_test': best_balance_result['source']['test_acc'],
            'target_test': best_balance_result['target']['test_acc']
        },
        'all_configs': [
            {
                'name': r['config_name'],
                'method': r.get('method', 'baseline'),
                'source_test': r['source']['test_acc'],
                'target_test': r['target']['test_acc'],
                'balance_score': r['balance_score']
            }
            for r in results
        ]
    }

    # 保存简化版本到统一目录
    summary_path = os.path.join(json_dir, f'exp3_v4_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nResults saved to {json_dir}/')

    # 保存前两名模型的权重
    # 按目标域准确率排序，取前两名
    top2_sorted = sorted(top2_models, key=lambda x: x['target_acc'], reverse=True)[:2]

    # 创建模型保存目录
    model_dir = os.path.join(base_dir, 'models', f'exp3_v4_{timestamp}')
    os.makedirs(model_dir, exist_ok=True)

    print(f'\nSaving top 2 model weights...')
    for rank, model_info in enumerate(top2_sorted, 1):
        config_name_safe = model_info['config_name'].replace(' ', '_').replace('(', '_').replace(')', '_').replace(',', '_').replace('=', '_')
        model_path = os.path.join(model_dir, f'top{rank}_{config_name_safe}_{timestamp}.pth')

        # 保存模型状态
        torch.save(model_info['model_state'], model_path)
        print(f'  Rank {rank}: {model_info["config_name"]} (Target Acc: {model_info["target_acc"]:.1f}%)')
        print(f'    Saved: {model_path}')

    # 生成前两名模型的可视化
    class_names = [f'C{i}' for i in range(source_data['n_classes'])]
    if 'fault_codes' in source_data:
        class_names = [str(c) for c in source_data['fault_codes']]

    print(f'\nGenerating visualizations for top 2 models...')

    for rank, model_info in enumerate(top2_sorted, 1):
        config_name_safe = model_info['config_name'].replace(' ', '_').replace('(', '_').replace(')', '_').replace(',', '_').replace('=', '_')
        prefix = f'top{rank}_{config_name_safe}'

        # 源域混淆矩阵
        plot_confusion_matrix(
            model_info['source_labels'], model_info['source_preds'], class_names,
            f'Source Domain Test - {model_info["config_name"]}',
            os.path.join(viz_dir, f'{prefix}_source_confusion_matrix.png')
        )

        # 目标域混淆矩阵
        plot_confusion_matrix(
            model_info['target_labels'], model_info['target_preds'], class_names,
            f'Target Domain Test - {model_info["config_name"]}',
            os.path.join(viz_dir, f'{prefix}_target_confusion_matrix.png')
        )

        # 训练曲线
        plot_training_curves(
            model_info['source_history'], model_info['target_history'],
            os.path.join(viz_dir, f'{prefix}_training_curves.png')
        )

    print(f'Visualizations saved to {viz_dir}/')

    # 生成前两名模型的注意力激活热力图
    print(f'\nGenerating attention heatmaps for top 2 models...')

    # 获取目标域测试集用于热力图生成
    target_test_dataset = MultiModalFaultDataset(
        target_data['X_test_time'],
        target_data['X_test_spec'],
        target_data['y_test'],
        augment=False
    )

    for rank, model_info in enumerate(top2_sorted, 1):
        config_name_safe = model_info['config_name'].replace(' ', '_').replace('(', '_').replace(')', '_').replace(',', '_').replace('=', '_')

        # 创建热力图保存目录
        heatmap_dir = os.path.join(base_dir, 'visualizations', f'attention_v4_{timestamp}', f'top{rank}_{config_name_safe}')

        # 临时创建模型来加载权重
        temp_model = CrossViTFaultDiagnosis(
            in_channels=3,
            num_classes=source_data['n_classes'],
            time_seq_len=source_data['window_size'],
            spec_height=source_data['spec_size'][0],
            spec_width=source_data['spec_size'][1],
            embed_dim=configs[0]['embed_dim'],  # 使用默认配置
            num_heads=configs[0]['num_heads'],
            num_layers=configs[0]['num_layers'],
            dropout=configs[0]['dropout']
        ).to(device)

        # 加载模型状态
        temp_model.load_state_dict(model_info['model_state'])

        print(f'  Rank {rank}: {model_info["config_name"]}')
        generate_attention_heatmaps(
            temp_model, target_test_dataset, heatmap_dir,
            model_info['config_name'], device, num_samples=5
        )

    print(f'Attention heatmaps saved!')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_power', type=str, default='1.0kW')
    parser.add_argument('--target_power', type=str, default='3.0kW')
    parser.add_argument('--base_path', type=str, default=str(DATASETS_DIR))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    try_improved_models_v4(args.source_power, args.target_power, args.base_path, seed=args.seed)
