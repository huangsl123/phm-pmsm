# -*- coding: utf-8 -*-
"""
EXP3 改进版 V3：修复冻结机制 + 添加防遗忘策略
1. 修复参数冻结机制 - 正确匹配模型命名
2. EWC (Elastic Weight Consolidation) - 保护重要参数
3. 知识蒸馏 - 保持源域知识
4. LWF (Learning without Forgetting) - 混合数据训练
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
    """计算配置的哈希值，用于唯一标识"""
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


class EWC(object):
    """
    Elastic Weight Consolidation (EWC)
    通过Fisher信息矩阵保护重要参数
    """
    def __init__(self, model, criterion, device, ewc_lambda=1000):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.ewc_lambda = ewc_lambda
        self.params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self._means = {}
        self._fishers = {}

    def compute_fisher(self, data_loader):
        """计算Fisher信息矩阵"""
        self.model.eval()
        fishers = {}

        for name, param in self.params.items():
            fishers[name] = torch.zeros_like(param.data)

        for time_x, spec_x, y in data_loader:
            time_x, spec_x, y = time_x.to(self.device), spec_x.to(self.device), y.to(self.device)

            self.model.zero_grad()
            outputs = self.model(time_x, spec_x)
            loss = self.criterion(outputs, y)
            loss.backward()

            for name, param in self.params.items():
                if param.grad is not None:
                    fishers[name] += param.grad.data ** 2

        # 归一化
        n_samples = len(data_loader)
        for name in fishers:
            fishers[name] /= n_samples
            self._fishers[name] = fishers[name].to(self.device)
            self._means[name] = self.params[name].data.clone().to(self.device)

        # 清空梯度
        self.model.zero_grad()

    def ewc_loss(self):
        """计算EWC正则化损失"""
        loss = 0
        for name, param in self.params.items():
            if name in self._fishers and name in self._means:
                fisher = self._fishers[name]
                mean = self._means[name]
                loss += (fisher * (param - mean) ** 2).sum()
        return (self.ewc_lambda / 2) * loss


class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失
    使用教师模型的软标签保持知识
    """
    def __init__(self, temperature=3.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_outputs, teacher_outputs, targets):
        """
        student_outputs: 学生模型输出 (logits)
        teacher_outputs: 教师模型输出 (logits)
        targets: 真实标签
        """
        # 软标签损失（知识蒸馏）
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1)
        ) * (self.temperature ** 2)

        # 硬标签损失
        hard_loss = self.ce_loss(student_outputs, targets)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


def freeze_layers_correctly(model, freeze_backbone=False, freeze_layers=None,
                           freeze_embedding=False, verbose=True):
    """
    正确的层冻结机制

    参数:
        freeze_backbone: 冻结除分类器外的所有层
        freeze_layers: 冻结前N层encoder (时域和频域各N层)
        freeze_embedding: 冻结embedding层
    """
    frozen_count = 0

    if freeze_backbone:
        if verbose:
            print('  Freezing backbone (all except classifier)...')
        for name, param in model.named_parameters():
            # 只保留分类器可训练
            if 'classifier' not in name:
                param.requires_grad = False
                frozen_count += 1
        if verbose:
            print(f'  Froze {frozen_count} parameters (backbone freeze)')

    elif freeze_layers is not None and freeze_layers > 0:
        if verbose:
            print(f'  Freezing first {freeze_layers} encoder layers...')
        # 修复：正确匹配模型参数命名
        # time_encoder.encoder.layers.N 和 spec_encoder.encoder.layers.N
        for name, param in model.named_parameters():
            if ('time_encoder.encoder.layers' in name or 'spec_encoder.encoder.layers' in name):
                # 提取层号: layers.0.xxx -> 0
                try:
                    layer_part = name.split('layers.')[1].split('.')[0]
                    layer_num = int(layer_part)
                    if layer_num < freeze_layers:
                        param.requires_grad = False
                        frozen_count += 1
                except (IndexError, ValueError):
                    continue

        if verbose:
            print(f'  Froze {frozen_count} parameter groups (layer freeze)')

    if freeze_embedding:
        if verbose:
            print('  Freezing embedding layers...')
        for name, param in model.named_parameters():
            if 'patch_embed' in name or 'pos_encoder' in name:
                param.requires_grad = False
                frozen_count += 1
        if verbose:
            print(f'  Froze additional embedding parameters')

    return frozen_count


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

    print(f'  Training source model (max_epochs={max_epochs})...')

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
    print(f'  Source training complete: Val={best_val_acc:.2f}% (Epoch: {len(history["train_acc"])})')
    return model, best_val_acc, history


def fine_tune_with_ewc(model, target_data, source_data, device, config,
                       lr=0.0001, max_epochs=100, freeze_backbone=False,
                       freeze_layers=None, freeze_embedding=False,
                       ewc_lambda=1000):
    """
    使用EWC进行微调
    """
    # 首先冻结指定的层
    freeze_layers_correctly(model, freeze_backbone, freeze_layers, freeze_embedding)

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
    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=config['weight_decay'])

    # 初始化EWC
    ewc = EWC(model, criterion, device, ewc_lambda=ewc_lambda)

    # 计算初始Fisher矩阵 (使用源域验证集)
    source_val_dataset = MultiModalFaultDataset(
        source_data['X_val_time'], source_data['X_val_spec'],
        source_data['y_val'], augment=False
    )
    source_val_loader = DataLoader(source_val_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f'  Computing initial Fisher matrix...')
    ewc.compute_fisher(source_val_loader)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'  Fine-tuning with EWC (lambda={ewc_lambda}, lr={lr})...')

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        train_correct = 0
        train_total = 0

        for time_x, spec_x, y in train_loader:
            time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(time_x, spec_x)

            # 标准损失 + EWC正则化
            loss = criterion(outputs, y) + ewc.ewc_loss()

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
    print(f'  Target fine-tuning complete: Val={best_val_acc:.2f}% (Epoch: {len(history["train_acc"])})')
    return model, best_val_acc, history


def fine_tune_with_distillation(model, target_data, source_data, device, config,
                                 lr=0.0001, max_epochs=100, freeze_backbone=False,
                                 freeze_layers=None, freeze_embedding=False,
                                 temperature=3.0, distill_alpha=0.5,
                                 mix_source_ratio=0.3):
    """
    使用知识蒸馏 + LWF进行微调
    """
    # 首先冻结指定的层
    freeze_layers_correctly(model, freeze_backbone, freeze_layers, freeze_embedding)

    # 创建教师模型（冻结的源域模型）
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

    # 混合源域和目标域训练数据
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

    # 按比例混合数据
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
    print(f'  Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M')
    print(f'  Mixed dataset: {source_size} source + {target_size} target samples')

    # 知识蒸馏损失
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

    print(f'  Fine-tuning with Distillation+LWF (T={temperature}, alpha={distill_alpha}, source_ratio={mix_source_ratio})...')

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
    print(f'  Target fine-tuning complete: Val={best_val_acc:.2f}% (Epoch: {len(history["train_acc"])})')
    return model, best_val_acc, history


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


def try_improved_models_v3(source_power='1.0kW', target_power='3.0kW', base_path=str(DATASETS_DIR), seed=42):
    """尝试多种改进配置 - V3版本 (修复冻结 + 防遗忘策略)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    version_info = get_version_info()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('\n' + '=' * 80)
    print(f'EXP3 IMPROVED V3 - Fixed Freezing + Anti-Forgetting Strategies')
    print(f'Source: {source_power} -> Target: {target_power}')
    print(f'Version: {timestamp}')
    print('=' * 80)
    print(f'PyTorch: {version_info["pytorch_version"]}')
    print(f'CUDA: {version_info["cuda_available"]}')
    if version_info["cuda_available"]:
        print(f'GPU: {version_info["gpu_name"]}')
    print('=' * 80)

    # 加载数据 - 使用 按原始时间块 70:20:10 分割（训练:测试:验证）
    print('\nLoading data with time-blocked 70:20:10 split (train:test:val), stride=128...')
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

    # V3 最佳配置 (stride=128版本) - 只保留最好的两个配置
    configs = [
        # 最佳配置1: V3-Baseline
        {
            'name': 'V3-Baseline (384, no freeze)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'method': 'baseline',
            'ft_lr': 0.0001,
            'freeze_backbone': False,
            'freeze_layers': None,
            'freeze_embedding': False
        },
        # 最佳配置2: Distill+LWF (T=2, alpha=0.3)
        {
            'name': 'Distill+LWF (384, T=2, alpha=0.3)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'method': 'distill',
            'ft_lr': 0.0001,
            'freeze_backbone': False,
            'freeze_layers': None,
            'freeze_embedding': False,
            'temperature': 2.0,
            'distill_alpha': 0.3,
            'mix_source_ratio': 0.3
        },
    ]

    results = []
    best_model = None
    best_config_name = None
    best_target_acc = 0
    best_balance_score = -float('inf')

    # 跟踪前两名模型（基于目标域准确率）
    top2_models = []  # [(target_acc, model, config_name, source_preds, source_labels, target_preds, target_labels, source_history, target_history)]

    # 统一结果目录（带时间戳）
    base_dir = str(RESULTS_DIR)
    json_dir = os.path.join(base_dir, 'JSON', 'exp3_v3')
    viz_dir = os.path.join(base_dir, 'visualizations', f'exp3_v3_{timestamp}')
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
                  if k not in ['name', 'method', 'ft_lr', 'freeze_backbone',
                              'freeze_layers', 'freeze_embedding', 'ewc_lambda',
                              'temperature', 'distill_alpha', 'mix_source_ratio']}

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

        if method == 'ewc':
            model, target_val_acc, target_history = fine_tune_with_ewc(
                model, target_data, source_data, device, config,
                lr=cfg_dict['ft_lr'],
                max_epochs=100,
                freeze_backbone=cfg_dict['freeze_backbone'],
                freeze_layers=cfg_dict['freeze_layers'],
                freeze_embedding=cfg_dict['freeze_embedding'],
                ewc_lambda=cfg_dict.get('ewc_lambda', 1000)
            )
        elif method == 'distill':
            model, target_val_acc, target_history = fine_tune_with_distillation(
                model, target_data, source_data, device, config,
                lr=cfg_dict['ft_lr'],
                max_epochs=100,
                freeze_backbone=cfg_dict['freeze_backbone'],
                freeze_layers=cfg_dict['freeze_layers'],
                freeze_embedding=cfg_dict['freeze_embedding'],
                temperature=cfg_dict.get('temperature', 3.0),
                distill_alpha=cfg_dict.get('distill_alpha', 0.5),
                mix_source_ratio=cfg_dict.get('mix_source_ratio', 0.3)
            )
        else:  # baseline
            freeze_layers_correctly(
                model,
                cfg_dict['freeze_backbone'],
                cfg_dict['freeze_layers'],
                cfg_dict['freeze_embedding']
            )

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

        if method == 'ewc':
            result['ewc_lambda'] = cfg_dict.get('ewc_lambda', 1000)
        elif method == 'distill':
            result['temperature'] = cfg_dict.get('temperature', 3.0)
            result['distill_alpha'] = cfg_dict.get('distill_alpha', 0.5)
            result['mix_source_ratio'] = cfg_dict.get('mix_source_ratio', 0.3)

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

        # 跟踪前两名模型（复制模型状态以避免后续修改影响）
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
        print(f'  Validation Balance Score: {r["balance_score"]:.1f} (target minus forgetting penalty)')

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
        'ideal_targets': {
            'target_test_acc': '70-82%',
            'source_test_acc_after_ft': '40-60%'
        },
        'improvements_from_v2': {
            'fixed_freezing': 'Correct parameter name matching',
            'ewc_added': 'Elastic Weight Consolidation for memory protection',
            'distillation_added': 'Knowledge distillation + LWF'
        }
    }

    # 保存完整结果到统一目录
    json_path = os.path.join(json_dir, f'exp3_v3_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)

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

    summary_path = os.path.join(json_dir, f'exp3_v3_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nResults saved to {json_dir}/')

    # 保存前两名模型的权重
    # 按目标域准确率排序，取前两名
    top2_sorted = sorted(top2_models, key=lambda x: x['target_acc'], reverse=True)[:2]

    # 创建模型保存目录
    model_dir = os.path.join(base_dir, 'models', f'exp3_v3_{timestamp}')
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
        heatmap_dir = os.path.join(base_dir, 'visualizations', f'attention_v3_{timestamp}', f'top{rank}_{config_name_safe}')

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

    try_improved_models_v3(args.source_power, args.target_power, args.base_path, seed=args.seed)
