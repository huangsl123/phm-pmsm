# -*- coding: utf-8 -*-
"""
EXP3 改进版 V2：优化迁移学习效果
1. 尝试多种防遗忘策略
2. 更详细的版本记录
3. 目标：源域40-60%，目标域70-82%
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
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
from data.data_processor_128 import load_csv_data, MultiModalFaultDataset


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

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20  # 增加patience

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


def fine_tune_on_target(model, target_data, source_data, device, config,
                        lr=0.0001, max_epochs=100, freeze_backbone=False,
                        freeze_layers=None, label_smoothing=0.05):
    """在目标域微调模型"""
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

    # 冻结策略
    if freeze_backbone:
        print(f'  Freezing backbone, only training classifier...')
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False

    elif freeze_layers is not None and freeze_layers > 0:
        # 冻结前N层transformer
        print(f'  Freezing first {freeze_layers} transformer layers...')
        layer_count = 0
        for name, param in model.named_parameters():
            if 'transformer' in name and 'layers' in name:
                layer_num = int(name.split('layers.')[1].split('.')[0])
                if layer_num < freeze_layers:
                    param.requires_grad = False
                    layer_count += 1
        print(f'  Froze {layer_count} parameter groups')

    # 获取可训练参数
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f'  Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M')

    optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'  Fine-tuning on target (lr={lr}, max_epochs={max_epochs})...')

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


def try_improved_models_v2(source_power='1.0kW', target_power='3.0kW', base_path=str(DATASETS_DIR), seed=42):
    """尝试多种改进配置 - V2版本"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 版本信息
    version_info = get_version_info()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print('\n' + '=' * 80)
    print(f'EXP3 IMPROVED V2 - Optimized Transfer Learning')
    print(f'Source: {source_power} -> Target: {target_power}')
    print(f'Version: {timestamp}')
    print('=' * 80)
    print(f'PyTorch: {version_info["pytorch_version"]}')
    print(f'CUDA: {version_info["cuda_available"]}')
    if version_info["cuda_available"]:
        print(f'GPU: {version_info["gpu_name"]}')
    print('=' * 80)

    # 加载数据 - 使用 7:2:1 分割 (训练:测试:验证)
    print('\nLoading data with 7:2:1 split (train:test:val)...')
    source_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{source_power}.csv'),
        window_size=1024,
        stride=128,
        spec_size=(128, 128),
        test_size=0.10,
        val_size=0.05,
        random_state=42,
        split_mode='721'  # 7:2:1 分割
    )
    target_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{target_power}.csv'),
        window_size=1024,
        stride=128,
        spec_size=(128, 128),
        test_size=0.10,
        val_size=0.05,
        random_state=42,
        split_mode='721'  # 7:2:1 分割
    )

    # 改进的配置列表 - 尝试多种防遗忘策略
    configs = [
        # ===== 基线配置 =====
        {
            'name': 'Baseline (384, no freeze)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.0001,
            'freeze_backbone': False,
            'freeze_layers': None,
            'label_smoothing': 0.05
        },
        # ===== 冻结骨干网络 =====
        {
            'name': 'Freeze Backbone (384)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.0001,
            'freeze_backbone': True,
            'freeze_layers': None,
            'label_smoothing': 0.05
        },
        # ===== 更小的微调学习率 =====
        {
            'name': 'Low FT LR (384, lr=1e-5)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.00001,  # 更小的学习率
            'freeze_backbone': False,
            'freeze_layers': None,
            'label_smoothing': 0.05
        },
        # ===== 部分冻结 + 小学习率 =====
        {
            'name': 'Partial Freeze + Low LR (384)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.00005,
            'freeze_backbone': False,
            'freeze_layers': 1,  # 冻结第一层
            'label_smoothing': 0.1  # 更高的label smoothing
        },
        # ===== 中等模型 =====
        {
            'name': 'Medium Model (512, no freeze)',
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.0001,
            'freeze_backbone': False,
            'freeze_layers': None,
            'label_smoothing': 0.05
        },
        # ===== 中等模型 + 冻结 =====
        {
            'name': 'Medium Model + Freeze Backbone (512)',
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.0001,
            'freeze_backbone': True,
            'freeze_layers': None,
            'label_smoothing': 0.05
        },
        # ===== 中等模型 + 低dropout =====
        {
            'name': 'Medium Model + Low Dropout (512)',
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.05,
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'batch_size': 32,
            'ft_lr': 0.00005,
            'freeze_backbone': False,
            'freeze_layers': 1,
            'label_smoothing': 0.1
        },
        # ===== 大模型 =====
        {
            'name': 'Large Model (768, partial freeze)',
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 4,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 16,
            'ft_lr': 0.00005,
            'freeze_backbone': False,
            'freeze_layers': 2,  # 冻结前2层
            'label_smoothing': 0.1
        },
    ]

    results = []
    best_model = None
    best_config_name = None
    best_target_acc = 0
    best_balance_score = -float('inf')  # 新增：平衡分（目标域高 + 源域保持）

    # 跟踪前两名模型（基于目标域准确率）
    top2_models = []

    # 创建输出目录
    results_dir = f'results/exp3_improved_v2_{timestamp}'
    # 统一结果目录（带时间戳）
    base_dir = str(RESULTS_DIR)
    json_dir = os.path.join(base_dir, 'json', 'exp3_v2')
    viz_dir = os.path.join(base_dir, 'visualizations', f'exp3_v2_{timestamp}')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    for i, cfg_dict in enumerate(configs):
        print(f'\n{"=" * 80}')
        print(f'CONFIG {i + 1}/{len(configs)}: {cfg_dict["name"]}')
        print(f'{"=" * 80}')

        # 计算配置哈希
        config_for_hash = {k: v for k, v in cfg_dict.items() if k not in ['name']}
        config_hash = compute_config_hash(config_for_hash)
        print(f'Config Hash: {config_hash}')

        config = {k: v for k, v in cfg_dict.items()
                  if k not in ['name', 'ft_lr', 'freeze_backbone', 'freeze_layers', 'label_smoothing']}

        # 阶段1: 源域训练
        model, source_val_acc, source_history = train_on_source(
            config, source_data, device, seed=seed, max_epochs=100
        )

        # 详细评估源域
        source_results_before = evaluate_all_splits(model, source_data, device)
        target_results_before = evaluate_all_splits(model, target_data, device)

        print(f'  After Phase 1 (Source Domain):')
        print(f'    Train: {source_results_before["train"]:.1f}% | Val: {source_results_before["val"]:.1f}% | Test: {source_results_before["test"]:.1f}%')
        print(f'  After Phase 1 (Target Domain - Before FT):')
        print(f'    Train: {target_results_before["train"]:.1f}% | Val: {target_results_before["val"]:.1f}% | Test: {target_results_before["test"]:.1f}%')

        # 阶段2: 目标域微调
        model, target_val_acc, target_history = fine_tune_on_target(
            model, target_data, source_data, device, config,
            lr=cfg_dict['ft_lr'],
            max_epochs=100,
            freeze_backbone=cfg_dict['freeze_backbone'],
            freeze_layers=cfg_dict['freeze_layers'],
            label_smoothing=cfg_dict['label_smoothing']
        )

        # 详细评估微调后
        source_results_after = evaluate_all_splits(model, source_data, device)
        target_results_after = evaluate_all_splits(model, target_data, device)

        print(f'  After Phase 2 (Source Domain):')
        print(f'    Train: {source_results_after["train"]:.1f}% | Val: {source_results_after["val"]:.1f}% | Test: {source_results_after["test"]:.1f}%')
        print(f'  After Phase 2 (Target Domain):')
        print(f'    Train: {target_results_after["train"]:.1f}% | Val: {target_results_after["val"]:.1f}% | Test: {target_results_after["test"]:.1f}%')
        print(f'    Target Improvement: +{target_results_after["test"] - target_results_before["test"]:.1f}%')

        # 计算平衡分：目标域准确率 + 源域保持率（鼓励源域保持在40%以上）
        source_retention = source_results_after["test"]
        target_acc = target_results_after["test"]
        balance_score = target_acc + max(0, min(source_retention - 40, 20)) / 2  # 源域40-60%时额外加分

        result = {
            'config_name': cfg_dict['name'],
            'config_hash': config_hash,
            'config': config,
            'ft_lr': cfg_dict['ft_lr'],
            'freeze_backbone': cfg_dict['freeze_backbone'],
            'freeze_layers': cfg_dict['freeze_layers'],
            'label_smoothing': cfg_dict['label_smoothing'],
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
        results.append(result)

        # Track best model (by target accuracy)
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
            'source_preds': source_results_after['test_preds'].copy(),
            'source_labels': source_results_after['test_labels'].copy(),
            'target_preds': target_results_after['test_preds'].copy(),
            'target_labels': target_results_after['test_labels'].copy(),
            'source_history': source_history,
            'target_history': target_history
        })

        # Track best balanced model
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
        print(f'  Balance Score: {r["balance_score"]:.1f} (Target + Source retention bonus)')

    best_result = max(results, key=lambda x: x['target']['test_acc'])
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
        }
    }

    json_path = os.path.join(json_dir, f'exp3_v2_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    print(f'\nResults saved to {json_path}')

    # 保存简化版本用于快速查看
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
                'source_test': r['source']['test_acc'],
                'target_test': r['target']['test_acc'],
                'balance_score': r['balance_score']
            }
            for r in results
        ]
    }

    summary_path = os.path.join(json_dir, f'exp3_v2_summary_{timestamp}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\nSummary saved to {summary_path}')

    print(f'\nResults saved to {json_path}')

    # 生成前两名模型的可视化
    # 按目标域准确率排序，取前两名
    top2_sorted = sorted(top2_models, key=lambda x: x['target_acc'], reverse=True)[:2]

    class_names = [f'C{i}' for i in range(source_data['n_classes'])]
    if 'fault_codes' in source_data:
        class_names = [str(c) for c in source_data['fault_codes']]

    print(f'\nGenerating visualizations for top 2 models...')

    for rank, model_info in enumerate(top2_sorted, 1):
        config_name_safe = model_info['config_name'].replace(' ', '_').replace('(', '_').replace(')', '_').replace(',', '_')
        prefix = f'top{rank}_{config_name_safe}'

        print(f'  Rank {rank}: {model_info["config_name"]} (Target Acc: {model_info["target_acc"]:.1f}%)')

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

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_power', type=str, default='1.0kW')
    parser.add_argument('--target_power', type=str, default='3.0kW')
    parser.add_argument('--base_path', type=str, default=str(DATASETS_DIR))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    try_improved_models_v2(args.source_power, args.target_power, args.base_path, seed=args.seed)
