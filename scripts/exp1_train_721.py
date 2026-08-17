# -*- coding: utf-8 -*-
"""
主训练脚本
支持三种实验模式：
- Exp1: 单域基线 (mode='single')
- Exp2: 跨域无适应 (mode='cross_domain')
- Exp3: 域适应 (mode='domain_adapt')
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import argparse
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

import sys
# 支持本地和Cloud Studio运行
if os.path.exists('models'):
    sys.path.append('.')
else:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _project_paths import DATASETS_DIR, PROJECT_ROOT, RESULTS_DIR

from models.crossvit import CrossViTFaultDiagnosis
from modules.domain_adaptation import get_default_domain_adaptation_module
from data.data_processor_v2 import (
    load_csv_data, load_cross_domain_data,
    create_domain_adapt_loaders, MultiModalFaultDataset,
    UnlabeledMultiModalDataset
)


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


def plot_training_curves(history, save_path):
    """Plot training curves (loss and accuracy)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    if 'train_loss' in history and history['train_loss']:
        axes[0].plot(history['train_loss'], label='Train Loss', marker='o', markersize=3)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_title('Loss Curve')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy curve
    if 'train_acc' in history and history['train_acc']:
        axes[1].plot(history['train_acc'], label='Train Acc', marker='o', markersize=3)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['val_acc'], label='Val Acc', marker='s', markersize=3)
    axes[1].set_title('Accuracy Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_with_preds(model, dataloader, device):
    """评估模型并返回预测结果"""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                time_x, spec_x, y = batch
                time_x, spec_x = time_x.to(device), spec_x.to(device)
                y = y.to(device)
                outputs = model(time_x, spec_x)
            else:
                continue

            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = 100. * correct / total
    return accuracy, np.array(all_preds), np.array(all_labels)


def train_epoch_single(model, dataloader, criterion, optimizer, device):
    """单域训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, (time_x, spec_x, y) in enumerate(pbar):
        time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)

        # 检查输入数据
        if batch_idx == 0:
            print(f'[DEBUG] time_x: min={time_x.min():.4f}, max={time_x.max():.4f}, nan={torch.isnan(time_x).any()}')
            print(f'[DEBUG] spec_x: min={spec_x.min():.4f}, max={spec_x.max():.4f}, nan={torch.isnan(spec_x).any()}')

        optimizer.zero_grad()

        try:
            outputs = model(time_x, spec_x)
            if batch_idx == 0:
                print(f'[DEBUG] outputs: min={outputs.min():.4f}, max={outputs.max():.4f}, nan={torch.isnan(outputs).any()}')

            loss = criterion(outputs, y)
            if batch_idx == 0:
                print(f'[DEBUG] loss: {loss.item():.4f}, nan={torch.isnan(loss).any()}')

            # 检查NaN
            if torch.isnan(loss):
                print(f'[ERROR] NaN loss at batch {batch_idx}')
                print(f'  time_x nan: {torch.isnan(time_x).any()}')
                print(f'  spec_x nan: {torch.isnan(spec_x).any()}')
                print(f'  outputs nan: {torch.isnan(outputs).any()}')
                continue

            loss.backward()

            # 检查梯度
            has_nan_grad = False
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f'[ERROR] NaN gradient in {name}')
                    has_nan_grad = True
            if has_nan_grad:
                continue

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})

        except Exception as e:
            print(f'[ERROR] Exception at batch {batch_idx}: {e}')
            import traceback
            traceback.print_exc()
            continue

    if total == 0:
        print('[ERROR] No valid batches processed!')
        return 0.0, 0.0

    return total_loss / len(dataloader), 100. * correct / total


def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 3:
                time_x, spec_x, y = batch
                time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
                outputs = model(time_x, spec_x)
                loss = criterion(outputs, y)
            else:
                time_x, spec_x = batch
                time_x, spec_x = time_x.to(device), spec_x.to(device)
                continue

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return total_loss / len(dataloader), 100. * correct / total, all_preds, all_labels


def train_epoch_domain_adapt(model, source_loader, target_loader, da_module,
                             criterion_cls, criterion_domain, optimizer_model,
                             optimizer_disc, device, epoch, num_epochs):
    """域适应训练一个epoch"""
    model.train()
    da_module.domain_discriminator.train()

    total_loss_cls = 0
    total_loss_mmd = 0
    total_loss_adv = 0
    total_loss = 0
    correct = 0
    total = 0

    iter_source = iter(source_loader)
    iter_target = iter(target_loader)

    num_batches = min(len(source_loader), len(target_loader))
    pbar = tqdm(range(num_batches), desc='Training')

    progress = epoch / num_epochs

    for batch_idx in pbar:
        try:
            time_s, spec_s, y_s = next(iter_source)
        except StopIteration:
            iter_source = iter(source_loader)
            time_s, spec_s, y_s = next(iter_source)

        try:
            time_t, spec_t = next(iter_target)
        except StopIteration:
            iter_target = iter(target_loader)
            time_t, spec_t = next(iter_target)

        time_s, spec_s, y_s = time_s.to(device), spec_s.to(device), y_s.to(device)
        time_t, spec_t = time_t.to(device), spec_t.to(device)

        batch_size = min(time_s.size(0), time_t.size(0))
        time_s, spec_s, y_s = time_s[:batch_size], spec_s[:batch_size], y_s[:batch_size]
        time_t, spec_t = time_t[:batch_size], spec_t[:batch_size]

        # ==================
        # 1. 分类损失（源域）
        # ==================
        outputs_s, features_s = model(time_s, spec_s, return_features=True)
        loss_cls = criterion_cls(outputs_s, y_s)

        _, predicted = outputs_s.max(1)
        total += y_s.size(0)
        correct += predicted.eq(y_s).sum().item()

        # ==================
        # 2. 域适应损失
        # ==================
        _, features_t = model(time_t, spec_t, return_features=True)
        da_losses = da_module(features_s, features_t, progress=progress)

        loss_mmd = da_losses['mmd'] * da_module.lambda_mmd
        loss_adv = da_losses['adversarial']

        # ==================
        # 3. 更新域判别器
        # ==================
        # 域判别器需要最大化域分类准确率
        domain_labels_s = torch.zeros(batch_size, 1).to(device)
        domain_labels_t = torch.ones(batch_size, 1).to(device)

        with torch.no_grad():
            features_s_det = features_s.detach()
            features_t_det = features_t.detach()

        domain_out_s = da_module.domain_discriminator(features_s_det)
        domain_out_t = da_module.domain_discriminator(features_t_det)

        loss_disc_s = nn.functional.binary_cross_entropy(domain_out_s, domain_labels_s)
        loss_disc_t = nn.functional.binary_cross_entropy(domain_out_t, domain_labels_t)
        loss_disc = (loss_disc_s + loss_disc_t) / 2

        optimizer_disc.zero_grad()
        loss_disc.backward()
        optimizer_disc.step()

        # ==================
        # 4. 更新主模型
        # ==================
        loss = loss_cls + loss_mmd + loss_adv

        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()

        total_loss_cls += loss_cls.item()
        total_loss_mmd += loss_mmd.item()
        total_loss_adv += loss_adv.item()
        total_loss += loss.item()

        pbar.set_postfix({
            'L_cls': f'{loss_cls.item():.3f}',
            'L_mmd': f'{loss_mmd.item():.3f}',
            'L_adv': f'{loss_adv.item():.3f}',
            'Acc': f'{100.*correct/total:.1f}%'
        })

    return {
        'loss_cls': total_loss_cls / num_batches,
        'loss_mmd': total_loss_mmd / num_batches,
        'loss_adv': total_loss_adv / num_batches,
        'loss': total_loss / num_batches,
        'accuracy': 100. * correct / total
    }


def run_exp1_single_domain(power='1.0kW', base_path=None, num_epochs=50):
    """
    Exp1: 单域基线实验（CrossViT双分支 - 改进版）
    """
    if base_path is None:
        base_path = str(DATASETS_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\n' + '=' * 80)
    print(f'Exp1: Single Domain Baseline - {power}')
    print('=' * 80)

    # 加载数据 - 使用原始时间块 70:20:10 分割（训练:测试:验证）
    data = load_csv_data(
        os.path.join(base_path, f'dataset2_{power}.csv'),
        window_size=1024,
        stride=128,
        spec_size=None,
        test_size=0.10,
        val_size=0.05,
        split_mode='time_blocked'  # time-blocked 70:20:10 split
    )

    print(f'Dataset size: Train={len(data["y_train"])}, Val={len(data["y_val"])}, Test={len(data["y_test"])}')

    # 创建数据加载器（关闭数据增强，数据太少时增强有害）
    train_dataset = MultiModalFaultDataset(
        data['X_train_time'], data['X_train_spec'], data['y_train'], augment=False
    )
    val_dataset = MultiModalFaultDataset(
        data['X_val_time'], data['X_val_spec'], data['y_val'], augment=False
    )
    test_dataset = MultiModalFaultDataset(
        data['X_test_time'], data['X_test_spec'], data['y_test'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # CrossViT模型（简化版）
    model = CrossViTFaultDiagnosis(
        in_channels=data['n_channels'],
        num_classes=data['n_classes'],
        time_seq_len=data['window_size'],
        spec_height=data['spec_size'][0],
        spec_width=data['spec_size'][1],
        embed_dim=256,      # 增大embed_dim
        num_heads=8,
        num_layers=2,       # 减少层数
        dropout=0.2
    ).to(device)

    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0
    best_model_state = None

    # Training history for curves
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_single(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        print(f'Epoch {epoch+1}/{num_epochs} - Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, Best: {best_val_acc:.2f}%')

        # Early stopping
        if epoch > 20 and best_val_acc < 30:
            print(f'Training stuck, stopping early')
            break

    # 测试
    model.load_state_dict(best_model_state)
    test_acc, test_preds, test_labels = evaluate_with_preds(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.2f}%')

    # 生成可视化
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = str(RESULTS_DIR)
    json_dir = os.path.join(base_dir, 'json', 'exp1')
    viz_dir = os.path.join(base_dir, 'visualizations', f'exp1_{timestamp}')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    print(f'\nGenerating visualizations...')

    # 获取类别名称
    class_names = [f'C{i}' for i in range(data['n_classes'])]
    if 'fault_codes' in data:
        class_names = [str(c) for c in data['fault_codes']]

    # 生成训练曲线
    plot_training_curves(
        history,
        os.path.join(viz_dir, f'training_curves_{power.replace(".", "_")}.png')
    )
    print(f'  Training curves saved')

    # 混淆矩阵
    plot_confusion_matrix(
        test_labels, test_preds, class_names,
        f'EXP1: Single Domain Baseline - {power}',
        os.path.join(viz_dir, f'confusion_matrix_{power.replace(".", "_")}.png')
    )
    print(f'  Confusion matrix saved')

    # 保存JSON结果
    import json
    json_path = os.path.join(json_dir, f'exp1_{power}_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'experiment': 'EXP1: Single Domain Baseline',
            'power': power,
            'test_acc': float(test_acc),
            'timestamp': timestamp
        }, f, indent=2)
    print(f'  JSON saved to {json_path}')

    print(f'Visualizations saved to {viz_dir}/')

    return test_acc


def run_exp2_cross_domain(source_power='1.0kW', target_power='3.0kW',
                          base_path=None, num_epochs=50):
    """
    Exp2: 跨域无适应实验
    """
    if base_path is None:
        base_path = str(DATASETS_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\n' + '=' * 80)
    print(f'Exp2: Cross Domain Without Adaptation')
    print(f'Source: {source_power}, Target: {target_power}')
    print('=' * 80)

    # 加载源域数据 - 使用原始时间块 70:20:10 分割（训练:测试:验证）
    source_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{source_power}.csv'),
        window_size=1024,
        stride=128,
        spec_size=None,
        test_size=0.10,
        val_size=0.05,
        split_mode='time_blocked'  # time-blocked 70:20:10 split
    )

    # 加载目标域测试数据 - 使用原始时间块 70:20:10 分割（训练:测试:验证）
    target_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{target_power}.csv'),
        window_size=1024,
        stride=128,
        spec_size=None,
        test_size=0.10,
        val_size=0.05,
        split_mode='time_blocked'  # time-blocked 70:20:10 split
    )

    # 创建数据加载器
    train_dataset = MultiModalFaultDataset(
        source_data['X_train_time'], source_data['X_train_spec'],
        source_data['y_train'], augment=True
    )
    val_dataset = MultiModalFaultDataset(
        source_data['X_val_time'], source_data['X_val_spec'],
        source_data['y_val'], augment=False
    )
    test_dataset = MultiModalFaultDataset(
        target_data['X_test_time'], target_data['X_test_spec'],
        target_data['y_test'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 创建模型
    model = CrossViTFaultDiagnosis(
        in_channels=source_data['n_channels'],
        num_classes=source_data['n_classes'],
        time_seq_len=source_data['window_size'],
        spec_height=source_data['spec_size'][0],
        spec_width=source_data['spec_size'][1],
        embed_dim=128, num_heads=8, num_layers=4, dropout=0.1
    ).to(device)

    # 训练
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc = 0
    best_model_state = None

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch_single(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

    # 测试
    model.load_state_dict(best_model_state)

    # 源域测试
    source_test_dataset = MultiModalFaultDataset(
        source_data['X_test_time'], source_data['X_test_spec'],
        source_data['y_test'], augment=False
    )
    source_test_loader = DataLoader(source_test_dataset, batch_size=32, shuffle=False)
    _, source_test_acc, _, _ = evaluate(model, source_test_loader, criterion, device)

    # 目标域测试
    _, target_test_acc, _, _ = evaluate(model, test_loader, criterion, device)

    print(f'Source Test Accuracy: {source_test_acc:.2f}%')
    print(f'Target Test Accuracy: {target_test_acc:.2f}%')
    print(f'Accuracy Drop: {source_test_acc - target_test_acc:.2f}%')

    return source_test_acc, target_test_acc


def run_exp3_domain_adaptation(source_power='1.0kW', target_power='3.0kW',
                               base_path=None, num_epochs=50,
                               lambda_mmd=0.1, lambda_adv=0.1):
    """
    Exp3: 域适应实验
    """
    if base_path is None:
        base_path = str(DATASETS_DIR)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('\n' + '=' * 80)
    print(f'Exp3: Domain Adaptation')
    print(f'Source: {source_power}, Target: {target_power}')
    print(f'Lambda MMD: {lambda_mmd}, Lambda Adv: {lambda_adv}')
    print('=' * 80)

    # 加载跨域数据
    cross_data = load_cross_domain_data(
        source_power, target_power, base_path,
        window_size=1024, stride=512, spec_size=None
    )
    loaders = create_domain_adapt_loaders(cross_data, batch_size=32, augment=True)

    # 创建模型
    source_data = cross_data['source']
    model = CrossViTFaultDiagnosis(
        in_channels=source_data['n_channels'],
        num_classes=source_data['n_classes'],
        time_seq_len=source_data['window_size'],
        spec_height=source_data['spec_size'][0],
        spec_width=source_data['spec_size'][1],
        embed_dim=128, num_heads=8, num_layers=4, dropout=0.1
    ).to(device)

    # 创建域适应模块
    da_module = get_default_domain_adaptation_module(feature_dim=256)
    da_module.lambda_mmd = lambda_mmd
    da_module.lambda_adv = lambda_adv
    da_module = da_module.to(device)

    # 优化器
    criterion_cls = nn.CrossEntropyLoss()
    criterion_domain = nn.BCELoss()

    optimizer_model = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    optimizer_disc = optim.Adam(da_module.domain_discriminator.parameters(), lr=0.001, weight_decay=1e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, mode='min', factor=0.5, patience=5)

    best_val_acc = 0
    best_model_state = None
    best_da_state = None

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        metrics = train_epoch_domain_adapt(
            model, loaders['source_train'], loaders['target_train'],
            da_module, criterion_cls, criterion_domain,
            optimizer_model, optimizer_disc, device, epoch, num_epochs
        )

        # 验证
        val_loss, val_acc, _, _ = evaluate(model, loaders['source_val'], criterion_cls, device)
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            best_da_state = da_module.state_dict()

        print(f'Val Acc: {val_acc:.2f}%, Best: {best_val_acc:.2f}%')

    # 测试
    model.load_state_dict(best_model_state)
    da_module.load_state_dict(best_da_state)

    # 源域测试
    _, source_test_acc, _, _ = evaluate(model, loaders['source_test'], criterion_cls, device)

    # 目标域测试
    _, target_test_acc, _, _ = evaluate(model, loaders['target_test'], criterion_cls, device)

    print(f'\nSource Test Accuracy: {source_test_acc:.2f}%')
    print(f'Target Test Accuracy: {target_test_acc:.2f}%')

    return source_test_acc, target_test_acc


def main():
    parser = argparse.ArgumentParser(description='CrossViT Fault Diagnosis with Domain Adaptation')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'cross_domain', 'domain_adapt'],
                       help='Experiment mode')
    parser.add_argument('--source_power', type=str, default='1.0kW',
                       choices=['1.0kW', '1.5kW', '3.0kW'],
                       help='Source domain power')
    parser.add_argument('--target_power', type=str, default='3.0kW',
                       choices=['1.0kW', '1.5kW', '3.0kW'],
                       help='Target domain power')
    parser.add_argument('--base_path', type=str,
                       default=str(DATASETS_DIR),
                       help='Data base path')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lambda_mmd', type=float, default=0.1,
                       help='MMD loss weight')
    parser.add_argument('--lambda_adv', type=float, default=0.1,
                       help='Adversarial loss weight')

    args = parser.parse_args()

    if args.mode == 'single':
        run_exp1_single_domain(args.source_power, args.base_path, args.num_epochs)

    elif args.mode == 'cross_domain':
        run_exp2_cross_domain(args.source_power, args.target_power,
                             args.base_path, args.num_epochs)

    elif args.mode == 'domain_adapt':
        run_exp3_domain_adaptation(args.source_power, args.target_power,
                                  args.base_path, args.num_epochs,
                                  args.lambda_mmd, args.lambda_adv)


if __name__ == '__main__':
    main()
