# -*- coding: utf-8 -*-
"""
EXP3 改进版：尝试多种优化策略
1. 更大的模型 (embed_dim=512)
2. 更多的训练轮数
3. 不同的微调学习率
4. 降低 dropout
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from _project_paths import DATASETS_DIR, PROJECT_ROOT, RESULTS_DIR

from models.crossvit import CrossViTFaultDiagnosis
from data.data_processor_128 import load_csv_data, MultiModalFaultDataset


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
    """Plot training curves"""
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


def plot_training_curves_dual(source_history, target_history, save_path):
    """Plot training curves for both phases (source and target)"""
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

    train_dataset = MultiModalFaultDataset(
        source_data['X_train_time'], source_data['X_train_spec'],
        source_data['y_train'], augment=False
    )
    val_dataset = MultiModalFaultDataset(
        source_data['X_val_time'], source_data['X_val_spec'],
        source_data['y_val'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

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
    patience = 15

    # Record training history
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
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    model.load_state_dict(best_model_state)
    print(f'  Source training complete: Val={best_val_acc:.2f}%')
    return model, best_val_acc, history


def fine_tune_on_target(model, target_data, source_data, device, config,
                       lr=0.0001, max_epochs=100, freeze_backbone=False):
    """在目标域微调模型"""
    train_dataset = MultiModalFaultDataset(
        target_data['X_train_time'], target_data['X_train_spec'],
        target_data['y_train'], augment=False
    )
    val_dataset = MultiModalFaultDataset(
        target_data['X_val_time'], target_data['X_val_spec'],
        target_data['y_val'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    # 可选：冻结骨干网络
    if freeze_backbone:
        print(f'  Freezing backbone, only training classifier...')
        for name, param in model.named_parameters():
            if 'classifier' not in name and 'fc' not in name:
                param.requires_grad = False

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 15

    # Record training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'  Fine-tuning on target (lr={lr}, max_epochs={max_epochs}, freeze={freeze_backbone})...')

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
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_state)
    print(f'  Target fine-tuning complete: Val={best_val_acc:.2f}%')
    return model, best_val_acc, history


def evaluate(model, data, device):
    """评估模型"""
    test_dataset = MultiModalFaultDataset(
        data['X_test_time'], data['X_test_spec'],
        data['y_test'], augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for time_x, spec_x, y in test_loader:
            time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)
            outputs = model(time_x, spec_x)
            _, predicted = outputs.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()

    return 100. * test_correct / test_total


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
    else:  # test
        dataset = MultiModalFaultDataset(
            data['X_test_time'], data['X_test_spec'],
            data['y_test'], augment=False
        )

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

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


def try_improved_models(source_power='1.0kW', target_power='3.0kW', base_path=str(DATASETS_DIR), seed=42):
    """尝试多种改进配置"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('\n' + '=' * 80)
    print(f'EXP3 IMPROVED: Trying Different Strategies')
    print(f'Source: {source_power} -> Target: {target_power}')
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
        split_mode='721'  # 7:2:1 分割
    )
    target_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{target_power}.csv'),
        window_size=1024,
        stride=128,
        spec_size=(128, 128),
        test_size=0.10,
        val_size=0.05,
        split_mode='721'  # 7:2:1 分割
    )

    # 尝试不同配置
    configs = [
        {
            'name': 'Baseline (embed_dim=384)',
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.0001,
            'freeze': False
        },
        {
            'name': 'Larger Model (embed_dim=512)',
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.0001,
            'freeze': False
        },
        {
            'name': 'Larger Model + Lower Dropout',
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.05,
            'lr': 0.0001,
            'weight_decay': 0.0001,
            'batch_size': 32,
            'ft_lr': 0.0001,
            'freeze': False
        },
        {
            'name': 'Larger Model + Higher FT LR',
            'embed_dim': 512,
            'num_heads': 8,
            'num_layers': 3,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32,
            'ft_lr': 0.0005,
            'freeze': False
        },
        {
            'name': 'Largest Model (embed_dim=768)',
            'embed_dim': 768,
            'num_heads': 12,
            'num_layers': 4,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 16,
            'ft_lr': 0.0001,
            'freeze': False
        },
    ]

    results = []
    best_model = None
    best_config_name = None
    best_target_acc = 0

    # 跟踪前两名模型（基于目标域准确率）
    top2_models = []

    for i, cfg_dict in enumerate(configs):
        print(f'\n{"=" * 80}')
        print(f'CONFIG {i + 1}/{len(configs)}: {cfg_dict["name"]}')
        print(f'{"=" * 80}')

        config = {k: v for k, v in cfg_dict.items() if k not in ['name', 'ft_lr', 'freeze']}

        # 阶段1: 源域训练
        model, source_val_acc, source_history = train_on_source(config, source_data, device, seed=seed, max_epochs=100)

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
            lr=cfg_dict['ft_lr'], max_epochs=100, freeze_backbone=cfg_dict['freeze']
        )

        # 详细评估微调后
        source_results_after = evaluate_all_splits(model, source_data, device)
        target_results_after = evaluate_all_splits(model, target_data, device)

        print(f'  After Phase 2 (Source Domain):')
        print(f'    Train: {source_results_after["train"]:.1f}% | Val: {source_results_after["val"]:.1f}% | Test: {source_results_after["test"]:.1f}%')
        print(f'  After Phase 2 (Target Domain):')
        print(f'    Train: {target_results_after["train"]:.1f}% | Val: {target_results_after["val"]:.1f}% | Test: {target_results_after["test"]:.1f}%')
        print(f'    Target Improvement: +{target_results_after["test"] - target_results_before["test"]:.1f}%')
        print(f'    vs EXP2 (9.2%): +{target_results_after["test"] - 9.2:.1f}%')

        result = {
            'config_name': cfg_dict['name'],
            'config': config,
            'ft_lr': cfg_dict['ft_lr'],
            'freeze': cfg_dict['freeze'],
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
            'vs_exp2': float(target_results_after['test'] - 9.2)
        }
        results.append(result)

        # Track best model
        if target_results_after['test'] > best_target_acc:
            best_target_acc = target_results_after['test']
            best_model = model
            best_config_name = cfg_dict['name']
            # Save best predictions for visualization
            best_source_preds = source_results_after['test_preds']
            best_source_labels = source_results_after['test_labels']
            best_target_preds = target_results_after['test_preds']
            best_target_labels = target_results_after['test_labels']
            best_source_history = source_history
            best_target_history = target_history

        # 跟踪前两名模型
        top2_models.append({
            'target_acc': target_results_after['test'],
            'config_name': cfg_dict['name'],
            'source_preds': source_results_after['test_preds'].copy(),
            'source_labels': source_results_after['test_labels'].copy(),
            'target_preds': target_results_after['test_preds'].copy(),
            'target_labels': target_results_after['test_labels'].copy(),
            'source_history': source_history,
            'target_history': target_history
        })

    # 最终报告
    print('\n' + '=' * 80)
    print('ALL RESULTS SUMMARY')
    print('=' * 80)

    for r in results:
        print(f'\n{r["config_name"]}:')
        print(f'  Source - Train: {r["source"]["train_acc"]:.1f}% | Val: {r["source"]["val_acc"]:.1f}% | Test: {r["source"]["test_acc"]:.1f}%')
        print(f'  Target - Train: {r["target"]["train_acc"]:.1f}% | Val: {r["target"]["val_acc"]:.1f}% | Test: {r["target"]["test_acc"]:.1f}%')
        print(f'  Target Improvement: +{r["target"]["improvement"]:.1f}% (vs EXP2: +{r["vs_exp2"]:.1f}%)')

    best_result = max(results, key=lambda x: x['target']['test_acc'])
    print(f'\n🏆 Best config: {best_result["config_name"]}')
    print(f'   Source - Train: {best_result["source"]["train_acc"]:.1f}% | Val: {best_result["source"]["val_acc"]:.1f}% | Test: {best_result["source"]["test_acc"]:.1f}%')
    print(f'   Target - Train: {best_result["target"]["train_acc"]:.1f}% | Val: {best_result["target"]["val_acc"]:.1f}% | Test: {best_result["target"]["test_acc"]:.1f}%')

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/exp3_improved_{timestamp}'
    # 统一结果目录（带时间戳）
    base_dir = str(RESULTS_DIR)
    json_dir = os.path.join(base_dir, 'json', 'exp3_v1')
    viz_dir = os.path.join(base_dir, 'visualizations', f'exp3_v1_{timestamp}')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    json_path = os.path.join(json_dir, f'exp3_v1_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'all_results': results,
            'best_result': best_result,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f'\nResults saved to {json_path}')

    # 生成前两名模型的可视化
    # 按目标域准确率排序，取前两名
    top2_sorted = sorted(top2_models, key=lambda x: x['target_acc'], reverse=True)[:2]

    class_names = [f'C{i}' for i in range(source_data['n_classes'])]
    if 'fault_codes' in source_data:
        class_names = [str(c) for c in source_data['fault_codes']]

    print(f'\nGenerating visualizations for top 2 models...')

    for rank, model_info in enumerate(top2_sorted, 1):
        config_name_safe = model_info['config_name'].replace(' ', '_').replace('(', '_').replace(')', '_')
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

        # 训练曲线（源域和目标域）
        plot_training_curves_dual(
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

    try_improved_models(args.source_power, args.target_power, args.base_path, seed=args.seed)
