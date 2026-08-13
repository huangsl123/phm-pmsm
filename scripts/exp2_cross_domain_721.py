# -*- coding: utf-8 -*-
"""
EXP2: 跨域泛化实验（无适应）
目标：测试模型在1.0kW上训练后，直接在3.0kW上的泛化能力
预期：准确率会显著下降，证明域偏移问题
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


def evaluate_with_preds(model, target_data, device):
    """评估模型并返回预测结果"""
    test_dataset = MultiModalFaultDataset(
        target_data['X_test_time'], target_data['X_test_spec'],
        target_data['y_test'], augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for time_x, spec_x, y in test_loader:
            time_x, spec_x = time_x.to(device), spec_x.to(device)
            y = y.to(device)

            outputs = model(time_x, spec_x)
            _, predicted = outputs.max(1)

            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()

    test_acc = 100. * test_correct / test_total
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return test_acc, all_preds, all_labels


class WarmupCosineScheduler:
    """带预热的余弦退火学习率调度器"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr


def train_on_source(config, source_data, device, max_epochs=100, warmup_epochs=10):
    """在源域（1.0kW）上训练模型"""

    # 创建数据加载器（只用训练集）
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

    # 创建模型
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
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=warmup_epochs, total_epochs=max_epochs,
                                     min_lr=config['lr'] * 0.01)

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 15

    # Training history for curves
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print(f'\n{"=" * 60}')
    print(f'Training on source domain (1.0kW)')
    print(f'{"=" * 60}')
    print(f'Max epochs: {max_epochs}, Warmup epochs: {warmup_epochs}')

    for epoch in range(max_epochs):
            print(f'Starting epoch {epoch+1}/{max_epochs}...')
            # 训练
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            batch_count = 0

            for batch_idx, (time_x, spec_x, y) in enumerate(train_loader):
                if batch_idx == 0:
                    print(f'  First batch: time_x shape={time_x.shape}, spec_x shape={spec_x.shape}, y shape={y.shape}')

                time_x, spec_x, y = time_x.to(device), spec_x.to(device), y.to(device)

                optimizer.zero_grad()
                outputs = model(time_x, spec_x)
                loss = criterion(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += y.size(0)
                train_correct += predicted.eq(y).sum().item()
                batch_count += 1

            train_acc = 100. * train_correct / train_total
            print(f'  Epoch {epoch+1} completed: batches={batch_count}, train_acc={train_acc:.2f}%')

            # 验证
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

            # Record history
            history['train_loss'].append(train_loss / len(train_loader))
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)

            current_lr = scheduler.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1}/{max_epochs} | LR: {current_lr:.6f} | '
                      f'Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f'  Early stopping at epoch {epoch + 1}')
                break

    # 加载最佳模型
    model.load_state_dict(best_model_state)

    print(f'\nSource domain training complete: Val Acc = {best_val_acc:.2f}%')

    return model, history


def evaluate_on_target(model, target_data, device, domain_name='Target'):
    """在目标域上评估模型"""

    test_dataset = MultiModalFaultDataset(
        target_data['X_test_time'], target_data['X_test_spec'],
        target_data['y_test'], augment=False
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

    test_acc = 100. * test_correct / test_total

    return test_acc


def run_exp2(source_power='1.0kW', target_power='3.0kW', base_path=str(DATASETS_DIR), config=None, seed=42):
    """
    EXP2: 跨域泛化实验（无适应）

    用源域数据训练，直接在目标域测试

    Parameters:
    -----------
    seed : int
        随机种子，用于结果复现
    """
    # 设置随机种子
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} (seed={seed})')

    print('\n' + '=' * 80)
    print(f'EXP2: Cross-Domain Generalization (No Adaptation)')
    print(f'Source (Train): {source_power}')
    print(f'Target (Test):  {target_power}')
    print('=' * 80)

    # 时间戳（用于结果和可视化）
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 使用最佳配置
    if config is None:
        config = {
            'window_size': 1024,
            'stride': 128,
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32
        }

    print(f'\nConfig: {config}')
    """
    EXP2: 跨域泛化实验（无适应）

    用源域数据训练，直接在目标域测试

    Parameters:
    -----------
    seed : int
        随机种子，用于结果复现
    """
    # 设置随机种子
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device} (seed={seed})')

    print('\n' + '=' * 80)
    print(f'EXP2: Cross-Domain Generalization (No Adaptation)')
    print(f'Source (Train): {source_power}')
    print(f'Target (Test):  {target_power}')
    print('=' * 80)

    # 使用最佳配置
    if config is None:
        config = {
            'window_size': 1024,
            'stride': 128,
            'embed_dim': 384,
            'num_heads': 4,
            'num_layers': 2,
            'dropout': 0.1,
            'lr': 0.0001,
            'weight_decay': 0.0005,
            'batch_size': 32
        }

    print(f'\nConfig: {config}')

    # 加载源域数据（用于训练）
    print(f'\n{"=" * 60}')
    print(f'Loading source domain data: {source_power}')
    print(f'{"=" * 60}')

    source_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{source_power}.csv'),
        window_size=config['window_size'],
        stride=config['stride'],
        spec_size=(128, 128),
        test_size=0.10,
        val_size=0.05,
        split_mode='721'  # 7:2:1 分割
    )
    print(f'Source: Train={len(source_data["y_train"])}, Val={len(source_data["y_val"])}, '
          f'Test={len(source_data["y_test"])}')

    # 加载目标域数据（用于测试）
    print(f'\n{"=" * 60}')
    print(f'Loading target domain data: {target_power}')
    print(f'{"=" * 60}')

    target_data = load_csv_data(
        os.path.join(base_path, f'dataset2_{target_power}.csv'),
        window_size=config['window_size'],
        stride=config['stride'],
        spec_size=(128, 128),
        test_size=0.10,
        val_size=0.05,
        split_mode='721'  # 7:2:1 分割
    )
    print(f'Target: Train={len(target_data["y_train"])}, Val={len(target_data["y_val"])}, '
          f'Test={len(target_data["y_test"])}')

    # 在源域上训练
    model, train_history = train_on_source(config, source_data, device, max_epochs=100, warmup_epochs=10)

    # 在源域测试集上评估
    print(f'\n{"=" * 60}')
    print(f'Evaluating on source test set...')
    source_test_acc, source_preds, source_labels = evaluate_with_preds(model, source_data, device)
    print(f'Source Test Acc: {source_test_acc:.2f}%')

    # 在目标域测试集上评估
    print(f'\n{"=" * 60}')
    print(f'Evaluating on target test set...')
    target_test_acc, target_preds, target_labels = evaluate_with_preds(model, target_data, device)
    print(f'Target Test Acc: {target_test_acc:.2f}%')

    # 计算性能下降
    performance_drop = source_test_acc - target_test_acc

    # 结果报告
    print('\n' + '=' * 80)
    print('EXP2 RESULTS')
    print('=' * 80)
    print(f'Source ({source_power}) Test Acc: {source_test_acc:.2f}%')
    print(f'Target ({target_power}) Test Acc: {target_test_acc:.2f}%')
    print(f'Performance Drop: {performance_drop:.2f}%')
    print('=' * 80)

    if performance_drop > 15:
        print(f'\n✓ Significant domain shift detected! ({performance_drop:.2f}% drop)')
        print('  This justifies the need for domain adaptation methods.')
    else:
        print(f'\n✗ Minimal domain shift ({performance_drop:.2f}% drop)')

    # 保存结果到统一目录
    base_dir = str(RESULTS_DIR)
    json_dir = os.path.join(base_dir, 'json', 'exp2')
    viz_dir = os.path.join(base_dir, 'visualizations', f'exp2_{timestamp}')
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    filename = f'exp2_{source_power}_to_{target_power}_{timestamp}.json'
    json_path = os.path.join(json_dir, filename)

    with open(json_path, 'w') as f:
        json.dump({
            'experiment': 'EXP2: Cross-Domain (No Adaptation)',
            'source_power': source_power,
            'target_power': target_power,
            'config': config,
            'source_test_acc': source_test_acc,
            'target_test_acc': target_test_acc,
            'performance_drop': performance_drop,
            'source_samples': {
                'train': len(source_data['y_train']),
                'val': len(source_data['y_val']),
                'test': len(source_data['y_test'])
            },
            'target_samples': {
                'train': len(target_data['y_train']),
                'val': len(target_data['y_val']),
                'test': len(target_data['y_test'])
            },
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f'\nResults saved to {json_path}')

    print(f'\nGenerating visualizations...')

    # 获取类别名称
    class_names = [f'C{i}' for i in range(source_data['n_classes'])]
    if 'fault_codes' in source_data:
        class_names = [str(c) for c in source_data['fault_codes']]

    # 生成训练曲线
    plot_training_curves(
        train_history,
        os.path.join(viz_dir, 'training_curves.png')
    )
    print(f'  Training curves saved')

    # 源域混淆矩阵
    plot_confusion_matrix(
        source_labels, source_preds, class_names,
        f'EXP2: Source Domain ({source_power}) - No Adaptation',
        os.path.join(viz_dir, f'source_confusion_matrix_{source_power.replace(".", "_")}.png')
    )
    print(f'  Source confusion matrix saved')

    # 目标域混淆矩阵
    plot_confusion_matrix(
        target_labels, target_preds, class_names,
        f'EXP2: Target Domain ({target_power}) - No Adaptation',
        os.path.join(viz_dir, f'target_confusion_matrix_{target_power.replace(".", "_")}.png')
    )
    print(f'  Target confusion matrix saved')

    print(f'Visualizations saved to {viz_dir}/')

    return {
        'source_test_acc': source_test_acc,
        'target_test_acc': target_test_acc,
        'performance_drop': performance_drop
    }


def run_exp2_multiple(source_power='1.0kW', target_power='3.0kW', base_path=str(DATASETS_DIR),
                      config=None, runs=5, start_seed=42):
    """
    EXP2: 多次运行取平均

    Parameters:
    -----------
    runs : int
        运行次数
    start_seed : int
        起始随机种子，每次运行会递增
    """
    print('\n' + '=' * 80)
    print(f'EXP2: Cross-Domain Generalization ({runs} runs)')
    print(f'Source (Train): {source_power}')
    print(f'Target (Test):  {target_power}')
    print('=' * 80)

    results = []

    for run in range(runs):
        print(f'\n{"=" * 80}')
        print(f'Run {run + 1}/{runs} (seed={start_seed + run})')
        print(f'{"=" * 80}')

        result = run_exp2(source_power, target_power, base_path, config, seed=start_seed + run)
        results.append(result)

    # 统计结果
    source_accs = [r['source_test_acc'] for r in results]
    target_accs = [r['target_test_acc'] for r in results]
    drops = [r['performance_drop'] for r in results]

    print('\n' + '=' * 80)
    print(f'EXP2 SUMMARY ({runs} runs)')
    print('=' * 80)
    print(f'\nSource ({source_power}) Test Acc:')
    print(f'  Mean: {np.mean(source_accs):.2f}% ± {np.std(source_accs):.2f}%')
    print(f'  Range: [{min(source_accs):.2f}%, {max(source_accs):.2f}%]')

    print(f'\nTarget ({target_power}) Test Acc:')
    print(f'  Mean: {np.mean(target_accs):.2f}% ± {np.std(target_accs):.2f}%')
    print(f'  Range: [{min(target_accs):.2f}%, {max(target_accs):.2f}%]')

    print(f'\nPerformance Drop:')
    print(f'  Mean: {np.mean(drops):.2f}% ± {np.std(drops):.2f}%')
    print(f'  Range: [{min(drops):.2f}%, {max(drops):.2f}%]')

    if np.mean(drops) > 15:
        print(f'\n✓ Significant domain shift detected! (平均下降 {np.mean(drops):.2f}%)')
        print('  这证明了域适应方法的必要性。')
    else:
        print(f'\n✗ Minimal domain shift (平均下降 {np.mean(drops):.2f}%)')

    # 保存结果
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'exp2_{source_power}_to_{target_power}_{runs}runs_{timestamp}.json'

    with open(os.path.join(results_dir, filename), 'w') as f:
        json.dump({
            'experiment': 'EXP2: Cross-Domain (No Adaptation) - Multiple Runs',
            'source_power': source_power,
            'target_power': target_power,
            'config': config,
            'runs': runs,
            'source_test_acc': {
                'mean': float(np.mean(source_accs)),
                'std': float(np.std(source_accs)),
                'min': float(min(source_accs)),
                'max': float(max(source_accs)),
                'all': source_accs
            },
            'target_test_acc': {
                'mean': float(np.mean(target_accs)),
                'std': float(np.std(target_accs)),
                'min': float(min(target_accs)),
                'max': float(max(target_accs)),
                'all': target_accs
            },
            'performance_drop': {
                'mean': float(np.mean(drops)),
                'std': float(np.std(drops)),
                'min': float(min(drops)),
                'max': float(max(drops)),
                'all': drops
            },
            'all_results': results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f'\nResults saved to {results_dir}/{filename}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_power', type=str, default='1.0kW')
    parser.add_argument('--target_power', type=str, default='3.0kW')
    parser.add_argument('--base_path', type=str, default=str(DATASETS_DIR))
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs to average (default: 1)')
    parser.add_argument('--start_seed', type=int, default=42,
                       help='Starting random seed (default: 42)')
    args = parser.parse_args()

    if args.runs > 1:
        run_exp2_multiple(args.source_power, args.target_power, args.base_path,
                         runs=args.runs, start_seed=args.start_seed)
    else:
        run_exp2(args.source_power, args.target_power, args.base_path)
