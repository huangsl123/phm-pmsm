# -*- coding: utf-8 -*-
"""
精简版训练脚本 - 只训练V3/V4最佳配置并保存模型权重
基于 exp3_improved_v4.py 的设置
数据分割: test_size=0.10, val_size=0.05
评估: 在测试集上评估
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

from _project_paths import PROJECT_ROOT

from models.crossvit import CrossViTFaultDiagnosis
from data.data_processor_v2 import load_csv_data, MultiModalFaultDataset


# ============================================
# 最佳配置定义 (基于 exp3_improved_v4.py 结果)
# ============================================

BEST_CONFIGS = [
    {
        "name": "V4_Best_Distill_T1.8",
        "description": "V4最优蒸馏 - 平衡分80.0",
        "method": "distill",
        "config": {
            "embed_dim": 384,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "lr": 0.0001,
            "weight_decay": 0.0005,
            "batch_size": 32
        },
        "method_params": {
            "temperature": 1.8,
            "distill_alpha": 0.3,
            "mix_source_ratio": 0.3
        }
    },
    {
        "name": "V3_Best_Distill_T2",
        "description": "V3最优蒸馏 - 源域保留77.5%",
        "method": "distill",
        "config": {
            "embed_dim": 384,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "lr": 0.0001,
            "weight_decay": 0.0005,
            "batch_size": 32
        },
        "method_params": {
            "temperature": 2.0,
            "distill_alpha": 0.3,
            "mix_source_ratio": 0.3
        }
    },
    {
        "name": "V4_Medium_DANN",
        "description": "V4最优DANN - 目标域75%",
        "method": "dann",
        "config": {
            "embed_dim": 512,
            "num_heads": 8,
            "num_layers": 3,
            "dropout": 0.05,
            "lr": 0.0001,
            "weight_decay": 0.0001,
            "batch_size": 32
        },
        "method_params": {
            "dann_lambda": 0.1
        }
    },
    {
        "name": "V3_Baseline",
        "description": "V3基线 - 目标域77.5%",
        "method": "baseline",
        "config": {
            "embed_dim": 384,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.1,
            "lr": 0.0001,
            "weight_decay": 0.0005,
            "batch_size": 32
        },
        "method_params": {}
    }
]


# ============================================
# 评估函数
# ============================================

def evaluate_on_split(model, data, device, split='test'):
    """评估模型在指定数据分割上的准确率"""
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

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for time_x, spec_x, labels in loader:
            time_x = time_x.to(device)
            spec_x = spec_x.to(device)
            labels = labels.to(device)

            outputs = model(time_x, spec_x)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return 100. * correct / total


def evaluate_all_splits(model, source_data, target_data, device):
    """评估模型在所有数据分割上的准确率"""
    # 源域评估
    source_train = evaluate_on_split(model, source_data, device, 'train')
    source_val = evaluate_on_split(model, source_data, device, 'val')
    source_test = evaluate_on_split(model, source_data, device, 'test')

    # 目标域评估
    target_train = evaluate_on_split(model, target_data, device, 'train')
    target_val = evaluate_on_split(model, target_data, device, 'val')
    target_test = evaluate_on_split(model, target_data, device, 'test')

    return {
        'source': {'train': source_train, 'val': source_val, 'test': source_test},
        'target': {'train': target_train, 'val': target_val, 'test': target_test}
    }


# ============================================
# 训练函数
# ============================================

def train_source_domain(model, source_data, device, config, epochs=100):
    """在源域训练模型"""

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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            time_x, spec_x, labels = batch
            time_x = time_x.to(device)
            spec_x = spec_x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(time_x, spec_x)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100. * train_correct / train_total
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                time_x, spec_x, labels = batch
                time_x = time_x.to(device)
                spec_x = spec_x.to(device)
                labels = labels.to(device)

                outputs = model(time_x, spec_x)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 20 == 0:
            print(f'    Epoch {epoch+1}/{epochs}: Train={train_acc:.1f}%, Val={val_acc:.1f}%, Best={best_val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history, model


def fine_tune_with_distillation(model, source_data, target_data, teacher_model,
                                device, params, epochs=100):
    """使用知识蒸馏进行微调"""

    train_dataset = MultiModalFaultDataset(
        target_data['X_train_time'], target_data['X_train_spec'],
        target_data['y_train'], augment=False
    )
    val_dataset = MultiModalFaultDataset(
        target_data['X_val_time'], target_data['X_val_spec'],
        target_data['y_val'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    # 源域数据用于混合
    source_train_dataset = MultiModalFaultDataset(
        source_data['X_train_time'], source_data['X_train_spec'],
        source_data['y_train'], augment=False
    )
    source_loader = DataLoader(source_train_dataset, batch_size=32, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    T = params.get('temperature', 2.0)
    alpha = params.get('distill_alpha', 0.3)
    mix_ratio = params.get('mix_source_ratio', 0.3)

    teacher_model.eval()

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        # 混合源域和目标域数据
        mixed_data = []
        source_iter = iter(source_loader)

        for target_batch in train_loader:
            mixed_data.append(('target', target_batch))
            if np.random.random() < mix_ratio:
                try:
                    source_batch = next(source_iter)
                    mixed_data.append(('source', source_batch))
                except StopIteration:
                    source_iter = iter(source_loader)
                    source_batch = next(source_iter)
                    mixed_data.append(('source', source_batch))

        for data_type, batch in mixed_data:
            time_x, spec_x, labels = batch
            time_x = time_x.to(device)
            spec_x = spec_x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if data_type == 'target':
                # 软标签（来自教师模型）
                with torch.no_grad():
                    teacher_outputs = teacher_model(time_x, spec_x)
                    soft_targets = torch.softmax(teacher_outputs / T, dim=1)

                # 学生模型输出
                student_outputs = model(time_x, spec_x)
                soft_student = torch.log_softmax(student_outputs / T, dim=1)

                # 蒸馏损失 + 硬标签损失
                loss_distill = kl_loss(soft_student, soft_targets) * (T * T)
                loss_hard = criterion(student_outputs, labels)
                loss = alpha * loss_distill + (1 - alpha) * loss_hard
            else:  # 源域数据只用硬标签
                outputs = model(time_x, spec_x)
                loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                time_x, spec_x, labels = batch
                time_x = time_x.to(device)
                spec_x = spec_x.to(device)
                labels = labels.to(device)

                outputs = model(time_x, spec_x)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        if (epoch + 1) % 20 == 0:
            print(f'    Epoch {epoch+1}/{epochs}: Val={val_acc:.1f}%, Best={best_val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


def fine_tune_baseline(model, target_data, device, config, epochs=100):
    """基线微调（无防遗忘策略）"""

    train_dataset = MultiModalFaultDataset(
        target_data['X_train_time'], target_data['X_train_spec'],
        target_data['y_train'], augment=False
    )
    val_dataset = MultiModalFaultDataset(
        target_data['X_val_time'], target_data['X_val_spec'],
        target_data['y_val'], augment=False
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            time_x, spec_x, labels = batch
            time_x = time_x.to(device)
            spec_x = spec_x.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(time_x, spec_x)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                time_x, spec_x, labels = batch
                time_x = time_x.to(device)
                spec_x = spec_x.to(device)
                labels = labels.to(device)

                outputs = model(time_x, spec_x)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total

        if (epoch + 1) % 20 == 0:
            print(f'    Epoch {epoch+1}/{epochs}: Val={val_acc:.1f}%, Best={best_val_acc:.1f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'    Early stopping at epoch {epoch + 1}')
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model


# ============================================
# 主训练流程
# ============================================

def train_best_config(config_info, source_data_dir, target_data_dir,
                      output_dir, device, epochs=100):
    """训练单个最佳配置"""

    print(f'\n{"="*60}')
    print(f'训练配置: {config_info["name"]}')
    print(f'描述: {config_info["description"]}')
    print(f'{"="*60}\n')

    # 加载数据 - 使用与 exp3_improved_v4.py 相同的参数
    print('加载数据...')
    source_data = load_csv_data(
        source_data_dir,
        window_size=1024,
        stride=256,      # 重要：使用256而不是512，产生394个样本
        spec_size=None,
        test_size=0.10,
        val_size=0.05,
        random_state=42
    )
    target_data = load_csv_data(
        target_data_dir,
        window_size=1024,
        stride=256,      # 重要：使用256而不是512，产生394个样本
        spec_size=None,
        test_size=0.10,
        val_size=0.05,
        random_state=42
    )

    # 打印数据分割信息
    print(f'源域数据: 训练{len(source_data["y_train"])}样本, 验证{len(source_data["y_val"])}样本, 测试{len(source_data["y_test"])}样本')
    print(f'目标域数据: 训练{len(target_data["y_train"])}样本, 验证{len(target_data["y_val"])}样本, 测试{len(target_data["y_test"])}样本')

    # 创建模型
    model = CrossViTFaultDiagnosis(
        in_channels=3,
        num_classes=source_data['n_classes'],
        time_seq_len=source_data['window_size'],
        spec_height=source_data['spec_size'][0],
        spec_width=source_data['spec_size'][1],
        embed_dim=config_info['config']['embed_dim'],
        num_heads=config_info['config']['num_heads'],
        num_layers=config_info['config']['num_layers'],
        dropout=config_info['config']['dropout']
    ).to(device)

    # 1. 源域训练
    print('阶段1: 源域训练')
    _, model = train_source_domain(model, source_data, device,
                                   config_info['config'], epochs=epochs)

    # 保存源域模型（作为教师模型）
    source_model_path = os.path.join(output_dir, f'{config_info["name"]}_source.pth')
    torch.save(model.state_dict(), source_model_path)
    print(f'✓ 源域模型已保存: {source_model_path}')

    # 评估源域性能（在测试集上）
    results_before = evaluate_all_splits(model, source_data, target_data, device)
    print(f'  源域测试准确率: {results_before["source"]["test"]:.2f}%')
    print(f'  目标域测试准确率（微调前）: {results_before["target"]["test"]:.2f}%')

    # 2. 目标域微调
    print('\n阶段2: 目标域微调')

    # 创建教师模型副本
    teacher_model = CrossViTFaultDiagnosis(
        in_channels=3,
        num_classes=source_data['n_classes'],
        time_seq_len=source_data['window_size'],
        spec_height=source_data['spec_size'][0],
        spec_width=source_data['spec_size'][1],
        embed_dim=config_info['config']['embed_dim'],
        num_heads=config_info['config']['num_heads'],
        num_layers=config_info['config']['num_layers'],
        dropout=config_info['config']['dropout']
    ).to(device)
    teacher_model.load_state_dict(model.state_dict())
    teacher_model.eval()

    if config_info['method'] == 'distill':
        model = fine_tune_with_distillation(
            model, source_data, target_data, teacher_model,
            device, config_info['method_params'], epochs=epochs
        )
    else:
        model = fine_tune_baseline(
            model, target_data, device,
            config_info['config'], epochs=epochs
        )

    # 保存微调后的模型
    final_model_path = os.path.join(output_dir, f'{config_info["name"]}_final.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f'✓ 微调模型已保存: {final_model_path}')

    # 3. 最终评估（在测试集上）
    results_after = evaluate_all_splits(model, source_data, target_data, device)
    print(f'\n最终结果:')
    print(f'  源域测试准确率: {results_after["source"]["test"]:.2f}%')
    print(f'  目标域测试准确率: {results_after["target"]["test"]:.2f}%')

    # 计算平衡分
    source_test = results_after["source"]["test"]
    target_test = results_after["target"]["test"]
    balance_score = target_test + max(0, min(source_test - 40, 20)) / 2
    print(f'  平衡分: {balance_score:.2f}')

    # 保存结果摘要
    result = {
        "name": config_info["name"],
        "description": config_info["description"],
        "method": config_info["method"],
        "source_train_acc": results_after["source"]["train"],
        "source_val_acc": results_after["source"]["val"],
        "source_test_acc": results_after["source"]["test"],
        "target_train_acc": results_after["target"]["train"],
        "target_val_acc": results_after["target"]["val"],
        "target_test_acc": results_after["target"]["test"],
        "balance_score": balance_score,
        "model_path": final_model_path
    }

    return result


def main():
    """主函数"""

    import argparse

    parser = argparse.ArgumentParser(description='训练最佳配置并保存模型权重')
    parser.add_argument('--source_data', type=str, required=True,
                        help='源域数据目录 (1.0kW)')
    parser.add_argument('--target_data', type=str, required=True,
                        help='目标域数据目录 (3.0kW)')
    parser.add_argument('--output', type=str, default=None,
                        help='模型输出目录（默认: result_upgrade/models）')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--config', type=str, nargs='+',
                        choices=['V4_T1.8', 'V3_T2', 'V4_DANN', 'V3_Baseline', 'ALL'],
                        default=['ALL'],
                        help='要训练的配置')
    parser.add_argument('--device', type=str, default='cpu',
                        help='训练设备')

    args = parser.parse_args()

    # 默认输出到 result_upgrade/models/
    if args.output is None:
        output_dir = os.path.join(PROJECT_ROOT, 'result_upgrade', 'models')
    else:
        output_dir = args.output

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 选择要训练的配置
    if 'ALL' in args.config:
        configs_to_train = BEST_CONFIGS
    else:
        config_map = {
            'V4_T1.8': 0,
            'V3_T2': 1,
            'V4_DANN': 2,
            'V3_Baseline': 3
        }
        configs_to_train = [BEST_CONFIGS[config_map[c]] for c in args.config if c in config_map]

    device = torch.device(args.device)

    # 训练每个配置
    all_results = []

    for config_info in configs_to_train:
        try:
            result = train_best_config(
                config_info,
                args.source_data,
                args.target_data,
                output_dir,
                device,
                epochs=args.epochs
            )
            all_results.append(result)
        except Exception as e:
            print(f'✗ 训练失败: {config_info["name"]}, 错误: {e}')
            import traceback
            traceback.print_exc()

    # 保存训练摘要
    summary_path = os.path.join(output_dir, 'training_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f'\n{"="*60}')
    print(f'训练完成! 共训练 {len(all_results)} 个配置')
    print(f'摘要已保存: {summary_path}')
    print(f'模型保存在: {output_dir}/')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()
