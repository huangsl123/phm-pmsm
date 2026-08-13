# -*- coding: utf-8 -*-
"""
分析源域和目标域故障代码的一一对应关系
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np

from _project_paths import DATASETS_DIR

print("=" * 80)
print("故障代码一一对应关系分析")
print("=" * 80)

# 读取数据
source = pd.read_csv(DATASETS_DIR / 'dataset2_1.0kW.csv')
target = pd.read_csv(DATASETS_DIR / 'dataset2_3.0kW.csv')

# 获取每个故障代码的详细信息
def get_fault_info(df):
    """获取故障代码的统计信息"""
    info = []
    for code in sorted(df['fault_code'].unique()):
        code_df = df[df['fault_code'] == code]
        info.append({
            'code': code,
            'category': code_df['fault_category'].iloc[0],
            'data_type': code_df['data_type'].iloc[0],
            'count': len(code_df)
        })
    return info

source_info = get_fault_info(source)
target_info = get_fault_info(target)

print("\n## 一、源域 (1.0kW) 故障代码统计\n")
print(f"{'代码':<15} {'类型':<10} {'数据类型':<10} {'样本数':>8}")
print("-" * 50)
for info in source_info:
    print(f"{info['code']:<15} {info['category']:<10} {info['data_type']:<10} {info['count']:>8}")

print(f"\n源域统计:")
print(f"  总故障代码数: {len(source_info)}")
print(f"  coil 类型: {sum(1 for i in source_info if i['category'] == 'coil')} 个")
print(f"  interturn 类型: {sum(1 for i in source_info if i['category'] == 'interturn')} 个")
print(f"  unknown 类型: {sum(1 for i in source_info if i['category'] == 'unknown')} 个")

print("\n## 二、目标域 (3.0kW) 故障代码统计\n")
print(f"{'代码':<15} {'类型':<10} {'数据类型':<10} {'样本数':>8}")
print("-" * 50)
for info in target_info:
    print(f"{info['code']:<15} {info['category']:<10} {info['data_type']:<10} {info['count']:>8}")

print(f"\n目标域统计:")
print(f"  总故障代码数: {len(target_info)}")
print(f"  coil 类型: {sum(1 for i in target_info if i['category'] == 'coil')} 个")
print(f"  interturn 类型: {sum(1 for i in target_info if i['category'] == 'interturn')} 个")
print(f"  unknown 类型: {sum(1 for i in target_info if i['category'] == 'unknown')} 个")

print("\n## 三、按故障类型分组对比\n")

# 按 category 和 data_type 分组
def group_by_type(info_list):
    """按故障类型和数据类型分组"""
    groups = {}
    for info in info_list:
        if info['category'] == 'unknown':
            continue
        key = (info['category'], info['data_type'])
        if key not in groups:
            groups[key] = []
        groups[key].append(info['code'])
    return groups

source_groups = group_by_type(source_info)
target_groups = group_by_type(target_info)

print("源域分组:")
for (category, data_type), codes in sorted(source_groups.items()):
    print(f"  ({category}, {data_type}): {len(codes)} 个代码 - {codes}")

print("\n目标域分组:")
for (category, data_type), codes in sorted(target_groups.items()):
    print(f"  ({category}, {data_type}): {len(codes)} 个代码 - {codes}")

print("\n## 四、可能的对应关系分析\n")

# 分析 coil + current 组
source_coil_current = [i['code'] for i in source_info if i['category'] == 'coil' and i['data_type'] == 'current']
target_coil_current = [i['code'] for i in target_info if i['category'] == 'coil' and i['data_type'] == 'current']

print(f"coil + current 组:")
print(f"  源域 ({len(source_coil_current)} 个): {source_coil_current}")
print(f"  目标域 ({len(target_coil_current)} 个): {target_coil_current}")

if len(source_coil_current) == len(target_coil_current):
    print(f"  ✓ 数量相同！可能存在一一对应关系")
    for i, (s_code, t_code) in enumerate(zip(source_coil_current, target_coil_current)):
        print(f"    映射 {i}: {s_code} -> {t_code}")
else:
    print(f"  ✗ 数量不同，无法直接一一对应")

# 分析 interturn + current 组
source_inter_current = [i['code'] for i in source_info if i['category'] == 'interturn' and i['data_type'] == 'current']
target_inter_current = [i['code'] for i in target_info if i['category'] == 'interturn' and i['data_type'] == 'current']

print(f"\ninterturn + current 组:")
print(f"  源域 ({len(source_inter_current)} 个): {source_inter_current}")
print(f"  目标域 ({len(target_inter_current)} 个): {target_inter_current}")

if len(source_inter_current) == len(target_inter_current):
    print(f"  ✓ 数量相同！可能存在一一对应关系")
    for i, (s_code, t_code) in enumerate(zip(source_inter_current, target_inter_current)):
        print(f"    映射 {i}: {s_code} -> {t_code}")
else:
    print(f"  ✗ 数量不同，无法直接一一对应")

# 包含 0_00 的分析
print("\n## 五、包含 0_00 的完整对应关系\n")

source_all = [i['code'] for i in source_info if i['code'] == '0_00' or (i['category'] != 'unknown' and i['data_type'] == 'current')]
target_all = [i['code'] for i in target_info if i['code'] == '0_00' or (i['category'] != 'unknown' and i['data_type'] == 'current')]

print(f"包含 0_00 的所有电流数据故障代码:")
print(f"  源域 ({len(source_all)} 个): {source_all}")
print(f"  目标域 ({len(target_all)} 个): {target_all}")

if len(source_all) == len(target_all):
    print(f"\n  ✓ 数量完全相同！({len(source_all)} 个)")
    print(f"  建议的一一对应映射:")
    for i, (s_code, t_code) in enumerate(zip(source_all, target_all)):
        s_cat = next(i['category'] for i in source_info if i['code'] == s_code)
        t_cat = next(i['category'] for i in target_info if i['code'] == t_code)
        match = "✓" if s_cat == t_cat else "✗"
        print(f"    {i:2d}: {s_code:<15} ({s_cat:<10}) -> {t_code:<15} ({t_cat:<10}) {match}")
else:
    print(f"\n  ✗ 数量不同: 源域 {len(source_all)} 个, 目标域 {len(target_all)} 个")

print("\n## 六、结论\n")

if len(source_all) == len(target_all):
    print("✓ 发现完整的故障代码一一对应关系!")
    print(f"  源域和目标域都有 {len(source_all)} 个故障代码")
    print("  可以创建一个故障代码映射表用于跨域迁移学习")
else:
    print("✗ 未发现完整的一一对应关系")
    print(f"  源域有 {len(source_all)} 个故障代码")
    print(f"  目标域有 {len(target_all)} 个故障代码")

print("\n" + "=" * 80)
