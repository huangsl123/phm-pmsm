# EXP3 V6 标准无监督迁移归档

本目录完整保存标准无监督DANN、MMD和伪标签实验。训练阶段目标域标签不传入适应函数；
目标标签只用于训练完成后的离线诊断和预锁定测试。

## 结果摘要

| 预锁定方法 | 源域测试 | 目标域测试 |
|---|---:|---:|
| DANN λ=0.1 | 84.05% | 1.63% |
| MMD λ=0.5 | 84.51% | 3.73% |
| 伪标签 threshold=0.8 | 84.44% | 6.67% |

这是应当保留的负结果：标准边缘域对齐和朴素伪标签没有解决当前15类跨功率对应问题。

## 目录

- `code/`：V6训练、V5公共绘图/数据函数及无量纲特征代码快照；
- `dataset_metadata/`：数据索引CSV与manifest；
- `report/`：完整V6实验报告；
- `results/models/`：9配置×2种子，共18个单模型；
- `results/json/`：各配置验证诊断与无监督损失历史；
- `results/visualizations/`：18张训练曲线、18张验证混淆矩阵、6张测试混淆矩阵和2张柱状图；
- `results/prelocked_protocol.json`：训练前锁定的测试配置；
- `results/summary.json`：完整汇总；
- `tests/`：证明适应接口无法接收目标标签的测试快照。

## 复现

建议从项目根目录执行：

```bash
/home/eai/Tools/miniforge3/envs/bym50/bin/python \
  scripts/exp3_v6_unsupervised_da.py --seeds 42 123
```

注意：目标域诊断曲线是在每个配置训练全部完成后，使用内存中的固定epoch快照离线计算；这些
准确率不参与梯度、早停或模型选择。
