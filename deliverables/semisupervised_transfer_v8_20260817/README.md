# EXP3 V8 半监督迁移归档

V8仅开放5%、10%、20%、30%、40%的3.0 kW目标训练标签，其余目标训练样本通过X-only加载器
作为未标注数据。每档比较少量标签基线与FixMatch半监督迁移，源/目标共用一个模型和一个15类头。

## 核心结果

| 标签比例 | 基线目标测试 | FixMatch目标测试 | FixMatch源域测试 |
|---:|---:|---:|---:|
| 5% | 55.56% | 52.03% | 81.76% |
| 10% | 62.75% | 62.22% | 84.64% |
| 20% | 69.08% | 66.01% | 85.75% |
| 30% | 77.12% | 74.64% | 84.38% |
| 40% | 78.10% | 77.12% | 84.90% |

FixMatch在30%标签时双域超过70%，但没有超过相同标签预算的基线，因此不能声称未标注数据带来
额外性能提升。五档预算均通过验证集锁定置信阈值0.95。

## 目录

- `code/`：V8训练脚本、公共模型/绘图及特征代码快照；
- `dataset_metadata/`：源/目标索引CSV与重新解析数据manifest；
- `report/`：V8完整实验报告；
- `results/models/`：30个checkpoint，包含开放/隐藏目标训练索引；
- `results/json/`：30次训练的验证结果和伪标签覆盖率；
- `results/visualizations/`：82张训练曲线、混淆矩阵和柱状图；
- `results/locked_configs.json`：每档预算锁定配置；
- `results/summary.json`：最终测试汇总；
- `tests/`：标签隔离、分层划分和X-only加载器协议测试快照。

## 复现

```bash
/home/eai/Tools/miniforge3/envs/bym50/bin/python \
  scripts/exp3_v8_semisupervised.py --seeds 42 123
```

训练曲线中的Target train只对应开放标签子集。隐藏目标训练标签不进入适应函数。
