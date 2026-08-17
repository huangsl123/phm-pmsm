# EXP3 V7 四种有监督迁移策略归档

本目录完整保存标准LwF、源记忆软标签蒸馏、EWC和经验回放实验。四种策略都使用3.0 kW
目标训练标签，因此均属于有监督迁移学习；两域始终共用一个模型和一个15类分类头。

## 锁定测试结果

| 方法 | 源域测试 | 目标域测试 |
|---|---:|---:|
| LwF α=0.5 | 63.73% ± 3.42% | 69.35% ± 0.65% |
| Distill β=0.2 | 86.54% ± 1.48% | 84.18% ± 0.37% |
| EWC λ=10000 | 72.55% ± 1.66% | 83.53% ± 0.00% |
| Replay γ=1.0 | 84.84% ± 0.55% | 84.90% ± 0.28% |

“±”为seed 42和123之间的样本标准差。配置由验证集双域下界锁定，测试集未用于调参。

## 目录

- `code/`：V7训练脚本、共享模型/绘图函数和数据特征代码快照；
- `dataset_metadata/`：1.0/3.0 kW索引CSV与重新解析数据manifest；
- `report/`：完整V7实验报告；
- `results/models/`：24个模型checkpoint；
- `results/json/`：24个逐次结果；
- `results/visualizations/`：57张双域训练曲线、混淆矩阵和参数柱状图；
- `results/summary.json`、`locked_configs.json`、`validation_aggregates.json`：汇总与锁定证据；
- `tests/`：V7方法性质、Fisher数据边界和策略分离测试快照。

## 复现

从项目根目录执行：

```bash
/home/eai/Tools/miniforge3/envs/bym50/bin/python \
  scripts/exp3_v7_supervised_strategies.py --seeds 42 123
```

重新运行会创建新的时间戳结果目录，不覆盖本归档。源数据每个功率、每类只有一条长记录，报告
中的时间块测试不能替代跨独立采集批次验证。
