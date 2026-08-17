# 有监督迁移学习归档包

本目录集中保存本项目中使用目标域训练标签的迁移学习实验，便于后续查找。当前推荐结果为
`current_v5/`；`legacy_v1_v4/` 仅用于追溯历史实验。

## 方法性质说明

| 版本/方法 | 是否使用目标域训练标签 | 性质 |
|---|---|---|
| V1/V2 基础微调与冻结 | 是 | 有监督迁移 |
| V3 EWC | 是，目标域交叉熵 | 有监督迁移＋抗遗忘约束 |
| V3 蒸馏/LwF | 是，目标硬标签参与蒸馏损失 | 有监督迁移＋知识蒸馏/回放 |
| V4 DANN | 是，当前实现对目标标签计算分类损失 | 有监督域对抗，不是纯无监督DANN |
| V4 MMD | 是，当前实现对目标标签计算分类损失 | 有监督分布对齐，不是纯无监督MMD |
| V4 伪标签 | 当前实现仍从带真实标签的数据集构造子集 | 不能作为严格无监督结果 |
| V5 单模型回放 | 是 | 有监督目标域适应＋源域经验回放 |

因此，不能把此前V1–V4整体描述成“目标域标签不参与的无监督迁移”。如果后续需要严格无监督
实验，应另建协议：训练阶段禁止读取 `target y_train`，只允许目标域无标签输入参与对齐或伪标签。

## 目录结构

```text
supervised_transfer_learning_20260817/
├── README.md
├── current_v5/
│   ├── code/                 # V5特征处理、数据加载、训练和数据重建脚本快照
│   ├── dataset_metadata/     # 1.0/3.0 kW索引CSV与格式v2 manifest
│   ├── report/               # V5单模型迁移实验报告
│   ├── results/              # 41 MB完整结果
│   │   ├── feature_cache/    # 无量纲特征缓存
│   │   ├── models/           # 8配置×2种子，共16个单模型权重
│   │   ├── validation/       # 每配置、每种子的验证JSON与训练历史
│   │   ├── visualizations/   # 35张曲线、混淆矩阵和柱状图
│   │   ├── locked_config.json
│   │   ├── validation_ranking.json
│   │   └── summary.json
│   └── tests/                # 幅值缩放不变性测试
└── legacy_v1_v4/
    ├── code/                 # 历史V1–V4及统一重训脚本快照
    ├── report/               # 修正后的综合实验报告
    └── results_11GB_link     # 指向原11 GB结果目录的相对链接，不重复占空间
```

## 当前推荐结果

- 协议：一个共享模型、一个共享15类头；源域预训练后进行目标域监督适应和源域1:1回放；
- 锁定配置：8192点窗口，MLP 376→512→256→15，dropout=0.2，适应学习率5e-4；
- 源域测试：85.29% ± 0.28%；
- 目标域测试：86.14% ± 0.55%；
- 随机种子：42、123。

首先查看：

1. `current_v5/report/EXP3_V5_单模型迁移实验报告.md`
2. `current_v5/results/summary.json`
3. `current_v5/results/visualizations/validation_config_bars.png`
4. `current_v5/results/visualizations/locked_test/`

## 复现位置

归档中的代码是便于查阅的快照。直接复现建议仍在项目根目录执行：

```bash
/home/eai/Tools/miniforge3/envs/bym50/bin/python \
  scripts/exp3_v5_single_model_transfer.py --seeds 42 123
```

原始TDMS ZIP和重建后的全部NPY未重复复制进本归档，以免额外占用大量空间；CSV、manifest、
特征缓存、全部模型和全部实验输出已经保留。
