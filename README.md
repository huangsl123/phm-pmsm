# PHM-PMSM

Research project for cross-condition permanent-magnet synchronous motor (PMSM) stator-fault diagnosis using time-frequency CrossViT and anti-forgetting transfer learning.

> **Data validity notice:** `datasets/dataset2_*.csv` are format-v2 recording
> indexes, not point-by-point signal tables. Rebuild them and the referenced
> signal arrays from the original TDMS ZIP files before training.

## Corrected data pipeline

```bash
python -m pip install -r requirements.txt
python scripts/rebuild_dataset.py
python -m unittest discover -s tests -v
```

The rebuild reads all 48 three-phase current recordings, applies anti-aliased
resampling from 100 kHz to 10 kHz, and writes a canonical manifest under
`datasets/processed_v2/`. Training scripts now use
`data/data_processor_v2.py`, which applies one 15-class mapping across domains
and splits contiguous recording blocks before windowing.

The three `dataset2_*.csv` files are regenerated as 15-row recording indexes;
full signal samples are stored in the referenced NPY files.

Previous result JSON files and figures were generated with the invalid CSV
pipeline. They are retained for audit history only and must not be cited as
corrected results. The latest LaTeX manuscript withdraws those values pending a
fresh, multi-seed run under the corrected protocol.

## Repository structure

- `scripts/`: final experiment, analysis, and visualization scripts
- `datasets/`: 1.0 kW, 1.5 kW, and 3.0 kW PMSM datasets
- `models/`, `modules/`, `data/`: model architecture, transfer-learning modules, and preprocessing code
- `result_upgrade/`: final experiment JSON results and visualizations used by the paper
- `PHM_会议论文_中英文与图/latest/`: final v14 paper PDF and LaTeX package
- `_project_docs/`: experiment reports and paper-development notes

Large `.pth` checkpoints and local archive folders are intentionally excluded from Git history. The final paper version is v14.
