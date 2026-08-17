#!/usr/bin/env python3
"""Regenerate complete dual-domain figures for a corrected retraining run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _project_paths import DATASETS_DIR
from data.data_processor_v2 import load_csv_data
from retrain_corrected_experiments import (
    evaluate, load_checkpoint_model, plot_dual_confusion, plot_dual_history,
)


def plot_bars(rows: list[dict], path: Path, title: str) -> None:
    names = [row["config_name"] for row in rows]
    positions = np.arange(len(rows))
    width = 0.25
    fig, axis = plt.subplots(figsize=(max(13, len(rows) * 0.9), 7))
    axis.bar(positions - width, [row["source_val_mean"] for row in rows], width,
             yerr=[row["source_val_std"] for row in rows], capsize=3,
             label="Source validation")
    axis.bar(positions, [row["target_val_mean"] for row in rows], width,
             yerr=[row["target_val_std"] for row in rows], capsize=3,
             label="Target validation")
    axis.bar(positions + width, [row["balance_mean"] for row in rows], width,
             yerr=[row["balance_std"] for row in rows], capsize=3,
             label="Validation balance")
    axis.axhline(70, color="#c0392b", linestyle="--", linewidth=1.2,
                 label="70% target")
    axis.set_xticks(positions, names, rotation=62, ha="right", fontsize=8)
    axis.set_ylabel("Score / Accuracy (%)")
    axis.set_title(title, fontweight="bold")
    axis.grid(axis="y", alpha=0.28)
    axis.legend(ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = load_csv_data(
        str(DATASETS_DIR / f"dataset2_{config['source_power']}.csv"),
        max_windows_per_class=config["max_windows_per_class"],
    )
    target = load_csv_data(
        str(DATASETS_DIR / f"dataset2_{config['target_power']}.csv"),
        max_windows_per_class=config["max_windows_per_class"],
    )

    for version in ("exp3_v1", "exp3_v2", "exp3_v3", "exp3_v4"):
        result_paths = sorted((run_dir / "json" / version / "configs").glob("*.json"))
        total = len(result_paths)
        for index, result_path in enumerate(result_paths, 1):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            stem = result_path.stem
            visual_dir = run_dir / "visualizations" / version
            plot_dual_history(
                result["source_history"], result["target_history"],
                visual_dir / "training_curves_dual" / f"{stem}.png",
                f"{version.upper()} - {result['config_name']} - seed {result['seed']}",
            )
            model, _ = load_checkpoint_model(Path(result["checkpoint"]), source, device)
            source_val = evaluate(model, source, device, "val")
            target_val = evaluate(model, target, device, "val")
            plot_dual_confusion(
                source_val, target_val,
                visual_dir / "validation_confusion_dual" / f"{stem}.png",
                f"{version.upper()} - {result['config_name']} - seed {result['seed']}",
            )
            if index % 5 == 0 or index == total:
                print(f"{version}: {index}/{total}", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        plot_bars(
            summary[version]["validation_aggregates"],
            run_dir / "visualizations" / version / "validation_config_bars.png",
            f"{version.upper()} - All Configuration Validation Results",
        )
        for row in summary[version]["final_test_runs"]:
            plot_dual_confusion(
                {
                    "accuracy": row["source_test_accuracy"],
                    "labels": row["source_labels"],
                    "predictions": row["source_predictions"],
                },
                {
                    "accuracy": row["target_test_accuracy"],
                    "labels": row["target_labels"],
                    "predictions": row["target_predictions"],
                },
                run_dir / "visualizations" / version / "locked_test_dual" /
                f"confusion_seed{row['seed']}.png",
                f"{version.upper()} - Locked Configuration - seed {row['seed']}",
                split_name="Test",
            )

    print("Visualization regeneration complete.", flush=True)


if __name__ == "__main__":
    main()
