#!/usr/bin/env python3
"""EXP3-V8: semi-supervised 1.0 kW -> 3.0 kW transfer with FixMatch.

All source labels are available.  Only a stratified fraction of target TRAIN
labels is exposed; the remaining target training loader contains X only.
Target validation labels are used for checkpoint/config selection and target
test labels are used once after locking.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from datetime import datetime
from itertools import cycle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT)]

from _project_paths import RESULTS_DIR
from exp3_v5_single_model_transfer import (
    TransferMLP,
    atomic_json,
    evaluate,
    load_cached,
    plot_confusion,
    plot_curves,
    standardize,
)
from exp3_v7_supervised_strategies import source_pretrain, state


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def xy_loader(X, y, batch=128, shuffle=True):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=batch, shuffle=shuffle, num_workers=0, drop_last=shuffle,
    )


def x_loader(X, batch=128, shuffle=True):
    """The unlabeled target loader deliberately has no label tensor."""
    return DataLoader(
        TensorDataset(torch.from_numpy(X)), batch_size=batch,
        shuffle=shuffle, num_workers=0, drop_last=shuffle,
    )


def stratified_target_partition(X, y, fraction: float, seed: int):
    """Expose an equal label count per class, then discard all other labels."""
    rng = np.random.default_rng(seed)
    labeled, unlabeled = [], []
    per_class = {}
    for class_id in range(15):
        indices = np.flatnonzero(y == class_id)
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * fraction)))
        labeled.extend(indices[:count]); unlabeled.extend(indices[count:])
        per_class[str(class_id)] = count
    labeled = np.asarray(sorted(labeled), dtype=np.int64)
    unlabeled = np.asarray(sorted(unlabeled), dtype=np.int64)
    return {
        "X_labeled": X[labeled], "y_labeled": y[labeled],
        "X_unlabeled": X[unlabeled],
        "labeled_indices": labeled, "unlabeled_indices": unlabeled,
        "labels_per_class": per_class,
    }


def weak_augment(x):
    return x + torch.randn_like(x) * 0.01


def strong_augment(x):
    value = x + torch.randn_like(x) * 0.05
    keep = torch.rand_like(value) >= 0.05
    return value * keep


@torch.no_grad()
def update_ema(teacher, student, decay=0.995):
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        teacher_buffer.copy_(student_buffer)


def configurations():
    rows = []
    for fraction in (.05, .10, .20, .30, .40):
        rows.append({"method": "labeled_only", "labeled_fraction": fraction, "threshold": None})
        rows.extend({"method": "fixmatch", "labeled_fraction": fraction, "threshold": threshold}
                    for threshold in (.80, .95))
    return rows


def adapt_semisupervised(model, source, target_labeled, target_unlabeled_X,
                         target_validation, config, device, epochs=70):
    """Adapt with labeled target subset and X-only target remainder."""
    source_batches = xy_loader(source["X_train"], source["y_train"])
    labeled_batches = xy_loader(target_labeled["X_train"], target_labeled["y_train"])
    unlabeled_batches = x_loader(target_unlabeled_X)
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema_teacher = copy.deepcopy(model).eval()
    for parameter in ema_teacher.parameters():
        parameter.requires_grad = False

    history = {domain: {f"{split}_{metric}": []
                        for split in ("train", "val") for metric in ("loss", "accuracy")}
               for domain in ("source", "target")}
    unsupervised_history = []
    best = ((-1.0, -1.0), None)
    stale = 0
    for epoch in range(epochs + 1):
        coverage_values, pseudo_losses = [], []
        if epoch:
            model.train()
            if config["method"] == "fixmatch":
                iterator = zip(cycle(source_batches), cycle(labeled_batches), unlabeled_batches)
            else:
                iterator = ((source_batch, labeled_batch, None)
                            for source_batch, labeled_batch in zip(source_batches, cycle(labeled_batches)))
            for (sx, sy), (lx, ly), unlabeled_batch in iterator:
                sx, sy = sx.to(device), sy.to(device)
                lx, ly = lx.to(device), ly.to(device)
                optimizer.zero_grad()
                loss = criterion(model(sx), sy) + criterion(model(lx), ly)
                coverage = 0.0
                pseudo_loss = torch.zeros((), device=device)
                if config["method"] == "fixmatch":
                    (ux,) = unlabeled_batch
                    ux = ux.to(device)
                    ema_teacher.eval()
                    with torch.no_grad():
                        probability = F.softmax(ema_teacher(weak_augment(ux)), dim=1)
                        confidence, pseudo = probability.max(dim=1)
                        mask = confidence >= config["threshold"]
                    coverage = float(mask.float().mean())
                    if mask.any():
                        pseudo_loss = F.cross_entropy(model(strong_augment(ux[mask])), pseudo[mask])
                        ramp = min(1.0, epoch / 10.0)
                        loss = loss + ramp * pseudo_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                update_ema(ema_teacher, model)
                coverage_values.append(coverage)
                pseudo_losses.append(float(pseudo_loss.detach()))
            scheduler.step()

        unsupervised_history.append({
            "epoch": epoch,
            "pseudo_coverage": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "pseudo_loss": float(np.mean(pseudo_losses)) if pseudo_losses else 0.0,
        })
        current = {}
        domain_data = {
            "source": source,
            "target": {
                "X_train": target_labeled["X_train"], "y_train": target_labeled["y_train"],
                "X_val": target_validation["X_val"], "y_val": target_validation["y_val"],
            },
        }
        for domain, data in domain_data.items():
            for split in ("train", "val"):
                result = evaluate(model, data[f"X_{split}"], data[f"y_{split}"], device, criterion)
                history[domain][f"{split}_loss"].append(result["loss"])
                history[domain][f"{split}_accuracy"].append(result["accuracy"])
                if split == "val":
                    current[domain] = result
        key = (min(current["source"]["accuracy"], current["target"]["accuracy"]),
               float(np.mean([current["source"]["accuracy"], current["target"]["accuracy"]])))
        if key > best[0]:
            best = (key, state(model)); stale = 0
        else:
            stale += 1
        if stale >= 20:
            break

    model.load_state_dict(best[1])
    final = {
        "source": evaluate(model, source["X_val"], source["y_val"], device, criterion),
        "target": evaluate(model, target_validation["X_val"], target_validation["y_val"], device, criterion),
    }
    return history, unsupervised_history, final, best[0]


def plot_bars(aggregates, path, title, test=False):
    labels = [row["label"] for row in aggregates]
    x = np.arange(len(labels)); width = .36
    fig, axis = plt.subplots(figsize=(max(11, len(labels) * 1.35), 7))
    axis.bar(x - width / 2, [row["source"] for row in aggregates], width,
             label=f"Source {'test' if test else 'validation'}")
    axis.bar(x + width / 2, [row["target"] for row in aggregates], width,
             label=f"Target {'test' if test else 'validation'}")
    axis.axhline(70, color="tab:red", linestyle="--", label="70%")
    axis.axhline(80, color="tab:green", linestyle=":", label="80%")
    axis.set_xticks(x, labels); axis.set_ylim(0, 105); axis.set_ylabel("Accuracy (%)")
    axis.set_title(title); axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = parser.parse_args()
    run = args.run_dir or RESULTS_DIR / f"exp3_v8_semisupervised_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_source = load_cached(run, "1.0kW", 8192, 256)
    raw_target = load_cached(run, "3.0kW", 8192, 256)
    source, target, mean, scale = standardize(raw_source, raw_target)

    bases = {}
    for seed in args.seeds:
        bases[seed] = source_pretrain(source, seed, device)

    rows = []
    for index, config in enumerate(configurations(), 1):
        for seed in args.seeds:
            seed_all(seed)
            partition = stratified_target_partition(
                target["X_train"], target["y_train"], config["labeled_fraction"], seed)
            target_labeled = {"X_train": partition["X_labeled"], "y_train": partition["y_labeled"]}
            model = copy.deepcopy(bases[seed])
            threshold = "none" if config["threshold"] is None else str(config["threshold"])
            stem = (f"{index:02d}_{config['method']}_f{config['labeled_fraction']}_t{threshold}_seed{seed}"
                    .replace(".", "p"))
            print(stem, flush=True)
            history, unsupervised, validation, key = adapt_semisupervised(
                model, source, target_labeled, partition["X_unlabeled"], target, config, device)
            checkpoint = run / "models" / f"{stem}.pth"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": state(model), "config": config, "seed": seed,
                "mean": mean, "scale": scale, "single_shared_model": True,
                "target_training_labels_total": len(target["y_train"]),
                "target_training_labels_exposed": len(partition["y_labeled"]),
                "labeled_indices": partition["labeled_indices"],
                "unlabeled_indices": partition["unlabeled_indices"],
            }, checkpoint)
            row = {
                "stem": stem, "config_index": index, "config": config, "seed": seed,
                "target_training_labels_total": len(target["y_train"]),
                "target_training_labels_exposed": len(partition["y_labeled"]),
                "labels_per_class": partition["labels_per_class"],
                "source_val_accuracy": validation["source"]["accuracy"],
                "target_val_accuracy": validation["target"]["accuracy"],
                "minimum_val_accuracy": key[0], "validation": validation,
                "unsupervised_history": unsupervised, "checkpoint": str(checkpoint),
            }
            atomic_json(run / "json" / f"{stem}.json", row); rows.append(row)
            plot_curves(history, run / "visualizations/training_curves" / f"{stem}.png", stem)
            plot_confusion(validation, run / "visualizations/validation_confusion" / f"{stem}.png",
                           stem, "validation")
            print(f"  labels={len(partition['y_labeled'])}/{len(target['y_train'])} "
                  f"source={row['source_val_accuracy']:.2f} target={row['target_val_accuracy']:.2f}", flush=True)

    aggregates = []
    for index, config in enumerate(configurations(), 1):
        selected = [row for row in rows if row["config_index"] == index]
        aggregates.append({
            "config_index": index, "config": config,
            "source_val_mean": float(np.mean([row["source_val_accuracy"] for row in selected])),
            "target_val_mean": float(np.mean([row["target_val_accuracy"] for row in selected])),
            "minimum_val_mean": float(np.mean([row["minimum_val_accuracy"] for row in selected])),
            "labels_exposed": selected[0]["target_training_labels_exposed"],
        })
    atomic_json(run / "validation_aggregates.json", aggregates)
    validation_bars = [{
        "label": (f"{row['config']['method']}\n{int(row['config']['labeled_fraction']*100)}%"
                  + ("" if row["config"]["threshold"] is None else f" / {row['config']['threshold']}")),
        "source": row["source_val_mean"], "target": row["target_val_mean"],
    } for row in aggregates]
    plot_bars(validation_bars, run / "visualizations/validation_config_bars.png",
              "V8 semi-supervised validation comparison")

    locked = {}
    for fraction in (.05, .10, .20, .30, .40):
        baseline = next(row for row in aggregates
                        if row["config"]["labeled_fraction"] == fraction
                        and row["config"]["method"] == "labeled_only")
        fixmatch = max((row for row in aggregates
                        if row["config"]["labeled_fraction"] == fraction
                        and row["config"]["method"] == "fixmatch"),
                       key=lambda row: (row["minimum_val_mean"], row["target_val_mean"]))
        locked[f"{fraction:.2f}"] = {"labeled_only": baseline, "fixmatch": fixmatch}
    atomic_json(run / "locked_configs.json", locked)

    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    tests = []
    for fraction_key, methods in locked.items():
        for method, chosen in methods.items():
            for seed in args.seeds:
                row = next(item for item in rows
                           if item["config_index"] == chosen["config_index"] and item["seed"] == seed)
                checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
                model = TransferMLP(376, 512, .2).to(device)
                model.load_state_dict(checkpoint["model_state"])
                result = {domain: evaluate(model, data["X_test"], data["y_test"], device, criterion)
                          for domain, data in (("source", source), ("target", target))}
                test = {
                    "labeled_fraction": float(fraction_key), "method": method, "seed": seed,
                    "source_accuracy": result["source"]["accuracy"],
                    "target_accuracy": result["target"]["accuracy"],
                    "results": result, "checkpoint": row["checkpoint"],
                }
                tests.append(test)
                plot_confusion(result, run / "visualizations/locked_test" /
                               f"f{fraction_key}_{method}_seed{seed}.png",
                               f"V8 {int(float(fraction_key)*100)}% {method} seed {seed}", "test")

    test_bars = []
    for fraction in (.05, .10, .20, .30, .40):
        for method in ("labeled_only", "fixmatch"):
            selected = [row for row in tests
                        if row["labeled_fraction"] == fraction and row["method"] == method]
            test_bars.append({
                "label": f"{method}\n{int(fraction*100)}%",
                "source": float(np.mean([row["source_accuracy"] for row in selected])),
                "target": float(np.mean([row["target_accuracy"] for row in selected])),
            })
    plot_bars(test_bars, run / "visualizations/locked_test_bars.png",
              "V8 locked semi-supervised test comparison", test=True)
    summary = {
        "protocol": "semi-supervised target adaptation; stratified target label subset plus X-only remainder",
        "target_validation_labels_used_for_selection": True,
        "target_test_labels_used_after_locking_only": True,
        "locked": locked, "tests": tests,
    }
    atomic_json(run / "summary.json", summary)
    for row in test_bars:
        print(row["label"].replace("\n", " "), row["source"], row["target"], flush=True)


if __name__ == "__main__":
    main()
