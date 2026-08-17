#!/usr/bin/env python3
"""True single-model supervised transfer with source replay.

One shared MLP and one shared 15-class head are source-pretrained, then adapted
with labelled target training windows plus source replay.  Both domains are
always evaluated with the exact same state dict.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from _project_paths import DATASETS_DIR, RESULTS_DIR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.data_processor_v2 import CLASS_NAMES
from data.transfer_features import load_transfer_features


SHORT_NAMES = ["H"] + [f"IC-{x}" for x in (6, 5, 4, 3, 2, 1, .5)] + [
    f"IT-{x}" for x in (6, 5, 4, 3, 2, 1, .5)
]


class TransferMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 15),
        )

    def forward(self, x):
        return self.network(x)


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_cached(run_dir: Path, power: str, window: int, max_windows: int) -> dict:
    path = run_dir / "feature_cache" / f"{power}_w{window}_n{max_windows}.npz"
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        return {key: loaded[key] for key in loaded.files}
    print(f"extract {power}, window={window}", flush=True)
    data = load_transfer_features(DATASETS_DIR / f"dataset2_{power}.csv", window, max_windows)
    arrays = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def standardize(source: dict, target: dict) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    # Target training features may be used without labels for domain calibration.
    train = np.concatenate([source["X_train"], target["X_train"]])
    mean = train.mean(axis=0); scale = np.maximum(train.std(axis=0), 1e-5)
    converted = []
    for domain in (source, target):
        copy = dict(domain)
        for split in ("train", "val", "test"):
            copy[f"X_{split}"] = ((domain[f"X_{split}"] - mean) / scale).astype(np.float32)
        converted.append(copy)
    return converted[0], converted[1], mean, scale


def loader(X, y, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(model, X, y, device, criterion) -> dict:
    model.eval(); total_loss = correct = total = 0; predictions = []
    with torch.no_grad():
        for xb, yb in loader(X, y, 256, False):
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += float(criterion(logits, yb).item()) * len(yb)
            pred = logits.argmax(1); predictions.extend(pred.cpu().tolist())
            correct += int((pred == yb).sum()); total += len(yb)
    return {"loss": total_loss / total, "accuracy": 100.0 * correct / total,
            "labels": y.tolist(), "predictions": predictions}


def pretrain(model, source: dict, config: dict, device) -> tuple[dict, list[dict]]:
    criterion = nn.CrossEntropyLoss(label_smoothing=config["smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["pretrain_lr"], weight_decay=1e-4)
    train_loader = loader(source["X_train"], source["y_train"], config["batch_size"], True)
    best_score = -1.0; best_state = None; stale = 0; history = []
    for epoch in range(config["pretrain_epochs"]):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = criterion(model(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
        val = evaluate(model, source["X_val"], source["y_val"], device, criterion)
        history.append({"epoch": epoch, "val_accuracy": val["accuracy"], "val_loss": val["loss"]})
        if val["accuracy"] > best_score:
            best_score = val["accuracy"]; stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else: stale += 1
        if stale >= 15: break
    model.load_state_dict(best_state)
    return best_state, history


def adapt(model, source: dict, target: dict, config: dict, device):
    criterion_none = nn.CrossEntropyLoss(label_smoothing=config["smoothing"], reduction="none")
    criterion = nn.CrossEntropyLoss(label_smoothing=config["smoothing"])
    X = np.concatenate([source["X_train"], target["X_train"]])
    y = np.concatenate([source["y_train"], target["y_train"]])
    domain = np.concatenate([np.zeros(len(source["y_train"]), dtype=np.int64),
                             np.ones(len(target["y_train"]), dtype=np.int64)])
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(domain)),
                              batch_size=config["batch_size"], shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["adapt_lr"], weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["adapt_epochs"])
    history = {d: {f"{s}_{m}": [] for s in ("train", "val") for m in ("loss", "accuracy")}
               for d in ("source", "target")}
    best_key = (-1.0, -1.0); best_state = None; stale = 0
    for epoch in range(config["adapt_epochs"] + 1):
        if epoch:
            model.train()
            for xb, yb, db in train_loader:
                xb, yb, db = xb.to(device), yb.to(device), db.to(device); optimizer.zero_grad()
                losses = criterion_none(model(xb), yb)
                weights = torch.where(db == 0, config["source_weight"], 1.0)
                loss = (losses * weights).sum() / weights.sum()
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            scheduler.step()
        results = {}
        for name, data in (("source", source), ("target", target)):
            for split in ("train", "val"):
                result = evaluate(model, data[f"X_{split}"], data[f"y_{split}"], device, criterion)
                history[name][f"{split}_loss"].append(result["loss"])
                history[name][f"{split}_accuracy"].append(result["accuracy"])
                if split == "val": results[name] = result
        key = (min(results["source"]["accuracy"], results["target"]["accuracy"]),
               np.mean([results["source"]["accuracy"], results["target"]["accuracy"]]))
        if key > best_key:
            best_key = key; stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else: stale += 1
        if epoch and epoch % 10 == 0:
            print(f"    epoch={epoch} source={results['source']['accuracy']:.2f} target={results['target']['accuracy']:.2f}", flush=True)
        if stale >= 20: break
    model.load_state_dict(best_state)
    final = {name: evaluate(model, data["X_val"], data["y_val"], device, criterion)
             for name, data in (("source", source), ("target", target))}
    return history, final, best_key


def plot_curves(history: dict, path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    for column, domain in enumerate(("source", "target")):
        for row, metric in enumerate(("loss", "accuracy")):
            axis = axes[row, column]
            axis.plot(history[domain][f"train_{metric}"], label="Train")
            axis.plot(history[domain][f"val_{metric}"], label="Validation")
            axis.set(title=f"{domain.title()} {metric}", xlabel="Adaptation epoch", ylabel=metric.title())
            if metric == "accuracy": axis.set_ylim(0, 105)
            axis.grid(alpha=.25); axis.legend()
    fig.suptitle(title); fig.tight_layout(rect=(0, 0, 1, .97))
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_confusion(results: dict, path: Path, title: str, split: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(22, 9.5))
    for axis, domain in zip(axes, ("source", "target")):
        result = results[domain]
        matrix = confusion_matrix(result["labels"], result["predictions"], labels=np.arange(15)).astype(float)
        matrix = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) * 100
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=100)
        for row in range(15):
            for column in range(15):
                if matrix[row, column] >= .5:
                    axis.text(column, row, f"{matrix[row, column]:.0f}", ha="center", va="center", fontsize=6,
                              color="white" if matrix[row, column] >= 55 else "black")
        axis.set_xticks(range(15), SHORT_NAMES, rotation=60, ha="right", fontsize=8)
        axis.set_yticks(range(15), SHORT_NAMES, fontsize=8)
        axis.set(xlabel="Predicted", ylabel="True", title=f"{domain.title()} {split}: {result['accuracy']:.2f}%")
        fig.colorbar(image, ax=axis, fraction=.046, pad=.04)
    fig.suptitle(title); fig.tight_layout(rect=(0, 0, 1, .96))
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def plot_config_bars(aggregates: list[dict], path: Path) -> None:
    ordered = sorted(aggregates, key=lambda row: row["config_index"])
    positions = np.arange(len(ordered)); width = .36
    labels = [f"C{row['config_index']}\nw{row['config']['window']}" for row in ordered]
    fig, axis = plt.subplots(figsize=(14, 7))
    axis.bar(positions - width / 2, [row["source_val_mean"] for row in ordered], width,
             label="Source validation")
    axis.bar(positions + width / 2, [row["target_val_mean"] for row in ordered], width,
             label="Target validation")
    axis.axhline(70, color="tab:red", linestyle="--", label="70%")
    axis.axhline(80, color="tab:green", linestyle=":", label="80%")
    axis.set_xticks(positions, labels); axis.set_ylim(0, 105)
    axis.set(xlabel="Configuration", ylabel="Mean accuracy (%)",
             title="Single-model transfer validation comparison (two seeds)")
    axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def configs() -> list[dict]:
    variants = [
        (256, .15, 5e-4, .5), (256, .25, 1e-3, 1.0),
        (512, .20, 5e-4, 1.0), (512, .35, 3e-4, 2.0),
    ]
    return [{"window": window, "hidden": hidden, "dropout": dropout,
             "adapt_lr": lr, "source_weight": weight, "pretrain_lr": 8e-4,
             "batch_size": 128, "pretrain_epochs": 60, "adapt_epochs": 100,
             "smoothing": .05}
            for window in (4096, 8192) for hidden, dropout, lr, weight in variants]


def name(index: int, config: dict) -> str:
    return (f"{index:02d}_w{config['window']}_h{config['hidden']}_d{config['dropout']}_"
            f"lr{config['adapt_lr']}_replay{config['source_weight']}").replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--max-windows", type=int, default=256)
    args = parser.parse_args()
    run_dir = args.run_dir or RESULTS_DIR / f"exp3_v5_single_transfer_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for index, config in enumerate(configs(), 1):
        raw_source = load_cached(run_dir, "1.0kW", config["window"], args.max_windows)
        raw_target = load_cached(run_dir, "3.0kW", config["window"], args.max_windows)
        source, target, mean, scale = standardize(raw_source, raw_target)
        for seed in args.seeds:
            stem = f"{name(index, config)}_seed{seed}"
            result_path = run_dir / "validation" / f"{stem}.json"
            if result_path.exists(): rows.append(json.loads(result_path.read_text())); continue
            print(f"[{index}/8 seed={seed}] {stem}", flush=True); seed_all(seed)
            model = TransferMLP(source["X_train"].shape[1], config["hidden"], config["dropout"]).to(device)
            source_state, pretrain_history = pretrain(model, source, config, device)
            history, validation, key = adapt(model, source, target, config, device)
            checkpoint = run_dir / "models" / f"{stem}.pth"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "config": config, "mean": mean, "scale": scale, "seed": seed,
                        "class_names": CLASS_NAMES, "single_shared_model": True}, checkpoint)
            row = {"stem": stem, "config_index": index, "config": config, "seed": seed,
                   "source_val_accuracy": validation["source"]["accuracy"],
                   "target_val_accuracy": validation["target"]["accuracy"],
                   "minimum_val_accuracy": key[0], "mean_val_accuracy": key[1],
                   "validation": validation, "history": history,
                   "pretrain_history": pretrain_history, "checkpoint": str(checkpoint)}
            atomic_json(result_path, row); rows.append(row)
            plot_curves(history, run_dir / "visualizations/training_curves" / f"{stem}.png", stem)
            plot_confusion(validation, run_dir / "visualizations/validation_confusion" / f"{stem}.png", stem, "validation")
            print(f"  val source={validation['source']['accuracy']:.2f} target={validation['target']['accuracy']:.2f}", flush=True)
    aggregates = []
    for index, config in enumerate(configs(), 1):
        selected = [row for row in rows if row["config_index"] == index]
        aggregates.append({"config_index": index, "config": config,
                           "source_val_mean": float(np.mean([r["source_val_accuracy"] for r in selected])),
                           "target_val_mean": float(np.mean([r["target_val_accuracy"] for r in selected])),
                           "minimum_val_mean": float(np.mean([r["minimum_val_accuracy"] for r in selected]))})
    aggregates.sort(key=lambda r: (r["minimum_val_mean"], (r["source_val_mean"] + r["target_val_mean"]) / 2), reverse=True)
    atomic_json(run_dir / "validation_ranking.json", aggregates)
    plot_config_bars(aggregates, run_dir / "visualizations" / "validation_config_bars.png")
    locked = aggregates[0]; atomic_json(run_dir / "locked_config.json", locked)
    tests = []
    for seed in args.seeds:
        row = next(r for r in rows if r["config_index"] == locked["config_index"] and r["seed"] == seed)
        checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        source, target, _, _ = standardize(
            load_cached(run_dir, "1.0kW", config["window"], args.max_windows),
            load_cached(run_dir, "3.0kW", config["window"], args.max_windows))
        model = TransferMLP(source["X_train"].shape[1], config["hidden"], config["dropout"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        criterion = nn.CrossEntropyLoss(label_smoothing=config["smoothing"])
        result = {domain: evaluate(model, data["X_test"], data["y_test"], device, criterion)
                  for domain, data in (("source", source), ("target", target))}
        tests.append({"seed": seed, "source_accuracy": result["source"]["accuracy"],
                      "target_accuracy": result["target"]["accuracy"], "results": result,
                      "checkpoint": row["checkpoint"]})
        plot_confusion(result, run_dir / "visualizations/locked_test" / f"seed{seed}.png",
                       f"Locked single-model transfer seed {seed}", "test")
    summary = {"protocol": "one shared model and one shared 15-class head; source pretrain then labelled target adaptation with source replay",
               "locked": locked, "tests": tests,
               "source_test_mean": float(np.mean([r["source_accuracy"] for r in tests])),
               "target_test_mean": float(np.mean([r["target_accuracy"] for r in tests]))}
    atomic_json(run_dir / "summary.json", summary)
    print(f"LOCKED TEST source={summary['source_test_mean']:.2f} target={summary['target_test_mean']:.2f}", flush=True)


if __name__ == "__main__":
    main()
