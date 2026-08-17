#!/usr/bin/env python3
"""Run the corrected EXP1/EXP2/EXP3 study without test-set model selection.

The runner is deliberately resumable.  Every (version, configuration, seed)
checkpoint and validation result is written immediately.  EXP3 configurations
are ranked only by validation metrics; test blocks are opened only for the
locked configuration of each version.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from _project_paths import DATASETS_DIR, PROJECT_ROOT, RESULTS_DIR
from data.data_processor_v2 import CLASS_NAMES, MultiModalFaultDataset, load_csv_data
from models.crossvit import CrossViTFaultDiagnosis
import exp3_v3_721 as v3
import exp3_v4_721 as v4


MODEL_KEYS = (
    "embed_dim", "num_heads", "num_layers", "dropout", "lr",
    "weight_decay", "batch_size",
)


class Tee:
    """Line-buffered stdout/stderr mirror used for durable long-run logs."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def base_config(**updates):
    value = {
        "embed_dim": 384,
        "num_heads": 4,
        "num_layers": 2,
        "dropout": 0.1,
        "lr": 1e-4,
        "weight_decay": 5e-4,
        "batch_size": 32,
        "method": "baseline",
        "ft_lr": 1e-4,
        "freeze_backbone": False,
        "freeze_layers": None,
        "freeze_embedding": False,
        "label_smoothing": 0.05,
    }
    value.update(updates)
    return value


def experiment_configs() -> dict[str, list[dict]]:
    v1_configs = [
        base_config(name="Baseline (embed_dim=384)"),
        base_config(name="Larger Model (embed_dim=512)", embed_dim=512,
                    num_heads=8, num_layers=3),
        base_config(name="Larger Model + Lower Dropout", embed_dim=512,
                    num_heads=8, num_layers=3, dropout=0.05,
                    weight_decay=1e-4),
        base_config(name="Larger Model + Higher FT LR", embed_dim=512,
                    num_heads=8, num_layers=3, ft_lr=5e-4),
        base_config(name="Largest Model (embed_dim=768)", embed_dim=768,
                    num_heads=12, num_layers=4, batch_size=16),
    ]

    v2_configs = [
        base_config(name="Baseline (384, no freeze)"),
        base_config(name="Freeze Backbone (384)", freeze_backbone=True),
        base_config(name="Low FT LR (384, lr=1e-5)", ft_lr=1e-5),
        base_config(name="Partial Freeze + Low LR (384)", ft_lr=5e-5,
                    freeze_layers=1, label_smoothing=0.1),
        base_config(name="Medium Model (512, no freeze)", embed_dim=512,
                    num_heads=8, num_layers=3),
        base_config(name="Medium Model + Freeze Backbone (512)", embed_dim=512,
                    num_heads=8, num_layers=3, freeze_backbone=True),
        base_config(name="Medium Model + Low Dropout (512)", embed_dim=512,
                    num_heads=8, num_layers=3, dropout=0.05,
                    weight_decay=1e-4, ft_lr=5e-5, freeze_layers=1,
                    label_smoothing=0.1),
        base_config(name="Large Model (768, partial freeze)", embed_dim=768,
                    num_heads=12, num_layers=4, batch_size=16,
                    ft_lr=5e-5, freeze_layers=2, label_smoothing=0.1),
    ]

    v3_configs = [
        base_config(name="V3-Baseline (384, no freeze)"),
        base_config(name="Fixed Freeze (384, first layer)", freeze_layers=1),
        base_config(name="Fixed Freeze + Embed (384)", freeze_layers=1,
                    freeze_embedding=True),
        base_config(name="EWC (384, lambda=100)", method="ewc", ewc_lambda=100),
        base_config(name="EWC (384, lambda=500)", method="ewc", ewc_lambda=500),
        base_config(name="EWC (384, lambda=1000)", method="ewc", ewc_lambda=1000),
        base_config(name="EWC + Freeze (384, lambda=500)", method="ewc",
                    ewc_lambda=500, freeze_layers=1),
        base_config(name="Distill+LWF (384, T=2, alpha=0.3)", method="distill",
                    temperature=2.0, distill_alpha=0.3, mix_source_ratio=0.3),
        base_config(name="Distill+LWF (384, T=3, alpha=0.5)", method="distill",
                    temperature=3.0, distill_alpha=0.5, mix_source_ratio=0.3),
        base_config(name="Distill+LWF (384, T=4, alpha=0.7)", method="distill",
                    temperature=4.0, distill_alpha=0.7, mix_source_ratio=0.5),
        base_config(name="Medium + EWC (512, lambda=500)", method="ewc",
                    embed_dim=512, num_heads=8, num_layers=3, ewc_lambda=500),
        base_config(name="Medium + Distill (512, T=3)", method="distill",
                    embed_dim=512, num_heads=8, num_layers=3,
                    temperature=3.0, distill_alpha=0.5, mix_source_ratio=0.4),
    ]

    v4_configs = [
        base_config(name="V3-Best Distill (T=2, alpha=0.3)", method="distill",
                    temperature=2.0, distill_alpha=0.3, mix_source_ratio=0.3),
        base_config(name="Distill (T=1.5, alpha=0.25)", method="distill",
                    temperature=1.5, distill_alpha=0.25, mix_source_ratio=0.3),
        base_config(name="Distill (T=1.8, alpha=0.3)", method="distill",
                    temperature=1.8, distill_alpha=0.3, mix_source_ratio=0.3),
        base_config(name="Distill (T=2.2, alpha=0.35)", method="distill",
                    temperature=2.2, distill_alpha=0.35, mix_source_ratio=0.3),
        base_config(name="Distill (T=2.5, alpha=0.4)", method="distill",
                    temperature=2.5, distill_alpha=0.4, mix_source_ratio=0.3),
        base_config(name="DANN (lambda=0.05)", method="dann", dann_lambda=0.05),
        base_config(name="DANN (lambda=0.1)", method="dann", dann_lambda=0.1),
        base_config(name="DANN (lambda=0.2)", method="dann", dann_lambda=0.2),
        base_config(name="Medium + DANN (512, lambda=0.1)", method="dann",
                    embed_dim=512, num_heads=8, num_layers=3, dann_lambda=0.1),
        base_config(name="MMD (lambda=0.05)", method="mmd", mmd_lambda=0.05),
        base_config(name="MMD (lambda=0.1)", method="mmd", mmd_lambda=0.1),
        base_config(name="Pseudo (threshold=0.85)", method="pseudo", threshold=0.85),
        base_config(name="Pseudo (threshold=0.90)", method="pseudo", threshold=0.90),
        base_config(name="Pseudo (threshold=0.95)", method="pseudo", threshold=0.95),
        base_config(name="Medium + Pseudo (512, threshold=0.90)", method="pseudo",
                    embed_dim=512, num_heads=8, num_layers=3, threshold=0.90),
        base_config(name="Distill + Pseudo", method="distill_pseudo",
                    temperature=2.0, distill_alpha=0.3,
                    mix_source_ratio=0.3, threshold=0.90),
    ]
    return {
        "exp3_v1": v1_configs,
        "exp3_v2": v2_configs,
        "exp3_v3": v3_configs,
        "exp3_v4": v4_configs,
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def model_config(config: dict) -> dict:
    return {key: config[key] for key in MODEL_KEYS}


def make_model(config: dict, data: dict, device: torch.device):
    return CrossViTFaultDiagnosis(
        in_channels=data["n_channels"], num_classes=data["n_classes"],
        time_seq_len=data["window_size"], spec_height=data["spec_size"][0],
        spec_width=data["spec_size"][1], embed_dim=config["embed_dim"],
        num_heads=config["num_heads"], num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)


def cpu_state(model) -> dict:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def dataset_for_split(data: dict, split: str):
    return MultiModalFaultDataset(
        data[f"X_{split}_time"], data[f"X_{split}_spec"],
        data[f"y_{split}"], augment=False,
    )


def evaluate(model, data: dict, device: torch.device, split: str) -> dict:
    loader = DataLoader(dataset_for_split(data, split), batch_size=64,
                        shuffle=False, num_workers=0)
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for time_x, spec_x, label in loader:
            logits = model(time_x.to(device), spec_x.to(device))
            predictions.append(logits.argmax(1).cpu().numpy())
            labels.append(label.numpy())
    predictions = np.concatenate(predictions)
    labels = np.concatenate(labels)
    return {
        "accuracy": float((predictions == labels).mean() * 100.0),
        "predictions": predictions.tolist(),
        "labels": labels.tolist(),
    }


def metric_only(result: dict) -> dict:
    return {"accuracy": result["accuracy"]}


def source_cache_key(config: dict, seed: int) -> str:
    payload = json.dumps(
        {"model": model_config(config), "seed": seed}, sort_keys=True
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def train_or_load_source(config: dict, source_data: dict, device: torch.device,
                         seed: int, max_epochs: int, run_dir: Path):
    key = source_cache_key(config, seed)
    path = run_dir / "models" / "source_cache" / f"source_{key}.pth"
    if path.exists():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = make_model(config, source_data, device)
        model.load_state_dict(checkpoint["model_state"])
        return model, checkpoint["history"], path

    seed_everything(seed)
    model, best_val, history = v3.train_on_source(
        model_config(config), source_data, device, seed=seed, max_epochs=max_epochs
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "source_model", "seed": seed,
        "config": model_config(config), "best_source_val_accuracy": float(best_val),
        "class_names": CLASS_NAMES, "model_state": cpu_state(model),
        "history": history,
    }, path)
    return model, history, path


def fine_tune_baseline(model, target_data: dict, config: dict,
                       device: torch.device, seed: int, max_epochs: int,
                       patience: int):
    seed_everything(seed)
    for parameter in model.parameters():
        parameter.requires_grad = True
    v3.freeze_layers_correctly(
        model, config.get("freeze_backbone", False),
        config.get("freeze_layers"), config.get("freeze_embedding", False),
    )
    train_loader = DataLoader(
        MultiModalFaultDataset(
            target_data["X_train_time"], target_data["X_train_spec"],
            target_data["y_train"], augment=False,
        ), batch_size=config["batch_size"], shuffle=True, num_workers=0,
    )
    val_loader = DataLoader(dataset_for_split(target_data, "val"),
                            batch_size=config["batch_size"], shuffle=False,
                            num_workers=0)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.get("label_smoothing", 0.05)
    )
    optimizer = optim.Adam(parameters, lr=config["ft_lr"],
                           weight_decay=config["weight_decay"])
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_accuracy, stale, best_state = -1.0, 0, None
    for epoch in range(max_epochs):
        model.train()
        total_loss = total = correct = 0
        for time_x, spec_x, labels in train_loader:
            time_x, spec_x, labels = time_x.to(device), spec_x.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(time_x, spec_x)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total += labels.numel()
            correct += int((logits.argmax(1) == labels).sum().item())
        history["train_loss"].append(total_loss / len(train_loader))
        history["train_acc"].append(100.0 * correct / total)

        model.eval()
        val_loss = val_total = val_correct = 0
        with torch.no_grad():
            for time_x, spec_x, labels in val_loader:
                time_x, spec_x, labels = time_x.to(device), spec_x.to(device), labels.to(device)
                logits = model(time_x, spec_x)
                val_loss += float(criterion(logits, labels).item())
                val_total += labels.numel()
                val_correct += int((logits.argmax(1) == labels).sum().item())
        val_accuracy = 100.0 * val_correct / val_total
        history["val_loss"].append(val_loss / len(val_loader))
        history["val_acc"].append(val_accuracy)
        if val_accuracy > best_accuracy:
            best_accuracy, stale, best_state = val_accuracy, 0, cpu_state(model)
        else:
            stale += 1
        if (epoch + 1) % 10 == 0:
            print(f"    target epoch {epoch + 1}/{max_epochs}: val={val_accuracy:.2f}, best={best_accuracy:.2f}", flush=True)
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_accuracy, history


def merge_histories(first: dict, second: dict) -> dict:
    return {key: list(first.get(key, [])) + list(second.get(key, []))
            for key in set(first) | set(second)}


def fine_tune(config: dict, model, source_data: dict, target_data: dict,
              device: torch.device, seed: int, max_epochs: int, patience: int):
    seed_everything(seed)
    method = config.get("method", "baseline")
    if method == "baseline":
        return fine_tune_baseline(model, target_data, config, device, seed,
                                  max_epochs, patience)
    if method == "ewc":
        return v3.fine_tune_with_ewc(
            model, target_data, source_data, device, model_config(config),
            lr=config["ft_lr"], max_epochs=max_epochs,
            freeze_backbone=config.get("freeze_backbone", False),
            freeze_layers=config.get("freeze_layers"),
            freeze_embedding=config.get("freeze_embedding", False),
            ewc_lambda=config["ewc_lambda"],
        )
    if method == "distill":
        return v3.fine_tune_with_distillation(
            model, target_data, source_data, device, model_config(config),
            lr=config["ft_lr"], max_epochs=max_epochs,
            freeze_backbone=config.get("freeze_backbone", False),
            freeze_layers=config.get("freeze_layers"),
            freeze_embedding=config.get("freeze_embedding", False),
            temperature=config["temperature"],
            distill_alpha=config["distill_alpha"],
            mix_source_ratio=config["mix_source_ratio"],
        )
    if method == "dann":
        return v4.fine_tune_with_dann(
            model, target_data, source_data, device, model_config(config),
            lr=config["ft_lr"], max_epochs=max_epochs,
            dann_lambda=config["dann_lambda"],
        )
    if method == "mmd":
        return v4.fine_tune_with_mmd(
            model, target_data, source_data, device, model_config(config),
            lr=config["ft_lr"], max_epochs=max_epochs,
            mmd_lambda=config["mmd_lambda"],
        )
    if method == "pseudo":
        return v4.fine_tune_with_pseudo_labels(
            model, target_data, source_data, device, model_config(config),
            lr=config["ft_lr"], max_epochs=max_epochs,
            threshold=config["threshold"],
        )
    if method == "distill_pseudo":
        first_epochs = max(1, max_epochs // 2)
        model, _, first = v4.fine_tune_with_distillation(
            model, target_data, source_data, device, model_config(config),
            lr=config["ft_lr"], max_epochs=first_epochs,
            temperature=config["temperature"],
            distill_alpha=config["distill_alpha"],
            mix_source_ratio=config["mix_source_ratio"],
        )
        model, best, second = v4.fine_tune_with_pseudo_labels(
            model, target_data, source_data, device, model_config(config),
            lr=config["ft_lr"] * 0.5,
            max_epochs=max(1, max_epochs - first_epochs),
            threshold=config["threshold"],
        )
        return model, best, merge_histories(first, second)
    raise ValueError(f"Unknown transfer method: {method}")


def plot_history(history: dict, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.get("train_loss", []), label="train")
    axes[0].plot(history.get("val_loss", []), label="validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(history.get("train_acc", []), label="train")
    axes[1].plot(history.get("val_acc", []), label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dual_history(source_history: dict, target_history: dict,
                      path: Path, title: str) -> None:
    """Plot source/target loss and accuracy as four panels and eight curves."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    panels = (
        (axes[0, 0], source_history, "loss", "Source Domain - Loss", "Loss"),
        (axes[0, 1], source_history, "acc", "Source Domain - Accuracy", "Accuracy (%)"),
        (axes[1, 0], target_history, "loss", "Target Domain - Loss", "Loss"),
        (axes[1, 1], target_history, "acc", "Target Domain - Accuracy", "Accuracy (%)"),
    )
    for axis, history, metric, panel_title, ylabel in panels:
        axis.plot(history.get(f"train_{metric}", []), marker="o", markersize=2,
                  linewidth=1.6, label="Train")
        axis.plot(history.get(f"val_{metric}", []), marker="s", markersize=2,
                  linewidth=1.6, label="Validation")
        axis.set_title(panel_title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.28)
        axis.legend()
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(labels: list[int], predictions: list[int], path: Path,
                   title: str) -> None:
    matrix = confusion_matrix(labels, predictions, labels=list(range(15)))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 9))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(15), yticks=range(15),
             xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel="Predicted", ylabel="True", title=title)
    axis.tick_params(axis="x", rotation=90, labelsize=7)
    axis.tick_params(axis="y", labelsize=7)
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dual_confusion(source_result: dict, target_result: dict, path: Path,
                        title: str, split_name: str = "Validation") -> None:
    """Plot readable row-normalized source and target confusion matrices."""
    short_names = ["H"] + [f"IC-{value}Ω" for value in (6, 5, 4, 3, 2, 1, 0.5)] + [
        f"IT-{value}Ω" for value in (6, 5, 4, 3, 2, 1, 0.5)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(23, 10), constrained_layout=True)
    image = None
    for axis, domain, result in (
        (axes[0], "Source", source_result), (axes[1], "Target", target_result)
    ):
        matrix = confusion_matrix(
            result["labels"], result["predictions"], labels=list(range(15))
        ).astype(float)
        denominator = np.maximum(matrix.sum(axis=1, keepdims=True), 1.0)
        normalized = matrix / denominator * 100.0
        image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=100)
        for row in range(15):
            for column in range(15):
                value = normalized[row, column]
                if value >= 0.5:
                    axis.text(column, row, f"{value:.0f}", ha="center", va="center",
                              fontsize=6, color="white" if value >= 55 else "#202020")
        axis.set(xticks=range(15), yticks=range(15),
                 xticklabels=short_names, yticklabels=short_names,
                 xlabel="Predicted class", ylabel="True class",
                 title=f"{domain} {split_name} - Acc {result['accuracy']:.2f}%")
        axis.tick_params(axis="x", rotation=60, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
    fig.colorbar(image, ax=axes, shrink=0.84, label="Row-normalized percentage (%)")
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_config_comparison(aggregates: list[dict], path: Path, version: str) -> None:
    names = [item["config_name"] for item in aggregates]
    target = [item["target_val_mean"] for item in aggregates]
    source = [item["source_val_mean"] for item in aggregates]
    positions = np.arange(len(names))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(max(12, len(names) * 0.75), 6))
    axis.bar(positions - 0.2, source, width=0.4, label="source validation")
    axis.bar(positions + 0.2, target, width=0.4, label="target validation")
    axis.set_xticks(positions, names, rotation=65, ha="right", fontsize=8)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(f"{version.upper()} validation comparison")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validation_balance(source_before: float, source_after: float,
                       target_after: float) -> float:
    return target_after - max(0.0, source_before - source_after)


def save_model_checkpoint(path: Path, model, config: dict, seed: int,
                          metrics: dict, source_checkpoint: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "target_finetuned_model", "seed": seed,
        "config": config, "class_names": CLASS_NAMES,
        "label_mapping": "healthy; intercoil 6..0.5 ohm; interturn 6..0.5 ohm",
        "source_checkpoint": str(source_checkpoint), "metrics": metrics,
        "model_state": cpu_state(model),
    }, path)


def load_checkpoint_model(path: Path, data: dict, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(checkpoint["config"], data, device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def run_exp1_exp2(args, source_data, target_data, device, run_dir: Path,
                  summary: dict) -> None:
    config = base_config(name="Corrected CrossViT baseline")
    exp1_runs, exp2_runs = [], []
    for seed in args.seeds:
        model, history, source_path = train_or_load_source(
            config, source_data, device, seed, args.max_epochs, run_dir
        )
        exp1_test = evaluate(model, source_data, device, "test")
        exp2_target = evaluate(model, target_data, device, "test")
        model_copy = run_dir / "models" / "exp1" / f"exp1_seed{seed}.pth"
        model_copy.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        checkpoint["exp1_test_accuracy"] = exp1_test["accuracy"]
        torch.save(checkpoint, model_copy)
        exp1_runs.append({"seed": seed, "test_accuracy": exp1_test["accuracy"],
                          "model": str(model_copy)})
        exp2_runs.append({
            "seed": seed, "source_test_accuracy": exp1_test["accuracy"],
            "target_test_accuracy": exp2_target["accuracy"],
            "performance_drop": exp1_test["accuracy"] - exp2_target["accuracy"],
            "model": str(model_copy),
        })
        plot_history(history, run_dir / "visualizations" / "exp1" /
                     f"training_seed{seed}.png", f"EXP1 source training, seed {seed}")
        plot_confusion(exp1_test["labels"], exp1_test["predictions"],
                       run_dir / "visualizations" / "exp1" /
                       f"confusion_seed{seed}.png", f"EXP1 test, seed {seed}")
        plot_confusion(exp2_target["labels"], exp2_target["predictions"],
                       run_dir / "visualizations" / "exp2" /
                       f"target_confusion_seed{seed}.png", f"EXP2 target test, seed {seed}")
    summary["exp1"] = {
        "runs": exp1_runs,
        "test_mean": float(np.mean([item["test_accuracy"] for item in exp1_runs])),
        "test_std": float(np.std([item["test_accuracy"] for item in exp1_runs])),
    }
    summary["exp2"] = {
        "runs": exp2_runs,
        "source_test_mean": float(np.mean([item["source_test_accuracy"] for item in exp2_runs])),
        "target_test_mean": float(np.mean([item["target_test_accuracy"] for item in exp2_runs])),
        "target_test_std": float(np.std([item["target_test_accuracy"] for item in exp2_runs])),
    }
    atomic_json(run_dir / "json" / "exp1.json", summary["exp1"])
    atomic_json(run_dir / "json" / "exp2.json", summary["exp2"])


def aggregate_config_results(configs: list[dict], per_run: list[dict]) -> list[dict]:
    aggregates = []
    for config in configs:
        rows = [row for row in per_run if row["config_name"] == config["name"]]
        aggregates.append({
            "config_name": config["name"], "method": config["method"],
            "config": config,
            "source_val_mean": float(np.mean([row["source_after"]["val_accuracy"] for row in rows])),
            "source_val_std": float(np.std([row["source_after"]["val_accuracy"] for row in rows])),
            "target_val_mean": float(np.mean([row["target_after"]["val_accuracy"] for row in rows])),
            "target_val_std": float(np.std([row["target_after"]["val_accuracy"] for row in rows])),
            "balance_mean": float(np.mean([row["validation_balance"] for row in rows])),
            "balance_std": float(np.std([row["validation_balance"] for row in rows])),
            "seeds": [row["seed"] for row in rows],
        })
    return aggregates


def run_exp3_version(version: str, configs: list[dict], args, source_data,
                     target_data, device, run_dir: Path) -> dict:
    print(f"\n{'=' * 88}\n{version.upper()}: {len(configs)} configurations x {len(args.seeds)} seeds\n{'=' * 88}", flush=True)
    config_json_dir = run_dir / "json" / version / "configs"
    model_dir = run_dir / "models" / version
    visual_dir = run_dir / "visualizations" / version
    per_run = []
    for config_index, config in enumerate(configs, 1):
        for seed in args.seeds:
            stem = f"{config_index:02d}_{slug(config['name'])}_seed{seed}"
            result_path = config_json_dir / f"{stem}.json"
            checkpoint_path = model_dir / f"{stem}.pth"
            if result_path.exists() and checkpoint_path.exists():
                print(f"[{version}] resume: {stem}", flush=True)
                per_run.append(json.loads(result_path.read_text(encoding="utf-8")))
                continue
            print(f"[{version}] config {config_index}/{len(configs)}, seed={seed}: {config['name']}", flush=True)
            source_model, source_history, source_path = train_or_load_source(
                config, source_data, device, seed, args.max_epochs, run_dir
            )
            source_before_train = evaluate(source_model, source_data, device, "train")
            source_before_val = evaluate(source_model, source_data, device, "val")
            target_before_train = evaluate(source_model, target_data, device, "train")
            target_before_val = evaluate(source_model, target_data, device, "val")

            model = copy.deepcopy(source_model)
            model, best_target_val, target_history = fine_tune(
                config, model, source_data, target_data, device, seed,
                args.max_epochs, args.patience,
            )
            source_after_train = evaluate(model, source_data, device, "train")
            source_after_val = evaluate(model, source_data, device, "val")
            target_after_train = evaluate(model, target_data, device, "train")
            target_after_val = evaluate(model, target_data, device, "val")
            balance = validation_balance(
                source_before_val["accuracy"], source_after_val["accuracy"],
                target_after_val["accuracy"],
            )
            metrics = {
                "source_before": {
                    "train_accuracy": source_before_train["accuracy"],
                    "val_accuracy": source_before_val["accuracy"],
                },
                "target_before": {
                    "train_accuracy": target_before_train["accuracy"],
                    "val_accuracy": target_before_val["accuracy"],
                },
                "source_after": {
                    "train_accuracy": source_after_train["accuracy"],
                    "val_accuracy": source_after_val["accuracy"],
                },
                "target_after": {
                    "train_accuracy": target_after_train["accuracy"],
                    "val_accuracy": target_after_val["accuracy"],
                },
                "validation_balance": balance,
            }
            save_model_checkpoint(checkpoint_path, model, config, seed, metrics, source_path)
            result = {
                "version": version, "config_index": config_index,
                "config_name": config["name"], "config": config,
                "method": config["method"], "seed": seed,
                **metrics, "best_target_val_from_training": float(best_target_val),
                "source_history": source_history, "target_history": target_history,
                "checkpoint": str(checkpoint_path),
                "test_evaluated": False,
            }
            atomic_json(result_path, result)
            plot_dual_history(
                source_history, target_history,
                visual_dir / "training_curves" / f"{stem}.png",
                f"{version.upper()} - {config['name']} - seed {seed}",
            )
            plot_dual_confusion(
                source_after_val, target_after_val,
                visual_dir / "validation_confusion" / f"{stem}.png",
                f"{version.upper()} - {config['name']} - seed {seed}",
            )
            per_run.append(result)
            atomic_json(run_dir / "progress.json", {
                "updated_at": datetime.now().isoformat(), "active_version": version,
                "completed_config_runs": len(per_run),
                "total_config_runs": len(configs) * len(args.seeds),
                "last_completed": stem,
            })
            del model, source_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    aggregates = aggregate_config_results(configs, per_run)
    aggregates.sort(key=lambda item: (item["balance_mean"], item["target_val_mean"]),
                    reverse=True)
    selected = aggregates[0]
    selected_index = next(index for index, config in enumerate(configs, 1)
                          if config["name"] == selected["config_name"])
    print(f"[{version}] locked by validation: {selected['config_name']}", flush=True)

    final_tests = []
    for seed in args.seeds:
        stem = f"{selected_index:02d}_{slug(selected['config_name'])}_seed{seed}"
        checkpoint_path = model_dir / f"{stem}.pth"
        model, _ = load_checkpoint_model(checkpoint_path, source_data, device)
        source_test = evaluate(model, source_data, device, "test")
        target_test = evaluate(model, target_data, device, "test")
        final_tests.append({
            "seed": seed, "source_test_accuracy": source_test["accuracy"],
            "target_test_accuracy": target_test["accuracy"],
            "source_predictions": source_test["predictions"],
            "source_labels": source_test["labels"],
            "target_predictions": target_test["predictions"],
            "target_labels": target_test["labels"],
            "checkpoint": str(checkpoint_path),
        })
        plot_confusion(source_test["labels"], source_test["predictions"],
                       visual_dir / "locked_test" / f"source_confusion_seed{seed}.png",
                       f"{version.upper()} locked source test, seed {seed}")
        plot_confusion(target_test["labels"], target_test["predictions"],
                       visual_dir / "locked_test" / f"target_confusion_seed{seed}.png",
                       f"{version.upper()} locked target test, seed {seed}")

    plot_config_comparison(
        aggregates, visual_dir / "validation_config_comparison.png", version
    )
    result = {
        "version": version, "selection_protocol": (
            "All configurations ranked by mean validation balance across seeds; "
            "test evaluated only for the locked configuration"
        ),
        "validation_aggregates": aggregates,
        "selected_config": selected,
        "final_test_runs": final_tests,
        "source_test_mean": float(np.mean([row["source_test_accuracy"] for row in final_tests])),
        "source_test_std": float(np.std([row["source_test_accuracy"] for row in final_tests])),
        "target_test_mean": float(np.mean([row["target_test_accuracy"] for row in final_tests])),
        "target_test_std": float(np.std([row["target_test_accuracy"] for row in final_tests])),
    }
    atomic_json(run_dir / "json" / version / f"{version}_complete.json", result)
    return result


def fmt(value: float) -> str:
    return f"{value:.2f}%"


def generate_report(summary: dict, run_dir: Path) -> Path:
    report = [
        "# 修正后 EXP3 V1/V2/V3/V4 综合实验报告",
        "",
        "## 实验概述",
        "",
        f"- 任务：{summary['source_power']} → {summary['target_power']} 跨功率迁移学习",
        "- 类别：15类（健康类1个、线圈间7级、匝间7级）",
        "- 划分：训练/测试/验证 = 70%/20%/10%，按原始记录连续时间块切分",
        f"- 随机种子：{summary['seeds']}",
        "- 选择协议：仅使用验证集选择配置；锁定后测试集只评估一次",
        "",
        "## 一、数据与评估协议修正",
        "",
        "本轮结果来自重新解析的 TDMS 格式 v2 数据。跨功率标签按旁路电阻和故障类型对齐；健康重复记录合并。窗口不会跨越记录或集合边界，集合边界保留一个完整窗口的隔离带。",
        "",
        "## 二、EXP1 单域基线",
        "",
        f"源域测试准确率：{fmt(summary['exp1']['test_mean'])} ± {fmt(summary['exp1']['test_std'])}。",
        "",
        "## 三、EXP2 跨域无适应",
        "",
        f"源域测试：{fmt(summary['exp2']['source_test_mean'])}；目标域测试：{fmt(summary['exp2']['target_test_mean'])} ± {fmt(summary['exp2']['target_test_std'])}。",
        "",
        "## 四、EXP3 各版本验证集配置比较",
        "",
    ]
    for version in ("exp3_v1", "exp3_v2", "exp3_v3", "exp3_v4"):
        if version not in summary:
            continue
        value = summary[version]
        report.extend([
            f"### {version.upper()}", "",
            "| 排名 | 配置 | 方法 | 源域验证 | 目标域验证 | 验证平衡分 |",
            "|---:|---|---|---:|---:|---:|",
        ])
        for rank, row in enumerate(value["validation_aggregates"], 1):
            report.append(
                f"| {rank} | {row['config_name']} | {row['method']} | "
                f"{row['source_val_mean']:.2f}±{row['source_val_std']:.2f} | "
                f"{row['target_val_mean']:.2f}±{row['target_val_std']:.2f} | "
                f"{row['balance_mean']:.2f}±{row['balance_std']:.2f} |"
            )
        report.extend([
            "", f"锁定配置：**{value['selected_config']['config_name']}**。",
            f"最终源域测试：{fmt(value['source_test_mean'])} ± {fmt(value['source_test_std'])}；"
            f"目标域测试：{fmt(value['target_test_mean'])} ± {fmt(value['target_test_std'])}。",
            "",
        ])
    report.extend([
        "## 五、可视化",
        "",
        "每个配置和随机种子的训练曲线均保存在本次运行目录的 `visualizations/` 下；每个版本包含完整验证配置对比图，以及锁定配置在源域和目标域测试集上的混淆矩阵。",
        "",
        "## 六、结论",
        "",
        "结论仅依据本轮修正数据和锁定测试结果生成。历史格式 v1 数据产生的指标不参与本报告比较。具体最佳方法、源域遗忘和目标域迁移表现见上表。",
        "",
        "## 七、复现信息",
        "",
        f"- 运行目录：`{run_dir}`",
        f"- Python：{summary['environment']['python']}",
        f"- PyTorch：{summary['environment']['torch']}",
        f"- CUDA设备：{summary['environment']['device']}",
        f"- 最大训练轮数：{summary['max_epochs']}，早停耐心：{summary['patience']}",
        "",
    ])
    path = run_dir / "EXP3_CORRECTED_EXPERIMENT_REPORT.md"
    path.write_text("\n".join(report), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-power", default="1.0kW")
    parser.add_argument("--target-power", default="3.0kW")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--max-windows-per-class", type=int, default=256)
    parser.add_argument("--stages", nargs="+", default=[
        "exp1", "exp2", "exp3_v1", "exp3_v2", "exp3_v3", "exp3_v4"
    ])
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.seeds = [42]
        args.max_epochs = min(args.max_epochs, 2)
        args.patience = 2
        args.max_windows_per_class = 30

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (args.run_dir or RESULTS_DIR / f"corrected_retrain_{timestamp}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "training.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_stream)
    sys.stderr = Tee(sys.__stderr__, log_stream)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Run directory: {run_dir}", flush=True)
    print(f"Device: {device}", flush=True)

    source_data = load_csv_data(
        str(DATASETS_DIR / f"dataset2_{args.source_power}.csv"),
        split_mode="time_blocked", max_windows_per_class=args.max_windows_per_class,
    )
    target_data = load_csv_data(
        str(DATASETS_DIR / f"dataset2_{args.target_power}.csv"),
        split_mode="time_blocked", max_windows_per_class=args.max_windows_per_class,
    )
    summary = {
        "run_dir": str(run_dir), "started_at": datetime.now().isoformat(),
        "source_power": args.source_power, "target_power": args.target_power,
        "seeds": args.seeds, "max_epochs": args.max_epochs,
        "patience": args.patience, "max_windows_per_class": args.max_windows_per_class,
        "class_names": CLASS_NAMES, "split_protocol": source_data["split_protocol"],
        "environment": {
            "python": platform.python_version(), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        },
    }
    atomic_json(run_dir / "run_config.json", summary)

    if "exp1" in args.stages or "exp2" in args.stages:
        run_exp1_exp2(args, source_data, target_data, device, run_dir, summary)
        atomic_json(run_dir / "summary.json", summary)

    configs_by_version = experiment_configs()
    for version in ("exp3_v1", "exp3_v2", "exp3_v3", "exp3_v4"):
        if version not in args.stages:
            continue
        configs = configs_by_version[version]
        if args.smoke:
            configs = configs[:1]
        summary[version] = run_exp3_version(
            version, configs, args, source_data, target_data, device, run_dir
        )
        atomic_json(run_dir / "summary.json", summary)

    summary["completed_at"] = datetime.now().isoformat()
    report = generate_report(summary, run_dir)
    summary["report"] = str(report)
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(run_dir / "progress.json", {
        "status": "complete", "completed_at": summary["completed_at"],
        "report": str(report),
    })
    print(f"Complete. Report: {report}", flush=True)


if __name__ == "__main__":
    main()
