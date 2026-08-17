#!/usr/bin/env python3
"""EXP3-V6: standard unsupervised DANN, MMD and pseudo-label adaptation."""

from __future__ import annotations

import argparse, copy, json, random, sys
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
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))
from _project_paths import RESULTS_DIR
from exp3_v5_single_model_transfer import (
    atomic_json, evaluate, load_cached, plot_confusion, plot_curves, standardize,
)


class V6Model(nn.Module):
    def __init__(self, input_dim=376, hidden=512, dropout=.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden // 2, 15)

    def forward(self, x, return_features=False):
        features = self.encoder(x); logits = self.classifier(features)
        return (logits, features) if return_features else logits


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient; return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def xy_loader(X, y, batch=128, shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
                      batch_size=batch, shuffle=shuffle, num_workers=0, drop_last=shuffle)


def x_loader(X, batch=128, shuffle=True):
    # Deliberately no target label tensor in this loader.
    return DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch,
                      shuffle=shuffle, num_workers=0, drop_last=shuffle)


def pretrain_source(model, source, device, epochs=50):
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    train = xy_loader(source["X_train"], source["y_train"])
    best = (-1, None)
    for _ in range(epochs):
        model.train()
        for xb, yb in train:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
        score = evaluate(model, source["X_val"], source["y_val"], device, criterion)["accuracy"]
        if score > best[0]: best = (score, copy.deepcopy(model.state_dict()))
    model.load_state_dict(best[1])


def rbf_mmd(source, target):
    joined = torch.cat([source, target], 0)
    distance = torch.cdist(joined, joined).square()
    bandwidth = distance.detach().median().clamp_min(1e-6)
    kernels = sum(torch.exp(-distance / (bandwidth * scale)) for scale in (.25, .5, 1., 2., 4.))
    n = len(source)
    return kernels[:n, :n].mean() + kernels[n:, n:].mean() - 2 * kernels[:n, n:].mean()


def adapt_unlabeled(model, source, target_X, method, parameter, device, epochs=60):
    """Adapt without accepting target labels; return fixed-epoch snapshots."""
    source_loader = xy_loader(source["X_train"], source["y_train"])
    target_loader = x_loader(target_X)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    source_criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    discriminator = None
    if method == "dann":
        discriminator = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 2)).to(device)
        optimizer = torch.optim.AdamW(list(model.parameters()) + list(discriminator.parameters()),
                                      lr=5e-4, weight_decay=3e-4)
    snapshots, unsupervised = [], []
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            for (sx, sy), (tx,) in zip(source_loader, cycle(target_loader)):
                sx, sy, tx = sx.to(device), sy.to(device), tx.to(device); optimizer.zero_grad()
                source_logits, source_feat = model(sx, True)
                _, target_feat = model(tx, True)
                loss = source_criterion(source_logits, sy)
                extra = torch.tensor(0., device=device); coverage = 0.
                if method == "dann":
                    progress = (epoch - 1) / max(epochs - 1, 1)
                    coefficient = parameter * (2 / (1 + np.exp(-10 * progress)) - 1)
                    features = ReverseGradient.apply(torch.cat([source_feat, target_feat]), coefficient)
                    domain_y = torch.cat([torch.zeros(len(sx)), torch.ones(len(tx))]).long().to(device)
                    extra = F.cross_entropy(discriminator(features), domain_y)
                    loss = loss + extra
                elif method == "mmd":
                    extra = rbf_mmd(source_feat, target_feat)
                    loss = loss + parameter * extra
                elif method == "pseudo":
                    with torch.no_grad():
                        probability = F.softmax(model(tx), dim=1)
                        confidence, pseudo = probability.max(1)
                        mask = confidence >= parameter
                    coverage = float(mask.float().mean())
                    if mask.any():
                        extra = F.cross_entropy(model(tx[mask]), pseudo[mask])
                        loss = loss + extra
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            unsupervised.append({"extra_loss": float(extra.detach()), "pseudo_coverage": coverage})
        snapshots.append({key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    return snapshots, unsupervised


def offline_history(model, snapshots, source, target, device):
    """Use target labels only after adaptation has completely finished."""
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    history = {d: {f"{s}_{m}": [] for s in ("train", "val") for m in ("loss", "accuracy")}
               for d in ("source", "target")}
    for state in snapshots:
        model.load_state_dict(state)
        for domain, data in (("source", source), ("target", target)):
            for split in ("train", "val"):
                result = evaluate(model, data[f"X_{split}"], data[f"y_{split}"], device, criterion)
                history[domain][f"{split}_loss"].append(result["loss"])
                history[domain][f"{split}_accuracy"].append(result["accuracy"])
    model.load_state_dict(snapshots[-1])
    final = {domain: evaluate(model, data["X_val"], data["y_val"], device, criterion)
             for domain, data in (("source", source), ("target", target))}
    return history, final


def configurations():
    return ([{"method": "dann", "parameter": x} for x in (.05, .1, .2)] +
            [{"method": "mmd", "parameter": x} for x in (.1, .5, 1.)] +
            [{"method": "pseudo", "parameter": x} for x in (.7, .8, .9)])


def plot_bars(rows, tests, output_dir):
    grouped = []
    for config in configurations():
        selected = [r for r in rows if r["config"] == config]
        grouped.append((f"{config['method']}\n{config['parameter']}",
                        np.mean([r["source_val_accuracy"] for r in selected]),
                        np.mean([r["target_val_accuracy_diagnostic_only"] for r in selected])))
    x = np.arange(len(grouped)); width = .36
    fig, axis = plt.subplots(figsize=(15, 7))
    axis.bar(x - width / 2, [r[1] for r in grouped], width, label="Source validation")
    axis.bar(x + width / 2, [r[2] for r in grouped], width, label="Target validation (diagnostic only)")
    axis.axhline(100 / 15, color="gray", linestyle=":", label="15-class chance")
    axis.set_xticks(x, [r[0] for r in grouped]); axis.set_ylim(0, 100)
    axis.set(ylabel="Accuracy (%)", title="V6 unsupervised adaptation validation diagnostics")
    axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "validation_config_bars.png", dpi=180, bbox_inches="tight"); plt.close(fig)
    methods = ["dann", "mmd", "pseudo"]
    source_means = [np.mean([t["results"]["source"]["accuracy"] for t in tests if t["method"] == m]) for m in methods]
    target_means = [np.mean([t["results"]["target"]["accuracy"] for t in tests if t["method"] == m]) for m in methods]
    fig, axis = plt.subplots(figsize=(10, 6))
    x = np.arange(3)
    axis.bar(x - width / 2, source_means, width, label="Source test")
    axis.bar(x + width / 2, target_means, width, label="Target test")
    axis.axhline(100 / 15, color="gray", linestyle=":", label="15-class chance")
    axis.set_xticks(x, [m.upper() for m in methods]); axis.set_ylim(0, 100)
    axis.set(ylabel="Accuracy (%)", title="V6 prelocked test comparison")
    axis.grid(axis="y", alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(output_dir / "locked_test_method_bars.png", dpi=180, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    args = parser.parse_args(); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = args.run_dir or RESULTS_DIR / f"exp3_v6_unsupervised_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True, exist_ok=True)
    raw_s = load_cached(run, "1.0kW", 8192, 256); raw_t = load_cached(run, "3.0kW", 8192, 256)
    source, target, mean, scale = standardize(raw_s, raw_t)
    # Fixed before any target-label evaluation: one representative per method.
    atomic_json(run / "prelocked_protocol.json", {"test_configs": {"dann": .1, "mmd": .5, "pseudo": .8},
        "selection": "prelocked; target labels excluded from adaptation and selection"})
    rows = []
    for index, config in enumerate(configurations(), 1):
        for seed in args.seeds:
            stem = f"{index:02d}_{config['method']}_{config['parameter']}_seed{seed}".replace(".", "p")
            print(stem, flush=True); seed_all(seed)
            model = V6Model().to(device); pretrain_source(model, source, device)
            snapshots, unsupervised = adapt_unlabeled(
                model, source, target["X_train"], config["method"], config["parameter"], device)
            history, validation = offline_history(model, snapshots, source, target, device)
            checkpoint = run / "models" / f"{stem}.pth"; checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": snapshots[-1], "config": config, "seed": seed,
                        "mean": mean, "scale": scale, "target_labels_used_for_training": False}, checkpoint)
            row = {"stem": stem, "config": config, "seed": seed, "validation": validation,
                   "source_val_accuracy": validation["source"]["accuracy"],
                   "target_val_accuracy_diagnostic_only": validation["target"]["accuracy"],
                   "unsupervised_history": unsupervised, "checkpoint": str(checkpoint)}
            atomic_json(run / "json" / f"{stem}.json", row); rows.append(row)
            plot_curves(history, run / "visualizations/training_curves" / f"{stem}.png", stem)
            plot_confusion(validation, run / "visualizations/validation_confusion" / f"{stem}.png",
                           stem, "validation diagnostic")
            print(f"  source={row['source_val_accuracy']:.2f} target_diag={row['target_val_accuracy_diagnostic_only']:.2f}", flush=True)
    tests = []
    locked = {"dann": .1, "mmd": .5, "pseudo": .8}
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    for row in rows:
        if row["config"]["parameter"] != locked[row["config"]["method"]]: continue
        checkpoint = torch.load(row["checkpoint"], map_location="cpu", weights_only=False)
        model = V6Model().to(device); model.load_state_dict(checkpoint["model_state"])
        result = {d: evaluate(model, data["X_test"], data["y_test"], device, criterion)
                  for d, data in (("source", source), ("target", target))}
        tests.append({"method": row["config"]["method"], "seed": row["seed"], "results": result,
                      "checkpoint": row["checkpoint"]})
        plot_confusion(result, run / "visualizations/locked_test" / f"{row['stem']}.png",
                       f"V6 prelocked {row['config']['method']} seed {row['seed']}", "test")
    summary = {"training_protocol": "strict unsupervised target adaptation; target labels diagnostic/test only",
               "all_validation_runs": rows, "prelocked_tests": tests}
    atomic_json(run / "summary.json", summary)
    plot_bars(rows, tests, run / "visualizations")
    for method in locked:
        chosen = [x for x in tests if x["method"] == method]
        print(method, np.mean([x["results"]["source"]["accuracy"] for x in chosen]),
              np.mean([x["results"]["target"]["accuracy"] for x in chosen]), flush=True)


if __name__ == "__main__": main()
