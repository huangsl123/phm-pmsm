#!/usr/bin/env python3
"""EXP3-V7: corrected supervised LwF, KD, EWC and experience replay."""

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

SCRIPT_DIR = Path(__file__).resolve().parent; PROJECT_ROOT = SCRIPT_DIR.parent
sys.path[:0] = [str(SCRIPT_DIR), str(PROJECT_ROOT)]
from _project_paths import RESULTS_DIR
from exp3_v5_single_model_transfer import (
    TransferMLP, atomic_json, evaluate, load_cached, plot_confusion, plot_curves, standardize,
)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def loader(X, y, batch=128, shuffle=True):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=batch,
                      shuffle=shuffle, num_workers=0, drop_last=shuffle)


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def kd_loss(student, teacher, temperature):
    return F.kl_div(F.log_softmax(student / temperature, dim=1),
                    F.softmax(teacher / temperature, dim=1), reduction="batchmean") * temperature ** 2


def source_pretrain(source, seed, device):
    seed_all(seed); model = TransferMLP(376, 512, .2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=.05); train = loader(source["X_train"], source["y_train"])
    best = (-1., None); stale = 0
    for _ in range(60):
        model.train()
        for xb, yb in train:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad()
            loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
        accuracy = evaluate(model, source["X_val"], source["y_val"], device, criterion)["accuracy"]
        if accuracy > best[0]: best = (accuracy, state(model)); stale = 0
        else: stale += 1
        if stale >= 15: break
    model.load_state_dict(best[1]); return model


def empirical_fisher(model, source, device):
    """Diagonal empirical Fisher: mean of squared per-example log-loss gradients."""
    criterion = nn.CrossEntropyLoss(); fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    data = loader(source["X_train"], source["y_train"], batch=1, shuffle=False)
    model.eval()
    for xb, yb in data:
        xb, yb = xb.to(device), yb.to(device); model.zero_grad(set_to_none=True)
        criterion(model(xb), yb).backward()
        for n, p in model.named_parameters():
            if p.grad is not None: fisher[n] += p.grad.detach().square()
    for n in fisher: fisher[n] /= len(data)
    model.zero_grad(set_to_none=True); return fisher


def ewc_penalty(model, fisher, means):
    return sum((fisher[n] * (p - means[n]).square()).sum() for n, p in model.named_parameters()) / 2


def configs():
    return ([{"method": "lwf", "alpha": a, "temperature": 2.0} for a in (.2, .5, .8)] +
            [{"method": "distill", "weight": w, "temperature": 2.0} for w in (.2, .5, 1.0)] +
            [{"method": "ewc", "lambda": w} for w in (100., 1000., 10000.)] +
            [{"method": "replay", "weight": w} for w in (.25, .5, 1.0)])


def adapt(model, teacher, source, target, config, fisher, device, epochs=80):
    source_train = loader(source["X_train"], source["y_train"])
    target_train = loader(target["X_train"], target["y_train"])
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    means = {n: p.detach().clone() for n, p in teacher.named_parameters()}
    history = {d: {f"{s}_{m}": [] for s in ("train", "val") for m in ("loss", "accuracy")}
               for d in ("source", "target")}
    best = ((-1., -1.), None); stale = 0
    for epoch in range(epochs + 1):
        if epoch:
            model.train()
            if config["method"] == "lwf":
                iterator = ((None, batch) for batch in target_train)
            else:
                iterator = zip(cycle(source_train), target_train)
            for source_batch, (tx, ty) in iterator:
                tx, ty = tx.to(device), ty.to(device); optimizer.zero_grad()
                target_logits = model(tx); target_ce = criterion(target_logits, ty)
                if config["method"] == "lwf":
                    with torch.no_grad(): teacher_target = teacher(tx)
                    loss = ((1 - config["alpha"]) * target_ce + config["alpha"] *
                            kd_loss(target_logits, teacher_target, config["temperature"]))
                elif config["method"] == "distill":
                    sx, _ = source_batch; sx = sx.to(device)
                    with torch.no_grad(): teacher_source = teacher(sx)
                    loss = target_ce + config["weight"] * kd_loss(model(sx), teacher_source, config["temperature"])
                elif config["method"] == "ewc":
                    loss = target_ce + config["lambda"] * ewc_penalty(model, fisher, means)
                else:
                    sx, sy = source_batch; sx, sy = sx.to(device), sy.to(device)
                    loss = target_ce + config["weight"] * criterion(model(sx), sy)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); optimizer.step()
            scheduler.step()
        current = {}
        for domain, data in (("source", source), ("target", target)):
            for split in ("train", "val"):
                result = evaluate(model, data[f"X_{split}"], data[f"y_{split}"], device, criterion)
                history[domain][f"{split}_loss"].append(result["loss"])
                history[domain][f"{split}_accuracy"].append(result["accuracy"])
                if split == "val": current[domain] = result
        key = (min(current["source"]["accuracy"], current["target"]["accuracy"]),
               np.mean([current["source"]["accuracy"], current["target"]["accuracy"]]))
        if key > best[0]: best = (key, state(model)); stale = 0
        else: stale += 1
        if stale >= 20: break
    model.load_state_dict(best[1])
    final = {d: evaluate(model, x["X_val"], x["y_val"], device, criterion)
             for d, x in (("source", source), ("target", target))}
    return history, final, best[0]


def plot_bars(aggregates, path):
    labels = [f"{x['method']}\n{x['parameter']}" for x in aggregates]; p = np.arange(len(labels)); w=.36
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(p-w/2,[x["source_val_mean"] for x in aggregates],w,label="Source validation")
    ax.bar(p+w/2,[x["target_val_mean"] for x in aggregates],w,label="Target validation")
    ax.axhline(70,color="tab:red",ls="--"); ax.axhline(80,color="tab:green",ls=":")
    ax.set_xticks(p,labels); ax.set_ylim(0,105); ax.set_ylabel("Accuracy (%)"); ax.legend(); ax.grid(axis="y",alpha=.25)
    fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=180,bbox_inches="tight"); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--seeds",type=int,nargs="+",default=[42,123]); args=parser.parse_args()
    run=args.run_dir or RESULTS_DIR/f"exp3_v7_supervised_{datetime.now():%Y%m%d_%H%M%S}"; run.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_s=load_cached(run,"1.0kW",8192,256); raw_t=load_cached(run,"3.0kW",8192,256)
    source,target,mean,scale=standardize(raw_s,raw_t); rows=[]
    bases={}; fishers={}
    for seed in args.seeds:
        bases[seed]=source_pretrain(source,seed,device); fishers[seed]=empirical_fisher(bases[seed],source,device)
    for index,config in enumerate(configs(),1):
        parameter=config.get("alpha",config.get("weight",config.get("lambda")))
        for seed in args.seeds:
            stem=f"{index:02d}_{config['method']}_{parameter}_seed{seed}".replace(".","p"); print(stem,flush=True)
            teacher=copy.deepcopy(bases[seed]).eval(); model=copy.deepcopy(bases[seed])
            for p in teacher.parameters(): p.requires_grad=False
            history,validation,key=adapt(model,teacher,source,target,config,fishers[seed],device)
            checkpoint=run/"models"/f"{stem}.pth"; checkpoint.parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model_state":state(model),"config":config,"seed":seed,"mean":mean,"scale":scale,
                        "single_shared_model":True,"supervised_target_adaptation":True},checkpoint)
            row={"stem":stem,"config_index":index,"method":config["method"],"parameter":parameter,"config":config,"seed":seed,
                 "source_val_accuracy":validation["source"]["accuracy"],"target_val_accuracy":validation["target"]["accuracy"],
                 "minimum_val_accuracy":key[0],"validation":validation,"checkpoint":str(checkpoint)}
            atomic_json(run/"json"/f"{stem}.json",row); rows.append(row)
            plot_curves(history,run/"visualizations/training_curves"/f"{stem}.png",stem)
            plot_confusion(validation,run/"visualizations/validation_confusion"/f"{stem}.png",stem,"validation")
            print(f"  source={row['source_val_accuracy']:.2f} target={row['target_val_accuracy']:.2f}",flush=True)
    aggregates=[]
    for index,config in enumerate(configs(),1):
        x=[r for r in rows if r["config_index"]==index]
        aggregates.append({"config_index":index,"method":x[0]["method"],"parameter":x[0]["parameter"],"config":config,
            "source_val_mean":float(np.mean([r["source_val_accuracy"] for r in x])),
            "target_val_mean":float(np.mean([r["target_val_accuracy"] for r in x])),
            "minimum_val_mean":float(np.mean([r["minimum_val_accuracy"] for r in x]))})
    atomic_json(run/"validation_aggregates.json",aggregates); plot_bars(aggregates,run/"visualizations/validation_config_bars.png")
    locked={}
    for method in ("lwf","distill","ewc","replay"):
        locked[method]=max([x for x in aggregates if x["method"]==method],key=lambda x:(x["minimum_val_mean"],x["target_val_mean"]))
    atomic_json(run/"locked_configs.json",locked); criterion=nn.CrossEntropyLoss(label_smoothing=.05); tests=[]
    for method,chosen in locked.items():
        for seed in args.seeds:
            row=next(r for r in rows if r["config_index"]==chosen["config_index"] and r["seed"]==seed)
            ck=torch.load(row["checkpoint"],map_location="cpu",weights_only=False); model=TransferMLP(376,512,.2).to(device); model.load_state_dict(ck["model_state"])
            result={d:evaluate(model,x["X_test"],x["y_test"],device,criterion) for d,x in (("source",source),("target",target))}
            tests.append({"method":method,"seed":seed,"source_accuracy":result["source"]["accuracy"],"target_accuracy":result["target"]["accuracy"],"results":result,"checkpoint":row["checkpoint"]})
            plot_confusion(result,run/"visualizations/locked_test"/f"{method}_seed{seed}.png",f"V7 {method} seed {seed}","test")
    summary={"protocol":"supervised target adaptation, one shared model/head, dual-validation checkpointing","locked":locked,"tests":tests}
    atomic_json(run/"summary.json",summary)
    for method in locked:
        x=[r for r in tests if r["method"]==method]; print(method,np.mean([r["source_accuracy"] for r in x]),np.mean([r["target_accuracy"] for r in x]),flush=True)


if __name__=="__main__": main()
