"""Leakage-resistant loader for the TDMS-derived PMSM dataset (format v2)."""

from __future__ import annotations

import json
import csv
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy import signal
from torch.utils.data import DataLoader, Dataset


CLASS_NAMES = ["healthy"] + [f"intercoil_L{i}" for i in range(1, 8)] + [
    f"interturn_L{i}" for i in range(1, 8)
]


def compute_spectrogram(
    x: np.ndarray, n_fft: int = 256, hop_length: int = 64,
    win_length: Optional[int] = None,
) -> np.ndarray:
    """Return a native-resolution log-power STFT without image interpolation."""
    if x.ndim != 2:
        raise ValueError(f"Expected (time, channels), got {x.shape}")
    win_length = n_fft if win_length is None else win_length
    if not 0 < hop_length <= win_length <= len(x):
        raise ValueError("Require 0 < hop_length <= win_length <= signal length")
    specs = []
    for channel in range(x.shape[1]):
        _, _, zxx = signal.stft(
            x[:, channel], window="hann", nperseg=win_length,
            noverlap=win_length - hop_length, nfft=n_fft,
            boundary=None, padded=False,
        )
        power_db = 10.0 * np.log10(np.maximum(np.abs(zxx) ** 2, 1e-12))
        low, high = np.percentile(power_db, (1.0, 99.0))
        specs.append(np.clip((power_db - low) / max(high - low, 1e-8), 0.0, 1.0))
    return np.stack(specs).astype(np.float32, copy=False)


def normalize_signal(x: np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize each sensor channel over time (axis 0), not across phases."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected (time, channels), got {x.shape}")
    if method == "zscore":
        center, scale = x.mean(axis=0, keepdims=True), x.std(axis=0, keepdims=True)
    elif method == "robust":
        center = np.median(x, axis=0, keepdims=True)
        scale = np.percentile(x, 75, axis=0, keepdims=True) - np.percentile(
            x, 25, axis=0, keepdims=True
        )
    elif method == "minmax":
        low, high = x.min(axis=0, keepdims=True), x.max(axis=0, keepdims=True)
        return (2.0 * (x - low) / np.maximum(high - low, 1e-8) - 1.0).astype(
            np.float32, copy=False
        )
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return ((x - center) / np.maximum(scale, 1e-8)).astype(np.float32, copy=False)


class MultiModalFaultDataset(Dataset):
    def __init__(self, time_signals, spectrograms, labels, augment: bool = False):
        self.time_signals = torch.as_tensor(time_signals, dtype=torch.float32)
        self.spectrograms = torch.as_tensor(spectrograms, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        time_x = self.time_signals[index].clone()
        spec_x = self.spectrograms[index].clone()
        label = self.labels[index]
        if self.augment:
            if torch.rand(()) < 0.5:
                time_x += torch.randn_like(time_x) * 0.01
            if torch.rand(()) < 0.5:
                scale = 0.9 + 0.2 * torch.rand(())
                time_x *= scale
        return time_x, spec_x, label


class UnlabeledMultiModalDataset(Dataset):
    def __init__(self, time_signals, spectrograms):
        self.time_signals = torch.as_tensor(time_signals, dtype=torch.float32)
        self.spectrograms = torch.as_tensor(spectrograms, dtype=torch.float32)

    def __len__(self):
        return len(self.time_signals)

    def __getitem__(self, index):
        return self.time_signals[index], self.spectrograms[index]


def _resolve_power(data_path: str) -> str:
    match = re.search(r"(1\.0|1\.5|3\.0)kW", str(data_path))
    if not match:
        raise ValueError(f"Cannot infer motor power from {data_path}")
    return f"{match.group(1)}kW"


def _choose_starts(
    length: int, window_size: int, stride: int, max_windows: int,
) -> np.ndarray:
    if length < window_size:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, length - window_size + 1, stride, dtype=np.int64)
    if len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows).round().astype(int)]
    return np.unique(starts)


def _split_regions(length: int, guard: int) -> dict[str, tuple[int, int]]:
    train_end = int(length * 0.70)
    test_end = int(length * 0.90)
    regions = {
        "train": (0, max(0, train_end - guard)),
        "test": (min(length, train_end + guard), max(0, test_end - guard)),
        "val": (min(length, test_end + guard), length),
    }
    if any(end <= begin for begin, end in regions.values()):
        raise ValueError(
            f"Recording length {length} is too short for a {guard}-sample guard "
            "around train/test/validation boundaries"
        )
    return regions


def load_csv_data(
    data_path: str, window_size: int = 1024, stride: int = 128,
    n_fft: int = 256, hop_length: int = 64,
    spec_size: Optional[Tuple[int, int]] = None,
    test_size: float = 0.2, val_size: float = 0.1,
    random_state: int = 42, split_mode: str = "time_blocked",
    max_windows_per_class: int = 256, normalization: str = "zscore",
) -> Dict:
    """Load format-v2 data; the historical CSV argument is used only for power.

    Splits are contiguous regions of each original recording.  A one-window
    guard band separates regions, so overlapping samples cannot cross splits.
    The old ``721`` value is accepted as an alias for the train/test/validation
    ratio 70/20/10.
    """
    del test_size, val_size, random_state
    if split_mode not in {"time_blocked", "721"}:
        raise ValueError("Only leakage-resistant split_mode='time_blocked' is supported")
    index_path = Path(data_path).resolve()
    power = _resolve_power(str(index_path))
    datasets_dir = index_path.parent
    processed_dir = datasets_dir / "processed_v2"
    manifest_path = processed_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run `python scripts/rebuild_dataset.py` first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 2:
        raise ValueError(f"Unsupported manifest version: {manifest.get('format_version')}")
    with index_path.open(newline="", encoding="utf-8") as stream:
        records = sorted(csv.DictReader(stream), key=lambda r: int(r["class_id"]))
    if not records or {r["dataset_format"] for r in records} != {
        "pmsm_tdms_v2_recording_index"
    }:
        raise ValueError(f"{index_path} is not a format-v2 recording index")
    if {r["power_kW"] for r in records} != {power}:
        raise ValueError(f"Power metadata in {index_path} does not match its filename")
    if [int(r["class_id"]) for r in records] != list(range(15)):
        raise ValueError(f"{power} does not contain exactly the canonical 15 classes")

    if max_windows_per_class < 10:
        raise ValueError("max_windows_per_class must be at least 10 for a 7:2:1 split")
    per_split_limit = {
        "train": round(max_windows_per_class * 0.70),
        "val": round(max_windows_per_class * 0.10),
        "test": max_windows_per_class - round(max_windows_per_class * 0.70)
        - round(max_windows_per_class * 0.10),
    }
    collected = {name: {"time": [], "spec": [], "label": []} for name in per_split_limit}
    provenance = {name: [] for name in per_split_limit}

    for record in records:
        signal_path = datasets_dir / record["signal_file"]
        raw = np.load(signal_path, mmap_mode="r")
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise ValueError(f"Invalid three-phase current array: {signal_path} {raw.shape}")
        for split, (begin, end) in _split_regions(len(raw), window_size).items():
            starts = _choose_starts(
                end - begin, window_size, stride, per_split_limit[split]
            )
            for local_start in starts:
                start = begin + int(local_start)
                window = normalize_signal(raw[start:start + window_size], normalization)
                spec = compute_spectrogram(window, n_fft=n_fft, hop_length=hop_length)
                if spec_size is not None and tuple(spec.shape[1:]) != tuple(spec_size):
                    raise ValueError(
                        f"Requested spec_size={spec_size}, but the native STFT is "
                        f"{tuple(spec.shape[1:])}. Resizing is intentionally disabled."
                    )
                collected[split]["time"].append(window)
                collected[split]["spec"].append(spec)
                collected[split]["label"].append(int(record["class_id"]))
                provenance[split].append(
                    (record["source_tdms"], start, start + window_size)
                )

    result: Dict = {
        "n_classes": 15,
        "n_channels": 3,
        "window_size": window_size,
        "spec_size": (n_fft // 2 + 1, 1 + (window_size - n_fft) // hop_length),
        "fault_codes": CLASS_NAMES.copy(),
        "class_names": CLASS_NAMES.copy(),
        "sample_rate_hz": int(records[0]["sample_rate_hz"]),
        "split_protocol": "train/test/validation = 70/20/10 contiguous time blocks with one-window guards",
        "source_index": str(index_path),
        "source_manifest": str(manifest_path),
        "provenance": provenance,
    }
    for split in ("train", "test", "val"):
        result[f"X_{split}_time"] = np.stack(collected[split]["time"])
        result[f"X_{split}_spec"] = np.stack(collected[split]["spec"])
        result[f"y_{split}"] = np.asarray(collected[split]["label"], dtype=np.int64)
    return result


def load_cross_domain_data(source_power: str, target_power: str, base_path: str, **kwargs):
    base = Path(base_path)
    source = load_csv_data(str(base / f"dataset2_{source_power}.csv"), **kwargs)
    target = load_csv_data(str(base / f"dataset2_{target_power}.csv"), **kwargs)
    if source["fault_codes"] != target["fault_codes"]:
        raise ValueError("Source and target label semantics differ")
    return {"source": source, "target": target}


def create_data_loaders(data: Dict, batch_size: int = 32, num_workers: int = 0):
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = MultiModalFaultDataset(
            data[f"X_{split}_time"], data[f"X_{split}_spec"], data[f"y_{split}"],
            augment=(split == "train"),
        )
        loaders[split] = DataLoader(
            dataset, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers,
        )
    return loaders


def create_domain_adapt_loaders(
    source_data: Dict, target_data: Dict, batch_size: int = 32,
    num_workers: int = 0,
):
    source = create_data_loaders(source_data, batch_size, num_workers)
    target = create_data_loaders(target_data, batch_size, num_workers)
    return {
        "source_train": source["train"], "source_val": source["val"],
        "source_test": source["test"], "target_train": target["train"],
        "target_val": target["val"], "target_test": target["test"],
    }


__all__ = [
    "CLASS_NAMES", "MultiModalFaultDataset", "UnlabeledMultiModalDataset",
    "compute_spectrogram", "normalize_signal", "load_csv_data",
    "load_cross_domain_data", "create_data_loaders", "create_domain_adapt_loaders",
]
