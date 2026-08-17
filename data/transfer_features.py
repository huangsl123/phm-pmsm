"""Dimensionless three-phase features for single-model transfer learning."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .data_processor_v2 import CLASS_NAMES, _choose_starts, _split_regions


def dimensionless_features(window: np.ndarray, sample_rate_hz: int, bins: int = 96) -> np.ndarray:
    """Remove absolute operating amplitude while retaining phase and spectral shape."""
    x = np.asarray(window, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(x * x)) + 1e-12
    x = x / scale
    rms = np.sqrt(np.mean(x * x, axis=0) + 1e-18)
    std = np.maximum(x.std(axis=0), 1e-12)
    z = x / std
    stats = np.column_stack([
        rms, np.mean(np.abs(x), axis=0), np.max(np.abs(x), axis=0),
        np.ptp(x, axis=0), np.max(np.abs(x), axis=0) / rms,
        np.mean(z ** 3, axis=0), np.mean(z ** 4, axis=0) - 3.0,
    ]).ravel()
    corr = np.corrcoef(x, rowvar=False)[np.triu_indices(3, 1)]
    phase = np.concatenate([
        corr,
        [rms.std() / max(rms.mean(), 1e-12),
         np.sqrt(np.mean(np.square(x.sum(axis=1))))],
    ])

    fft = np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)
    power = np.square(np.abs(fft)) + 1e-20
    frequencies = np.fft.rfftfreq(len(x), 1.0 / sample_rate_hz)
    edges = np.linspace(0.0, 2500.0, bins + 1)
    pooled = np.empty((bins, 3), dtype=np.float64)
    for index in range(bins):
        mask = (frequencies > edges[index]) & (frequencies <= edges[index + 1])
        pooled[index] = power[mask].mean(axis=0) if np.any(mask) else 1e-20
    # One global denominator preserves phase imbalance but removes absolute power.
    spectrum = np.log10(pooled / max(float(pooled.sum()), 1e-20) + 1e-12).T.ravel()

    fundamental_mask = (frequencies >= 20.0) & (frequencies <= 100.0)
    indices = np.flatnonzero(fundamental_mask)
    fundamental = indices[np.argmax(power[fundamental_mask].sum(axis=1))]
    harmonics = []
    for order in range(1, 21):
        index = min(int(round(fundamental * order)), len(fft) - 1)
        ratio = np.abs(fft[index]) / np.maximum(np.abs(fft[fundamental]), 1e-12)
        harmonics.extend(np.log10(ratio + 1e-12))
    phasor = fft[fundamental]
    a = np.exp(2j * np.pi / 3)
    seq1 = abs((phasor[0] + a * phasor[1] + a ** 2 * phasor[2]) / 3)
    seq2 = abs((phasor[0] + a ** 2 * phasor[1] + a * phasor[2]) / 3)
    positive, negative = max(seq1, seq2), min(seq1, seq2)
    zero = abs(phasor.sum() / 3)
    sequence = [negative / max(positive, 1e-12), zero / max(positive, 1e-12)]
    value = np.concatenate([stats, phase, spectrum, harmonics, sequence])
    return np.nan_to_num(value, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)


def load_transfer_features(index_csv: str | Path, window_size: int, max_windows: int = 256,
                           bins: int = 96, stride: int = 512) -> dict:
    """Read original-record NPY files with guarded 70/20/10 time blocks."""
    index_path = Path(index_csv).resolve()
    with index_path.open(newline="", encoding="utf-8") as stream:
        records = sorted(csv.DictReader(stream), key=lambda row: int(row["class_id"]))
    if [int(row["class_id"]) for row in records] != list(range(15)):
        raise ValueError("Expected canonical class IDs 0..14")
    limits = {"train": round(max_windows * .7), "val": round(max_windows * .1)}
    limits["test"] = max_windows - limits["train"] - limits["val"]
    values = {split: {"X": [], "y": [], "starts": []} for split in limits}
    sample_rate = int(records[0]["sample_rate_hz"])
    for record in records:
        raw = np.load(index_path.parent / record["signal_file"], mmap_mode="r")
        for split, (begin, end) in _split_regions(len(raw), window_size).items():
            starts = _choose_starts(end - begin, window_size, stride, limits[split])
            for local in starts:
                start = begin + int(local)
                values[split]["X"].append(
                    dimensionless_features(raw[start:start + window_size], sample_rate, bins)
                )
                values[split]["y"].append(int(record["class_id"]))
                values[split]["starts"].append(start)
    result = {"class_names": CLASS_NAMES, "window_size": window_size}
    for split in ("train", "val", "test"):
        result[f"X_{split}"] = np.stack(values[split]["X"])
        result[f"y_{split}"] = np.asarray(values[split]["y"], dtype=np.int64)
        result[f"starts_{split}"] = np.asarray(values[split]["starts"], dtype=np.int64)
    return result


__all__ = ["dimensionless_features", "load_transfer_features"]
