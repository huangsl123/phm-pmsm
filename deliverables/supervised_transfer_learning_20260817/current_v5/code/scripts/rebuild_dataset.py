#!/usr/bin/env python3
"""Rebuild the PMSM current dataset directly from the original TDMS ZIP files.

The old CSV files in this repository were produced by taking every 2000th raw
sample without an anti-aliasing filter.  They also lost recording boundaries
and encoded classes independently in each power domain.  This script keeps one
resampled signal file per original recording and writes a canonical manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import zipfile
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

try:
    from nptdms import TdmsFile
except ImportError as exc:  # pragma: no cover - exercised by CLI users
    raise SystemExit(
        "nptdms is required. Install it with `python -m pip install nptdms`."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASETS = PROJECT_ROOT / "datasets"
FORMAT_VERSION = 2

# Table 2 of the source data article.  Percent fault ratios differ with motor
# power because the stator resistance differs; the common physical condition is
# the bypass-resistance setting, not the percentage string in the filename.
# The healthy baseline has no injected bypass circuit, so its resistance is
# intentionally missing rather than 0 ohm (which would denote a hard short).
BYPASS_RESISTANCE_OHM = [None, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5]
EXPECTED_SEVERITY_PERCENT = {
    "intercoil": {
        "1.0kW": [0.0, 0.68, 0.81, 1.01, 1.34, 2.00, 3.93, 7.56],
        "1.5kW": [0.0, 4.79, 5.70, 7.02, 9.15, 13.12, 23.20, 37.66],
        "3.0kW": [0.0, 2.49, 2.98, 3.69, 4.86, 7.12, 13.30, 23.48],
    },
    "interturn": {
        "1.0kW": [0.0, 2.26, 2.70, 3.35, 4.41, 6.48, 12.17, 21.69],
        "1.5kW": [0.0, 1.57, 1.88, 2.34, 3.10, 4.57, 8.74, 16.08],
        "3.0kW": [0.0, 1.78, 2.13, 2.65, 3.50, 5.16, 9.81, 17.86],
    },
}

NAME_RE = re.compile(
    r"^(?P<power>\d+)W_(?P<severity>\d+(?:[_.]\d+)?)_"
    r"(?P<modality>current|vibration)_(?P<fault>intercoil|interturn|coil)\.tdms$",
    re.IGNORECASE,
)


def parse_recording_name(member_name: str) -> dict:
    """Parse a TDMS member name without relying on underscore positions."""
    match = NAME_RE.fullmatch(Path(member_name).name)
    if not match:
        raise ValueError(f"Unrecognised TDMS filename: {member_name}")
    values = match.groupdict()
    fault_type = values["fault"].lower()
    if fault_type == "coil":
        fault_type = "intercoil"
    severity_text = values["severity"].replace("_", ".")
    return {
        "power_w": int(values["power"]),
        "power_kW": f"{int(values['power']) / 1000:.1f}kW",
        "severity_percent": float(severity_text),
        "severity_code": severity_text.replace(".", "_"),
        "modality": values["modality"].lower(),
        "fault_type": fault_type,
    }


def canonical_labels(records: list[dict]) -> None:
    """Assign cross-domain IDs using fault type and bypass-resistance level."""
    by_power: dict[str, list[dict]] = {}
    for record in records:
        if record["modality"] == "current":
            by_power.setdefault(record["power_kW"], []).append(record)

    for power, power_records in by_power.items():
        for fault_type in ("intercoil", "interturn"):
            group = sorted(
                (r for r in power_records if r["fault_type"] == fault_type),
                key=lambda r: r["severity_percent"],
            )
            if len(group) != 8:
                raise ValueError(
                    f"{power}/{fault_type}: expected 8 recordings, found {len(group)}"
                )
            expected = EXPECTED_SEVERITY_PERCENT[fault_type].get(power)
            actual = [record["severity_percent"] for record in group]
            if expected is None or not np.allclose(actual, expected, atol=1e-6):
                raise ValueError(
                    f"{power}/{fault_type}: severities {actual} do not match "
                    f"the source article table {expected}"
                )
            for severity_rank, record in enumerate(group):
                record["severity_rank"] = severity_rank
                record["bypass_resistance_ohm"] = BYPASS_RESISTANCE_OHM[severity_rank]
                if severity_rank == 0:
                    record["class_id"] = 0
                    record["class_name"] = "healthy"
                    record["include_in_dataset"] = fault_type == "intercoil"
                    record["duplicate_of_class_id"] = (
                        None if fault_type == "intercoil" else 0
                    )
                else:
                    offset = 0 if fault_type == "intercoil" else 7
                    record["class_id"] = offset + severity_rank
                    record["class_name"] = f"{fault_type}_L{severity_rank}"
                    record["include_in_dataset"] = True
                    record["duplicate_of_class_id"] = None

        baselines = [
            record for record in power_records if record["severity_rank"] == 0
        ]
        if len(baselines) != 2 or len({r["zip_crc32"] for r in baselines}) != 1:
            raise ValueError(
                f"{power}: the two published healthy baselines are not exact duplicates"
            )

    label_sets = {
        power: sorted(
            (r["class_id"], r["class_name"])
            for r in values if r["include_in_dataset"]
        )
        for power, values in by_power.items()
    }
    expected = [(0, "healthy")] + [
        (i, f"intercoil_L{i}") for i in range(1, 8)
    ] + [(7 + i, f"interturn_L{i}") for i in range(1, 8)]
    for power, labels in label_sets.items():
        if labels != expected:
            raise ValueError(f"Canonical label validation failed for {power}: {labels}")


def find_signal_channels(tdms: TdmsFile, modality: str):
    channels = [channel for group in tdms.groups() for channel in group.channels()]
    expected = 3 if modality == "current" else 1
    candidates = [channel for channel in channels if len(channel) > 0]
    if len(candidates) != expected:
        raise ValueError(
            f"Expected {expected} {modality} channel(s), found {len(candidates)}: "
            f"{[c.path for c in candidates]}"
        )
    return candidates


def sampling_rate(channel) -> float:
    increment = channel.properties.get("wf_increment")
    if not increment or increment <= 0:
        raise ValueError(f"Missing wf_increment in {channel.path}")
    return 1.0 / float(increment)


def read_metadata(archive: zipfile.ZipFile, record: dict) -> dict:
    with archive.open(record["zip_member"]) as stream:
        with TdmsFile.open(stream) as tdms:
            channels = find_signal_channels(tdms, record["modality"])
            rates = [sampling_rate(channel) for channel in channels]
            lengths = [len(channel) for channel in channels]
            if max(rates) - min(rates) > 1e-6 or len(set(lengths)) != 1:
                raise ValueError(f"Channel mismatch in {record['zip_member']}")
            return {
                "source_sample_rate_hz": rates[0],
                "source_samples": lengths[0],
                "source_channels": [channel.name for channel in channels],
                "units": [channel.properties.get("unit_string", "") for channel in channels],
            }


def read_and_resample(
    archive: zipfile.ZipFile, record: dict, output_rate_hz: int
) -> tuple[np.ndarray, dict]:
    with archive.open(record["zip_member"]) as stream:
        tdms = TdmsFile.read(stream)
    channels = find_signal_channels(tdms, record["modality"])
    input_rate = sampling_rate(channels[0])
    lengths = {len(channel) for channel in channels}
    rates = {round(sampling_rate(channel), 6) for channel in channels}
    if len(lengths) != 1 or len(rates) != 1:
        raise ValueError(f"Channel mismatch in {record['zip_member']}")

    ratio = Fraction(output_rate_hz / input_rate).limit_denominator(10000)
    output_length = int(np.ceil(next(iter(lengths)) * ratio.numerator / ratio.denominator))
    signal = np.empty((output_length, len(channels)), dtype=np.float32)
    for column, channel in enumerate(channels):
        # resample_poly applies a zero-phase low-pass FIR before downsampling.
        resampled = resample_poly(
            np.asarray(channel[:], dtype=np.float64), ratio.numerator, ratio.denominator
        )
        signal[:, column] = resampled[:output_length].astype(np.float32, copy=False)

    metadata = {
        "source_sample_rate_hz": input_rate,
        "source_samples": next(iter(lengths)),
        "source_channels": [channel.name for channel in channels],
        "units": [channel.properties.get("unit_string", "") for channel in channels],
        "sample_rate_hz": output_rate_hz,
        "samples": len(signal),
        "anti_alias_filter": "scipy.signal.resample_poly default zero-phase FIR",
        "resample_up": ratio.numerator,
        "resample_down": ratio.denominator,
    }
    return signal, metadata


def discover(datasets_dir: Path, powers: set[str]) -> list[dict]:
    records: list[dict] = []
    for zip_path in sorted(datasets_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".tdms"):
                    continue
                parsed = parse_recording_name(info.filename)
                if parsed["power_kW"] not in powers:
                    continue
                parsed.update(
                    {
                        "source_zip": zip_path.name,
                        "zip_member": info.filename,
                        "zip_crc32": f"{info.CRC:08x}",
                        "source_uncompressed_bytes": info.file_size,
                    }
                )
                records.append(parsed)
    canonical_labels(records)
    return records


def class_map() -> list[dict]:
    return [{
        "class_id": 0,
        "class_name": "healthy",
        "fault_type": "healthy",
        "severity_rank": 0,
        "bypass_resistance_ohm": None,
        "severity_percent_by_power": {
            power: 0.0 for power in ("1.0kW", "1.5kW", "3.0kW")
        },
    }] + [
        {
            "class_id": offset + rank,
            "class_name": f"{fault_type}_L{rank}",
            "fault_type": fault_type,
            "severity_rank": rank,
            "bypass_resistance_ohm": BYPASS_RESISTANCE_OHM[rank],
            "severity_percent_by_power": {
                power: EXPECTED_SEVERITY_PERCENT[fault_type][power][rank]
                for power in ("1.0kW", "1.5kW", "3.0kW")
            },
        }
        for fault_type, offset in (("intercoil", 0), ("interturn", 7))
        for rank in range(1, 8)
    ]


def export_recording_csvs(
    datasets_dir: Path, output_dir: Path, records: list[dict]
) -> list[Path]:
    """Atomically replace legacy point CSVs with compact format-v2 indexes.

    Signal samples remain in typed NPY arrays.  Repeating roughly 60 million
    rows as decimal text would multiply storage and parse time without adding
    information; these CSVs provide the human-readable recording-level index.
    """
    fields = [
        "dataset_format", "power_kW", "class_id", "class_name", "fault_type",
        "severity_rank", "bypass_resistance_ohm", "severity_percent",
        "sample_rate_hz", "samples", "channels", "signal_file", "source_zip",
        "source_tdms", "source_crc32", "anti_alias_filter",
    ]
    written = []
    for power in sorted({record["power_kW"] for record in records}):
        power_records = sorted(
            (
                record for record in records
                if record["power_kW"] == power and record["include_in_dataset"]
            ),
            key=lambda record: record["class_id"],
        )
        destination = datasets_dir / f"dataset2_{power}.csv"
        temporary = destination.with_suffix(".tmp.csv")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            for record in power_records:
                signal_path = output_dir / record["processed_file"]
                writer.writerow(
                    {
                        "dataset_format": "pmsm_tdms_v2_recording_index",
                        "power_kW": power,
                        "class_id": record["class_id"],
                        "class_name": record["class_name"],
                        "fault_type": record["fault_type"],
                        "severity_rank": record["severity_rank"],
                        "bypass_resistance_ohm": record["bypass_resistance_ohm"],
                        "severity_percent": record["severity_percent"],
                        "sample_rate_hz": record["sample_rate_hz"],
                        "samples": record["samples"],
                        "channels": ";".join(record["source_channels"]),
                        "signal_file": str(signal_path.relative_to(datasets_dir)),
                        "source_zip": record["source_zip"],
                        "source_tdms": record["zip_member"],
                        "source_crc32": record["zip_crc32"],
                        "anti_alias_filter": record["anti_alias_filter"],
                    }
                )
        os.replace(temporary, destination)
        written.append(destination)
        print(f"Wrote {destination}", flush=True)
    return written


def rebuild(args: argparse.Namespace) -> Path:
    datasets_dir = args.datasets_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover(datasets_dir, set(args.powers))
    current_records = [
        record for record in records
        if record["modality"] == "current" and record["include_in_dataset"]
    ]
    if not current_records:
        raise ValueError("No current TDMS recordings found")

    archives: dict[str, zipfile.ZipFile] = {}
    try:
        for index, record in enumerate(current_records, 1):
            archive = archives.setdefault(
                record["source_zip"], zipfile.ZipFile(datasets_dir / record["source_zip"])
            )
            relative = Path(record["power_kW"]) / (
                f"class_{record['class_id']:02d}_{record['class_name']}.npy"
            )
            destination = output_dir / relative
            print(
                f"[{index:02d}/{len(current_records):02d}] {record['zip_member']} "
                f"-> class {record['class_id']:02d}",
                flush=True,
            )
            if args.metadata_only:
                metadata = read_metadata(archive, record)
            elif destination.exists() and not args.overwrite:
                mmap = np.load(destination, mmap_mode="r")
                metadata = read_metadata(archive, record)
                metadata.update(
                    {
                        "sample_rate_hz": args.output_rate,
                        "samples": int(mmap.shape[0]),
                        "anti_alias_filter": "scipy.signal.resample_poly default zero-phase FIR",
                        "resample_up": 1,
                        "resample_down": round(metadata["source_sample_rate_hz"] / args.output_rate),
                    }
                )
                record["processed_file"] = str(relative)
            else:
                signal, metadata = read_and_resample(archive, record, args.output_rate)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".tmp.npy")
                np.save(temporary, signal, allow_pickle=False)
                os.replace(temporary, destination)
                record["processed_file"] = str(relative)
            record.update(metadata)
    finally:
        for archive in archives.values():
            archive.close()

    manifest = {
        "format_version": FORMAT_VERSION,
        "description": "PMSM three-phase current recordings rebuilt from original TDMS files",
        "label_semantics": "15 distinct classes: one shared healthy class and seven bypass-resistance levels for each fault type",
        "excluded_source_records": [
            record for record in records
            if record["modality"] == "current" and not record["include_in_dataset"]
        ],
        "class_map": class_map(),
        "records": current_records,
    }
    manifest_path = output_dir / "manifest.json"
    temporary_manifest = output_dir / "manifest.tmp.json"
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, manifest_path)
    print(f"Wrote {manifest_path}", flush=True)
    if not args.metadata_only and not args.no_export_csv:
        export_recording_csvs(datasets_dir, output_dir, current_records)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DATASETS / "processed_v2"
    )
    parser.add_argument(
        "--powers", nargs="+", default=["1.0kW", "1.5kW", "3.0kW"]
    )
    parser.add_argument("--output-rate", type=int, default=10_000)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument(
        "--no-export-csv", action="store_true",
        help="Do not replace dataset2_*.csv with format-v2 recording indexes",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    try:
        rebuild(build_parser().parse_args())
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
