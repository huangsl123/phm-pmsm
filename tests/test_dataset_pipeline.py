import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_processor_v2 import _split_regions, compute_spectrogram, normalize_signal
from scripts.rebuild_dataset import canonical_labels, parse_recording_name


class FilenameParsingTests(unittest.TestCase):
    def test_dot_severity_is_not_split_into_a_fake_field(self):
        parsed = parse_recording_name("1000W_1.01_current_intercoil.tdms")
        self.assertEqual(parsed["severity_code"], "1_01")
        self.assertEqual(parsed["fault_type"], "intercoil")
        self.assertEqual(parsed["modality"], "current")

    def test_coil_alias_is_canonicalized(self):
        parsed = parse_recording_name("3000W_2_49_vibration_coil.tdms")
        self.assertEqual(parsed["fault_type"], "intercoil")

    def test_labels_are_type_and_rank_based(self):
        records = []
        for power, coil, turn in (
            ("1.0kW", [0, .68, .81, 1.01, 1.34, 2, 3.93, 7.56],
             [0, 2.26, 2.70, 3.35, 4.41, 6.48, 12.17, 21.69]),
            ("1.5kW", [0, 4.79, 5.70, 7.02, 9.15, 13.12, 23.20, 37.66],
             [0, 1.57, 1.88, 2.34, 3.10, 4.57, 8.74, 16.08]),
            ("3.0kW", [0, 2.49, 2.98, 3.69, 4.86, 7.12, 13.30, 23.48],
             [0, 1.78, 2.13, 2.65, 3.50, 5.16, 9.81, 17.86]),
        ):
            for fault_type, severities in (("intercoil", coil), ("interturn", turn)):
                records.extend(
                    {"power_kW": power, "modality": "current",
                     "fault_type": fault_type, "severity_percent": severity,
                     "zip_crc32": (
                         f"{power}-healthy" if severity == 0
                         else f"{power}-{fault_type}-{severity}"
                     )}
                    for severity in severities
                )
        canonical_labels(records)
        for power in ("1.0kW", "1.5kW", "3.0kW"):
            labels = sorted(
                r["class_id"] for r in records
                if r["power_kW"] == power and r["include_in_dataset"]
            )
            self.assertEqual(labels, list(range(15)))
            excluded = [
                r for r in records
                if r["power_kW"] == power and not r["include_in_dataset"]
            ]
            self.assertEqual(len(excluded), 1)
            self.assertEqual(excluded[0]["duplicate_of_class_id"], 0)
        coil_l1 = [
            r for r in records
            if r["fault_type"] == "intercoil" and r["severity_rank"] == 1
        ]
        self.assertEqual({r["severity_percent"] for r in coil_l1}, {0.68, 4.79, 2.49})
        self.assertEqual({r["bypass_resistance_ohm"] for r in coil_l1}, {6.0})
        healthy = [r for r in records if r["severity_rank"] == 0]
        self.assertTrue(healthy)
        self.assertTrue(all(r["bypass_resistance_ohm"] is None for r in healthy))


class SignalProcessingTests(unittest.TestCase):
    def test_normalization_operates_over_time_per_channel(self):
        x = np.array([[0, 100, 10], [1, 100, 20], [2, 100, 30]], dtype=np.float32)
        normalized = normalize_signal(x, "zscore")
        np.testing.assert_allclose(normalized.mean(axis=0), 0, atol=1e-6)
        np.testing.assert_allclose(normalized[:, 1], 0, atol=1e-6)

    def test_stft_keeps_native_resolution(self):
        x = np.random.default_rng(0).normal(size=(1024, 3)).astype(np.float32)
        spec = compute_spectrogram(x, n_fft=256, hop_length=64)
        self.assertEqual(spec.shape, (3, 129, 13))
        self.assertTrue(np.isfinite(spec).all())
        self.assertGreaterEqual(float(spec.min()), 0.0)
        self.assertLessEqual(float(spec.max()), 1.0)

    def test_time_splits_have_guard_bands(self):
        regions = _split_regions(100_000, guard=1024)
        self.assertGreaterEqual(regions["test"][0] - regions["train"][1], 2048)
        self.assertGreaterEqual(regions["val"][0] - regions["test"][1], 2048)
        self.assertEqual(regions["test"][1], 90_000 - 1024)


if __name__ == "__main__":
    unittest.main()
