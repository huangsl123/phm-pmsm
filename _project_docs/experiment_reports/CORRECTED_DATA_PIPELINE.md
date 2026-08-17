# Corrected PMSM data and evaluation pipeline

## Status

The original TDMS archives have been reparsed. The generated format-v2 dataset
contains 48 published source records, including three excluded duplicate
healthy records, and 45 distinct training records (15 classes for each power).
Generated `.npy` arrays are ignored
by Git because they are reproducible; `datasets/processed_v2/manifest.json`
records their provenance and processing metadata.

## Corrections

| Earlier failure | Correction |
|---|---|
| Every 2000th sample retained without filtering | Polyphase FIR anti-alias filtering, 100 kHz to 10 kHz |
| `1.01_current` parsed as a fault code | Regex parsing accepts both `1.01` and `1_01` |
| Label derived from `fault_code` only | Global `(fault_type, severity_rank)` mapping |
| Different label meaning in each power domain | Fixed classes 0--14 in all three domains |
| Identical healthy signal assigned to two labels | Merge both 0% files into one healthy class |
| Current and vibration rows concatenated | One array per original three-phase current recording |
| Windows crossed recording boundaries | Windows never leave their source recording |
| Random split after 87.5% overlapping windows | Contiguous train/test/validation 70/20/10 blocks split before windowing |
| No separation around split boundaries | One 1024-sample guard on both sides |
| Normalization over the three phases at each instant | Per-phase normalization over time |
| STFT stretched to 128 by 128 | Native 3 by 129 by 13 STFT, no image interpolation |
| Test accuracy used for configuration selection | Validation target accuracy and forgetting penalty |
| Paper described three-channel vibration | Paper now states three-phase current dual-view input |
| Invalid results presented as evidence | Values and derived figures withdrawn pending rerun |

## Canonical classes

- 0: shared healthy class; the duplicate 0% source file is excluded.
- 1--7: inter-coil bypass-resistance levels 6, 5, 4, 3, 2, 1, and 0.5 ohm.
- 8--14: inter-turn faults at the same seven resistance levels.

Percent fault ratios differ by motor capacity, so the percentage string is
metadata rather than the cross-domain class key. The shared bypass-resistance
setting is the physical correspondence key. This mapping follows the
eight-condition table for each fault type in the source data article.

## Required rerun

Old checkpoints, JSON metrics, confusion matrices, attention maps, and paper
figures are not compatible with format v2. EXP1, EXP2, and all EXP3 candidates
must be retrained with fixed seeds. Candidate selection must use source and
target validation blocks only. Test blocks may be evaluated once after the
configuration is locked; report every seed, mean, standard deviation, and
class-wise confusion matrices.
