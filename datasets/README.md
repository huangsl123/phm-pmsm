# PMSM dataset processing

The three historical `dataset2_*.csv` files are retained only as legacy
artifacts. They must not be used for new experiments: they discard 1,999 of
every 2,000 samples without an anti-alias filter, lose original recording
boundaries, and do not preserve common class semantics across motor powers.

Rebuild the current-signal dataset directly from the original ZIP files:

```bash
python scripts/rebuild_dataset.py
```

This creates `processed_v2/manifest.json` plus one 10 kHz, anti-aliased
three-phase current `.npy` file per original TDMS recording. Class IDs are
fixed across all powers:

The command also replaces `dataset2_*.csv` with compact, human-readable
recording indexes. Each CSV has 15 rows and points to the corresponding typed
signal arrays. The samples themselves are not duplicated as decimal CSV text:
at 10 kHz that would require roughly 60 million rows across the three domains
and several gigabytes while being slower and less precise than NPY.

- `0`: the single shared healthy class.
- `1..7`: seven inter-coil bypass-resistance levels.
- `8..14`: seven inter-turn bypass-resistance levels.

The two published 0% TDMS files in each power domain are byte-for-byte
identical. The inter-turn copy is retained in the JSON manifest as excluded
source provenance, but it is not repeated in the CSV or used for training.

The fault percentages are not compared directly between motor powers. The
healthy baseline has no injected bypass circuit, so its bypass-resistance field
is empty in CSV and `null` in JSON. Fault labels are aligned by the common
physical bypass-resistance settings `6, 5, 4, 3, 2, 1, 0.5` ohm; the manifest
retains each power-specific fault percentage as metadata.

The loader in `data/data_processor_v2.py` splits each recording into contiguous
training/test/validation blocks in the ratio 70/20/10 before windowing and
leaves a full-window guard at each boundary. It therefore prevents overlapping
windows from crossing between the three sets.
