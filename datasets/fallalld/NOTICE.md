# FallAllD — vendored copy

**Contents:** the public Kaggle mirror
[`sankalpsinghvishen/derived-fallalld-dataset`](https://www.kaggle.com/datasets/sankalpsinghvishen/derived-fallalld-dataset)
of the FallAllD dataset:

- `FallAllD.pkl` (441.23 MB) — the official pickled dataset, exactly as the
  pipeline's `fdlib/datasets/fallalld.py` expects (Waist device used, 238 Hz)
- `FallAllD_40SamplesPerSec_ActivityIdsFiltered.pkl` (230.59 MB) — 40 Hz
  derived variant
- `activity_info.pkl` — activity metadata (small, tracked directly)

14 subjects, waist/neck/wrist placements, ±8 g / ±2000 dps int16-scaled.

## Why the `.part-*` files, and how to rebuild

GitHub rejects any single file over **100 MB**, so the two large pickles are
stored split into ≤ 95 MiB parts. Every byte is in this tree; rebuilding the
originals takes one command after cloning:

```bash
cd datasets/fallalld
cat FallAllD.pkl.part-* > FallAllD.pkl
cat FallAllD_40SamplesPerSec_ActivityIdsFiltered.pkl.part-* > \
    FallAllD_40SamplesPerSec_ActivityIdsFiltered.pkl
```

Then verify integrity against `datasets/SHA256SUMS.txt`:

```bash
cd datasets
sha256sum -c SHA256SUMS.txt --ignore-missing   # FallAllD lines must say OK
```

Expected:

```text
5ada697ff106525c0a4ab4e88d6993f2ea732a59e95747c792a73fd3686788dd  FallAllD.pkl
ed897e2fd24350ceecdc1512662fd842baa3401fbf578f13e5b7773609866542  FallAllD_40SamplesPerSec_ActivityIdsFiltered.pkl
af7b04c9a08bd75a17ff241ecccc7497a09cebfe1561c97d24785a45081d5baf  activity_info.pkl
```

The parser only opens the joined `FallAllD.pkl` (and `activity_info.pkl` for
labels), so if you only browse the data you never need to rebuild anything.

**Citation obligation:** Saleh M, Abbas M, Le Bouquin Jeannès R. *FallAllD:
An Open Dataset of Human Falls and Activities of Daily Living for Classical
and Deep Learning Applications.* IEEE Sensors Journal 2020. BibTeX:
`paper_support/refs.bib`.

Canonical upstream: IEEE DataPort
(https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living)
— free account required there; this vendored copy exists so the repo is
self-contained for research reproducibility.
