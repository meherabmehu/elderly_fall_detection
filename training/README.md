# training/ — the ML pipeline (fdlib)

This is the **real-data** training and evaluation pipeline (GEN-2 in
`docs/KNOWLEDGE_MAP.md`): the shared `fdlib` library, seven Kaggle notebooks
(`nb00`–`nb07`), and the driver scripts. The resulting artifacts (model,
headers, results) are what the rest of the repository deploys and cites.

Provenance: reference repository
[`arifshekhk8/preimpact-fall-detection-tinyml`](https://github.com/arifshekhk8/preimpact-fall-detection-tinyml)
— see `ATTRIBUTION.md`. The full engineering context is in the reference
`PROJECT_PLAN.md` (included here), which records every dataset finding and
deviation from the original experimental plan.

## Layout

```text
src/fdlib/             the shared library — single source of truth
  config.py            every constant: 50 Hz, 100x6, stride 25, seed 1337
  preprocess.py        the frozen seven-step contract + axis canonicalisation
  datasets/            sisfall, kfall, fallalld, umafall -> Trial objects
  windowing.py         windowing and the two labelling rules (post-fall / pre-impact)
  cv.py                grouped / LOSO / LODO folds, class weights, inner splits
  models.py            proposed separable CNN (instance_norm in graph or pipeline) + comparators
  baselines.py         SMV threshold, SVM, Random Forest, window features
  adapt.py             instance norm, CORAL, DANN (target labels never used)
  metrics.py           classification + trial-level + lead time + false alarms/hour
  experiment.py        fold runner (per-fold CSV append + resume), evaluate()
  tflite_export.py     INT8 conversion, desktop verification, model.h export
kaggle/nb00..nb07      probe, preprocess, E1, E2, E3, E5, export, final
scripts/               sync_fdlib.py, run_kernel.py, build_tables.py,
                       debounce_analysis.py, trial_level_eval.py
Makefile               make sync|preprocess|e1|e2|e3|e5|export|all|tables|lint
requirements.txt       local driver env (Kaggle provides its own on the kernels)
PROJECT_PLAN.md        execution plan + all dataset deviations (read this)
SHOWCASE.md            demo-day run sheet
```

## Design guarantees (why results from this pipeline are trustworthy)

1. One definition of preprocessing: training, export and firmware all trace
   to `fdlib.preprocess`; corpora are stamped with `preprocess_signature()`
   (`2574e6104c95`).
2. Splits by subject or dataset, never by window.
3. Fold-level CSV append + resume; every run logged in
   `results/kaggle_reference/experiment_log.csv`.
4. Representative dataset for INT8 drawn from the training split only.
5. Desktop FP32↔INT8 verification is a gate before any firmware runs.

## Running

```bash
python -m venv ../.venv && ../.venv/bin/pip install -r requirements.txt
make sync          # publish fdlib as YOUR Kaggle dataset (edit slugs in scripts/)
make preprocess    # nb01 — corpus + sanity gates (must print gate_report OK)
make e1 e2 e3 e5 export
../.venv/bin/python scripts/run_kernel.py nb07_final
make tables        # rebuild paper tables from result CSVs
```

Kernel rules enforced by `scripts/run_kernel.py`: one kernel at a time, halt
on first failure, T4×2 requested and verified.

## One stale comment, deliberately not "fixed"

`src/fdlib/config.py` says the stride is "50 samples" in a comment; the value
is **25** (0.5 s at 50 Hz). The executed value is correct everywhere; the
comment is stale. The file is kept byte-identical with the library version
that produced the Kaggle results (`fdlib_version.json`) — fix it there, not
here, if a new training run is planned.
