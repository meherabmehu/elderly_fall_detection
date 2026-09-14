# Datasets

## Where the datasets are

**SisFall and UMAFall raw files are vendored directly in this tree** — clone
the repo and the data is ready, no account or download step needed:

| Dataset | In-tree path | Contents | Licence / obligation |
|---|---|---|---|
| **SisFall** | `datasets/sisfall/SisFall_dataset/` | complete original release: 38 subjects, 4,505 trial `.txt` files + Readme + Supplementary | public research dataset — **cite Sucerquia et al. 2017** |
| **UMAFall** | `datasets/umafall/` | 746 trial `.csv` files (corrected version) | **CC BY 4.0** — cite Casilari et al. 2017 |
| **FallAllD** | *not vendored* — IEEE DataPort (free account): [link](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living) | | cite Saleh et al. 2020 |
| **KFall** | *not vendored* — its terms **forbid transferring the dataset to any third party**: [register at the official site](https://sites.google.com/view/kfalldataset/home) | | cite Yu et al. 2021 |

The same two archives (SisFall.zip, UMAFall_Dataset.zip) also sit in the
[`datasets-v1` release](https://github.com/meherabmehu/elderly_fall_detection/releases/tag/datasets-v1)
as checksummed single-file downloads (`datasets/SHA256SUMS.txt`) for anyone
who prefers an archive over a full clone. *Why not everything in the tree:*
KFall's licence forbids redistribution anywhere; FallAllD requires a personal
IEEE DataPort login.

## Verify / re-fetch

The vendored files are already laid out for the parsers, so normally nothing
to do. To re-fetch the two archives instead of cloning (e.g. for a fresh
machine without git):

```bash
python tools/download_datasets.py --extract   # pulls the release assets
# or download from the release page and
sha256sum -c datasets/SHA256SUMS.txt
```

## Expected local layouts (what the parsers look for)

```text
datasets/
├── sisfall/     SisFall_dataset/SAxx/Fxx_SAxx_Rxx.txt …   (vendored in-tree)
├── umafall/     UMAFall_Subject_XX_(ADL|Fall)_*.csv       (vendored in-tree)
├── fallalld/    FallAllD/ Subject*/(ADL|Fall)*/(acc|gyr)_*.csv or official .pkl   (you fetch)
└── kfall/       KFall Dataset/…/sensor_data/SAxxTxxRxx.csv + label_data/*.xlsx   (you register & fetch)
```

The parsers (`training/src/fdlib/datasets/`) accept the **original file
formats** — never convert files offline; all conversion happens inside
`fdlib.preprocess` so it is defined in exactly one place. Folder slugs must
contain `sisfall` / `kfall` / `fallalld` / `umafall` for Kaggle auto-mapping.

## Roles (settled by the nb00 probe)

- **KFall**: primary pre-impact source (video-grounded onset/impact frames);
  5,075 trials / 32 subjects in the mirror.
- **SisFall**: training corpus + E1 post-fall baseline. Note: the reference
  Kaggle runs used the 25-subject Kaggle mirror (`adityavvvn/sisfall`); the
  release asset here is the **complete 38-subject** official release. SisFall
  has no usable temporal labels (peak-acceleration proxy used).
- **FallAllD**: independent lab/hardware; LODO fold; placement ablation
  (waist/neck/wrist at 238 Hz).
- **UMAFall**: held-out stress test only (pocket smartphone at 200 Hz,
  accelerometer-only; waist SensorTag is 20 Hz and unusable at the 50 Hz
  working rate). The release asset is the corrected version used by the
  pipeline conventions; gyro is zero in the 200 Hz pocket files.

## Single-trial excerpt (already in the tree)

`wokwi/final_model/src/replay_data.h` — one 268-sample KFall trial
(S06T20R01) as a 50 Hz g/dps excerpt for the Wokwi replay demo. Review/demo
excerpt only; the full KFall dataset is not redistributed.

## Mandatory citations

Sucerquia et al. 2017 (SisFall); Musci et al. 2018/2020 (SisFall Enhanced
annotations, if used); Yu et al. 2021 (KFall); Saleh et al. 2020 (FallAllD);
Casilari et al. 2017 (UMAFall) — BibTeX in `paper_support/refs.bib`.
