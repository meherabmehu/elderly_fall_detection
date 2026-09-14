# Datasets

## Where the datasets are

Two datasets are attached to this repository's GitHub release
[`datasets-v1`](https://github.com/meherabmehu/elderly_fall_detection/releases/tag/datasets-v1),
verifiable against `SHA256SUMS.txt` and fetchable with
[`tools/download_datasets.py`](../tools/download_datasets.py):

| Dataset | Where | License / obligation |
|---|---|---|
| **SisFall** | Release asset `SisFall.zip` (213.6 MB; full original 38 subjects, 4,585 files) | public research dataset — **cite Sucerquia et al. 2017** |
| **UMAFall** | Release asset `UMAFall_Dataset.zip` (78.5 MB; 746 CSV trials, corrected version) | **CC BY 4.0** — cite Casilari et al. 2017 |
| **FallAllD** | IEEE DataPort (free account required) — [link](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living) | cite Saleh et al. 2020 |
| **KFall** | **Never redistributable** — register at the [official site](https://sites.google.com/view/kfalldataset/home); its terms forbid transferring the dataset to third parties | cite Yu et al. 2021 |

Why not committed into the git tree: KFall's license forbids it; SisFall's
archive exceeds GitHub's 100 MB file limit; raw data in git history makes
every future clone heavy forever. Release assets (2 GB each) are the right
place, checksummed here in the repo.

## Fetch + verify + extract

```bash
python tools/download_datasets.py --extract
# or manually: download from the release page, then
sha256sum -c datasets/SHA256SUMS.txt
```

## Expected local layouts (what the parsers look for)

```text
datasets/
├── sisfall/     SisFall_dataset/SAxx/Fxx_SAxx_Rxx.txt …   (from release asset)
├── umafall/     UMAFall_Subject_XX_(ADL|Fall)_*.csv       (from release asset)
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
