# Datasets

Raw datasets are **intentionally not in this repository**. Sizes range from
hundreds of MB to several GB, and KFall's terms require that its raw files are
never redistributed. Everything the pipeline needs to know about layouts lives
in `training/src/fdlib/datasets/` plus the acquisition notes below.

## Sources used by the reference Kaggle runs

| Dataset | Kaggle source (as used) | Canonical upstream |
|---|---|---|
| SisFall | `adityavvvn/sisfall` | [SisFall (Sucerquia 2017)](https://www.kaggle.com/datasets/nvnikhil0001/sisfall-enhanced/data) mirror |
| KFall | `usmanabbasi2002/kfall-dataset` | [KFall site](https://sites.google.com/view/kfalldataset/home) |
| FallAllD | `sankalpsinghvishen/derived-fallalld-dataset` | [IEEE DataPort](https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living) |
| UMAFall | `thanushanth/umafall` | [Figshare](https://figshare.com/articles/dataset/UMA_ADL_FALL_Dataset_zip/4214283) |

Follow each source's access, license and citation rules; record your access
dates in `training/PROJECT_PLAN.md`-style when re-running.

## Expected local layouts (what the parsers look for)

```text
SisFall/                             *.txt (9-column raw ADC), Readme.txt
KFall/KFall Dataset/.../sensor_data/ SAxxTxxRxx.csv (+ label_data/*.xlsx)
FallAllD/                         FallAllD.pkl (official) or Subject*/(ADL|Fall)*/(acc|gyr)_*.csv
UMAFall/                          UMAFall_Subject_XX_(ADL|Fall)_*.csv
```

Folder names slugs must contain `sisfall` / `kfall` / `fallalld` / `umafall`
for auto-mapping on Kaggle. The parsers otherwise accept the original file
formats — do not convert offline; all conversion happens inside
`fdlib.preprocess` so it is defined in exactly one place.

## Roles (settled by the nb00 probe)

- **KFall**: primary pre-impact source (video-grounded onset/impact frames);
  5,075 trials / 32 subjects in the mirror.
- **SisFall**: training corpus + E1 post-fall baseline; mirror has 25 of 38
  subjects; no usable temporal labels (peak-acceleration proxy used).
- **FallAllD**: independent lab/hardware; LODO fold; placement ablation
  (waist/neck/wrist at 238 Hz).
- **UMAFall**: held-out stress test only (pocket smartphone at 200 Hz,
  accelerometer only; waist SensorTag is 20 Hz and unusable at 50 Hz).

## Single-trial excerpt

`wokwi/final_model/src/replay_data.h` contains one 268-sample trial
(KFall S06T20R01) as a 50 Hz g/dps excerpt, kept for the Wokwi replay
integration demo. It is a review/demo excerpt; the full dataset is not
redistributed.

## Mandatory citations

Sucerquia et al. 2017; Musci et al. 2018/2020; Yu et al. 2021; Saleh et al.
2020; Casilari et al. 2017 — BibTeX in `paper_support/refs.bib`.
