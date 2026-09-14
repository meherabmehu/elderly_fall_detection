# Kaggle-এ নিজে মডেল ট্রেন করার সম্পূর্ণ গাইড (ম্যানুয়াল পথ)

> কেন এই পথ? প্ল্যান §7 বলেছে: বড় ট্রেনিং (LOSO, LODO grids, ablations) Kaggle-এ হবে,
> আর deployment আর্টিফ্যাক্ট (model + normalization constants) এখানকার একমাত্র জিনিস যা
> ফার্মওয়্যারে যায়। এই গাইডে আপনি **শুধু মডেল ট্রেন + কোয়ান্টাইজ + আর্টিফ্যাক্ট ডাউনলোড**
> করবেন — বাকি সব (ফার্মওয়্যার, E1–E5, টেবিল) লোকালেই চলে।

---

## ধাপ ১ — `fall-sim-code.zip` ডাউনলোড করুন (এই রিপোজিটরির সাথে দেওয়া)

ভেতরে আছে: `config.py`, `src/`, `tools/`, `requirements.txt`, `kaggle_training.ipynb`।

## ধাপ ২ — Kaggle Notebook খুলুন

1. [kaggle.com](https://kaggle.com) → **Create → Notebook**।
2. নোটবুকের বাম প্যানেল → **File (উপরের ⊞ আইকন)** → **Upload** →
   `fall-sim-code.zip` ড্র্যাগ-ড্রপ করুন। (এক্সট্রাক্ট হয়ে `/kaggle/working/fall-sim/` হবে।)
3. বাম প্যানেল → **Add Input → Kaggle Dataset** → আপনার ৪টা ডেটাসেট যোগ করুন
   (পরের ধাপ দেখুন), তারপর নিচের ড্রপডাউনে প্রতিটার পাথ **কপি করে নোট করুন** (যেমন
   `/kaggle/input/sisfall-enhanced/...`)।

## ধাপ ৩ — ডেটাসেট আপলোড (৪টার যেকোনোটা; না দিলেও চলবে — ডেমো ডেটা)

প্রতিটা ডেটাসেট আলাদা **Kaggle Dataset** হিসেবে আপলোড করুন (নাম যেকোনো, নোটবুক নিজে চিনবে —
slug-এ `sisfall`/`kfall`/`fallalld`/`umafall` থাকলেই ম্যাপ হবে):

| Dataset | ভেতরে যা থাকবে (ক্যানোনিক্যাল লেআউট, `data/raw/<name>/`-এর মতোই) |
|---|---|
| **SisFall** (Enhanced) | `F01_SA06_R02.txt` (৯ কলাম raw ADC) + `F01_SA06_R02.labels.csv` (0/1/2 per-sample) + `trials_meta.csv` (file,subject,age,kind,onset_frame,impact_frame) + `imus.json` (Readme.txt-এর আসল conversion constants) |
| **KFall** | `SA06T01R01.csv` (time,ax,ay,az,gx,gy,gz) + `labels.csv` (file,subject,age,kind,onset_frame,impact_frame) |
| **FallAllD** | `Subject1/ADL_01/acc_0.csv, gyr_0.csv, acc_1.csv, gyr_1.csv` + `subjects.csv` (subject_dir,kind,subject,age,onset_frame,impact_frame) |
| **UMAFall** | `UMA_<subj>_<trial>.csv` (time, w_ax..w_gz, r_ax..r_gz) + `trials_meta.csv` |

> ⚠️ **গুরুত্বপূর্ণ:** নোটবুক শুধু **ম্যাপিং** করে — ডেটাসেটের ভেতরের ফাইল-লেআউট আসল
> ফরম্যাটেই থাকতে হবে। সবচেয়ে সহজ উপায়: লোকালে `data/raw/<name>/` ঠিক করে সাজিয়ে
> পুরো ফোল্ডার একটা zip করে Upload as Dataset করুন। (`README.md` §৪-এ লেআউট আছে।)
> কোনো ডেটাসেট ম্যাপ না হলে নোটবুক **বিল্ট-ইন ডেমো ডেটায়** ট্রেন করবে — পাইপলাইন
> প্রমাণিত হবে, কিন্তু সংখ্যাগুলো পেপারে যাবে না।

**আসল ডেটা না দিলে বড় সতর্কতা:** মুসসি-অ্যানোটেশন/ডেটাসেট সাইজের কারণে পুরো SisFall
~1 GB, KFall কয়েক GB — Kaggle-এর 20 GB ইনপুট কোটার মধ্যে রাখতে কম্প্রেস করে আপলোড করুন
(ডেটাসেট তৈরি করার সময় Kaggle নিজেই জিপ/আনজিপ করে নেয়)।

## ধাপ ৪ — Run All (উপর থেকে নিচে)

- **Cell 1–2:** কোড এক্সট্রাক্ট + ডেটা ম্যাপ।
- **Cell 3:** `tools/kaggle_train.py` চলে — ট্রেনিং, held-out evaluation, INT8
  quantisation + verification, তারপর **`fall_deploy_artifacts.zip`** বানায়।
- **Cell 4:** সামারি দেখায়।

ট্রেনিং কনফিগ (প্ল্যান §6-এর হুবহু): separable CNN (8,219 params), 3-class head
(non-fall/alert/fall), Adam 1e-3 + cosine decay, early stop on val **macro-F1**
(patience 8), class weighting, subject-wise split only।

## ধাপ ৫ — `fall_deploy_artifacts.zip` ডাউনলোড

বাম প্যানেল → **Output** ফোল্ডার → `/kaggle/working/fall-sim/fall_deploy_artifacts.zip`
→ ডাউনলোড বাটন। ভেতরে:

```
models/proposed_fp32.keras       ← FP32 মডেল (ডিবাগ/রিপ্রোডিউসিবিলিটি)
models/model.tflite              ← INT8 মডেল (ESP32-এ যাবে)
output/frozen_normalization.json ← ফ্রোজেন norm (ফার্মওয়্যারে যাবে)
output/kaggle_summary.json       ← এই রানের সব মেট্রিক
```

## ধাপ ৬ — লোকালে বসানো + ফার্মওয়্যার এক্সপোর্ট

```bash
# 1. zip-এর ফাইলগুলো এই জায়গায় রাখুন (পাথ হুবহু):
fall-sim/models/proposed_fp32.keras
fall-sim/models/model.tflite
fall-sim/output/frozen_normalization.json
fall-sim/output/kaggle_summary.json

# 2. (full-size synthetic mirror চাইলে) ফুল ডেটা জেনারেট:
cd fall-sim
python run_all.py --stage gen && python run_all.py --stage preprocess

# 3. নতুন ট্রেনিং ছাড়াই ফার্মওয়্যার ফাইল রি-এক্সপোর্ট + Table IV আপডেট:
python run_all.py --stage e4 --pretrained
#    → firmware/fall_detector/model_data.h + model_config.h (আপনার ট্রেন করা মডেল!)
#    → output/e4_summary.json (নতুন verified numbers)

# 4. ESP32-তে ফ্ল্যাশ (Arduino IDE → firmware/fall_detector) — README §৫-৬
```

## ধাপ ৭ — (ঐচ্ছিক) পুরো এক্সপেরিমেন্ট-সেট লোকালে

`python run_all.py --stage all` (E1–E5 + টেবিল + ফিগার) — এটি নিজে নিজে ট্রেন করবে;
যদি চান আপনার Kaggle-মডেল শুধু E4-তে (deployment) ব্যবহার হোক, আগের ধাপ অনুযায়ী
`e4 --pretrained` দিন, বাকি এক্সপেরিমেন্ট আলাদা চালান।

---

## FAQ

**প্রশ্ন: ট্রেনিং ছাড়া কি সিমুলেশন চলে?**
না — E1/E2/E3/E4 সবই মডেল ট্রেন করে (সিমুলেশন মানেই এক্সপেরিমেন্ট চালানো)।
তবে এই রিপোজিটরিতে **ট্রেন করা মডেল + ফুল-রান রেজাল্ট** দেওয়া আছে (`models/`,
`output/reference_results/`, `firmware/`) — সেগুলো দিয়ে ট্রেনিং ছাড়াই ব্যবহার/ফ্ল্যাশ করা
যায়। আপনার নিজের মডেল চাইলে উপরের পথে Kaggle-এ ট্রেন করে `e4 --pretrained` দিন।

**প্রশ্ন: ডেমো ডেটা (`data/synthetic`) কী?**
ছোট (৫+৩ সাবজেক্ট/ডেটাসেট) কিন্তু পুরো পাইপলাইন-ভিত্তিক মিরর — ৩ মিনিটে `--stage all`
চালিয়ে সবকিছু টেস্ট করা যায়। ফুল-সাইজ ডেটা: `python run_all.py --stage gen --force`
(ওভাররাইট) বা `rm -rf data/synthetic` দিয়ে আবার জেনারেট (কনফিগ-ডিফল্ট = ফুল)।

**প্রশ্ন: `--epochs 100` কেন?**
Kaggle-এর 12-ঘণ্টা সেশন-কোটার মধ্যে ৪০ এপক খুবই নিরাপদ; পেপার-কোয়ালিটি সংখ্যার জন্য
১০০ এপক + patience 15 ব্যবহার করুন (`--full`)।

**প্রশ্ন: INT8 delta 2 pp-এর বেশি হলে?**
`kaggle_summary.json`-এর `quant.delta_pp` দেখুন। বেশি হলে representative dataset
বাড়ান (`config.py`-র `QUANT["rep_windows"]` 200 → 500) — ইনপুট scale আরও ভালো হবে।
