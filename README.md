# Formative 2 — Multimodal Data Preprocessing
User Identity & Product Recommendation System: face recognition → voiceprint verification → product recommendation, with ACCESS DENIED pathways on any failed step.

## ⚠️ Placeholder data — replace before submission
This repo currently runs on **synthetic stand-ins** so the full pipeline is testable end-to-end. Swap in real data (the code auto-detects real files and skips generation):

| Replace with | Location | Naming |
|---|---|---|
|  ~~Course CSVs~~ ✅ **real course datasets already included** (sourced from the shared assignment data) | `data/raw/` | keep the same filenames & columns (adjust `data_merge.py` join keys if the real schema differs) |
| Your face photos (neutral, smiling, surprised) | `data/images/raw/` | `member1_neutral.jpg`, `member1_smiling.jpg`, `member1_surprised.jpg` (add `member2_…` etc. for teammates) |
| Your voice recordings | `data/audio/raw/` | `member1_yes_approve.wav`, `member1_confirm_transaction.wav` |

Keep at least one non-member identity (impostor) in images/audio so the auth models have negative examples.

## Repo structure
```
├── app.py                          # CLI simulation (Task 6)
├── Formative2_Multimodal_Pipeline.ipynb  # executed notebook, full narrative
├── scripts/
│   ├── generate_data.py            # synthetic CSV stand-ins (delete once real data added)
│   ├── data_merge.py               # Task 1: clean, merge, EDA, feature engineering
│   ├── image_pipeline.py           # Task 2: collect, augment, extract → image_features.csv
│   ├── audio_pipeline.py           # Task 3: collect, visualize, augment → audio_features.csv
│   ├── train_models.py             # Task 4: 3 models + accuracy/F1/loss → models/
│   └── build_notebook.py           # regenerates the notebook
├── data/
│   ├── raw/                        # source CSVs
│   ├── images/{raw,augmented}/
│   ├── audio/{raw,augmented}/
│   └── processed/                  # merged_dataset.csv, image_features.csv, audio_features.csv
├── models/                         # face/voice/product .joblib + metrics.json
└── outputs/plots/                  # EDA, image grid, waveforms/spectrograms
```

## How to run
```bash
pip install -r requirements.txt

python scripts/generate_data.py    # skip once real CSVs are in data/raw/
python scripts/data_merge.py
python scripts/image_pipeline.py
python scripts/audio_pipeline.py
python scripts/train_models.py

# Authorized full transaction
python app.py --image data/images/raw/member1_smiling.jpg \
              --audio data/audio/raw/member1_yes_approve.wav --customer A5003

# Unauthorized attempt (denied at face step)
python app.py --image data/images/raw/impostor2_neutral.jpg \
              --audio data/audio/raw/member1_yes_approve.wav
```

## Current metrics (real tabular data; synthetic face/voice placeholders)
| Model | Accuracy | F1 (weighted) | Log loss |
|---|---|---|---|
| Face recognition (RandomForest) | 1.00 | 1.00 | 0.06 |
| Voiceprint verification (LogReg) | 0.88 | 0.87 | 0.20 |
| Product recommendation (RandomForest) | 0.20 | 0.19 | 1.62 — does not beat majority baseline (0.25); real dataset carries no product signal, documented honestly in notebook |

## Multimodal decision logic
`P(face authorized) ≥ 0.60` **AND** `P(voice authorized) ≥ 0.60` → product model runs; any failure → ACCESS DENIED.

## Team contributions
- Member 1 — *(fill in per member once the group divides the work)*
