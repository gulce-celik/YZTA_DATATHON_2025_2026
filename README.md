https://colab.research.google.com/drive/13jopU0hgnCWTXHgo4-tsd-KYkpAOGHwx?usp=sharing

# DATATHON Team 106

**YZTA Datathon 2025–2026** — regression solution for predicting cognitive performance (**`bilissel_performans_skoru`**) from sleep, lifestyle, and demographic features.

This repository ships our **final submission pipeline**: exploratory analysis, feature engineering, multi-model cross-validation, stacking, and CSV export. The same logic is available as a **Google Colab notebook** (link above) and as a **standalone Python script** in this repo.

## Team

| Name | Role |
|------|------|
| **Gülce Çelik** | Team member |
| **Gamze Akemoğlu** | Team member |
| **Mehmet Vural** | Team member |
| **Anıl Dinç** | Team member |

---

## Quick start — Google Colab

1. Open the **[Colab notebook](https://colab.research.google.com/drive/13jopU0hgnCWTXHgo4-tsd-KYkpAOGHwx?usp=sharing)** (same link as the first line of this README).
2. Upload **`train.csv`** and **`test_x.csv`** (competition files) into the Colab environment (e.g. `/content` if you use the default paths in the EDA section).
3. Run all cells **from top to bottom**. The first part produces EDA figures; the second part trains models and writes the submission file.

Colab already includes many scientific libraries; the notebook installs additional ones with `%pip` where needed.

---

## Quick start — local Python

Use **`FINAL_TEAM_SOLUTION.py`** for a single file that runs the **full** workflow (EDA + training + submission).

- **Paths:** The EDA block uses `DATA_DIR = Path("/content")` for plots; on your machine, either change that to the folder where your CSVs live or copy the files under `/content` if you mirror Colab.
- **Training block** reads `train.csv` and `test_x.csv` from the **current working directory** (same convention as the notebook’s modeling section).
- **`%pip` vs script:** In Colab, dependencies are installed with `%pip install …`. In the `.py` file, that line is replaced with an equivalent `subprocess.check_call([sys.executable, "-m", "pip", "install", …])` so the same packages install when you run `python FINAL_TEAM_SOLUTION.py`.

Install extra libraries if your environment does not already have them (the modeling stack uses **XGBoost**; EDA uses **matplotlib** and **seaborn**):

```bash
pip install xgboost matplotlib seaborn lightgbm catboost pandas scikit-learn
```

---

## What this solution does (overview)

1. **EDA** — Missingness, target distribution, Pearson / Spearman correlations, heatmaps, scatter plots, and categorical cardinality to understand the data before modeling.
2. **Cleaning** — Lowercasing and light normalization of categorical text, country aliases, and conservative clipping on selected numeric fields.
3. **Imputation** — Hybrid approach: **IterativeImputer** (MICE-style) on numeric columns with missing values, **constant** fill for categoricals.
4. **Features** — Hand-crafted sleep and stress interactions (e.g. sleep quality score, fragmentation, weekend shift, caffeine / screen / stress terms).
5. **Encoding** — Label encoding for tree models, **out-of-fold target encoding** for selected high-cardinality columns, with scaling where applied.
6. **Pseudo-labeling** — A preliminary **CatBoost** model generates pseudo-targets on the test set to augment training folds.
7. **Stacking** — **10-fold** out-of-fold predictions from **CatBoost**, **XGBoost**, **LightGBM**, and **KNN**; a **Ridge** meta-model blends them. Final predictions are mean-calibrated and clipped using train quantile bounds.

Output file: **`TEAM_VOLTRAN_FRANKENSTEIN_OPTUNA.csv`** (submission-ready `id` + target column).

---

## Repository layout

| File | Description |
|------|-------------|
| `FINAL_TEAM_SOLUTION.py` | End-to-end script: EDA + training + CSV export |
| `README.md` | This documentation |

Competition datasets are **not** bundled here; obtain `train.csv` and `test_x.csv` from the official datathon source.

---

## Links

- **GitHub:** [https://github.com/gulce-celik/YZTA_DATATHON_2025_2026](https://github.com/gulce-celik/YZTA_DATATHON_2025_2026)
- **Colab:** [Open notebook](https://colab.research.google.com/drive/13jopU0hgnCWTXHgo4-tsd-KYkpAOGHwx?usp=sharing)

---

*DATATHON Team 106 — YZTA 2025–2026.*
