https://colab.research.google.com/drive/13jopU0hgnCWTXHgo4-tsd-KYkpAOGHwx?usp=sharing

# YZTA Datathon 2025–2026

Official team solution for predicting **`bilissel_performans_skoru`**.

## Run in Google Colab

Open the notebook in Colab using the link on the **first line** of this README, upload `train.csv` and `test_x.csv`, and run all cells.

## Run as a Python script

The file **`FINAL_TEAM_SOLUTION.py`** contains the same pipeline end-to-end (EDA plots + training + `TEAM_VOLTRAN_FRANKENSTEIN_OPTUNA.csv`).

- **Colab / notebook:** the original `%pip install …` cell is preserved in the Colab workflow.
- **Script:** the `%pip` line is replaced with a `subprocess.check_call` to `python -m pip install …` using the **same packages** (`lightgbm`, `catboost`, `pandas`, `scikit-learn`) so the file runs outside IPython.

Place `train.csv` and `test_x.csv` in the working directory (or adjust `DATA_DIR` at the top of the EDA section for the exploratory plots). The training section reads `train.csv` / `test_x.csv` from the current directory, matching the notebook.

If your environment is missing optional libraries used in the script (for example **XGBoost**, **matplotlib**, or **seaborn**), install them before running:

```bash
pip install xgboost matplotlib seaborn
```

## Repository

https://github.com/gulce-celik/YZTA_DATATHON_2025_2026
