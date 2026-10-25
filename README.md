# Forecasting next-day NEWS2 risk in the ICU with an FT-Transformer (MIMIC-IV)

Code for the study *"Forecasting Next-Day NEWS2 Risk in the ICU with a Feature-Tokenizer Transformer: A Temporally Separated, Patient-Level Evaluation on MIMIC-IV"* (Nabil Walid Rafi, Bangladesh University of Professionals).

From data recorded in the first 24 h of an ICU stay, the models predict (A) the official NEWS2 clinical-risk category over the next 24 h (24–48 h after ICU admission) and (B) High risk or death in that window. A tuned FT-Transformer with periodic numerical embeddings is compared with NEWS2 persistence, logistic regression, random forest, XGBoost and LightGBM on the same features and the same patient-level split.

## Data

This repository contains **no data**. The study uses [MIMIC-IV v3.1](https://doi.org/10.13026/kpb9-mt58), which is available only to credentialed PhysioNet users who have completed the required training and signed the data use agreement. Keep any copies, extracts and notebook outputs private.

Tables used: `icu/icustays`, `icu/chartevents`, `icu/procedureevents`, `icu/inputevents`, `icu/d_items`, `hosp/admissions`, `hosp/patients`, `hosp/labevents`.

## Notebooks (run in this order)

| Notebook | What it does | Runtime (Kaggle) |
| --- | --- | --- |
| `make-mimic-slim.ipynb` | One-time extraction of the required rows from the MIMIC-IV CSVs into small parquet files (vital signs, GCS and oxygen for the first 48 h; labs from −6 h; ventilation and vasopressor episodes). No values are changed. | ~15–90 min, CPU |
| `tune-hyperparameters.ipynb` | Builds the cohort, 61 features and patient-level split, then tunes the FT-Transformer, XGBoost, LightGBM and random forest with Optuna on validation patients only. Writes `best_params.json`. | ~4–5 h, GPU T4 |
| `mimic-ftt-early-warning.ipynb` | Reads the tuned settings and runs the full analysis: five-seed FT-Transformer ensemble and comparators, patient-clustered bootstrap CIs, High-or-death thresholds, permutation importance and SHAP, and sensitivity analyses. Writes all tables and figures to `results_archive.zip`. | ~2.5 h, GPU T4 |

On Kaggle: attach the MIMIC-IV files to the first notebook; attach its output to the second; attach the outputs of the first and second to the third. Use **Save Version → Save & Run All**.

## Design safeguards

- Predictors only from 0–24 h; outcome only from 24–48 h.
- Patient-level split (StratifiedGroupKFold, seed 42): no patient in both training and test sets.
- Imputation and scaling fitted on training data only.
- Tuning, early stopping, checkpoint selection and decision thresholds use validation patients only; the test set is used once.

## Main software

Python 3, PyTorch, scikit-learn, XGBoost, LightGBM, Optuna, SHAP, pandas, pyarrow.

## Citation

If you use this code, please cite the paper and MIMIC-IV:

- Johnson A, et al. MIMIC-IV (version 3.1). PhysioNet; 2024. https://doi.org/10.13026/kpb9-mt58
- Johnson AEW, et al. MIMIC-IV, a freely accessible electronic health record dataset. *Scientific Data*. 2023;10:1.
