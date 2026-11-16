# Cancer Prediction

Binary classification: predict whether a breast tumour is **benign or malignant** from 30 measurements of cell nuclei.

Built during the Machine Learning Internship at **Acmegrade (Feb–Apr 2024)**.

> Research / learning project on a public dataset. **Not a medical device and not for clinical use.**

## Dataset

**Wisconsin Breast Cancer Diagnostic dataset** (`data.csv`, included in this repo)

- 569 samples: 357 benign, 212 malignant
- 30 numeric features (`radius_mean` … `fractal_dimension_worst`)
- Target: `diagnosis` (B → 0, M → 1); the `id` column is not a feature
- No missing values

## What the notebook does

1. Loads and explores the data (class counts, dtypes, missing values)
2. Correlation heatmap + per-feature distribution plots by diagnosis
3. 80/20 train/test split — 455 train / 114 test (`test_size=0.2`, `np.random.seed(10)`)
4. Trains a **Random Forest with 2,000 trees** and evaluates it
5. Scales the features with `StandardScaler` (fit on train only), reports 5-fold cross-validation with `cross_val_score`, tunes all three models with `GridSearchCV`, and saves the tuned models to `models/` with `joblib`

## Results (held-out test set, 114 samples)

| Model | Test accuracy |
|---|---|
| **Random Forest** | **98.2% (0.9825)** |
| SVC | 96.5% (0.9649) |
| XGBoost | 96.5% (0.9649) |

Random Forest classification report (test set):

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 — Benign | 1.00 | 0.97 | 0.99 | 75 |
| 1 — Malignant | 0.95 | 1.00 | 0.97 | 39 |

Confusion matrix: `[[73, 2], [0, 39]]` — all 39 malignant cases in the test set were caught; 2 benign cases were flagged as malignant.

The 98.2% was re-verified by re-running the notebook's exact pipeline (seed 10, same split) in a clean environment.

## A number NOT to quote

The last cell of the notebook scores the model on the **full dataset, including the data it trained on**, giving 99.6%. That is not a real result — it is expected to be high. The honest number is the **98.2% held-out test accuracy** above.

## Saved models (`models/`)

The three `.pkl` files are `GridSearchCV` objects saved by the notebook itself (`joblib.dump`, scikit-learn 1.3.2 / XGBoost 1.7.6). Random Forest keeps its 2,000 trees and tunes depth instead; SVC and XGBoost are tuned on the scaled data:

| File | Best params | Best CV score |
|---|---|---|
| `Random Forest.pkl` | `{max_depth: None, min_samples_split: 2}` (2,000 trees) | 0.9582 |
| `SVC.pkl` | `{C: 10, gamma: 0.01}` | 0.9802 |
| `XGBoost.pkl` | `{n_estimators: 200}` | 0.9604 |

They need a matching scikit-learn / XGBoost version to load — pickles are version-fragile, which is why the notebook (retrain from `data.csv`) is the source of truth, not the `.pkl` files. The old files (tuned in a session that was never in the notebook) are replaced by these.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook "Cancer Prediction Project.ipynb"
```

## Files

- `Cancer Prediction Project.ipynb` — the analysis (markdown section headers added in cleanup; modelling section added Nov 2026: `LabelEncoder`, `StandardScaler`, `cross_val_score`, `GridSearchCV` tuning and `joblib` saving for all three models)
- `data.csv` — Wisconsin Breast Cancer Diagnostic dataset
- `models/` — tuned models saved with pickle (see note above)
- `requirements.txt` — dependencies
