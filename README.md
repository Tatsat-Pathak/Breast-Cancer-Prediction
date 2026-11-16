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

## Results (held-out test set, 114 samples)

| Model | Test accuracy |
|---|---|
| **Random Forest** | **98.2% (0.9825)** |
| SVC | 93.0% (0.9298) |
| XGBoost | CV score 95.8% (test accuracy not reproducible from the saved file — see note) |

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

The three `.pkl` files are `GridSearchCV` objects from a tuning session (scikit-learn 1.3.0):

| File | Best params | Best CV score |
|---|---|---|
| `Random Forest.pkl` | `{n_estimators: 2000}` | 0.9603 |
| `SVC.pkl` | `{C: 1, gamma: 0.0001}` | 0.9405 |
| `XGBoost.pkl` | `{n_estimators: 100}` | 0.9581 |

They need a matching scikit-learn / XGBoost version to load — pickles are version-fragile, which is why the notebook (retrain from `data.csv`) is the source of truth, not the `.pkl` files. The XGBoost file in particular cannot be scored with current XGBoost versions (`use_label_encoder` was removed), so only its CV score is reported here.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook "Cancer Prediction Project.ipynb"
```

## Files

- `Cancer Prediction Project.ipynb` — the analysis (markdown section headers added in cleanup; code unchanged)
- `data.csv` — Wisconsin Breast Cancer Diagnostic dataset
- `models/` — tuned models saved with pickle (see note above)
- `requirements.txt` — dependencies
