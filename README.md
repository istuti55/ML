# Titanic Survival Prediction — Logistic Regression

A beginner-friendly end-to-end machine learning workflow that predicts passenger survival on the Titanic using **Logistic Regression**, built with `pandas`, `seaborn`, and `scikit-learn`.

**Suggested notebook filename:** `titanic_survival_logistic_regression.ipynb`

## Overview

This notebook walks through a complete classification pipeline on the classic Titanic dataset (loaded via `seaborn.load_dataset('titanic')`):

1. **Data Loading & Exploration** — inspect shape, dtypes, and summary statistics (`.info()`, `.describe()`).
2. **Data Cleaning**
   - Fill missing `age` values with the median.
   - Fill missing `embarked` values with the mode.
   - Drop redundant / high-missing / leakage-prone columns: `deck`, `embark_town`, `class`, `alive`, `who`.
3. **Encoding**
   - Map `sex` to binary (0 = male, 1 = female).
   - Convert `adult_male` and `alone` booleans to integers.
   - One-hot encode `embarked` (drop-first to avoid the dummy trap).
4. **Outlier Handling** — remove `fare` outliers using the IQR method.
5. **Feature Scaling** — standardize `age` and `fare` with `StandardScaler`.
6. **Feature Engineering**
   - `family_size` = `sibsp` + `parch` + 1
   - `is_alone` flag (dropped later after correlation review)
7. **Exploratory Visualization**
   - Boxplot of `fare` to spot outliers.
   - Correlation heatmap of all numeric features.
8. **Modeling**
   - 80/20 train-test split (`random_state=42`).
   - `LogisticRegression(max_iter=500)` from scikit-learn.
9. **Evaluation**
   - Accuracy score (~**82%** on the test set).
   - Confusion matrix visualized with a `seaborn` heatmap.

## Final Feature Set

| Feature | Description |
|---|---|
| `pclass` | Passenger class (1st, 2nd, 3rd) |
| `sex` | 0 = male, 1 = female |
| `age` | Standardized age |
| `sibsp` | # of siblings/spouses aboard |
| `parch` | # of parents/children aboard |
| `fare` | Standardized fare (outliers removed) |
| `adult_male` | 1 if adult male, else 0 |
| `alone` | 1 if traveling alone, else 0 |
| `embarked_Q`, `embarked_S` | One-hot encoded embarkation port |
| `family_size` | Total family members aboard (incl. self) |

**Target:** `survived` (0 = did not survive, 1 = survived)

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## How to Run

1. Open the notebook in Jupyter Lab/Notebook or VS Code.
2. Run all cells top to bottom — the Titanic dataset is fetched automatically via `seaborn.load_dataset('titanic')` (no manual download needed).
3. Review the final accuracy score and confusion matrix in the last two cells.

## Results

- **Model:** Logistic Regression
- **Test Accuracy:** ≈ 81.9%
- **Evaluation:** Confusion matrix (true vs. predicted survival)

## Notes / Possible Improvements

- Some `FutureWarning`s appear from chained `inplace=True` assignments (e.g., `data['age'].fillna(..., inplace=True)`); consider using `data['age'] = data['age'].fillna(...)` instead for forward compatibility with pandas 3.0.
- `classification_report` and `LabelEncoder` are imported but unused — could be added for a fuller evaluation (precision/recall/F1) or removed for cleanliness.
- Could extend with additional models (Random Forest, XGBoost) for comparison, or use cross-validation instead of a single train/test split.
