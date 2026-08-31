# Data Folder

This folder must contain `creditcard.csv` before running `src/train_model.py`, `src/predict.py`, or `notebooks/eda.py`.

The dataset is **not included in this repository** (it is excluded via `.gitignore`) because of its size (~150 MB).

## How to get it

1. Download **Credit Card Fraud Detection** from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Extract the archive.
3. Place `creditcard.csv` directly inside this `data/` folder, so the path is:
   ```
   data/creditcard.csv
   ```

## About the dataset

- 284,807 transactions, 492 of which are fraud (highly imbalanced, ~0.17% fraud rate)
- Features: `Time`, `V1`–`V28` (PCA-transformed, anonymized), `Amount`
- Target: `Class` (0 = Normal, 1 = Fraud)
