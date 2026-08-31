import os
import sys
import json

import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


DATA_PATH = "data/creditcard.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Load the dataset
if not os.path.exists(DATA_PATH):
    sys.exit(
        f"\nERROR: Could not find '{DATA_PATH}'.\n"
        "Download the dataset from "
        "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
        "and place 'creditcard.csv' inside the 'data/' folder.\n"
    )

df = pd.read_csv(DATA_PATH)


# Remove duplicated rows
df = df.drop_duplicates()


# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# --- Handle class imbalance (training set only) ---
# The dataset is highly imbalanced (~0.17% fraud). Relying only on
# class_weight="balanced" tends to over-correct for Logistic Regression:
# it pushes Recall up but tanks Precision (the model flags far too many
# normal transactions as fraud). SMOTE fixes this at the data level by
# generating synthetic "Fraud" samples so every model sees a properly
# balanced training set instead of relying on per-model weighting.
#
# IMPORTANT: SMOTE is fit/applied ONLY on X_train / y_train. The test set
# (X_test / y_test) is left exactly as-is, at the real-world imbalance,
# so evaluation metrics reflect how the model would actually perform in
# production and no synthetic information leaks into the test set.
print("Class distribution before SMOTE:", y_train.value_counts().to_dict())

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("Class distribution after SMOTE:", y_train_res.value_counts().to_dict())


# Train Logistic Regression
logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_res, y_train_res)


# Train Decision Tree
decision_tree_model = DecisionTreeClassifier(
    random_state=42
)

decision_tree_model.fit(X_train_res, y_train_res)


# Train Random Forest
random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train_res, y_train_res)


# Train XGBoost
xgboost_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgboost_model.fit(X_train_res, y_train_res)


# Store models
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model,
    "XGBoost": xgboost_model
}


# Evaluate models and keep track of results so the best one can be
# selected automatically instead of being hardcoded.
all_results = {}
best_model_name = None
best_f1 = -1.0

for model_name, model in models.items():

    y_pred = model.predict(X_test_scaled)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 50)
    print(model_name)
    print("=" * 50)

    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    print("\nConfusion Matrix:")
    print(cm)

    all_results[model_name] = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm.tolist()
    }

    # Save a confusion matrix plot for each model (used in the README /
    # screenshots for documentation).
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Fraud"]
    )
    disp.plot(cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(f"{RESULTS_DIR}/confusion_matrix_{safe_name}.png")
    plt.close()

    # Track the best model based on F1 score, since accuracy is
    # misleading on this highly imbalanced dataset.
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = model_name

best_model = models[best_model_name]

print("\n" + "=" * 50)
print(f"Best model selected: {best_model_name} (F1 Score: {best_f1:.4f})")
print("=" * 50)


# Save evaluation results for documentation / README
with open(f"{RESULTS_DIR}/evaluation_results.json", "w") as f:
    json.dump(
        {"results": all_results, "best_model": best_model_name},
        f,
        indent=4
    )

# Save the best model (chosen automatically by F1 score).
# Saved as "best_model.pkl" so the filename always matches whichever
# algorithm actually won, instead of assuming it's Random Forest.
joblib.dump(best_model, f"{MODELS_DIR}/best_model.pkl")

# Save the scaler
joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")

print(f"\nBest model ({best_model_name}) and scaler saved successfully!")
print(f"Evaluation results saved to {RESULTS_DIR}/evaluation_results.json")
