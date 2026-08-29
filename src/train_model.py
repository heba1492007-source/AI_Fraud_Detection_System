import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# Load the dataset
df = pd.read_csv("data/creditcard.csv")


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


# Train Logistic Regression
logistic_model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_scaled, y_train)


# Train Decision Tree
decision_tree_model = DecisionTreeClassifier(
    class_weight="balanced",
    random_state=42
)

decision_tree_model.fit(X_train_scaled, y_train)


# Train Random Forest
random_forest_model = RandomForestClassifier(
    class_weight="balanced",
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(X_train_scaled, y_train)


# Store models
models = {
    "Logistic Regression": logistic_model,
    "Decision Tree": decision_tree_model,
    "Random Forest": random_forest_model
}


# Evaluate models
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


# Save the best model
joblib.dump(random_forest_model, "models/random_forest_model.pkl")

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

print("\nBest model and scaler saved successfully!")