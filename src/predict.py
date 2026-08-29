import pandas as pd
import joblib


# Load the trained model
model = joblib.load("models/random_forest_model.pkl")

# Load the scaler
scaler = joblib.load("models/scaler.pkl")


def predict_transaction(transaction_data):
    """
    Predict whether a transaction is Normal or Fraud.
    """

    # Convert input data to DataFrame
    transaction_df = pd.DataFrame(
        [transaction_data]
    )

    # Scale the input
    transaction_scaled = scaler.transform(transaction_df)

    # Make prediction
    prediction = model.predict(transaction_scaled)[0]

    # Get fraud probability
    fraud_probability = model.predict_proba(
        transaction_scaled
    )[0][1]

    # Convert prediction to label
    if prediction == 1:
        result = "Fraud"
    else:
        result = "Normal"

    return result, fraud_probability


# Test prediction using a real fraud transaction
df = pd.read_csv("data/creditcard.csv")
df = df.drop_duplicates()

features = df.drop("Class", axis=1)
target = df["Class"]

fraud_index = target[target == 1].index[0]

sample_transaction = features.loc[fraud_index].to_dict()

result, probability = predict_transaction(sample_transaction)

print("Prediction:", result)
print("Fraud Probability:", probability)