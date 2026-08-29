import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("AI Fraud Detection System")
st.write("Enter the transaction details below.")

# Features required by the trained model
feature_names = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

st.subheader("Transaction Input")

# Default values
transaction_data = {}

for feature in feature_names:
    transaction_data[feature] = st.number_input(
        feature,
        value=0.0,
        key=feature
    )

# Test Fraud Transaction
if st.button("Test Fraud Transaction"):
    st.session_state["fraud_test"] = True

if st.session_state.get("fraud_test", False):

    fraud_values = {
        "Time": 406.0,
        "V1": -2.312227,
        "V2": 1.951992,
        "V3": -1.609851,
        "V4": 3.997906,
        "V5": -0.522188,
        "V6": -1.426545,
        "V7": -2.537387,
        "V8": 1.391657,
        "V9": -2.770089,
        "V10": -2.772272,
        "V11": 3.202033,
        "V12": -2.899907,
        "V13": -0.595222,
        "V14": -4.289254,
        "V15": 0.389724,
        "V16": -1.140747,
        "V17": -2.830056,
        "V18": -0.016823,
        "V19": 0.416956,
        "V20": 0.126911,
        "V21": 0.517232,
        "V22": -0.035049,
        "V23": -0.465211,
        "V24": 0.320198,
        "V25": 0.044519,
        "V26": 0.177839,
        "V27": 0.261145,
        "V28": -0.143276,
        "Amount": 0.0
    }

    st.info("Fraud test transaction loaded. Scroll down and click Predict Transaction.")

    transaction_data = fraud_values

# Prediction
if st.button("Predict Transaction"):

    transaction_df = pd.DataFrame([transaction_data], columns=feature_names)

    transaction_scaled = scaler.transform(transaction_df)

    prediction = model.predict(transaction_scaled)[0]

    fraud_probability = model.predict_proba(transaction_scaled)[0][1]

    if prediction == 1:
        st.error("Fraud")
    else:
        st.success("Normal")

    st.write(f"Fraud Probability: {fraud_probability:.2%}")