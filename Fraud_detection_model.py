import streamlit as st
import joblib
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier


# Load the pre-trained fraud detection model
model = joblib.load("Fraud_detection_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")
iso = joblib.load("iso_model.pkl")
model_base = joblib.load("model.pkl")


st.title("Fraud Detection Model")
st.write("Enter the transaction details to predict if it's fraudulent or not.")

# Define input fields for transaction features
amount = st.number_input("Transaction Amount in USD", min_value=0.0, step=0.01)
transaction_type = st.selectbox("Transaction Type", ["POS", "Online", "other"])
city = st.text_input("Account Holder City")
age = st.number_input("Account Holder Age", min_value=0, step=1)
category = st.text_input(
    "Transaction Category: 'Food', 'Electronics', 'Clothing', 'other'"
)
state = st.text_input("Account Holder State")
hour = st.number_input("Transaction Hour (0-23)", min_value=0, max_value=23, step=1)
day = st.number_input(
    "Transaction Day of the Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, step=1
)
month = st.number_input("Transaction Month (1-12)", min_value=1, max_value=12, step=1)
distance_from_home = st.number_input(
    "Distance from Home (in kilometers)", min_value=0.0, step=0.1
)
gender = st.selectbox("Account Holder Gender", ["M", "F"])

if st.button("Predict Fraud"):
    # Create a DataFrame for the input features
    import pandas as pd

    input_data = pd.DataFrame(
        {
            "category": [category],
            "amt": [amount],
            "gender": [gender],
            "city": [city],
            "state": [state],
            "age": [age],
            "hour": [hour],
            "weekday": [day],
            "month": [month],
            "distance_km": [distance_from_home],
            "channel": [transaction_type],
        }
    )

    new_probs = model_base.predict_proba(input_data)[:, 1]

    # Isolation Forest (needs transformed features)
    transformed_input = preprocessor.transform(input_data)
    new_scores = iso.decision_function(transformed_input)[0]

    # Meta model input
    df_predictor = np.column_stack((new_probs, new_scores))
    prediction = model.predict(df_predictor)
    result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
    st.success(f"The transaction is predicted to be: {result}")
