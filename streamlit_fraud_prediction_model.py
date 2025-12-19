import pandas as pd
import numpy as np
import streamlit as st
import joblib

model = joblib.load("fraud_prediction_model.pkl")
preprocessor = joblib.load("fraud_preprocessor.pkl")
iso = joblib.load("fraud_iso.pkl")
model_base = joblib.load("base_fraud_prediction_model.pkl")

df = pd.read_csv(
    r"C:\Users\emmanuel\Nigeria_fraud_detection_model\nibss_fraud_dataset.csv"
)


st.title("Fraud Prediction Model")
st.header("Input Transaction Details")
amount = st.number_input("Transaction Amount", min_value=0.0, value=1000.0)
hour = st.number_input(
    "Hour of Transaction (0-23)", min_value=0, max_value=23, value=12
)
weekday = st.number_input(
    "Day of the Week (0=Monday, 6=Sunday)", min_value=0, max_value=6, value=2
)
month = st.number_input("Month of the Year (1-12)", min_value=1, max_value=12, value=6)
channel = st.selectbox(
    "Transaction Channel", options=["Online", "POS", "ATM", "Mobile"]
)
merchant_category = st.selectbox(
    "Merchant Category", options=["Retail", "Food", "Travel", "Entertainment"]
)
location = st.selectbox("Transaction Location", options=["Urban", "Suburban", "Rural"])
age_group = st.selectbox(
    "Customer Age Group", options=["18-25", "26-35", "36-45", "46-55", "56+"]
)

numeric_cols = [
    "tx_count_24h",
    "amount_sum_24h",
    "amount_mean_7d",
    "amount_std_7d",
    "tx_count_total",
    "velocity_score",
]

global_averages = df[numeric_cols].mean()
print(global_averages)

tx_count_24h_avg = global_averages["tx_count_24h"]
amount_sum_24h_avg = global_averages["amount_sum_24h"]
amount_mean_7d_avg = global_averages["amount_mean_7d"]
amount_std_7d_avg = global_averages["amount_std_7d"]
tx_count_total_avg = global_averages["tx_count_total"]
velocity_score_avg = global_averages["velocity_score"]

amount_log = np.log(amount + 1)
amount_vs_mean_ratio = amount / amount_mean_7d_avg

# Hour
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)

# Day
day_sin = np.sin(2 * np.pi * weekday / 7)
day_cos = np.cos(2 * np.pi * weekday / 7)

# Month
month_sin = np.sin(2 * np.pi * (month - 1) / 12)
month_cos = np.cos(2 * np.pi * (month - 1) / 12)

if st.button("Predict Fraud"):
    input_data = pd.DataFrame(
        [
            {
                "channel": channel,
                "merchant_category": merchant_category,
                "location": location,
                "age_group": age_group,
                "tx_count_24h": tx_count_24h_avg,
                "amount_sum_24h": amount_sum_24h_avg,
                "amount_mean_7d": amount_mean_7d_avg,
                "amount_std_7d": amount_std_7d_avg,
                "amount_vs_mean_ratio": amount_vs_mean_ratio,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "day_sin": day_sin,
                "day_cos": day_cos,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "amount_log": amount_log,
                "velocity_score": velocity_score_avg,
            }
        ],
    )
st.subheader("Prepared Input Data for Prediction")
for col in model_base.feature_names_:
    if col not in input_data.columns:
        input_data[col] = 0  # or a meaningful default

input_data = input_data[model_base.feature_names_]

new_probs = model_base.predict_proba(input_data)[:, 1]

# Isolation Forest (needs transformed features)
transformed_input = preprocessor.transform(input_data)
new_scores = iso.decision_function(transformed_input)[0]

# Meta model input
df_predictor = np.array([[new_probs, new_scores]])
prediction = model.predict(df_predictor)
result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
st.success(f"The transaction is predicted to be: {result}")
