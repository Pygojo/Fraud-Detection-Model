# Fraud-Detection-Model

## 📌 Overview
This project applies machine learning techniques to detect fraudulent transactions.  
It combines **supervised learning** (Logistic Regression / XGBoost) with **unsupervised anomaly detection** (Isolation Forest) to flag suspicious activity.  
The model is deployed using **Streamlit**, providing an interactive web interface for predictions.

---

## 🚀 Features
- Preprocessing pipeline for transaction data (encoding, scaling, feature engineering).
- Supervised fraud detection model trained on labeled data.
- Isolation Forest anomaly detector for unseen fraud patterns.
- Streamlit app for user‑friendly predictions.
- Modular design for easy extension and deployment.

---

## 🛠 Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/Pygojo/Fraud-Detection-Model.git
cd Fraud-Detection-Model
pip install -r requirements.txt

---
## Usage
streamlit run Fraud_detection_Model.py

## 📂 Project Structure
fraud-detection-model/
│
├── app.py                  # Streamlit app
├── Fraud_detection_model.pkl  # Trained supervised model
├── preprocessor.pkl        # Preprocessing pipeline
├── iso_model.pkl           # Isolation Forest anomaly detector
├── requirements.txt        # Dependencies
└── README.md               # Project documentation

## 📜 License
This project is licensed under the MIT License.
