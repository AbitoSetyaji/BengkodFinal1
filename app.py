import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load Model
model = joblib.load("best_model_churn.pkl")

st.title("🔮 Telco Customer Churn Prediction")
st.write("Aplikasi Prediksi Churn Pelanggan Telco – UAS Bengkel Koding UDINUS")

# Input Features
st.header("📌 Masukkan Data Pelanggan")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", 0, 100, 1)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method",
                             ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

# Convert to DataFrame for prediction
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [SeniorCitizen],
    "Partner": [Partner],
    "Dependents": [Dependents],
    "tenure": [tenure],
    "PhoneService": [PhoneService],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "OnlineBackup": [OnlineBackup],
    "DeviceProtection": [DeviceProtection],
    "TechSupport": [TechSupport],
    "StreamingTV": [StreamingTV],
    "StreamingMovies": [StreamingMovies],
    "Contract": [Contract],
    "PaperlessBilling": [PaperlessBilling],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

# Encode input using label encoding from model training
# NOTE: Untuk kemudahan, Streamlit memakai model yang sudah di-label encode saat training
# Pastikan kolom urutannya sama seperti dataset training

if st.button("🔮 Prediksi Churn"):
    prediction = model.predict(input_data)[0]
    result = "CHURN" if prediction == 1 else "TIDAK CHURN"

    st.subheader("📢 Hasil Prediksi:")
    st.success(f"Pelanggan diprediksi: **{result}**")
