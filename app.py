import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(page_title="Best Model Churn Prediction", layout="centered")

st.title("📊 Telco Customer Churn Prediction — Best Model")

# Load model
try:
    with open("best_model_churn.pkl", "rb") as f:
        model = pickle.load(f)
except Exception as e:
    st.error("❌ Gagal memuat model. Pastikan file best_model_churn.pkl ada dan kompatibel.")
    st.code(str(e))
    st.stop()

st.write("Masukkan data pelanggan berikut untuk memprediksi apakah pelanggan akan churn atau tidak.")

# Form input
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    depend = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (bulan)", 0, 100, 1)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multilines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    stream_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

    submit = st.form_submit_button("Prediksi")

# Predict
if submit:
    data = pd.DataFrame(
        {
            "gender": [gender],
            "SeniorCitizen": [senior],
            "Partner": [partner],
            "Dependents": [depend],
            "tenure": [tenure],
            "PhoneService": [phone],
            "MultipleLines": [multilines],
            "InternetService": [internet],
            "OnlineSecurity": [online_sec],
            "OnlineBackup": [online_backup],
            "DeviceProtection": [device_prot],
            "TechSupport": [tech],
            "StreamingTV": [stream_tv],
            "StreamingMovies": [stream_movies],
            "Contract": [contract],
            "PaperlessBilling": [paperless],
            "PaymentMethod": [payment],
            "MonthlyCharges": [monthly],
            "TotalCharges": [total],
        }
    )

    try:
        pred = model.predict(data)[0]
        proba = model.predict_proba(data)[0][1]
    except Exception as e:
        st.error("❌ Error saat melakukan prediksi. Model kemungkinan tidak kompatibel dengan input.")
        st.code(str(e))
        st.stop()

    st.subheader("Hasil Prediksi")
    if pred == "Yes":
        st.error(f"⚠️ Pelanggan diprediksi **Churn** dengan probabilitas {proba:.2f}")
    else:
        st.success(f"✅ Pelanggan diprediksi **Tidak Churn** dengan probabilitas {proba:.2f}")
