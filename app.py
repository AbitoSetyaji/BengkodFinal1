import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ==============================
# LOAD MODEL & PREPROCESSOR
# ==============================
st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("📊 Telco Customer Churn Prediction")

try:
    model = joblib.load("best_model_churn.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error("❌ Gagal memuat model atau encoder.")
    st.code(str(e))
    st.stop()

st.write("Masukkan data pelanggan berikut:")

# ==============================
# FORM INPUT
# ==============================
with st.form("form"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (bulan)", 0, 100, 1)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multilines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_prot = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    stream_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    stream_mov = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    monthly = st.number_input("Monthly Charges", 0.0, 500.0, 50.0)
    total = st.number_input("Total Charges", 0.0, 10000.0, 100.0)

    submit = st.form_submit_button("Prediksi")

# ==============================
# PROCESSING
# ==============================
if submit:
    raw = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multilines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_mov,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # ============ APPLY LABEL ENCODER ============
    df = raw.copy()
    for col in df.columns:
        if col in label_encoders:
            df[col] = label_encoders[col].transform(df[col].astype(str))

    # ============ SCALE NUMERIC ============
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    # ============ PREDICT ============
    try:
        pred = model.predict(df)[0]
    except Exception as e:
        st.error("Model error saat prediksi.")
        st.code(str(e))
        st.stop()

    st.subheader("Hasil Prediksi")
    if pred == 1:
        st.error("⚠️ Pelanggan diprediksi CHURN")
    else:
        st.success("✅ Pelanggan diprediksi TIDAK CHURN")
