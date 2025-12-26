import streamlit as st
import joblib
import pandas as pd

st.set_page_config(
    page_title="Best Model Churn Prediction",
    layout="centered"
)

st.title("📊 Telco Customer Churn Prediction — Best Model")

# =============================
# LOAD MODEL
# =============================
try:
    model = joblib.load("best_model_churn.joblib")
except Exception as e:
    st.error("❌ Gagal memuat model.")
    st.error("Pastikan file `best_model_churn.joblib` ada dan dibuat dengan joblib.")
    st.code(str(e))
    st.stop()

st.write("Masukkan data pelanggan berikut untuk memprediksi churn.")

# =============================
# FORM INPUT
# =============================
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
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

# =============================
# PREDICTION
# =============================
if submit:
    input_df = pd.DataFrame([{
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
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])

    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
    except Exception as e:
        st.error("❌ Model tidak kompatibel dengan input.")
        st.code(str(e))
        st.stop()

    st.subheader("Hasil Prediksi")
    if pred == "Yes":
        st.error(f"⚠️ Pelanggan diprediksi **CHURN** (Probabilitas: {proba:.2f})")
    else:
        st.success(f"✅ Pelanggan diprediksi **TIDAK CHURN** (Probabilitas: {proba:.2f})")
