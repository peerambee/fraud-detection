import streamlit as st
import numpy as np
import joblib

# -----------------------------
# Load Model & Tools
# -----------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Fraud Detection System", page_icon="💳", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
    💳 AI Fraud Detection System
    </h1>
""", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("📝 Enter Transaction Details")

amount = st.sidebar.number_input("💰 Transaction Amount", min_value=0.0)

payment = st.sidebar.selectbox("💳 Payment Method", encoders['Payment Method'].classes_)
category = st.sidebar.selectbox("📦 Product Category", encoders['Product Category'].classes_)

quantity = st.sidebar.number_input("🔢 Quantity", min_value=1)
age = st.sidebar.number_input("👤 Customer Age", min_value=1)

location = st.sidebar.selectbox("🌍 Customer Location", encoders['Customer Location'].classes_)
device = st.sidebar.selectbox("📱 Device Used", encoders['Device Used'].classes_)

shipping = st.sidebar.selectbox("🚚 Shipping Address", encoders['Shipping Address'].classes_)
billing = st.sidebar.selectbox("🏠 Billing Address", encoders['Billing Address'].classes_)

account_age = st.sidebar.number_input("📅 Account Age Days", min_value=0)
hour = st.sidebar.slider("⏰ Transaction Hour", 0, 23)

day = st.sidebar.slider("📆 Transaction Day", 1, 31)
month = st.sidebar.slider("📅 Transaction Month", 1, 12)

# -----------------------------
# Encoding Function
# -----------------------------
def encode(col, value):
    return encoders[col].transform([value])[0]

# -----------------------------
# Prepare Input
# -----------------------------
input_data = np.array([[
    amount,
    encode('Payment Method', payment),
    encode('Product Category', category),
    quantity,
    age,
    encode('Customer Location', location),
    encode('Device Used', device),
    encode('Shipping Address', shipping),
    encode('Billing Address', billing),
    account_age,
    hour,
    day,
    month
]])

# Scale input
input_scaled = scaler.transform(input_data)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Check Transaction", use_container_width=True):

    prob = model.predict_proba(input_scaled)[0][1]

    st.markdown("## 🧠 Prediction Result")

    # Risk Levels
    if prob < 0.4:
        st.markdown(f"""
        <div style='background-color:#4CAF50;padding:20px;border-radius:10px;text-align:center'>
        <h2 style='color:white;'>✅ Genuine Transaction</h2>
        <h3 style='color:white;'>Risk Score: {prob*100:.2f}% (Low Risk)</h3>
        </div>
        """, unsafe_allow_html=True)

    elif prob < 0.7:
        st.markdown(f"""
        <div style='background-color:#FFA500;padding:20px;border-radius:10px;text-align:center'>
        <h2 style='color:white;'>⚠️ Suspicious Transaction</h2>
        <h3 style='color:white;'>Risk Score: {prob*100:.2f}% (Medium Risk)</h3>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style='background-color:#ff4d4d;padding:20px;border-radius:10px;text-align:center'>
        <h2 style='color:white;'>🚨 Fraudulent Transaction</h2>
        <h3 style='color:white;'>Risk Score: {prob*100:.2f}% (High Risk)</h3>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    "<center>🚀 AI Fraud Detection System | Final Year Project</center>",
    unsafe_allow_html=True
)