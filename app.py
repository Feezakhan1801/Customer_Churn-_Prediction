import joblib
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Customer Churn AI Dashboard",
    layout="wide",
    page_icon="📊"
)

# =========================
# CUSTOM CSS (PRO UI LOOK)
# =========================
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .block-container {
        padding: 2rem 2rem;
    }
    h1, h2, h3 {
        color: #38bdf8;
    }
    .stMetric {
        background-color: #1e293b;
        padding: 15px;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("models/model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    return model, encoders

model, encoders = load_model()

# =========================
# ENCODING
# =========================
def encode_input(df):
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col])
    return df

# =========================
# RECOMMENDATION ENGINE
# =========================
def get_recommendation(prob, row):
    recs = []

    if prob > 0.7:
        recs = [
            "URGENT: Offer 20–30% discount",
            "Assign dedicated support agent",
            "Provide loyalty rewards",
        ]
    elif prob > 0.3:
        recs = [
            "Offer 10–15% discount",
            "Suggest plan upgrade",
            "Run engagement campaign",
        ]
    else:
        recs = [
            "Maintain engagement",
            "Referral bonus",
            "Collect feedback",
        ]

    if row["MonthlyCharges"].values[0] > 80:
        recs.append("Optimize pricing plan")

    return recs

# =========================
# PDF REPORT
# =========================
def generate_pdf(pred, prob, risk, recs, features):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Customer Churn Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 720, f"Prediction: {'CHURN' if pred else 'NO CHURN'}")
    c.drawString(50, 700, f"Probability: {round(prob*100,2)}%")
    c.drawString(50, 680, f"Risk: {risk}")

    y = 640
    c.drawString(50, y, "Recommendations:")
    y -= 20

    for r in recs[:6]:
        c.drawString(60, y, f"- {r}")
        y -= 15

    c.save()
    buffer.seek(0)
    return buffer

# =========================
# HEADER
# =========================
st.title("📊 Customer Churn Prediction Dashboard")
st.caption("AI-powered retention & churn analysis system")

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0, 200, 70)
TotalCharges = st.sidebar.number_input("Total Charges", 0, 10000, 500)

# =========================
# INPUT DF
# =========================
input_data = pd.DataFrame([[
    gender, SeniorCitizen, Partner, Dependents, tenure,
    PhoneService, MultipleLines, InternetService,
    OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies,
    Contract, PaperlessBilling, PaymentMethod,
    MonthlyCharges, TotalCharges
]], columns=[
    "gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService",
    "OnlineSecurity","OnlineBackup","DeviceProtection",
    "TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges"
])

input_data = encode_input(input_data)

# =========================
# PREDICT BUTTON
# =========================
if st.button("🚀 Predict Churn", use_container_width=True):

    with st.spinner("Analyzing customer behavior..."):
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

    # Risk
    if prob < 0.3:
        risk = "Low 🟢"
    elif prob < 0.7:
        risk = "Medium 🟡"
    else:
        risk = "High 🔴"

    # =========================
    # DASHBOARD CARDS
    # =========================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", "CHURN" if pred else "NO CHURN")

    with col2:
        st.metric("Probability", f"{round(prob*100,2)}%")

    with col3:
        st.metric("Risk Level", risk)

    # =========================
    # RETENTION SCORE
    # =========================
    retention_score = int((1 - prob) * 100)

    st.subheader("📊 Retention Score")
    st.progress(retention_score)

    if retention_score > 70:
        st.success("Loyal Customer")
    elif retention_score > 40:
        st.warning("At Risk Customer")
    else:
        st.error("High Risk Customer")

    # =========================
    # RECOMMENDATIONS
    # =========================
    st.subheader("🤖 AI Recommendations")
    recs = get_recommendation(prob, input_data)

    for r in recs:
        st.write("•", r)

    # =========================
    # SHAP EXPLAINABILITY
    # =========================
    st.subheader("🧠 Feature Impact (Explainable AI)")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        shap_df = pd.DataFrame({
            "Feature": input_data.columns,
            "Impact": shap_values[0]
        }).sort_values("Impact", ascending=False)

        st.dataframe(shap_df, use_container_width=True)
        st.bar_chart(shap_df.set_index("Feature"))

    except:
        st.warning("SHAP explanation not available for this model.")

    # =========================
    # SUMMARY
    # =========================
    st.subheader("🧾 AI Summary")

    if prob < 0.3:
        st.info("Customer is stable with low churn risk.")
    elif prob < 0.7:
        st.warning("Moderate churn risk detected.")
    else:
        st.error("High churn risk — immediate action needed.")

    # =========================
    # DOWNLOAD REPORT
    # =========================
    st.subheader("📄 Export Report")

    pdf_file = generate_pdf(pred, prob, risk, recs, None)

    st.download_button(
        "⬇️ Download PDF Report",
        data=pdf_file,
        file_name="churn_report.pdf",
        mime="application/pdf"
    )