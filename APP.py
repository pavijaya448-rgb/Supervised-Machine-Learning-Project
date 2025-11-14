import streamlit as st
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load model and encoders
model = joblib.load(r"C:\Users\WELCOME\PycharmProjects\MachineLearningProject\model\best_model.pkl")
le = joblib.load(r"C:\Users\WELCOME\PycharmProjects\MachineLearningProject\model\label_encoders.pkl")
scl = joblib.load(r"C:\Users\WELCOME\PycharmProjects\MachineLearningProject\model\scaler.pkl")

st.set_page_config(page_title="Telco Churn Prediction", layout="centered")

st.title("📊 Telco Churn Prediction")
st.write("Predict whether a customer will **churn** or **stay** based on their details.")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.number_input("Tenure (months)", min_value=0, max_value=80, value=10)
MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaymentMethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0)

# DataFrame for input
input_data = pd.DataFrame({
    "gender": [gender],
    "tenure": [tenure],
    "MultipleLines": [MultipleLines],
    "InternetService": [InternetService],
    "OnlineSecurity": [OnlineSecurity],
    "TechSupport": [TechSupport],
    "Contract": [Contract],
    "PaymentMethod": [PaymentMethod],
    "MonthlyCharges": [MonthlyCharges],
    "TotalCharges": [TotalCharges]
})

from sklearn.preprocessing import LabelEncoder

le_temp = LabelEncoder()
input_data["gender"] = le_temp.fit(["Male", "Female"]).transform(input_data["gender"])
input_data["MultipleLines"] = le_temp.fit(["No", "Yes", "No phone service"]).transform(input_data["MultipleLines"])
input_data["InternetService"] = le_temp.fit(["DSL", "Fiber optic", "No"]).transform(input_data["InternetService"])
input_data["OnlineSecurity"] = le_temp.fit(["No", "Yes", "No internet service"]).transform(input_data["OnlineSecurity"])
input_data["TechSupport"] = le_temp.fit(["No", "Yes", "No internet service"]).transform(input_data["TechSupport"])
input_data["Contract"] = le_temp.fit(["Month-to-month", "One year", "Two year"]).transform(input_data["Contract"])
input_data["PaymentMethod"] = le_temp.fit([
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
]).transform(input_data["PaymentMethod"])

# Scale numeric values
input_scaled = scl.transform(input_data)

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ The customer is likely to **CHURN**.")
    else:
        st.success("✅ The customer is likely to **STAY**.")
