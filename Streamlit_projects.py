import streamlit as st
import pandas as pd
import pickle
 
# Load model
with open("loan_model.pkl", "rb") as f:
    model = pickle.load(f)
 
st.title("Loan Approval Predictor")
st.write("Fill in the applicant details and click **Predict**.")
 
# ── Input fields ──────────────────────────────────────────────────────────────
gender         = st.selectbox("Gender", ["Female", "Male"])
married        = st.selectbox("Married", ["No", "Yes"])
dependents     = st.selectbox("Number of Dependents", [0, 1, 2, 3])
education      = st.selectbox("Education", ["Not Graduate", "Graduate"])
self_employed  = st.selectbox("Self Employed", ["No", "Yes"])
applicant_income   = st.number_input("Applicant Income ($/month)", min_value=0, value=5000, step=500)
coapplicant_income = st.number_input("Co-applicant Income ($/month)", min_value=0, value=0, step=500)
loan_amount    = st.slider("Loan Amount (thousands $)", 50, 700, 150)
loan_term      = st.selectbox("Loan Term (months)", [120, 180, 240, 360])
credit_history = st.selectbox("Credit History", ["Bad (0)", "Good (1)"])
property_area  = st.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])
 
# ── Encode inputs ─────────────────────────────────────────────────────────────
input_data = pd.DataFrame([{
    "Gender":            1 if gender == "Male" else 0,
    "Married":           1 if married == "Yes" else 0,
    "Dependents":        dependents,
    "Education":         1 if education == "Graduate" else 0,
    "Self_Employed":     1 if self_employed == "Yes" else 0,
    "ApplicantIncome":   applicant_income,
    "CoapplicantIncome": coapplicant_income,
    "LoanAmount":        loan_amount,
    "Loan_Amount_Term":  loan_term,
    "Credit_History":    1 if credit_history == "Good (1)" else 0,
    "Property_Area":     ["Rural", "Semiurban", "Urban"].index(property_area)
}])
 
# ── Predict button ────────────────────────────────────────────────────────────
if st.button("Predict"):
    result = model.predict(input_data)[0]
    prob   = model.predict_proba(input_data)[0][result]
 
    if result == 1:
        st.success(f"✅ Loan APPROVED  (confidence: {prob*100:.1f}%)")
    else:
        st.error(f"❌ Loan REJECTED  (confidence: {prob*100:.1f}%)")
 
# ── Feature importance chart ──────────────────────────────────────────────────
st.subheader("Feature Importance")
feature_names = [
    "Gender", "Married", "Dependents", "Education", "Self Employed",
    "Applicant Income", "Coapplicant Income", "Loan Amount",
    "Loan Term", "Credit History", "Property Area"
]
importance_df = pd.DataFrame(
    {"Importance": model.feature_importances_},
    index=feature_names
)
st.bar_chart(importance_df)
 