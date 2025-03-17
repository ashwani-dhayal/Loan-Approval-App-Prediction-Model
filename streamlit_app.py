import streamlit as st
import requests
import pandas as pd

st.title("Loan Prediction App")

# Create form for user input
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    
    with col2:
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
        loan_amount = st.number_input("Loan Amount", min_value=0)
    
    loan_term = st.selectbox("Loan Term", [360, 180, 240, 120])
    credit_history = st.selectbox("Credit History", [1.0, 0.0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        # Prepare data for API request
        data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": property_area
        }
        
        # Make API request to Flask backend
        response = requests.post("http://localhost:5000/predict", json=data)
        result = response.json()
        
        if "loan_approval" in result:
            if result["loan_approval"]:
                st.success("Loan Approved!")
            else:
                st.error("Loan Not Approved")
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")

