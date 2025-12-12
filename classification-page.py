# EMI Eligibility Classification Module 
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import json

def render():
    # Page Header with Modern Design
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                color: white; 
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h1 style="margin: 0; font-size: 2.5rem;">🎯 EMI Eligibility Predictor</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Predict loan eligibility using advanced classification models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Configuration
    MODEL_PATHS = {
        "XGBoost Classifier": "models/classification/xgboost_classifier.pkl",
        "Random Forest Classifier": "models/classification/random_forest_classifier.pkl",
        "Logistic Regression": "models/classification/logistic_regression.pkl"
    }
    
    PREPROCESSOR_PATH = "models/classification/preprocessor.pkl"
    LABEL_ENCODER_PATH = "models/classification/label_encoder.pkl"
    FEATURE_ORDER_PATH = "models/classification/feature_order.json"
    
    # Model Selection
    st.markdown("### 🤖 **Select Classification Model**")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Choose a model for prediction:",
            list(MODEL_PATHS.keys()),
            help="Different models offer varying prediction approaches"
        )
    
    with col2:
        st.info(f"**Selected:** {selected_model}")
    
    # Load Model and Artifacts
    model_path = MODEL_PATHS[selected_model]
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}")
        st.info("Please ensure models are trained and saved in the correct directory.")
        return
    
    try:
        model = joblib.load(model_path)
        label_encoder = joblib.load(LABEL_ENCODER_PATH) if os.path.exists(LABEL_ENCODER_PATH) else None
        
        # IMPORTANT: Load feature order from training
        feature_order = None
        if os.path.exists(FEATURE_ORDER_PATH):
            with open(FEATURE_ORDER_PATH, 'r') as f:
                feature_order = json.load(f)
        
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return
    
    st.markdown("---")
    
    # Input Form
    st.markdown("### 📝 **Enter Applicant Details**")
    
    with st.form("prediction_form"):
        # Personal Information
        st.markdown("#### 👤 Personal Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=30, key="age")
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
        with col3:
            marital_status = st.selectbox("Marital Status", ["Single", "Married"], key="marital_status")
        with col4:
            education = st.selectbox("Education", 
                ["High School", "Graduate", "Post Graduate", "Professional"], 
                key="education")
        
        st.markdown("#### 💼 Employment Details")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_salary = st.number_input("Monthly Salary (₹)", 
                min_value=10000, max_value=500000, value=50000, step=1000, key="salary")
        with col2:
            employment_type = st.selectbox("Employment Type", 
                ["Private", "Government", "Self-employed"], key="employment")
        with col3:
            years_of_employment = st.number_input("Years of Employment", 
                min_value=0.0, max_value=40.0, value=5.0, step=0.5, key="years_emp")
        with col4:
            company_type = st.selectbox("Company Type", 
                ["Startup", "Mid-size", "MNC", "Large Indian"], key="company")
        
        st.markdown("#### 🏠 Housing & Family")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            house_type = st.selectbox("House Type", ["Own", "Rented", "Family"], key="house")
        with col2:
            monthly_rent = st.number_input("Monthly Rent (₹)", 
                min_value=0, max_value=100000, value=0, step=500, key="rent")
        with col3:
            family_size = st.number_input("Family Size", 
                min_value=1, max_value=10, value=3, key="family")
        with col4:
            dependents = st.number_input("Dependents", 
                min_value=0, max_value=8, value=1, key="dependents")
        
        st.markdown("#### 💰 Financial Details")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            credit_score = st.slider("Credit Score", 300, 900, 650, key="credit")
        with col2:
            bank_balance = st.number_input("Bank Balance (₹)", 
                min_value=0, max_value=10000000, value=100000, step=5000, key="balance")
        with col3:
            emergency_fund = st.number_input("Emergency Fund (₹)", 
                min_value=0, max_value=5000000, value=50000, step=5000, key="emergency")
        with col4:
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"], key="loans")
        
        st.markdown("#### 📊 Monthly Expenses")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            school_fees = st.number_input("School Fees (₹)", 
                min_value=0, max_value=50000, value=0, step=500, key="school")
        with col2:
            college_fees = st.number_input("College Fees (₹)", 
                min_value=0, max_value=100000, value=0, step=1000, key="college")
        with col3:
            travel_expenses = st.number_input("Travel Expenses (₹)", 
                min_value=0, max_value=50000, value=3000, step=500, key="travel")
        with col4:
            groceries_utilities = st.number_input("Groceries & Utilities (₹)", 
                min_value=0, max_value=50000, value=10000, step=500, key="groceries")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            other_monthly_expenses = st.number_input("Other Expenses (₹)", 
                min_value=0, max_value=50000, value=5000, step=500, key="other")
        with col2:
            current_emi_amount = st.number_input("Current EMI (₹)", 
                min_value=0, max_value=200000, value=0, step=1000, key="current_emi")
        with col3:
            requested_amount = st.number_input("Requested Loan Amount (₹)", 
                min_value=10000, max_value=5000000, value=200000, step=10000, key="req_amount")
        with col4:
            requested_tenure = st.number_input("Requested Tenure (months)", 
                min_value=6, max_value=360, value=60, step=6, key="tenure")
        
        st.markdown("#### 📋 Loan Details")
        col1, col2 = st.columns(2)
        
        with col1:
            emi_scenario = st.selectbox("EMI Scenario", 
                ["Home Appliances EMI", "Vehicle EMI", "Education EMI", 
                 "Personal Loan EMI", "E-commerce Shopping EMI"], 
                key="scenario")
        
        # Submit Button
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Predict Eligibility", use_container_width=True)
    
    # Prediction Logic
    if submitted:
        with st.spinner("🔄 Analyzing financial profile..."):
            try:
                # ====== STEP 1: Prepare input data ======
                input_data = {
                    'age': age,
                    'gender': gender,
                    'marital_status': marital_status,
                    'education': education,
                    'monthly_salary': monthly_salary,
                    'employment_type': employment_type,
                    'years_of_employment': years_of_employment,
                    'company_type': company_type,
                    'house_type': house_type,
                    'monthly_rent': monthly_rent,
                    'family_size': family_size,
                    'dependents': dependents,
                    'school_fees': school_fees,
                    'college_fees': college_fees,
                    'travel_expenses': travel_expenses,
                    'groceries_utilities': groceries_utilities,
                    'other_monthly_expenses': other_monthly_expenses,
                    'existing_loans': 1 if existing_loans == "Yes" else 0,
                    'current_emi_amount': current_emi_amount,
                    'credit_score': credit_score,
                    'bank_balance': bank_balance,
                    'emergency_fund': emergency_fund,
                    'emi_scenario': emi_scenario,
                    'requested_amount': requested_amount,
                    'requested_tenure': requested_tenure
                }
                
                input_df = pd.DataFrame([input_data])
                
                # ====== STEP 2: Engineer Features (Same as Training) ======
                input_df['total_expenses'] = (input_df['monthly_rent'] + input_df['school_fees'] + 
                                              input_df['college_fees'] + input_df['travel_expenses'] + 
                                              input_df['groceries_utilities'] + input_df['other_monthly_expenses'] + 
                                              input_df['current_emi_amount'])
                
                input_df['net_monthly_income'] = input_df['monthly_salary'] - input_df['total_expenses']
                input_df['expense_to_income_ratio'] = input_df['total_expenses'] / (input_df['monthly_salary'] + 1e-9)
                input_df['dependency_ratio'] = input_df['dependents'] / (input_df['family_size'] + 1e-9)
                
                input_df['risk_score'] = (
                    (1 - (input_df['credit_score'] - 300) / 600) * 0.7 + 
                    (input_df['current_emi_amount'] / (input_df['monthly_salary'] + 1e-9)) * 0.3
                )
                input_df['risk_score'] = input_df['risk_score'].clip(0, 1)
                
                input_df['loan_to_income_ratio'] = input_df['requested_amount'] / ((input_df['monthly_salary'] * 12) + 1e-9)
                input_df['savings_capacity'] = input_df['bank_balance'] + input_df['emergency_fund']
                
                # ====== STEP 3: One-Hot Encode Categorical Variables ======
                categorical_columns = ['gender', 'marital_status', 'education', 'employment_type',
                                      'company_type', 'house_type', 'emi_scenario']
                
                input_df_encoded = pd.get_dummies(input_df, columns=categorical_columns, drop_first=False)
                
                # ====== STEP 4: Ensure Column Order Matches Training ======
                # This is CRITICAL to avoid feature name mismatch errors
                if feature_order is not None:
                    # Add missing columns with 0 values
                    for col in feature_order:
                        if col not in input_df_encoded.columns:
                            input_df_encoded[col] = 0
                    
                    # Reorder columns to match training order
                    input_df_encoded = input_df_encoded[feature_order]
                
                # ====== STEP 5: Make Prediction ======
                prediction = model.predict(input_df_encoded)[0]
                
                # Get prediction probabilities if available
                prediction_proba = None
                if hasattr(model, 'predict_proba'):
                    prediction_proba = model.predict_proba(input_df_encoded)[0]
                
                # Decode prediction
                if label_encoder is not None:
                    try:
                        decoded_prediction = label_encoder.inverse_transform([int(prediction)])[0]
                    except:
                        decoded_prediction = prediction
                else:
                    decoded_prediction = prediction
                
                # ====== STEP 6: Display Results ======
                st.markdown("---")
                st.markdown("### 🎉 **Prediction Results**")
                
                col1, col2, col3 = st.columns(3)
                
                # Determine status
                status_lower = str(decoded_prediction).lower()
                
                with col1:
                    if 'eligible' in status_lower and 'not' not in status_lower:
                        st.success(f"### ✅ {decoded_prediction}")
                        st.markdown("**Status:** Approved")
                    elif 'high' in status_lower or 'risk' in status_lower:
                        st.warning(f"### ⚠️ {decoded_prediction}")
                        st.markdown("**Status:** Conditional")
                    else:
                        st.error(f"### ❌ {decoded_prediction}")
                        st.markdown("**Status:** Not Approved")
                
                with col2:
                    if prediction_proba is not None:
                        confidence = np.max(prediction_proba) * 100
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                    st.metric("Credit Score", credit_score)
                
                with col3:
                    net_income = monthly_salary - (input_df['total_expenses'].iloc[0])
                    st.metric("Net Monthly Income", f"₹{net_income:,.0f}")
                    risk_ratio = input_df['expense_to_income_ratio'].iloc[0] * 100
                    st.metric("Expense Ratio", f"{risk_ratio:.1f}%")
                
                # Additional Insights
                if prediction_proba is not None:
                    st.markdown("#### 📊 **Prediction Probabilities**")
                    proba_df = pd.DataFrame({
                        'Class': ['Eligible', 'High Risk', 'Not Eligible'],
                        'Probability': prediction_proba * 100
                    })
                    st.bar_chart(proba_df.set_index('Class'))
                
                # Financial Summary
                st.markdown("#### 💡 **Financial Summary**")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    - **Monthly Salary:** ₹{monthly_salary:,}
                    - **Total Expenses:** ₹{input_df['total_expenses'].iloc[0]:,.0f}
                    - **Net Income:** ₹{net_income:,.0f}
                    - **Expense Ratio:** {risk_ratio:.1f}%
                    """)
                
                with col2:
                    st.markdown(f"""
                    - **Credit Score:** {credit_score}
                    - **Bank Balance:** ₹{bank_balance:,}
                    - **Emergency Fund:** ₹{emergency_fund:,}
                    - **Existing Loans:** {existing_loans}
                    """)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.write("**Debug Info:**")
                st.write(f"Error: {type(e).__name__}: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    render()