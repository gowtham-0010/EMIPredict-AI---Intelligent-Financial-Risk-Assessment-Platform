# EMI Amount Regression Module
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import json

def render():
    # Page Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                color: white; 
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h1 style="margin: 0; font-size: 2.5rem;">💵 EMI Amount Predictor</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Calculate maximum affordable monthly EMI using regression models
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Configuration
    MODEL_PATHS = {
        "XGBoost Regressor": "models/regression/xgboost_regressor.pkl",
        "Random Forest Regressor": "models/regression/random_forest_regressor.pkl",
        "Linear Regression": "models/regression/linear_regression.pkl"
    }
    
    FEATURE_ORDER_PATH = "models/classification/feature_order.json"  # Same as classification
    
    # Model Selection
    st.markdown("### 🤖 **Select Regression Model**")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Choose a model for EMI prediction:",
            list(MODEL_PATHS.keys()),
            help="Different models provide varying prediction accuracies"
        )
    
    with col2:
        st.info(f"**Selected:** {selected_model}")
    
    # Load Model
    model_path = MODEL_PATHS[selected_model]
    
    if not os.path.exists(model_path):
        st.error(f"⚠️ Model file not found: {model_path}")
        st.info("Please ensure models are trained and saved in the correct directory.")
        return
    
    try:
        model = joblib.load(model_path)
        
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
    st.markdown("### 📝 **Enter Financial Profile**")
    
    with st.form("emi_prediction_form"):
        # Personal Information
        st.markdown("#### 👤 Personal Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=70, value=35)
        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"], key="gender_reg")
        with col3:
            marital_status = st.selectbox("Marital Status", ["Single", "Married"], key="marital_reg")
        with col4:
            education = st.selectbox("Education", 
                ["High School", "Graduate", "Post Graduate", "Professional"], key="education_reg")
        
        st.markdown("#### 💼 Employment & Income")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            monthly_salary = st.number_input("Monthly Salary (₹)", 
                min_value=15000, max_value=500000, value=75000, step=5000)
        with col2:
            employment_type = st.selectbox("Employment Type", 
                ["Private", "Government", "Self-employed"], key="employment_reg")
        with col3:
            years_of_employment = st.number_input("Years of Employment", 
                min_value=0.0, max_value=40.0, value=8.0, step=0.5)
        with col4:
            company_type = st.selectbox("Company Type", 
                ["Startup", "Mid-size", "MNC", "Large Indian"], key="company_reg")
        
        st.markdown("#### 🏠 Housing & Living")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            house_type = st.selectbox("House Type", ["Own", "Rented", "Family"], key="house_reg")
        with col2:
            monthly_rent = st.number_input("Monthly Rent (₹)", 
                min_value=0, max_value=100000, value=0, step=1000)
        with col3:
            family_size = st.number_input("Family Size", 
                min_value=1, max_value=10, value=4)
        with col4:
            dependents = st.number_input("Dependents", 
                min_value=0, max_value=8, value=2)
        
        st.markdown("#### 💰 Financial Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            credit_score = st.slider("Credit Score", 300, 900, 700)
        with col2:
            bank_balance = st.number_input("Bank Balance (₹)", 
                min_value=0, max_value=10000000, value=150000, step=10000)
        with col3:
            emergency_fund = st.number_input("Emergency Fund (₹)", 
                min_value=0, max_value=5000000, value=80000, step=10000)
        with col4:
            existing_loans = st.selectbox("Existing Loans", ["Yes", "No"], key="loans_reg")
        
        st.markdown("#### 📊 Monthly Obligations")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            school_fees = st.number_input("School Fees (₹)", 
                min_value=0, max_value=50000, value=0, step=500)
        with col2:
            college_fees = st.number_input("College Fees (₹)", 
                min_value=0, max_value=100000, value=0, step=1000)
        with col3:
            travel_expenses = st.number_input("Travel Expenses (₹)", 
                min_value=0, max_value=50000, value=5000, step=500)
        with col4:
            groceries_utilities = st.number_input("Groceries & Utilities (₹)", 
                min_value=0, max_value=50000, value=12000, step=500)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            other_monthly_expenses = st.number_input("Other Expenses (₹)", 
                min_value=0, max_value=50000, value=8000, step=500)
        with col2:
            current_emi_amount = st.number_input("Current EMI (₹)", 
                min_value=0, max_value=200000, value=0, step=1000)
        with col3:
            requested_amount = st.number_input("Requested Loan Amount (₹)", 
                min_value=10000, max_value=5000000, value=500000, step=10000)
        with col4:
            requested_tenure = st.number_input("Requested Tenure (months)", 
                min_value=6, max_value=360, value=84, step=6)
        
        st.markdown("#### 📋 Loan Type")
        emi_scenario = st.selectbox("EMI Scenario", 
            ["Vehicle EMI", "Home Appliances EMI", "Education EMI", 
             "Personal Loan EMI", "E-commerce Shopping EMI"], key="scenario_reg")
        
        # Submit Button
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("💵 Calculate Maximum EMI", use_container_width=True)
    
    # Prediction Logic
    if submitted:
        with st.spinner("🔄 Calculating affordable EMI amount..."):
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
                predicted_emi = model.predict(input_df_encoded)[0]
                
                # Calculate additional metrics
                total_expenses = input_df['total_expenses'].iloc[0]
                net_income = monthly_salary - total_expenses
                available_for_emi = net_income * 0.40  # 40% of net income
                
                # Calculate loan details
                monthly_interest_rate = 0.10 / 12  # Assuming 10% annual interest
                n_months = requested_tenure
                
                # EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)
                if monthly_interest_rate > 0:
                    emi_for_requested = (requested_amount * monthly_interest_rate * 
                                        (1 + monthly_interest_rate)**n_months) / \
                                       ((1 + monthly_interest_rate)**n_months - 1)
                else:
                    emi_for_requested = requested_amount / n_months
                
                # ====== STEP 6: Display Results ======
                st.markdown("---")
                st.markdown("### 🎉 **EMI Prediction Results**")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                padding: 2rem; 
                                border-radius: 12px; 
                                color: white; 
                                text-align: center;
                                box-shadow: 0 5px 15px rgba(0,0,0,0.2);">
                        <h4 style="margin: 0;">Maximum Affordable EMI</h4>
                        <h2 style="margin: 0.5rem 0; font-size: 2rem;">₹{:,.0f}</h2>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">per month</p>
                    </div>
                    """.format(predicted_emi), unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                                padding: 2rem; 
                                border-radius: 12px; 
                                color: white; 
                                text-align: center;
                                box-shadow: 0 5px 15px rgba(0,0,0,0.2);">
                        <h4 style="margin: 0;">Requested EMI</h4>
                        <h2 style="margin: 0.5rem 0; font-size: 2rem;">₹{:,.0f}</h2>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">per month</p>
                    </div>
                    """.format(emi_for_requested), unsafe_allow_html=True)
                
                with col3:
                    affordability = (predicted_emi / emi_for_requested) * 100 if emi_for_requested > 0 else 100
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                                padding: 2rem; 
                                border-radius: 12px; 
                                color: white; 
                                text-align: center;
                                box-shadow: 0 5px 15px rgba(0,0,0,0.2);">
                        <h4 style="margin: 0;">Affordability Index</h4>
                        <h2 style="margin: 0.5rem 0; font-size: 2rem;">{:.1f}%</h2>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">of requested EMI</p>
                    </div>
                    """.format(affordability), unsafe_allow_html=True)
                
                # Decision Indicator
                st.markdown("#### 🎯 **Loan Assessment**")
                
                if predicted_emi >= emi_for_requested:
                    st.success("""
                    ✅ **APPROVED** - You can comfortably afford the requested loan amount.
                    
                    The predicted maximum EMI (₹{:,.0f}) exceeds the required EMI (₹{:,.0f}) 
                    for your requested loan amount.
                    """.format(predicted_emi, emi_for_requested))
                else:
                    st.warning("""
                    ⚠️ **REVIEW REQUIRED** - The requested loan amount may strain your finances.
                    
                    Consider reducing the loan amount or increasing the tenure. 
                    Maximum recommended EMI: ₹{:,.0f} | Required EMI: ₹{:,.0f}
                    """.format(predicted_emi, emi_for_requested))
                
                # Financial Breakdown
                st.markdown("#### 💡 **Financial Breakdown**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Income & Expenses**")
                    breakdown_data = {
                        'Category': ['Monthly Salary', 'Total Expenses', 'Net Income', 
                                   'Available for EMI (40%)', 'Predicted Max EMI'],
                        'Amount (₹)': [monthly_salary, total_expenses, net_income, 
                                      available_for_emi, predicted_emi]
                    }
                    breakdown_df = pd.DataFrame(breakdown_data)
                    st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**Loan Details**")
                    loan_data = {
                        'Parameter': ['Requested Amount', 'Tenure', 'Interest Rate (Est.)', 
                                    'Required EMI', 'Total Payment'],
                        'Value': [f'₹{requested_amount:,}', f'{requested_tenure} months', 
                                '10.0%', f'₹{emi_for_requested:,.0f}', 
                                f'₹{emi_for_requested * requested_tenure:,.0f}']
                    }
                    loan_df = pd.DataFrame(loan_data)
                    st.dataframe(loan_df, use_container_width=True, hide_index=True)
                
                # Recommendations
                st.markdown("#### 📋 **Recommendations**")
                
                if predicted_emi >= emi_for_requested:
                    st.info("""
                    **Suggestions for Optimal Loan Management:**
                    - ✓ Your financial profile supports this loan
                    - Consider setting up auto-debit for timely payments
                    - Maintain emergency fund equal to 6 months EMI
                    - Review insurance coverage for loan protection
                    """)
                else:
                    # Calculate alternative scenarios
                    max_affordable_loan = (predicted_emi * ((1 + monthly_interest_rate)**n_months - 1)) / \
                                         (monthly_interest_rate * (1 + monthly_interest_rate)**n_months)
                    
                    st.info(f"""
                    **Alternative Options:**
                    - 💰 Reduce loan amount to ₹{max_affordable_loan:,.0f}
                    - ⏱️ Extend tenure to {int((requested_amount / predicted_emi) * 1.1)} months
                    - 💳 Improve credit score for better terms
                    - 📊 Reduce monthly expenses by ₹{(emi_for_requested - predicted_emi):,.0f}
                    """)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.write("**Debug Info:**")
                st.write(f"Error: {type(e).__name__}: {str(e)}")
                st.exception(e)

if __name__ == "__main__":
    render()