# Data Management and CRUD Operations Module
import streamlit as st
import pandas as pd
import os
from datetime import datetime

def render():
    # Page Header
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                padding: 2rem; 
                border-radius: 15px; 
                color: white; 
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
        <h1 style="margin: 0; font-size: 2.5rem;">💾 Data Management</h1>
        <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem; opacity: 0.9;">
            Manage financial records with full CRUD operations
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data file path
    DATA_FILE = "financial_records.csv"
    
    # Initialize session state for data
    if 'data' not in st.session_state:
        if os.path.exists(DATA_FILE):
            st.session_state.data = pd.read_csv(DATA_FILE)
        else:
            # Create empty dataframe with schema
            st.session_state.data = pd.DataFrame(columns=[
                'id', 'age', 'gender', 'marital_status', 'education', 'monthly_salary',
                'employment_type', 'years_of_employment', 'company_type', 'house_type',
                'monthly_rent', 'family_size', 'dependents', 'school_fees', 'college_fees',
                'travel_expenses', 'groceries_utilities', 'other_monthly_expenses',
                'existing_loans', 'current_emi_amount', 'credit_score', 'bank_balance',
                'emergency_fund', 'emi_scenario', 'requested_amount', 'requested_tenure',
                'created_at', 'updated_at'
            ])
    
    # Tabs for CRUD operations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 View Data", "➕ Add Record", "✏️ Update Record", "🗑️ Delete Record", "📥 Import/Export"])
    
    # Tab 1: View Data
    with tab1:
        st.markdown("### 📊 **Financial Records Database**")
        
        if len(st.session_state.data) > 0:
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(st.session_state.data))
            with col2:
                avg_salary = st.session_state.data['monthly_salary'].mean() if 'monthly_salary' in st.session_state.data.columns else 0
                st.metric("Avg Salary", f"₹{avg_salary:,.0f}")
            with col3:
                avg_credit = st.session_state.data['credit_score'].mean() if 'credit_score' in st.session_state.data.columns else 0
                st.metric("Avg Credit Score", f"{avg_credit:.0f}")
            with col4:
                total_requested = st.session_state.data['requested_amount'].sum() if 'requested_amount' in st.session_state.data.columns else 0
                st.metric("Total Requested", f"₹{total_requested:,.0f}")
            
            st.markdown("---")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'emi_scenario' in st.session_state.data.columns:
                    scenarios = ['All'] + list(st.session_state.data['emi_scenario'].unique())
                    selected_scenario = st.selectbox("Filter by EMI Scenario", scenarios)
            
            with col2:
                if 'employment_type' in st.session_state.data.columns:
                    emp_types = ['All'] + list(st.session_state.data['employment_type'].unique())
                    selected_emp = st.selectbox("Filter by Employment", emp_types)
            
            with col3:
                search_term = st.text_input("🔍 Search", placeholder="Search by ID or name...")
            
            # Apply filters
            filtered_data = st.session_state.data.copy()
            
            if selected_scenario != 'All' and 'emi_scenario' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['emi_scenario'] == selected_scenario]
            
            if selected_emp != 'All' and 'employment_type' in filtered_data.columns:
                filtered_data = filtered_data[filtered_data['employment_type'] == selected_emp]
            
            if search_term:
                filtered_data = filtered_data[
                    filtered_data.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
                ]
            
            # Display data
            st.dataframe(filtered_data, use_container_width=True, height=400)
            
            # Download button
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Filtered Data",
                data=csv,
                file_name=f"emi_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.info("📂 No records found. Add your first record in the 'Add Record' tab.")
    
    # Tab 2: Add Record
    with tab2:
        st.markdown("### ➕ **Add New Financial Record**")
        
        with st.form("add_record_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                age = st.number_input("Age", 18, 70, 30)
                gender = st.selectbox("Gender", ["Male", "Female"])
                marital_status = st.selectbox("Marital Status", ["Single", "Married"])
                education = st.selectbox("Education", ["High School", "Graduate", "Post Graduate", "Professional"])
            
            with col2:
                monthly_salary = st.number_input("Monthly Salary", 10000, 500000, 50000)
                employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-employed"])
                years_of_employment = st.number_input("Years of Employment", 0.0, 40.0, 5.0, 0.5)
                company_type = st.selectbox("Company Type", ["Startup", "Mid-size", "MNC", "Large Indian"])
            
            with col3:
                house_type = st.selectbox("House Type", ["Own", "Rented", "Family"])
                monthly_rent = st.number_input("Monthly Rent", 0, 100000, 0)
                family_size = st.number_input("Family Size", 1, 10, 3)
                dependents = st.number_input("Dependents", 0, 8, 1)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                school_fees = st.number_input("School Fees", 0, 50000, 0)
                college_fees = st.number_input("College Fees", 0, 100000, 0)
                travel_expenses = st.number_input("Travel Expenses", 0, 50000, 3000)
            
            with col2:
                groceries_utilities = st.number_input("Groceries & Utilities", 0, 50000, 10000)
                other_monthly_expenses = st.number_input("Other Expenses", 0, 50000, 5000)
                current_emi_amount = st.number_input("Current EMI", 0, 200000, 0)
            
            with col3:
                credit_score = st.slider("Credit Score", 300, 900, 650)
                bank_balance = st.number_input("Bank Balance", 0, 10000000, 100000)
                emergency_fund = st.number_input("Emergency Fund", 0, 5000000, 50000)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])
            with col2:
                emi_scenario = st.selectbox("EMI Scenario", 
                    ["Home Appliances EMI", "Vehicle EMI", "Education EMI", "Personal Loan EMI", "E-commerce Shopping EMI"])
            with col3:
                requested_amount = st.number_input("Requested Amount", 10000, 5000000, 200000)
                requested_tenure = st.number_input("Requested Tenure (months)", 6, 360, 60)
            
            submitted = st.form_submit_button("➕ Add Record", use_container_width=True)
        
        if submitted:
            # Generate new ID
            if len(st.session_state.data) > 0 and 'id' in st.session_state.data.columns:
                new_id = st.session_state.data['id'].max() + 1
            else:
                new_id = 1
            
            # Create new record
            new_record = {
                'id': new_id,
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
                'existing_loans': existing_loans,
                'current_emi_amount': current_emi_amount,
                'credit_score': credit_score,
                'bank_balance': bank_balance,
                'emergency_fund': emergency_fund,
                'emi_scenario': emi_scenario,
                'requested_amount': requested_amount,
                'requested_tenure': requested_tenure,
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Add to dataframe
            st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_record])], ignore_index=True)
            
            # Save to file
            st.session_state.data.to_csv(DATA_FILE, index=False)
            
            st.success(f"✅ Record added successfully! ID: {new_id}")
            st.balloons()
    
    # Tab 3: Update Record
    with tab3:
        st.markdown("### ✏️ **Update Existing Record**")
        
        if len(st.session_state.data) > 0 and 'id' in st.session_state.data.columns:
            record_ids = st.session_state.data['id'].tolist()
            selected_id = st.selectbox("Select Record ID to Update", record_ids)
            
            if selected_id:
                record = st.session_state.data[st.session_state.data['id'] == selected_id].iloc[0]
                
                with st.form("update_record_form"):
                    st.markdown(f"**Updating Record ID: {selected_id}**")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        age = st.number_input("Age", 18, 70, int(record['age']))
                        monthly_salary = st.number_input("Monthly Salary", 10000, 500000, int(record['monthly_salary']))
                    
                    with col2:
                        credit_score = st.slider("Credit Score", 300, 900, int(record['credit_score']))
                        bank_balance = st.number_input("Bank Balance", 0, 10000000, int(record['bank_balance']))
                    
                    with col3:
                        requested_amount = st.number_input("Requested Amount", 10000, 5000000, int(record['requested_amount']))
                        requested_tenure = st.number_input("Tenure", 6, 360, int(record['requested_tenure']))
                    
                    updated = st.form_submit_button("💾 Update Record", use_container_width=True)
                
                if updated:
                    # Update record
                    st.session_state.data.loc[st.session_state.data['id'] == selected_id, 'age'] = age
                    st.session_state.data.loc[st.session_state.data['id'] == selected_id, 'monthly_salary'] = monthly_salary
                    st.session_state.data.loc[st.session_state.data['id'] == selected_id, 'credit_score'] = credit_score
                    st.session_state.data.loc[st.session_state.data['id'] == selected_id, 'bank_balance'] = bank_balance
                    st.session_state.data.loc[st.session_state.data['id'] == selected_id, 'requested_amount'] = requested_amount
                    st.session_state.data.loc[st.session_state.data['id'] == selected_id, 'requested_tenure'] = requested_tenure
                    st.session_state.data.loc[st.session_state.data['id'] == selected_id, 'updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Save to file
                    st.session_state.data.to_csv(DATA_FILE, index=False)
                    
                    st.success(f"✅ Record {selected_id} updated successfully!")
        else:
            st.info("📂 No records available to update.")
    
    # Tab 4: Delete Record
    with tab4:
        st.markdown("### 🗑️ **Delete Record**")
        
        if len(st.session_state.data) > 0 and 'id' in st.session_state.data.columns:
            record_ids = st.session_state.data['id'].tolist()
            selected_id = st.selectbox("Select Record ID to Delete", record_ids, key="delete_id")
            
            if selected_id:
                record = st.session_state.data[st.session_state.data['id'] == selected_id].iloc[0]
                
                st.warning(f"⚠️ You are about to delete Record ID: {selected_id}")
                st.json(record.to_dict())
                
                col1, col2, col3 = st.columns([1, 1, 2])
                with col1:
                    if st.button("🗑️ Confirm Delete", type="primary"):
                        st.session_state.data = st.session_state.data[st.session_state.data['id'] != selected_id]
                        st.session_state.data.to_csv(DATA_FILE, index=False)
                        st.success(f"✅ Record {selected_id} deleted successfully!")
                        st.rerun()
        else:
            st.info("📂 No records available to delete.")
    
    # Tab 5: Import/Export
    with tab5:
        st.markdown("### 📥 **Import/Export Data**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📤 Export Data")
            st.info("Download current database as CSV file")
            
            if len(st.session_state.data) > 0:
                csv = st.session_state.data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Complete Database",
                    data=csv,
                    file_name=f"emi_database_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.warning("No data to export")
        
        with col2:
            st.markdown("#### 📤 Import Data")
            st.info("Upload CSV file to import records")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    new_data = pd.read_csv(uploaded_file)
                    st.success(f"✅ File loaded: {len(new_data)} records")
                    
                    if st.button("🔄 Import Data", use_container_width=True):
                        st.session_state.data = new_data
                        st.session_state.data.to_csv(DATA_FILE, index=False)
                        st.success("✅ Data imported successfully!")
                        st.rerun()
                except Exception as e:
                    st.error(f"❌ Error loading file: {str(e)}")

if __name__ == "__main__":
    render()
