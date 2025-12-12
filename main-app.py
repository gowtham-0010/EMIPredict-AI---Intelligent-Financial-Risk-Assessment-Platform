# EMIPredict AI - Complete Streamlit Application with Navigation
import streamlit as st
from pathlib import Path
import sys

# Add pages to path
sys.path.append(str(Path(__file__).parent))

# Page configuration
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/EMIPredict-AI',
        'Report a bug': 'https://github.com/yourusername/EMIPredict-AI/issues',
        'About': '# EMIPredict AI\n\nIntelligent Financial Risk Assessment Platform'
    }
)

# Import page modules
try:
    # Since we're using a multi-page structure, import render functions
    import importlib.util
    
    def load_page_module(page_name):
        """Dynamically load page modules"""
        module_path = Path(__file__).parent / f"{page_name}.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location(page_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None
    
except Exception as e:
    st.error(f"Error loading modules: {e}")

# Sidebar Navigation with Enhanced UI
st.sidebar.markdown("""
<div style="text-align: center; 
            padding: 1.5rem; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            border-radius: 15px; 
            color: white; 
            margin-bottom: 2rem;
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);">
    <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">💰 EMIPredict AI</h1>
    <p style="margin: 0.5rem 0 0 0; font-size: 1rem; opacity: 0.95; font-weight: 500;">
        Financial Risk Assessment
    </p>
</div>
""", unsafe_allow_html=True)

# Navigation Menu - UPDATED (Removed Analytics & MLflow pages)
page = st.sidebar.radio(
    "📍 Navigation",
    [
        "🏠 Home",
        "🎯 Eligibility Predictor",
        "💵 EMI Amount Predictor",
        "💾 Data Management",
        "ℹ️ About"
    ]
)

# Sidebar Information
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 **Platform Insights**")

# Enhanced stats display
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
            padding: 1rem; 
            border-radius: 10px; 
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 10px rgba(240, 147, 251, 0.3);">
    <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;">Total Records</div>
    <div style="font-size: 1.8rem; font-weight: 700;">400K+</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
            padding: 1rem; 
            border-radius: 10px; 
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 10px rgba(79, 172, 254, 0.3);">
    <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;">ML Models</div>
    <div style="font-size: 1.8rem; font-weight: 700;">6+</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
            padding: 1rem; 
            border-radius: 10px; 
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 4px 10px rgba(67, 233, 123, 0.3);">
    <div style="font-size: 0.85rem; opacity: 0.9; margin-bottom: 0.3rem;">Accuracy Rate</div>
    <div style="font-size: 1.8rem; font-weight: 700;">90%+</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 **Quick Guide**")
st.sidebar.markdown("""
<div style="background: rgba(102, 126, 234, 0.1); 
            padding: 1rem; 
            border-radius: 10px; 
            border-left: 4px solid #667eea;">
    <div style="font-size: 0.9rem; line-height: 1.6;">
        <strong>1. Eligibility Check</strong><br/>
        Predict loan approval status<br/><br/>
        <strong>2. EMI Calculation</strong><br/>
        Get maximum affordable EMI<br/><br/>
        <strong>3. Data Management</strong><br/>
        Manage financial records
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 **Support**")
st.sidebar.markdown("""
<div style="font-size: 0.85rem;">
    <a href="https://github.com/gowtham-0010" style="text-decoration: none; color: #667eea;">
        📖 Documentation
    </a><br/>
    <a href="https://github.com/gowtham-0010" style="text-decoration: none; color: #667eea;">
        🐛 Report Bug
    </a><br/>
    <a href="https://github.com/gowtham-0010" style="text-decoration: none; color: #667eea;">
        💡 Feature Request
    </a>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; font-size: 0.75rem; color: #888; line-height: 1.5;">
    <p style="margin: 0.5rem 0;">Built with ❤️ using</p>
    <p style="margin: 0.3rem 0;"><strong>Streamlit • XGBoost • scikit-learn</strong></p>
    <p style="margin: 0.5rem 0; font-size: 0.7rem;">© 2025 EMIPredict AI</p>
</div>
""", unsafe_allow_html=True)

# Page Routing
if page == "🏠 Home":
    # Enhanced Home Page Content
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem 2rem; 
                border-radius: 20px; 
                color: white; 
                text-align: center; 
                margin-bottom: 2.5rem; 
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);">
        <div style="font-size: 3.5rem; font-weight: 800; margin-bottom: 1rem; 
                    text-shadow: 2px 2px 8px rgba(0,0,0,0.2);">
            💰 EMIPredict AI
        </div>
        <div style="font-size: 1.4rem; opacity: 0.95; margin-bottom: 0.8rem; font-weight: 500;">
            Intelligent Financial Risk Assessment Platform
        </div>
        <div style="font-size: 1rem; opacity: 0.85;">
            Powered by Advanced Machine Learning | 400,000+ Financial Profiles Analyzed
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 🎯 **Welcome to EMIPredict AI**")
        st.markdown("""
        EMIPredict AI is a cutting-edge platform that leverages machine learning to provide:
        
        - **🎯 Smart Eligibility Assessment**: Classify loan applicants into Eligible, High Risk, or Not Eligible
        - **💵 Accurate EMI Prediction**: Calculate maximum affordable monthly EMI based on financial capacity
        - **📊 Real-time Analysis**: Instant risk assessment using 22+ financial parameters
        - **💾 Data Management**: Complete CRUD operations for financial records
        - **📈 Proven Accuracy**: Built on 400,000 realistic financial profiles with 90%+ accuracy
        """)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 2rem; 
                    border-radius: 15px; 
                    color: white;
                    text-align: center;
                    box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);">
            <h2 style="margin: 0; font-size: 3rem;">🚀</h2>
            <h3 style="margin: 1rem 0 0.5rem 0;">Get Started</h3>
            <p style="margin: 0; font-size: 0.95rem; opacity: 0.9;">
                Navigate using the sidebar to explore different modules
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Statistics Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.8rem; 
                    border-radius: 15px; 
                    color: white; 
                    text-align: center;
                    box-shadow: 0 6px 15px rgba(102, 126, 234, 0.3);">
            <div style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.3rem;">400K+</div>
            <div style="font-size: 0.95rem; opacity: 0.9;">Financial Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.8rem; 
                    border-radius: 15px; 
                    color: white; 
                    text-align: center;
                    box-shadow: 0 6px 15px rgba(240, 147, 251, 0.3);">
            <div style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.3rem;">22+</div>
            <div style="font-size: 0.95rem; opacity: 0.9;">Features Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.8rem; 
                    border-radius: 15px; 
                    color: white; 
                    text-align: center;
                    box-shadow: 0 6px 15px rgba(79, 172, 254, 0.3);">
            <div style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.3rem;">6+</div>
            <div style="font-size: 0.95rem; opacity: 0.9;">ML Models Trained</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.8rem; 
                    border-radius: 15px; 
                    color: white; 
                    text-align: center;
                    box-shadow: 0 6px 15px rgba(67, 233, 123, 0.3);">
            <div style="font-size: 2.8rem; font-weight: 800; margin-bottom: 0.3rem;">90%+</div>
            <div style="font-size: 0.95rem; opacity: 0.9;">Accuracy Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🎨 **Platform Features**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; 
                    padding: 2.5rem; 
                    border-radius: 15px; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.08); 
                    border-left: 5px solid #667eea; 
                    height: 100%;
                    transition: transform 0.3s ease;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">🎯</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #333; margin-bottom: 0.8rem;">
                Eligibility Prediction
            </div>
            <div style="color: #666; font-size: 1rem; line-height: 1.7;">
                Advanced classification models to determine loan eligibility status. 
                Uses XGBoost, Random Forest, and Logistic Regression for accurate predictions.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; 
                    padding: 2.5rem; 
                    border-radius: 15px; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.08); 
                    border-left: 5px solid #f093fb; 
                    height: 100%;
                    transition: transform 0.3s ease;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">💵</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #333; margin-bottom: 0.8rem;">
                EMI Amount Prediction
            </div>
            <div style="color: #666; font-size: 1rem; line-height: 1.7;">
                Regression models calculate the maximum monthly EMI based on comprehensive 
                financial analysis and risk assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; 
                    padding: 2.5rem; 
                    border-radius: 15px; 
                    box-shadow: 0 10px 25px rgba(0,0,0,0.08); 
                    border-left: 5px solid #4facfe; 
                    height: 100%;
                    transition: transform 0.3s ease;">
            <div style="font-size: 3.5rem; margin-bottom: 1rem;">💾</div>
            <div style="font-size: 1.6rem; font-weight: 700; color: #333; margin-bottom: 0.8rem;">
                Data Management
            </div>
            <div style="color: #666; font-size: 1rem; line-height: 1.7;">
                Complete CRUD operations for financial records with import/export functionality 
                and real-time statistics dashboard.
            </div>
        </div>
        """, unsafe_allow_html=True)

elif page == "🎯 Eligibility Predictor":
    module = load_page_module("classification-page")
    if module and hasattr(module, 'render'):
        module.render()
    else:
        st.error("Classification page module not found. Please ensure 'classification-page.py' exists.")

elif page == "💵 EMI Amount Predictor":
    module = load_page_module("regression-page")
    if module and hasattr(module, 'render'):
        module.render()
    else:
        st.error("Regression page module not found. Please ensure 'regression-page.py' exists.")

elif page == "💾 Data Management":
    module = load_page_module("data-management-page")
    if module and hasattr(module, 'render'):
        module.render()
    else:
        st.error("Data management page module not found. Please ensure 'data-management-page.py' exists.")

elif page == "ℹ️ About":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 2.5rem 2rem; 
                border-radius: 20px; 
                color: white; 
                margin-bottom: 2.5rem;
                box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);">
        <h1 style="margin: 0; font-size: 2.8rem; font-weight: 700;">ℹ️ About EMIPredict AI</h1>
        <p style="margin: 0.8rem 0 0 0; font-size: 1.15rem; opacity: 0.95;">
            Project Information and Documentation
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📖 **Project Overview**")
    st.markdown("""
    **EMIPredict AI** is a comprehensive financial risk assessment platform built to address 
    the critical issue of poor financial planning and inadequate risk assessment in EMI-based lending.
    
    The platform delivers:
    - **Dual ML Problem Solving**: Classification (eligibility) and Regression (EMI amount)
    - **Real-time Assessment**: Using 400,000 financial records
    - **Advanced Features**: 22 financial and demographic variables + 8 engineered features
    - **Production Ready**: Streamlit deployment with CRUD operations
    """)
    
    st.markdown("### 🎯 **Business Use Cases**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    color: white;
                    margin-bottom: 1rem;">
            <h4 style="margin: 0 0 1rem 0;">Financial Institutions</h4>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                <li>Automate loan approval (80% time reduction)</li>
                <li>Risk-based pricing strategies</li>
                <li>Real-time eligibility assessment</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    color: white;">
            <h4 style="margin: 0 0 1rem 0;">FinTech Companies</h4>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                <li>Instant EMI eligibility checks</li>
                <li>Mobile app integration</li>
                <li>Automated risk scoring</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    color: white;
                    margin-bottom: 1rem;">
            <h4 style="margin: 0 0 1rem 0;">Banks & Credit Agencies</h4>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                <li>Data-driven loan recommendations</li>
                <li>Portfolio risk management</li>
                <li>Regulatory compliance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 1.5rem; 
                    border-radius: 12px; 
                    color: white;">
            <h4 style="margin: 0 0 1rem 0;">Loan Officers</h4>
            <ul style="margin: 0; padding-left: 1.2rem; line-height: 1.8;">
                <li>AI-powered recommendations</li>
                <li>Comprehensive profile analysis</li>
                <li>Performance tracking</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 💻 **Technology Stack**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🐍 Machine Learning**
        - Scikit-learn
        - XGBoost
        - Pandas & NumPy
        - Feature Engineering
        """)
    
    with col2:
        st.markdown("""
        **📊 Visualization**
        - Matplotlib
        - Seaborn
        - Plotly
        - Streamlit Charts
        """)
    
    with col3:
        st.markdown("""
        **🚀 Deployment**
        - Streamlit
        - Streamlit Cloud
        - GitHub Integration
        - Real-time Predictions
        """)
    
    st.markdown("### 📊 **Model Performance**")
    st.success("""
    **Classification Models:**
    - Expected Accuracy: >90%
    - Best Model: XGBoost Classifier
    
    **Regression Models:**
    - Expected RMSE: <2000 INR
    - Best Model: XGBoost Regressor
    """)
    
    st.markdown("### 📞 **Contact & Support**")
    st.info("""
    - **GitHub**: [EMIPredict-AI Repository](https://github.com/gowtham-0010)
    - **Documentation**: [Project Wiki](https://github.com/gowtham-0010/wiki)
    - **Issues**: [Report Bug]https://github.com/gowtham-0010/issues)
    - **Email**: sgowtham1009@gmail.com
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p style="margin: 0.5rem 0;"><strong>EMIPredict AI Platform</strong></p>
        <p style="margin: 0.5rem 0;">Built with ❤️ using Streamlit, XGBoost, and Advanced Machine Learning</p>
        <p style="margin: 0.5rem 0; font-size: 0.9rem;">© 2025 EMIPredict AI | Intelligent Financial Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)