# EMIPredict AI - Main Application Entry Point
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="EMIPredict AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI/UX
st.markdown("""
<style>
    /* Main Theme Colors */
    :root {
        --primary-color: #6C63FF;
        --secondary-color: #4CAF50;
        --danger-color: #FF6B6B;
        --warning-color: #FFD93D;
        --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        opacity: 0.95;
        margin-bottom: 0.5rem;
    }
    
    /* Card Styles */
    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border-left: 5px solid #667eea;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.2);
    }
    
    .card-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    .card-description {
        color: #666;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Stats Section */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .stat-box {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
        color: #666;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <div class="hero-title">💰 EMIPredict AI</div>
    <div class="hero-subtitle">Intelligent Financial Risk Assessment Platform</div>
    <p>Powered by Advanced Machine Learning | 400,000+ Financial Profiles Analyzed</p>
</div>
""", unsafe_allow_html=True)

# Introduction Section
col1, col2 = st.columns([2, 1])
with col1:
    st.markdown("### 🎯 **Welcome to EMIPredict AI**")
    st.markdown("""
    EMIPredict AI is a cutting-edge platform that leverages machine learning to provide:
    
    - **🎯 Smart Eligibility Assessment**: Classify loan applicants into Eligible, High Risk, or Not Eligible
    - **💵 Accurate EMI Prediction**: Calculate maximum affordable monthly EMI based on financial capacity
    - **📊 Real-time Analysis**: Instant risk assessment using 22+ financial parameters
    - **🔬 MLflow Integration**: Comprehensive model tracking and experiment management
    - **📈 Data-Driven Insights**: Built on 400,000 realistic financial profiles
    """)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h2 style="color: #667eea;">🚀</h2>
        <h3>Get Started</h3>
        <p style="color: #666;">Navigate using the sidebar to explore different modules</p>
    </div>
    """, unsafe_allow_html=True)

# Stats Section
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">400K+</div>
        <div class="stat-label">Financial Records</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">22+</div>
        <div class="stat-label">Features Analyzed</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">6+</div>
        <div class="stat-label">ML Models Trained</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-number">90%+</div>
        <div class="stat-label">Accuracy Rate</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Feature Cards
st.markdown("### 🎨 **Platform Features**")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="card-icon">🎯</div>
        <div class="card-title">Eligibility Prediction</div>
        <div class="card-description">
            Advanced classification models to determine loan eligibility status. 
            Uses XGBoost, Random Forest, and Logistic Regression for accurate predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="card-icon">💵</div>
        <div class="card-title">EMI Amount Prediction</div>
        <div class="card-description">
            Regression models calculate the maximum monthly EMI based on comprehensive 
            financial analysis and risk assessment.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="card-icon">📊</div>
        <div class="card-title">Data Analytics</div>
        <div class="card-description">
            Explore interactive visualizations, EDA insights, and model performance 
            metrics tracked through MLflow.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="card-icon">🔬</div>
        <div class="card-title">MLflow Tracking</div>
        <div class="card-description">
            Complete experiment tracking with model versioning, parameter logging, 
            and performance comparison across all trained models.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="card-icon">💾</div>
        <div class="card-title">Data Management</div>
        <div class="card-description">
            Full CRUD operations for financial records. Add, update, delete, and 
            manage loan application data efficiently.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="card-icon">🎓</div>
        <div class="card-title">Model Insights</div>
        <div class="card-description">
            Detailed feature importance analysis, SHAP values, and comprehensive 
            model evaluation metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)

# How It Works Section
st.markdown("---")
st.markdown("### 🔄 **How It Works**")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem; color: #667eea;">1️⃣</div>
        <h4>Input Data</h4>
        <p style="color: #666;">Enter financial and demographic details</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem; color: #667eea;">2️⃣</div>
        <h4>Feature Engineering</h4>
        <p style="color: #666;">Advanced preprocessing and feature creation</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem; color: #667eea;">3️⃣</div>
        <h4>ML Prediction</h4>
        <p style="color: #666;">Models analyze and predict outcomes</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="font-size: 3rem; color: #667eea;">4️⃣</div>
        <h4>Get Results</h4>
        <p style="color: #666;">Instant eligibility and EMI recommendations</p>
    </div>
    """, unsafe_allow_html=True)

# Technology Stack
st.markdown("---")
st.markdown("### 💻 **Technology Stack**")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🐍 Machine Learning**
    - Scikit-learn
    - XGBoost
    - LightGBM
    - Feature Engineering
    """)

with col2:
    st.markdown("""
    **📊 Data & Analytics**
    - Pandas & NumPy
    - Matplotlib & Seaborn
    - Plotly for Interactive Viz
    - MLflow for Tracking
    """)

with col3:
    st.markdown("""
    **🚀 Deployment**
    - Streamlit Framework
    - Streamlit Cloud
    - GitHub Integration
    - Real-time Predictions
    """)

# Call to Action
st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 2rem; 
            border-radius: 15px; 
            text-align: center; 
            color: white;">
    <h2>🚀 Ready to Get Started?</h2>
    <p style="font-size: 1.1rem; margin-bottom: 1rem;">
        Navigate through the sidebar to explore different modules and start making predictions!
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p><strong>EMIPredict AI Platform</strong></p>
    <p>Built with ❤️ using Streamlit, MLflow, and Advanced Machine Learning</p>
    <p style="font-size: 0.85rem; color: #999;">
        © 2025 EMIPredict AI | Intelligent Financial Risk Assessment
    </p>
</div>
""", unsafe_allow_html=True)
