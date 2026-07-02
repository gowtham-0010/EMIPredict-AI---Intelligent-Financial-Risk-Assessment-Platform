# EMIPredict AI - Intelligent Financial Risk Assessment Platform

[![Streamlit App](https://img.shields.io/badge/Live%20App-Streamlit%20Cloud-ff4b4b?logo=streamlit)](https://dzbekmhtwqsmbnh9bpyrzs.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/gowtham-0010/EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform)

EMIPredict AI is an end-to-end machine learning web app that predicts loan eligibility and estimates maximum affordable EMI using financial and demographic inputs. The project is deployed on Streamlit Cloud and includes classification and regression workflows, feature engineering, model training, and an interactive user interface [web:62][web:64].

## Live Demo
- **Streamlit App:** [EMIPredict AI](https://dzbekmhtwqsmbnh9bpyrzs.streamlit.app/)
- **GitHub Repository:** [EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform](https://github.com/gowtham-0010/EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform)

## Features
- Loan eligibility prediction using classification models.
- EMI estimation using regression models.
- Feature engineering from raw financial records.
- Interactive Streamlit dashboard with multiple pages.
- Model training pipeline with saved artifacts.
- Deployed cloud application for real-time predictions.

## Tech Stack
- Python
- Streamlit
- scikit-learn
- XGBoost
- pandas
- NumPy
- joblib

## Project Structure
- `app.py` / `main-app.py` - Main Streamlit application.
- `classification-page.py` - Eligibility prediction page.
- `regression-page.py` - EMI prediction page.
- `data-management-page.py` - Data management page.
- `train-models.py` - Model training pipeline.
- `financial_records.csv` - Financial dataset used for training.
- `models/` - Saved trained model artifacts.

## How It Works
1. Loads financial dataset from CSV.
2. Cleans and preprocesses raw inputs.
3. Engineers derived financial features.
4. Trains classification and regression models.
5. Serves predictions through a Streamlit interface.

## Installation
```bash
git clone https://github.com/gowtham-0010/EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform.git
cd EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform
pip install -r requirements.txt
streamlit run main-app.py
```

## Resume Highlights
- Built a classification model predicting EMI eligibility on a financial dataset.
- Engineered financial features from raw user and loan data.
- Evaluated multiple ML algorithms and deployed the best-performing models.
- Deployed a Streamlit app for real-time financial risk assessment.

## License
This project is for educational and demonstration purposes.
