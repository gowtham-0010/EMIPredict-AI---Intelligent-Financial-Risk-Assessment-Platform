# Model Training Script with MLflow Integration

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, mean_squared_error,
                             mean_absolute_error, r2_score)
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import os
import json
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class EMIPredictTrainer:
    """
    Complete ML training pipeline for EMI Prediction with MLflow integration
    """

    def __init__(self, data_path, experiment_name="EMIPredict_AI"):
        """
        Initialize trainer
        Args:
            data_path: Path to the EMI dataset CSV
            experiment_name: Name for MLflow experiment
        """
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train_clf = None
        self.y_test_clf = None
        self.y_train_reg = None
        self.y_test_reg = None
        self.label_encoder = LabelEncoder()
        # Setup directories
        self.setup_directories()
        # Setup MLflow
        mlflow.set_experiment(self.experiment_name)

    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'models/classification',
            'models/regression',
            'mlruns',
            'artifacts'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def load_and_explore_data(self):
        """Load dataset and perform initial exploration"""
        print(" Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"\n Dataset loaded successfully!")
        print(f"   Shape: {self.df.shape}")
        print(f"   Features: {self.df.shape[1]}")
        print(f"   Records: {self.df.shape[0]}")

        print("\n Dataset Info:")
        print(self.df.info())

        print("\n Target Variable Distribution:")
        print(self.df['emi_eligibility'].value_counts())

        print("\n EMI Amount Statistics:")
        print(self.df['max_monthly_emi'].describe())

        return self.df

    def preprocess_data(self):
        """Data preprocessing and feature engineering"""
        print("\n Starting data preprocessing...")

        # Fix: Convert all numeric columns to float (clean up commas or spaces)
        numeric_columns = [
            'age', 'monthly_salary', 'years_of_employment', 'monthly_rent', 'family_size',
            'dependents', 'school_fees', 'college_fees', 'travel_expenses', 'groceries_utilities',
            'other_monthly_expenses', 'current_emi_amount', 'credit_score', 'bank_balance',
            'emergency_fund', 'requested_amount', 'requested_tenure', 'max_monthly_emi'
        ]
        for col in numeric_columns:
            if col in self.df.columns:
                # Remove commas and spaces, convert to float, errors to NaN
                self.df[col] = pd.to_numeric(self.df[col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

        # Handle missing values
        self.df = self.df.fillna(0)

        print("   Creating derived features...")

        # Total monthly expenses
        self.df['total_expenses'] = (
            self.df['monthly_rent'] +
            self.df['school_fees'] +
            self.df['college_fees'] +
            self.df['travel_expenses'] +
            self.df['groceries_utilities'] +
            self.df['other_monthly_expenses'] +
            self.df['current_emi_amount']
        )

        # Net monthly income
        self.df['net_monthly_income'] = self.df['monthly_salary'] - self.df['total_expenses']

        # Expense to income ratio
        self.df['expense_to_income_ratio'] = self.df['total_expenses'] / (self.df['monthly_salary'] + 1e-9)

        # Dependency ratio
        self.df['dependency_ratio'] = self.df['dependents'] / (self.df['family_size'] + 1e-9)

        # Risk score calculation
        self.df['risk_score'] = (
            (1 - (self.df['credit_score'] - 300) / 600) * 0.7 +
            (self.df['current_emi_amount'] / (self.df['monthly_salary'] + 1e-9)) * 0.3
        )
        self.df['risk_score'] = self.df['risk_score'].clip(0, 1)

        # Loan to income ratio
        self.df['loan_to_income_ratio'] = self.df['requested_amount'] / ((self.df['monthly_salary'] * 12) + 1e-9)

        # Savings capacity
        self.df['savings_capacity'] = self.df['bank_balance'] + self.df['emergency_fund']

        # Encode existing_loans if it's text
        if self.df['existing_loans'].dtype == 'object' or str(self.df['existing_loans'].dtype).startswith("string"):
            self.df['existing_loans'] = self.df['existing_loans'].map({'Yes': 1, 'No': 0}).fillna(0)

        print("    Feature engineering completed!")

        # Encode categorical variables
        categorical_columns = ['gender', 'marital_status', 'education', 'employment_type',
                               'company_type', 'house_type', 'emi_scenario']

        print("   Encoding categorical variables...")
        self.df_encoded = pd.get_dummies(self.df, columns=categorical_columns, drop_first=False)

        print(f"    Features after encoding: {self.df_encoded.shape[1]}")

        return self.df_encoded

    def prepare_train_test_split(self):
        """Split data into train and test sets"""
        print("\n  Splitting data into train and test sets...")

        # Features for modeling (exclude target variables)
        exclude_cols = ['emi_eligibility', 'max_monthly_emi']
        feature_cols = [col for col in self.df_encoded.columns if col not in exclude_cols]

        X = self.df_encoded[feature_cols]

        # Classification target
        y_clf = self.df['emi_eligibility']
        self.label_encoder.fit(y_clf)
        y_clf_encoded = self.label_encoder.transform(y_clf)

        # Regression target
        y_reg = self.df['max_monthly_emi']

        # Train-test split
        self.X_train, self.X_test, self.y_train_clf, self.y_test_clf, self.y_train_reg, self.y_test_reg = \
            train_test_split(X, y_clf_encoded, y_reg, test_size=0.2, random_state=42, stratify=y_clf_encoded)

        print(f"    Training set: {self.X_train.shape[0]} samples")
        print(f"    Test set: {self.X_test.shape[0]} samples")

        # Save label encoder
        joblib.dump(self.label_encoder, 'models/classification/label_encoder.pkl')
        print("    Label encoder saved!")

        # Save feature names
        with open('models/classification/feature_order.json', 'w') as f:
            json.dump(list(X.columns), f)
        print("    Feature order saved!")

        return self.X_train, self.X_test

    def train_classification_models(self):
        """Train multiple classification models with MLflow tracking"""
        print("\n Training Classification Models...")
        print("="*60)

        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost Classifier': XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
        }

        best_model = None
        best_score = 0

        for model_name, model in models.items():
            print(f"\n Training {model_name}...")

            with mlflow.start_run(run_name=f"Classification_{model_name}"):
                # Train model
                model.fit(self.X_train, self.y_train_clf)

                # Predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None

                # Metrics
                accuracy = accuracy_score(self.y_test_clf, y_pred)
                precision = precision_score(self.y_test_clf, y_pred, average='weighted', zero_division=0)
                recall = recall_score(self.y_test_clf, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(self.y_test_clf, y_pred, average='weighted', zero_division=0)

                # Log parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())

                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                # Log model
                if 'XGBoost' in model_name:
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")

                # Print metrics
                print(f"    Accuracy: {accuracy:.4f}")
                print(f"    Precision: {precision:.4f}")
                print(f"    Recall: {recall:.4f}")
                print(f"    F1-Score: {f1:.4f}")

                # Track best model
                if accuracy > best_score:
                    best_score = accuracy
                    best_model = (model_name, model)

                # Save model
                model_filename = model_name.lower().replace(' ', '_') + '.pkl'
                joblib.dump(model, f'models/classification/{model_filename}')
                print(f"    Model saved: {model_filename}")

        print(f"\n Best Classification Model: {best_model[0]} (Accuracy: {best_score:.4f})")

        return best_model

    def train_regression_models(self):
        """Train multiple regression models with MLflow tracking"""
        print("\n Training Regression Models...")
        print("="*60)

        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost Regressor': XGBRegressor(n_estimators=100, random_state=42)
        }

        best_model = None
        best_score = float('inf')

        for model_name, model in models.items():
            print(f"\n Training {model_name}...")

            with mlflow.start_run(run_name=f"Regression_{model_name}"):
                # Train model
                model.fit(self.X_train, self.y_train_reg)

                # Predictions
                y_pred = model.predict(self.X_test)

                # Metrics
                rmse = np.sqrt(mean_squared_error(self.y_test_reg, y_pred))
                mae = mean_absolute_error(self.y_test_reg, y_pred)
                r2 = r2_score(self.y_test_reg, y_pred)
                mape = np.mean(np.abs((self.y_test_reg - y_pred) / (self.y_test_reg + 1e-9))) * 100

                # Log parameters
                if hasattr(model, 'get_params'):
                    mlflow.log_params(model.get_params())

                # Log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mape", mape)

                # Log model
                if 'XGBoost' in model_name:
                    mlflow.xgboost.log_model(model, "model")
                else:
                    mlflow.sklearn.log_model(model, "model")

                # Print metrics
                print(f"    RMSE: {rmse:.2f}")
                print(f"    MAE: {mae:.2f}")
                print(f"    R² Score: {r2:.4f}")
                print(f"    MAPE: {mape:.2f}%")

                # Track best model
                if rmse < best_score:
                    best_score = rmse
                    best_model = (model_name, model)

                # Save model
                model_filename = model_name.lower().replace(' ', '_') + '.pkl'
                joblib.dump(model, f'models/regression/{model_filename}')
                print(f"    Model saved: {model_filename}")

        print(f"\n Best Regression Model: {best_model[0]} (RMSE: {best_score:.2f})")
        return best_model

    def train_all_models(self):
        """Complete training pipeline"""
        print("\n" + "="*60)
        print(" EMIPredict AI - Model Training Pipeline")
        print("="*60)

        # Load and explore data
        self.load_and_explore_data()

        # Preprocess data
        self.preprocess_data()

        # Prepare train-test split
        self.prepare_train_test_split()

        # Train classification models
        best_clf = self.train_classification_models()

        # Train regression models
        best_reg = self.train_regression_models()

        print("\n" + "="*60)
        print(" Training Completed Successfully!")
        print("="*60)
        print(f"\n Summary:")
        print(f"   Best Classification Model: {best_clf[0]}")
        print(f"   Best Regression Model: {best_reg[0]}")
        print(f"\n Models saved in: models/ directory")
        print(f" MLflow tracking: mlruns/ directory")
        print(f"\n Run 'mlflow ui' to view experiment tracking")

        return best_clf, best_reg


# Main execution
if __name__ == "__main__":
    # Path to your dataset
    DATA_PATH = "data/EMI_dataset.csv"  # dataset path

    # Initialize and run trainer
    trainer = EMIPredictTrainer(DATA_PATH)
    best_classification, best_regression = trainer.train_all_models()
