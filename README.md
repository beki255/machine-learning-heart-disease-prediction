# Heart Disease Prediction System

An AI-powered Streamlit app for heart disease prediction using 4 ML models with mean imputation preprocessing.

## Features
- Login System (Admin/Doctor roles)
- 13 Medical Features Input
- 4 ML Models (Random Forest, Logistic Regression, SVM, Decision Tree)
- 10-Fold Cross-Validation
- Mean Imputation (keeps all data rows)
- PDF/HTML Reports
- Prediction Explainability (Top 5 features)
- Batch Prediction (CSV upload)
- Modern Violet UI with enlarged fonts

## Tech Stack
Python, Streamlit, Scikit-learn, SHAP, Pandas, Plotly

## How to Run
1. pip install -r requirements.txt
2. python src/train_model.py  (CRITICAL - generates imputer.pkl)
3. streamlit run app.py

## Login
- admin / admin123 (full access)
- doctor / doctor123 (prediction + reports)

## Project Structure
- app.py - Main app (1266 lines)
- preprocess.ipynb - Teaching notebook
- src/train_model.py - Model training
- src/predictor.py - Prediction class
- data/heartt_cleveland_cleaned.csv - Dataset
- data/photo_2026-04-29_21-43-55.jpg - Background image

## New Features
- PDF/HTML Report Download
- Key Prediction Factors Display
- Batch Prediction Tab
- Welcome Page with Background Image
- Fonts Increased (between original and previous)
- Mean Imputation (replaces dropna())

## Important
**Must run 'python src/train_model.py' first** to generate imputer.pkl file!

Built with for educational purposes
