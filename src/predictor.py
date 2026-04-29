import joblib
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HeartDiseasePredictor:
    def __init__(self, model_path=None, 
                 scaler_path=None,
                 imputer_path=None,
                 features_path=None):
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'models', 'model.pkl')
        if scaler_path is None:
            scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl')
        if imputer_path is None:
            imputer_path = os.path.join(PROJECT_ROOT, 'models', 'imputer.pkl')
        if features_path is None:
            features_path = os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl')
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        self.feature_names = joblib.load(features_path)
        
    def predict(self, patient_data):
        patient_array = np.array(patient_data).reshape(1, -1)
        patient_imputed = self.imputer.transform(patient_array)
        patient_scaled = self.scaler.transform(patient_imputed)
        
        prediction = self.model.predict(patient_scaled)[0]
        probability = self.model.predict_proba(patient_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'label': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'confidence': {
                'no_disease': round(probability[0] * 100, 1),
                'disease': round(probability[1] * 100, 1)
            }
        }
    
    def get_feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return None


FEATURE_INFO = {
    'age': {'name': 'Age', 'unit': 'years', 'range': '29-77'},
    'sex': {'name': 'Sex', 'unit': '', 'options': '1=Male, 0=Female'},
    'cp': {'name': 'Chest Pain Type', 'unit': '', 'options': '1=Typical, 2=Atypical, 3=Non-anginal, 4=Asymptomatic'},
    'trestbps': {'name': 'Resting Blood Pressure', 'unit': 'mmHg', 'range': '94-200'},
    'chol': {'name': 'Cholesterol', 'unit': 'mg/dl', 'range': '126-564'},
    'fbs': {'name': 'Fasting Blood Sugar > 120', 'unit': '', 'options': '1=Yes, 0=No'},
    'restecg': {'name': 'Resting ECG', 'unit': '', 'options': '0=Normal, 1=ST-T wave, 2=Left ventricular'},
    'thalach': {'name': 'Max Heart Rate', 'unit': 'bpm', 'range': '71-202'},
    'exang': {'name': 'Exercise Induced Angina', 'unit': '', 'options': '1=Yes, 0=No'},
    'oldpeak': {'name': 'ST Depression', 'unit': 'mm', 'range': '0-6.2'},
    'slope': {'name': 'ST Segment Slope', 'unit': '', 'options': '1=Up, 2=Flat, 3=Down'},
    'ca': {'name': 'Major Vessels', 'unit': '', 'range': '0-4'},
    'thal': {'name': 'Thalassemia', 'unit': '', 'options': '3=Normal, 6=Fixed, 7=Reversible'}
}


def get_default_values():
    return {
        'age': 55,
        'sex': 1,
        'cp': 2,
        'trestbps': 130,
        'chol': 250,
        'fbs': 0,
        'restecg': 1,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 1.0,
        'slope': 2,
        'ca': 1,
        'thal': 7
    }
