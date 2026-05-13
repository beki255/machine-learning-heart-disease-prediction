import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'heartt_cleveland_cleaned.csv')
    df = pd.read_csv(data_path)
    return df


def train_models():
    print("Loading data...")
    df = load_data()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Mean imputation instead of dropping rows
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    print("\n" + "="*60)
    print("MODEL TRAINING & EVALUATION")
    print("="*60)
    
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10)
        
        results[name] = {
            'accuracy': acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"  Test Accuracy: {acc:.4f}")
        print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test_scaled)
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    print("="*60)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_best))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(cm)
    
    
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(models_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(imputer, os.path.join(models_dir, 'imputer.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, 'feature_names.pkl'))
    
    print("\nModel saved to: models/model.pkl")
    print("Scaler saved to: models/scaler.pkl")
    print("Imputer saved to: models/imputer.pkl")
    
    return best_model_name, results
if __name__ == "__main__":
    train_models()
