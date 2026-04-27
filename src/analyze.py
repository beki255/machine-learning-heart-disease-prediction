import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_curve, auc, confusion_matrix,
                           classification_report, roc_auc_score)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_data():
    data_path = os.path.join(PROJECT_ROOT, 'data', 'heartt_cleveland_cleaned.csv')
    df = pd.read_csv(data_path)
    return df


def analyze_data(df):
    print("="*60)
    print("DATA ANALYSIS")
    print("="*60)
    
    df_clean = df.dropna()
    
    print(f"\nDataset Shape: {df_clean.shape}")
    print(f"\nTarget Distribution:")
    print(df_clean['target'].value_counts())
    print(f"\nClass Balance: {df_clean['target'].value_counts(normalize=True).round(3).to_dict()}")
    
    print(f"\n--- Statistical Summary ---")
    print(df_clean.describe().round(2))
    
    print(f"\n--- Correlation with Target ---")
    correlations = df_clean.corr()['target'].sort_values(ascending=False)
    print(correlations)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_clean.corr(), annot=True, cmap='RdBu_r', center=0, 
                fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=150)
    plt.close()
    print("\nSaved: correlation_heatmap.png")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(df_clean[df_clean['target']==0]['age'], bins=20, alpha=0.7, label='No Disease', color='green')
    axes[0, 0].hist(df_clean[df_clean['target']==1]['age'], bins=20, alpha=0.7, label='Disease', color='red')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Age Distribution')
    axes[0, 0].legend()
    
    axes[0, 1].hist(df_clean[df_clean['target']==0]['chol'], bins=20, alpha=0.7, label='No Disease', color='green')
    axes[0, 1].hist(df_clean[df_clean['target']==1]['chol'], bins=20, alpha=0.7, label='Disease', color='red')
    axes[0, 1].set_xlabel('Cholesterol')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Cholesterol Distribution')
    axes[0, 1].legend()
    
    axes[1, 0].hist(df_clean[df_clean['target']==0]['thalach'], bins=20, alpha=0.7, label='No Disease', color='green')
    axes[1, 0].hist(df_clean[df_clean['target']==1]['thalach'], bins=20, alpha=0.7, label='Disease', color='red')
    axes[1, 0].set_xlabel('Max Heart Rate')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Max Heart Rate Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].hist(df_clean[df_clean['target']==0]['oldpeak'], bins=20, alpha=0.7, label='No Disease', color='green')
    axes[1, 1].hist(df_clean[df_clean['target']==1]['oldpeak'], bins=20, alpha=0.7, label='Disease', color='red')
    axes[1, 1].set_xlabel('ST Depression')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('ST Depression Distribution')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('distributions.png', dpi=150)
    plt.close()
    print("Saved: distributions.png")


def train_with_tuning():
    print("\n" + "="*60)
    print("MODEL TRAINING WITH HYPERPARAMETER TUNING")
    print("="*60)
    
    df = load_data()
    df = df.dropna()
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {}
    
    print("\n1. Random Forest (with GridSearchCV)...")
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    rf = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
    rf.fit(X_train_scaled, y_train)
    models['Random Forest'] = rf.best_estimator_
    print(f"   Best params: {rf.best_params_}")
    
    print("\n2. Logistic Regression...")
    lr_params = {'C': [0.01, 0.1, 1, 10]}
    lr = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), lr_params, cv=5)
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr.best_estimator_
    print(f"   Best params: {lr.best_params_}")
    
    print("\n3. SVM...")
    svm_params = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
    svm = GridSearchCV(SVC(kernel='rbf', probability=True, random_state=42), svm_params, cv=5)
    svm.fit(X_train_scaled, y_train)
    models['SVM'] = svm.best_estimator_
    print(f"   Best params: {svm.best_params_}")
    
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'model': model,
            'y_prob': y_prob,
            'y_pred': y_pred
        }
        
        print(f"\n{name}:")
        print(f"  Accuracy:  {results[name]['accuracy']:.4f}")
        print(f"  Precision: {results[name]['precision']:.4f}")
        print(f"  Recall:   {results[name]['recall']:.4f}")
        print(f"  F1 Score: {results[name]['f1']:.4f}")
        print(f"  ROC-AUC:  {results[name]['roc_auc']:.4f}")
    
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best = results[best_model_name]
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {best['accuracy']:.4f}, ROC-AUC: {best['roc_auc']:.4f}")
    print("="*60)
    
    print(f"\nClassification Report ({best_model_name}):")
    print(classification_report(y_test, best['y_pred']))
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, best['y_pred'])
    print(cm)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    plt.title(f'{best_model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print("\nSaved: confusion_matrix.png")
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green']
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        plt.plot(fpr, tpr, color=color, label=f'{name} (AUC = {res["roc_auc"]:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=150)
    plt.close()
    print("Saved: roc_curves.png")
    
    if hasattr(best['model'], 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': best['model'].feature_importances_
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'], color='steelblue')
        plt.xlabel('Importance')
        plt.title(f'{best_model_name} - Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150)
        plt.close()
        print("Saved: feature_importance.png")
    
    plt.figure(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    values = [best['accuracy'], best['precision'], best['recall'], best['f1'], best['roc_auc']]
    bars = plt.bar(metrics, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title(f'{best_model_name} - Performance Metrics')
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('metrics.png', dpi=150)
    plt.close()
    print("Saved: metrics.png")
    
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(best['model'], os.path.join(models_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    joblib.dump(X.columns.tolist(), os.path.join(models_dir, 'feature_names.pkl'))
    
    print("\nModel saved to models/")
    
    return best_model_name, results


if __name__ == "__main__":
    df = load_data()
    analyze_data(df)
    train_with_tuning()
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - correlation_heatmap.png")
    print("  - distributions.png")
    print("  - confusion_matrix.png")
    print("  - roc_curves.png")
    print("  - feature_importance.png")
    print("  - metrics.png")