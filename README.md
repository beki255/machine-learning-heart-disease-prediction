# Heart Disease Prediction System

A machine learning application to predict heart disease for hospital/clinic use using the Cleveland Heart Disease dataset.

## Overview

This is a real-world heart disease prediction system that uses machine learning to predict whether a patient has heart disease based on 13 medical features.

## Features

- **Real-time Prediction**: Instant heart disease detection
- **Statistics Dashboard**: View prediction statistics
- **History Tracking**: Keep track of all predictions
- **Export to CSV**: Save predictions for records
- **User-friendly GUI**: Easy to use interface

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python app.py
```

## Buttons

| Button | Function |
|--------|----------|
| Predict | Make heart disease prediction |
| Stats | Open statistics dashboard |
| Clear | Reset patient form |
| Export | Save history to CSV |
| Clear History | Clear prediction table |

## Project Structure

```
ML PROJECT/
├── app.py                 # Main GUI application
├── src/
│   ├── predictor.py      # ML prediction module
│   └── analyze.py      # Data analysis
├── models/              # Trained model files
├── data/
│   └── heartt_cleveland_cleaned.csv
├── *.png               # Analysis charts
└── README.md
```

## Dataset

14 medical features:
- Age, Sex
- Chest Pain Type (1-4)
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Max Heart Rate
- Exercise Induced Angina
- ST Depression
- ST Segment Slope
- Major Vessels (0-4)
- Thalassemia

## Model Performance

- Random Forest: ~87% accuracy
- SVM: ~85% accuracy
- Logistic Regression: ~83% accuracy
- ROC-AUC: ~95%