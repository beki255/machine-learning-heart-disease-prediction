import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime
import json

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = os.path.abspath(".")

USERS_FILE = os.path.join(PROJECT_ROOT, "data", "users.json")

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {
        'admin': {'password': 'admin123', 'role': 'admin'},
        'doctor': {'password': 'doctor123', 'role': 'doctor'}
    }

def save_users(users):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

USERS = load_users()

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%); }
    html, body { font-size: 16px; }
    h1 { font-size: 34px; font-weight: bold; }
    h2 { font-size: 28px; font-weight: bold; }
    h3 { font-size: 24px; font-weight: bold; color: #c0392b; }
    .stTextInput label, .stNumberInput label, .stSelectbox label { 
        font-size: 20px; font-weight: 700; color: #1a1a2e; 
    }
    .stButton button { 
        font-size: 18px; padding: 14px 28px; border-radius: 10px; font-weight: 600;
    }
    .stAlert { font-size: 20px; border-radius: 12px; padding: 20px; }
    [data-testid="stMetricValue"] { font-size: 28px; }
    [data-testid="stMetricLabel"] { font-size: 16px; }
    .stDataFrame { font-size: 16px; }
    .stTabs [data-testid="stTabBarButton"] { font-size: 18px; padding: 12px 20px; }
    .header-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px; border-radius: 16px; margin-bottom: 20px;
    }
    .card {
        background: white; border-radius: 16px; padding: 24px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin: 8px 0;
    }
    .stat-card {
        background: white; border-radius: 12px; padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1); text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'model.pkl'))
    scaler = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl'))
    return model, scaler, feature_names


@st.cache_resource
def load_all_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'heartt_cleveland_cleaned.csv'))
    df = df.dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=10, scoring='accuracy')
        test_acc = model.score(X_test_scaled, y_test)
        results[name] = {'model': model, 'accuracy': test_acc, 'cv_accuracy': cv_scores.mean()}
    
    return results, scaler, X.columns.tolist()


FEATURE_LABELS = {
    'age': 'Age', 'sex': 'Sex', 'cp': 'Chest Pain Type',
    'trestbps': 'Blood Pressure', 'chol': 'Cholesterol',
    'fbs': 'Fasting Blood Sugar', 'restecg': 'Resting ECG',
    'thalach': 'Max Heart Rate', 'exang': 'Exercise Angina',
    'oldpeak': 'ST Depression', 'slope': 'ST Slope',
    'ca': 'Major Vessels', 'thal': 'Thalassemia'
}

FEATURE_OPTIONS = {
    'sex': {'Male': 1, 'Female': 0},
    'cp': {'Typical Angina': 1, 'Atypical Angina': 2, 'Non-anginal': 3, 'Asymptomatic': 4},
    'fbs': {'<= 120 mg/dl': 0, '> 120 mg/dl': 1},
    'restecg': {'Normal': 0, 'ST-T wave': 1, 'Left ventricular': 2},
    'exang': {'No': 0, 'Yes': 1},
    'slope': {'Up': 1, 'Flat': 2, 'Down': 3},
    'ca': {'0': 0, '1': 1, '2': 2, '3': 3},
    'thal': {'Normal': 3, 'Fixed': 6, 'Reversible': 7}
}


if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'role' not in st.session_state:
    st.session_state.role = ""
if 'history' not in st.session_state:
    st.session_state.history = []
if 'patient_counter' not in st.session_state:
    st.session_state.patient_counter = 0


def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="header-banner">
            <h1 style="color: white; margin: 0;">Heart Disease Prediction</h1>
            <p style="color: white; opacity: 0.9; margin: 10px 0 0 0;">
                AI-Powered Detection System
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Login Required")
        
        username = st.text_input("Username", placeholder="Enter username")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        
        if st.button("Login", type="primary", width='stretch'):
            if username in USERS and USERS[username]['password'] == password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.role = USERS[username]['role']
                st.rerun()
            else:
                st.error("Invalid credentials!")
        
        st.markdown("---")
        st.caption("Admin: admin/admin123 | Doctor: doctor/doctor123")


if not st.session_state.logged_in:
    login()
    st.stop()


def predict_single(data):
    model, scaler, feature_names = load_model()
    data_array = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data_array)
    
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0]
    
    return int(pred), prob[0] * 100, prob[1] * 100


def predict_all_models(data):
    models_dict, scaler, feature_names = load_all_models()
    data_array = np.array(data).reshape(1, -1)
    data_scaled = scaler.transform(data_array)
    
    results = {}
    for name, info in models_dict.items():
        model = info['model']
        pred = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]
        results[name] = {
            'prediction': 'Heart Disease' if pred == 1 else 'No Disease',
            'confidence': round(max(prob) * 100, 1),
            'disease_prob': round(prob[1] * 100, 1),
            'accuracy': info['accuracy'],
            'cv_accuracy': info['cv_accuracy']
        }
    
    return results


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: #e74c3c; margin: 0;">Heart</h2>
        <p style="color: #7f8c8d; margin: 5px 0;">Disease AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    role_badge = "Admin" if st.session_state.role == 'admin' else "Doctor"
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 12px; border-radius: 12px; color: white; text-align: center;">
        <div style="font-size: 12px; opacity: 0.8;">Logged in as</div>
        <div style="font-size: 16px; font-weight: bold;">{st.session_state.username}</div>
        <div style="font-size: 14px;">{role_badge}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.role == 'admin':
        with st.expander("Admin Panel"):
            st.markdown("### Manage Users")
            
            users_list = list(USERS.keys())
            selected = st.selectbox("Select User", users_list)
            
            if selected:
                st.write(f"**Role:** {USERS[selected]['role']}")
            
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password", type="password")
            new_role = st.selectbox("Role", ["doctor"])
            
            if st.button("Add User"):
                if new_user and new_pass:
                    if new_user in USERS:
                        st.error("User already exists!")
                    else:
                        USERS[new_user] = {'password': new_pass, 'role': new_role}
                        save_users(USERS)
                        st.success(f"User {new_user} added!")
                        st.rerun()
                else:
                    st.error("Fill all fields!")
            
            if selected and selected != 'admin':
                if st.button("Remove User"):
                    del USERS[selected]
                    save_users(USERS)
                    st.success(f"User {selected} removed!")
                    st.rerun()


# Main
st.markdown("# Heart Disease Prediction System")

if st.session_state.role == 'admin':
    tabs = st.tabs(["Users", "Predict", "Compare", "Stats", "Features", "History"])
    tab_u, tab1, tab2, tab3, tab4, tab5 = tabs
else:
    tabs = st.tabs(["Predict", "Compare", "Stats", "Features", "History"])
    tab1, tab2, tab3, tab4, tab5 = tabs

# User Management (Admin only)
if st.session_state.role == 'admin':
    with tab_u:
        st.markdown("## User Management")
        
        st.markdown("### All Users")
        users_data = []
        for user, info in USERS.items():
            users_data.append({
                'Username': user,
                'Role': info['role'],
                'Password': info['password']
            })
        st.dataframe(pd.DataFrame(users_data), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Add New User")
        
        c1, c2 = st.columns(2)
        with c1:
            new_user = st.text_input("Username")
        with c2:
            new_pass = st.text_input("Password", type="password")
        
        new_role = st.selectbox("Role", ["doctor"])
        
        if st.button("Add User"):
            if new_user and new_pass:
                if new_user in USERS:
                    st.error("User already exists!")
                else:
                    USERS[new_user] = {'password': new_pass, 'role': new_role}
                    save_users(USERS)
                    st.success(f"User {new_user} added!")
                    st.rerun()
            else:
                st.error("Fill all fields!")
        
        st.markdown("---")
        st.markdown("### Remove User")
        
        rem_user = st.selectbox("Select user to remove", [u for u in USERS.keys() if u != 'admin'])
        
        if st.button("Remove User"):
            if rem_user:
                del USERS[rem_user]
                save_users(USERS)
                st.success(f"User {rem_user} removed!")
                st.rerun()


# Prediction Tab
with tab1:
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown('<div class="card"><h3>Patient Information</h3></div>', unsafe_allow_html=True)
        patient_id = f"P{str(st.session_state.patient_counter + 1).zfill(4)}"
        patient_name = st.text_input("Patient Name", placeholder="Enter name")
        age = st.number_input("Age", min_value=1, max_value=120, value=55, key="age_pred")
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True, key="sex_pred")
    
    with c2:
        st.markdown('<div class="card"><h3>Medical Features</h3></div>', unsafe_allow_html=True)
        
        c_bp, c_chol = st.columns(2)
        with c_bp:
            trestbps = st.number_input("Blood Pressure (mmHg)", 94, 200, 130, key="bp_pred")
        with c_chol:
            chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 250, key="chol_pred")
        
        c_cp, c_fbs = st.columns(2)
        with c_cp:
            cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"], key="cp_pred")
        with c_fbs:
            fbs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"], key="fbs_pred")
        
        c_ecg, c_hr = st.columns(2)
        with c_ecg:
            restecg = st.selectbox("Resting ECG", ["Normal", "ST-T wave", "Left ventricular"], key="ecg_pred")
        with c_hr:
            thalach = st.number_input("Max Heart Rate", 60, 220, 150, key="hr_pred")
        
        c_ex, c_old = st.columns(2)
        with c_ex:
            exang = st.selectbox("Exercise Angina", ["No", "Yes"], key="ex_pred")
        with c_old:
            oldpeak = st.number_input("ST Depression", 0.0, 10.0, 1.0, 0.1, key="old_pred")
        
        c_slope, c_ca = st.columns(2)
        with c_slope:
            slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key="slope_pred")
        with c_ca:
            ca = st.selectbox("Major Vessels", ["0", "1", "2", "3"], key="ca_pred")
        
        c_thal = st.selectbox("Thalassemia", ["Normal", "Fixed", "Reversible"], key="thal_pred")

    st.markdown("---")
    
    if st.button("PREDICT", type="primary", width='stretch'):
        sex_val = 1 if sex == "Male" else 0
        cp_val = FEATURE_OPTIONS['cp'][cp]
        fbs_val = FEATURE_OPTIONS['fbs'][fbs]
        restecg_val = FEATURE_OPTIONS['restecg'][restecg]
        exang_val = FEATURE_OPTIONS['exang'][exang]
        slope_val = FEATURE_OPTIONS['slope'][slope]
        ca_val = FEATURE_OPTIONS['ca'][ca]
        thal_val = FEATURE_OPTIONS['thal'][c_thal]
        
        data = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca_val, thal_val]
        
        pred, prob_no, prob_yes = predict_single(data)
        
        rec = {
            'id': patient_id,
            'name': patient_name,
            'age': age,
            'sex': sex,
            'trestbps': trestbps,
            'chol': chol,
            'cp': cp,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': c_thal,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'doctor': st.session_state.username,
            'prediction': 'Heart Disease' if pred == 1 else 'No Disease',
            'confidence': round(prob_yes if pred == 1 else prob_no, 1)
        }
        st.session_state.history.append(rec)
        st.session_state.patient_counter += 1
        
        if pred == 1:
            st.error(f"**HEART DISEASE DETECTED**\n\nConfidence: {rec['confidence']}%")
        else:
            st.success(f"**NO HEART DISEASE**\n\nConfidence: {rec['confidence']}%")


# Compare Tab
with tab2:
    st.markdown("## Model Comparison")
    st.markdown("Compare predictions from 8 different ML models")
    
    st.markdown("### Enter Patient Features")
    
    c1, c2 = st.columns(2)
    with c1:
        age_c = st.number_input("Age", 55, key="age_cmp")
        sex_c = st.radio("Sex", ["Male", "Female"], key="sex_cmp")
        cp_c = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"], key="cp_cmp")
        trestbps_c = st.number_input("Blood Pressure", 130, key="bp_cmp")
        chol_c = st.number_input("Cholesterol", 250, key="chol_cmp")
    with c2:
        fbs_c = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"], key="fbs_cmp")
        restecg_c = st.selectbox("Resting ECG", ["Normal", "ST-T wave", "Left ventricular"], key="ecg_cmp")
        thalach_c = st.number_input("Max Heart Rate", 150, key="hr_cmp")
        exang_c = st.selectbox("Exercise Angina", ["No", "Yes"], key="ex_cmp")
        oldpeak_c = st.number_input("ST Depression", 1.0, key="old_cmp")
    
    c3, c4 = st.columns(2)
    with c3:
        slope_c = st.selectbox("ST Slope", ["Up", "Flat", "Down"], key="slope_cmp")
        ca_c = st.selectbox("Major Vessels", ["0", "1", "2", "3"], key="ca_cmp")
    with c4:
        thal_c = st.selectbox("Thalassemia", ["Normal", "Fixed", "Reversible"], key="thal_cmp")

    if st.button("Compare All Models", type="primary"):
        sex_val = 1 if sex_c == "Male" else 0
        cp_val = FEATURE_OPTIONS['cp'][cp_c]
        fbs_val = FEATURE_OPTIONS['fbs'][fbs_c]
        restecg_val = FEATURE_OPTIONS['restecg'][restecg_c]
        exang_val = FEATURE_OPTIONS['exang'][exang_c]
        slope_val = FEATURE_OPTIONS['slope'][slope_c]
        ca_val = FEATURE_OPTIONS['ca'][ca_c]
        thal_val = FEATURE_OPTIONS['thal'][thal_c]
        
        data = [age_c, sex_val, cp_val, trestbps_c, chol_c, fbs_val, restecg_val, thalach_c, exang_val, oldpeak_c, slope_val, ca_val, thal_val]
        
        results = predict_all_models(data)
        
        st.markdown("### Results")
        
        for name, res in results.items():
            with st.expander(f"{name}"):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Prediction", res['prediction'])
                with col_b:
                    st.metric("Test Accuracy", f"{res['accuracy']*100:.1f}%")
                with col_c:
                    st.metric("CV Accuracy", f"{res['cv_accuracy']*100:.1f}%")
        
        st.markdown("#### Model Accuracy Comparison (10-Fold CV)")
        
        models_dict, _, _ = load_all_models()
        acc_data = {'Model': list(models_dict.keys()), 
                  'Test Accuracy': [info['accuracy'] for info in models_dict.values()],
                  'CV Accuracy': [info['cv_accuracy'] for info in models_dict.values()]}
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=acc_data['Model'],
            y=acc_data['Test Accuracy'],
            name='Test Accuracy',
            marker_color='#667eea'
        ))
        fig.add_trace(go.Bar(
            x=acc_data['Model'],
            y=acc_data['CV Accuracy'],
            name='CV Accuracy',
            marker_color='#e74c3c'
        ))
        fig.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Model",
            yaxis_title="Accuracy",
            yaxis_tickformat='.0%',
            height=400,
            barmode='group',
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)


# Stats Tab
with tab3:
    st.markdown("## Statistics Dashboard")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        st.markdown("### Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Predictions", len(df))
        with c2:
            disease_count = len(df[df['prediction'] == 'Heart Disease'])
            st.metric("Heart Disease", disease_count)
        with c3:
            no_disease = len(df[df['prediction'] == 'No Disease'])
            st.metric("No Disease", no_disease)
        with c4:
            avg_conf = df['confidence'].mean()
            st.metric("Avg Confidence", f"{avg_conf:.1f}%")
        
        st.markdown("### Predictions by Date")
        
        df['date_only'] = pd.to_datetime(df['date']).dt.date
        date_counts = df.groupby('date_only')['prediction'].value_counts().unstack(fill_value=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=date_counts.index, y=date_counts.get('Heart Disease', []),
            mode='lines+markers', name='Heart Disease', line=dict(color='#e74c3c', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=date_counts.index, y=date_counts.get('No Disease', []),
            mode='lines+markers', name='No Disease', line=dict(color='#27ae60', width=3)
        ))
        fig.update_layout(
            title="Predictions Over Time",
            xaxis_title="Date",
            yaxis_title="Count",
            height=350,
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Age Distribution")
        
        c5, c6 = st.columns(2)
        with c5:
            avg_age = df['age'].mean()
            st.metric("Average Age", f"{avg_age:.0f} years")
        with c6:
            min_age = df['age'].min()
            max_age = df['age'].max()
            st.metric("Age Range", f"{min_age} - {max_age}")
        
        fig_age = go.Figure()
        fig_age.add_trace(go.Histogram(
            x=df['age'], nbinsx=20, marker_color='#c0392b'
        ))
        fig_age.update_layout(
            title="Age Distribution",
            xaxis_title="Age",
            yaxis_title="Count",
            height=300,
            plot_bgcolor='white'
        )
        st.plotly_chart(fig_age, use_container_width=True)
        
        st.markdown("### Gender Distribution")
        
        gender_counts = df['sex'].value_counts()
        fig_gender = go.Figure(go.Pie(
            labels=gender_counts.index,
            values=gender_counts.values,
            hole=0.4,
            marker=dict(colors=['#667eea', '#e74c3c'])
        ))
        fig_gender.update_layout(
            title="Gender Distribution",
            height=300
        )
        st.plotly_chart(fig_gender, use_container_width=True)
        
    else:
        st.info("No statistics available. Make some predictions first!")


# Features Tab
with tab4:
    st.markdown("## Feature Importance")
    st.markdown("Top risk factors for heart disease")
    
    model, scaler, feature_names = load_model()
    
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_names, model.feature_importances_))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        fig = go.Figure(go.Bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation='h',
            marker_color='#c0392b'
        ))
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance",
            height=500,
            font_size=14,
            plot_bgcolor='white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### About Model")
    st.markdown("""
    - **Model:** Random Forest Classifier
    - **Accuracy:** 86.7%
    - **Features:** 13 medical indicators
    
    Based on Cleveland heart disease dataset.
    """)


# History Tab
with tab5:
    st.markdown("## Prediction History")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        st.dataframe(df, use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Predictions", len(st.session_state.history))
        with c2:
            disease_count = sum(1 for h in st.session_state.history if h['prediction'] == 'Heart Disease')
            st.metric("Disease Detected", disease_count)
        
        st.markdown("---")
        st.markdown("### Export Data")
        
        csv = df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "predictions.csv",
            "text/csv"
        )
        
        if st.button("Clear History"):
            st.session_state.history = []
            st.session_state.patient_counter = 0
            st.rerun()
    else:
        st.info("No predictions yet!")


st.markdown("---")
st.caption("Heart Disease Prediction System | ML Powered")