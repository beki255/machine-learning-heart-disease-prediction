import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.graph_objects as go
from datetime import datetime
import json
import io
import base64

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
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

# Clean white background theme with larger fonts
st.markdown("""
<style>
    /* Main Background - Clean White */
    .stApp { 
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
    }
    
    /* Header Banner with Background Image */
    .header-banner {
        background: linear-gradient(rgba(138,43,226,0.85), rgba(153,50,204,0.85)), 
                    url('https://images.unsplash.com/photo-1559757148-5c350d0e053?w=1200') no-repeat center center;
        background-size: cover;
        padding: 30px; border-radius: 20px;
        margin-bottom: 30px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        display: flex; flex-direction: column; align-items: center;
    }
            
    html, body { 
        font-size: 24px !important; 
    }
    
    /* Headers */
    h1 { 
        font-size: 64px !important; 
        font-weight: 800 !important; 
        color: #1e3c72 !important;
        margin-bottom: 25px;
    }
    h2 { 
        font-size: 48px !important; 
        font-weight: 700 !important; 
        color: #2c3e50 !important;
        margin-bottom: 25px;
    }
    h3 { 
        font-size: 40px !important; 
        font-weight: 600 !important; 
        color: #e74c3c !important;
        margin-bottom: 20px;
    }
    h4 {
        font-size: 32px !important;
        font-weight: 600 !important;
        color: #34495e !important;
    }
    
    /* Labels - MUCH LARGER */
    .stTextInput label, .stNumberInput label, .stSelectbox label, 
    .stRadio label, .stCheckbox label, .stDateInput label {
        font-size: 28px !important; 
        font-weight: 700 !important; 
        color: #1e3c72 !important;
        margin-bottom: 15px !important;
    }
    
    /* Input values text */
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
        font-size: 24px !important;
        font-weight: 500 !important;
    }
    
    /* Buttons - BIGGER */
    .stButton button { 
        font-size: 26px !important; 
        padding: 22px 44px !important; 
        border-radius: 15px !important; 
        font-weight: 700 !important;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border: none;
        color: white;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Primary Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
    }
    
    /* Alerts - BIGGER */
    .stAlert { 
        font-size: 26px !important; 
        border-radius: 15px; 
        padding: 30px; 
        font-weight: 600;
    }
    
    /* Metrics - BIGGER */
    [data-testid="stMetricValue"] { 
        font-size: 56px !important; 
        font-weight: 800 !important;
        color: #1e3c72 !important;
    }
    [data-testid="stMetricLabel"] { 
        font-size: 24px !important; 
        font-weight: 700 !important;
        color: #34495e !important;
    }
    
    /* TABS - MUCH BIGGER */
    .stTabs [data-testid="stTabBarButton"] { 
        font-size: 36px !important; 
        padding: 26px 52px !important; 
        font-weight: 800 !important;
        background: rgba(255,255,255,0.95);
        border-radius: 18px 18px 0 0;
        margin-right: 10px;
        transition: all 0.3s ease;
        color: #1e3c72 !important;
        letter-spacing: 1.5px;
    }
    .stTabs [data-testid="stTabBarButton"]:hover {
        background: white;
        transform: translateY(-4px);
        box-shadow: 0 -6px 16px rgba(0,0,0,0.1);
    }
    .stTabs [data-testid="stTabBarButton"][aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white !important;
        box-shadow: 0 -6px 16px rgba(0,0,0,0.15);
    }
    
    /* Tab content container */
    .stTabs [data-testid="stTabContent"] {
        background: white;
        border-radius: 0 22px 22px 22px;
        padding: 35px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }
    
/* Header Banner - Violet */
.header-banner {
    background: linear-gradient(135deg, #8A2BE2 0%, #9932CC 100%); padding: 50px; border-radius: 20px;
    margin-bottom: 40px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    display: flex; flex-direction: column; align-items: center;
}
    
    /* Cards */
    .card {
        background: #f8f9fa;
        border-radius: 20px;
        padding: 35px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        margin: 25px 0;
        transition: all 0.3s ease;
        border: 1px solid #e0e6ed;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.12);
        border-color: #1e3c72;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 2px solid #e0e6ed;
        padding: 25px;
    }
    
    /* DataFrame/Table - MUCH LARGER FONTS */
    .stDataFrame {
        font-size: 28px !important;
    }
    .stDataFrame table {
        font-size: 28px !important;
        width: 100% !important;
    }
    .stDataFrame th {
        font-size: 30px !important;
        font-weight: 800 !important;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        padding: 20px !important;
        text-align: center !important;
    }
    .stDataFrame td {
        font-size: 26px !important;
        padding: 16px !important;
        border-bottom: 1px solid #e0e6ed !important;
    }
    .stDataFrame tr:hover {
        background-color: #f0f4f8 !important;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 35px;
        background: white;
        border-radius: 15px;
        margin-top: 45px;
        border: 1px solid #e0e6ed;
        font-size: 20px !important;
    }
    
    /* Info/Warning boxes */
    [data-testid="stInfo"] {
        background: #e3f2fd;
        border-left: 6px solid #2196f3;
        border-radius: 12px;
        padding: 22px;
        font-size: 20px;
    }
    
    [data-testid="stSuccess"] {
        background: #e8f5e9;
        border-left: 6px solid #4caf50;
        border-radius: 12px;
        padding: 22px;
        font-size: 20px;
    }
    
    [data-testid="stError"] {
        background: #ffebee;
        border-left: 6px solid #f44336;
        border-radius: 12px;
        padding: 22px;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'model.pkl'))
    scaler = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'scaler.pkl'))
    imputer = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'imputer.pkl'))
    feature_names = joblib.load(os.path.join(PROJECT_ROOT, 'models', 'feature_names.pkl'))
    return model, scaler, imputer, feature_names


@st.cache_resource
def load_all_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    import numpy as np
    
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'heartt_cleveland_cleaned.csv'))
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Mean imputation instead of dropping rows
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    
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

# Full names for display in history table
FULL_NAMES = {
    'id': 'Patient ID',
    'name': 'Patient Name',
    'age': 'Age (years)',
    'sex': 'Gender',
    'trestbps': 'Blood Pressure (mmHg)',
    'chol': 'Cholesterol (mg/dl)',
    'cp': 'Chest Pain Type',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG Results',
    'thalach': 'Maximum Heart Rate',
    'exang': 'Exercise Induced Angina',
    'oldpeak': 'ST Depression (mm)',
    'slope': 'ST Slope',
    'ca': 'Major Vessels (0-3)',
    'thal': 'Thalassemia',
    'date': 'Date & Time',
    'doctor': 'Doctor Name',
    'prediction': 'Prediction Result',
    'confidence': 'Confidence (%)'
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
    # Encode the local image for background
    import base64
    with open(os.path.join(PROJECT_ROOT, 'data', 'photo_2026-04-29_21-43-55.jpg'), 'rb') as f:
        img_data = base64.b64encode(f.read()).decode()
    
    if not st.session_state.get('show_login_form', False):
        # Welcome Page with Background Image
        st.markdown(f"""
        <div style="background: linear-gradient(rgba(138,43,226,0.88), rgba(153,50,204,0.88)), 
                    url('data:image/jpeg;base64,{img_data}') no-repeat center center;
                    background-size: cover;
                    padding: 100px 50px; border-radius: 20px; margin: 50px 0;
                    text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.15);
                    min-height: 500px; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 120px; margin-bottom: 30px;">❤️</div>
            <h1 style="color: white; margin: 0 0 20px 0; font-size: 72px;">Heart Disease Prediction System</h1>
            <p style="color: white; opacity: 0.95; margin: 0; font-size: 28px; font-weight: 500;">
                Advanced AI-Powered Heart Disease Detection using Machine Learning
            </p>
            <p style="color: white; opacity: 0.8; margin: 30px 0 0 0; font-size: 20px;">
                Click the button below to access the system
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔐 Login to System", type="primary", use_container_width=True):
            st.session_state.show_login_form = True
            st.rerun()
    
    else:
        # Login Form
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="header-banner">
                <img src="https://img.freepik.com/free-vector/cardiology-concept-illustration_114360-2265.jpg" 
                     style="width: 120px; height: 120px; border-radius: 50%; margin-bottom: 20px; object-fit: cover;">
                <h1 style="color: white; margin: 0;">Heart Disease Prediction</h1>
                <p style="color: white; opacity: 0.95; margin: 15px 0 0 0; font-size: 26px; font-weight: 500;">
                    AI-Powered Detection System
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown("### Login Required")
                st.markdown("Please sign in to access the system")
            
            username = st.text_input("Username", placeholder="Enter your username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")
            
            if st.button("Login", type="primary", use_container_width=True):
                if username in USERS and USERS[username]['password'] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.role = USERS[username]['role']
                    st.rerun()
                else:
                    st.error("Invalid credentials! Please check your username and password.")
            
            st.markdown("---")
            st.caption("Default Admin: admin / admin123 | Doctor: doctor/doctor123")


if not st.session_state.logged_in:
    login()
    st.stop()


def predict_single(data):
    model, scaler, imputer, feature_names = load_model()
    data_array = np.array(data).reshape(1, -1)
    data_imputed = imputer.transform(data_array)
    data_scaled = scaler.transform(data_imputed)
    
    pred = model.predict(data_scaled)[0]
    prob = model.predict_proba(data_scaled)[0]
    
    return int(pred), prob[0] * 100, prob[1] * 100


def predict_all_models(data):
    import joblib
    from sklearn.impute import SimpleImputer
    import pandas as pd
    import os
    
    models_dict, scaler, feature_names = load_all_models()
    data_array = np.array(data).reshape(1, -1)
    
    # Load saved imputer and apply transform (not fit_transform)
    imputer_path = os.path.join(PROJECT_ROOT, 'models', 'imputer.pkl')
    imputer = joblib.load(imputer_path)
    
    X = pd.DataFrame(data_array, columns=feature_names)
    X_imputed = imputer.transform(X)
    data_scaled = scaler.transform(X_imputed)
    
    results = {}
    for name, info in models_dict.items():
        model = info['model']
        pred = model.predict(data_scaled)[0]
        prob = model.predict_proba(data_scaled)[0]
        results[name] = {
            'prediction': '❤️ Heart Disease' if pred == 1 else '✅ No Disease',
            'confidence': round(max(prob) * 100, 1),
            'disease_prob': round(prob[1] * 100, 1),
            'accuracy': info['accuracy'],
            'cv_accuracy': info['cv_accuracy']
        }
    
    return results


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 18px; margin-bottom: 30px;">
        <div style="font-size: 56px;">❤️</div>
        <h3 style="color: white; margin: 15px 0 8px 0;">Heart Disease</h3>
        <p style="color: rgba(255,255,255,0.9); margin: 0; font-size: 18px;">AI Prediction System</p>
    </div>
    """, unsafe_allow_html=True)
    
    role_badge = "👑 Admin" if st.session_state.role == 'admin' else "👨‍⚕️ Doctor"
    st.info(f"**Logged in as:** {st.session_state.username}\n\n**Role:** {role_badge}")
    
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.role = ""
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.role == 'admin':
        with st.expander("⚙️ Admin Panel", expanded=False):
            st.markdown("### 👥 Manage Users")
            
            users_list = list(USERS.keys())
            selected = st.selectbox("Select User", users_list)
            
            if selected:
                st.write(f"**Role:** {USERS[selected]['role'].capitalize()}")
            
            st.markdown("---")
            st.markdown("### ➕ Add New User")
            
            new_user = st.text_input("Username", placeholder="Enter username")
            new_pass = st.text_input("Password", type="password", placeholder="Enter password")
            new_role = st.selectbox("Role", ["doctor"])
            
            if st.button("➕ Add User", use_container_width=True):
                if new_user and new_pass:
                    if new_user in USERS:
                        st.error("❌ User already exists!")
                    else:
                        USERS[new_user] = {'password': new_pass, 'role': new_role}
                        save_users(USERS)
                        st.success(f"✅ User {new_user} added successfully!")
                        st.rerun()
                else:
                    st.error("❌ Please fill all fields!")
            
            if selected and selected != 'admin':
                st.markdown("---")
                st.markdown("### 🗑️ Remove User")
                if st.button("🗑️ Remove User", use_container_width=True):
                    del USERS[selected]
                    save_users(USERS)
                    st.success(f"✅ User {selected} removed successfully!")
                    st.rerun()


# Main content
# Read and encode the local image
import base64
with open(os.path.join(PROJECT_ROOT, 'data', 'photo_2026-04-29_21-43-55.jpg'), 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

st.markdown(f"""
<div class="header-banner" style="background: linear-gradient(rgba(138,43,226,0.85), rgba(153,50,204,0.85)), 
            url('data:image/jpeg;base64,{img_data}') no-repeat center center; 
            background-size: cover; padding: 30px; border-radius: 15px; margin-bottom: 25px;
            text-align: center; display: flex; flex-direction: column; align-items: center;">
    <div style="font-size: 48px;">❤️</div>
    <h1 style="color: white; margin: 8px 0 0 0; font-size: 48px;">Heart Disease Prediction System</h1>
    <p style="color: white; opacity: 0.95; margin: 10px 0 0 0; font-size: 20px; font-weight: 500;">
        Advanced AI-Powered Heart Disease Detection using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

if st.session_state.role == 'admin':
    tabs = st.tabs(["USERS", "PREDICT", "COMPARE", "STATS", "FEATURES", "ANALYSIS", "HISTORY", "BATCH"])
    tab_u, tab1, tab2, tab3, tab4, tab5, tab6, tab_batch = tabs
else:
    tabs = st.tabs(["PREDICT", "COMPARE", "STATS", "FEATURES", "ANALYSIS", "HISTORY", "BATCH"])
    tab1, tab2, tab3, tab4, tab5, tab6, tab_batch = tabs

# User Management (Admin only)
if st.session_state.role == 'admin':
    with tab_u:
        st.markdown("## 👥 User Management")
        st.markdown("Manage doctor accounts in the system")
        
        st.markdown("### 📋 All Users")
        users_data = []
        for user, info in USERS.items():
            users_data.append({
                'Username': user,
                'Role': info['role'].capitalize(),
                'Password': info['password']
            })
        st.dataframe(pd.DataFrame(users_data), use_container_width=True, height=400)
        
        st.markdown("---")
        st.markdown("### ➕ Add New Doctor")
        
        c1, c2 = st.columns(2)
        with c1:
            new_user = st.text_input("Username", placeholder="Enter username", key="admin_new_user")
        with c2:
            new_pass = st.text_input("Password", type="password", placeholder="Enter password", key="admin_new_pass")
        
        new_role = st.selectbox("Role", ["doctor"], key="admin_role")
        
        if st.button("✅ Add Doctor", use_container_width=True, key="admin_add"):
            if new_user and new_pass:
                if new_user in USERS:
                    st.error("❌ User already exists!")
                else:
                    USERS[new_user] = {'password': new_pass, 'role': new_role}
                    save_users(USERS)
                    st.success(f"✅ Doctor {new_user} added successfully!")
                    st.rerun()
            else:
                st.error("❌ Please fill all fields!")
        
        st.markdown("---")
        st.markdown("### 🗑️ Remove Doctor")
        
        rem_user = st.selectbox("Select doctor to remove", [u for u in USERS.keys() if u != 'admin'])
        
        if st.button("🗑️ Remove Doctor", use_container_width=True, key="admin_remove"):
            if rem_user:
                del USERS[rem_user]
                save_users(USERS)
                st.success(f"✅ Doctor {rem_user} removed successfully!")
                st.rerun()


# Prediction Tab
with tab1:
    st.markdown("## 🔮 Patient Assessment")
    st.markdown("Enter patient information for heart disease prediction")
    
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown('<div class="card"><h3>📋 Patient Information</h3></div>', unsafe_allow_html=True)
        patient_id = f"P{str(st.session_state.patient_counter + 1).zfill(4)}"
        st.info(f"**Patient ID:** {patient_id}")
        patient_name = st.text_input("📝 Patient Name", placeholder="Enter full name", key="pred_name")
        age = st.number_input("🎂 Age", min_value=1, max_value=120, value=55, step=1, key="age_pred", help="Patient's age in years")
        sex = st.radio("⚥ Sex", ["Male", "Female"], horizontal=True, key="sex_pred")
    
    with c2:
        st.markdown('<div class="card"><h3>🩺 Medical Features</h3></div>', unsafe_allow_html=True)
        
        c_bp, c_chol = st.columns(2)
        with c_bp:
            trestbps = st.number_input("💓 Blood Pressure (mmHg)", min_value=94, max_value=200, value=130, step=5, key="bp_pred", help="Resting blood pressure")
        with c_chol:
            chol = st.number_input("🩸 Cholesterol (mg/dl)", min_value=100, max_value=600, value=250, step=10, key="chol_pred", help="Serum cholesterol")
        
        c_cp, c_fbs = st.columns(2)
        with c_cp:
            cp = st.selectbox("💔 Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"], key="cp_pred")
        with c_fbs:
            fbs = st.selectbox("🍬 Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"], key="fbs_pred")
        
        c_ecg, c_hr = st.columns(2)
        with c_ecg:
            restecg = st.selectbox("📊 Resting ECG", ["Normal", "ST-T wave", "Left ventricular"], key="ecg_pred")
        with c_hr:
            thalach = st.number_input("💗 Max Heart Rate", min_value=60, max_value=220, value=150, step=5, key="hr_pred", help="Maximum heart rate achieved")
        
        c_ex, c_old = st.columns(2)
        with c_ex:
            exang = st.selectbox("🏃 Exercise Angina", ["No", "Yes"], key="ex_pred")
        with c_old:
            oldpeak = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="old_pred", help="ST depression induced by exercise")
        
        c_slope, c_ca = st.columns(2)
        with c_slope:
            slope = st.selectbox("📈 ST Slope", ["Up", "Flat", "Down"], key="slope_pred")
        with c_ca:
            ca = st.selectbox("🫀 Major Vessels", ["0", "1", "2", "3"], key="ca_pred", help="Number of major vessels colored by fluoroscopy")
        
        with c2:
            thal = st.selectbox("🔬 Thalassemia", ["Normal", "Fixed", "Reversible"], key="thal_pred")

    st.markdown("---")
    
    if st.button("🔍 PREDICT HEART DISEASE", type="primary", use_container_width=True):
        sex_val = 1 if sex == "Male" else 0
        cp_val = FEATURE_OPTIONS['cp'][cp]
        fbs_val = FEATURE_OPTIONS['fbs'][fbs]
        restecg_val = FEATURE_OPTIONS['restecg'][restecg]
        exang_val = FEATURE_OPTIONS['exang'][exang]
        slope_val = FEATURE_OPTIONS['slope'][slope]
        ca_val = FEATURE_OPTIONS['ca'][ca]
        thal_val = FEATURE_OPTIONS['thal'][thal]
        
        data = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca_val, thal_val]
        
        pred, prob_no, prob_yes = predict_single(data)
        
        rec = {
            'id': patient_id,
            'name': patient_name if patient_name else "Anonymous",
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
            'thal': thal,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
            'doctor': st.session_state.username,
            'prediction': 'Heart Disease' if pred == 1 else 'No Disease',
            'confidence': round(prob_yes if pred == 1 else prob_no, 1)
        }
        st.session_state.history.append(rec)
        st.session_state.patient_counter += 1
        st.session_state.show_pdf = True
        st.session_state.show_features = True
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if pred == 1:
                st.error(f"""
                ### ❌ Heart Disease Detected
                
                **Patient ID:** {patient_id}
                **Patient Name:** {patient_name if patient_name else "Anonymous"}
                **Confidence:** {rec['confidence']}%
                
                ⚠️ **Recommendation:** Please advise patient for immediate further testing and consultation with a cardiologist.
                """)
            else:
                st.success(f"""
                ### ✅ No Heart Disease Detected
                
                **Patient ID:** {patient_id}
                **Patient Name:** {patient_name if patient_name else "Anonymous"}
                **Confidence:** {rec['confidence']}%
                
                💚 **Recommendation:** Patient appears healthy. Continue regular check-ups and healthy lifestyle.
                """)
        
        # PDF Download
        if st.session_state.get('show_pdf', False):
            html_report = f"""
            <html><body>
            <h1>Heart Disease Prediction Report</h1>
            <p><b>Patient:</b> {patient_name if patient_name else 'Anonymous'}</p>
            <p><b>ID:</b> {patient_id}</p>
            <p><b>Prediction:</b> {rec['prediction']}</p>
            <p><b>Confidence:</b> {rec['confidence']}%</p>
            <p><b>Doctor:</b> {st.session_state.username}</p>
            <p><b>Date:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            </body></html>
            """
            st.download_button(
                "📥 Download HTML Report",
                html_report,
                f"report_{patient_id}.html",
                "text/html",
                use_container_width=True,
                key="pdf_download"
            )
        
        # Feature Importance
        if st.session_state.get('show_features', False):
            if st.button("🔍 Show Key Prediction Factors", use_container_width=True, key="show_factors"):
                st.session_state.show_factors = True
            
            if st.session_state.get('show_factors', False):
                try:
                    model, _, imputer, feature_names = load_model()
                    data = [age, 1 if sex == "Male" else 0, FEATURE_OPTIONS['cp'][cp], 
                            trestbps, chol, FEATURE_OPTIONS['fbs'][fbs], 
                            FEATURE_OPTIONS['restecg'][restecg], thalach, 
                            FEATURE_OPTIONS['exang'][exang], oldpeak, 
                            FEATURE_OPTIONS['slope'][slope], 
                            FEATURE_OPTIONS['ca'][ca], 
                            FEATURE_OPTIONS['thal'][thal]]
                    data_array = np.array(data).reshape(1, -1)
                    data_imputed = imputer.transform(data_array)
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        st.write("### Key Factors in This Prediction")
                        sorted_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
                        for fname, imp in sorted_features:
                            st.progress(float(imp), text=f"{FEATURE_LABELS.get(fname, fname)}: {imp:.3f}")
                    else:
                        st.info("Model doesn't support feature importance display")
                except Exception as e:
                    st.error(f"Explanation failed: {str(e)}")


# Compare Tab
with tab2:
    st.markdown("## 📊 Model Comparison")
    st.markdown("Compare predictions from 4 different ML models to get comprehensive insights")
    
    st.markdown("### 🩺 Enter Patient Features")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        age_c = st.number_input("🎂 Age", min_value=1, max_value=120, value=55, step=1, key="age_cmp")
        sex_c = st.radio("⚥ Sex", ["Male", "Female"], horizontal=True, key="sex_cmp")
        cp_c = st.selectbox("💔 Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"], key="cp_cmp")
        trestbps_c = st.number_input("💓 Blood Pressure (mmHg)", min_value=94, max_value=200, value=130, step=5, key="bp_cmp")
        chol_c = st.number_input("🩸 Cholesterol (mg/dl)", min_value=100, max_value=600, value=250, step=10, key="chol_cmp")
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fbs_c = st.selectbox("🍬 Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"], key="fbs_cmp")
        restecg_c = st.selectbox("📊 Resting ECG", ["Normal", "ST-T wave", "Left ventricular"], key="ecg_cmp")
        thalach_c = st.number_input("💗 Max Heart Rate", min_value=60, max_value=220, value=150, step=5, key="hr_cmp")
        exang_c = st.selectbox("🏃 Exercise Angina", ["No", "Yes"], key="ex_cmp")
        oldpeak_c = st.number_input("📉 ST Depression", min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="old_cmp")
        st.markdown('</div>', unsafe_allow_html=True)
    
    c3, c4 = st.columns(2)
    with c3:
        slope_c = st.selectbox("📈 ST Slope", ["Up", "Flat", "Down"], key="slope_cmp")
        ca_c = st.selectbox("🫀 Major Vessels", ["0", "1", "2", "3"], key="ca_cmp")
    with c4:
        thal_c = st.selectbox("🔬 Thalassemia", ["Normal", "Fixed", "Reversible"], key="thal_cmp")

    if st.button("🔄 Compare All Models", type="primary", use_container_width=True):
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
        
        st.markdown("### 📊 Model Results")
        
        for name, res in results.items():
            with st.expander(f"🤖 {name}", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("📋 Prediction", res['prediction'])
                with col_b:
                    st.metric("🎯 Test Accuracy", f"{res['accuracy']*100:.1f}%")
                with col_c:
                    st.metric("🔄 CV Accuracy", f"{res['cv_accuracy']*100:.1f}%")
        
        st.markdown("### 📈 Model Accuracy Comparison (10-Fold CV)")
        
        models_dict, _, _ = load_all_models()
        acc_data = {'Model': list(models_dict.keys()), 
                  'Test Accuracy': [info['accuracy'] for info in models_dict.values()],
                  'CV Accuracy': [info['cv_accuracy'] for info in models_dict.values()]}
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=acc_data['Model'],
            y=acc_data['Test Accuracy'],
            name='Test Accuracy',
            marker_color='#1e3c72',
            text=[f"{x:.1%}" for x in acc_data['Test Accuracy']],
            textposition='auto',
        ))
        fig.add_trace(go.Bar(
            x=acc_data['Model'],
            y=acc_data['CV Accuracy'],
            name='CV Accuracy',
            marker_color='#2a5298',
            text=[f"{x:.1%}" for x in acc_data['CV Accuracy']],
            textposition='auto',
        ))
        fig.update_layout(
            title="Model Accuracy Comparison",
            title_font_size=28,
            xaxis_title="Model",
            xaxis_title_font_size=22,
            yaxis_title="Accuracy",
            yaxis_title_font_size=22,
            yaxis_tickformat='.0%',
            height=550,
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18)
        )
        st.plotly_chart(fig, use_container_width=True)


# Stats Tab
with tab3:
    st.markdown("## 📈 Statistics Dashboard")
    st.markdown("View comprehensive analytics and insights from all predictions")
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        st.markdown("### 📊 Overview")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("📝 Total Predictions", len(df))
        with c2:
            disease_count = len(df[df['prediction'] == 'Heart Disease'])
            st.metric("❤️ Heart Disease Cases", disease_count)
        with c3:
            no_disease = len(df[df['prediction'] == 'No Disease'])
            st.metric("✅ No Disease Cases", no_disease)
        with c4:
            avg_conf = df['confidence'].mean()
            st.metric("📊 Avg Confidence", f"{avg_conf:.1f}%")
        
        st.markdown("### 📅 Predictions Over Time")
        
        df['date_only'] = pd.to_datetime(df['date']).dt.date
        date_counts = df.groupby('date_only')['prediction'].value_counts().unstack(fill_value=0)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=date_counts.index, y=date_counts.get('Heart Disease', []),
            mode='lines+markers', name='❤️ Heart Disease', 
            line=dict(color='#e74c3c', width=5),
            marker=dict(size=12)
        ))
        fig.add_trace(go.Scatter(
            x=date_counts.index, y=date_counts.get('No Disease', []),
            mode='lines+markers', name='✅ No Disease', 
            line=dict(color='#27ae60', width=5),
            marker=dict(size=12)
        ))
        fig.update_layout(
            title="Predictions Over Time",
            title_font_size=28,
            xaxis_title="Date",
            xaxis_title_font_size=22,
            yaxis_title="Count",
            yaxis_title_font_size=22,
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.markdown("### 👤 Age Distribution")
            
            fig_age = go.Figure()
            fig_age.add_trace(go.Histogram(
                x=df['age'], nbinsx=20, 
                marker_color='#1e3c72',
                marker_line_color='white',
                marker_line_width=2
            ))
            fig_age.update_layout(
                title="Age Distribution of Patients",
                title_font_size=24,
                xaxis_title="Age (years)",
                xaxis_title_font_size=20,
                yaxis_title="Number of Patients",
                yaxis_title_font_size=20,
                height=450,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=16)
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col_right:
            st.markdown("### ⚥ Gender Distribution")
            
            gender_counts = df['sex'].value_counts()
            fig_gender = go.Figure(go.Pie(
                labels=gender_counts.index,
                values=gender_counts.values,
                hole=0.4,
                marker=dict(colors=['#1e3c72', '#e74c3c']),
                textfont=dict(size=18)
            ))
            fig_gender.update_layout(
                title="Gender Distribution of Patients",
                title_font_size=24,
                height=450,
                font=dict(size=16)
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        
        st.markdown("### 📊 Detailed Statistics")
        
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            avg_age = df['age'].mean()
            std_age = df['age'].std()
            st.metric("Average Age", f"{avg_age:.0f} ± {std_age:.0f} years")
        
        with col_stats2:
            avg_bp = df['trestbps'].mean()
            std_bp = df['trestbps'].std()
            st.metric("Average Blood Pressure", f"{avg_bp:.0f} ± {std_bp:.0f} mmHg")
        
        with col_stats3:
            avg_chol = df['chol'].mean()
            std_chol = df['chol'].std()
            st.metric("Average Cholesterol", f"{avg_chol:.0f} ± {std_chol:.0f} mg/dl")
        
    else:
        st.info("📭 No statistics available. Make some predictions first!")


# Features Tab
with tab4:
    st.markdown("## 📋 Feature Importance Analysis")
    st.markdown("Understanding the key risk factors for heart disease")
    
    model, scaler, imputer, feature_names = load_model()
    
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feature_names, model.feature_importances_))
        # Use full names for display
        importance_display = {FEATURE_LABELS.get(k, k): v for k, v in importance.items()}
        importance_display = dict(sorted(importance_display.items(), key=lambda x: x[1], reverse=True))
        
        fig = go.Figure(go.Bar(
            x=list(importance_display.values()),
            y=list(importance_display.keys()),
            orientation='h',
            marker_color='#1e3c72',
            marker_line_color='white',
            marker_line_width=2,
            text=[f"{x:.1%}" for x in importance_display.values()],
            textposition='outside',
        ))
        fig.update_layout(
            title="Top Risk Factors for Heart Disease",
            title_font_size=28,
            xaxis_title="Importance Score",
            xaxis_title_font_size=22,
            yaxis_title="Medical Features",
            yaxis_title_font_size=22,
            height=650,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=18),
            xaxis_tickformat='.0%'
        )
        st.plotly_chart(fig, use_container_width=True)

# Analysis Tab - Additional Graphs
with tab5:
    st.markdown("## Advanced Analysis")
    st.markdown("ROC Curves, Confusion Matrix, Correlation Heatmap & Distributions")
    
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'heartt_cleveland_cleaned.csv'))
    df = df.dropna()
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Train/test split for metrics
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # ROC Curve
    st.markdown("### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.3f})', 
                               line=dict(color='#8A2BE2', width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', 
                               line=dict(color='gray', dash='dash')))
    fig_roc.update_layout(title='Receiver Operating Characteristic (ROC) Curve',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          height=400, plot_bgcolor='white')
    st.plotly_chart(fig_roc, use_container_width=True)
    
    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = go.Figure(data=go.Heatmap(z=cm, x=['No Disease', 'Heart Disease'], 
                                     y=['No Disease', 'Heart Disease'],
                                     colorscale='Purples', showscale=True))
    fig_cm.update_layout(title='Confusion Matrix',
                        xaxis_title='Predicted',
                        yaxis_title='Actual',
                        height=350)
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Correlation Heatmap
    st.markdown("### Correlation Heatmap")
    corr = df.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                       colorscale='RdBu', zmid=0))
    fig_corr.update_layout(title='Feature Correlations',
                        height=500, width=None)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Distributions
    st.markdown("### Feature Distributions")
    
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### Age Distribution")
        fig_age = go.Figure()
        fig_age.add_trace(go.Histogram(x=df['age'], nbinsx=20, name='Age',
                                   marker_color='#8A2BE2'))
        fig_age.update_layout(title='Age Distribution', xaxis_title='Age', 
                      height=300, plot_bgcolor='white')
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col_d2:
        st.markdown("#### Target Distribution")
        target_counts = y.value_counts()
        fig_target = go.Figure(go.Pie(labels=['No Disease', 'Heart Disease'],
                                  values=target_counts.values,
                                  hole=0.4,
                                  marker_colors=['#27ae60', '#e74c3c']))
        fig_target.update_layout(title='Heart Disease vs No Disease',
                             height=300)
        st.plotly_chart(fig_target, use_container_width=True)

# History Tab - WITH FULL FEATURE NAMES
with tab6:
    st.markdown("## 📜 Prediction History")
    st.markdown("View and manage all past predictions with complete medical details")
    
    if st.session_state.history:
        # Create DataFrame with all records
        df = pd.DataFrame(st.session_state.history)
        
        # Rename columns to full descriptive names
        df_display = df.rename(columns=FULL_NAMES)
        
        # Reorder columns for better readability
        column_order = ['Patient ID', 'Patient Name', 'Age (years)', 'Gender', 'Blood Pressure (mmHg)', 
                       'Cholesterol (mg/dl)', 'Chest Pain Type', 'Fasting Blood Sugar', 
                       'Resting ECG Results', 'Maximum Heart Rate', 'Exercise Induced Angina', 
                       'ST Depression (mm)', 'ST Slope', 'Major Vessels (0-3)', 'Thalassemia',
                       'Prediction Result', 'Confidence (%)', 'Date & Time', 'Doctor Name']
        
        df_display = df_display[[col for col in column_order if col in df_display.columns]]
        
        # Display the dataframe with full names
        st.dataframe(df_display, use_container_width=True, height=500)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Predictions", len(st.session_state.history))
        with col2:
            disease_count = sum(1 for h in st.session_state.history if h['prediction'] == 'Heart Disease')
            st.metric("❤️ Heart Disease Detected", disease_count)
        with col3:
            healthy_count = sum(1 for h in st.session_state.history if h['prediction'] == 'No Disease')
            st.metric("✅ No Disease Detected", healthy_count)
        
        st.markdown("---")
        st.markdown("### 💾 Export Data")
        
        # Export with full column names
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name=f"heart_disease_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        st.markdown("### 🗑️ Clear All History")
        if st.button("⚠️ Clear All Predictions", use_container_width=True):
            st.session_state.history = []
            st.session_state.patient_counter = 0
            st.success("✅ All predictions cleared successfully!")
            st.rerun()
    else:
        st.info("📭 No predictions yet! Go to the Predict tab to make your first prediction.")



# ========== NEW FEATURES: PDF, SHAP, BATCH PREDICTION ==========

def generate_pdf_report(patient_data, prediction, confidence, doctor):
    """Generate PDF report for a prediction"""
    html_content = f"""
    <html>
    <head><title>Heart Disease Prediction Report</title></head>
    <body>
    <h1>Heart Disease Prediction Report</h1>
    <hr>
    <p><b>Patient ID:</b> {patient_data.get('id', 'N/A')}</p>
    <p><b>Patient Name:</b> {patient_data.get('name', 'Anonymous')}</p>
    <p><b>Age:</b> {patient_data.get('age', 'N/A')}</p>
    <p><b>Prediction:</b> {prediction}</p>
    <p><b>Confidence:</b> {confidence}%</p>
    <p><b>Doctor:</b> {doctor}</p>
    <p><b>Date:</b> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
    <hr>
    <p><i>This report was generated by the Heart Disease Prediction System</i></p>
    </body>
    </html>
    """
    return html_content.encode('utf-8')




def get_shap_explanation(data, model, feature_names):
    """Get SHAP values for a prediction"""
    try:
        explainer = shap.TreeExplainer(model)
        data_array = np.array(data).reshape(1, -1)
        shap_values = explainer.shap_values(data_array)
        return shap_values, feature_names
    except:
        return None, None

# Batch Prediction Function
def batch_predict(uploaded_file, model, scaler, imputer, feature_names):
    """Process batch predictions from uploaded CSV"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if all required features are present
        missing_cols = [col for col in feature_names if col not in df.columns]
        if missing_cols:
            return None, f"Missing columns: {missing_cols}"
        
        # Apply imputation and scaling
        X = df[feature_names]
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Add results to dataframe
        df['Prediction'] = ['Heart Disease' if p == 1 else 'No Disease' for p in predictions]
        df['Confidence_No_Disease'] = probabilities[:, 0] * 100
        df['Confidence_Heart_Disease'] = probabilities[:, 1] * 100
        
        return df, "Success"
    except Exception as e:
        return None, str(e)


# Add Batch Prediction Tab to UI (if logged in as doctor or admin)
if st.session_state.get('logged_in', False) and 'batch_tab' not in st.session_state:
    st.session_state.batch_tab = True

# Add PDF download button after prediction
if 'show_pdf' not in st.session_state:
    st.session_state.show_pdf = False

# SHAP explanation in prediction result
if 'shap_values' not in st.session_state:
    st.session_state.shap_values = None

# ========== END NEW FEATURES ==========


# Batch Prediction Tab
with tab_batch:
    st.markdown("## 📁 Batch Prediction")
    st.markdown("Upload a CSV file with patient data for bulk predictions")
    
    st.markdown("### 📋 Required Columns:")
    st.code(", ".join(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                      'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']))
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("🔮 Run Batch Prediction", type="primary", use_container_width=True):
            with st.spinner("Processing..."):
                try:
                    model, scaler, imputer, feature_names = load_model()
                    df, msg = batch_predict(uploaded_file, model, scaler, imputer, feature_names)
                    
                    if df is not None:
                        st.success(f"✅ Processed {len(df)} patients!")
                        st.dataframe(df, use_container_width=True)
                        
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results CSV",
                            csv,
                            f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    else:
                        st.error(f"❌ Error: {msg}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p style="font-size: 18px;">❤️ Heart Disease Prediction System | Powered by Advanced AI & Machine Learning</p>
    <p style="font-size: 16px; opacity: 0.7;">© 2026 - AI Healthcare System | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)