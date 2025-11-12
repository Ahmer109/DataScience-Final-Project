import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
import pickle
import os
import datetime
import json
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Advanced Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern theme with transparent charts
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f172a, #1e293b, #334155);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 100%);
        border-right: 1px solid #475569;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
    }
    
    /* Text elements */
    p, span, div, label {
        color: #cbd5e1 !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #60a5fa !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%) !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
        transition: all 0.3s !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%) !important;
    }
    
    /* Input fields */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    /* Input field backgrounds and text */
    input, select, textarea {
        color: white !important;
        background-color: rgba(255,255,255,0.1) !important;
        border: 2px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    /* Selectbox specific styling */
    div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.1) !important;
        border: 2px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    div[data-baseweb="select"] span {
        color: #1e293b !important;
    }
    
    /* Dropdown menu styling */
    [data-baseweb="popover"] {
        background-color: rgba(255,255,255,0.1) !important;
    }
    
    ul[role="listbox"] {
        background-color: rgba(255,255,255,0.1) !important;
    }
    
    li[role="option"] {
        background-color: rgba(255,255,255,0.1) !important;
        color: #1e293b !important;
    }
    
    li[role="option"]:hover {
        background-color: rgba(255,255,255,0.1) !important;
        color: rgba(255,255,255,0.1) !important;
    }
    
    /* Selected option in dropdown */
    li[role="option"][aria-selected="true"] {
        background-color: rgba(255,255,255,0.5) !important;
        color: rgba(255,255,255,0.1) !important;
    }
    
    /* Number input styling */
    input[type="number"] {
        background-color: rgba(255,255,255,0.1) !important;
        color: white !important;
        font-weight: 500 !important;
    }
    
    /* Slider value display */
    .stSlider > div > div > div {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    /* Slider track */
    .stSlider > div > div > div > div {
        background-color: #3b82f6 !important;
    }
    
    /* Radio buttons */
    .stRadio label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    
    .stRadio > div {
        color: #e2e8f0 !important;
    }
    
    /* Info boxes */
    .stAlert {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 10px !important;
    }
    
    /* Dataframe */
    .dataframe {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
    }
    
    .dataframe th {
        background-color: #334155 !important;
        color: #f1f5f9 !important;
    }
    
    .dataframe td {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        border-color: #475569 !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: #064e3b !important;
        color: #a7f3d0 !important;
        border-color: #059669 !important;
    }
    
    .stError {
        background-color: #7f1d1d !important;
        color: #fecaca !important;
        border-color: #dc2626 !important;
    }
    
    .stWarning {
        background-color: #78350f !important;
        color: #fed7aa !important;
        border-color: #d97706 !important;
    }
    
    .stInfo {
        background-color: #1e3a8a !important;
        color: #bfdbfe !important;
        border-color: #3b82f6 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #e2e8f0 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }
    
    .streamlit-expanderContent {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
    }
    
    /* Plotly chart backgrounds - TRANSPARENT */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {
        background-color: transparent !important;
    }
    
    /* Plotly chart text - ensure visibility */
    .js-plotly-plot .plotly .modebar,
    .js-plotly-plot .plotly .modebar-group,
    .js-plotly-plot .plotly .modebar-btn {
        background-color: rgba(30, 41, 59, 0.9) !important;
    }
    
    .js-plotly-plot .plotly text {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
    }
    
    .js-plotly-plot .plotly .legendtext {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
    }
    
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
    }
    
    .js-plotly-plot .plotly .gtitle {
        fill: #e2e8f0 !important;
        color: #e2e8f0 !important;
    }
    
    /* Card-like containers */
    .stContainer {
        background-color: #1e293b !important;
        border-radius: 15px !important;
        padding: 20px !important;
        border: 1px solid #475569 !important;
    }
    
    /* Divider */
    hr {
        border-color: #475569 !important;
    }
    
    /* Learning progress specific */
    .learning-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        display: inline-block;
        margin: 2px;
    }
    
    /* Auto-learning badge */
    .auto-learning-badge {
        background: linear-gradient(135deg, #8b5cf6, #7c3aed);
        color: white;
        padding: 8px 15px;
        border-radius: 25px;
        font-size: 14px;
        font-weight: bold;
        display: inline-block;
        margin: 5px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Chat styles */
    .chat-container {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #475569;
        height: 500px;
        overflow-y: auto;
    }
    
    .user-message {
        background: linear-gradient(135deg, #3b82f6, #1d4ed8);
        color: white;
        padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        word-wrap: break-word;
    }
    
    .bot-message {
        background: rgba(255,255,255,0.1);
        color: #e2e8f0;
        padding: 12px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        margin-right: auto;
        border: 1px solid #475569;
        word-wrap: break-word;
    }
    
    .chat-input {
        background: rgba(255,255,255,0.1) !important;
        border: 2px solid #475569 !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 12px !important;
    }
    
    .quick-question-btn {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid #475569 !important;
        color: #e2e8f0 !important;
        padding: 8px 12px !important;
        margin: 5px 0 !important;
        border-radius: 8px !important;
        text-align: left !important;
        font-size: 14px !important;
    }
    
    .quick-question-btn:hover {
        background: rgba(255,255,255,0.2) !important;
        border-color: #3b82f6 !important;
    }
    
    /* Algorithm comparison cards */
    .algorithm-card {
        background: rgba(255,255,255,0.05);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #475569;
        transition: all 0.3s ease;
    }
    
    .algorithm-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.2);
        border-color: #3b82f6;
    }
    
    .algorithm-card.best {
        border: 2px solid #10b981;
        background: rgba(16, 185, 129, 0.1);
    }
    
    .algorithm-icon {
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = []
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = None
if 'accuracy' not in st.session_state:
    st.session_state.accuracy = 0.0
if 'dataset_available' not in st.session_state:
    st.session_state.dataset_available = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'learning_updates' not in st.session_state:
    st.session_state.learning_updates = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'auto_learning_enabled' not in st.session_state:
    st.session_state.auto_learning_enabled = True
if 'pending_feedback' not in st.session_state:
    st.session_state.pending_feedback = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'current_student_data' not in st.session_state:
    st.session_state.current_student_data = None
if 'selected_algorithm' not in st.session_state:
    st.session_state.selected_algorithm = "Random Forest"
if 'algorithm_comparison' not in st.session_state:
    st.session_state.algorithm_comparison = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'all_models' not in st.session_state:
    st.session_state.all_models = {}
if 'all_encoders' not in st.session_state:
    st.session_state.all_encoders = {}
if 'all_scalers' not in st.session_state:
    st.session_state.all_scalers = {}
if 'all_accuracies' not in st.session_state:
    st.session_state.all_accuracies = {}
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None

# Model file paths
MODEL_FILE = 'student_performance_model.pkl'
ENCODERS_FILE = 'label_encoders.pkl'
SCALER_FILE = 'feature_scaler.pkl'
ALL_MODELS_FILE = 'all_models.pkl'
LEARNING_LOG_FILE = 'learning_log.csv'
PREDICTION_HISTORY_FILE = 'prediction_history.csv'
PENDING_FEEDBACK_FILE = 'pending_feedback.pkl'

# Gemini AI Configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = "AIzaSyAblEEGSq7wO8ihLLuBN1Mjw2vehg_spi0"

def update_random_forest(df):
    """Update Random Forest model"""
    feature_cols = ['gender', 'parental_education', 'lunch', 'test_prep', 
                    'math_score', 'reading_score', 'writing_score', 
                    'study_hours', 'attendance', 'extracurricular', 'avg_score']
    
    X = df[feature_cols].copy()
    y = df['performance']
    
    encoders = {}
    for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    new_accuracy = accuracy_score(y_test, y_pred)
    accuracy_change = new_accuracy - st.session_state.accuracy
    
    save_model(model, encoders, X.columns.tolist(), new_accuracy, algorithm="Random Forest")
    
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.feature_names = X.columns.tolist()
    st.session_state.accuracy = new_accuracy
    st.session_state.model_trained = True
    st.session_state.learning_updates += 1
    st.session_state.last_update = datetime.datetime.now()
    st.session_state.selected_algorithm = "Random Forest"
    
    # Update all models
    st.session_state.all_models["Random Forest"] = model
    st.session_state.all_encoders["Random Forest"] = encoders
    st.session_state.all_accuracies["Random Forest"] = new_accuracy
    save_all_models()
    
    return True, accuracy_change

def update_xgboost(df):
    """Update XGBoost model"""
    feature_cols = ['gender', 'parental_education', 'lunch', 'test_prep', 
                    'math_score', 'reading_score', 'writing_score', 
                    'study_hours', 'attendance', 'extracurricular', 'avg_score']
    
    X = df[feature_cols].copy()
    y = df['performance']
    
    encoders = {}
    for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create label encoder for y for XGBoost
    le_y = LabelEncoder()
    y_train_encoded = le_y.fit_transform(y_train)
    y_test_encoded = le_y.transform(y_test)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train, y_train_encoded)
    
    y_pred_encoded = model.predict(X_test)
    y_pred = le_y.inverse_transform(y_pred_encoded)
    new_accuracy = accuracy_score(y_test, y_pred)
    accuracy_change = new_accuracy - st.session_state.accuracy
    
    save_model(model, encoders, X.columns.tolist(), new_accuracy, algorithm="XGBoost", target_encoder=le_y)
    
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.feature_names = X.columns.tolist()
    st.session_state.accuracy = new_accuracy
    st.session_state.model_trained = True
    st.session_state.learning_updates += 1
    st.session_state.last_update = datetime.datetime.now()
    st.session_state.selected_algorithm = "XGBoost"
    st.session_state.target_encoder = le_y
    
    # Update all models
    st.session_state.all_models["XGBoost"] = model
    st.session_state.all_encoders["XGBoost"] = encoders
    st.session_state.all_accuracies["XGBoost"] = new_accuracy
    st.session_state.all_encoders["XGBoost_y"] = le_y  # Save y encoder for XGBoost
    save_all_models()
    
    return True, accuracy_change

def update_neural_network(df):
    """Update Neural Network model"""
    feature_cols = ['gender', 'parental_education', 'lunch', 'test_prep', 
                    'math_score', 'reading_score', 'writing_score', 
                    'study_hours', 'attendance', 'extracurricular', 'avg_score']
    
    X = df[feature_cols].copy()
    y = df['performance']
    
    encoders = {}
    for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Scale features for neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create label encoder for y
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
    
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    model.fit(X_train, y_train)
    
    y_pred_encoded = model.predict(X_test)
    y_pred = le_y.inverse_transform(y_pred_encoded)
    y_test_original = le_y.inverse_transform(y_test)
    new_accuracy = accuracy_score(y_test_original, y_pred)
    accuracy_change = new_accuracy - st.session_state.accuracy
    
    save_model(model, encoders, feature_cols, new_accuracy, scaler=scaler, algorithm="Neural Network", target_encoder=le_y)
    
    st.session_state.model = model
    st.session_state.encoders = encoders
    st.session_state.feature_names = feature_cols
    st.session_state.accuracy = new_accuracy
    st.session_state.model_trained = True
    st.session_state.learning_updates += 1
    st.session_state.last_update = datetime.datetime.now()
    st.session_state.scaler = scaler
    st.session_state.selected_algorithm = "Neural Network"
    st.session_state.target_encoder = le_y
    
    # Update all models
    st.session_state.all_models["Neural Network"] = model
    st.session_state.all_encoders["Neural Network"] = encoders
    st.session_state.all_scalers["Neural Network"] = scaler
    st.session_state.all_accuracies["Neural Network"] = new_accuracy
    st.session_state.all_encoders["Neural Network_y"] = le_y  # Save y encoder for Neural Network
    save_all_models()
    
    return True, accuracy_change

def efficient_model_update(new_data_points):
    """Efficiently update model with multiple new data points"""
    try:
        if not new_data_points:
            return False, 0
            
        df = st.session_state.df.copy()
        
        new_rows = []
        for data_point in new_data_points:
            new_row = data_point.copy()
            new_row['avg_score'] = (new_row['math_score'] + new_row['reading_score'] + new_row['writing_score']) / 3
            new_rows.append(new_row)
        
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        st.session_state.df = df
        
        # Use current algorithm for update
        if st.session_state.selected_algorithm == "Random Forest":
            return update_random_forest(df)
        elif st.session_state.selected_algorithm == "XGBoost":
            return update_xgboost(df)
        elif st.session_state.selected_algorithm == "Neural Network":
            return update_neural_network(df)
        else:
            return update_random_forest(df)
        
    except Exception as e:
        st.error(f"Batch update error: {e}")
        return False, 0

def process_pending_feedback():
    """Process all pending feedback and update model"""
    if not st.session_state.pending_feedback:
        return
    
    try:
        feedback_by_performance = {}
        for feedback in st.session_state.pending_feedback:
            actual_perf = feedback['actual_performance']
            if actual_perf not in feedback_by_performance:
                feedback_by_performance[actual_perf] = []
            feedback_by_performance[actual_perf].append(feedback['student_data'])
        
        total_updates = 0
        total_accuracy_change = 0
        
        for actual_perf, student_data_list in feedback_by_performance.items():
            for student_data in student_data_list:
                student_data['performance'] = actual_perf
            
            success, accuracy_change = efficient_model_update(student_data_list)
            if success:
                total_updates += len(student_data_list)
                total_accuracy_change += accuracy_change
        
        if total_updates > 0:
            st.success(f"🔄 Processed {total_updates} feedback entries in batch update!")
            if total_accuracy_change != 0:
                change_type = "improved" if total_accuracy_change > 0 else "decreased"
                st.info(f"📈 Model accuracy {change_type} by {abs(total_accuracy_change)*100:.2f}%")
            
            st.session_state.pending_feedback = []
            save_pending_feedback()
            
    except Exception as e:
        st.error(f"Error processing pending feedback: {e}")


def query_gemini(prompt):
    """Query Gemini AI with a concise, complete response"""
    try:
        # Add instruction for concise replies
        instruction = (
            "You are an AI assistant. Respond concisely (2-3 sentences) "
            "but include all key details. Avoid long explanations."
        )

        data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": f"{instruction}\n\nUser: {prompt}"}]
                }
            ]
        }

        response = requests.post(f"{GEMINI_API_URL}?key={GEMINI_API_KEY}", json=data)

        if response.status_code == 200:
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return f"⚠️ Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"⚠️ Connection error: {str(e)}"

def get_dataset_context():
    """Get comprehensive dataset context for AI"""
    if st.session_state.df is None:
        return "No dataset available."
    
    df = st.session_state.df
    
    # Basic dataset info
    context = f"""
DATASET OVERVIEW:
- Total Students: {len(df)}
- Features: {len(df.columns)}
- Performance Distribution: {dict(df['performance'].value_counts())}

KEY STATISTICS:
- Average Math Score: {df['math_score'].mean():.1f}
- Average Reading Score: {df['reading_score'].mean():.1f}
- Average Writing Score: {df['writing_score'].mean():.1f}
- Average Study Hours: {df['study_hours'].mean():.1f}
- Average Attendance: {df['attendance'].mean():.1f}%

PERFORMANCE BREAKDOWN:
"""
    
    # Performance by categories
    for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
        if col in df.columns:
            performance_by_cat = df.groupby(col)['performance'].value_counts().unstack().fillna(0)
            context += f"\n{col.upper()}:\n{performance_by_cat.to_string()}\n"
    
    # Correlation insights
    numeric_cols = ['math_score', 'reading_score', 'writing_score', 'study_hours', 'attendance']
    correlations = df[numeric_cols].corr()
    context += f"\nSCORE CORRELATIONS:\n{correlations.round(2).to_string()}"
    
    return context

def get_prediction_context():
    """Get current prediction context for AI"""
    if not st.session_state.current_prediction or not st.session_state.current_student_data:
        return "No recent prediction available."
    
    student_data = st.session_state.current_student_data
    prediction = st.session_state.current_prediction
    
    context = f"""
CURRENT PREDICTION ANALYSIS:

STUDENT PROFILE:
- Gender: {student_data['gender']}
- Parental Education: {student_data['parental_education']}
- Lunch Type: {student_data['lunch']}
- Test Preparation: {student_data['test_prep']}
- Math Score: {student_data['math_score']}/100
- Reading Score: {student_data['reading_score']}/100
- Writing Score: {student_data['writing_score']}/100
- Study Hours: {student_data['study_hours']} hours/day
- Attendance: {student_data['attendance']}%
- Extracurricular: {student_data['extracurricular']}

PREDICTION RESULTS:
- Predicted Performance: {prediction['performance']}
- Confidence Level: {prediction['confidence']*100:.1f}%
- Average Score: {prediction['avg_score']:.1f}/100
- Algorithm Used: {prediction.get('algorithm', 'Random Forest')}

STRENGTHS AND AREAS FOR IMPROVEMENT:
"""
    
    # Analyze strengths and weaknesses
    scores = {
        'Math': student_data['math_score'],
        'Reading': student_data['reading_score'], 
        'Writing': student_data['writing_score']
    }
    
    strong_subjects = [subj for subj, score in scores.items() if score >= 80]
    weak_subjects = [subj for subj, score in scores.items() if score < 70]
    
    if strong_subjects:
        context += f"- Strong in: {', '.join(strong_subjects)}\n"
    if weak_subjects:
        context += f"- Needs improvement in: {', '.join(weak_subjects)}\n"
    
    # Study habits analysis
    if student_data['study_hours'] >= 6:
        context += "- Good study habits (adequate study hours)\n"
    else:
        context += "- Could benefit from more study time\n"
    
    if student_data['attendance'] >= 85:
        context += "- Excellent attendance\n"
    else:
        context += "- Attendance could be improved\n"
    
    return context

def get_ai_response(user_message):
    """Get AI response with full context"""
    # Build comprehensive context
    dataset_context = get_dataset_context()
    prediction_context = get_prediction_context()
    model_context = f"Model Accuracy: {st.session_state.accuracy*100:.2f}%" if st.session_state.model_trained else "Model not trained"
    algorithm_context = f"Current Algorithm: {st.session_state.selected_algorithm}"
    
    system_prompt = f"""
You are an AI Educational Assistant specialized in student performance analysis. You have access to:

1. DATASET INFORMATION:
{dataset_context}

2. CURRENT PREDICTION CONTEXT:
{prediction_context}

3. MODEL PERFORMANCE:
{model_context}

4. ALGORITHM INFORMATION:
{algorithm_context}

Guidelines:
- Provide data-driven insights based on the available dataset
- Reference specific statistics when relevant
- Offer practical, actionable advice for improvement
- Consider the student's current prediction and profile
- Be encouraging and constructive in your feedback
- If asking about specific predictions, use the current prediction context
- For dataset questions, reference the actual statistics provided
- Consider the algorithm used for predictions in your analysis

User Question: {user_message}

Please provide a helpful, educational response:
"""
    
    return query_gemini(system_prompt)

def save_model(model, encoders, feature_names, accuracy, scaler=None, algorithm="Random Forest", target_encoder=None):
    """Save trained model and encoders to files"""
    try:
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'algorithm': algorithm,
            'target_encoder': target_encoder
        }
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        with open(ENCODERS_FILE, 'wb') as f:
            pickle.dump(encoders, f)
        if scaler:
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump(scaler, f)
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False

def save_all_models():
    """Save all trained models to file"""
    try:
        all_models_data = {
            'models': st.session_state.all_models,
            'encoders': st.session_state.all_encoders,
            'scalers': st.session_state.all_scalers,
            'accuracies': st.session_state.all_accuracies,
            'target_encoder': st.session_state.target_encoder
        }
        with open(ALL_MODELS_FILE, 'wb') as f:
            pickle.dump(all_models_data, f)
        return True
    except Exception as e:
        st.error(f"Error saving all models: {e}")
        return False

def load_model():
    """Load trained model and encoders from files"""
    try:
        if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODERS_FILE):
            return False
            
        with open(MODEL_FILE, 'rb') as f:
            model_data = pickle.load(f)
        with open(ENCODERS_FILE, 'rb') as f:
            encoders = pickle.load(f)
        
        scaler = None
        if os.path.exists(SCALER_FILE):
            with open(SCALER_FILE, 'rb') as f:
                scaler = pickle.load(f)
        
        st.session_state.model = model_data['model']
        st.session_state.encoders = encoders
        st.session_state.feature_names = model_data['feature_names']
        st.session_state.accuracy = model_data['accuracy']
        st.session_state.model_trained = True
        st.session_state.scaler = scaler
        st.session_state.selected_algorithm = model_data.get('algorithm', 'Random Forest')
        st.session_state.target_encoder = model_data.get('target_encoder')
        
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return False

def load_all_models():
    """Load all trained models from file"""
    try:
        if not os.path.exists(ALL_MODELS_FILE):
            return False
            
        with open(ALL_MODELS_FILE, 'rb') as f:
            all_models_data = pickle.load(f)
        
        st.session_state.all_models = all_models_data.get('models', {})
        st.session_state.all_encoders = all_models_data.get('encoders', {})
        st.session_state.all_scalers = all_models_data.get('scalers', {})
        st.session_state.all_accuracies = all_models_data.get('accuracies', {})
        st.session_state.target_encoder = all_models_data.get('target_encoder')
        
        return True
    except Exception as e:
        st.error(f"Error loading all models: {e}")
        return False

def save_prediction_history():
    """Save prediction history to CSV"""
    try:
        if st.session_state.prediction_history:
            df = pd.DataFrame(st.session_state.prediction_history)
            df.to_csv(PREDICTION_HISTORY_FILE, index=False)
    except Exception as e:
        st.warning(f"Could not save prediction history: {e}")

def load_prediction_history():
    """Load prediction history from CSV"""
    try:
        if os.path.exists(PREDICTION_HISTORY_FILE):
            df = pd.read_csv(PREDICTION_HISTORY_FILE)
            st.session_state.prediction_history = df.to_dict('records')
    except Exception as e:
        st.warning(f"Could not load prediction history: {e}")

def save_pending_feedback():
    """Save pending feedback to file"""
    try:
        with open(PENDING_FEEDBACK_FILE, 'wb') as f:
            pickle.dump(st.session_state.pending_feedback, f)
    except Exception as e:
        st.warning(f"Could not save pending feedback: {e}")

def load_pending_feedback():
    """Load pending feedback from file"""
    try:
        if os.path.exists(PENDING_FEEDBACK_FILE):
            with open(PENDING_FEEDBACK_FILE, 'rb') as f:
                st.session_state.pending_feedback = pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load pending feedback: {e}")

def log_prediction(student_data, predicted_performance, confidence, algorithm):
    """Log prediction for future learning"""
    try:
        prediction_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'gender': student_data['gender'],
            'parental_education': student_data['parental_education'],
            'lunch': student_data['lunch'],
            'test_prep': student_data['test_prep'],
            'math_score': student_data['math_score'],
            'reading_score': student_data['reading_score'],
            'writing_score': student_data['writing_score'],
            'study_hours': student_data['study_hours'],
            'attendance': student_data['attendance'],
            'extracurricular': student_data['extracurricular'],
            'predicted_performance': predicted_performance,
            'prediction_confidence': confidence,
            'algorithm': algorithm,
            'actual_performance': None,
            'feedback_provided': False
        }
        
        st.session_state.prediction_history.append(prediction_entry)
        save_prediction_history()
        
        return True
    except Exception as e:
        st.warning(f"Could not log prediction: {e}")
        return False

def log_learning_update(student_data, predicted_performance, actual_performance, accuracy_change):
    """Log learning updates for tracking"""
    try:
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'math_score': student_data['math_score'],
            'reading_score': student_data['reading_score'],
            'writing_score': student_data['writing_score'],
            'study_hours': student_data['study_hours'],
            'attendance': student_data['attendance'],
            'predicted_performance': predicted_performance,
            'actual_performance': actual_performance,
            'accuracy_change': accuracy_change,
            'model_accuracy': st.session_state.accuracy,
            'algorithm': st.session_state.selected_algorithm
        }
        
        if os.path.exists(LEARNING_LOG_FILE):
            log_df = pd.read_csv(LEARNING_LOG_FILE)
            log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        else:
            log_df = pd.DataFrame([log_entry])
        
        log_df.to_csv(LEARNING_LOG_FILE, index=False)
        return True
    except Exception as e:
        st.warning(f"Could not save learning log: {e}")
        return False

# ==================== CONSTANTS ====================
RANDOM_STATE = 42
TEST_SIZE = 0.2

DATASET_FILE = "StudentsPerformance.csv"
def detect_existing_models():
    """Detect and load existing trained models"""
    model_files = {
        'single': MODEL_FILE,
        'all_models': ALL_MODELS_FILE,
        'dataset': DATASET_FILE
    }
    
    existing_models = {}
    
    # Check for single model file
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                model_data = pickle.load(f)
                existing_models['single'] = {
                    'algorithm': model_data.get('algorithm', 'Random Forest'),
                    'accuracy': model_data.get('accuracy', 0.0),
                    'version': model_data.get('version', 'v1.0')
                }
        except Exception as e:
            st.warning(f"Single model file exists but could not be read: {e}")
    
    # Check for all models file
    if os.path.exists(ALL_MODELS_FILE):
        try:
            with open(ALL_MODELS_FILE, 'rb') as f:
                all_models_data = pickle.load(f)
                algorithms = list(all_models_data.get('models', {}).keys())
                existing_models['all_models'] = {
                    'algorithms': algorithms,
                    'accuracies': all_models_data.get('accuracies', {}),
                    'version': all_models_data.get('version', 'v1.0')
                }
        except Exception as e:
            st.warning(f"All models file exists but could not be read: {e}")
    
    return existing_models

def robust_load_model():
    """Robust model loading with fallback options"""
    try:
        # First try to load the comprehensive all_models file
        if os.path.exists(ALL_MODELS_FILE):
            with open(ALL_MODELS_FILE, 'rb') as f:
                all_models_data = pickle.load(f)
                
            st.session_state.all_models = all_models_data.get('models', {})
            st.session_state.all_accuracies = all_models_data.get('accuracies', {})
            st.session_state.all_encoders = all_models_data.get('encoders', {})
            st.session_state.all_scalers = all_models_data.get('scalers', {})
            st.session_state.feature_names = all_models_data.get('feature_names', [])
            st.session_state.target_encoder = all_models_data.get('target_encoder')
            
            # Set the current model to the best available
            if st.session_state.all_models:
                best_algo = max(st.session_state.all_accuracies, key=st.session_state.all_accuracies.get)
                st.session_state.model = st.session_state.all_models[best_algo]
                st.session_state.encoders = st.session_state.all_encoders.get(best_algo, {})
                st.session_state.accuracy = st.session_state.all_accuracies.get(best_algo, 0.0)
                st.session_state.scaler = st.session_state.all_scalers.get(best_algo)
                st.session_state.selected_algorithm = best_algo
                st.session_state.model_trained = True
                
                st.sidebar.success(f"✅ Loaded {best_algo} model ({st.session_state.accuracy*100:.1f}%)")
                return True
        
        # Fallback to single model file
        elif os.path.exists(MODEL_FILE) and os.path.exists(ENCODERS_FILE):
            with open(MODEL_FILE, 'rb') as f:
                model_data = pickle.load(f)
            with open(ENCODERS_FILE, 'rb') as f:
                encoders = pickle.load(f)
                
            st.session_state.model = model_data['model']
            st.session_state.encoders = encoders
            st.session_state.feature_names = model_data.get('feature_names', [])
            st.session_state.accuracy = model_data.get('accuracy', 0.0)
            st.session_state.model_trained = True
            st.session_state.selected_algorithm = model_data.get('algorithm', 'Random Forest')
            
            if os.path.exists(SCALER_FILE):
                with open(SCALER_FILE, 'rb') as f:
                    st.session_state.scaler = pickle.load(f)
            
            # Also populate all_models for consistency
            st.session_state.all_models[st.session_state.selected_algorithm] = st.session_state.model
            st.session_state.all_accuracies[st.session_state.selected_algorithm] = st.session_state.accuracy
            st.session_state.all_encoders[st.session_state.selected_algorithm] = st.session_state.encoders
            if st.session_state.scaler:
                st.session_state.all_scalers[st.session_state.selected_algorithm] = st.session_state.scaler
            
            st.sidebar.success(f"✅ Loaded single {st.session_state.selected_algorithm} model ({st.session_state.accuracy*100:.1f}%)")
            return True
            
    except Exception as e:
        st.sidebar.error(f"❌ Model loading failed: {e}")
    
    return False

def enhanced_save_model(model, encoders, feature_names, accuracy, scaler=None, algorithm="Random Forest", target_encoder=None):
    """Enhanced model saving with versioning"""
    try:
        # Save single model
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy,
            'algorithm': algorithm,
            'target_encoder': target_encoder,
            'version': 'v2.0',
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save encoders
        with open(ENCODERS_FILE, 'wb') as f:
            pickle.dump(encoders, f)
        
        # Save scaler if exists
        if scaler:
            with open(SCALER_FILE, 'wb') as f:
                pickle.dump(scaler, f)
        
        # Save all models
        all_models_data = {
            'models': st.session_state.all_models,
            'accuracies': st.session_state.all_accuracies,
            'encoders': st.session_state.all_encoders,
            'scalers': st.session_state.all_scalers,
            'feature_names': feature_names,
            'target_encoder': target_encoder,
            'version': 'v2.0',
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open(ALL_MODELS_FILE, 'wb') as f:
            pickle.dump(all_models_data, f)
        
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False

def initialize_model_system():
    """Initialize the complete model system"""
    # Detect existing models
    existing_models = detect_existing_models()
    
    if existing_models:
        st.sidebar.info(f"🤖 Found {len(existing_models)} model file(s)")
        
        # Try to load models
        if robust_load_model():
            st.sidebar.success("✅ Models loaded successfully!")
        else:
            st.sidebar.warning("⚠️ Models detected but couldn't load. Will train new ones.")
    
    # Load dataset if available
    if st.session_state.df is None:
        load_kaggle_data()
    
    # Load additional data
    load_prediction_history()
    load_pending_feedback()

# Enhanced train_model function with better error handling
def train_model(df, algorithm="Random Forest"):
    """Enhanced model training with better error handling"""
    try:
        # Check if Performance column exists
        if 'performance' not in df.columns:
            st.error(f"❌ Performance column not found. Available columns: {list(df.columns)}")
            return None, None, None, None, None, None, None
        
        # Prepare data - use consistent feature set
        feature_cols = ['gender', 'parental_education', 'lunch', 'test_prep', 
                       'math_score', 'reading_score', 'writing_score', 
                       'study_hours', 'attendance', 'extracurricular', 'avg_score']
        
        # Ensure all required features exist
        missing_features = [f for f in feature_cols if f not in df.columns]
        if missing_features:
            st.warning(f"⚠️ Missing features: {missing_features}. Creating them...")
            # Create missing features with default values
            for feature in missing_features:
                if feature == 'avg_score':
                    df['avg_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
                else:
                    df[feature] = 0  # Default value for other features
        
        X = df[feature_cols]
        y = df['performance']
        
        # Encode categorical variables
        encoders = {}
        X_encoded = X.copy()
        for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
            if col in X_encoded.columns:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col])
                encoders[col] = le
        
        # Encode target variable
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        encoders['target'] = target_encoder
        
        # Split data with FIXED random_state
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model based on algorithm with FIXED parameters
        if algorithm == "Random Forest":
            model = RandomForestClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            model.fit(X_train_scaled, y_train)
            
        elif algorithm == "XGBoost":
            model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            )
            model.fit(X_train_scaled, y_train)
            
        elif algorithm == "Neural Network":
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20
            )
            model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Store in all_models and all_accuracies
        st.session_state.all_models[algorithm] = model
        st.session_state.all_accuracies[algorithm] = accuracy
        st.session_state.all_encoders[algorithm] = encoders
        st.session_state.all_scalers[algorithm] = scaler
        
        # Update session state for current model
        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.scaler = scaler
        st.session_state.accuracy = accuracy
        st.session_state.model_trained = True
        st.session_state.X_test = X_test_scaled
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.feature_names = feature_cols
        st.session_state.selected_algorithm = algorithm
        st.session_state.target_encoder = target_encoder
        
        # Save models to disk
        enhanced_save_model(model, encoders, feature_cols, accuracy, scaler, algorithm, target_encoder)
        
        return model, encoders, accuracy, X_test_scaled, y_test, y_pred, feature_cols
    
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None, None, None, None, None, None, None

def compare_algorithms(df):
    """Compare all algorithms with consistent parameters"""
    results = {}
    
    # Check if Performance column exists
    if 'performance' not in df.columns:
        st.error(f"❌ Performance column not found. Available columns: {list(df.columns)}")
        return {}, None, None
    
    # Prepare data
    feature_cols = ['gender', 'parental_education', 'lunch', 'test_prep', 
                   'math_score', 'reading_score', 'writing_score', 
                   'study_hours', 'attendance', 'extracurricular', 'avg_score']
    
    X = df[feature_cols]
    y = df['performance']
    
    # Encode categorical variables
    encoders = {}
    X_encoded = X.copy()
    for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
        if col in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            encoders[col] = le
    
    # Encode target variable
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    encoders['target'] = target_encoder
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=RANDOM_STATE,
        max_depth=10,
        min_samples_split=5
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    results['Random Forest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'predictions': rf_pred
    }
    
    # Train and evaluate XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        max_depth=6,
        learning_rate=0.1
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)
    results['XGBoost'] = {
        'model': xgb_model,
        'accuracy': xgb_accuracy,
        'predictions': xgb_pred
    }
    
    # Train and evaluate Neural Network
    nn_model = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    nn_model.fit(X_train_scaled, y_train)
    nn_pred = nn_model.predict(X_test_scaled)
    nn_accuracy = accuracy_score(y_test, nn_pred)
    results['Neural Network'] = {
        'model': nn_model,
        'accuracy': nn_accuracy,
        'predictions': nn_pred
    }
    
    # Find best algorithm
    best_algorithm = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_algorithm]['accuracy']
    
    # Store all models and their accuracies
    st.session_state.all_models = {name: res['model'] for name, res in results.items()}
    st.session_state.all_accuracies = {name: res['accuracy'] for name, res in results.items()}
    st.session_state.all_encoders = {name: encoders for name in results.keys()}
    st.session_state.all_scalers = {name: scaler for name in results.keys()}
    st.session_state.encoders = encoders
    st.session_state.scaler = scaler
    st.session_state.feature_names = feature_cols
    st.session_state.target_encoder = target_encoder
    
    # Set the best model as current
    best_model = results[best_algorithm]['model']
    best_pred = results[best_algorithm]['predictions']
    
    st.session_state.model = best_model
    st.session_state.accuracy = best_accuracy
    st.session_state.model_trained = True
    st.session_state.X_test = X_test_scaled
    st.session_state.y_test = y_test
    st.session_state.y_pred = best_pred
    st.session_state.selected_algorithm = best_algorithm
    
    # Save to disk
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump({
            'model': best_model,
            'encoders': encoders,
            'scaler': scaler,
            'accuracy': best_accuracy,
            'algorithm': best_algorithm,
            'feature_names': feature_cols,
            'target_encoder': target_encoder
        }, f)
    
    with open(ALL_MODELS_FILE, 'wb') as f:
        pickle.dump({
            'models': st.session_state.all_models,
            'accuracies': st.session_state.all_accuracies,
            'encoders': st.session_state.all_encoders,
            'scalers': st.session_state.all_scalers,
            'feature_names': feature_cols,
            'target_encoder': target_encoder
        }, f)
    
    return results, best_algorithm, best_accuracy

def ensure_feature_consistency(input_data, expected_features):
    """Ensure input data has all expected features"""
    input_features = set(input_data.columns)
    expected_features_set = set(expected_features)
    
    # Add missing features with default values
    for feature in expected_features_set - input_features:
        if feature == 'avg_score':
            # Calculate avg_score from the three scores
            input_data['avg_score'] = (input_data['math_score'] + input_data['reading_score'] + input_data['writing_score']) / 3
        else:
            # Add missing feature with default value
            input_data[feature] = 0
    
    # Remove extra features
    for feature in input_features - expected_features_set:
        input_data = input_data.drop(feature, axis=1)
    
    # Ensure correct order
    input_data = input_data[expected_features]
    
    return input_data

def make_prediction(input_data, algorithm, model, encoders, scaler, target_encoder):
    """Make prediction using the specified algorithm"""
    try:
        # Transform categorical features
        for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
            if col in input_data.columns and col in encoders:
                try:
                    input_data[col] = encoders[col].transform(input_data[col])
                except Exception as e:
                    # Handle unknown categories by using a default value
                    input_data[col] = 0

        # Scale features if scaler exists
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        # Make prediction
        if algorithm == "Random Forest":
            prediction_encoded = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
        elif algorithm == "XGBoost":
            prediction_encoded = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
        elif algorithm == "Neural Network":
            prediction_encoded = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]
        
        # Convert back to original label
        if target_encoder is not None:
            prediction = target_encoder.inverse_transform([prediction_encoded])[0]
        else:
            # Fallback if no target encoder
            class_mapping = {0: 'High', 1: 'Medium', 2: 'Low'}
            prediction = class_mapping.get(prediction_encoded, 'Medium')
        
        confidence = max(prediction_proba)
        
        return prediction, confidence
        
    except Exception as e:
        st.error(f"❌ Prediction error for {algorithm}: {str(e)}")
        return None, 0.0

def load_kaggle_data():
    """Load Kaggle Students Performance Dataset"""
    try:
        df = pd.read_csv("StudentsPerformance.csv")
        
        st.sidebar.info(f"📊 Loaded Kaggle Dataset: {len(df)} students")
        
        column_mapping = {
            'math score': 'math_score',
            'reading score': 'reading_score', 
            'writing score': 'writing_score',
            'test preparation course': 'test_prep',
            'parental level of education': 'parental_education',
            'lunch': 'lunch',
            'gender': 'gender',
            'race/ethnicity': 'ethnicity'
        }
        df = df.rename(columns=column_mapping)
        
        if 'study_hours' not in df.columns:
            np.random.seed(42)
            base_study = (df['math_score'] + df['reading_score'] + df['writing_score']) / 300 * 10
            noise = np.random.normal(0, 1.5, len(df))
            df['study_hours'] = np.clip(base_study + noise, 0, 10).astype(int)
        
        if 'attendance' not in df.columns:
            base_attendance = 70 + (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
            noise = np.random.normal(0, 8, len(df))
            df['attendance'] = np.clip(base_attendance + noise, 50, 100).astype(int)
        
        if 'extracurricular' not in df.columns:
            prob_yes = (df['math_score'] + df['reading_score'] + df['writing_score']) / 300 * 0.8
            df['extracurricular'] = np.random.binomial(1, prob_yes, len(df))
            df['extracurricular'] = df['extracurricular'].map({1: 'yes', 0: 'no'})
        
        df['avg_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
        
        def calculate_performance(row):
            score = row['avg_score']
            if row['test_prep'] == 'completed':
                score += 5
            if row['parental_education'] in ["bachelor's degree", "master's degree"]:
                score += 3
            if row['lunch'] == 'standard':
                score += 2
            if row['study_hours'] >= 6:
                score += 4
            if row['attendance'] >= 85:
                score += 3
            if row['extracurricular'] == 'yes':
                score += 2
                
            if score >= 85:
                return 'High'
            elif score >= 65:
                return 'Medium'
            else:
                return 'Low'
        
        df['performance'] = df.apply(calculate_performance, axis=1)
        
        st.session_state.df = df
        st.session_state.dataset_available = True
        return df
        
    except Exception as e:
        st.session_state.dataset_available = False
        
        if load_model():
            st.sidebar.info("📁 Using pre-trained model")
            return None
        else:
            st.sidebar.info("📁 Using sample data")
            return generate_sample_data()

def generate_sample_data():
    """Fallback sample data generator"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'gender': np.random.choice(['male', 'female'], n_samples),
        'parental_education': np.random.choice([
            'some high school', 'high school', 'some college', 
            "associate's degree", "bachelor's degree", "master's degree"
        ], n_samples),
        'lunch': np.random.choice(['standard', 'free/reduced'], n_samples, p=[0.7, 0.3]),
        'test_prep': np.random.choice(['completed', 'none'], n_samples, p=[0.6, 0.4]),
        'math_score': np.random.randint(0, 101, n_samples),
        'reading_score': np.random.randint(0, 101, n_samples),
        'writing_score': np.random.randint(0, 101, n_samples),
        'study_hours': np.random.randint(0, 11, n_samples),
        'attendance': np.random.randint(50, 101, n_samples),
        'extracurricular': np.random.choice(['yes', 'no'], n_samples, p=[0.65, 0.35])
    }
    
    df = pd.DataFrame(data)
    df['avg_score'] = (df['math_score'] + df['reading_score'] + df['writing_score']) / 3
    
    def calculate_performance(row):
        score = row['avg_score']
        if row['test_prep'] == 'completed':
            score += 5
        if row['parental_education'] in ["bachelor's degree", "master's degree"]:
            score += 3
        if row['lunch'] == 'standard':
            score += 2
        if row['study_hours'] >= 6:
            score += 4
        if row['attendance'] >= 85:
            score += 3
        if row['extracurricular'] == 'yes':
            score += 2
            
        if score >= 85:
            return 'High'
        elif score >= 65:
            return 'Medium'
        else:
            return 'Low'
    
    df['performance'] = df.apply(calculate_performance, axis=1)
    
    st.session_state.df = df
    st.session_state.dataset_available = True
    return df

def display_learning_stats():
    """Display model learning statistics in sidebar"""
    if st.session_state.df is not None:
        df = st.session_state.df
        
        st.markdown("<h3 style='color: white !important;'>📈 Learning Progress</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(df))
            st.metric("Learning Updates", st.session_state.learning_updates)
        with col2:
            st.metric("Model Accuracy", f"{st.session_state.accuracy*100:.1f}%")
            if st.session_state.last_update:
                last_update_str = st.session_state.last_update.strftime("%H:%M")
                st.metric("Last Update", last_update_str)
        
        if st.session_state.learning_updates > 0:
            st.markdown("<div class='learning-badge'>🤖 ACTIVE LEARNING</div>", unsafe_allow_html=True)
            if st.session_state.learning_updates >= 10:
                st.markdown("<div class='learning-badge' style='background:linear-gradient(135deg, #8b5cf6, #7c3aed)'>🚀 ADVANCED LEARNER</div>", unsafe_allow_html=True)
        
        if st.session_state.auto_learning_enabled:
            st.markdown("<div class='auto-learning-badge'>🔄 AUTO-LEARNING: ON</div>", unsafe_allow_html=True)
        
        if st.session_state.pending_feedback:
            st.warning(f"📥 {len(st.session_state.pending_feedback)} pending feedback entries")

# Load data on startup
if st.session_state.df is None:
    load_kaggle_data()

# Load prediction history and pending feedback
load_prediction_history()
load_pending_feedback()

# Try to load pre-trained model on startup
if not st.session_state.model_trained:
    load_model()

# Load all models
load_all_models()

# Header
st.markdown("""
<div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
    <h1 style='color: white !important; margin: 0; font-size: 42px;'>🎓 Advanced Student Performance Predictor</h1>
    <p style='color: #e0e7ff !important; margin: 10px 0 0 0; font-size: 18px;'>AI-Powered Academic Analytics & Predictions | Real Kaggle Dataset with Continuous Self-Learning</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h2 style='color: white !important; text-align: center;'>🎯 Navigation</h2>", unsafe_allow_html=True)
    
    tab_selection = st.radio(
        "Choose a section:",
        ["📖 Overview", "🔮 Predict Performance", "📊 Analytics Dashboard", "🤖 Train Model", "📁 Dataset Info", "🔄 Learning Center", "🤖 AI Assistant"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Algorithm Selection
    st.markdown("<h3 style='color: white !important;'>🤖 Algorithm Selection</h3>", unsafe_allow_html=True)
    algorithm = st.selectbox(
        "Choose ML Algorithm:",
        ["Random Forest", "XGBoost", "Neural Network"],
        index=["Random Forest", "XGBoost", "Neural Network"].index(st.session_state.selected_algorithm) if st.session_state.selected_algorithm in ["Random Forest", "XGBoost", "Neural Network"] else 0
    )
    
    if algorithm != st.session_state.selected_algorithm:
        st.session_state.selected_algorithm = algorithm
        # Update current model if it exists in all_models
        if algorithm in st.session_state.all_models:
            st.session_state.model = st.session_state.all_models[algorithm]
            st.session_state.encoders = st.session_state.all_encoders.get(algorithm, {})
            st.session_state.accuracy = st.session_state.all_accuracies.get(algorithm, 0.0)
            st.session_state.scaler = st.session_state.all_scalers.get(algorithm)
            st.session_state.target_encoder = st.session_state.target_encoder
        st.info(f"Algorithm changed to: {algorithm}")
    
    # Algorithm descriptions
    if algorithm == "Random Forest":
        st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.2); padding: 10px; border-radius: 8px; margin: 10px 0;'>
            <p style='color: #bfdbfe !important; margin: 0; font-size: 12px;'>
            <strong>🌲 Random Forest:</strong> Ensemble method, robust, handles mixed data types well
            </p>
        </div>
        """, unsafe_allow_html=True)
    elif algorithm == "XGBoost":
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.2); padding: 10px; border-radius: 8px; margin: 10px 0;'>
            <p style='color: #a7f3d0 !important; margin: 0; font-size: 12px;'>
            <strong>🚀 XGBoost:</strong> Gradient boosting, high performance, good with tabular data
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:  # Neural Network
        st.markdown("""
        <div style='background: rgba(139, 92, 246, 0.2); padding: 10px; border-radius: 8px; margin: 10px 0;'>
            <p style='color: #ddd6fe !important; margin: 0; font-size: 12px;'>
            <strong>🧠 Neural Network:</strong> Deep learning, captures complex patterns, requires more data
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Auto-learning toggle
    st.markdown("<h3 style='color: white !important;'>🤖 Auto-Learning</h3>", unsafe_allow_html=True)
    auto_learning = st.toggle("Enable Continuous Self-Learning", 
                             value=st.session_state.auto_learning_enabled,
                             help="Model automatically learns from predictions and improves over time")
    
    if auto_learning != st.session_state.auto_learning_enabled:
        st.session_state.auto_learning_enabled = auto_learning
        st.rerun()
    
    if st.session_state.auto_learning_enabled:
        st.success("🔄 Auto-learning enabled")
    else:
        st.warning("⏸️ Auto-learning paused")
    
    st.markdown("---")
    
    # Learning Statistics
    display_learning_stats()
    
    st.markdown("---")
    
    # Model status
    if st.session_state.model_trained:
        st.markdown(f"""
        <div style='background: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #22c55e;'>
            <p style='color: white !important; margin: 0; font-size: 14px;'>🤖 Model Status</p>
            <p style='color: #22c55e !important; margin: 5px 0 0 0; font-size: 16px; font-weight: bold;'>TRAINED & LEARNING</p>
            <p style='color: #86efac !important; margin: 0; font-size: 12px;'>Algorithm: {st.session_state.selected_algorithm}</p>
            <p style='color: #86efac !important; margin: 0; font-size: 12px;'>Accuracy: {st.session_state.accuracy*100:.1f}%</p>
            <p style='color: #86efac !important; margin: 0; font-size: 12px;'>Updates: {st.session_state.learning_updates}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.2); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #ef4444;'>
            <p style='color: white !important; margin: 0; font-size: 14px;'>🤖 Model Status</p>
            <p style='color: #ef4444 !important; margin: 5px 0 0 0; font-size: 16px; font-weight: bold;'>NOT TRAINED</p>
            <p style='color: #fca5a5 !important; margin: 0; font-size: 12px;'>Train model first</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Dataset status
    if st.session_state.dataset_available and st.session_state.df is not None:
        st.markdown(f"""
        <div style='background: rgba(34, 197, 94, 0.2); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #22c55e;'>
            <p style='color: white !important; margin: 0; font-size: 14px;'>📁 Dataset Status</p>
            <p style='color: #22c55e !important; margin: 5px 0 0 0; font-size: 16px; font-weight: bold;'>AVAILABLE</p>
            <p style='color: #86efac !important; margin: 0; font-size: 12px;'>{len(st.session_state.df)} students</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background: rgba(245, 158, 11, 0.2); padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 4px solid #f59e0b;'>
            <p style='color: white !important; margin: 0; font-size: 14px;'>📁 Dataset Status</p>
            <p style='color: #f59e0b !important; margin: 5px 0 0 0; font-size: 16px; font-weight: bold;'>NOT AVAILABLE</p>
            <p style='color: #fed7aa !important; margin: 0; font-size: 12px;'>Using pre-trained model</p>
        </div>
        """, unsafe_allow_html=True)

# Main content based on tab selection
if tab_selection == "📖 Overview":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>📖 Project Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 30px; border-radius: 20px; margin-bottom: 20px;'>
            <h3 style='color: #f1f5f9 !important;'>🎓 Advanced Student Performance Predictor</h3>
            <p style='color: #cbd5e1 !important; font-size: 16px; line-height: 1.6;'>
            A comprehensive AI-powered web application that predicts student academic performance using multiple machine learning algorithms 
            and provides data-driven insights for educational improvement. This system combines traditional data analysis 
            with cutting-edge AI capabilities to help educators and students understand performance patterns.
            </p>
            <p style='color: #60a5fa !important; font-size: 14px; font-weight: bold; margin-top: 15px;'>
            🔄 Smart Data Handling: If Kaggle dataset is unavailable, the system automatically uses pre-trained models. 
            If both are unavailable, it generates sample data for seamless predictions.
            </p>
            <p style='color: #8b5cf6 !important; font-size: 14px; font-weight: bold; margin-top: 10px;'>
            🤖 Multi-Algorithm Support: Choose between Random Forest, XGBoost, and Neural Network for optimal predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features Section
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🚀 Key Features</h3>", unsafe_allow_html=True)
        
        features = [
            {
                "icon": "🔮",
                "title": "Smart Predictions",
                "description": "Predict student performance using multiple ML algorithms with real-time accuracy metrics"
            },
            {
                "icon": "🤖",
                "title": "Multi-Algorithm Support",
                "description": "Choose from Random Forest, XGBoost, or Neural Network for optimal performance"
            },
            {
                "icon": "🔄",
                "title": "Continuous Self-Learning",
                "description": "Model automatically improves over time with user feedback and new data"
            },
            {
                "icon": "📊",
                "title": "Interactive Analytics",
                "description": "Visualize data patterns with interactive charts and performance dashboards"
            },
            {
                "icon": "💾",
                "title": "Auto-Save Model",
                "description": "Automatically save and load trained models for persistent learning"
            },
            {
                "icon": "🎯",
                "title": "Personalized Recommendations",
                "description": "Get tailored study strategies based on individual student profiles"
            }
        ]
        
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            with cols[idx % 2]:
                st.markdown(f"""
                <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 4px solid #3b82f6;'>
                    <h4 style='color: #f1f5f9 !important; margin: 0 0 10px 0;'>{feature['icon']} {feature['title']}</h4>
                    <p style='color: #cbd5e1 !important; margin: 0; font-size: 14px;'>{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        # Technology Stack
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🛠️ Technology Stack</h3>", unsafe_allow_html=True)
        
        technologies = [
            {"name": "Streamlit", "purpose": "Web Framework"},
            {"name": "Scikit-learn", "purpose": "Machine Learning"},
            {"name": "XGBoost", "purpose": "Gradient Boosting"},
            {"name": "Neural Networks", "purpose": "Deep Learning"},
            {"name": "Plotly", "purpose": "Data Visualization"},
            {"name": "Gemini AI", "purpose": "AI Assistant"},
            {"name": "Pandas/Numpy", "purpose": "Data Processing"},
            {"name": "Kaggle Dataset", "purpose": "Real Student Data"}
        ]
        
        for tech in technologies:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
                <p style='color: #60a5fa !important; margin: 0; font-weight: bold;'>{tech['name']}</p>
                <p style='color: #cbd5e1 !important; margin: 0; font-size: 12px;'>{tech['purpose']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Algorithm Comparison
        st.markdown("<h3 style='color: #f1f5f9 !important; margin-top: 20px;'>🤖 Supported Algorithms</h3>", unsafe_allow_html=True)
        
        algorithms = [
            {
                "name": "🌲 Random Forest",
                "strength": "Robust, handles mixed data well",
                "best_for": "General purpose, reliable predictions"
            },
            {
                "name": "🚀 XGBoost", 
                "strength": "High performance, fast training",
                "best_for": "Tabular data, competition-grade results"
            },
            {
                "name": "🧠 Neural Network",
                "strength": "Complex pattern recognition",
                "best_for": "Large datasets, non-linear relationships"
            }
        ]
        
        for algo in algorithms:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
                <p style='color: #60a5fa !important; margin: 0; font-weight: bold; font-size: 14px;'>{algo['name']}</p>
                <p style='color: #cbd5e1 !important; margin: 2px 0; font-size: 11px;'><strong>Strength:</strong> {algo['strength']}</p>
                <p style='color: #cbd5e1 !important; margin: 0; font-size: 11px;'><strong>Best for:</strong> {algo['best_for']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Quick Stats
    st.markdown("<h3 style='color: #f1f5f9 !important; margin-top: 20px;'>📈 Project Stats</h3>", unsafe_allow_html=True)
    
    if st.session_state.df is not None:
        total_students = len(st.session_state.df)
        high_performers = (st.session_state.df['performance'] == 'High').sum()
        success_rate = (high_performers / total_students) * 100
        data_source = "Kaggle Dataset" if st.session_state.dataset_available else "Sample Data"
    else:
        total_students = "N/A"
        success_rate = "N/A"
        data_source = "Pre-trained Model"
    
    stats = [
        {"label": "Total Students", "value": total_students},
        {"label": "Current Algorithm", "value": st.session_state.selected_algorithm},
        {"label": "Model Accuracy", "value": f"{st.session_state.accuracy*100:.1f}%" if st.session_state.model_trained else "Not Trained"},
        {"label": "Learning Updates", "value": st.session_state.learning_updates},
        {"label": "Success Rate", "value": f"{success_rate:.1f}%" if st.session_state.df is not None else "N/A"},
        {"label": "Data Source", "value": data_source}
    ]
    
    cols = st.columns(3)
    for idx, stat in enumerate(stats):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
                <p style='color: #cbd5e1 !important; margin: 0; font-size: 12px;'>{stat['label']}</p>
                <p style='color: #60a5fa !important; margin: 0; font-weight: bold; font-size: 16px;'>{stat['value']}</p>
            </div>
            """, unsafe_allow_html=True)

elif tab_selection == "🔮 Predict Performance":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>🔮 Student Performance Prediction</h2>", unsafe_allow_html=True)

    # Show model status
    # if st.session_state.model_trained:
    #     st.success(f"✅ Using trained {st.session_state.selected_algorithm} model (Accuracy: {st.session_state.accuracy*100:.2f}%)")
    # else:
    #     st.warning("⚠️ No trained model available. Please train the model first in the 'Train Model' section.")
    # Enhanced initialization
    initialize_model_system()

    # Display model status in sidebar
    if st.session_state.model_trained:
        st.sidebar.success(f"✅ {st.session_state.selected_algorithm} Model Active")
        st.sidebar.metric("Model Accuracy", f"{st.session_state.accuracy*100:.1f}%")
    else:
        st.sidebar.warning("⚠️ No trained model detected")

    # Model selection for prediction
    st.markdown("<h3 style='color: #f1f5f9 !important;'>🤖 Select Prediction Model</h3>", unsafe_allow_html=True)
    
    available_models = []
    if st.session_state.all_models:
        for algo in ["Random Forest", "XGBoost", "Neural Network"]:
            if algo in st.session_state.all_models:
                accuracy = st.session_state.all_accuracies.get(algo, 0.0)
                available_models.append(f"{algo} ({accuracy*100:.1f}%)")
    
    if available_models:
        selected_model_display = st.selectbox(
            "Choose model for prediction:",
            available_models,
            index=available_models.index(f"{st.session_state.selected_algorithm} ({st.session_state.accuracy*100:.1f}%)") if f"{st.session_state.selected_algorithm} ({st.session_state.accuracy*100:.1f}%)" in available_models else 0
        )
        
        # Extract algorithm name from display string
        selected_algorithm = selected_model_display.split(" (")[0]
        
        if selected_algorithm != st.session_state.selected_algorithm and selected_algorithm in st.session_state.all_models:
            st.session_state.selected_algorithm = selected_algorithm
            st.session_state.model = st.session_state.all_models[selected_algorithm]
            st.session_state.encoders = st.session_state.all_encoders.get(selected_algorithm, {})
            st.session_state.accuracy = st.session_state.all_accuracies.get(selected_algorithm, 0.0)
            st.session_state.scaler = st.session_state.all_scalers.get(selected_algorithm)
            st.info(f"🔄 Switched to {selected_algorithm} model")

    col1 = st.columns(1)[0]

    with col1:
        st.markdown("<h3 style='color: #f1f5f9 !important;'>📝 Student Information</h3>", unsafe_allow_html=True)
        gender = st.selectbox("👤 Gender", ["female", "male"])
        parental_education = st.selectbox(
            "🎓 Parental Education Level",
            ["some high school", "high school", "some college",
             "associate's degree", "bachelor's degree", "master's degree"]
        )
        lunch = st.selectbox("🍽️ Lunch Type", ["standard", "free/reduced"])
        test_prep = st.selectbox("📚 Test Preparation Course", ["completed", "none"])
        st.markdown("<h4 style='color: #f1f5f9 !important;'>📊 Test Scores</h4>", unsafe_allow_html=True)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            math_score = st.number_input("Math", 0, 100, 75)
        with col_b:
            reading_score = st.number_input("Reading", 0, 100, 75)
        with col_c:
            writing_score = st.number_input("Writing", 0, 100, 75)

        study_hours = st.slider("⏰ Daily Study Hours", 0, 10, 5)
        attendance = st.slider("📅 Attendance Rate (%)", 0, 100, 85)
        extracurricular = st.selectbox("🎨 Extracurricular Activities", ["yes", "no"])

        predict_button = st.button("🎯 Predict Performance", use_container_width=True)

    # Prediction logic
    if predict_button:
        if not st.session_state.model_trained:
            if not load_model():
                if st.session_state.dataset_available and st.session_state.df is not None:
                    with st.spinner(f"Training {st.session_state.selected_algorithm} model..."):
                        result = train_model(st.session_state.df, st.session_state.selected_algorithm)
                        if result[0] is not None:
                            model, encoders, accuracy, X_test, y_test, y_pred, feature_names = result
                            st.session_state.model = model
                            st.session_state.encoders = encoders
                            st.session_state.feature_names = feature_names
                            st.session_state.model_trained = True
                            st.session_state.X_test = X_test
                            st.session_state.y_test = y_test
                            st.session_state.y_pred = y_pred
                            st.session_state.accuracy = accuracy
                            st.success(f"✅ {st.session_state.selected_algorithm} model trained successfully! Accuracy: {accuracy*100:.2f}%")
                        else:
                            st.error("❌ Failed to train model")
                            st.stop()
                else:
                    st.error("❌ No dataset available to train model and no pre-trained model found.")
                    st.stop()

        # Prepare input data with ALL expected features
        input_data = pd.DataFrame({
            'gender': [gender],
            'parental_education': [parental_education],
            'lunch': [lunch],
            'test_prep': [test_prep],
            'math_score': [math_score],
            'reading_score': [reading_score],
            'writing_score': [writing_score],
            'study_hours': [study_hours],
            'attendance': [attendance],
            'extracurricular': [extracurricular],
            'avg_score': [(math_score + reading_score + writing_score) / 3]
        })
        
        # Ensure we have the right feature names from the trained model
        if hasattr(st.session_state, 'feature_names') and st.session_state.feature_names:
            expected_features = st.session_state.feature_names
            input_data = ensure_feature_consistency(input_data, expected_features)
        else:
            expected_features = ['gender', 'parental_education', 'lunch', 'test_prep', 
                            'math_score', 'reading_score', 'writing_score', 
                            'study_hours', 'attendance', 'extracurricular', 'avg_score']
            input_data = ensure_feature_consistency(input_data, expected_features)

        # Make prediction
        try:
            prediction, confidence = make_prediction(
                input_data, 
                st.session_state.selected_algorithm,
                st.session_state.model,
                st.session_state.encoders,
                st.session_state.scaler,
                st.session_state.target_encoder
            )
            
            if prediction is not None:
                avg_score = (math_score + reading_score + writing_score) / 3

                student_data = {
                    'gender': gender,
                    'parental_education': parental_education,
                    'lunch': lunch,
                    'test_prep': test_prep,
                    'math_score': math_score,
                    'reading_score': reading_score,
                    'writing_score': writing_score,
                    'study_hours': study_hours,
                    'attendance': attendance,
                    'extracurricular': extracurricular
                }

                log_prediction(student_data, prediction, confidence, st.session_state.selected_algorithm)

                # Store in session state
                st.session_state.current_prediction = {
                    'performance': prediction,
                    'confidence': confidence,
                    'avg_score': avg_score,
                    'algorithm': st.session_state.selected_algorithm
                }
                st.session_state.current_student_data = student_data
            else:
                st.error("❌ Prediction failed")
                st.stop()
            
        except Exception as e:
            st.error(f"❌ Prediction failed: {str(e)}")
            st.stop()

    # Display Prediction Results
    if "current_prediction" in st.session_state and st.session_state.current_prediction is not None:
        prediction = st.session_state.current_prediction['performance']
        confidence = st.session_state.current_prediction['confidence']
        avg_score = st.session_state.current_prediction['avg_score']
        algorithm = st.session_state.current_prediction['algorithm']
        student_data = st.session_state.current_student_data

        st.markdown("<h3 style='color: #f1f5f9 !important;'>🎯 Prediction Results</h3>", unsafe_allow_html=True)

        color_map = {
            "High": "linear-gradient(135deg, #10b981, #059669)",
            "Medium": "linear-gradient(135deg, #f59e0b, #d97706)",
            "Low": "linear-gradient(135deg, #ef4444, #dc2626)"
        }
        st.markdown(f'''
            <div style="background: {color_map.get(prediction, '#64748b')};
                        color: white; padding: 30px; border-radius: 20px; text-align: center;
                        font-size: 24px; font-weight: bold; margin: 20px 0;">
                {prediction.upper()} PERFORMANCE
            </div>
        ''', unsafe_allow_html=True)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Average Score", f"{avg_score:.1f}")
        col_m2.metric("Confidence", f"{confidence*100:.1f}%")
        col_m3.metric("Prediction", prediction)
        col_m4.metric("Algorithm", algorithm)
        st.info(f"🧠 Using {algorithm} model with {st.session_state.accuracy*100:.1f}% accuracy")

        # Radar Chart
        st.markdown("<h3 style='color: #f1f5f9 !important;'>📊 Performance Profile</h3>", unsafe_allow_html=True)
        categories = ['Math', 'Reading', 'Writing', 'Study Habits', 'Attendance']
        values = [math_score, reading_score, writing_score, study_hours*10, attendance]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            fillcolor='rgba(139, 92, 246, 0.3)',
            line=dict(color='rgb(139, 92, 246)', width=2),
            name='Performance'
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

        # AI Analysis with improvement suggestions
        st.markdown("---")
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🤖 AI Improvement Suggestions</h3>", unsafe_allow_html=True)

        if st.button("🎯 Get AI Improvement Plan", use_container_width=True):
            with st.spinner("🤖 Generating improvement plan..."):
                try:
                    ai_prompt = f"""
                    Student profile:
                    Gender: {gender}
                    Parental Education: {parental_education}
                    Lunch: {lunch}
                    Test Prep: {test_prep}
                    Scores: Math {math_score}, Reading {reading_score}, Writing {writing_score}
                    Study Hours: {study_hours}, Attendance: {attendance}%
                    Extracurricular: {extracurricular}
                    Current predicted performance: {prediction}
                    Algorithm used: {algorithm}

                    Provide actionable suggestions to improve performance to HIGH, in bullet points.
                    Include study habits, attendance, test preparation, motivation, and weak subjects focus.
                    Consider the algorithm's strengths in your recommendations.
                    """
                    ai_analysis = get_ai_response(ai_prompt)
                    
                    # Display analysis nicely
                    for line in ai_analysis.split("\n"):
                        if line.strip():
                            st.info(line.strip())
                except Exception as e:
                    st.error(f"❌ Failed to generate AI suggestions. Error: {str(e)}")

        # Recommendations
        st.markdown("<h3 style='color: #f1f5f9 !important;'>💡 Personalized Recommendations</h3>", unsafe_allow_html=True)
        recommendations = []
        if prediction == 'High':
            recommendations = [
                "✅ Maintain consistent study habits",
                "🎯 Consider advanced placement courses",
                "🌟 Explore leadership opportunities",
                "🤝 Mentor other students"
            ]
        elif prediction == 'Medium':
            recommendations = [
                "📚 Increase study hours to 6-8 per day",
                "👥 Join study groups for peer learning",
                "🎯 Focus on weaker subjects",
                "📝 Utilize test preparation resources"
            ]
        else:
            recommendations = [
                "🆘 Seek tutoring support immediately",
                "📅 Create a structured study schedule",
                "📈 Improve attendance to above 85%",
                "💬 Discuss with academic counselor"
            ]
        for rec in recommendations:
            st.info(rec)

elif tab_selection == "🤖 Train Model":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>🤖 Model Training & Evaluation</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("📁 Dataset not available for training.")
        if st.session_state.model_trained:
            st.info(f"✅ Using existing pre-trained {st.session_state.selected_algorithm} model with {st.session_state.accuracy*100:.2f}% accuracy")
        else:
            st.error("❌ No dataset available and no pre-trained model found.")
    else:
        # Algorithm comparison section
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🔍 Algorithm Comparison</h3>", unsafe_allow_html=True)
        
        if st.button("🔄 Compare All Algorithms", use_container_width=True):
            with st.spinner("Training and comparing all algorithms..."):
                results, best_algorithm, best_accuracy = compare_algorithms(st.session_state.df)
                st.session_state.algorithm_comparison = results
                
                # Display comparison results
                st.markdown("<h4 style='color: #f1f5f9 !important;'>📊 Algorithm Performance</h4>", unsafe_allow_html=True)
                
                cols = st.columns(3)
                
                for idx, (algo_name, result) in enumerate(results.items()):
                    with cols[idx]:
                        is_best = algo_name == best_algorithm
                        card_class = "algorithm-card best" if is_best else "algorithm-card"
                        badge = " 🏆" if is_best else ""
                        
                        st.markdown(f"""
                        <div class='{card_class}'>
                            <div class='algorithm-icon'>{"🌲" if algo_name == "Random Forest" else "🚀" if algo_name == "XGBoost" else "🧠"}</div>
                            <h4 style='color: #f1f5f9 !important; margin: 10px 0;'>{algo_name}{badge}</h4>
                            <p style='color: #60a5fa !important; font-size: 24px; font-weight: bold; margin: 0;'>{result['accuracy']*100:.2f}%</p>
                            <p style='color: #cbd5e1 !important; margin: 5px 0; font-size: 12px;'>Accuracy Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.success(f"🎯 Best Algorithm: {best_algorithm} ({best_accuracy*100:.2f}% accuracy)")
                st.session_state.selected_algorithm = best_algorithm
                st.info(f"🤖 Selected {best_algorithm} as the primary algorithm for predictions")
        
        st.markdown("---")
        
        # Individual algorithm training
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🚀 Train Specific Algorithm</h3>", unsafe_allow_html=True)
        
        col_train1, col_train2, col_train3 = st.columns(3)
        
        with col_train1:
            if st.button("🌲 Train Random Forest", use_container_width=True):
                with st.spinner("Training Random Forest model..."):
                    result = train_model(st.session_state.df, "Random Forest")
                    if result[0] is not None:
                        st.success("✅ Random Forest model trained successfully!")
                        st.session_state.selected_algorithm = "Random Forest"
                        st.rerun()
        
        with col_train2:
            if st.button("🚀 Train XGBoost", use_container_width=True):
                with st.spinner("Training XGBoost model..."):
                    result = train_model(st.session_state.df, "XGBoost")
                    if result[0] is not None:
                        st.success("✅ XGBoost model trained successfully!")
                        st.session_state.selected_algorithm = "XGBoost"
                        st.rerun()
        
        with col_train3:
            if st.button("🧠 Train Neural Network", use_container_width=True):
                with st.spinner("Training Neural Network model..."):
                    result = train_model(st.session_state.df, "Neural Network")
                    if result[0] is not None:
                        st.success("✅ Neural Network model trained successfully!")
                        st.session_state.selected_algorithm = "Neural Network"
                        st.rerun()
        
        # Train all models button
        st.markdown("---")
        if st.button("🤖 Train All Models", use_container_width=True):
            with st.spinner("Training all models..."):
                algorithms = ["Random Forest", "XGBoost", "Neural Network"]
                results = {}
                
                for algo in algorithms:
                    with st.spinner(f"Training {algo}..."):
                        result = train_model(st.session_state.df, algo)
                        if result[0] is not None:
                            model, encoders, accuracy, X_test, y_test, y_pred, feature_names = result
                            results[algo] = accuracy
                
                if results:
                    best_algo = max(results, key=results.get)
                    st.success(f"✅ All models trained successfully! Best: {best_algo} ({results[best_algo]*100:.2f}%)")
                    st.session_state.selected_algorithm = best_algo
                    st.rerun()
    
    if st.session_state.model_trained:
        st.markdown("<br>", unsafe_allow_html=True)
        
        if os.path.exists(MODEL_FILE):
            file_size = os.path.getsize(MODEL_FILE) / 1024
            st.success(f"📁 Model file saved: 'student_performance_model.pkl' ({file_size:.1f} KB)")
        
        if os.path.exists(ALL_MODELS_FILE):
            file_size = os.path.getsize(ALL_MODELS_FILE) / 1024
            st.success(f"📁 All models file saved: 'all_models.pkl' ({file_size:.1f} KB)")
        
        # Show all trained models
        if st.session_state.all_models:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📋 Trained Models</h3>", unsafe_allow_html=True)
            
            cols = st.columns(3)
            for idx, (algo_name, model) in enumerate(st.session_state.all_models.items()):
                with cols[idx]:
                    # Get the stored accuracy for this specific algorithm
                    accuracy = st.session_state.all_accuracies.get(algo_name, 0.0)
                    is_current = algo_name == st.session_state.selected_algorithm
                    status = "🟢 CURRENT" if is_current else "⚪ AVAILABLE"
                    
                    st.markdown(f"""
                    <div class='algorithm-card {'best' if is_current else ''}'>
                        <div class='algorithm-icon'>{"🌲" if algo_name == "Random Forest" else "🚀" if algo_name == "XGBoost" else "🧠"}</div>
                        <h4 style='color: #f1f5f9 !important; margin: 10px 0;'>{algo_name}</h4>
                        <p style='color: #60a5fa !important; font-size: 20px; font-weight: bold; margin: 0;'>{accuracy*100:.2f}%</p>
                        <p style='color: #cbd5e1 !important; margin: 5px 0; font-size: 12px;'>Accuracy</p>
                        <p style='color: {'#10b981' if is_current else '#94a3b8'} !important; margin: 0; font-size: 10px; font-weight: bold;'>{status}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📊 Model Performance</h3>", unsafe_allow_html=True)
            
            accuracy_value = st.session_state.accuracy
            st.metric("Accuracy", f"{accuracy_value*100:.2f}%")
            st.metric("Algorithm", st.session_state.selected_algorithm)
            
            if st.session_state.y_test is not None and st.session_state.y_pred is not None:
                cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
                fig = px.imshow(
                    cm, 
                    text_auto=True, 
                    labels=dict(x="Predicted", y="Actual"),
                    x=['High', 'Low', 'Medium'],
                    y=['High', 'Low', 'Medium'],
                    color_continuous_scale='Blues'
                )
                fig.update_layout(
                    height=400, 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0', size=12),
                    xaxis=dict(color='#e2e8f0'),
                    yaxis=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Confusion matrix data not available")
        
        with col2:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>🎯 Feature Importance</h3>", unsafe_allow_html=True)
            
            if st.session_state.model is not None and hasattr(st.session_state.model, 'feature_importances_'):
                importances = st.session_state.model.feature_importances_
                feature_imp = pd.DataFrame({
                    'feature': st.session_state.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(
                    feature_imp, 
                    x='importance', 
                    y='feature',
                    orientation='h',
                    color='importance',
                    color_continuous_scale='Purples'
                )
                fig.update_layout(
                    height=400, 
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0', size=12),
                    xaxis=dict(color='#e2e8f0'),
                    yaxis=dict(color='#e2e8f0')
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Feature importance not available for Neural Network models")
        
        if st.session_state.y_test is not None and st.session_state.y_pred is not None:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📋 Detailed Classification Report</h3>", unsafe_allow_html=True)
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    
    elif st.session_state.model_trained:
        st.warning("⚠️ Model is trained but evaluation data is not available.")
        st.info(f"✅ Using pre-trained {st.session_state.selected_algorithm} model with {st.session_state.accuracy*100:.2f}% accuracy")
        
        if os.path.exists(MODEL_FILE):
            file_size = os.path.getsize(MODEL_FILE) / 1024
            st.success(f"📁 Pre-trained model available: 'student_performance_model.pkl' ({file_size:.1f} KB)")
    
    else:
        st.info("👆 Click the 'Compare All Algorithms' button to find the best algorithm, or train specific algorithms directly.")
        
        
elif tab_selection == "📊 Analytics Dashboard":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>📊 Analytics Dashboard</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("📊 Dataset not available for analytics. Please ensure the dataset file exists.")
        if st.session_state.model_trained:
            st.info("🔮 You can still use the prediction feature with the pre-trained model.")
    else:
        df = st.session_state.df
        
        # Learning progress metrics
        if st.session_state.learning_updates > 0:
            col_learn1, col_learn2, col_learn3, col_learn4 = st.columns(4)
            with col_learn1:
                st.metric("🔄 Learning Updates", st.session_state.learning_updates)
            with col_learn2:
                st.metric("📈 Current Accuracy", f"{st.session_state.accuracy*100:.1f}%")
            with col_learn3:
                st.metric("🤖 Algorithm", st.session_state.selected_algorithm)
            with col_learn4:
                if st.session_state.last_update:
                    last_update = st.session_state.last_update.strftime("%Y-%m-%d %H:%M")
                    st.metric("🕒 Last Update", last_update)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("👥 Total Students", len(df))
        with col2:
            st.metric("📈 Avg Score", f"{df['avg_score'].mean():.1f}")
        with col3:
            st.metric("🌟 High Performers", (df['performance']=='High').sum())
        with col4:
            st.metric("✅ Success Rate", f"{(df['performance']=='High').sum()/len(df)*100:.1f}%")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📊 Performance Distribution</h3>", unsafe_allow_html=True)
            perf_counts = df['performance'].value_counts()
            colors = ['#10b981', '#f59e0b', '#ef4444']
            fig = px.pie(
                values=perf_counts.values, 
                names=perf_counts.index,
                color_discrete_sequence=colors,
                hole=0.4
            )
            fig.update_layout(
                height=400, 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0', size=14),
                legend=dict(font=dict(color='#e2e8f0'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📈 Score Distribution</h3>", unsafe_allow_html=True)
            fig = px.histogram(
                df, 
                x='avg_score', 
                nbins=20,
                color_discrete_sequence=['#8b5cf6']
            )
            fig.update_layout(
                height=400, 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0', size=12),
                xaxis=dict(color='#e2e8f0'),
                yaxis=dict(color='#e2e8f0')
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Add algorithm comparison visualization if available
        if st.session_state.algorithm_comparison:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>🤖 Algorithm Comparison</h3>", unsafe_allow_html=True)
            algo_results = st.session_state.algorithm_comparison
            algo_names = list(algo_results.keys())
            accuracies = [algo_results[name]['accuracy'] for name in algo_names]
            
            fig = px.bar(
                x=algo_names,
                y=accuracies,
                color=algo_names,
                color_discrete_sequence=['#3b82f6', '#10b981', '#8b5cf6'],
                text=[f'{acc*100:.2f}%' for acc in accuracies]
            )
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0', size=12),
                xaxis=dict(color='#e2e8f0', title='Algorithm'),
                yaxis=dict(color='#e2e8f0', title='Accuracy', tickformat='.0%'),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # Show all models performance
        if st.session_state.all_accuracies:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📈 All Models Performance</h3>", unsafe_allow_html=True)
            
            models_data = []
            for algo, accuracy in st.session_state.all_accuracies.items():
                models_data.append({
                    'Algorithm': algo,
                    'Accuracy': accuracy * 100,
                    'Icon': '🌲' if algo == "Random Forest" else '🚀' if algo == "XGBoost" else '🧠'
                })
            
            models_df = pd.DataFrame(models_data)
            fig = px.bar(
                models_df,
                x='Algorithm',
                y='Accuracy',
                color='Algorithm',
                color_discrete_sequence=['#3b82f6', '#10b981', '#8b5cf6'],
                text='Accuracy',
                title='All Trained Models Accuracy Comparison'
            )
            fig.update_layout(
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0', size=12),
                xaxis=dict(color='#e2e8f0'),
                yaxis=dict(color='#e2e8f0', title='Accuracy (%)'),
                showlegend=False
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

elif tab_selection == "📁 Dataset Info":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>📁 Dataset Information</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("📁 Dataset not available.")
        if st.session_state.model_trained:
            st.info(f"✅ Using pre-trained model with {st.session_state.accuracy*100:.2f}% accuracy")
    else:
        df = st.session_state.df
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;'>
                <h3 style='color: #f1f5f9 !important;'>📚 Students Performance in Exams</h3>
                <p style='color: #cbd5e1 !important;'>This dataset contains student achievement data in secondary education 
                from two Portuguese schools. It includes student grades, demographic, social, and school-related features.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.metric("Dataset Source", "Kaggle")
            st.metric("Records", len(df))
            st.metric("Features", len(df.columns))
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #f1f5f9 !important;'>📋 Dataset Preview</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🔍 Column Information</h3>", unsafe_allow_html=True)
        
        column_info = {
            'gender': 'Student gender (male/female)',
            'parental_education': "Parent's education level",
            'lunch': 'Lunch type (standard/free/reduced)',
            'test_prep': 'Test preparation course completion',
            'math_score': 'Math exam score (0-100)',
            'reading_score': 'Reading exam score (0-100)',
            'writing_score': 'Writing exam score (0-100)',
            'study_hours': 'Daily study hours (0-10)',
            'attendance': 'Attendance rate (%)',
            'extracurricular': 'Extracurricular activities participation',
            'avg_score': 'Average of three exam scores',
            'performance': 'Performance level (High/Medium/Low)'
        }
        
        for col, desc in column_info.items():
            if col in df.columns:
                st.write(f"**{col}**: {desc}")
        
        st.markdown("<h3 style='color: #f1f5f9 !important;'>📊 Data Statistics</h3>", unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)

elif tab_selection == "🔄 Learning Center":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>🔄 Learning Center</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 25px; border-radius: 15px;'>
            <h3 style='color: #f1f5f9 !important;'>🤖 Continuous Self-Learning System</h3>
            <p style='color: #cbd5e1 !important;'>This system automatically learns from every prediction and feedback to improve accuracy over time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric("Auto-Learning", "ACTIVE" if st.session_state.auto_learning_enabled else "PAUSED")
        st.metric("Pending Updates", len(st.session_state.pending_feedback))
        st.metric("Total Predictions", len(st.session_state.prediction_history))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Learning Updates", st.session_state.learning_updates)
    with col_stat2:
        st.metric("Current Accuracy", f"{st.session_state.accuracy*100:.1f}%")
    with col_stat3:
        feedback_count = len([p for p in st.session_state.prediction_history if p.get('feedback_provided', False)])
        st.metric("Feedback Provided", feedback_count)
    with col_stat4:
        if st.session_state.last_update:
            last_update = st.session_state.last_update.strftime("%m/%d %H:%M")
            st.metric("Last Update", last_update)
    
    if st.session_state.pending_feedback:
        st.markdown("<h3 style='color: #f1f5f9 !important;'>📥 Pending Feedback</h3>", unsafe_allow_html=True)
        
        pending_df = pd.DataFrame(st.session_state.pending_feedback)
        st.dataframe(pending_df[['timestamp', 'predicted_performance', 'actual_performance']], use_container_width=True)
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Process All Feedback", use_container_width=True):
                with st.spinner("Processing feedback..."):
                    process_pending_feedback()
                    st.rerun()
        with col_btn2:
            if st.button("🗑️ Clear Pending Feedback", use_container_width=True):
                st.session_state.pending_feedback = []
                save_pending_feedback()
                st.success("Pending feedback cleared!")
                st.rerun()
    
    if st.session_state.prediction_history:
        st.markdown("<h3 style='color: #f1f5f9 !important;'>📋 Prediction History</h3>", unsafe_allow_html=True)
        
        pred_df = pd.DataFrame(st.session_state.prediction_history)
        
        recent_preds = pred_df.tail(10)
        st.dataframe(recent_preds[['timestamp', 'predicted_performance', 'prediction_confidence', 'actual_performance', 'algorithm']], use_container_width=True)
        
        st.markdown("<h4 style='color: #f1f5f9 !important;'>📊 Prediction Analytics</h4>", unsafe_allow_html=True)
        
        col_anal1, col_anal2, col_anal3 = st.columns(3)
        with col_anal1:
            pred_counts = pred_df['predicted_performance'].value_counts()
            fig = px.pie(values=pred_counts.values, names=pred_counts.index, 
                        color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444'])
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_anal2:
            fig = px.histogram(pred_df, x='prediction_confidence', nbins=20,
                             color_discrete_sequence=['#8b5cf6'])
            fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', 
                            xaxis=dict(color='#e2e8f0'), yaxis=dict(color='#e2e8f0'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_anal3:
            if 'algorithm' in pred_df.columns:
                algo_counts = pred_df['algorithm'].value_counts()
                fig = px.pie(values=algo_counts.values, names=algo_counts.index,
                           color_discrete_sequence=['#3b82f6', '#10b981', '#8b5cf6'])
                fig.update_layout(height=300, paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'))
                st.plotly_chart(fig, use_container_width=True)
    
    if os.path.exists(LEARNING_LOG_FILE):
        st.markdown("<h3 style='color: #f1f5f9 !important;'>📈 Learning Progress Over Time</h3>", unsafe_allow_html=True)
        try:
            learning_log = pd.read_csv(LEARNING_LOG_FILE)
            learning_log['timestamp'] = pd.to_datetime(learning_log['timestamp'])
            learning_log = learning_log.sort_values('timestamp')
            
            fig = px.line(learning_log, x='timestamp', y='model_accuracy',
                         title='Model Accuracy Improvement', markers=True)
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e2e8f0'), xaxis=dict(color='#e2e8f0'), 
                            yaxis=dict(color='#e2e8f0', tickformat='.0%'))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display learning progress: {e}")

elif tab_selection == "🤖 AI Assistant":
    import streamlit.components.v1 as components
    import html as html_lib
    
    # Initialize Chat State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Header
    st.markdown("""
    <div style='padding: 1.5rem; background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
                margin-bottom: 1.5rem; box-shadow: 0 2px 15px rgba(0,0,0,0.2);'>
        <h2 style='color: #f1f5f9; margin: 0; font-size: 1.8rem;'>🤖 AI Educational Assistant</h2>
        <p style='color: #94a3b8; margin: 8px 0 0; font-size: 0.95rem;'>
            Chat with AI to get answers about study tips, data analysis, and auto-learning features.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Build Chat HTML
    chat_messages_html = ""
    
    if not st.session_state.chat_history:
        chat_messages_html = """
        <div class='empty-chat'>
            <h3>💭 Start chatting with the AI Assistant</h3>
            <p style='color: #64748b; font-size: 0.9rem;'>Ask me about:</p>
            <p style='color: #64748b; font-size: 0.8rem;'>• Student performance predictions</p>
            <p style='color: #64748b; font-size: 0.8rem;'>• Auto-learning features</p>
            <p style='color: #64748b; font-size: 0.8rem;'>• Study strategies and tips</p>
            <p style='color: #64748b; font-size: 0.8rem;'>• Data analysis insights</p>
        </div>
        """
    else:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            avatar = "👤" if role == "user" else "🤖"
            avatar_class = "avatar-user" if role == "user" else "avatar-ai"
            content = html_lib.escape(msg["content"])
            
            chat_messages_html += f"""
            <div class='chat-message {role}'>
                <div class='chat-avatar {avatar_class}'>{avatar}</div>
                <div class='chat-bubble'>{content}</div>
            </div>
            """

    # Complete HTML with styles
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
        }}
        
        .chat-screen {{
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 1.5rem;
            height: 65vh;
            overflow-y: auto;
            box-shadow: 0 4px 25px rgba(0,0,0,0.3);
        }}

        .chat-message {{
            display: flex;
            align-items: flex-start;
            gap: 1rem;
            margin-bottom: 1.2rem;
            animation: fadeIn 0.3s ease-in;
        }}

        @keyframes fadeIn {{
            from {{opacity: 0; transform: translateY(10px);}}
            to {{opacity: 1; transform: translateY(0);}}
        }}

        .chat-message.user {{
            flex-direction: row-reverse;
        }}

        .chat-avatar {{
            width: 45px;
            height: 45px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.4rem;
            flex-shrink: 0;
        }}

        .avatar-user {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.4);
        }}

        .avatar-ai {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            box-shadow: 0 2px 10px rgba(16, 185, 129, 0.4);
        }}

        .chat-bubble {{
            padding: 1rem 1.4rem;
            border-radius: 1rem;
            max-width: 70%;
            color: #f1f5f9;
            line-height: 1.6;
            word-wrap: break-word;
            font-size: 0.95rem;
        }}

        .user .chat-bubble {{
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            border-radius: 1rem 1rem 0.3rem 1rem;
            box-shadow: 0 2px 10px rgba(99, 102, 241, 0.3);
        }}

        .assistant .chat-bubble {{
            background: rgba(30,41,59,0.85);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 1rem 1rem 1rem 0.3rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }}

        .empty-chat {{
            text-align: center;
            color: #64748b;
            padding-top: 15%;
        }}

        .chat-screen::-webkit-scrollbar {{
            width: 8px;
        }}
        
        .chat-screen::-webkit-scrollbar-thumb {{
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
        }}
    </style>
    </head>
    <body>
        <div class='chat-screen' id='chat-container'>
            {chat_messages_html}
        </div>
        <script>
            const chatContainer = document.getElementById('chat-container');
            if (chatContainer) {{
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }}
        </script>
    </body>
    </html>
    """

    # Chat and Input Container
    st.markdown("""
    <style>
        .chat-wrapper {
            position: relative;
            background: rgba(15, 23, 42, 0.5);
            border-radius: 20px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .stForm {
            margin-top: -6rem !important;
            padding-top: 0rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Render Chat
    components.html(full_html, height=500, scrolling=False)

    # Input Section
    with st.form("chat_form", clear_on_submit=True):
        col_input, col_send = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Message", 
                placeholder="Ask about auto-learning, predictions, or study tips...", 
                label_visibility="collapsed"
            )
        with col_send:
            send_btn = st.form_submit_button("📤 Send", use_container_width=True)

    # Handle Chat Logic
    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("🤖 AI is thinking..."):
            response = get_ai_response(user_input.strip())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Clear Chat
    if st.session_state.chat_history:
        col_clear, col_insights, col_export = st.columns([1, 1, 1])
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        with col_insights:
            if st.button("📊 Get Dataset Insights", use_container_width=True):
                st.session_state.chat_history.append({'role': 'user', 'content': "Provide key insights from the dataset and explain the auto-learning system"})
                with st.spinner("🤖 Analyzing dataset and auto-learning..."):
                    ai_response = get_ai_response("Provide key insights from the dataset and explain the auto-learning system")
                    st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
                st.rerun()
        with col_export:
            if st.button("💾 Export Chat", use_container_width=True):
                # Create downloadable chat history
                chat_text = "AI Assistant Chat History\n"
                chat_text += "=" * 30 + "\n\n"
                for msg in st.session_state.chat_history:
                    role = "You" if msg["role"] == "user" else "AI Assistant"
                    chat_text += f"{role}: {msg['content']}\n\n"
                
                st.download_button(
                    label="Download Chat",
                    data=chat_text,
                    file_name=f"ai_assistant_chat_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b !important;'>
    <p>🎓 Advanced Student Performance Predictor | Built with Streamlit & Scikit-learn | Powered by Kaggle Dataset & Gemini AI</p>
    <p style='font-size: 12px;'>🤖 Multi-Algorithm AI: Random Forest, XGBoost, and Neural Networks for Optimal Predictions</p>
</div>
""", unsafe_allow_html=True)
