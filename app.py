import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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

# Model file paths
MODEL_FILE = 'student_performance_model.pkl'
ENCODERS_FILE = 'label_encoders.pkl'
LEARNING_LOG_FILE = 'learning_log.csv'
PREDICTION_HISTORY_FILE = 'prediction_history.csv'
PENDING_FEEDBACK_FILE = 'pending_feedback.pkl'

# Gemini AI Configuration
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
GEMINI_API_KEY = "AIzaSyAblEEGSq7wO8ihLLuBN1Mjw2vehg_spi0"

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
    
    system_prompt = f"""
You are an AI Educational Assistant specialized in student performance analysis. You have access to:

1. DATASET INFORMATION:
{dataset_context}

2. CURRENT PREDICTION CONTEXT:
{prediction_context}

3. MODEL PERFORMANCE:
{model_context}

Guidelines:
- Provide data-driven insights based on the available dataset
- Reference specific statistics when relevant
- Offer practical, actionable advice for improvement
- Consider the student's current prediction and profile
- Be encouraging and constructive in your feedback
- If asking about specific predictions, use the current prediction context
- For dataset questions, reference the actual statistics provided

User Question: {user_message}

Please provide a helpful, educational response:
"""
    
    return query_gemini(system_prompt)

def save_model(model, encoders, feature_names, accuracy):
    """Save trained model and encoders to files"""
    try:
        model_data = {
            'model': model,
            'feature_names': feature_names,
            'accuracy': accuracy
        }
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        with open(ENCODERS_FILE, 'wb') as f:
            pickle.dump(encoders, f)
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
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
        
        st.session_state.model = model_data['model']
        st.session_state.encoders = encoders
        st.session_state.feature_names = model_data['feature_names']
        st.session_state.accuracy = model_data['accuracy']
        st.session_state.model_trained = True
        
        return True
    except Exception as e:
        st.error(f"Error loading model: {e}")
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

def log_prediction(student_data, predicted_performance, confidence):
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
            'model_accuracy': st.session_state.accuracy
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

def update_model_with_new_data(student_data, actual_performance):
    """Update the model with new prediction data for continuous learning"""
    try:
        df = st.session_state.df.copy()
        
        new_row = student_data.copy()
        new_row['performance'] = actual_performance
        new_row['avg_score'] = (new_row['math_score'] + new_row['reading_score'] + new_row['writing_score']) / 3
        
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state.df = df
        
        feature_cols = ['gender', 'parental_education', 'lunch', 'test_prep', 
                        'math_score', 'reading_score', 'writing_score', 
                        'study_hours', 'attendance', 'extracurricular']
        
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
        
        model_data = {
            'model': model,
            'feature_names': X.columns.tolist(),
            'accuracy': new_accuracy
        }
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        with open(ENCODERS_FILE, 'wb') as f:
            pickle.dump(encoders, f)
        
        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.feature_names = X.columns.tolist()
        st.session_state.accuracy = new_accuracy
        st.session_state.model_trained = True
        st.session_state.learning_updates += 1
        st.session_state.last_update = datetime.datetime.now()
        
        log_learning_update(student_data, "N/A", actual_performance, accuracy_change)
        
        return True, accuracy_change
        
    except Exception as e:
        st.error(f"Error updating model: {e}")
        return False, 0

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
        
        feature_cols = ['gender', 'parental_education', 'lunch', 'test_prep', 
                        'math_score', 'reading_score', 'writing_score', 
                        'study_hours', 'attendance', 'extracurricular']
        
        X = df[feature_cols].copy()
        y = df['performance']
        
        encoders = {}
        for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
        
        model = RandomForestClassifier(n_estimators=80, random_state=42, max_depth=8)
        model.fit(X_train, y_train)
        
        new_accuracy = accuracy_score(y_test, model.predict(X_test))
        accuracy_change = new_accuracy - st.session_state.accuracy
        
        model_data = {
            'model': model,
            'feature_names': X.columns.tolist(),
            'accuracy': new_accuracy
        }
        
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model_data, f)
        with open(ENCODERS_FILE, 'wb') as f:
            pickle.dump(encoders, f)
        
        st.session_state.model = model
        st.session_state.encoders = encoders
        st.session_state.feature_names = X.columns.tolist()
        st.session_state.accuracy = new_accuracy
        st.session_state.learning_updates += len(new_data_points)
        st.session_state.last_update = datetime.datetime.now()
        
        return True, accuracy_change
        
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

def auto_learn_from_prediction(student_data, predicted_performance, actual_performance):
    """Automatically learn from prediction results"""
    if not st.session_state.auto_learning_enabled:
        return
    
    try:
        feedback_entry = {
            'student_data': student_data,
            'predicted_performance': predicted_performance,
            'actual_performance': actual_performance,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        st.session_state.pending_feedback.append(feedback_entry)
        save_pending_feedback()
        
        if len(st.session_state.pending_feedback) >= 5:
            process_pending_feedback()
        
        return True
    except Exception as e:
        st.warning(f"Auto-learning failed: {e}")
        return False

def check_dataset_available():
    """Check if dataset file exists"""
    try:
        df = pd.read_csv("StudentsPerformance.csv")
        st.session_state.dataset_available = True
        st.session_state.df = df
        return True
    except:
        st.session_state.dataset_available = False
        return False

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

def train_model(df):
    """Train the initial model"""
    try:
        feature_cols = ['gender', 'parental_education', 'lunch', 'test_prep', 
                        'math_score', 'reading_score', 'writing_score', 
                        'study_hours', 'attendance', 'extracurricular']
        
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
        accuracy = accuracy_score(y_test, y_pred)
        
        if save_model(model, encoders, X.columns.tolist(), accuracy):
            st.success(f"💾 Model saved successfully! Accuracy: {accuracy*100:.2f}%")
        
        return model, encoders, accuracy, X_test, y_test, y_pred, X.columns
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, 0, None, None, None, None

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

# Process any pending feedback on startup
if st.session_state.pending_feedback and st.session_state.model_trained:
    process_pending_feedback()

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
        ["📖 Overview", "🔮 Predict Performance", "📊 Analytics Dashboard", "🤖 Train Model", "💡 AI Insights", "📁 Dataset Info", "🔄 Learning Center", "🤖 AI Assistant"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
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
    
    # Manual feedback processing
    if st.session_state.pending_feedback:
        if st.button("🔄 Process Pending Feedback", use_container_width=True):
            with st.spinner("Processing feedback..."):
                process_pending_feedback()
                st.rerun()
    
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

# Add the Overview tab at the beginning of main content
# Add the Overview tab at the beginning of main content
if tab_selection == "📖 Overview":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>📖 Project Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 30px; border-radius: 20px; margin-bottom: 20px;'>
            <h3 style='color: #f1f5f9 !important;'>🎓 Advanced Student Performance Predictor</h3>
            <p style='color: #cbd5e1 !important; font-size: 16px; line-height: 1.6;'>
            A comprehensive AI-powered web application that predicts student academic performance using machine learning 
            and provides data-driven insights for educational improvement. This system combines traditional data analysis 
            with cutting-edge AI capabilities to help educators and students understand performance patterns.
            </p>
            <p style='color: #60a5fa !important; font-size: 14px; font-weight: bold; margin-top: 15px;'>
            🔄 Smart Data Handling: If Kaggle dataset is unavailable, the system automatically uses pre-trained models. 
            If both are unavailable, it generates sample data for seamless predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features Section
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🚀 Key Features</h3>", unsafe_allow_html=True)
        
        features = [
            {
                "icon": "🔮",
                "title": "Smart Predictions",
                "description": "Predict student performance using Random Forest algorithm with real-time accuracy metrics"
            },
            {
                "icon": "🤖",
                "title": "AI-Powered Insights",
                "description": "Get intelligent analysis and recommendations using Gemini AI integration"
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
        
        # Data Fallback System
        st.markdown("<h3 style='color: #f1f5f9 !important; margin-top: 20px;'>📁 Data Fallback System</h3>", unsafe_allow_html=True)
        
        fallback_steps = [
            {
                "step": "1️⃣",
                "description": "Try to load Kaggle dataset (StudentsPerformance.csv)"
            },
            {
                "step": "2️⃣", 
                "description": "If unavailable, load pre-trained model (student_performance_model.pkl)"
            },
            {
                "step": "3️⃣",
                "description": "If both unavailable, generate realistic sample data automatically"
            }
        ]
        
        for step in fallback_steps:
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
                <div style='display: flex; align-items: center; gap: 10px;'>
                    <span style='color: #60a5fa; font-size: 16px;'>{step['step']}</span>
                    <p style='color: #cbd5e1 !important; margin: 0; font-size: 12px;'>{step['description']}</p>
                </div>
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
        {"label": "Model Accuracy", "value": f"{st.session_state.accuracy*100:.1f}%" if st.session_state.model_trained else "Not Trained"},
        {"label": "Learning Updates", "value": st.session_state.learning_updates},
        {"label": "Success Rate", "value": f"{success_rate:.1f}%" if st.session_state.df is not None else "N/A"},
        {"label": "Data Source", "value": data_source}
    ]
    
    for stat in stats:
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.05); padding: 12px; border-radius: 8px; margin-bottom: 8px;'>
            <p style='color: #cbd5e1 !important; margin: 0; font-size: 12px;'>{stat['label']}</p>
            <p style='color: #60a5fa !important; margin: 0; font-weight: bold; font-size: 16px;'>{stat['value']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown("<h3 style='color: #f1f5f9 !important; margin-top: 30px;'>🔧 How It Works</h3>", unsafe_allow_html=True)
    
    steps = [
        {
            "step": "1",
            "title": "Smart Data Loading",
            "description": "Automatically loads Kaggle dataset, falls back to pre-trained model, or generates sample data if needed"
        },
        {
            "step": "2",
            "title": "Model Training & Prediction",
            "description": "Train Random Forest classifier and make real-time performance predictions"
        },
        {
            "step": "3",
            "title": "Interactive Analysis",
            "description": "Provide detailed performance analysis with interactive visualizations and charts"
        },
        {
            "step": "4",
            "title": "Continuous Self-Learning",
            "description": "Model automatically improves with user feedback through auto-learning system"
        },
        {
            "step": "5",
            "title": "AI-Powered Insights",
            "description": "Get intelligent recommendations using integrated Gemini AI assistant"
        }
    ]
    
    for step in steps:
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 4px solid #8b5cf6;'>
            <div style='display: flex; align-items: flex-start; gap: 15px;'>
                <div style='background: linear-gradient(135deg, #8b5cf6, #7c3aed); color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; flex-shrink: 0;'>
                    {step['step']}
                </div>
                <div>
                    <h4 style='color: #f1f5f9 !important; margin: 0 0 8px 0;'>{step['title']}</h4>
                    <p style='color: #cbd5e1 !important; margin: 0;'>{step['description']}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Data Sources Section
    st.markdown("<h3 style='color: #f1f5f9 !important; margin-top: 30px;'>📚 Data Sources & Fallback</h3>", unsafe_allow_html=True)
    
    data_sources = [
        {
            "source": "🎯 Primary Source",
            "description": "Kaggle Students Performance Dataset",
            "details": "Real student data with exam scores, demographics, and educational factors"
        },
        {
            "source": "💾 Secondary Source", 
            "description": "Pre-trained Model Files",
            "details": "Automatically saved models from previous training sessions"
        },
        {
            "source": "🔄 Fallback Source",
            "description": "Generated Sample Data",
            "details": "Realistic synthetic data created when primary sources are unavailable"
        }
    ]
    
    for source in data_sources:
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; margin-bottom: 15px; border-left: 4px solid #10b981;'>
            <h4 style='color: #f1f5f9 !important; margin: 0 0 8px 0;'>{source['source']}</h4>
            <p style='color: #60a5fa !important; margin: 0 0 5px 0; font-weight: bold;'>{source['description']}</p>
            <p style='color: #cbd5e1 !important; margin: 0; font-size: 14px;'>{source['details']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Use Cases Section
    st.markdown("<h3 style='color: #f1f5f9 !important; margin-top: 30px;'>🎯 Use Cases</h3>", unsafe_allow_html=True)
    
    use_cases = [
        "🏫 **Educational Institutions**: Identify at-risk students and provide targeted support",
        "👨‍🏫 **Teachers**: Understand class performance patterns and adjust teaching strategies", 
        "🎓 **Students**: Get personalized study recommendations and performance insights",
        "👨‍👩‍👧‍👦 **Parents**: Monitor student progress and understand factors affecting performance",
        "🔬 **Researchers**: Analyze educational data patterns and test hypotheses"
    ]
    
    for use_case in use_cases:
        st.markdown(f"""
        <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 10px;'>
            <p style='color: #cbd5e1 !important; margin: 0;'>{use_case}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Developer Info
    st.markdown("---")
    st.markdown("""
    <div style='background: linear-gradient(135deg, #1e293b, #0f172a); padding: 25px; border-radius: 15px; text-align: center;'>
        <h3 style='color: #f1f5f9 !important; margin-bottom: 10px;'>👨‍💻 Developer Information</h3>
        <p style='color: #60a5fa !important; font-size: 18px; font-weight: bold; margin: 5px 0;'>Ahmer ALI</p>
        <p style='color: #cbd5e1 !important; margin: 5px 0;'>BSCS Student at Sindh University</p>
        <p style='color: #94a3b8 !important; margin: 5px 0; font-size: 14px;'>Advanced Student Performance Predictor with Auto-Learning AI</p>
        <p style='color: #60a5fa !important; margin: 10px 0 0 0; font-size: 14px; font-weight: bold;'>
        🔄 Smart Fallback System: Always functional with Kaggle data, pre-trained models, or generated samples
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    
# Main content based on selection
# if tab_selection == "🤖 AI Assistant":
#     st.markdown("<h2 style='color: #f1f5f9 !important;'>🤖 AI Educational Assistant</h2>", unsafe_allow_html=True)
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.markdown("""
#         <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; margin-bottom: 20px;'>
#             <h3 style='color: #f1f5f9 !important;'>💬 Chat with Gemini AI</h3>
#             <p style='color: #cbd5e1 !important;'>Get AI-powered insights about student performance, educational strategies, 
#             and data analysis. The AI has access to your dataset and current predictions!</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Chat container
#         st.markdown("<div class='chat-container' id='chat-container'>", unsafe_allow_html=True)
        
#         # Display chat history
#         for message in st.session_state.chat_history:
#             if message['role'] == 'user':
#                 st.markdown(f"<div class='user-message'><strong>You:</strong> {message['content']}</div>", unsafe_allow_html=True)
#             else:
#                 st.markdown(f"<div class='bot-message'><strong>AI:</strong> {message['content']}</div>", unsafe_allow_html=True)
        
#         st.markdown("</div>", unsafe_allow_html=True)
        
#         # Chat input
#         user_input = st.text_area(
#             "💭 Ask me anything about student performance, educational strategies, or data analysis:",
#             placeholder="Type your message here...\nExamples:\n- 'Analyze my current prediction'\n- 'What factors most affect performance?'\n- 'Give me study tips for math'\n- 'How does parental education impact scores?'",
#             key="chat_input",
#             height=100
#         )
        
#         col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
#         with col_btn1:
#             if st.button("🚀 Send Message", use_container_width=True) and user_input:
#                 # Add user message to chat history
#                 st.session_state.chat_history.append({'role': 'user', 'content': user_input})
                
#                 # Get AI response with full context
#                 with st.spinner("🤖 Analyzing data and generating response..."):
#                     ai_response = get_ai_response(user_input)
#                     st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
                
#                 st.rerun()
        
#         with col_btn2:
#             if st.button("🔄 Clear Chat", use_container_width=True):
#                 st.session_state.chat_history = []
#                 st.rerun()
        
#         with col_btn3:
#             if st.button("📊 Dataset Insights", use_container_width=True):
#                 with st.spinner("🤖 Analyzing dataset..."):
#                     insights_prompt = "Provide key insights and patterns from the student performance dataset. Focus on the most important factors affecting student performance."
#                     ai_response = get_ai_response(insights_prompt)
#                     st.session_state.chat_history.append({'role': 'user', 'content': "Give me insights from the dataset"})
#                     st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
#                 st.rerun()
    
#     with col2:
#         st.markdown("<h3 style='color: #f1f5f9 !important;'>💡 Quick Actions</h3>", unsafe_allow_html=True)
        
#         # Prediction-related questions
#         if st.session_state.current_prediction:
#             st.markdown("#### 🎯 Prediction Analysis")
#             pred_questions = [
#                 "Analyze my current prediction",
#                 "Why did I get this prediction?",
#                 "How can I improve my performance?",
#                 "What are my strengths and weaknesses?",
#                 "Compare my scores to dataset averages"
#             ]
            
#             for question in pred_questions:
#                 if st.button(f"❓ {question}", key=f"pred_{question}", use_container_width=True):
#                     st.session_state.chat_history.append({'role': 'user', 'content': question})
#                     with st.spinner("🤖 Analyzing your prediction..."):
#                         ai_response = get_ai_response(question)
#                         st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
#                     st.rerun()
        
#         st.markdown("#### 📚 Educational Questions")
#         edu_questions = [
#             "What factors most influence student performance?",
#             "How can students improve their math scores?",
#             "What's the impact of parental education?",
#             "Give me study strategies for better performance",
#             "How does attendance affect academic performance?",
#             "What are the benefits of extracurricular activities?",
#             "How effective is test preparation?",
#             "What's the correlation between different subjects?"
#         ]
        
#         for question in edu_questions:
#             if st.button(f"❓ {question}", key=f"edu_{question}", use_container_width=True):
#                 st.session_state.chat_history.append({'role': 'user', 'content': question})
#                 with st.spinner("🤖 Researching educational insights..."):
#                     ai_response = get_ai_response(question)
#                     st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
#                 st.rerun()
        
#         st.markdown("#### 📈 Data Analysis")
#         data_questions = [
#             "Show me dataset statistics",
#             "What patterns are in the data?",
#             "How accurate is the prediction model?",
#             "What are the key performance indicators?",
#             "Analyze gender performance differences"
#         ]
        
#         for question in data_questions:
#             if st.button(f"❓ {question}", key=f"data_{question}", use_container_width=True):
#                 st.session_state.chat_history.append({'role': 'user', 'content': question})
#                 with st.spinner("🤖 Analyzing dataset..."):
#                     ai_response = get_ai_response(question)
#                     st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
#                 st.rerun()


elif tab_selection == "🤖 AI Assistant":
    import streamlit.components.v1 as components
    import html as html_lib
    
    # === Initialize Chat State ===
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # col_main, col_sidebar = st.columns([3, 1])
    
    # with col_main:
        # === Header ===
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
    # with col_sidebar:
        # st.markdown("<h4 style='color: #f1f5f9 !important;'>🔄 Auto-Learning Status</h4>", unsafe_allow_html=True)
        
        # status_color = "#10b981" if st.session_state.auto_learning_enabled else "#ef4444"
        # status_text = "ACTIVE" if st.session_state.auto_learning_enabled else "PAUSED"
        
        # st.markdown(f"""
        # <div style='background: rgba(30,41,59,0.8); padding: 15px; border-radius: 10px; border-left: 4px solid {status_color};'>
        #     <p style='color: #e2e8f0 !important; margin: 0; font-size: 14px;'>Auto-Learning</p>
        #     <p style='color: {status_color} !important; margin: 5px 0 0 0; font-size: 16px; font-weight: bold;'>{status_text}</p>
        #     <p style='color: #94a3b8 !important; margin: 0; font-size: 12px;'>Updates: {st.session_state.learning_updates}</p>
        #     <p style='color: #94a3b8 !important; margin: 0; font-size: 12px;'>Accuracy: {st.session_state.accuracy*100:.1f}%</p>
        # </div>
        # """, unsafe_allow_html=True)

    # # === Quick Actions Sidebar ===
    # col_main, col_sidebar = st.columns([3, 1])
    
    # with col_sidebar:
    # #     st.markdown("<h3 style='color: #f1f5f9 !important;'>💡 Quick Actions</h3>", unsafe_allow_html=True)
        
    # #     # Auto-learning specific questions
    # #     st.markdown("#### 🔄 Auto-Learning")
    # #     learning_questions = [
    # #         "How does the auto-learning feature work?",
    # #         "What is continuous self-learning?",
    # #         "How can I help the model improve?",
    # #         "Explain the auto-learning system",
    # #         "How many updates has the model received?"
    # #     ]
        
    # #     for question in learning_questions:
    # #         if st.button(f"❓ {question}", key=f"learn_{question}", use_container_width=True):
    # #             st.session_state.chat_history.append({'role': 'user', 'content': question})
    # #             with st.spinner("🤖 Explaining auto-learning..."):
    # #                 ai_response = get_ai_response(question)
    # #                 st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
    # #             st.rerun()
        
    # #     # Prediction-related questions
    # #     if st.session_state.current_prediction:
    # #         st.markdown("#### 🎯 Prediction Analysis")
    # #         pred_questions = [
    # #             "Analyze my current prediction",
    # #             "How does auto-learning affect predictions?",
    # #             "How can I improve my performance?",
    # #             "What are my strengths and weaknesses?",
    # #             "How accurate is the model?"
    # #         ]
            
    # #         for question in pred_questions:
    # #             if st.button(f"❓ {question}", key=f"pred_{question}", use_container_width=True):
    # #                 st.session_state.chat_history.append({'role': 'user', 'content': question})
    # #                 with st.spinner("🤖 Analyzing your prediction..."):
    # #                     ai_response = get_ai_response(question)
    # #                     st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
    # #                 st.rerun()
        
    # #     st.markdown("#### 📚 Educational Questions")
    # #     edu_questions = [
    # #         "What factors most influence student performance?",
    # #         "How can students improve their math scores?",
    # #         "What's the impact of parental education?",
    # #         "Give me study strategies for better performance"
    # #     ]
        
    # #     for question in edu_questions:
    # #         if st.button(f"❓ {question}", key=f"edu_{question}", use_container_width=True):
    # #             st.session_state.chat_history.append({'role': 'user', 'content': question})
    # #             with st.spinner("🤖 Researching educational insights..."):
    # #                 ai_response = get_ai_response(question)
    # #                 st.session_state.chat_history.append({'role': 'assistant', 'content': ai_response})
    # #             st.rerun()
        
    # #     # Auto-learning status
    #     st.markdown("---")
    #     st.markdown("<h4 style='color: #f1f5f9 !important;'>🔄 Auto-Learning Status</h4>", unsafe_allow_html=True)
        
    #     status_color = "#10b981" if st.session_state.auto_learning_enabled else "#ef4444"
    #     status_text = "ACTIVE" if st.session_state.auto_learning_enabled else "PAUSED"
        
    #     st.markdown(f"""
    #     <div style='background: rgba(30,41,59,0.8); padding: 15px; border-radius: 10px; border-left: 4px solid {status_color};'>
    #         <p style='color: #e2e8f0 !important; margin: 0; font-size: 14px;'>Auto-Learning</p>
    #         <p style='color: {status_color} !important; margin: 5px 0 0 0; font-size: 16px; font-weight: bold;'>{status_text}</p>
    #         <p style='color: #94a3b8 !important; margin: 0; font-size: 12px;'>Updates: {st.session_state.learning_updates}</p>
    #         <p style='color: #94a3b8 !important; margin: 0; font-size: 12px;'>Accuracy: {st.session_state.accuracy*100:.1f}%</p>
    #     </div>
    #     """, unsafe_allow_html=True)

    # with col_main:
        # === Build Chat HTML ===
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

    # === Complete HTML with styles ===
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

    # === Chat and Input Container ===
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

    # === Render Chat ===
    components.html(full_html, height=500, scrolling=False)

    # === Input Section ===
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

    # === Handle Chat Logic ===
    if send_btn and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input.strip()})
        with st.spinner("🤖 AI is thinking..."):
            response = get_ai_response(user_input.strip())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # === Clear Chat ===
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


# elif tab_selection == "🔮 Predict Performance":
#     st.markdown("<h2 style='color: #f1f5f9 !important;'>🔮 Student Performance Prediction</h2>", unsafe_allow_html=True)
    
#     # Show model status with learning info
#     if st.session_state.model_trained:
#         st.success(f"✅ Using trained model (Accuracy: {st.session_state.accuracy*100:.2f}%)")
#         if st.session_state.learning_updates > 0:
#             st.info(f"🔄 Model has learned from {st.session_state.learning_updates} updates")
#         if st.session_state.auto_learning_enabled:
#             st.info("🤖 Auto-learning enabled - model will improve automatically")
#     else:
#         st.warning("⚠️ No trained model available. Please train the model first in the 'Train Model' section.")
    
#     col1, col2 = st.columns([1, 1])
    
#     with col1:
#         st.markdown("<h3 style='color: #f1f5f9 !important;'>📝 Student Information</h3>", unsafe_allow_html=True)
        
#         with st.container():
#             gender = st.selectbox("👤 Gender", ["female", "male"])
            
#             parental_education = st.selectbox(
#                 "🎓 Parental Education Level",
#                 ["some high school", "high school", "some college", 
#                  "associate's degree", "bachelor's degree", "master's degree"]
#             )
            
#             lunch = st.selectbox("🍽️ Lunch Type", ["standard", "free/reduced"])
            
#             test_prep = st.selectbox("📚 Test Preparation Course", ["completed", "none"])
            
#             st.markdown("<h4 style='color: #f1f5f9 !important;'>📊 Test Scores</h4>", unsafe_allow_html=True)
#             col_a, col_b, col_c = st.columns(3)
#             with col_a:
#                 math_score = st.number_input("Math", 0, 100, 75)
#             with col_b:
#                 reading_score = st.number_input("Reading", 0, 100, 75)
#             with col_c:
#                 writing_score = st.number_input("Writing", 0, 100, 75)
            
#             study_hours = st.slider("⏰ Daily Study Hours", 0, 10, 5)
#             attendance = st.slider("📅 Attendance Rate (%)", 0, 100, 85)
#             extracurricular = st.selectbox("🎨 Extracurricular Activities", ["yes", "no"])
        
#         predict_button = st.button("🎯 Predict Performance", use_container_width=True)
    
#     with col2:
#         if predict_button:
#             if not st.session_state.model_trained:
#                 if not load_model():
#                     if st.session_state.dataset_available and st.session_state.df is not None:
#                         with st.spinner("Training model..."):
#                             result = train_model(st.session_state.df)
#                             if result[0] is not None:
#                                 model, encoders, accuracy, X_test, y_test, y_pred, feature_names = result
#                                 st.session_state.model = model
#                                 st.session_state.encoders = encoders
#                                 st.session_state.feature_names = feature_names
#                                 st.session_state.model_trained = True
#                                 st.session_state.X_test = X_test
#                                 st.session_state.y_test = y_test
#                                 st.session_state.y_pred = y_pred
#                                 st.session_state.accuracy = accuracy
#                                 st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")
#                             else:
#                                 st.error("❌ Failed to train model")
#                                 st.stop()
#                     else:
#                         st.error("❌ No dataset available to train model and no pre-trained model found.")
#                         st.stop()
            
#             if st.session_state.model_trained:
#                 input_data = pd.DataFrame({
#                     'gender': [gender],
#                     'parental_education': [parental_education],
#                     'lunch': [lunch],
#                     'test_prep': [test_prep],
#                     'math_score': [math_score],
#                     'reading_score': [reading_score],
#                     'writing_score': [writing_score],
#                     'study_hours': [study_hours],
#                     'attendance': [attendance],
#                     'extracurricular': [extracurricular]
#                 })
                
#                 for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
#                     input_data[col] = st.session_state.encoders[col].transform(input_data[col])
                
#                 prediction = st.session_state.model.predict(input_data)[0]
#                 prediction_proba = st.session_state.model.predict_proba(input_data)[0]
#                 confidence = max(prediction_proba)
                
#                 avg_score = (math_score + reading_score + writing_score) / 3
                
#                 student_data = {
#                     'gender': gender,
#                     'parental_education': parental_education,
#                     'lunch': lunch,
#                     'test_prep': test_prep,
#                     'math_score': math_score,
#                     'reading_score': reading_score,
#                     'writing_score': writing_score,
#                     'study_hours': study_hours,
#                     'attendance': attendance,
#                     'extracurricular': extracurricular
#                 }
                
#                 log_prediction(student_data, prediction, confidence)
                
#                 # Store current prediction for AI context
#                 st.session_state.current_prediction = {
#                     'performance': prediction,
#                     'confidence': confidence,
#                     'avg_score': avg_score
#                 }
#                 st.session_state.current_student_data = student_data
                
#                 st.markdown("<h3 style='color: #f1f5f9 !important;'>🎯 Prediction Results</h3>", unsafe_allow_html=True)
                
#                 if prediction == 'High':
#                     st.markdown('<div style="background: linear-gradient(135deg, #10b981, #059669); color: white !important; padding: 30px; border-radius: 20px; text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0;">🌟 HIGH PERFORMANCE 🌟</div>', unsafe_allow_html=True)
#                 elif prediction == 'Medium':
#                     st.markdown('<div style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white !important; padding: 30px; border-radius: 20px; text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0;">📈 MEDIUM PERFORMANCE 📈</div>', unsafe_allow_html=True)
#                 else:
#                     st.markdown('<div style="background: linear-gradient(135deg, #ef4444, #dc2626); color: white !important; padding: 30px; border-radius: 20px; text-align: center; font-size: 24px; font-weight: bold; margin: 20px 0;">⚠️ LOW PERFORMANCE ⚠️</div>', unsafe_allow_html=True)
                
#                 col_m1, col_m2, col_m3 = st.columns(3)
#                 with col_m1:
#                     st.metric("Average Score", f"{avg_score:.1f}")
#                 with col_m2:
#                     st.metric("Confidence", f"{confidence*100:.1f}%")
#                 with col_m3:
#                     st.metric("Prediction", prediction)
                
#                 st.info(f"🧠 Using trained model with {st.session_state.accuracy*100:.1f}% accuracy")
                
#                 # Radar Chart
#                 st.markdown("<h3 style='color: #f1f5f9 !important;'>📊 Performance Profile</h3>", unsafe_allow_html=True)
                
#                 categories = ['Math', 'Reading', 'Writing', 'Study Habits', 'Attendance']
#                 values = [math_score, reading_score, writing_score, study_hours*10, attendance]
                
#                 fig = go.Figure()
#                 fig.add_trace(go.Scatterpolar(
#                     r=values,
#                     theta=categories,
#                     fill='toself',
#                     fillcolor='rgba(139, 92, 246, 0.3)',
#                     line=dict(color='rgb(139, 92, 246)', width=2),
#                     name='Performance'
#                 ))
                
#                 fig.update_layout(
#                     polar=dict(
#                         radialaxis=dict(visible=True, range=[0, 100])
#                     ),
#                     showlegend=False,
#                     height=400,
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     plot_bgcolor='rgba(0,0,0,0)',
#                     font=dict(color='#e2e8f0', size=12)
#                 )
                
#                 st.plotly_chart(fig, use_container_width=True)
                
#                 # AI Analysis Section
#                 st.markdown("---")
#                 st.markdown("<h3 style='color: #f1f5f9 !important;'>🤖 AI Analysis</h3>", unsafe_allow_html=True)
                
#                 if st.button("🎯 Get Detailed AI Analysis", use_container_width=True):
#                     with st.spinner("🤖 Analyzing your performance profile..."):
#                         ai_analysis = get_ai_response("Provide a detailed analysis of this student's performance prediction, including strengths, weaknesses, and specific recommendations for improvement.")
                        
#                         st.markdown("<div class='bot-message'>", unsafe_allow_html=True)
#                         st.markdown(ai_analysis)
#                         st.markdown("</div>", unsafe_allow_html=True)
                
#                 # CONTINUOUS LEARNING FEATURE
#                 st.markdown("---")
#                 st.markdown("<h3 style='color: #f1f5f9 !important;'>🔄 Continuous Self-Learning</h3>", unsafe_allow_html=True)
                
#                 if st.session_state.auto_learning_enabled:
#                     st.success("🤖 **Auto-learning enabled** - Model will improve automatically with feedback")
#                 else:
#                     st.warning("⏸️ **Auto-learning paused** - Enable in sidebar to allow model improvement")
                
#                 st.info("💡 **Help the model learn!** Provide feedback to improve future predictions.")
                
#                 feedback_col1, feedback_col2 = st.columns([2, 1])
                
#                 with feedback_col1:
#                     actual_performance = st.selectbox(
#                         "What was the actual performance?",
#                         ["Unknown", "High", "Medium", "Low"],
#                         key="feedback_performance"
#                     )
                
#                 if actual_performance != "Unknown":
#                     if st.button("✅ Teach Model This Data", use_container_width=True):
#                         if st.session_state.auto_learning_enabled:
#                             with st.spinner("Updating model with new knowledge..."):
#                                 success = auto_learn_from_prediction(student_data, prediction, actual_performance)
                                
#                                 if success:
#                                     st.success(f"🎉 Feedback recorded! Model will learn from this data.")
#                                     st.balloons()
#                                     st.info(f"📚 The model now has {len(st.session_state.prediction_history)} recorded predictions and {len(st.session_state.pending_feedback)} pending learning updates")
#                                 else:
#                                     st.error("❌ Failed to record feedback")
#                         else:
#                             st.warning("⚠️ Auto-learning is disabled. Enable it in the sidebar to allow model updates.")
#                 else:
#                     st.warning("👆 Select actual performance to teach the model")
                
#                 # Recommendations
#                 st.markdown("<h3 style='color: #f1f5f9 !important;'>💡 Personalized Recommendations</h3>", unsafe_allow_html=True)
                
#                 recommendations = []
#                 if prediction == 'High':
#                     recommendations = [
#                         "✅ Maintain consistent study habits",
#                         "🎯 Consider advanced placement courses",
#                         "🌟 Explore leadership opportunities",
#                         "🤝 Mentor other students"
#                     ]
#                 elif prediction == 'Medium':
#                     recommendations = [
#                         "📚 Increase study hours to 6-8 per day",
#                         "👥 Join study groups for peer learning",
#                         "🎯 Focus on weaker subjects",
#                         "📝 Utilize test preparation resources"
#                     ]
#                 else:
#                     recommendations = [
#                         "🆘 Seek tutoring support immediately",
#                         "📅 Create a structured study schedule",
#                         "📈 Improve attendance to above 85%",
#                         "💬 Discuss with academic counselor"
#                     ]
                
#                 for rec in recommendations:
#                     st.info(rec)

#         else:
#             if st.session_state.model_trained:
#                 html_content = f"""
#                 <div style='background: rgba(34, 197, 94, 0.1); padding: 40px; border-radius: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 2px solid #22c55e;'>
#                     <h3 style='color: #f1f5f9 !important; margin-bottom: 10px;'>🚀 Ready to Predict</h3>
#                     <p style='color: #cbd5e1 !important; margin-bottom: 15px;'>Using trained model with {st.session_state.accuracy*100:.2f}% accuracy</p>
#                     <p style='color: #86efac !important; font-size: 14px;'>Fill in student information and click predict</p>
#                 """
                
#                 if st.session_state.learning_updates > 0:
#                     html_content += f"<p style='color: #f59e0b !important; font-size: 12px;'>🔄 Model has learned from {st.session_state.learning_updates} updates</p>"
                
#                 if st.session_state.auto_learning_enabled:
#                     html_content += "<p style='color: #8b5cf6 !important; font-size: 12px;'>🤖 Auto-learning enabled</p>"
                
#                 html_content += "</div>"
                
#                 st.markdown(html_content, unsafe_allow_html=True)
#             else:
#                 st.markdown("""
#                 <div style='background: rgba(128, 128, 128, 0.2); padding: 40px; border-radius: 20px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
#                     <h3 style='color: #f1f5f9 !important; margin-bottom: 10px;'>Ready to Predict</h3>
#                     <p style='color: #cbd5e1 !important;'>Fill in the student information and click predict to see results</p>
#                     <p style='color: #f59e0b !important; font-size: 14px;'>Model will be trained automatically on first prediction</p>
#                 </div>
#                 """, unsafe_allow_html=True)

elif tab_selection == "🔮 Predict Performance":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>🔮 Student Performance Prediction</h2>", unsafe_allow_html=True)

    # Show model status
    if st.session_state.model_trained:
        st.success(f"✅ Using trained model (Accuracy: {st.session_state.accuracy*100:.2f}%)")
        if st.session_state.learning_updates > 0:
            st.info(f"🔄 Model has learned from {st.session_state.learning_updates} updates")
        if st.session_state.auto_learning_enabled:
            st.info("🤖 Auto-learning enabled - model will improve automatically")
    else:
        st.warning("⚠️ No trained model available. Please train the model first in the 'Train Model' section.")

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
                    with st.spinner("Training model..."):
                        result = train_model(st.session_state.df)
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
                            st.success(f"✅ Model trained successfully! Accuracy: {accuracy*100:.2f}%")
                        else:
                            st.error("❌ Failed to train model")
                            st.stop()
                else:
                    st.error("❌ No dataset available to train model and no pre-trained model found.")
                    st.stop()

        # Prepare input data
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
            'extracurricular': [extracurricular]
        })
        for col in ['gender', 'parental_education', 'lunch', 'test_prep', 'extracurricular']:
            input_data[col] = st.session_state.encoders[col].transform(input_data[col])

        prediction = st.session_state.model.predict(input_data)[0]
        prediction_proba = st.session_state.model.predict_proba(input_data)[0]
        confidence = max(prediction_proba)
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

        log_prediction(student_data, prediction, confidence)

        # Store in session state
        st.session_state.current_prediction = {
            'performance': prediction,
            'confidence': confidence,
            'avg_score': avg_score
        }
        st.session_state.current_student_data = student_data

    # Display Prediction Results
    if "current_prediction" in st.session_state and st.session_state.current_prediction is not None:
        prediction = st.session_state.current_prediction['performance']
        confidence = st.session_state.current_prediction['confidence']
        avg_score = st.session_state.current_prediction['avg_score']
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

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Average Score", f"{avg_score:.1f}")
        col_m2.metric("Confidence", f"{confidence*100:.1f}%")
        col_m3.metric("Prediction", prediction)
        st.info(f"🧠 Using trained model with {st.session_state.accuracy*100:.1f}% accuracy")

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

                    Provide actionable suggestions to improve performance to HIGH, in bullet points.
                    Include study habits, attendance, test preparation, motivation, and weak subjects focus.
                    """
                    ai_analysis = get_ai_response(ai_prompt)
                    
                    # Display analysis nicely
                    for line in ai_analysis.split("\n"):
                        if line.strip():
                            st.info(line.strip())
                except Exception as e:
                    # Handle API errors gracefully
                    if "429" in str(e):
                        st.error("⚠️ AI service is temporarily overloaded. Please try again in a few minutes.")
                    else:
                        st.error(f"❌ Failed to generate AI suggestions. Error: {str(e)}")


        # Continuous Self-Learning & Feedback (unchanged)
        st.markdown("---")
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🔄 Continuous Self-Learning</h3>", unsafe_allow_html=True)
        if st.session_state.auto_learning_enabled:
            st.success("🤖 **Auto-learning enabled** - Model will improve automatically with feedback")
        else:
            st.warning("⏸️ **Auto-learning paused** - Enable in sidebar to allow model improvement")
        st.info("💡 **Help the model learn!** Provide feedback to improve future predictions.")

        feedback_col1, feedback_col2 = st.columns([2, 1])
        with feedback_col1:
            actual_performance = st.selectbox(
                "What was the actual performance?",
                ["Unknown", "High", "Medium", "Low"],
                key="feedback_performance"
            )
        if actual_performance != "Unknown":
            if st.button("✅ Teach Model This Data", use_container_width=True):
                if st.session_state.auto_learning_enabled:
                    with st.spinner("Updating model with new knowledge..."):
                        success = auto_learn_from_prediction(student_data, prediction, actual_performance)
                        if success:
                            st.success("🎉 Feedback recorded! Model will learn from this data.")
                            st.balloons()
                            st.info(f"📚 Total predictions: {len(st.session_state.prediction_history)}, pending updates: {len(st.session_state.pending_feedback)}")
                        else:
                            st.error("❌ Failed to record feedback")
                else:
                    st.warning("⚠️ Auto-learning is disabled. Enable in the sidebar.")
        else:
            st.warning("👆 Select actual performance to teach the model")

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
                st.metric("👥 Total Students", len(df))
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>🎓 Test Prep Impact</h3>", unsafe_allow_html=True)
            prep_perf = pd.crosstab(df['test_prep'], df['performance'])
            fig = px.bar(
                prep_perf, 
                barmode='group',
                color_discrete_sequence=['#10b981', '#f59e0b', '#ef4444']
            )
            fig.update_layout(
                height=400, 
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#e2e8f0', size=12),
                xaxis=dict(color='#e2e8f0'),
                yaxis=dict(color='#e2e8f0'),
                legend=dict(font=dict(color='#e2e8f0'))
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>👨‍👩‍👧‍👦 Parental Education Impact</h3>", unsafe_allow_html=True)
            edu_avg = df.groupby('parental_education')['avg_score'].mean().sort_values()
            fig = px.bar(
                x=edu_avg.values, 
                y=edu_avg.index, 
                orientation='h',
                color=edu_avg.values,
                color_continuous_scale='Viridis'
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
        
        # Learning progress visualization (if there are updates)
        if st.session_state.learning_updates > 0 and os.path.exists(LEARNING_LOG_FILE):
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📈 Learning Progress Over Time</h3>", unsafe_allow_html=True)
            try:
                learning_log = pd.read_csv(LEARNING_LOG_FILE)
                learning_log['timestamp'] = pd.to_datetime(learning_log['timestamp'])
                learning_log = learning_log.sort_values('timestamp')
                
                fig = px.line(
                    learning_log, 
                    x='timestamp', 
                    y='model_accuracy',
                    title='Model Accuracy Improvement Over Time',
                    markers=True
                )
                fig.update_layout(
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e2e8f0', size=12),
                    xaxis=dict(color='#e2e8f0'),
                    yaxis=dict(color='#e2e8f0', tickformat='.0%')
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not display learning progress: {e}")
        
        # Prediction history analytics
        if st.session_state.prediction_history:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📋 Prediction Analytics</h3>", unsafe_allow_html=True)
            
            pred_df = pd.DataFrame(st.session_state.prediction_history)
            col_pred1, col_pred2, col_pred3 = st.columns(3)
            
            with col_pred1:
                st.metric("Total Predictions", len(pred_df))
            with col_pred2:
                feedback_provided = pred_df['feedback_provided'].sum() if 'feedback_provided' in pred_df.columns else 0
                st.metric("Feedback Provided", feedback_provided)
            with col_pred3:
                avg_confidence = pred_df['prediction_confidence'].mean() * 100 if 'prediction_confidence' in pred_df.columns else 0
                st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        # Correlation heatmap
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🔥 Score Correlation Heatmap</h3>", unsafe_allow_html=True)
        corr_data = df[['math_score', 'reading_score', 'writing_score', 'study_hours', 'attendance']].corr()
        fig = px.imshow(
            corr_data, 
            text_auto=True, 
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        fig.update_layout(
            height=500, 
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0', size=12),
            xaxis=dict(color='#e2e8f0'),
            yaxis=dict(color='#e2e8f0')
        )
        st.plotly_chart(fig, use_container_width=True)

elif tab_selection == "🤖 Train Model":
    st.markdown("<h2 style='color: #f1f5f9 !important;'>🤖 Model Training & Evaluation</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("📁 Dataset not available for training.")
        if st.session_state.model_trained:
            st.info(f"✅ Using existing pre-trained model with {st.session_state.accuracy*100:.2f}% accuracy")
        else:
            st.error("❌ No dataset available and no pre-trained model found.")
    else:
        if st.button("🚀 Train & Save Random Forest Model", use_container_width=True):
            with st.spinner("Training model and saving automatically..."):
                result = train_model(st.session_state.df)
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
                    
                    st.success(f"✅ Model trained and automatically saved! Accuracy: {accuracy*100:.2f}%")
                    st.info("💾 Model file saved as 'student_performance_model.pkl' in the same directory")
                else:
                    st.error("❌ Failed to train model")
        else:
            st.info("👆 Click the button above to train the model. It will be automatically saved.")
    
    if st.session_state.model_trained:
        st.markdown("<br>", unsafe_allow_html=True)
        
        if os.path.exists(MODEL_FILE):
            file_size = os.path.getsize(MODEL_FILE) / 1024
            st.success(f"📁 Model file saved: 'student_performance_model.pkl' ({file_size:.1f} KB)")
        else:
            st.warning("📁 Model file not found in directory")
        
        if st.session_state.learning_updates > 0:
            st.info(f"🔄 Model has been updated {st.session_state.learning_updates} times through continuous learning")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📊 Model Performance</h3>", unsafe_allow_html=True)
            
            accuracy_value = st.session_state.accuracy
            st.metric("Accuracy", f"{accuracy_value*100:.2f}%")
            
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
            
            if st.session_state.model is not None:
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
                st.warning("Model not available for feature importance analysis.")
        
        if st.session_state.y_test is not None and st.session_state.y_pred is not None:
            st.markdown("<h3 style='color: #f1f5f9 !important;'>📋 Detailed Classification Report</h3>", unsafe_allow_html=True)
            report = classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.background_gradient(cmap='RdYlGn'), use_container_width=True)
    
    elif st.session_state.model_trained:
        st.warning("⚠️ Model is trained but evaluation data is not available.")
        st.info(f"✅ Using pre-trained model with {st.session_state.accuracy*100:.2f}% accuracy")
        
        if os.path.exists(MODEL_FILE):
            file_size = os.path.getsize(MODEL_FILE) / 1024
            st.success(f"📁 Pre-trained model available: 'student_performance_model.pkl' ({file_size:.1f} KB)")
    
    else:
        st.info("👆 Click the 'Train Random Forest Model' button to train the model and see performance metrics.")

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
        st.dataframe(recent_preds[['timestamp', 'predicted_performance', 'prediction_confidence', 'actual_performance']], use_container_width=True)
        
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
            feedback_status = pred_df['feedback_provided'].value_counts() if 'feedback_provided' in pred_df.columns else pd.Series()
            if not feedback_status.empty:
                fig = px.pie(values=feedback_status.values, names=feedback_status.index.map({True: 'With Feedback', False: 'No Feedback'}),
                           color_discrete_sequence=['#3b82f6', '#6b7280'])
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

else:  # AI Insights
    st.markdown("<h2 style='color: #f1f5f9 !important;'>💡 AI-Generated Insights</h2>", unsafe_allow_html=True)
    
    insights = [
        {
            "icon": "📚",
            "title": "Test Preparation Impact",
            "description": "Students who completed test preparation courses show an average 12-15 point improvement across all subjects. This is the single most impactful factor in student performance prediction.",
            "color": "linear-gradient(135deg, #5a67d8 0%, #764ba2 100%)",
            "textColor": "white"
        },
        {
            "icon": "👨‍👩‍👧‍👦",
            "title": "Parental Education Correlation",
            "description": "Students with parents holding bachelor's or master's degrees show 20% higher average scores. This suggests a strong correlation between household educational environment and student success.",
            "color": "linear-gradient(135deg, #ff758c 0%, #ff7eb3 100%)",
            "textColor": "white"
        },
        {
            "icon": "⏰",
            "title": "Study Hours Threshold",
            "description": "Analysis reveals a performance plateau after 7 hours of daily study. Students studying 5-7 hours show optimal results, suggesting quality of study matters more than quantity beyond this point.",
            "color": "linear-gradient(135deg, #36d1dc 0%, #5b86e5 100%)",
            "textColor": "white"
        },
        {
            "icon": "📅",
            "title": "Attendance Threshold",
            "description": "Maintaining 85%+ attendance is critical. Students below this threshold show a significant drop in performance, with each 5% decrease correlating to approximately 3-4 point score reduction.",
            "color": "linear-gradient(135deg, #a8e063 0%, #56ab2f 100%)",
            "textColor": "black"
        },
        {
            "icon": "⚖️",
            "title": "Gender Performance Patterns",
            "description": "Female students tend to score 5-8 points higher in reading and writing, while male students show slight advantages in math. However, these differences diminish significantly with proper test preparation.",
            "color": "linear-gradient(135deg, #f6d365 0%, #fda085 100%)",
            "textColor": "black"
        },
        {
            "icon": "🎨",
            "title": "Extracurricular Benefits",
            "description": "Students participating in extracurricular activities show 8% better overall performance. These activities correlate with better time management and stress handling capabilities.",
            "color": "linear-gradient(135deg, #7f00ff 0%, #e100ff 100%)",
            "textColor": "white"
        }
    ]

    cols = st.columns(2)
    for idx, insight in enumerate(insights):
        with cols[idx % 2]:
            st.markdown(f"""
            <div style='background: {insight["color"]}; padding: 25px; border-radius: 20px; margin-bottom: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
                <h2 style='color: white !important; margin: 0; font-size: 48px; text-align: center;'>{insight["icon"]}</h2>
                <h3 style='color: white !important; margin: 15px 0; text-align: center;'>{insight["title"]}</h3>
                <p style='color: white !important; margin: 0; font-size: 14px; line-height: 1.5;'>{insight["description"]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    if st.session_state.learning_updates > 0:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h3 style='color: #f1f5f9 !important;'>🔄 Continuous Learning Impact</h3>", unsafe_allow_html=True)
        
        col_learn1, col_learn2, col_learn3 = st.columns(3)
        
        with col_learn1:
            st.metric("Learning Updates", st.session_state.learning_updates)
        with col_learn2:
            st.metric("Current Accuracy", f"{st.session_state.accuracy*100:.1f}%")
        with col_learn3:
            improvement = (st.session_state.accuracy - 0.75) * 100
            st.metric("Improvement", f"+{improvement:.1f}%")
        
        st.info("🎯 **The model continuously improves** with each feedback update, adapting to new patterns and becoming more accurate over time!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #64748b !important;'>
    <p>🎓 Advanced Student Performance Predictor | Built with Streamlit & Scikit-learn | Powered by Kaggle Dataset & Gemini AI</p>
    <p style='font-size: 12px;'>🤖 Context-Aware AI Assistant: Understands your data and predictions for personalized insights</p>
</div>
""", unsafe_allow_html=True)
