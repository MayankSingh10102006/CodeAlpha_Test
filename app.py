# ============================================================
#  AI-Powered Diabetes Prediction System
#  CodeAlpha Internship — Task 4: Disease Prediction
#  Developer: Mayank
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── ML Libraries ─────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             confusion_matrix, classification_report)

# ── Visualization ─────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── PDF Generation ────────────────────────────────────────────
from fpdf import FPDF

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Diabetes Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
#  CUSTOM CSS — Hospital-grade dark/light modern theme
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --primary: #0EA5E9;
        --primary-dark: #0284C7;
        --danger: #EF4444;
        --success: #22C55E;
        --warning: #F59E0B;
        --bg: #0F172A;
        --surface: #1E293B;
        --surface2: #334155;
        --border: #475569;
        --text: #F1F5F9;
        --text-muted: #94A3B8;
        --font: 'Plus Jakarta Sans', sans-serif;
        --mono: 'JetBrains Mono', monospace;
    }

    /* ── Base ── */
    html, body, [class*="css"] {
        font-family: var(--font) !important;
    }
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1a2744 50%, #0F172A 100%);
        color: var(--text);
    }
    .block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1829 0%, #1a2744 100%) !important;
        border-right: 1px solid rgba(14,165,233,0.2) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }

    /* ── Header Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #0c2340 0%, #0a3d62 40%, #1565a8 100%);
        border: 1px solid rgba(14,165,233,0.3);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
        box-shadow: 0 25px 50px rgba(0,0,0,0.5);
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%; right: -20%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(14,165,233,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-banner::after {
        content: '🏥';
        position: absolute;
        right: 3rem; top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.15;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff, #7dd3fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0 0 0.5rem 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #7dd3fc;
        font-size: 1rem;
        font-weight: 400;
        margin: 0;
        opacity: 0.85;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--surface);
        border: 1px solid rgba(71,85,105,0.5);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,0.3); }
    .metric-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary), #38bdf8);
    }
    .metric-value { font-size: 2rem; font-weight: 800; color: var(--primary); font-family: var(--mono); }
    .metric-label { font-size: 0.78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-top: 0.3rem; }

    /* ── Section Headers ── */
    .section-header {
        display: flex; align-items: center; gap: 0.8rem;
        font-size: 1.1rem; font-weight: 700;
        color: var(--text);
        border-bottom: 1px solid rgba(71,85,105,0.4);
        padding-bottom: 0.8rem;
        margin-bottom: 1.5rem;
    }
    .section-icon {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, var(--primary), #38bdf8);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }

    /* ── Input Cards ── */
    .input-card {
        background: var(--surface);
        border: 1px solid rgba(71,85,105,0.4);
        border-radius: 14px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: border-color 0.2s;
    }
    .input-card:hover { border-color: rgba(14,165,233,0.4); }
    .input-label {
        font-size: 0.82rem; font-weight: 600;
        color: var(--text-muted); text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 0.4rem;
    }
    .normal-range {
        font-size: 0.75rem; color: #22C55E;
        background: rgba(34,197,94,0.1);
        border-radius: 6px; padding: 0.2rem 0.5rem;
        display: inline-block; margin-top: 0.3rem;
        border: 1px solid rgba(34,197,94,0.2);
    }

    /* ── Predict Button ── */
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9, #0284C7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.85rem 2.5rem !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        font-family: var(--font) !important;
        letter-spacing: 0.03em !important;
        width: 100% !important;
        transition: all 0.2s !important;
        box-shadow: 0 4px 20px rgba(14,165,233,0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 30px rgba(14,165,233,0.5) !important;
    }

    /* ── Result Cards ── */
    .result-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.05));
        border: 1.5px solid rgba(239,68,68,0.5);
        border-radius: 18px; padding: 2rem;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.05));
        border: 1.5px solid rgba(34,197,94,0.5);
        border-radius: 18px; padding: 2rem;
        text-align: center;
    }
    .result-title { font-size: 1.8rem; font-weight: 800; margin: 0.5rem 0; }
    .result-prob { font-size: 3rem; font-weight: 800; font-family: var(--mono); }
    .result-high .result-title, .result-high .result-prob { color: #EF4444; }
    .result-low .result-title, .result-low .result-prob { color: #22C55E; }

    /* ── Recommendation Cards ── */
    .rec-card {
        background: var(--surface);
        border-left: 3px solid var(--primary);
        border-radius: 0 10px 10px 0;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.88rem;
        color: var(--text);
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--surface) !important;
        border-radius: 12px !important;
        padding: 0.3rem !important;
        border: 1px solid rgba(71,85,105,0.3) !important;
        gap: 0.2rem !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        padding: 0.5rem 1.2rem !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0EA5E9, #0284C7) !important;
        color: white !important;
    }

    /* ── Slider ── */
    .stSlider > div > div { background: rgba(14,165,233,0.2) !important; }
    .stSlider [data-testid="stThumbValue"] { color: var(--primary) !important; }

    /* ── Sidebar specific ── */
    .sidebar-section {
        background: rgba(14,165,233,0.05);
        border: 1px solid rgba(14,165,233,0.15);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
    }
    .sidebar-title {
        font-size: 0.75rem; font-weight: 700;
        color: var(--primary); text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 0.6rem;
    }

    /* ── History Table ── */
    .history-row {
        background: var(--surface);
        border-radius: 10px; padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        display: flex; justify-content: space-between; align-items: center;
        border: 1px solid rgba(71,85,105,0.3);
        font-size: 0.85rem;
    }

    /* ── Plotly charts ── */
    .js-plotly-plot { border-radius: 14px !important; overflow: hidden; }

    /* ── Misc ── */
    hr { border-color: rgba(71,85,105,0.3) !important; }
    .stExpander { background: var(--surface) !important; border-radius: 12px !important; border: 1px solid rgba(71,85,105,0.3) !important; }
    [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {
        background: var(--surface2) !important;
        color: var(--text) !important;
        border: 1px solid rgba(71,85,105,0.5) !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  DATA & MODEL TRAINING
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    """Load the Pima Indians Diabetes Dataset."""
    df = pd.read_csv("diabetes.csv")
    return df

@st.cache_resource
def train_model(df):
    """Train Random Forest model and return model + metrics."""
    # ── Preprocessing ──
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df_clean = df.copy()
    df_clean[cols_with_zeros] = df_clean[cols_with_zeros].replace(0, np.nan)
    df_clean.fillna(df_clean.median(), inplace=True)

    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # ── Train Random Forest ──
    model = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    # ── Metrics ──
    y_pred = model.predict(X_test_sc)
    y_prob = model.predict_proba(X_test_sc)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)
    cm     = confusion_matrix(y_test, y_pred)

    # Save model & scaler
    with open("model.pkl", "wb") as f: pickle.dump(model, f)
    with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)

    return model, scaler, acc, auc, cm, X_test_sc, y_test, y_prob, X.columns.tolist()

def predict(model, scaler, input_data):
    """Run prediction on user input."""
    input_arr = np.array(input_data).reshape(1, -1)
    input_sc  = scaler.transform(input_arr)
    pred      = model.predict(input_sc)[0]
    prob      = model.predict_proba(input_sc)[0]
    return pred, prob


# ══════════════════════════════════════════════════════════════
#  PDF REPORT GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_pdf_report(inputs, prediction, probability, feature_names):
    """Generate a downloadable PDF prediction report."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(15, 23, 42)

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(14, 165, 233)
    pdf.cell(0, 15, "AI Diabetes Prediction Report", ln=True, align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(5)

    # Result
    pdf.set_font("Helvetica", "B", 14)
    if prediction == 1:
        pdf.set_text_color(239, 68, 68)
        pdf.cell(0, 10, f"Result: HIGH RISK — {probability[1]*100:.1f}% Probability", ln=True)
    else:
        pdf.set_text_color(34, 197, 94)
        pdf.cell(0, 10, f"Result: LOW RISK — {probability[0]*100:.1f}% Confidence", ln=True)

    pdf.ln(5)

    # Input values
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "Patient Input Summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    labels = ["Pregnancies", "Glucose (mg/dL)", "Blood Pressure (mm Hg)",
              "Skin Thickness (mm)", "Insulin (mu U/ml)", "BMI",
              "Diabetes Pedigree Function", "Age"]
    for label, val in zip(labels, inputs):
        pdf.cell(90, 8, label, border=1)
        pdf.cell(0, 8, str(val), border=1, ln=True)

    pdf.ln(5)

    # Recommendations
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, "Medical Recommendations", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(71, 85, 105)
    if prediction == 1:
        recs = [
            "Consult an endocrinologist immediately.",
            "Monitor blood glucose levels daily.",
            "Adopt a low-carb, high-fiber diet.",
            "Exercise at least 150 min/week.",
            "Consider HbA1c and fasting glucose tests."
        ]
    else:
        recs = [
            "Maintain a balanced diet rich in vegetables.",
            "Exercise regularly — 30 min/day.",
            "Annual glucose screening recommended.",
            "Keep BMI within healthy range (18.5–24.9).",
            "Stay hydrated and reduce sugar intake."
        ]
    for r in recs:
        pdf.cell(0, 7, f"  • {r}", ln=True)

    pdf.ln(5)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, "Disclaimer: This is an AI prediction tool for educational purposes only.", ln=True)
    pdf.cell(0, 6, "Always consult a qualified medical professional for diagnosis.", ln=True)

    return pdf.output(dest="S").encode("latin-1")


# ══════════════════════════════════════════════════════════════
#  SESSION STATE — Prediction History
# ══════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []


# ══════════════════════════════════════════════════════════════
#  LOAD DATA & TRAIN MODEL
# ══════════════════════════════════════════════════════════════
try:
    df = load_data()
    model, scaler, acc, auc, cm, X_test_sc, y_test, y_prob, feature_names = train_model(df)
    model_ready = True
except FileNotFoundError:
    st.error("⚠️ **diabetes.csv not found!** Please place it in the same directory as app.py")
    model_ready = False
    st.stop()


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
        <div style='font-size:3rem;'>🏥</div>
        <div style='font-size:1rem; font-weight:800; color:#0EA5E9;'>DiabetesAI</div>
        <div style='font-size:0.72rem; color:#94A3B8;'>Powered by Random Forest</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Model metrics
    st.markdown("<div class='sidebar-title'>📊 Model Performance</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{acc*100:.1f}%")
    c2.metric("ROC-AUC", f"{auc:.3f}")

    st.divider()

    # About Dataset
    st.markdown("""
    <div class='sidebar-section'>
        <div class='sidebar-title'>📁 Dataset Info</div>
        <div style='font-size:0.82rem; color:#CBD5E1; line-height:1.6;'>
            <b>Name:</b> Pima Indians Diabetes<br>
            <b>Source:</b> UCI ML Repository<br>
            <b>Records:</b> 768 patients<br>
            <b>Features:</b> 8 medical attributes<br>
            <b>Target:</b> Diabetic / Non-Diabetic
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Developer info
    st.markdown("""
    <div class='sidebar-section'>
        <div class='sidebar-title'>👨‍💻 Developer</div>
        <div style='font-size:0.82rem; color:#CBD5E1; line-height:1.8;'>
            <b>Name:</b> Mayank<br>
            <b>Program:</b> CodeAlpha ML Internship<br>
            <b>Task:</b> Disease Prediction (Task 4)<br>
            <b>Model:</b> Random Forest Classifier
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div style='font-size:0.72rem; color:#64748B; padding: 0.8rem; background: rgba(239,68,68,0.05);
    border-radius:8px; border:1px solid rgba(239,68,68,0.15); margin-top:1rem;'>
    ⚠️ <b>Disclaimer:</b> For educational use only. Not a substitute for medical advice.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ══════════════════════════════════════════════════════════════

# ── Hero Banner ──
st.markdown("""
<div class='hero-banner'>
    <p class='hero-title'>AI-Powered Diabetes<br>Prediction System</p>
    <p class='hero-subtitle'>Advanced machine learning for early diabetes risk detection • Random Forest Classifier</p>
</div>
""", unsafe_allow_html=True)

# ── Top Metrics ──
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{len(df)}</div>
        <div class='metric-label'>Total Records</div>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{acc*100:.1f}%</div>
        <div class='metric-label'>Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{auc:.3f}</div>
        <div class='metric-label'>ROC-AUC Score</div>
    </div>""", unsafe_allow_html=True)
with m4:
    diabetic_pct = df['Outcome'].mean() * 100
    st.markdown(f"""<div class='metric-card'>
        <div class='metric-value'>{diabetic_pct:.1f}%</div>
        <div class='metric-label'>Diabetic in Dataset</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Predict", "📊 Analytics", "📈 Model Insights", "📋 History"])


# ══════════════════════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    # ── INPUT FORM ──
    with col_input:
        st.markdown("""<div class='section-header'>
            <div class='section-icon'>📝</div> Patient Input Parameters
        </div>""", unsafe_allow_html=True)

        r1c1, r1c2 = st.columns(2)
        with r1c1:
            pregnancies = st.slider("🤱 Pregnancies",
                min_value=0, max_value=17, value=3,
                help="Number of times pregnant")
            st.markdown("<div class='normal-range'>Normal: 0–5</div>", unsafe_allow_html=True)

        with r1c2:
            age = st.slider("🎂 Age",
                min_value=1, max_value=100, value=33,
                help="Patient age in years")
            st.markdown("<div class='normal-range'>Dataset range: 21–81</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        r2c1, r2c2 = st.columns(2)
        with r2c1:
            glucose = st.slider("🩸 Glucose Level (mg/dL)",
                min_value=0, max_value=300, value=120,
                help="Plasma glucose concentration (2hr oral glucose tolerance test)")
            st.markdown("<div class='normal-range'>Normal: 70–140 mg/dL</div>", unsafe_allow_html=True)

        with r2c2:
            blood_pressure = st.slider("💓 Blood Pressure (mm Hg)",
                min_value=0, max_value=150, value=72,
                help="Diastolic blood pressure")
            st.markdown("<div class='normal-range'>Normal: 60–90 mm Hg</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        r3c1, r3c2 = st.columns(2)
        with r3c1:
            skin_thickness = st.slider("📏 Skin Thickness (mm)",
                min_value=0, max_value=100, value=23,
                help="Triceps skin fold thickness")
            st.markdown("<div class='normal-range'>Normal: 10–40 mm</div>", unsafe_allow_html=True)

        with r3c2:
            insulin = st.slider("💉 Insulin Level (mu U/ml)",
                min_value=0, max_value=900, value=80,
                help="2-hour serum insulin level")
            st.markdown("<div class='normal-range'>Normal: 16–166 mu U/ml</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        r4c1, r4c2 = st.columns(2)
        with r4c1:
            bmi = st.slider("⚖️ BMI",
                min_value=0.0, max_value=70.0, value=31.5, step=0.1,
                help="Body Mass Index (weight in kg / height in m²)")
            st.markdown("<div class='normal-range'>Normal: 18.5–24.9</div>", unsafe_allow_html=True)

        with r4c2:
            dpf = st.slider("🧬 Diabetes Pedigree Function",
                min_value=0.0, max_value=3.0, value=0.47, step=0.001,
                help="Genetic likelihood of diabetes based on family history")
            st.markdown("<div class='normal-range'>Lower is better</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        predict_btn = st.button("🔬 Predict Diabetes Risk", use_container_width=True)

    # ── RESULT PANEL ──
    with col_result:
        st.markdown("""<div class='section-header'>
            <div class='section-icon'>📋</div> Prediction Result
        </div>""", unsafe_allow_html=True)

        user_inputs = [pregnancies, glucose, blood_pressure,
                       skin_thickness, insulin, bmi, dpf, age]

        if predict_btn:
            pred, prob = predict(model, scaler, user_inputs)

            # ── Store in history ──
            st.session_state.history.append({
                "time": datetime.now().strftime("%H:%M:%S"),
                "glucose": glucose, "bmi": bmi, "age": age,
                "result": "High Risk" if pred == 1 else "Low Risk",
                "probability": f"{max(prob)*100:.1f}%"
            })

            # ── Result Card ──
            if pred == 1:
                st.markdown(f"""
                <div class='result-high'>
                    <div style='font-size:3rem;'>⚠️</div>
                    <div class='result-title'>High Risk of Diabetes</div>
                    <div class='result-prob'>{prob[1]*100:.1f}%</div>
                    <div style='color:#94A3B8; font-size:0.85rem; margin-top:0.5rem;'>Diabetes Probability</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='result-low'>
                    <div style='font-size:3rem;'>✅</div>
                    <div class='result-title'>Low Risk of Diabetes</div>
                    <div class='result-prob'>{prob[0]*100:.1f}%</div>
                    <div style='color:#94A3B8; font-size:0.85rem; margin-top:0.5rem;'>Confidence Score</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauge Chart ──
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob[1] * 100,
                title={"text": "Diabetes Risk Score", "font": {"color": "#CBD5E1", "size": 14}},
                number={"suffix": "%", "font": {"color": "#F1F5F9", "size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                    "bar": {"color": "#EF4444" if pred == 1 else "#22C55E"},
                    "bgcolor": "#1E293B",
                    "bordercolor": "#475569",
                    "steps": [
                        {"range": [0, 30], "color": "rgba(34,197,94,0.15)"},
                        {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                        {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 2}, "value": prob[1]*100}
                }
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#CBD5E1"}, height=220, margin=dict(t=40, b=10, l=10, r=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # ── Recommendations ──
            st.markdown("**📌 Medical Recommendations:**")
            if pred == 1:
                recs = [
                    "🏥 Consult an endocrinologist immediately",
                    "🩸 Monitor blood glucose levels daily",
                    "🥗 Adopt a low-carb, high-fiber diet",
                    "🏃 Exercise at least 150 min/week",
                    "🧪 Get HbA1c & fasting glucose tests"
                ]
                color = "#EF4444"
            else:
                recs = [
                    "🥦 Maintain a balanced, vegetable-rich diet",
                    "🚶 Exercise 30 minutes daily",
                    "📅 Annual glucose screening recommended",
                    "⚖️ Keep BMI in healthy range (18.5–24.9)",
                    "💧 Stay hydrated and reduce sugar intake"
                ]
                color = "#22C55E"

            for r in recs:
                st.markdown(f"""<div class='rec-card' style='border-left-color:{color};'>{r}</div>""",
                            unsafe_allow_html=True)

            # ── Download PDF ──
            st.markdown("<br>", unsafe_allow_html=True)
            try:
                pdf_bytes = generate_pdf_report(user_inputs, pred, prob, feature_names)
                st.download_button(
                    "📥 Download Prediction Report (PDF)",
                    data=pdf_bytes,
                    file_name=f"diabetes_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception:
                st.info("Install `fpdf2` for PDF download: `pip install fpdf2`")

        else:
            # Placeholder before prediction
            st.markdown("""
            <div style='text-align:center; padding:3rem 1rem; color:#475569;'>
                <div style='font-size:4rem; margin-bottom:1rem;'>🔬</div>
                <div style='font-size:1rem; font-weight:600;'>Fill in patient parameters and click Predict</div>
                <div style='font-size:0.82rem; margin-top:0.5rem;'>Results will appear here with full analysis</div>
            </div>
            """, unsafe_allow_html=True)

            # Show live indicators even before prediction
            st.markdown("**📊 Live Indicators:**")

            # Glucose indicator
            fig_gl = go.Figure(go.Indicator(
                mode="gauge+number",
                value=glucose,
                title={"text": "Glucose Level", "font": {"color": "#CBD5E1", "size": 12}},
                number={"suffix": " mg/dL", "font": {"color": "#F1F5F9", "size": 20}},
                gauge={
                    "axis": {"range": [0, 300]},
                    "bar": {"color": "#0EA5E9"},
                    "bgcolor": "#1E293B",
                    "steps": [
                        {"range": [0, 70], "color": "rgba(245,158,11,0.2)"},
                        {"range": [70, 140], "color": "rgba(34,197,94,0.2)"},
                        {"range": [140, 300], "color": "rgba(239,68,68,0.2)"},
                    ]
                }
            ))
            fig_gl.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=180,
                margin=dict(t=40, b=5, l=5, r=5),
                font={"color": "#CBD5E1"}
            )
            st.plotly_chart(fig_gl, use_container_width=True)

            # BMI indicator
            fig_bmi = go.Figure(go.Indicator(
                mode="gauge+number",
                value=bmi,
                title={"text": "BMI", "font": {"color": "#CBD5E1", "size": 12}},
                number={"font": {"color": "#F1F5F9", "size": 20}},
                gauge={
                    "axis": {"range": [0, 70]},
                    "bar": {"color": "#8B5CF6"},
                    "bgcolor": "#1E293B",
                    "steps": [
                        {"range": [0, 18.5], "color": "rgba(245,158,11,0.2)"},
                        {"range": [18.5, 25], "color": "rgba(34,197,94,0.2)"},
                        {"range": [25, 30], "color": "rgba(245,158,11,0.2)"},
                        {"range": [30, 70], "color": "rgba(239,68,68,0.2)"},
                    ]
                }
            ))
            fig_bmi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", height=180,
                margin=dict(t=40, b=5, l=5, r=5),
                font={"color": "#CBD5E1"}
            )
            st.plotly_chart(fig_bmi, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""<div class='section-header'>
        <div class='section-icon'>📊</div> Dataset Analytics
    </div>""", unsafe_allow_html=True)

    a1, a2 = st.columns(2)

    # ── Outcome Distribution ──
    with a1:
        outcome_counts = df['Outcome'].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=["Non-Diabetic", "Diabetic"],
            values=[outcome_counts[0], outcome_counts[1]],
            hole=0.55,
            marker=dict(colors=["#22C55E", "#EF4444"],
                        line=dict(color="#0F172A", width=3)),
            textfont=dict(color="white", size=13)
        ))
        fig_pie.update_layout(
            title=dict(text="Outcome Distribution", font=dict(color="#CBD5E1", size=14)),
            paper_bgcolor="rgba(30,41,59,0.8)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"), height=320,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1")),
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ── Confusion Matrix ──
    with a2:
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=["Non-Diabetic", "Diabetic"],
            y=["Non-Diabetic", "Diabetic"],
            color_continuous_scale=[[0, "#1E293B"], [0.5, "#0369A1"], [1, "#0EA5E9"]]
        )
        fig_cm.update_layout(
            title=dict(text="Confusion Matrix", font=dict(color="#CBD5E1", size=14)),
            paper_bgcolor="rgba(30,41,59,0.8)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"), height=320,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        fig_cm.update_traces(textfont=dict(size=20, color="white"))
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── ROC Curve ──
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'ROC Curve (AUC = {auc:.3f})',
        line=dict(color='#0EA5E9', width=3),
        fill='tozeroy', fillcolor='rgba(14,165,233,0.08)'
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0,1], y=[0,1], mode='lines',
        name='Random Classifier',
        line=dict(color='#475569', dash='dash', width=1.5)
    ))
    fig_roc.update_layout(
        title=dict(text="ROC Curve — Random Forest", font=dict(color="#CBD5E1", size=14)),
        xaxis=dict(title="False Positive Rate", color="#94A3B8", gridcolor="rgba(71,85,105,0.3)"),
        yaxis=dict(title="True Positive Rate", color="#94A3B8", gridcolor="rgba(71,85,105,0.3)"),
        paper_bgcolor="rgba(30,41,59,0.8)", plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(color="#CBD5E1"), height=350,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#CBD5E1")),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_roc, use_container_width=True)

    # ── Feature Distribution ──
    st.markdown("""<div class='section-header' style='margin-top:1rem;'>
        <div class='section-icon'>📈</div> Feature Distributions by Outcome
    </div>""", unsafe_allow_html=True)

    feat_select = st.selectbox("Select Feature", feature_names, index=1)
    fig_hist = go.Figure()
    for outcome, color, label in zip([0, 1], ["#22C55E", "#EF4444"], ["Non-Diabetic", "Diabetic"]):
        fig_hist.add_trace(go.Histogram(
            x=df[df['Outcome'] == outcome][feat_select],
            name=label, opacity=0.75,
            marker_color=color, nbinsx=30
        ))
    fig_hist.update_layout(
        barmode='overlay',
        title=dict(text=f"{feat_select} Distribution by Outcome", font=dict(color="#CBD5E1", size=14)),
        xaxis=dict(title=feat_select, color="#94A3B8", gridcolor="rgba(71,85,105,0.3)"),
        yaxis=dict(title="Count", color="#94A3B8", gridcolor="rgba(71,85,105,0.3)"),
        paper_bgcolor="rgba(30,41,59,0.8)", plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(color="#CBD5E1"), height=320,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(t=50, b=40, l=40, r=20)
    )
    st.plotly_chart(fig_hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  TAB 3 — MODEL INSIGHTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""<div class='section-header'>
        <div class='section-icon'>🧠</div> Feature Importance — Random Forest
    </div>""", unsafe_allow_html=True)

    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig_fi = go.Figure(go.Bar(
        x=fi_df["Importance"],
        y=fi_df["Feature"],
        orientation="h",
        marker=dict(
            color=fi_df["Importance"],
            colorscale=[[0, "#1E3A5F"], [0.5, "#0369A1"], [1, "#0EA5E9"]],
            line=dict(color="rgba(0,0,0,0)")
        ),
        text=[f"{v:.3f}" for v in fi_df["Importance"]],
        textposition="outside",
        textfont=dict(color="#CBD5E1", size=11)
    ))
    fig_fi.update_layout(
        title=dict(text="Which features matter most for prediction?", font=dict(color="#CBD5E1", size=13)),
        xaxis=dict(title="Importance Score", color="#94A3B8", gridcolor="rgba(71,85,105,0.3)"),
        yaxis=dict(color="#94A3B8"),
        paper_bgcolor="rgba(30,41,59,0.8)", plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(color="#CBD5E1"), height=380,
        margin=dict(t=50, b=40, l=150, r=60)
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── Correlation Heatmap ──
    st.markdown("""<div class='section-header'>
        <div class='section-icon'>🔗</div> Feature Correlation Heatmap
    </div>""", unsafe_allow_html=True)

    corr = df.corr()
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.columns.tolist(),
        colorscale=[[0, "#1E3A5F"], [0.5, "#0F172A"], [1, "#0EA5E9"]],
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        zmin=-1, zmax=1
    ))
    fig_corr.update_layout(
        paper_bgcolor="rgba(30,41,59,0.8)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#CBD5E1"), height=420,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Model Info ──
    with st.expander("⚙️ Model Configuration Details"):
        st.json({
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "test_size": "20%",
            "preprocessing": "StandardScaler + median imputation",
            "accuracy": f"{acc*100:.2f}%",
            "roc_auc": f"{auc:.4f}"
        })


# ══════════════════════════════════════════════════════════════
#  TAB 4 — PREDICTION HISTORY
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""<div class='section-header'>
        <div class='section-icon'>📋</div> Prediction History (This Session)
    </div>""", unsafe_allow_html=True)

    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)

        # Summary stats
        h1, h2, h3 = st.columns(3)
        h1.metric("Total Predictions", len(history_df))
        h2.metric("High Risk", len(history_df[history_df["result"] == "High Risk"]))
        h3.metric("Low Risk", len(history_df[history_df["result"] == "Low Risk"]))

        st.markdown("<br>", unsafe_allow_html=True)

        for i, row in history_df.iloc[::-1].iterrows():
            color = "#EF4444" if row["result"] == "High Risk" else "#22C55E"
            icon  = "⚠️" if row["result"] == "High Risk" else "✅"
            st.markdown(f"""
            <div class='history-row'>
                <span style='color:#94A3B8; font-family:monospace;'>{row['time']}</span>
                <span>Glucose: <b>{row['glucose']}</b> | BMI: <b>{row['bmi']}</b> | Age: <b>{row['age']}</b></span>
                <span style='color:{color}; font-weight:700;'>{icon} {row['result']} ({row['probability']})</span>
            </div>
            """, unsafe_allow_html=True)

        # Download history
        st.markdown("<br>", unsafe_allow_html=True)
        csv = history_df.to_csv(index=False)
        st.download_button("📥 Export History as CSV", csv,
                           "prediction_history.csv", "text/csv",
                           use_container_width=True)

        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    else:
        st.markdown("""
        <div style='text-align:center; padding:3rem; color:#475569;'>
            <div style='font-size:3rem;'>📋</div>
            <div style='font-size:1rem; font-weight:600; margin-top:1rem;'>No predictions yet</div>
            <div style='font-size:0.82rem;'>Go to the Predict tab and run a prediction</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div style='text-align:center; padding: 2rem 0 0.5rem; color:#334155; font-size:0.78rem;'>
    🏥 AI Diabetes Prediction System • CodeAlpha ML Internship • Built with Streamlit & Scikit-learn<br>
    <span style='color:#1E293B;'>⚠️ For educational purposes only — not a substitute for medical advice</span>
</div>
""", unsafe_allow_html=True)
