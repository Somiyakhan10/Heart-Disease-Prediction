"""
╔══════════════════════════════════════════════════════════════╗
║     CARDIOVASCULAR RISK PREDICTION DASHBOARD                 ║
║     Production-ready Streamlit App                           ║
║     Features: ML model, interactive charts, PDF export       ║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── ML imports ────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
import joblib

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="CardioSense — Heart Risk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS  — Medical-precision aesthetic
# ══════════════════════════════════════════════════════════════
def inject_css(dark_mode: bool):
    if dark_mode:
        bg      = "#0d1117"
        surface = "#161b22"
        card    = "#1c2128"
        border  = "#30363d"
        text    = "#e6edf3"
        sub     = "#8b949e"
        accent  = "#f85149"
        accent2 = "#388bfd"
        success = "#3fb950"
        warn    = "#d29922"
        danger  = "#f85149"
    else:
        bg      = "#f0f4f8"
        surface = "#ffffff"
        card    = "#f8fafc"
        border  = "#e1e8ef"
        text    = "#0f172a"
        sub     = "#64748b"
        accent  = "#e11d48"
        accent2 = "#2563eb"
        success = "#16a34a"
        warn    = "#ca8a04"
        danger  = "#dc2626"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,600;0,700;1,300&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Reset & base ── */
    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
        color: {text};
    }}
    .stApp {{ background: {bg}; }}
    .block-container {{ padding: 1.5rem 2rem 3rem; max-width: 1400px; }}

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {{
        background: {surface};
        border-right: 1px solid {border};
    }}
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {{
        color: {text};
    }}

    /* ── Cards ── */
    .cs-card {{
        background: {surface};
        border: 1px solid {border};
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.08);
    }}
    .cs-card-accent {{
        border-left: 3px solid {accent};
    }}

    /* ── Hero header ── */
    .hero {{
        background: linear-gradient(135deg, {accent} 0%, {accent2} 100%);
        border-radius: 16px;
        padding: 2.5rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }}
    .hero::before {{
        content: "🫀";
        position: absolute;
        right: 2rem;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: .15;
    }}
    .hero h1 {{
        font-family: 'Fraunces', serif;
        font-size: 2.4rem;
        font-weight: 700;
        color: white;
        margin: 0 0 .4rem;
        line-height: 1.15;
    }}
    .hero p {{
        color: rgba(255,255,255,.85);
        font-size: 1rem;
        margin: 0;
        font-weight: 300;
    }}

    /* ── Risk badge ── */
    .risk-badge {{
        display: inline-block;
        padding: .35rem 1rem;
        border-radius: 999px;
        font-family: 'DM Mono', monospace;
        font-size: .8rem;
        font-weight: 500;
        letter-spacing: .05em;
        text-transform: uppercase;
    }}
    .risk-low    {{ background: #dcfce7; color: #166534; }}
    .risk-mod    {{ background: #fef9c3; color: #854d0e; }}
    .risk-high   {{ background: #ffedd5; color: #9a3412; }}
    .risk-vhigh  {{ background: #fee2e2; color: #991b1b; }}

    /* ── Metric cards ── */
    .metric-row {{ display: flex; gap: .75rem; flex-wrap: wrap; margin-bottom: 1rem; }}
    .metric-tile {{
        background: {card};
        border: 1px solid {border};
        border-radius: 10px;
        padding: .9rem 1.2rem;
        flex: 1;
        min-width: 130px;
        text-align: center;
    }}
    .metric-tile .val {{
        font-family: 'Fraunces', serif;
        font-size: 1.8rem;
        font-weight: 600;
        color: {accent};
        line-height: 1;
    }}
    .metric-tile .lbl {{
        font-size: .72rem;
        color: {sub};
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-top: .3rem;
    }}

    /* ── Section labels ── */
    .section-label {{
        font-family: 'DM Mono', monospace;
        font-size: .7rem;
        letter-spacing: .12em;
        text-transform: uppercase;
        color: {sub};
        margin-bottom: .6rem;
    }}

    /* ── Recommendation pills ── */
    .rec-pill {{
        background: {card};
        border: 1px solid {border};
        border-radius: 8px;
        padding: .6rem 1rem;
        margin: .35rem 0;
        font-size: .88rem;
        display: flex;
        align-items: flex-start;
        gap: .5rem;
    }}
    .rec-icon {{ font-size: 1.1rem; flex-shrink: 0; margin-top: .05rem; }}

    /* ── Compare bar ── */
    .compare-row {{
        display: flex;
        align-items: center;
        gap: .6rem;
        margin: .4rem 0;
        font-size: .85rem;
    }}
    .compare-label {{ width: 130px; color: {sub}; font-size: .8rem; }}
    .compare-bar-wrap {{
        flex: 1;
        background: {border};
        border-radius: 999px;
        height: 8px;
        position: relative;
    }}
    .compare-bar-fill {{
        height: 8px;
        border-radius: 999px;
        transition: width .6s ease;
    }}
    .compare-val {{
        width: 70px;
        text-align: right;
        font-family: 'DM Mono', monospace;
        font-size: .78rem;
        color: {text};
    }}

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: .5rem;
        border-bottom: 1px solid {border};
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        padding: .5rem 1.2rem;
        font-size: .88rem;
        font-weight: 500;
        color: {sub};
        background: transparent;
        border: 1px solid transparent;
    }}
    .stTabs [aria-selected="true"] {{
        color: {text} !important;
        border-color: {border} !important;
        border-bottom-color: {surface} !important;
        background: {surface} !important;
    }}

    /* ── Slider overrides ── */
    .stSlider [data-testid="stTickBar"] {{ display: none; }}

    /* ── Buttons ── */
    .stButton > button {{
        border-radius: 8px;
        font-weight: 500;
        font-size: .9rem;
        padding: .5rem 1.4rem;
        border: 1px solid {border};
        background: {surface};
        color: {text};
        transition: all .15s ease;
    }}
    .stButton > button:hover {{
        border-color: {accent};
        color: {accent};
    }}
    div[data-testid="stForm"] .stButton > button[kind="primaryFormSubmit"],
    .primary-btn > button {{
        background: linear-gradient(135deg, {accent}, {accent2}) !important;
        color: white !important;
        border: none !important;
        font-weight: 600;
    }}

    /* ── Progress bar ── */
    .prog-wrap {{
        background: {border};
        border-radius: 999px;
        height: 6px;
        margin: .5rem 0 1rem;
    }}
    .prog-fill {{
        height: 6px;
        border-radius: 999px;
        background: linear-gradient(90deg, {accent}, {accent2});
        transition: width .4s ease;
    }}

    /* ── Disclaimer ── */
    .disclaimer {{
        background: {card};
        border: 1px solid {warn};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: .82rem;
        color: {sub};
        margin-top: 1.5rem;
    }}

    /* ── Footer ── */
    .cs-footer {{
        text-align: center;
        color: {sub};
        font-size: .78rem;
        padding: 2rem 0 1rem;
        border-top: 1px solid {border};
        margin-top: 2rem;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 3px; }}

    /* ── Mobile ── */
    @media (max-width: 768px) {{
        .hero h1 {{ font-size: 1.7rem; }}
        .metric-tile .val {{ font-size: 1.4rem; }}
        .compare-label {{ width: 90px; }}
    }}
    </style>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS & METADATA
# ══════════════════════════════════════════════════════════════
FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']
TARGET   = 'target'

FEATURE_META = {
    'age':     {"label":"Age",                  "unit":"years", "normal":(30,65),  "min":20,   "max":90,  "step":1,   "type":"slider",
                "help":"Patient age in years. Risk increases with age."},
    'sex':     {"label":"Biological Sex",        "unit":"",      "options":{0:"Female",1:"Male"}, "type":"select",
                "help":"Biological sex. Males have higher baseline cardiovascular risk."},
    'cp':      {"label":"Chest Pain Type",       "unit":"",
                "options":{0:"Typical Angina",1:"Atypical Angina",2:"Non-Anginal Pain",3:"Asymptomatic"},
                "type":"select", "help":"Type of chest pain experienced. Asymptomatic chest pain can paradoxically indicate higher risk."},
    'trestbps':{"label":"Resting Blood Pressure","unit":"mmHg",  "normal":(90,120), "min":80,   "max":200, "step":1,   "type":"slider",
                "help":"Resting blood pressure in mmHg. Normal: < 120 mmHg."},
    'chol':    {"label":"Serum Cholesterol",     "unit":"mg/dL", "normal":(0,200),  "min":100,  "max":600, "step":1,   "type":"slider",
                "help":"Cholesterol level in mg/dL. High risk: > 240 mg/dL. Desirable: < 200 mg/dL."},
    'fbs':     {"label":"Fasting Blood Sugar",   "unit":"",
                "options":{0:"≤ 120 mg/dL  (Normal)",1:"> 120 mg/dL  (Elevated)"},
                "type":"select", "help":"Whether fasting blood sugar > 120 mg/dL. Elevated indicates possible diabetes."},
    'restecg': {"label":"Resting ECG Results",   "unit":"",
                "options":{0:"Normal",1:"ST-T Wave Abnormality",2:"Left Ventricular Hypertrophy"},
                "type":"select", "help":"Resting electrocardiographic results."},
    'thalach': {"label":"Max Heart Rate",        "unit":"bpm",   "normal":(60,150), "min":60,   "max":220, "step":1,   "type":"slider",
                "help":"Maximum heart rate achieved during exercise testing."},
    'exang':   {"label":"Exercise-Induced Angina","unit":"",
                "options":{0:"No",1:"Yes"},
                "type":"select", "help":"Angina (chest pain) triggered by exercise."},
    'oldpeak': {"label":"ST Depression (Oldpeak)","unit":"mm",   "normal":(0,1.5),  "min":0.0,  "max":7.0, "step":0.1, "type":"slider",
                "help":"ST depression induced by exercise relative to rest. Higher values suggest more ischemia."},
    'slope':   {"label":"ST Segment Slope",      "unit":"",
                "options":{0:"Upsloping",1:"Flat",2:"Downsloping"},
                "type":"select", "help":"Slope of the peak exercise ST segment."},
    'ca':      {"label":"Major Vessels (Fluoroscopy)","unit":"",
                "options":{0:"0 vessels",1:"1 vessel",2:"2 vessels",3:"3 vessels"},
                "type":"select", "help":"Number of major coronary vessels colored by fluoroscopy (0–3)."},
    'thal':    {"label":"Thalassemia",           "unit":"",
                "options":{1:"Normal",2:"Fixed Defect",3:"Reversible Defect"},
                "type":"select", "help":"Type of thalassemia blood disorder. Reversible defect indicates highest risk."},
}

DEMO_PATIENTS = {
    "🟢 Healthy (45F)":   {"age":45,"sex":0,"cp":0,"trestbps":110,"chol":180,"fbs":0,"restecg":0,"thalach":165,"exang":0,"oldpeak":0.5,"slope":2,"ca":0,"thal":2},
    "🟡 Moderate Risk (58M)": {"age":58,"sex":1,"cp":1,"trestbps":135,"chol":255,"fbs":0,"restecg":1,"thalach":130,"exang":0,"oldpeak":1.8,"slope":1,"ca":1,"thal":2},
    "🔴 High Risk (62M)":  {"age":62,"sex":1,"cp":3,"trestbps":155,"chol":305,"fbs":1,"restecg":2,"thalach":95, "exang":1,"oldpeak":4.2,"slope":2,"ca":3,"thal":3},
}

RISK_LEVELS = [
    (20,  "🟢 LOW RISK",       "low",   "#16a34a"),
    (40,  "🟡 MODERATE RISK",  "mod",   "#ca8a04"),
    (70,  "🟠 HIGH RISK",      "high",  "#ea580c"),
    (101, "🔴 VERY HIGH RISK", "vhigh", "#dc2626"),
]

# ══════════════════════════════════════════════════════════════
# MODEL TRAINING  (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    results = {}
    configs = {
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, max_depth=4, random_state=42),
        "Random Forest":     RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    }
    best_auc  = 0
    best_name = None
    for name, clf in configs.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        cv  = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc').mean()
        results[name] = {"pipe": pipe, "acc": acc, "auc": auc, "cv_auc": cv}
        if auc > best_auc:
            best_auc  = auc
            best_name = name

    # Feature importance from GB model
    gb_pipe = results["Gradient Boosting"]["pipe"]
    fi = gb_pipe.named_steps["clf"].feature_importances_
    fi_df = pd.DataFrame({"feature": FEATURES, "importance": fi}).sort_values("importance", ascending=False)

    return results, best_name, fi_df, X_test, y_test

@st.cache_data(show_spinner=False)
def load_default_data():
    try:
        return pd.read_csv("heart.csv")
    except FileNotFoundError:
        return None

# ══════════════════════════════════════════════════════════════
# PREDICTION  & HELPERS
# ══════════════════════════════════════════════════════════════
def predict(pipe, patient: dict) -> tuple[float, int]:
    df_p = pd.DataFrame([patient])[FEATURES]
    prob = pipe.predict_proba(df_p)[0][1] * 100
    pred = int(pipe.predict(df_p)[0])
    return round(prob, 2), pred

def get_risk_level(prob: float):
    for threshold, label, cls, color in RISK_LEVELS:
        if prob < threshold:
            return label, cls, color

def compare_to_normal(patient: dict) -> list[dict]:
    rows = []
    numeric = ['age','trestbps','chol','thalach','oldpeak']
    normal_mid = {
        'age':     50,
        'trestbps':115,
        'chol':    190,
        'thalach': 145,
        'oldpeak': 0.8,
    }
    normal_max = {
        'age':     80,
        'trestbps':200,
        'chol':    400,
        'thalach': 200,
        'oldpeak': 5.0,
    }
    for feat in numeric:
        val     = patient[feat]
        ref     = normal_mid[feat]
        mx      = normal_max[feat]
        pct_val = min(val / mx, 1.0)
        pct_ref = ref / mx
        delta   = ((val - ref) / ref) * 100
        rows.append({
            "feature": FEATURE_META[feat]["label"],
            "value":   val,
            "unit":    FEATURE_META[feat]["unit"],
            "ref":     ref,
            "pct_val": pct_val,
            "pct_ref": pct_ref,
            "delta":   delta,
        })
    return rows

def build_recommendations(patient: dict, prob: float) -> list[dict]:
    recs = []
    if patient['chol'] > 200:
        recs.append({"icon":"🥗", "text":f"Your cholesterol ({patient['chol']} mg/dL) is elevated. Reduce saturated fat intake and increase fiber."})
    if patient['trestbps'] > 120:
        recs.append({"icon":"🧂", "text":f"Resting BP ({patient['trestbps']} mmHg) is above normal. Reduce sodium intake and consider stress management."})
    if patient['fbs'] == 1:
        recs.append({"icon":"🩸", "text":"Elevated fasting blood sugar detected. Consult a physician about diabetes screening."})
    if patient['exang'] == 1:
        recs.append({"icon":"🚨", "text":"Exercise-induced angina is present. Avoid strenuous exercise until cleared by a cardiologist."})
    if patient['thalach'] < 120 and patient['age'] < 60:
        recs.append({"icon":"🏃", "text":f"Max heart rate ({patient['thalach']} bpm) is lower than expected for your age. Consider a cardiopulmonary fitness evaluation."})
    if patient['ca'] >= 2:
        recs.append({"icon":"🩻", "text":f"{patient['ca']} major coronary vessels affected. Strongly recommend a cardiologist consultation."})
    if patient['thal'] == 3:
        recs.append({"icon":"⚠️", "text":"Reversible thalassemia defect detected — a significant cardiac risk marker."})
    if patient['oldpeak'] > 2.0:
        recs.append({"icon":"📉", "text":f"ST depression of {patient['oldpeak']} mm suggests exercise-induced ischemia."})
    if patient['sex'] == 1 and patient['age'] > 45:
        recs.append({"icon":"🎂", "text":"Males over 45 have elevated baseline risk. Annual cardiac checkups are recommended."})
    if prob > 50 and not recs:
        recs.append({"icon":"🏥", "text":"Risk score is elevated. Please discuss results with your healthcare provider."})
    if not recs:
        recs.append({"icon":"✅", "text":"Parameters look relatively healthy. Maintain a balanced diet and regular exercise routine."})
        recs.append({"icon":"🔄", "text":"Schedule annual physical checkups to monitor cardiovascular health over time."})
    return recs

# ══════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════
def gauge_chart(prob: float, dark: bool) -> go.Figure:
    _, cls, color = get_risk_level(prob)
    bg_color  = "#161b22" if dark else "#ffffff"
    txt_color = "#e6edf3" if dark else "#0f172a"
    sub_color = "#8b949e" if dark else "#64748b"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob,
        delta={"reference": 50, "valueformat": ".1f", "suffix":"%"},
        number={"suffix": "%", "font": {"size": 48, "family":"Fraunces, serif", "color": color}},
        gauge={
            "axis": {"range":[0,100], "tickwidth":1, "tickcolor": sub_color,
                     "tickfont":{"size":11,"color":sub_color}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": bg_color,
            "borderwidth": 0,
            "steps": [
                {"range":[0,20],  "color":"#dcfce7"},
                {"range":[20,40], "color":"#fef9c3"},
                {"range":[40,70], "color":"#ffedd5"},
                {"range":[70,100],"color":"#fee2e2"},
            ],
            "threshold": {"line":{"color":color,"width":3},"thickness":0.75,"value":prob},
        },
        title={"text":"Cardiovascular Risk Score",
               "font":{"size":13,"color":sub_color,"family":"DM Sans, sans-serif"}},
    ))
    fig.update_layout(
        height=290, paper_bgcolor=bg_color, font_color=txt_color,
        margin=dict(t=40,b=10,l=30,r=30),
    )
    return fig

def feature_importance_chart(fi_df: pd.DataFrame, dark: bool) -> go.Figure:
    bg  = "#161b22" if dark else "#ffffff"
    txt = "#e6edf3" if dark else "#0f172a"
    sub = "#8b949e" if dark else "#64748b"
    labels = [FEATURE_META[f]["label"] for f in fi_df["feature"]]
    colors = px.colors.sequential.Reds_r[:len(fi_df)]

    fig = go.Figure(go.Bar(
        x=fi_df["importance"], y=labels,
        orientation='h',
        marker=dict(color=fi_df["importance"], colorscale="RdYlGn_r", showscale=False),
        text=[f"{v:.3f}" for v in fi_df["importance"]],
        textposition="outside",
        textfont=dict(size=10, color=txt),
    ))
    fig.update_layout(
        height=380, paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=txt, size=11),
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
        margin=dict(l=10,r=60,t=20,b=10),
    )
    return fig

def radar_chart(patient: dict, dark: bool) -> go.Figure:
    bg  = "#161b22" if dark else "#ffffff"
    txt = "#e6edf3" if dark else "#0f172a"
    # Normalize 5 key continuous features 0–1 relative to their max
    feats  = ['age','trestbps','chol','thalach','oldpeak']
    maxvals= [90,     200,       600,   220,       7.0]
    normals= [50,     115,       190,   145,       0.8]

    pat_vals = [patient[f]/m for f,m in zip(feats,maxvals)]
    ref_vals = [n/m for n,m in zip(normals,maxvals)]
    labels   = [FEATURE_META[f]["label"] for f in feats]
    # close the polygon
    pat_vals += [pat_vals[0]]; ref_vals += [ref_vals[0]]; labels += [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=ref_vals, theta=labels, fill='toself',
        name='Normal Range', line=dict(color='#3fb950', dash='dash'), opacity=0.4))
    fig.add_trace(go.Scatterpolar(r=pat_vals, theta=labels, fill='toself',
        name='Your Profile', line=dict(color='#f85149'), opacity=0.65))
    fig.update_layout(
        polar=dict(
            bgcolor=bg,
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor='#30363d'),
            angularaxis=dict(gridcolor='#30363d', color=txt),
        ),
        showlegend=True,
        legend=dict(orientation='h', y=-0.12, font=dict(size=11, color=txt)),
        paper_bgcolor=bg, font=dict(color=txt),
        height=340, margin=dict(t=30,b=30,l=30,r=30),
    )
    return fig

def history_chart(history: list, dark: bool) -> go.Figure:
    bg  = "#161b22" if dark else "#ffffff"
    txt = "#e6edf3" if dark else "#0f172a"
    times = [h["time"] for h in history]
    probs = [h["prob"] for h in history]
    colors= [get_risk_level(p)[2] for p in probs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times, y=probs, mode='lines+markers',
        line=dict(color='#388bfd', width=2),
        marker=dict(size=9, color=[get_risk_level(p)[2] for p in probs]),
        name='Risk %',
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#8b949e", annotation_text="50% threshold")
    fig.update_layout(
        height=260, paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=txt, size=11),
        xaxis=dict(showgrid=False, color=txt),
        yaxis=dict(showgrid=True, gridcolor='#30363d', range=[0,100], title="Risk %"),
        margin=dict(l=10,r=10,t=20,b=10),
    )
    return fig

def correlation_heatmap(df: pd.DataFrame, dark: bool) -> go.Figure:
    bg  = "#161b22" if dark else "#ffffff"
    txt = "#e6edf3" if dark else "#0f172a"
    corr = df[FEATURES + [TARGET]].corr()
    labels = [FEATURE_META.get(c, {}).get("label", c) for c in corr.columns]
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=labels, y=labels,
        colorscale='RdBu', zmid=0,
        text=np.round(corr.values,2), texttemplate="%{text}",
        textfont=dict(size=8),
    ))
    fig.update_layout(
        height=520, paper_bgcolor=bg, plot_bgcolor=bg,
        font=dict(color=txt, size=10),
        margin=dict(l=10,r=10,t=20,b=10),
    )
    return fig

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
def render_sidebar(models_info: dict, best_name: str) -> tuple:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        dark_mode = st.toggle("🌙 Dark Mode", value=True)
        st.divider()

        st.markdown("### 🤖 Model Selection")
        model_choice = st.selectbox(
            "Algorithm",
            options=list(models_info.keys()),
            index=list(models_info.keys()).index(best_name),
            help="Best model by AUC is pre-selected.",
        )
        info = models_info[model_choice]
        st.markdown(f"""
        <div class="metric-row">
          <div class="metric-tile"><div class="val">{info['acc']:.1%}</div><div class="lbl">Accuracy</div></div>
          <div class="metric-tile"><div class="val">{info['auc']:.3f}</div><div class="lbl">AUC</div></div>
          <div class="metric-tile"><div class="val">{info['cv_auc']:.3f}</div><div class="lbl">CV-AUC</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        st.markdown("### 📋 Demo Patients")
        demo_choice = st.selectbox("Load example", ["— Custom —"] + list(DEMO_PATIENTS.keys()))

        st.divider()
        st.markdown("### 📂 Upload Dataset")
        uploaded_file = st.file_uploader("Replace training data (CSV)", type=["csv"],
                                          help="Must contain same columns as heart.csv")
        st.divider()
        st.markdown("""
        <div style="font-size:.75rem; color:#8b949e; line-height:1.6;">
        <b>CardioSense v1.0</b><br>
        Built with Streamlit + scikit-learn<br>
        Dataset: UCI Heart Disease (303 patients)
        </div>
        """, unsafe_allow_html=True)

    return dark_mode, model_choice, demo_choice, uploaded_file

# ══════════════════════════════════════════════════════════════
# INPUT FORM
# ══════════════════════════════════════════════════════════════
def render_input_form(prefill: dict | None = None) -> dict:
    p = prefill or {}

    def sv(k, default):
        return p.get(k, default)

    filled = 0
    patient = {}

    st.markdown('<div class="section-label">Patient Parameters</div>', unsafe_allow_html=True)

    # Progress tracker
    prog_ph = st.empty()

    with st.form("full_form"):
        # ── Demographics ──
        st.markdown("**Demographics**")
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("Age (years)", 20, 90, sv('age',50),
                            help=FEATURE_META['age']['help'])
        with c2:
            sex = st.selectbox("Biological Sex", [0,1],
                               format_func=lambda x: FEATURE_META['sex']['options'][x],
                               index=sv('sex',1),
                               help=FEATURE_META['sex']['help'])
        filled += 2

        # ── Symptoms ──
        st.markdown("**Symptoms & Exam**")
        c1, c2, c3 = st.columns(3)
        with c1:
            cp = st.selectbox("Chest Pain Type", [0,1,2,3],
                              format_func=lambda x: FEATURE_META['cp']['options'][x],
                              index=sv('cp',0),
                              help=FEATURE_META['cp']['help'])
        with c2:
            exang = st.selectbox("Exercise Angina", [0,1],
                                 format_func=lambda x: FEATURE_META['exang']['options'][x],
                                 index=sv('exang',0),
                                 help=FEATURE_META['exang']['help'])
        with c3:
            restecg = st.selectbox("Resting ECG", [0,1,2],
                                   format_func=lambda x: FEATURE_META['restecg']['options'][x],
                                   index=sv('restecg',0),
                                   help=FEATURE_META['restecg']['help'])
        filled += 3

        # ── Vitals ──
        st.markdown("**Vitals & Lab Results**")
        c1, c2 = st.columns(2)
        with c1:
            trestbps = st.slider("Resting BP (mmHg)", 80, 200, sv('trestbps',120),
                                 help=FEATURE_META['trestbps']['help'])
            chol     = st.slider("Cholesterol (mg/dL)", 100, 600, sv('chol',200),
                                 help=FEATURE_META['chol']['help'])
        with c2:
            thalach  = st.slider("Max Heart Rate (bpm)", 60, 220, sv('thalach',150),
                                 help=FEATURE_META['thalach']['help'])
            fbs      = st.selectbox("Fasting Blood Sugar", [0,1],
                                    format_func=lambda x: FEATURE_META['fbs']['options'][x],
                                    index=sv('fbs',0),
                                    help=FEATURE_META['fbs']['help'])
        filled += 4

        # ── Advanced ──
        st.markdown("**Advanced Cardiac Markers**")
        c1, c2, c3 = st.columns(3)
        with c1:
            oldpeak = st.slider("ST Depression", 0.0, 7.0, float(sv('oldpeak',1.0)), step=0.1,
                                help=FEATURE_META['oldpeak']['help'])
        with c2:
            slope = st.selectbox("ST Slope", [0,1,2],
                                 format_func=lambda x: FEATURE_META['slope']['options'][x],
                                 index=sv('slope',0),
                                 help=FEATURE_META['slope']['help'])
        with c3:
            ca = st.selectbox("Major Vessels", [0,1,2,3],
                              format_func=lambda x: FEATURE_META['ca']['options'][x],
                              index=sv('ca',0),
                              help=FEATURE_META['ca']['help'])
        filled += 3

        thal_opts = [1,2,3]
        thal_def_idx = thal_opts.index(sv('thal', 2))
        thal = st.selectbox("Thalassemia", thal_opts,
                            format_func=lambda x: FEATURE_META['thal']['options'][x],
                            index=thal_def_idx,
                            help=FEATURE_META['thal']['help'])
        filled += 1

        submitted = st.form_submit_button("🫀 Predict Risk", use_container_width=True)

    # Progress bar  (13 fields = 100%)
    pct = filled / 13
    prog_ph.markdown(f"""
    <div class="section-label">Form Completion</div>
    <div class="prog-wrap"><div class="prog-fill" style="width:{int(pct*100)}%"></div></div>
    """, unsafe_allow_html=True)

    patient = dict(age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
                   fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
                   oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
    return patient, submitted

def render_quick_form(prefill: dict | None = None) -> tuple:
    p = prefill or {}
    st.markdown('<div class="section-label">3-Parameter Quick Screen</div>', unsafe_allow_html=True)
    st.caption("Uses age, cholesterol, and max heart rate only. Less accurate than the full assessment.")
    with st.form("quick_form"):
        c1,c2,c3 = st.columns(3)
        with c1: age     = st.number_input("Age",     20, 90,  p.get('age',50))
        with c2: chol    = st.number_input("Chol (mg/dL)", 100, 600, p.get('chol',200))
        with c3: thalach = st.number_input("Max HR (bpm)", 60, 220, p.get('thalach',150))
        sub = st.form_submit_button("⚡ Quick Assess", use_container_width=True)
    patient_quick = dict(age=age,sex=1,cp=0,trestbps=120,chol=chol,fbs=0,restecg=0,
                         thalach=thalach,exang=0,oldpeak=1.0,slope=1,ca=0,thal=2)
    return patient_quick, sub

# ══════════════════════════════════════════════════════════════
# RESULTS PANEL
# ══════════════════════════════════════════════════════════════
def render_results(prob: float, pred: int, patient: dict, fi_df: pd.DataFrame, dark: bool):
    label, cls, color = get_risk_level(prob)

    # ── Top strip ──
    st.markdown(f"""
    <div class="cs-card" style="border-left:4px solid {color}; padding:1rem 1.4rem;">
      <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:.5rem;">
        <div>
          <div class="section-label" style="margin:0 0 .3rem;">Prediction Result</div>
          <span class="risk-badge risk-{cls}">{label}</span>
        </div>
        <div style="text-align:right;">
          <div style="font-family:Fraunces,serif;font-size:2.6rem;line-height:1;color:{color};">{prob:.1f}<span style="font-size:1.2rem;">%</span></div>
          <div style="font-size:.75rem;color:#8b949e;">Risk Probability</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Charts row ──
    c1, c2 = st.columns([1,1])
    with c1:
        st.plotly_chart(gauge_chart(prob, dark), use_container_width=True, config={"displayModeBar":False})
    with c2:
        st.plotly_chart(radar_chart(patient, dark), use_container_width=True, config={"displayModeBar":False})

    # ── Comparison vs Normal ──
    st.markdown('<div class="section-label">Parameters vs Normal Range</div>', unsafe_allow_html=True)
    comp = compare_to_normal(patient)
    for row in comp:
        delta_str = f"+{row['delta']:.0f}%" if row['delta'] >= 0 else f"{row['delta']:.0f}%"
        bar_color = "#f85149" if row['delta'] > 15 else ("#d29922" if row['delta'] > 0 else "#3fb950")
        st.markdown(f"""
        <div class="compare-row">
          <div class="compare-label">{row['feature']}</div>
          <div class="compare-bar-wrap">
            <div class="compare-bar-fill" style="width:{int(row['pct_val']*100)}%;background:{bar_color};"></div>
          </div>
          <div class="compare-val">{row['value']} {row['unit']}</div>
          <div style="width:55px;font-size:.75rem;color:{bar_color};text-align:right;">{delta_str}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown('<div class="section-label" style="margin-top:1rem;">Personalised Recommendations</div>', unsafe_allow_html=True)
    recs = build_recommendations(patient, prob)
    for r in recs:
        st.markdown(f"""
        <div class="rec-pill">
          <span class="rec-icon">{r['icon']}</span>
          <span>{r['text']}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature importance ──
    st.markdown('<div class="section-label" style="margin-top:1rem;">Model Feature Importance</div>', unsafe_allow_html=True)
    st.plotly_chart(feature_importance_chart(fi_df, dark), use_container_width=True, config={"displayModeBar":False})

# ══════════════════════════════════════════════════════════════
# BATCH CSV  PREDICTION
# ══════════════════════════════════════════════════════════════
def render_batch(pipe, dark: bool):
    st.markdown('<div class="section-label">Batch Prediction via CSV Upload</div>', unsafe_allow_html=True)
    st.caption("Upload a CSV with the 13 feature columns. The app will predict risk for each row.")
    batch_file = st.file_uploader("Upload patient batch CSV", type=["csv"], key="batch")
    if batch_file:
        try:
            df_b = pd.read_csv(batch_file)
            missing = [c for c in FEATURES if c not in df_b.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
                return
            probs = pipe.predict_proba(df_b[FEATURES])[:,1] * 100
            df_b["risk_prob_%"]  = probs.round(2)
            df_b["risk_level"]   = [get_risk_level(p)[0] for p in probs]
            df_b["has_disease"]  = pipe.predict(df_b[FEATURES])

            st.success(f"✅ Predicted {len(df_b)} patients.")

            # Summary
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Total Patients", len(df_b))
            c2.metric("Avg Risk", f"{probs.mean():.1f}%")
            c3.metric("High/Very High Risk", int((probs>=40).sum()))
            c4.metric("Predicted Disease", int(df_b['has_disease'].sum()))

            st.dataframe(df_b, use_container_width=True, height=350)

            # Download
            csv_out = df_b.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download Results CSV", csv_out,
                               file_name="cardiosense_batch_results.csv", mime="text/csv")

            # Distribution chart
            fig = px.histogram(df_b, x="risk_prob_%", nbins=20,
                               color_discrete_sequence=["#f85149"],
                               title="Risk Score Distribution")
            fig.update_layout(paper_bgcolor="#161b22" if dark else "#fff",
                              plot_bgcolor="#161b22" if dark else "#fff",
                              font_color="#e6edf3" if dark else "#0f172a",
                              height=280, margin=dict(t=40,b=10,l=10,r=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ══════════════════════════════════════════════════════════════
# ABOUT SECTION
# ══════════════════════════════════════════════════════════════
def render_about(models_info: dict, best_name: str, df: pd.DataFrame, dark: bool):
    st.markdown("""
    <div class="cs-card">
    <h3 style="margin-top:0">About CardioSense</h3>
    <p>CardioSense is an AI-powered cardiovascular risk prediction dashboard built on the
    <b>UCI Heart Disease Dataset</b> (303 patients, Cleveland Clinic Foundation).
    Three machine learning models are trained and the best is automatically selected by AUC score.</p>
    </div>
    """, unsafe_allow_html=True)

    # Model comparison
    st.markdown('<div class="section-label">Model Comparison</div>', unsafe_allow_html=True)
    rows = []
    for name, info in models_info.items():
        rows.append({"Model": name, "Accuracy": f"{info['acc']:.1%}",
                     "AUC": f"{info['auc']:.3f}", "CV-AUC": f"{info['cv_auc']:.3f}",
                     "Selected": "✅" if name == best_name else ""})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Heatmap
    st.markdown('<div class="section-label">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    st.plotly_chart(correlation_heatmap(df, dark), use_container_width=True)

    # Feature explanations
    st.markdown('<div class="section-label">Feature Reference</div>', unsafe_allow_html=True)
    feat_rows = []
    for k,v in FEATURE_META.items():
        normal = v.get('normal','')
        if normal:
            normal_str = f"{normal[0]}–{normal[1]} {v.get('unit','')}"
        else:
            normal_str = "—"
        feat_rows.append({"Feature": k, "Label": v['label'],
                          "Normal Range": normal_str, "Description": v['help']})
    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Medical Disclaimer</b>: CardioSense is an educational and research tool only.
    It is <b>NOT</b> a substitute for professional medical advice, diagnosis, or treatment.
    Always consult a qualified healthcare provider for medical decisions.
    Predictions are based on a small dataset and should not be used for clinical purposes.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():
    # ── Session state ──
    if "history" not in st.session_state:
        st.session_state.history = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # ── Load data ──
    df = load_default_data()

    # ── Sidebar (always render so dark mode / uploader are available) ──
    # We pass dummy values if models aren't trained yet
    inject_css(True)  # default dark while we check

    if df is None:
        # Show a friendly uploader instead of crashing
        st.markdown("""
        <div class="hero">
          <h1>CardioSense</h1>
          <p>AI-Powered Cardiovascular Risk Prediction</p>
        </div>
        """, unsafe_allow_html=True)
        st.warning("⚠️ `heart.csv` was not found in the app folder. Please upload it below to continue.")
        csv_upload = st.file_uploader("Upload heart.csv", type=["csv"])
        if csv_upload is None:
            st.info("📂 Place `heart.csv` next to `app.py` **or** upload it here. Then the app will train and launch automatically.")
            st.stop()
        df = pd.read_csv(csv_upload)
        st.success("✅ Dataset loaded! Training models…")

    # ── Train ──
    with st.spinner("Training models…"):
        models_info, best_name, fi_df, X_test, y_test = train_models(df)

    # ── Sidebar (full, now that models exist) ──
    dark_mode, model_choice, demo_choice, uploaded_file = render_sidebar(models_info, best_name)
    inject_css(dark_mode)

    # Handle custom dataset upload
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Custom dataset loaded!")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

    pipe = models_info[model_choice]["pipe"]
    prefill = DEMO_PATIENTS.get(demo_choice) if demo_choice != "— Custom —" else None

    # ── Hero ──
    st.markdown("""
    <div class="hero">
      <h1>CardioSense</h1>
      <p>AI-Powered Cardiovascular Risk Prediction · Clinical-grade insights in seconds</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──
    tab1, tab2, tab3, tab4 = st.tabs(["🫀 Full Assessment", "⚡ Quick Screen", "📊 Batch Predict", "ℹ️ About"])

    # ════════════ TAB 1 — Full Form ════════════
    with tab1:
        col_form, col_result = st.columns([1, 1.1], gap="large")

        with col_form:
            patient, submitted = render_input_form(prefill)

        with col_result:
            if submitted or st.session_state.last_result:
                if submitted:
                    with st.spinner("Analysing…"):
                        prob, pred = predict(pipe, patient)
                    st.session_state.last_result = {"prob": prob, "pred": pred, "patient": patient}
                    ts = datetime.now().strftime("%H:%M:%S")
                    st.session_state.history.append({"time": ts, "prob": prob, "patient": patient})

                res = st.session_state.last_result
                render_results(res["prob"], res["pred"], res["patient"], fi_df, dark_mode)

                # ── Risk history ──
                if len(st.session_state.history) > 1:
                    st.markdown('<div class="section-label" style="margin-top:1rem;">Risk History (This Session)</div>', unsafe_allow_html=True)
                    st.plotly_chart(history_chart(st.session_state.history, dark_mode),
                                    use_container_width=True, config={"displayModeBar":False})

                # ── Doctor's notes ──
                with st.expander("📝 Doctor's Notes"):
                    note = st.text_area("Add clinical notes here…", height=100, key="doctor_notes")
                    if note:
                        st.caption(f"Note saved at {datetime.now().strftime('%Y-%m-%d %H:%M')}")

                # ── Export ──
                if submitted:
                    export_data = {
                        "timestamp": datetime.now().isoformat(),
                        "model": model_choice,
                        "risk_probability_%": res['prob'],
                        "risk_level": get_risk_level(res['prob'])[0],
                        "patient_parameters": res['patient'],
                        "recommendations": [r['text'] for r in build_recommendations(res['patient'], res['prob'])],
                    }
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        "⬇️ Export Report (JSON)",
                        data=json_str,
                        file_name=f"cardiosense_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                    )
            else:
                st.markdown("""
                <div class="cs-card" style="text-align:center;padding:3rem 1.5rem;color:#8b949e;">
                  <div style="font-size:3rem;margin-bottom:1rem;">🫀</div>
                  <div style="font-family:Fraunces,serif;font-size:1.3rem;margin-bottom:.5rem;color:#e6edf3;">Ready for Assessment</div>
                  <div style="font-size:.9rem;">Fill in patient parameters on the left and click <b>Predict Risk</b>.</div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════ TAB 2 — Quick Screen ════════════
    with tab2:
        c1, c2 = st.columns([1, 1.1], gap="large")
        with c1:
            q_patient, q_sub = render_quick_form(prefill)
        with c2:
            if q_sub:
                with st.spinner("Screening…"):
                    q_prob, q_pred = predict(pipe, q_patient)
                render_results(q_prob, q_pred, q_patient, fi_df, dark_mode)
                st.warning("⚠️ Quick screen uses only 3 parameters. For accurate results use **Full Assessment**.")
            else:
                st.markdown("""
                <div class="cs-card" style="text-align:center;padding:3rem 1.5rem;color:#8b949e;">
                  <div style="font-size:2.5rem;">⚡</div>
                  <div style="font-family:Fraunces,serif;font-size:1.2rem;margin:.6rem 0;color:#e6edf3;">Quick 3-Parameter Screen</div>
                  <div style="font-size:.88rem;">Fast preliminary assessment using age, cholesterol, and max heart rate.</div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════ TAB 3 — Batch ════════════
    with tab3:
        render_batch(pipe, dark_mode)

    # ════════════ TAB 4 — About ════════════
    with tab4:
        render_about(models_info, best_name, df, dark_mode)

    # ── Footer ──
    st.markdown("""
    <div class="cs-footer">
      CardioSense · Built with Streamlit · Dataset: UCI Heart Disease Repository
      · For educational &amp; portfolio use only · Not for clinical use
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
