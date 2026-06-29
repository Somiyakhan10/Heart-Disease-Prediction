# 🫀 CardioSense — Cardiovascular Risk Prediction Dashboard

> **AI-powered heart disease risk prediction · Production-ready Streamlit app · Portfolio-grade UI**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

⚠️ **Medical Disclaimer**: This project is for educational and research purposes only. It is **not** a substitute for professional medical advice, diagnosis, or treatment.

---

## Overview

CardioSense demonstrates a complete machine learning engineering pipeline — from raw data exploration and model training through to a deployed, user-facing product. The system accepts **13 clinical measurements** and returns a real-time **Low / Moderate / High / Very High** cardiovascular risk prediction with a confidence score, personalized health recommendations, and interactive visualizations.

The project was built to go beyond typical academic ML work: training a model is only the beginning. CardioSense wraps three trained classifiers inside a polished, production-grade **Streamlit** dashboard with custom CSS theming, animated Plotly charts, batch CSV inference, dark/light mode, and JSON report export.

---

## Model Performance

| Metric | Value |
|--------|-------|
| **Best Algorithm** | Gradient Boosting Classifier |
| **Dataset** | UCI Heart Disease Dataset (Cleveland Clinic) |
| **Total Samples** | 303 patient records |
| **Train / Test Split** | 80% / 20% with stratification |
| **Feature Scaling** | StandardScaler (z-score normalisation) |
| **Test Accuracy** | ~82% |
| **ROC-AUC Score** | ~0.90 |
| **Cross-Validation Mean** | ~0.89 ± 0.03 (5-fold) |
| **Class Distribution** | 165 Disease / 138 No Disease |

Three models are trained at startup and **the best by AUC is auto-selected**:

| Model | Typical Accuracy | Typical AUC |
|-------|-----------------|-------------|
| Gradient Boosting | ~82% | ~0.90 |
| Random Forest | ~80% | ~0.88 |
| Logistic Regression | ~78% | ~0.85 |

---

## UI Overview

```
┌─────────────────────────────────────────────────────────────┐
│  ⚙️ Sidebar          │  🫀 CardioSense Hero Banner           │
│  ─────────────────   │  ─────────────────────────────────── │
│  Dark/Light Toggle   │  [ Full Assessment ] [ Quick Screen ] │
│  Model Selector      │  [ Batch Predict  ] [ About        ] │
│  Model Metrics       │                                       │
│  Demo Patients       │  ┌──────────────┐ ┌───────────────┐  │
│  Upload Dataset      │  │ Input Form   │ │ Risk Gauge    │  │
│                      │  │ (13 sliders/ │ │ Radar Chart   │  │
│  v1.0 · scikit-learn │  │  dropdowns)  │ │ Comparisons   │  │
│                      │  │              │ │ Recs / Export │  │
│                      │  └──────────────┘ └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## How It Works

The system follows a straightforward inference pipeline. A user fills in 13 clinical parameters through the Streamlit form, which passes the feature vector through the saved **StandardScaler**, runs inference on the selected classifier, and returns the risk score with a full visual breakdown in real time.

```
User Input (13 features)
        ↓
StandardScaler transform
        ↓
Gradient Boosting / Random Forest / Logistic Regression
        ↓
{ risk_probability, risk_level, recommendations, charts }
```

---

## Features

### 4 Prediction Modes
| Mode | Parameters Used | Best For |
|------|----------------|----------|
| Full Assessment | All 13 features | Accurate screening |
| Quick Screen | Age, Cholesterol, Max HR | Fast preliminary check |
| Batch Predict | CSV upload | Multiple patients at once |
| Demo Patients | 3 preset cases | Testing & demonstrations |

### Visualizations
- **Risk Gauge** — Animated circular indicator (0–100%)
- **Radar Chart** — Patient profile vs normal range comparison
- **Feature Importance** — Which factors drive the prediction most
- **Correlation Heatmap** — Full 14×14 feature relationship matrix
- **Risk History** — Session-level trend line after multiple predictions

### Risk Levels
| Score | Level | Indicator |
|-------|-------|-----------|
| 0–19% | LOW RISK | 🟢 Green |
| 20–39% | MODERATE RISK | 🟡 Yellow |
| 40–69% | HIGH RISK | 🟠 Orange |
| 70–100% | VERY HIGH RISK | 🔴 Red |

### Additional Features
- 🌙 Dark / Light mode toggle
- 📋 Personalized health recommendations per patient profile
- 📊 Parameter vs normal range comparison bars with ±% deltas
- 📝 Doctor's Notes section
- ⬇️ JSON report export
- 📂 Custom dataset upload (replace training data on the fly)

---

## Feature Engineering

The 13 clinical input features used by the model:

| Feature | Description | Normal Range |
|---------|-------------|--------------|
| `age` | Patient age in years | 30–65 |
| `sex` | Biological sex (0 = Female, 1 = Male) | — |
| `cp` | Chest pain type (0–3) | — |
| `trestbps` | Resting blood pressure | < 120 mmHg |
| `chol` | Serum cholesterol | < 200 mg/dL |
| `fbs` | Fasting blood sugar > 120 mg/dL | 0 (Normal) |
| `restecg` | Resting ECG results (0–2) | 0 (Normal) |
| `thalach` | Maximum heart rate achieved | 60–150 bpm |
| `exang` | Exercise-induced angina | 0 (No) |
| `oldpeak` | ST depression induced by exercise | 0–1.5 mm |
| `slope` | Slope of peak exercise ST segment | — |
| `ca` | Major vessels coloured by fluoroscopy (0–3) | 0 |
| `thal` | Thalassemia type | 2 (Normal) |

---

## Tech Stack

- **Machine Learning**: Python, scikit-learn, pandas, NumPy, Gradient Boosting, Random Forest, Logistic Regression, StandardScaler
- **Frontend & Dashboard**: Streamlit 1.32+, Custom CSS, Plotly
- **Visualizations**: Plotly Graph Objects (gauge, radar, heatmap, bar, line)
- **Infrastructure**: Streamlit Community Cloud / Render / Heroku / Docker

---

## Project Structure

```
cardiosense/
├── app.py               ← Main Streamlit application (all-in-one, 1100+ lines)
├── heart.csv            ← UCI Heart Disease dataset (303 patients)
├── requirements.txt     ← Python dependencies
├── .streamlit/
│   └── config.toml      ← Dark theme & server configuration
└── README.md
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.10+
- pip

### 1. Clone / download
```bash
git clone https://github.com/somiyakhan/cardiosense.git
cd cardiosense
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

> ℹ️ `heart.csv` must be in the same directory as `app.py`. If it's missing, the app will prompt you to upload it directly in the browser.

---

## ☁️ Deployment Guide

### Option A — Streamlit Community Cloud (Free, Recommended)

1. Push your project to a **public GitHub repo** (include `heart.csv`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Connect your GitHub repo and set:
   - **Branch**: `main`
   - **Main file**: `app.py`
4. Click **Deploy** — live in ~2 minutes.

---

### Option B — Render (Free Tier)

1. Push to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service**.
3. Connect your repo and set:
   - **Build command**: `pip install -r requirements.txt`
   - **Start command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Click **Create Web Service**.

---

### Option C — Heroku

Create a `Procfile`:
```
web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Then:
```bash
heroku create your-cardiosense-app
git push heroku main
heroku open
```

---

### Option D — Docker

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

```bash
docker build -t cardiosense .
docker run -p 8501:8501 cardiosense
```

---

## 🎨 Customization

### Change Color Theme
Edit `inject_css()` in `app.py`:
```python
accent  = "#your_color"   # Primary accent (default: red #f85149)
accent2 = "#your_color"   # Secondary accent (default: blue #388bfd)
```

### Add a New Feature
1. Add a new key to the `FEATURE_META` dict.
2. Add the corresponding input widget in `render_input_form()`.
3. `train_models()` picks up new columns automatically on restart.

### Change Model Hyperparameters
Find the `configs` dict in `train_models()`:
```python
"Gradient Boosting": GradientBoostingClassifier(
    n_estimators=300,    # More trees = higher accuracy
    learning_rate=0.05,  # Lower = more conservative
    max_depth=5,
)
```

### Add a Recommendation Rule
In `build_recommendations()`:
```python
if patient['your_feature'] > threshold:
    recs.append({"icon": "🔔", "text": "Your custom recommendation."})
```

---

## Dataset

**Source**: [UCI Machine Learning Repository — Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

- **Patients**: 303 (Cleveland Clinic Foundation)
- **Features**: 13 clinical attributes
- **Target**: Binary — 0 = No Disease, 1 = Disease
- **Class balance**: ~54% Disease, ~46% No Disease
- **Missing values**: None

The dataset was collected at the Cleveland Clinic Foundation and has been a benchmark classification dataset since the early 1990s.

---

## Future Work

Several natural extensions exist for researchers and students who want to build on this project. Integrating **SHAP values** for per-prediction explainability would make the model's reasoning transparent to clinicians. Adding a **time-series risk tracker** that stores patient history across sessions would enable longitudinal monitoring. Replacing static thresholds in the recommendation engine with a **rule-based clinical decision support system** aligned to ACC/AHA guidelines would bring the tool closer to real clinical utility. Training on the full **multi-site UCI Heart Disease dataset** (Cleveland + Hungarian + Switzerland + VA Long Beach) with 920 combined records would likely improve model generalization.

---

## Portfolio Notes

This project demonstrates:
- ✅ End-to-end ML pipeline (data → train → evaluate → deploy)
- ✅ Production-grade Streamlit UI with fully custom CSS theming
- ✅ Multiple model training, comparison, and auto-selection by AUC
- ✅ Interactive Plotly visualizations (gauge, radar, heatmap, bar, line)
- ✅ Batch inference via CSV upload with downloadable results
- ✅ Session state management and risk history tracking
- ✅ Dark / light mode toggle
- ✅ JSON report export per prediction
- ✅ Modular, well-commented, production-ready code (1100+ lines)

---

## ⚠️ Medical Disclaimer

CardioSense is an **educational and portfolio tool only**.

- It is **NOT** a substitute for professional medical advice
- Predictions are based on a small (303-patient) dataset
- **Do not use for clinical decision-making**
- Always consult a qualified healthcare provider for medical decisions

---

## Author

**Somiya Khan**

[![GitHub](https://img.shields.io/badge/GitHub-somiyakhan-181717?style=flat-square&logo=github)](https://github.com/somiyakhan)
[![Kaggle](https://img.shields.io/badge/Kaggle-somiyakhan-20BEFF?style=flat-square&logo=kaggle)](https://www.kaggle.com/somiyakhan)

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

*CardioSense v1.0 · Built with ❤️ for learning and research · Dataset: UCI Heart Disease Repository*

*If this project helped you, consider giving it a ⭐ on GitHub*
