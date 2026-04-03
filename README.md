# ❤️ CardioSense — AI-Powered Cardiovascular Risk Prediction System

An Interpretable Machine Learning System for Heart Disease Risk Assessment with Multi-Model Comparison and Clinical Decision Support

---

> ⚠️ **Medical Disclaimer:** CardioSense is developed strictly for research and educational purposes. It is NOT a certified medical device and must not be used as the sole basis for any clinical decision. All predictions should be reviewed and validated by a qualified healthcare professional.

---

---

## 🔬 Overview

**CardioSense** is a full-stack machine learning application designed to assist researchers and clinicians in assessing cardiovascular disease risk. The system integrates multiple ML algorithms — including **Random Forest**, **Gradient Boosting**, and **Logistic Regression** — with an intuitive clinical interface styled for healthcare professionals.

### Key Features:

- **Multi-Model Comparison** — Compare Random Forest, Gradient Boosting, and Logistic Regression side-by-side
- **Clinical Input Interface** — 14 patient parameters organised by clinical categories
- **Real-Time Risk Assessment** — Instant predictions with confidence scores
- **Batch Prediction** — Upload CSV files for bulk patient risk assessment
- **Interpretable Outputs** — Probability scores, risk categories, and feature importance
- **Responsive Design** — Works on desktops, tablets, and mobile devices


---

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│   React 19 + Tailwind CSS 4 + Recharts + Axios                  │
│   ┌──────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────┐ │
│   │  Home    │  │   Predict    │  │    Batch     │  │ About  │ │
│   │  Page    │  │    Page      │  │    Page      │  │ Page   │ │
│   └──────────┘  └──────────────┘  └──────────────┘  └────────┘ │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP / REST (Axios)
┌─────────────────────────▼───────────────────────────────────────┐
│                        API LAYER                                │
│   Flask 3.1 + Flask-CORS + Gunicorn                             │
│   ┌──────────────────┐   ┌─────────────────┐   ┌─────────────┐ │
│   │  GET /api/health │   │ POST /api/predict│   │POST /api/   │ │
│   │                  │   │                  │   │batch-predict│ │
│   └──────────────────┘   └─────────────────┘   └─────────────┘ │
│   ┌──────────────────┐   ┌─────────────────┐                    │
│   │ GET /api/models  │   │ GET /api/       │                    │
│   │                  │   │features         │                    │
│   └──────────────────┘   └─────────────────┘                    │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                      INFERENCE LAYER                            │
│   scikit-learn + pandas + numpy + joblib                        │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│   │   Random     │  │  Gradient    │  │  Logistic    │         │
│   │   Forest     │  │   Boosting   │  │  Regression  │         │
│   │   (83.6%)    │  │   (82.1%)    │  │   (81.4%)    │         │
│   └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              StandardScaler + Feature Pipeline           │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow for Single Prediction:

1. User enters patient parameters via React form
2. Axios POSTs JSON data to `/api/predict`
3. Flask validates input and scales features
4. All three models generate predictions
5. Results are returned as JSON with probabilities
6. React renders risk gauge, confidence bars, and recommendations

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Name** | UCI Heart Disease Dataset |
| **Original Source** | UCI Machine Learning Repository |
| **Total Samples** | 303 patient records |
| **Features** | 14 clinical parameters |
| **Target** | Presence of heart disease (0 = No, 1 = Yes) |
| **Class Distribution** | 138 No Disease / 165 Disease |

### Feature Descriptions:

| Feature | Description | Type |
|---------|-------------|------|
| **age** | Age in years | Continuous |
| **sex** | 1 = male, 0 = female | Binary |
| **cp** | Chest pain type (0-3) | Categorical |
| **trestbps** | Resting blood pressure (mm Hg) | Continuous |
| **chol** | Serum cholesterol (mg/dl) | Continuous |
| **fbs** | Fasting blood sugar > 120 mg/dl (1 = true) | Binary |
| **restecg** | Resting ECG results (0-2) | Categorical |
| **thalach** | Maximum heart rate achieved | Continuous |
| **exang** | Exercise induced angina (1 = yes) | Binary |
| **oldpeak** | ST depression induced by exercise | Continuous |
| **slope** | Slope of peak exercise ST segment | Categorical |
| **ca** | Number of major vessels (0-3) | Categorical |
| **thal** | Thalassemia (0-2) | Categorical |
| **target** | Diagnosis (0 = no disease, 1 = disease) | Target |

---

## 🤖 Models & Performance

### Model Comparison

| Model | Accuracy | ROC-AUC | Cross-Validation (5-fold) | Best Parameters |
|-------|----------|---------|---------------------------|-----------------|
| **Random Forest** | **83.6%** | **0.916** | 81.2% ± 3.4% | n_estimators=100, max_depth=10 |
| Gradient Boosting | 82.1% | 0.908 | 80.5% ± 2.8% | learning_rate=0.1, n_estimators=100 |
| Logistic Regression | 81.4% | 0.895 | 79.8% ± 3.1% | C=1.0, penalty=l2 |

### Feature Importance (Random Forest)

```
1. thalach (Max Heart Rate)     ████████████████░░░░  18.2%
2. cp (Chest Pain Type)         ██████████████░░░░░░  16.5%
3. ca (Major Vessels)           ████████████░░░░░░░░  13.8%
4. oldpeak (ST Depression)      ████████████░░░░░░░░  12.9%
5. age                          ████████░░░░░░░░░░░░   8.5%
6. thal                         ████████░░░░░░░░░░░░   8.1%
7. exang                        ██████░░░░░░░░░░░░░░   6.2%
8. chol                         ████░░░░░░░░░░░░░░░░   4.5%
9. trestbps                     ███░░░░░░░░░░░░░░░░░   3.2%
10. sex                         ██░░░░░░░░░░░░░░░░░░   2.1%
```

---

## 🔧 Feature Engineering

### Input Features (14 Parameters)

| Category | Features |
|----------|----------|
| **Demographic** | Age, Sex |
| **Vitals** | Resting BP (trestbps), Max Heart Rate (thalach) |
| **Lab Results** | Cholesterol (chol), Fasting Blood Sugar (fbs) |
| **Cardiac Markers** | Chest Pain Type (cp), Exercise Angina (exang), ST Depression (oldpeak), ST Slope (slope) |
| **Clinical Findings** | Resting ECG (restecg), Major Vessels (ca), Thalassemia (thal) |

### Preprocessing Pipeline

```python
# Feature scaling (StandardScaler)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Cross-validation (5-fold)
cv_scores = cross_val_score(model, X_scaled, y, cv=5)
```

---

## 📁 Project Structure

```
CardioSense/
│
├── 📓 notebooks/
│   └── Somiya_Khan_Heart_Disease_Analysis.ipynb   # Full EDA + model training
│
├── ⚙️ backend/
│   ├── app.py                           # Flask application
│   ├── requirements.txt                 # Python dependencies
│   ├── Dockerfile                       # Container definition
│   ├── models/
│   │   ├── random_forest.pkl
│   │   ├── gradient_boosting.pkl
│   │   ├── logistic_regression.pkl
│   │   └── scaler.pkl
│   ├── routes/
│   │   ├── predict.py                   # Single prediction endpoint
│   │   ├── batch_predict.py             # Batch CSV prediction
│   │   └── health.py                    # Health check endpoint
│   └── utils/
│       ├── preprocessing.py             # Feature scaling & validation
│       └── feature_definitions.py       # Feature names & descriptions
│
├── 🖥️ frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── HomePage.jsx             # Dashboard with model stats
│   │   │   ├── PredictPage.jsx          # Single patient prediction
│   │   │   ├── BatchPage.jsx            # CSV batch prediction
│   │   │   └── AboutPage.jsx            # Methodology & documentation
│   │   ├── components/
│   │   │   ├── RiskGauge.jsx            # Animated probability gauge
│   │   │   ├── FeatureInput.jsx         # Individual input with tooltip
│   │   │   ├── ModelCard.jsx            # Model performance card
│   │   │   ├── ResultPanel.jsx          # Prediction results display
│   │   │   └── FeatureImportance.jsx    # Bar chart visualization
│   │   ├── services/
│   │   │   └── api.js                   # Axios API client
│   │   └── index.css                    # Tailwind + custom styles
│   ├── package.json
│   └── vite.config.js
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

### Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.10 |
| Node.js | ≥ 18.0 |
| npm | ≥ 9.0 |
| Git | latest |

### 1. Clone the Repository

```bash
git clone https://github.com/somiyakhan01/CardioSense.git
cd CardioSense
```

### 2. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train models (if needed)
python train_models.py

# Start the Flask server
python app.py
# → Running on http://127.0.0.1:5000
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start the Vite dev server
npm run dev
# → Running on http://localhost:5173
```

### 4. Training Notebook (Optional)

Upload `Somiya_Khan_Heart_Disease_Analysis.ipynb` to:
- **Kaggle** (with GPU enabled)
- **Google Colab**
- **Local Jupyter**

Run all cells to regenerate model files.

---


## 🗺️ Roadmap

- [x] **Phase 1** — Data collection, EDA, feature engineering
- [x] **Phase 2** — Model training (Random Forest, Gradient Boosting, Logistic Regression)
- [x] **Phase 3** — Cross-validation and hyperparameter tuning
- [x] **Phase 4** — Flask REST API development
- [x] **Phase 5** — React dashboard with Tailwind CSS
- [x] **Phase 6** — Batch prediction for CSV uploads
- [x] **Phase 7** — Docker deployment on Render + Vercel
- [ ] **Phase 8** — SHAP/LIME explainability integration
- [ ] **Phase 9** — XGBoost and Neural Network model addition
- [ ] **Phase 10** — Mobile app (React Native) development
- [ ] **Phase 11** — FHIR/HL7 integration for EHR systems

---

## 🤝 Contributing

Contributions from researchers, developers, and clinicians are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

For major changes, please open an Issue first to discuss.

---

## 📖 Citation

If you use this codebase in your research, please cite as:

```bibtex
@software{khan2026cardiosense,
  author    = {Khan, Somiya},
  title     = {CardioSense: An AI-Powered Cardiovascular Risk Prediction System},
  year      = {2026},


}
```

---

##  Acknowledgements

- **Dataset:** UCI Heart Disease Dataset — [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Original Study:** Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

**This software is for research and educational purposes only. Not for clinical use without proper regulatory approval.**

---

<div align="center">

**Made with ❤️ by Somiya Khan**



⭐ If this project helped your research, please consider giving it a star on GitHub! ⭐

</div>
