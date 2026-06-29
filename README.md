#  CardioSense — AI-Powered Cardiovascular Risk Prediction System

**An Interpretable Machine Learning System for Heart Disease Risk Assessment with Multi-Model Comparison and Clinical Decision Support**

---


<div align="center"> <a href="https://huggingface.co/spaces/somiya-khan01/heart_disease_risk_prediction" target="_blank"> <button style=" background: linear-gradient(135deg, #FFD21E, #FFA500); border: none; color: #000; font-size: 24px; font-weight: bold; padding: 16px 48px; border-radius: 50px; cursor: pointer; box-shadow: 0 8px 25px rgba(255, 210, 30, 0.4); transition: all 0.3s ease; text-decoration: none; "> 🎯 Live Demo </button> </a> <br><br> <p><em>Click the button above to test the model with your own inputs!</em></p> </div>


</div>

---

>  **Medical Disclaimer:** CardioSense is developed strictly for research and educational purposes. It is NOT a certified medical device and must not be used as the sole basis for any clinical decision. All predictions should be reviewed and validated by a qualified healthcare professional.


##  Overview

**CardioSense** is a full-stack machine learning application designed to assist researchers and clinicians in assessing cardiovascular disease risk. The system integrates multiple ML algorithms — including **Random Forest**, **Gradient Boosting**, and **Logistic Regression** — with an intuitive clinical interface styled for healthcare professionals.

### Key Features:

| Feature | Description |
|---------|-------------|
| **Multi-Model Comparison** | Compare Random Forest, Gradient Boosting, and Logistic Regression side-by-side |
| **Clinical Input Interface** | 14 patient parameters organised by clinical categories |
| **Real-Time Risk Assessment** | Instant predictions with confidence scores |
| **Batch Prediction** | Upload CSV files for bulk patient risk assessment |
| **Interpretable Outputs** | Probability scores, risk categories, and feature importance |
| **Responsive Design** | Works on desktops, tablets, and mobile devices |

---

##  Screenshots

### Home Dashboard
*Model comparison cards, quick stats, and clinical workflow overview*

<img width="1887" height="886" alt="image" src="https://github.com/user-attachments/assets/82d21777-f973-4497-a78f-9b16546bf2d2" />

### Batch Prediction
*CSV upload interface for bulk patient risk assessment*

<img width="1894" height="891" alt="image" src="https://github.com/user-attachments/assets/30e6fe5e-e438-41c7-ba56-960ae3d087f3" />


### About Page
*Methodology, dataset information, and model documentation*

<img width="1884" height="802" alt="image" src="https://github.com/user-attachments/assets/877612e6-b472-4157-8163-82534cf124d0" />


---

##  System Architecture

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

##  Dataset

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

##  Models & Performance

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

##  Feature Engineering

### Input Features (14 Parameters)

| Category | Features |
|----------|----------|
| **Demographic** | Age, Sex |
| **Vitals** | Resting BP (trestbps), Max Heart Rate (thalach) |
| **Lab Results** | Cholesterol (chol), Fasting Blood Sugar (fbs) |
| **Cardiac Markers** | Chest Pain Type (cp), Exercise Angina (exang), ST Depression (oldpeak), ST Slope (slope) |
| **Clinical Findings** | Resting ECG (restecg), Major Vessels (ca), Thalassemia (thal) |



---

##  Technology Stack

| Layer | Technology | Version | Purpose |
|-------|------------|---------|---------|
| **Model** | scikit-learn | 1.6.1 | ML algorithms (RF, GB, LR) |
| **Model** | pandas | 2.2.2 | Data manipulation |
| **Model** | numpy | 2.0.2 | Numerical operations |
| **Backend** | Flask | 3.1.0 | REST API server |
| **Backend** | Flask-CORS | 4.0.1 | Cross-origin resource sharing |
| **Backend** | Gunicorn | 22.0.0 | Production WSGI server |
| **Frontend** | React | 19.0 | UI framework |
| **Frontend** | Tailwind CSS | 4.0 | Utility-first styling |
| **Frontend** | Recharts | latest | Charts & visualizations |
| **Frontend** | Axios | latest | HTTP client |
| **Deployment** | Docker | latest | Containerization |
| **Deployment** | Hugging Face Spaces | — | Backend hosting |
| **Deployment** | Vercel | — | Frontend hosting |
| **Training** | Jupyter | latest | EDA & model development |

---

##  Roadmap

- [x] **Phase 1** — Data collection, EDA, feature engineering
- [x] **Phase 2** — Model training (Random Forest, Gradient Boosting, Logistic Regression)
- [x] **Phase 3** — Cross-validation and hyperparameter tuning
- [x] **Phase 4** — Flask REST API development
- [x] **Phase 5** — React dashboard with Tailwind CSS
- [x] **Phase 6** — Batch prediction for CSV uploads
- [x] **Phase 7** — Docker deployment on Hugging Face Spaces
- [ ] **Phase 8** — SHAP/LIME explainability integration
- [ ] **Phase 9** — XGBoost and Neural Network model addition
- [ ] **Phase 10** — Mobile app (React Native) development
- [ ] **Phase 11** — FHIR/HL7 integration for EHR systems

---

##  Contributing

Contributions from researchers, developers, and clinicians are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

For major changes, please open an Issue first to discuss.

---




---

##  Acknowledgements

- **Dataset:** UCI Heart Disease Dataset — [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Original Study:** Detrano, R., et al. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. *American Journal of Cardiology*.

---

##  License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

**This software is for research and educational purposes only. Not for clinical use without proper regulatory approval.**

---

<div align="center">

**Made with ❤️ by Somiya Khan**

 If this project helped your research, please consider giving it a star on GitHub! 

</div>
