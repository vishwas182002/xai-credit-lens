# 🔍 XAI Credit Lens

**An Explainable AI Framework for Fair Credit Decisioning**

> Production-grade ML pipeline combining credit default prediction, multi-method explainability (SHAP + LIME + Counterfactuals), automated fairness auditing, and regulatory compliance mapping — all in one interactive dashboard.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 Why This Exists

Credit decisions affect millions of lives, yet most ML models in lending operate as black boxes. Regulators (Fed SR 11-7, EU AI Act, ECOA) increasingly demand that institutions explain *why* a decision was made and prove it wasn't biased.

**XAI Credit Lens** bridges the gap between model performance and regulatory compliance by providing:

- **Transparent predictions** with global and local explanations
- **Actionable counterfactuals** — telling applicants exactly what to change to get approved
- **Automated fairness audits** across protected demographics
- **Compliance-ready reports** mapped to real regulatory frameworks

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard (Layer 5)           │
│  Applicant Input → Prediction → Explanation → Fairness  │
├─────────────┬──────────────┬──────────────┬─────────────┤
│  Regulatory │  Fairness    │ Explainability│  Prediction │
│  Mapping    │  Audit       │  Suite        │  Engine     │
│  (Layer 4)  │  (Layer 3)   │  (Layer 2)    │  (Layer 1)  │
├─────────────┴──────────────┴──────────────┴─────────────┤
│              Data Pipeline & Feature Engineering         │
├─────────────────────────────────────────────────────────┤
│              Taiwan Credit Dataset (UCI)                 │
└─────────────────────────────────────────────────────────┘
```

### Layer 1 — Prediction Engine
- XGBoost, LightGBM, Random Forest ensemble
- Optuna hyperparameter tuning with Stratified 5-Fold cross-validation
- Automated model comparison & best-model selection

### Layer 2 — Explainability Suite
- **SHAP**: Global feature importance + local waterfall plots
- **LIME**: Instance-level interpretable explanations
- **Counterfactual Explanations**: DiCE-based actionable recourse
  - *"If your debt-to-income ratio dropped by 8%, you'd be approved"*
  - This is the differentiator — real actionable feedback for applicants

### Layer 3 — Fairness Audit
- Disparate impact ratio across demographic groups
- Statistical parity, equalized odds, predictive parity
- Automated bias detection with configurable thresholds
- Audit-ready visualizations and summary statistics

### Layer 4 — Regulatory Mapping
- **EU AI Act**: Transparency & documentation requirements for high-risk AI
- **US ECOA / Reg B**: Adverse action notice generation
- **Fed SR 11-7**: Model risk management documentation
- Automated compliance checklist generation

### Layer 5 — Interactive Dashboard
- Single-screen applicant decisioning view
- Real-time prediction + explanation + fairness flag
- Counterfactual scenario explorer
- Exportable audit reports

---

## 📊 Dataset

**[Default of Credit Card Clients — UCI ML Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)**

- 30,000 credit card clients from a Taiwan bank (2005)
- 23 features: demographics, credit limit, payment history, bill amounts
- Binary target: default on payment (next month)
- Well-studied benchmark with known demographic attributes for fairness analysis

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/vishwas182002/xai-credit-lens.git
cd xai-credit-lens

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Download and prepare data
python -m src.data.download_data
python -m src.data.preprocess

# Train models (includes Optuna tuning — takes ~10 min)
python -m src.models.train

# Generate explanations for a sample applicant
python -m src.explainability.run_explanations

# Run fairness audit
python -m src.fairness.audit

# Run regulatory compliance checks
python -m src.regulatory.compliance_engine

# Launch dashboard
streamlit run src/dashboard/app.py
```

---

## 📁 Project Structure

```
xai-credit-lens/
├── configs/
│   ├── model_config.yaml          # Model hyperparameters & tuning config
│   ├── fairness_config.yaml       # Fairness thresholds & protected attributes
│   └── regulatory_config.yaml     # Compliance mapping rules
├── data/
│   ├── raw/                       # Original downloaded dataset
│   ├── processed/                 # Cleaned & feature-engineered data
│   └── splits/                    # Train/val/test splits
├── notebooks/
│   └── 01_eda.py                  # Exploratory data analysis
├── reports/
│   ├── figures/                   # Generated plots and charts
│   └── audit/                     # Compliance audit outputs (JSON)
├── src/
│   ├── data/
│   │   ├── download_data.py       # Dataset acquisition from UCI
│   │   └── preprocess.py          # Cleaning, feature engineering, splitting
│   ├── models/
│   │   └── train.py               # Optuna tuning + training + evaluation
│   ├── explainability/
│   │   ├── shap_explainer.py      # SHAP global + local explanations
│   │   ├── lime_explainer.py      # LIME instance explanations
│   │   ├── counterfactual.py      # DiCE counterfactual generation
│   │   └── run_explanations.py    # Orchestrator for all explainers
│   ├── fairness/
│   │   ├── metrics.py             # Fairness metric calculations
│   │   └── audit.py               # Automated bias audit pipeline
│   ├── regulatory/
│   │   ├── eu_ai_act.py           # EU AI Act compliance checks
│   │   ├── ecoa.py                # ECOA adverse action mapping
│   │   ├── sr11_7.py              # Fed SR 11-7 documentation
│   │   └── compliance_engine.py   # Unified compliance orchestrator
│   ├── dashboard/
│   │   └── app.py                 # Main Streamlit application
│   └── utils/
│       ├── logger.py              # Structured logging
│       └── config.py              # Configuration loader
├── tests/
│   ├── test_data.py               # Data pipeline tests
│   ├── test_models.py             # Model training tests
│   ├── test_explainability.py     # SHAP & LIME tests
│   └── test_fairness.py           # Fairness audit tests
├── .gitignore
├── .gitattributes
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🧪 Testing

```bash
pytest tests/ -v --cov=src
```

---

## 📜 Regulatory Framework Coverage

| Regulation | Aspect Covered | Module |
|---|---|---|
| EU AI Act (2024) | High-risk AI transparency, human oversight documentation | `regulatory/eu_ai_act.py` |
| US ECOA / Reg B | Adverse action reasons, prohibited basis monitoring | `regulatory/ecoa.py` |
| Fed SR 11-7 | Model validation, performance monitoring, documentation | `regulatory/sr11_7.py` |

---

## 🛠️ Tech Stack

- **ML**: XGBoost, LightGBM, scikit-learn, Optuna
- **Explainability**: SHAP, LIME, DiCE
- **Fairness**: Custom implementation (disparate impact, statistical parity, equalized odds, predictive parity)
- **Dashboard**: Streamlit
- **Data**: pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Testing**: pytest
- **Config**: PyYAML

---

## 👤 Author

**Vishwas Kothari**
MS Computer Science, University of Colorado Boulder
[LinkedIn](https://www.linkedin.com/in/vishwas-kothari) | [Email](mailto:vishwasvkothari@gmail.com)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
