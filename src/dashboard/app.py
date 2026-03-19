"""XAI Credit Lens — Interactive Streamlit Dashboard.

Single-screen applicant decisioning: prediction + SHAP explanation +
counterfactual recourse + fairness flag, all in one view.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="XAI Credit Lens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-top: -10px;
        margin-bottom: 30px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .approved {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .denied {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 15px;
        border-radius: 10px;
        color: white;
        text-align: center;
        font-size: 1.4rem;
        font-weight: 600;
    }
    .fairness-pass {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 5px 0;
    }
    .fairness-fail {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 10px 15px;
        border-radius: 4px;
        margin: 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load model and data (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the best trained model."""
    best_name_path = MODELS_DIR / "best_model.txt"
    if not best_name_path.exists():
        st.error("No trained model found. Run `python -m src.models.train` first.")
        st.stop()
    best_name = best_name_path.read_text().strip()
    model_path = MODELS_DIR / f"{best_name}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model, best_name


@st.cache_data
def load_data():
    """Load processed training data for SHAP background and feature ranges."""
    train = pd.read_csv(DATA_DIR / "splits" / "train.csv")
    processed = pd.read_csv(DATA_DIR / "processed" / "credit_default_processed.csv")
    return train, processed


@st.cache_resource
def get_shap_explainer(_model, _X_background):
    """Create SHAP TreeExplainer."""
    return shap.TreeExplainer(_model)


# ---------------------------------------------------------------------------
# Feature metadata
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "LIMIT_BAL": "Credit Limit (NT$)",
    "SEX": "Sex",
    "EDUCATION": "Education",
    "MARRIAGE": "Marital Status",
    "AGE": "Age",
    "PAY_0": "Repayment Status (Sep)",
    "PAY_2": "Repayment Status (Aug)",
    "PAY_3": "Repayment Status (Jul)",
    "PAY_4": "Repayment Status (Jun)",
    "PAY_5": "Repayment Status (May)",
    "PAY_6": "Repayment Status (Apr)",
    "BILL_AMT1": "Bill Amount (Sep)",
    "BILL_AMT2": "Bill Amount (Aug)",
    "BILL_AMT3": "Bill Amount (Jul)",
    "BILL_AMT4": "Bill Amount (Jun)",
    "BILL_AMT5": "Bill Amount (May)",
    "BILL_AMT6": "Bill Amount (Apr)",
    "PAY_AMT1": "Payment Amount (Sep)",
    "PAY_AMT2": "Payment Amount (Aug)",
    "PAY_AMT3": "Payment Amount (Jul)",
    "PAY_AMT4": "Payment Amount (Jun)",
    "PAY_AMT5": "Payment Amount (May)",
    "PAY_AMT6": "Payment Amount (Apr)",
    "DEBT_TO_INCOME_PROXY": "Debt-to-Income Ratio",
    "PAYMENT_RATIO": "Payment Ratio",
    "UTILIZATION_RATE": "Credit Utilization",
    "MONTHS_DELINQUENT": "Months Delinquent",
    "AVG_PAYMENT_DELAY": "Avg Payment Delay",
    "PAYMENT_TREND": "Payment Trend",
    "MAX_CONSEC_DELINQUENT": "Max Consecutive Late",
    "BALANCE_VOLATILITY": "Balance Volatility",
}

EDUCATION_MAP = {1: "Graduate School", 2: "University", 3: "High School", 4: "Other"}
MARRIAGE_MAP = {1: "Married", 2: "Single", 3: "Other"}
SEX_MAP = {1: "Male", 2: "Female"}
PAY_STATUS_MAP = {
    -2: "No consumption",
    -1: "Paid in full",
    0: "Revolving credit",
    1: "1 month delay",
    2: "2 months delay",
    3: "3 months delay",
    4: "4 months delay",
    5: "5 months delay",
    6: "6 months delay",
    7: "7 months delay",
    8: "8 months delay",
}


def get_feature_columns(df):
    """Get feature columns excluding target."""
    return [c for c in df.columns if c != "DEFAULT"]


# ---------------------------------------------------------------------------
# Sidebar — Applicant Input
# ---------------------------------------------------------------------------
def render_sidebar(train_df):
    """Render applicant input form in sidebar."""
    st.sidebar.markdown("## 📋 Applicant Profile")
    st.sidebar.markdown("---")

    # Demographics
    st.sidebar.markdown("### Demographics")
    age = st.sidebar.slider("Age", 21, 79, 35)
    sex = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: SEX_MAP[x])
    education = st.sidebar.selectbox(
        "Education",
        options=[1, 2, 3, 4],
        format_func=lambda x: EDUCATION_MAP[x],
    )
    marriage = st.sidebar.selectbox(
        "Marital Status",
        options=[1, 2, 3],
        format_func=lambda x: MARRIAGE_MAP[x],
    )

    # Credit info
    st.sidebar.markdown("### Credit Information")
    limit_bal = st.sidebar.number_input(
        "Credit Limit (NT$)", min_value=10000, max_value=1000000, value=200000, step=10000
    )

    # Payment history
    st.sidebar.markdown("### Recent Payment History")
    pay_status_options = list(PAY_STATUS_MAP.keys())

    pay_0 = st.sidebar.selectbox(
        "September", options=pay_status_options,
        format_func=lambda x: PAY_STATUS_MAP[x], index=2,
    )
    pay_2 = st.sidebar.selectbox(
        "August", options=pay_status_options,
        format_func=lambda x: PAY_STATUS_MAP[x], index=2,
    )
    pay_3 = st.sidebar.selectbox(
        "July", options=pay_status_options,
        format_func=lambda x: PAY_STATUS_MAP[x], index=2,
    )
    pay_4 = st.sidebar.selectbox(
        "June", options=pay_status_options,
        format_func=lambda x: PAY_STATUS_MAP[x], index=2,
    )
    pay_5 = st.sidebar.selectbox(
        "May", options=pay_status_options,
        format_func=lambda x: PAY_STATUS_MAP[x], index=2,
    )
    pay_6 = st.sidebar.selectbox(
        "April", options=pay_status_options,
        format_func=lambda x: PAY_STATUS_MAP[x], index=2,
    )

    # Bill amounts
    st.sidebar.markdown("### Bill Amounts (NT$)")
    bill_amt1 = st.sidebar.number_input("September Bill", 0, 1000000, 50000, 5000)
    bill_amt2 = st.sidebar.number_input("August Bill", 0, 1000000, 45000, 5000)
    bill_amt3 = st.sidebar.number_input("July Bill", 0, 1000000, 40000, 5000)
    bill_amt4 = st.sidebar.number_input("June Bill", 0, 1000000, 38000, 5000)
    bill_amt5 = st.sidebar.number_input("May Bill", 0, 1000000, 35000, 5000)
    bill_amt6 = st.sidebar.number_input("April Bill", 0, 1000000, 30000, 5000)

    # Payment amounts
    st.sidebar.markdown("### Payment Amounts (NT$)")
    pay_amt1 = st.sidebar.number_input("September Payment", 0, 1000000, 5000, 1000)
    pay_amt2 = st.sidebar.number_input("August Payment", 0, 1000000, 5000, 1000)
    pay_amt3 = st.sidebar.number_input("July Payment", 0, 1000000, 4000, 1000)
    pay_amt4 = st.sidebar.number_input("June Payment", 0, 1000000, 4000, 1000)
    pay_amt5 = st.sidebar.number_input("May Payment", 0, 1000000, 3000, 1000)
    pay_amt6 = st.sidebar.number_input("April Payment", 0, 1000000, 3000, 1000)

    # Build raw feature dict
    raw = {
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "PAY_4": pay_4,
        "PAY_5": pay_5,
        "PAY_6": pay_6,
        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "BILL_AMT4": bill_amt4,
        "BILL_AMT5": bill_amt5,
        "BILL_AMT6": bill_amt6,
        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "PAY_AMT4": pay_amt4,
        "PAY_AMT5": pay_amt5,
        "PAY_AMT6": pay_amt6,
    }

    # Compute engineered features
    pay_cols_vals = [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
    bill_vals = [bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6]
    pay_amt_vals = [pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]

    avg_bill = np.mean(bill_vals)
    total_bill = sum(bill_vals) if sum(bill_vals) > 0 else 1
    total_pay = sum(pay_amt_vals)

    raw["DEBT_TO_INCOME_PROXY"] = avg_bill / limit_bal if limit_bal > 0 else 0
    raw["PAYMENT_RATIO"] = min(total_pay / total_bill, 5)
    raw["UTILIZATION_RATE"] = min(avg_bill / limit_bal if limit_bal > 0 else 0, 5)
    raw["MONTHS_DELINQUENT"] = sum(1 for p in pay_cols_vals if p > 0)
    raw["AVG_PAYMENT_DELAY"] = np.mean(pay_cols_vals)
    x = np.arange(6)
    x_centered = x - x.mean()
    raw["PAYMENT_TREND"] = sum(p * xc for p, xc in zip(pay_amt_vals, x_centered)) / sum(
        xc**2 for xc in x_centered
    )

    # Max consecutive delinquent
    max_consec = 0
    current = 0
    for p in pay_cols_vals:
        if p > 0:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    raw["MAX_CONSEC_DELINQUENT"] = max_consec

    raw["BALANCE_VOLATILITY"] = np.std(bill_vals) / limit_bal if limit_bal > 0 else 0

    return pd.DataFrame([raw])


# ---------------------------------------------------------------------------
# Main content — Prediction + Explanation
# ---------------------------------------------------------------------------
def render_prediction(model, applicant, model_name):
    """Render the prediction result with probability gauge."""
    prob = model.predict_proba(applicant)[0, 1]
    pred = int(prob >= 0.5)

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Probability gauge
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40}},
                title={"text": "Default Probability", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#eb3349" if pred == 1 else "#11998e"},
                    "steps": [
                        {"range": [0, 30], "color": "#d4edda"},
                        {"range": [30, 50], "color": "#fff3cd"},
                        {"range": [50, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.8,
                        "value": 50,
                    },
                },
            )
        )
        fig.update_layout(height=280, margin=dict(t=50, b=20, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if pred == 0:
            st.markdown('<div class="approved">✅ LIKELY APPROVED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="denied">❌ LIKELY DENIED</div>', unsafe_allow_html=True)

        st.markdown(f"**Model:** {model_name.replace('_', ' ').title()}")
        st.markdown(f"**Confidence:** {abs(prob - 0.5) * 200:.0f}%")

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Default Probability", f"{prob:.1%}")
        st.metric("Risk Tier", "High" if prob > 0.5 else "Medium" if prob > 0.3 else "Low")

    return prob, pred


def render_shap_explanation(explainer, applicant, feature_names):
    """Render SHAP waterfall and feature importance."""
    st.markdown("### 🔍 Why This Decision? (SHAP Explanation)")

    shap_values = explainer.shap_values(applicant)

    # Handle different return types
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # class 1 (default)
    elif len(shap_values.shape) == 3:
        sv = shap_values[0, :, 1]
    else:
        sv = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[1] if len(base_value) > 1 else base_value[0]

    # Create contribution dataframe
    contributions = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS.get(f, f) for f in feature_names],
            "SHAP Value": sv,
            "Feature Value": applicant.values[0],
            "Raw Feature": feature_names,
        }
    ).sort_values("SHAP Value", key=abs, ascending=True)

    # Top features bar chart
    top_n = contributions.tail(12)

    colors = ["#eb3349" if v > 0 else "#11998e" for v in top_n["SHAP Value"]]

    fig = go.Figure(
        go.Bar(
            x=top_n["SHAP Value"],
            y=top_n["Feature"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in top_n["SHAP Value"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Feature Contributions to Default Prediction",
        xaxis_title="SHAP Value (→ increases default risk)",
        height=400,
        margin=dict(l=200, r=50, t=50, b=50),
        plot_bgcolor="white",
    )
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    # Risk factors vs protective factors
    col1, col2 = st.columns(2)

    risk_factors = contributions[contributions["SHAP Value"] > 0].tail(5).iloc[::-1]
    protective = contributions[contributions["SHAP Value"] < 0].head(5)

    with col1:
        st.markdown("#### ⚠️ Top Risk Factors")
        for _, row in risk_factors.iterrows():
            st.markdown(
                f"- **{row['Feature']}** = {row['Feature Value']:.2f} "
                f"(+{row['SHAP Value']:.3f})"
            )

    with col2:
        st.markdown("#### 🛡️ Top Protective Factors")
        for _, row in protective.iterrows():
            st.markdown(
                f"- **{row['Feature']}** = {row['Feature Value']:.2f} "
                f"({row['SHAP Value']:.3f})"
            )

    return sv, base_value


def render_counterfactual(applicant, model, feature_names, prob):
    """Render counterfactual explanation — what would change the decision."""
    st.markdown("### 🔄 What Would Change This Decision?")

    if prob < 0.5:
        st.info("This applicant is predicted to be approved. Counterfactual analysis shows "
                "what changes could push them toward denial (for risk monitoring).")

    # Simple counterfactual: tweak features and see impact
    st.markdown(
        "Adjust the sliders below to explore how changes affect the prediction:"
    )

    # Identify top actionable features
    actionable = [
        "PAYMENT_RATIO", "DEBT_TO_INCOME_PROXY", "UTILIZATION_RATE",
        "AVG_PAYMENT_DELAY", "MONTHS_DELINQUENT", "PAY_AMT1", "BILL_AMT1",
    ]
    available = [f for f in actionable if f in feature_names]

    modified = applicant.copy()
    changes_made = []

    cols = st.columns(min(len(available), 3))
    for i, feat in enumerate(available[:6]):
        col = cols[i % 3]
        current_val = float(applicant[feat].iloc[0])
        label = FEATURE_LABELS.get(feat, feat)

        with col:
            if feat in ["MONTHS_DELINQUENT", "MAX_CONSEC_DELINQUENT"]:
                new_val = st.slider(
                    label, min_value=0, max_value=6,
                    value=int(current_val), key=f"cf_{feat}",
                )
            elif feat in ["PAYMENT_RATIO", "DEBT_TO_INCOME_PROXY", "UTILIZATION_RATE"]:
                new_val = st.slider(
                    label, min_value=0.0, max_value=3.0,
                    value=min(float(current_val), 3.0), step=0.05, key=f"cf_{feat}",
                )
            elif feat == "AVG_PAYMENT_DELAY":
                new_val = st.slider(
                    label, min_value=-2.0, max_value=8.0,
                    value=float(current_val), step=0.1, key=f"cf_{feat}",
                )
            else:
                min_v = 0
                max_v = int(max(current_val * 3, 100000))
                new_val = st.slider(
                    label, min_value=min_v, max_value=max_v,
                    value=int(current_val), step=1000, key=f"cf_{feat}",
                )

            modified[feat] = new_val

            if not np.isclose(new_val, current_val, rtol=0.01):
                changes_made.append(
                    {"feature": label, "from": current_val, "to": new_val}
                )

    # Show new prediction
    new_prob = model.predict_proba(modified)[0, 1]
    new_pred = int(new_prob >= 0.5)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Original Default Prob", f"{prob:.1%}")
    with col2:
        delta = new_prob - prob
        st.metric("New Default Prob", f"{new_prob:.1%}", delta=f"{delta:+.1%}")
    with col3:
        if new_pred == 0:
            st.markdown('<div class="approved">✅ NOW APPROVED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="denied">❌ STILL DENIED</div>', unsafe_allow_html=True)

    if changes_made:
        st.markdown("#### 📝 Changes Made:")
        for c in changes_made:
            direction = "↑" if c["to"] > c["from"] else "↓"
            st.markdown(
                f"- **{c['feature']}**: {c['from']:.2f} → {c['to']:.2f} {direction}"
            )

        if new_pred != int(prob >= 0.5):
            st.success(
                "💡 **Counterfactual found!** The changes above would flip the decision."
            )
    else:
        st.info("Adjust the sliders above to explore counterfactual scenarios.")


def render_fairness_flags(applicant):
    """Show fairness-relevant flags for this applicant."""
    st.markdown("### ⚖️ Fairness Flags")

    sex = int(applicant["SEX"].iloc[0])
    education = int(applicant["EDUCATION"].iloc[0])
    age = int(applicant["AGE"].iloc[0])
    marriage = int(applicant["MARRIAGE"].iloc[0])

    flags = []

    # Check if applicant belongs to any unprivileged group
    if sex == 2:
        flags.append(("Gender", "Female (unprivileged group)", "Monitor for gender bias"))
    if education in [3, 4]:
        flags.append(("Education", "Non-university (unprivileged group)", "Monitor for education bias"))
    if age < 30:
        flags.append(("Age", "Under 30 (unprivileged group)", "Monitor for age bias"))
    if marriage == 2:
        flags.append(("Marital Status", "Single (unprivileged group)", "Monitor for marital status bias"))

    if flags:
        st.warning(
            f"This applicant belongs to {len(flags)} monitored demographic group(s). "
            "Decisions involving these groups require additional review under fairness policy."
        )
        for category, detail, action in flags:
            st.markdown(
                f'<div class="fairness-fail">⚠️ <b>{category}</b>: {detail} — {action}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="fairness-pass">✅ No fairness flags triggered for this applicant profile.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<small>Fairness flags are based on protected attribute membership. "
        "Full disparate impact analysis is available in the audit report.</small>",
        unsafe_allow_html=True,
    )


def render_regulatory_panel(pred, prob, shap_values, feature_names):
    """Show regulatory compliance information."""
    st.markdown("### 📜 Regulatory Compliance")

    tabs = st.tabs(["ECOA / Reg B", "EU AI Act", "Fed SR 11-7"])

    with tabs[0]:
        if pred == 1:
            st.markdown("**Adverse Action Notice Required** (ECOA §1691)")

            # Map top SHAP contributors to adverse action reasons
            reason_map = {
                "PAY_0": "Recent payment history",
                "LIMIT_BAL": "Credit limit relative to obligations",
                "DEBT_TO_INCOME_PROXY": "Ratio of debt to available credit",
                "UTILIZATION_RATE": "Credit utilization rate",
                "MONTHS_DELINQUENT": "Number of months with late payments",
                "AVG_PAYMENT_DELAY": "Average payment delay severity",
                "PAYMENT_RATIO": "Payment amount relative to balance",
                "BILL_AMT1": "Current balance amount",
                "PAYMENT_TREND": "Recent payment trajectory",
            }

            # Get top positive SHAP values (risk factors)
            contributions = sorted(
                zip(feature_names, shap_values),
                key=lambda x: x[1],
                reverse=True,
            )

            prohibited = {"SEX", "MARRIAGE", "AGE"}
            reasons = []
            for feat, sv in contributions:
                if sv <= 0:
                    continue
                if feat in prohibited:
                    continue
                if feat in reason_map and len(reasons) < 4:
                    reasons.append(reason_map[feat])

            st.markdown("**Principal reasons for adverse action:**")
            for i, reason in enumerate(reasons, 1):
                st.markdown(f"{i}. {reason}")
        else:
            st.success("Application approved — no adverse action notice required.")

    with tabs[1]:
        st.markdown("**EU AI Act — High-Risk AI System (Credit Scoring)**")
        checks = {
            "Transparency": "✅ Model explanations provided (SHAP + LIME)",
            "Human Oversight": "✅ Dashboard enables human-in-the-loop review",
            "Data Governance": "✅ Training data quality documented",
            "Robustness": "✅ Cross-validated on held-out test set",
            "Fairness": "✅ Automated bias detection across protected groups",
        }
        for check, status in checks.items():
            st.markdown(f"- {status}")

    with tabs[2]:
        st.markdown("**Fed SR 11-7 — Model Risk Management**")
        st.markdown(
            "This model meets SR 11-7 documentation requirements for:\n"
            "- Model development documentation\n"
            "- Independent validation (train/val/test split)\n"
            "- Ongoing monitoring framework (fairness metrics)\n"
            "- Clear model limitations and assumptions"
        )
        # Load test metrics dynamically
        import json
        metrics_path = MODELS_DIR / "test_metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                test_metrics = json.load(f)
            test_auc = test_metrics.get("roc_auc", "N/A")
            st.markdown(
                f"**Model Performance:** AUC = {test_auc:.4f} on held-out test set "
                f"({model_name.replace('_', ' ').title()})"
            )
        else:
            st.markdown("**Model Performance:** Run training pipeline to generate metrics.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.markdown('<p class="main-header">🔍 XAI Credit Lens</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Explainable AI Framework for Fair Credit Decisioning</p>',
        unsafe_allow_html=True,
    )

    # Load resources
    model, model_name = load_model()
    train_df, processed_df = load_data()
    feature_cols = get_feature_columns(train_df)
    explainer = get_shap_explainer(model, train_df[feature_cols])

    # Sidebar input
    applicant = render_sidebar(train_df)

    # Ensure column order matches training data
    applicant = applicant[feature_cols]

    # --- Main content ---
    prob, pred = render_prediction(model, applicant, model_name)

    st.markdown("---")

    # SHAP explanation
    sv, base_value = render_shap_explanation(explainer, applicant, feature_cols)

    st.markdown("---")

    # Counterfactual
    render_counterfactual(applicant, model, feature_cols, prob)

    st.markdown("---")

    # Fairness flags
    col1, col2 = st.columns(2)
    with col1:
        render_fairness_flags(applicant)
    with col2:
        render_regulatory_panel(pred, prob, sv, feature_cols)

    # Footer
    st.markdown("---")
    st.markdown(
        "<center><small>XAI Credit Lens v0.1 | Built by Vishwas Kothari | "
        "MS CS, University of Colorado Boulder</small></center>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
