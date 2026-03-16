"""Run the full fairness audit on the test set.

Evaluates model predictions across all protected demographic groups
and generates a compliance-ready audit report.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.fairness.metrics import FairnessAuditor
from src.data.preprocess import get_feature_columns
from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT

logger = get_logger("fairness.audit")

MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
REPORTS_DIR = PROJECT_ROOT / "reports" / "audit"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


def run_audit():
    """Execute full fairness audit and save results."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load best model
    best_name = (MODELS_DIR / "best_model.txt").read_text().strip()
    with open(MODELS_DIR / f"{best_name}.pkl", "rb") as f:
        model = pickle.load(f)
    logger.info(f"Loaded model: {best_name}")

    # Load test data
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    feature_cols = get_feature_columns(test)
    X_test = test[feature_cols]
    y_test = test["DEFAULT"].values

    # Predictions
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    logger.info(f"Test set: {len(test)} samples, {y_pred.mean():.2%} predicted default rate")

    # Run audit
    auditor = FairnessAuditor()
    sensitive_features = test[["SEX", "EDUCATION", "MARRIAGE", "AGE"]]

    audit_report = auditor.run_full_audit(y_test, y_pred, y_prob, sensitive_features)

    # Print summary
    print(audit_report["summary"])
    print(f"\nOverall Status: {audit_report['overall_status']}")
    print(f"Attributes Audited: {audit_report['total_attributes_audited']}")
    print(f"Attributes with Violations: {audit_report['attributes_with_violations']}")

    # Print detailed metrics for each attribute
    for attr_key, result in audit_report["audit_results"].items():
        print(f"\n{'='*50}")
        print(f"{result['label']} (n_priv={result['privileged_count']}, n_unpriv={result['unprivileged_count']})")
        print(f"{'='*50}")
        for metric, value in result["metrics"].items():
            flag = " ⚠️" if result["flags"].get(metric, False) else " ✅"
            print(f"  {metric}: {value:.4f}{flag}")
        if result["recommendations"]:
            print("  Recommendations:")
            for rec in result["recommendations"]:
                print(f"    → {rec}")

    # Save JSON report
    # Convert for JSON serialization
    json_report = json.loads(json.dumps(audit_report, default=str))
    with open(REPORTS_DIR / "fairness_audit_report.json", "w") as f:
        json.dump(json_report, f, indent=2)
    logger.info(f"Saved audit report to {REPORTS_DIR / 'fairness_audit_report.json'}")

    # Generate visualizations
    generate_audit_visualizations(audit_report)

    return audit_report


def generate_audit_visualizations(audit_report: dict):
    """Generate plotly visualizations for the fairness audit."""

    results = audit_report["audit_results"]

    # 1. Disparate Impact across all attributes
    attrs = []
    di_values = []
    colors = []
    for attr_key, result in results.items():
        attrs.append(result["label"])
        di = result["metrics"]["disparate_impact"]
        di_values.append(di)
        colors.append("#dc3545" if di < 0.8 else "#28a745")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=attrs, y=di_values,
        marker_color=colors,
        text=[f"{v:.3f}" for v in di_values],
        textposition="outside",
    ))
    fig.add_hline(y=0.8, line_dash="dash", line_color="red",
                  annotation_text="Four-Fifths Threshold (0.80)")
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray",
                  annotation_text="Perfect Parity (1.00)")
    fig.update_layout(
        title="Disparate Impact Ratio by Protected Attribute",
        yaxis_title="Disparate Impact Ratio",
        yaxis_range=[0, max(di_values) * 1.3],
        height=400,
        plot_bgcolor="white",
    )
    fig.write_html(str(FIGURES_DIR / "disparate_impact.html"))
    fig.write_image(str(FIGURES_DIR / "disparate_impact.png"), scale=2)
    logger.info("Saved disparate impact chart")

    # 2. Selection rates comparison
    fig2 = go.Figure()
    for attr_key, result in results.items():
        label = result["label"]
        priv_rate = result["metrics"]["privileged_favorable_rate"]
        unpriv_rate = result["metrics"]["unprivileged_favorable_rate"]

        fig2.add_trace(go.Bar(
            name=f"{label} - Privileged",
            x=[label], y=[priv_rate],
            marker_color="#4a90d9",
            text=[f"{priv_rate:.1%}"],
            textposition="outside",
        ))
        fig2.add_trace(go.Bar(
            name=f"{label} - Unprivileged",
            x=[label], y=[unpriv_rate],
            marker_color="#e8913a",
            text=[f"{unpriv_rate:.1%}"],
            textposition="outside",
        ))

    fig2.update_layout(
        title="Favorable Outcome Rates: Privileged vs Unprivileged Groups",
        yaxis_title="Favorable Outcome Rate (Non-Default)",
        barmode="group",
        height=400,
        plot_bgcolor="white",
        showlegend=False,
    )
    fig2.write_html(str(FIGURES_DIR / "selection_rates.html"))
    fig2.write_image(str(FIGURES_DIR / "selection_rates.png"), scale=2)
    logger.info("Saved selection rates chart")

    # 3. Fairness metrics heatmap
    metric_keys = [
        "disparate_impact",
        "statistical_parity_difference",
        "equal_opportunity_difference",
        "equalized_odds_difference",
        "predictive_parity_difference",
    ]
    metric_labels = [
        "Disparate Impact",
        "Statistical Parity Diff",
        "Equal Opportunity Diff",
        "Equalized Odds Diff",
        "Predictive Parity Diff",
    ]

    attr_labels = [r["label"] for r in results.values()]
    heatmap_data = []
    annotations = []

    for i, mk in enumerate(metric_keys):
        row = []
        for j, (attr_key, result) in enumerate(results.items()):
            val = result["metrics"][mk]
            row.append(val)
            flag = result["flags"].get(mk, False)
            annotations.append(dict(
                x=j, y=i,
                text=f"{val:.3f}" + (" ⚠️" if flag else ""),
                showarrow=False,
                font=dict(color="white" if abs(val) > 0.5 else "black"),
            ))
        heatmap_data.append(row)

    fig3 = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=attr_labels,
        y=metric_labels,
        colorscale="RdYlGn",
        text=[[f"{v:.3f}" for v in row] for row in heatmap_data],
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    fig3.update_layout(
        title="Fairness Metrics Dashboard",
        height=400,
        margin=dict(l=200),
    )
    fig3.write_html(str(FIGURES_DIR / "fairness_heatmap.html"))
    fig3.write_image(str(FIGURES_DIR / "fairness_heatmap.png"), scale=2)
    logger.info("Saved fairness heatmap")

    print(f"\nVisualizations saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_audit()
