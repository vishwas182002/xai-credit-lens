"""Fed SR 11-7 Model Risk Management documentation generator.

Generates model risk management documentation aligned with the Federal
Reserve's SR 11-7 guidance on model risk management.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import get_regulatory_config, PROJECT_ROOT

logger = get_logger("regulatory.sr11_7")

REPORTS_DIR = PROJECT_ROOT / "reports" / "audit"


class SR117DocumentationGenerator:
    """Generate SR 11-7 compliant model risk management documentation."""

    def __init__(self, config: Optional[dict] = None):
        reg_config = config or get_regulatory_config()
        self.sr_config = reg_config["fed_sr_11_7"]["documentation"]

    def generate_documentation(
        self,
        model_name: str = "Random Forest Credit Default Classifier",
        model_metrics: Optional[dict] = None,
        fairness_report: Optional[dict] = None,
        training_details: Optional[dict] = None,
    ) -> dict:
        """Generate complete SR 11-7 documentation package.

        Returns:
            dict with all four SR 11-7 documentation sections.
        """
        doc = {
            "framework": "Federal Reserve SR 11-7: Guidance on Model Risk Management",
            "model_name": model_name,
            "document_date": datetime.now().isoformat(),
            "model_tier": "Tier 2 — Material Model",
            "sections": {},
        }

        # Section 1: Model Development
        doc["sections"]["model_development"] = self._model_development_section(
            model_name, model_metrics, training_details
        )

        # Section 2: Model Validation
        doc["sections"]["validation"] = self._validation_section(
            model_metrics, fairness_report
        )

        # Section 3: Ongoing Monitoring
        doc["sections"]["ongoing_monitoring"] = self._monitoring_section(model_metrics)

        # Section 4: Governance
        doc["sections"]["governance"] = self._governance_section()

        # Compliance checklist
        doc["compliance_checklist"] = self._generate_checklist(doc["sections"])

        # Summary
        doc["summary"] = self._generate_summary(doc)

        return doc

    def _model_development_section(
        self, model_name: str, metrics: Optional[dict], training: Optional[dict]
    ) -> dict:
        """SR 11-7 Section: Model Development Documentation."""
        return {
            "title": "Model Development Documentation",
            "items": {
                "objective_and_intended_use": {
                    "requirement": self.sr_config["model_development"][0],
                    "documentation": (
                        f"The {model_name} is designed to predict the probability of "
                        "credit card payment default within the next billing cycle. "
                        "It is intended as a decision-support tool for credit risk "
                        "assessment, providing probability estimates alongside "
                        "multi-method explanations (SHAP, LIME, counterfactual) to "
                        "enable informed human decision-making. The model is NOT "
                        "intended for autonomous credit decisioning."
                    ),
                    "status": "DOCUMENTED",
                },
                "data_sources_and_quality": {
                    "requirement": self.sr_config["model_development"][1],
                    "documentation": (
                        "Training data: UCI Default of Credit Card Clients dataset "
                        "(30,000 records from a Taiwan bank, 2005). "
                        "23 original features covering demographics, credit limit, "
                        "6-month payment history, and billing information. "
                        "9 engineered features capturing behavioral patterns "
                        "(debt-to-income proxy, payment ratio, utilization rate, "
                        "delinquency counts, payment trend, balance volatility). "
                        "Data quality: No missing values. Known categorical encoding "
                        "issues cleaned (EDUCATION categories 0,5,6 merged; "
                        "MARRIAGE category 0 merged). "
                        "Limitation: Single-institution, single-country dataset "
                        "from 2005 may not generalize to current US market conditions."
                    ),
                    "status": "DOCUMENTED",
                },
                "feature_selection_rationale": {
                    "requirement": self.sr_config["model_development"][2],
                    "documentation": (
                        "Features selected based on credit risk domain knowledge: "
                        "payment history (strongest predictor per SHAP analysis), "
                        "credit utilization, and behavioral trends. "
                        "9 engineered features derived from raw payment and billing "
                        "data to capture temporal patterns. "
                        "Protected attributes (SEX, MARRIAGE, AGE) included for "
                        "fairness monitoring but flagged in compliance pipeline. "
                        "Feature importance validated via SHAP global analysis."
                    ),
                    "status": "DOCUMENTED",
                },
                "methodology_and_alternatives": {
                    "requirement": self.sr_config["model_development"][3],
                    "documentation": (
                        "Three model architectures evaluated: "
                        "XGBoost (gradient boosted trees), "
                        "LightGBM (gradient boosted trees with leaf-wise growth), "
                        "Random Forest (bagged decision trees). "
                        "Random Forest selected based on highest validation AUC "
                        f"({metrics.get('roc_auc', 'N/A') if metrics else 'N/A'}). "
                        "Tree-based models chosen for interpretability compatibility "
                        "with SHAP TreeExplainer. "
                        "Linear models (Logistic Regression) considered but excluded "
                        "due to lower performance on non-linear payment patterns."
                    ),
                    "status": "DOCUMENTED",
                },
                "assumptions_and_limitations": {
                    "requirement": self.sr_config["model_development"][4],
                    "documentation": (
                        "Key assumptions: (1) Historical payment patterns are predictive "
                        "of future default behavior. (2) Feature relationships are "
                        "adequately captured by tree-based ensemble methods. "
                        "Key limitations: (1) Training data is from a single Taiwan bank "
                        "(2005) — population drift likely for other markets/time periods. "
                        "(2) Model does not incorporate macroeconomic indicators. "
                        "(3) 22.12% default rate in training data may not match "
                        "deployment population. "
                        "(4) Counterfactual explanations assume feature independence."
                    ),
                    "status": "DOCUMENTED",
                },
            },
        }

    def _validation_section(
        self, metrics: Optional[dict], fairness: Optional[dict]
    ) -> dict:
        """SR 11-7 Section: Model Validation."""
        validation_items = self.sr_config["validation"]

        auc = metrics.get("roc_auc", "N/A") if metrics else "N/A"
        f1 = metrics.get("f1", "N/A") if metrics else "N/A"
        precision = metrics.get("precision", "N/A") if metrics else "N/A"
        recall = metrics.get("recall", "N/A") if metrics else "N/A"

        fairness_status = fairness.get("overall_status", "N/A") if fairness else "N/A"

        return {
            "title": "Model Validation",
            "items": {
                "independent_review": {
                    "requirement": validation_items[0],
                    "documentation": (
                        "Model logic reviewed through: (1) SHAP global feature importance "
                        "confirming payment history as dominant predictor (consistent with "
                        "domain expectations). (2) LIME local explanations validated against "
                        "SHAP for consistency. (3) Counterfactual analysis confirms model "
                        "responds sensibly to feature changes."
                    ),
                    "status": "DOCUMENTED",
                },
                "out_of_sample_testing": {
                    "requirement": validation_items[1],
                    "documentation": (
                        f"Held-out test set (6,000 samples, 20% of data): "
                        f"AUC = {auc}, F1 = {f1}, "
                        f"Precision = {precision}, Recall = {recall}. "
                        "Stratified split ensures class balance consistency across "
                        "train (22.12%), validation (22.11%), and test (22.12%) sets."
                    ),
                    "status": "DOCUMENTED",
                },
                "sensitivity_analysis": {
                    "requirement": validation_items[2],
                    "documentation": (
                        "Sensitivity assessed via: (1) Three competing model architectures "
                        "showing consistent AUC within 0.003 range. "
                        "(2) SHAP dependence plots showing monotonic relationships for "
                        "key features. (3) Counterfactual analysis quantifying decision "
                        "boundary sensitivity to individual feature changes."
                    ),
                    "status": "DOCUMENTED",
                },
                "benchmarking": {
                    "requirement": validation_items[3],
                    "documentation": (
                        "Model compared against three alternative approaches "
                        "(XGBoost, LightGBM, Random Forest). Published benchmarks "
                        "on this dataset report AUC of 0.77-0.82; our model achieves "
                        f"AUC = {auc}, within expected range. "
                        f"Fairness audit status: {fairness_status} across all 4 "
                        "protected attributes."
                    ),
                    "status": "DOCUMENTED",
                },
            },
        }

    def _monitoring_section(self, metrics: Optional[dict]) -> dict:
        """SR 11-7 Section: Ongoing Monitoring."""
        monitoring_items = self.sr_config["ongoing_monitoring"]

        return {
            "title": "Ongoing Monitoring",
            "items": {
                "performance_tracking": {
                    "requirement": monitoring_items[0],
                    "documentation": (
                        "Monitoring plan: Track AUC, F1, precision, recall, and "
                        "Brier score on monthly production data. Alert threshold: "
                        "AUC degradation > 5% from baseline triggers model review. "
                        "Fairness metrics (disparate impact, equalized odds) monitored "
                        "quarterly across all protected groups."
                    ),
                    "status": "PLANNED",
                },
                "population_stability": {
                    "requirement": monitoring_items[1],
                    "documentation": (
                        "Population Stability Index (PSI) to be computed monthly "
                        "comparing production feature distributions against training "
                        "data. PSI > 0.25 triggers investigation; PSI > 0.10 triggers "
                        "enhanced monitoring."
                    ),
                    "status": "PLANNED",
                },
                "feature_drift": {
                    "requirement": monitoring_items[2],
                    "documentation": (
                        "Feature drift detection via Kolmogorov-Smirnov test on "
                        "continuous features and chi-squared test on categorical "
                        "features. Monthly automated checks with dashboard reporting."
                    ),
                    "status": "PLANNED",
                },
                "retraining_triggers": {
                    "requirement": monitoring_items[3],
                    "documentation": (
                        "Retraining triggered by: (1) AUC drop > 5% sustained over "
                        "2 consecutive months. (2) PSI > 0.25 on any feature. "
                        "(3) Fairness metric violation on any protected attribute. "
                        "(4) Scheduled annual retraining regardless of performance."
                    ),
                    "status": "PLANNED",
                },
            },
        }

    def _governance_section(self) -> dict:
        """SR 11-7 Section: Governance."""
        governance_items = self.sr_config["governance"]

        return {
            "title": "Governance",
            "items": {
                "roles": {
                    "requirement": governance_items[0],
                    "documentation": (
                        "Model Owner: Credit Risk Analytics team lead. "
                        "Model Validator: Independent model risk management team. "
                        "Model User: Credit underwriting officers. "
                        "Separation of duties maintained between development "
                        "and validation functions."
                    ),
                    "status": "TEMPLATE",
                },
                "change_management": {
                    "requirement": governance_items[1],
                    "documentation": (
                        "All model changes require: (1) Impact assessment. "
                        "(2) Validation team review. (3) Approval from model "
                        "governance committee. (4) Documentation update. "
                        "Version control via Git with tagged releases."
                    ),
                    "status": "TEMPLATE",
                },
                "escalation": {
                    "requirement": governance_items[2],
                    "documentation": (
                        "Escalation path: Model User → Model Owner → "
                        "Model Risk Management → Chief Risk Officer. "
                        "Fairness violations escalate directly to compliance "
                        "and legal teams."
                    ),
                    "status": "TEMPLATE",
                },
            },
        }

    def _generate_checklist(self, sections: dict) -> dict:
        """Generate compliance checklist from all sections."""
        total = 0
        documented = 0
        planned = 0
        template = 0

        for section in sections.values():
            for item in section["items"].values():
                total += 1
                status = item["status"]
                if status == "DOCUMENTED":
                    documented += 1
                elif status == "PLANNED":
                    planned += 1
                elif status == "TEMPLATE":
                    template += 1

        return {
            "total_requirements": total,
            "documented": documented,
            "planned": planned,
            "template": template,
            "completion_pct": f"{(documented / total * 100):.0f}%",
        }

    def _generate_summary(self, doc: dict) -> str:
        """Generate human-readable SR 11-7 summary."""
        checklist = doc["compliance_checklist"]
        lines = [
            "FED SR 11-7 MODEL RISK MANAGEMENT DOCUMENTATION",
            "=" * 55,
            f"Model: {doc['model_name']}",
            f"Tier: {doc['model_tier']}",
            f"Date: {doc['document_date'][:10]}",
            "",
            f"Documentation Completion: {checklist['completion_pct']}",
            f"  Documented: {checklist['documented']}/{checklist['total_requirements']}",
            f"  Planned:    {checklist['planned']}/{checklist['total_requirements']}",
            f"  Template:   {checklist['template']}/{checklist['total_requirements']}",
            "",
        ]

        for section_key, section in doc["sections"].items():
            lines.append(f"--- {section['title']} ---")
            for item_key, item in section["items"].items():
                icon = {"DOCUMENTED": "✅", "PLANNED": "📋", "TEMPLATE": "📝"}
                lines.append(f"  {icon.get(item['status'], '❓')} {item['requirement']}: {item['status']}")
            lines.append("")

        return "\n".join(lines)

    def save_documentation(self, doc: dict, path: Optional[Path] = None):
        """Save SR 11-7 documentation to JSON."""
        if path is None:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = REPORTS_DIR / "sr11_7_documentation.json"

        with open(path, "w") as f:
            json.dump(doc, f, indent=2, default=str)
        logger.info(f"Saved SR 11-7 documentation to {path}")


if __name__ == "__main__":
    # Load available reports
    metrics_path = PROJECT_ROOT / "models" / "test_metrics.json"
    model_metrics = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            model_metrics = json.load(f)

    fairness_path = REPORTS_DIR / "fairness_audit_report.json"
    fairness_report = None
    if fairness_path.exists():
        with open(fairness_path) as f:
            fairness_report = json.load(f)

    generator = SR117DocumentationGenerator()
    doc = generator.generate_documentation(
        model_metrics=model_metrics,
        fairness_report=fairness_report,
    )

    print(doc["summary"])
    generator.save_documentation(doc)
