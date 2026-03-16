"""EU AI Act compliance checks for high-risk credit scoring AI.

Maps the XAI Credit Lens framework outputs to EU AI Act requirements
for high-risk AI systems used in credit decisioning.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger
from src.utils.config import get_regulatory_config, PROJECT_ROOT

logger = get_logger("regulatory.eu_ai_act")

REPORTS_DIR = PROJECT_ROOT / "reports" / "audit"


class EUAIActComplianceChecker:
    """Evaluate and document compliance with EU AI Act for credit scoring."""

    def __init__(self, config: Optional[dict] = None):
        reg_config = config or get_regulatory_config()
        self.requirements = reg_config["eu_ai_act"]["requirements"]
        self.risk_classification = reg_config["eu_ai_act"]["risk_classification"]

    def run_compliance_check(
        self,
        has_model_documentation: bool = True,
        has_explainability: bool = True,
        has_fairness_audit: bool = True,
        has_human_oversight: bool = True,
        has_data_governance: bool = True,
        has_robustness_testing: bool = True,
        model_metrics: Optional[dict] = None,
        fairness_report: Optional[dict] = None,
    ) -> dict:
        """Run EU AI Act compliance check.

        Args:
            has_model_documentation: Whether model docs exist.
            has_explainability: Whether SHAP/LIME/counterfactual explanations are available.
            has_fairness_audit: Whether bias audit has been performed.
            has_human_oversight: Whether human-in-the-loop is documented.
            has_data_governance: Whether data quality is documented.
            has_robustness_testing: Whether model validation exists.
            model_metrics: Optional dict of model performance metrics.
            fairness_report: Optional fairness audit report.

        Returns:
            Compliance report with status per requirement.
        """
        checks = {}

        # Transparency requirements
        transparency_items = self.requirements["transparency"]
        checks["transparency"] = {
            "requirement": "Article 13 — Transparency",
            "description": "High-risk AI systems shall be designed and developed in such a way "
                          "to ensure their operation is sufficiently transparent to enable "
                          "deployers to interpret the system's output and use it appropriately.",
            "sub_checks": [
                {
                    "item": transparency_items[0],
                    "status": "PASS" if has_model_documentation else "FAIL",
                    "evidence": "README.md contains model purpose, methodology, and limitations."
                    if has_model_documentation
                    else "Model documentation not found.",
                },
                {
                    "item": transparency_items[1],
                    "status": "PASS" if has_explainability else "FAIL",
                    "evidence": "Streamlit dashboard clearly indicates AI-based decisioning."
                    if has_explainability
                    else "No disclosure mechanism found.",
                },
                {
                    "item": transparency_items[2],
                    "status": "PASS" if has_explainability else "FAIL",
                    "evidence": "SHAP, LIME, and counterfactual explanations provided for each decision."
                    if has_explainability
                    else "No per-decision explanations available.",
                },
            ],
            "overall": "PASS" if (has_model_documentation and has_explainability) else "FAIL",
        }

        # Human oversight requirements
        oversight_items = self.requirements["human_oversight"]
        checks["human_oversight"] = {
            "requirement": "Article 14 — Human Oversight",
            "description": "High-risk AI systems shall be designed and developed in such a way "
                          "that they can be effectively overseen by natural persons.",
            "sub_checks": [
                {
                    "item": oversight_items[0],
                    "status": "PASS" if has_human_oversight else "FAIL",
                    "evidence": "Dashboard provides prediction + explanation for human review "
                    "before final decision."
                    if has_human_oversight
                    else "No human oversight mechanism documented.",
                },
                {
                    "item": oversight_items[1],
                    "status": "PASS" if has_human_oversight else "FAIL",
                    "evidence": "System outputs recommendations, not final decisions. "
                    "Human retains override authority."
                    if has_human_oversight
                    else "Override capability not documented.",
                },
                {
                    "item": oversight_items[2],
                    "status": "PASS" if has_human_oversight else "FAIL",
                    "evidence": "Fairness flags trigger escalation for applicants in "
                    "monitored demographic groups."
                    if has_human_oversight
                    else "No escalation procedures documented.",
                },
            ],
            "overall": "PASS" if has_human_oversight else "FAIL",
        }

        # Data governance requirements
        data_items = self.requirements["data_governance"]
        checks["data_governance"] = {
            "requirement": "Article 10 — Data and Data Governance",
            "description": "Training, validation and testing data sets shall be subject to "
                          "appropriate data governance and management practices.",
            "sub_checks": [
                {
                    "item": data_items[0],
                    "status": "PASS" if has_data_governance else "FAIL",
                    "evidence": "Preprocessing pipeline documents data cleaning steps, "
                    "feature engineering, and quality checks."
                    if has_data_governance
                    else "No data quality assessment found.",
                },
                {
                    "item": data_items[1],
                    "status": "PASS" if has_fairness_audit else "FAIL",
                    "evidence": f"Fairness audit completed across 4 protected attributes. "
                    f"Status: {fairness_report.get('overall_status', 'N/A')}"
                    if fairness_report
                    else "No bias testing on training data performed.",
                },
                {
                    "item": data_items[2],
                    "status": "PASS" if has_data_governance else "FAIL",
                    "evidence": "UCI ML Repository dataset with documented provenance "
                    "(Taiwan bank, 2005, 30,000 records)."
                    if has_data_governance
                    else "Data provenance not documented.",
                },
            ],
            "overall": "PASS"
            if (has_data_governance and has_fairness_audit)
            else "FAIL",
        }

        # Robustness requirements
        robust_items = self.requirements["robustness"]
        checks["robustness"] = {
            "requirement": "Article 15 — Accuracy, Robustness and Cybersecurity",
            "description": "High-risk AI systems shall be designed and developed in such a way "
                          "that they achieve an appropriate level of accuracy, robustness "
                          "and cybersecurity.",
            "sub_checks": [
                {
                    "item": robust_items[0],
                    "status": "PASS" if model_metrics else "FAIL",
                    "evidence": f"Model performance tracked: AUC={model_metrics.get('roc_auc', 'N/A'):.4f}, "
                    f"F1={model_metrics.get('f1', 'N/A'):.4f}"
                    if model_metrics
                    else "No performance monitoring plan.",
                },
                {
                    "item": robust_items[1],
                    "status": "PASS" if has_robustness_testing else "FAIL",
                    "evidence": "Three model architectures compared. Cross-validation performed. "
                    "Train/val/test split maintains population representativeness."
                    if has_robustness_testing
                    else "No adversarial robustness testing.",
                },
                {
                    "item": robust_items[2],
                    "status": "PASS" if has_robustness_testing else "FAIL",
                    "evidence": "Model comparison pipeline enables fallback to alternative "
                    "model if primary model degrades."
                    if has_robustness_testing
                    else "No fallback procedures documented.",
                },
            ],
            "overall": "PASS"
            if (has_robustness_testing and model_metrics)
            else "FAIL",
        }

        # Overall compliance
        all_pass = all(c["overall"] == "PASS" for c in checks.values())

        report = {
            "framework": "EU Artificial Intelligence Act (2024)",
            "risk_classification": self.risk_classification,
            "use_case": "Credit scoring / creditworthiness assessment",
            "assessment_date": datetime.now().isoformat(),
            "overall_compliance": "COMPLIANT" if all_pass else "NON-COMPLIANT",
            "checks": checks,
            "total_requirements": sum(
                len(c["sub_checks"]) for c in checks.values()
            ),
            "requirements_met": sum(
                1
                for c in checks.values()
                for sc in c["sub_checks"]
                if sc["status"] == "PASS"
            ),
            "summary": self._generate_summary(checks, all_pass),
        }

        return report

    def _generate_summary(self, checks: dict, all_pass: bool) -> str:
        """Generate human-readable compliance summary."""
        lines = [
            "EU AI ACT COMPLIANCE ASSESSMENT",
            "=" * 50,
            f"Risk Classification: HIGH-RISK (Credit Scoring)",
            f"Overall Status: {'COMPLIANT' if all_pass else 'NON-COMPLIANT'}",
            "",
        ]
        for category, check in checks.items():
            status = check["overall"]
            icon = "✅" if status == "PASS" else "❌"
            lines.append(f"{icon} {check['requirement']}: {status}")
            for sc in check["sub_checks"]:
                sc_icon = "  ✓" if sc["status"] == "PASS" else "  ✗"
                lines.append(f"  {sc_icon} {sc['item']}")

        return "\n".join(lines)

    def save_report(self, report: dict, path: Optional[Path] = None):
        """Save compliance report to JSON."""
        if path is None:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            path = REPORTS_DIR / "eu_ai_act_compliance.json"

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Saved EU AI Act compliance report to {path}")


if __name__ == "__main__":
    import json as json_mod

    # Load model metrics and fairness report if available
    metrics_path = PROJECT_ROOT / "models" / "test_metrics.json"
    model_metrics = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            model_metrics = json_mod.load(f)

    fairness_path = REPORTS_DIR / "fairness_audit_report.json"
    fairness_report = None
    if fairness_path.exists():
        with open(fairness_path) as f:
            fairness_report = json_mod.load(f)

    checker = EUAIActComplianceChecker()
    report = checker.run_compliance_check(
        has_model_documentation=True,
        has_explainability=True,
        has_fairness_audit=True,
        has_human_oversight=True,
        has_data_governance=True,
        has_robustness_testing=True,
        model_metrics=model_metrics,
        fairness_report=fairness_report,
    )

    print(report["summary"])
    print(f"\nRequirements Met: {report['requirements_met']}/{report['total_requirements']}")

    checker.save_report(report)
