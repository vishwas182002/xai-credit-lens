"""ECOA / Regulation B adverse action notice generator.

Maps model explanations to legally compliant adverse action reasons
as required by the Equal Credit Opportunity Act.
"""

from typing import Optional

import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import get_regulatory_config

logger = get_logger("regulatory.ecoa")


class ECOAAdverseActionGenerator:
    """Generate ECOA-compliant adverse action notices from model explanations."""

    def __init__(self, config: Optional[dict] = None):
        reg_config = config or get_regulatory_config()
        self.ecoa_config = reg_config["ecoa_reg_b"]
        self.max_reasons = self.ecoa_config["adverse_action"]["max_reasons"]
        self.reason_mapping = self.ecoa_config["adverse_action"]["reason_mapping"]
        self.prohibited_bases = self.ecoa_config["prohibited_bases"]

    def generate_notice(
        self,
        shap_explanation: dict,
        prediction: int,
        probability: float,
    ) -> dict:
        """Generate an adverse action notice from SHAP explanation.

        Args:
            shap_explanation: Output from SHAPExplainer.local_explanation()
            prediction: Binary prediction (1 = default/denied)
            probability: Default probability

        Returns:
            Structured adverse action notice with reasons and compliance flags.
        """
        if prediction == 0:
            return {
                "action": "APPROVED",
                "notice_required": False,
                "message": "Application approved. No adverse action notice required.",
            }

        # Get top contributing features driving the denial
        risk_factors = shap_explanation.get("top_risk_factors", [])

        # Filter out prohibited bases and map to ECOA-compliant reasons
        compliant_reasons = []
        prohibited_used = []

        for factor in risk_factors:
            feature = factor["feature"]

            # Check if this feature is a prohibited basis
            if feature.lower() in [p.lower() for p in self.prohibited_bases]:
                prohibited_used.append(feature)
                continue

            # Map to ECOA reason text
            reason_text = self.reason_mapping.get(feature)
            if reason_text and len(compliant_reasons) < self.max_reasons:
                compliant_reasons.append(
                    {
                        "reason_code": feature,
                        "reason_text": reason_text,
                        "impact_score": float(factor["shap_value"]),
                    }
                )

        # Build the notice
        notice = {
            "action": "DENIED",
            "notice_required": True,
            "default_probability": float(probability),
            "adverse_action_reasons": compliant_reasons,
            "num_reasons": len(compliant_reasons),
            "compliance_flags": {
                "max_reasons_respected": len(compliant_reasons) <= self.max_reasons,
                "prohibited_basis_excluded": len(prohibited_used) == 0,
                "prohibited_features_detected": prohibited_used,
            },
            "notice_text": self._format_notice_text(compliant_reasons),
            "regulatory_reference": (
                "Equal Credit Opportunity Act (15 U.S.C. §1691) and "
                "Regulation B (12 CFR Part 1002)"
            ),
        }

        if prohibited_used:
            logger.warning(
                f"Prohibited basis features detected in top factors: {prohibited_used}. "
                "These were excluded from the adverse action notice but may indicate "
                "proxy discrimination that requires investigation."
            )

        return notice

    def _format_notice_text(self, reasons: list[dict]) -> str:
        """Format reasons into a consumer-facing adverse action notice."""
        if not reasons:
            return "We were unable to approve your application at this time."

        lines = [
            "NOTICE OF ADVERSE ACTION",
            "",
            "Your application for credit was not approved.",
            "The principal reason(s) for this decision are:",
            "",
        ]
        for i, reason in enumerate(reasons, 1):
            lines.append(f"  {i}. {reason['reason_text']}")

        lines.extend(
            [
                "",
                "Under the Equal Credit Opportunity Act, you have the right to know "
                "why your application was denied. If you have questions, please contact "
                "our credit department.",
                "",
                "You have the right to obtain a free copy of your credit report within "
                "60 days from the consumer reporting agency used in this decision.",
            ]
        )
        return "\n".join(lines)
