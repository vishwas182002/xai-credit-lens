"""Fairness metrics and automated bias audit for credit decisions."""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.logger import get_logger
from src.utils.config import get_fairness_config

logger = get_logger("fairness.metrics")


class FairnessAuditor:
    """Automated fairness audit across protected demographic groups."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or get_fairness_config()
        self.protected_attrs = self.config["protected_attributes"]
        self.metric_thresholds = self.config["metrics"]

    def run_full_audit(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        sensitive_features: pd.DataFrame,
    ) -> dict:
        """Run complete fairness audit across all protected attributes.

        Args:
            y_true: Ground truth labels.
            y_pred: Binary predictions.
            y_prob: Predicted probabilities.
            sensitive_features: DataFrame with protected attribute columns.

        Returns:
            Comprehensive audit report with metrics, flags, and recommendations.
        """
        audit_results = {}

        for attr_key, attr_config in self.protected_attrs.items():
            col = attr_config["column"]
            if col not in sensitive_features.columns:
                logger.warning(f"Protected attribute '{col}' not found in data. Skipping.")
                continue

            groups = sensitive_features[col].values

            # Define privileged/unprivileged masks
            priv_mask, unpriv_mask = self._get_group_masks(groups, attr_config)

            if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
                logger.warning(f"Empty group for '{attr_key}'. Skipping.")
                continue

            metrics = self._compute_all_metrics(
                y_true, y_pred, y_prob, priv_mask, unpriv_mask
            )

            # Check against thresholds and flag violations
            flags = self._check_thresholds(metrics)

            audit_results[attr_key] = {
                "label": attr_config["label"],
                "privileged_count": int(priv_mask.sum()),
                "unprivileged_count": int(unpriv_mask.sum()),
                "metrics": metrics,
                "flags": flags,
                "has_violations": any(flags.values()),
                "recommendations": self._generate_recommendations(attr_key, metrics, flags),
            }

        # Overall summary
        total_violations = sum(
            r["has_violations"] for r in audit_results.values()
        )

        return {
            "audit_results": audit_results,
            "overall_status": "FAIL" if total_violations > 0 else "PASS",
            "total_attributes_audited": len(audit_results),
            "attributes_with_violations": total_violations,
            "summary": self._generate_summary(audit_results),
        }

    def _get_group_masks(self, groups, attr_config) -> tuple[np.ndarray, np.ndarray]:
        """Create boolean masks for privileged and unprivileged groups."""
        priv = attr_config.get("privileged")
        unpriv = attr_config.get("unprivileged")

        if isinstance(priv, list):
            priv_mask = np.isin(groups, priv)
        elif isinstance(priv, (int, float)):
            priv_mask = groups == priv
        else:
            # Handle condition-based (e.g., age >= 30)
            cond = attr_config.get("privileged_condition", "")
            if ">=" in cond:
                threshold = float(cond.split(">=")[1].strip())
                priv_mask = groups >= threshold
            elif "<" in cond:
                threshold = float(cond.split("<")[1].strip())
                priv_mask = groups < threshold
            else:
                priv_mask = np.ones(len(groups), dtype=bool)

        if isinstance(unpriv, list):
            unpriv_mask = np.isin(groups, unpriv)
        elif isinstance(unpriv, (int, float)):
            unpriv_mask = groups == unpriv
        else:
            cond = attr_config.get("unprivileged_condition", "")
            if ">=" in cond:
                threshold = float(cond.split(">=")[1].strip())
                unpriv_mask = groups >= threshold
            elif "<" in cond:
                threshold = float(cond.split("<")[1].strip())
                unpriv_mask = groups < threshold
            else:
                unpriv_mask = ~priv_mask

        return priv_mask, unpriv_mask

    def _compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        priv_mask: np.ndarray,
        unpriv_mask: np.ndarray,
    ) -> dict:
        """Compute all fairness metrics between privileged and unprivileged groups."""
        favorable = 0  # Not defaulting is favorable

        # Selection rates (rate of favorable outcome)
        priv_favorable_rate = (y_pred[priv_mask] == favorable).mean()
        unpriv_favorable_rate = (y_pred[unpriv_mask] == favorable).mean()

        # Disparate impact
        di = (
            unpriv_favorable_rate / priv_favorable_rate
            if priv_favorable_rate > 0
            else 0
        )

        # Statistical parity difference
        spd = unpriv_favorable_rate - priv_favorable_rate

        # True positive rates (for favorable outcome = 0, this is TNR)
        priv_tpr = self._true_positive_rate(y_true[priv_mask], y_pred[priv_mask])
        unpriv_tpr = self._true_positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask])
        eod_tpr = unpriv_tpr - priv_tpr

        # False positive rates
        priv_fpr = self._false_positive_rate(y_true[priv_mask], y_pred[priv_mask])
        unpriv_fpr = self._false_positive_rate(y_true[unpriv_mask], y_pred[unpriv_mask])
        eod_fpr = unpriv_fpr - priv_fpr

        # Equalized odds (max of TPR and FPR differences)
        equalized_odds_diff = max(abs(eod_tpr), abs(eod_fpr))

        # Predictive parity (PPV difference)
        priv_ppv = self._positive_predictive_value(y_true[priv_mask], y_pred[priv_mask])
        unpriv_ppv = self._positive_predictive_value(y_true[unpriv_mask], y_pred[unpriv_mask])
        ppd = unpriv_ppv - priv_ppv

        # Average predicted probability by group
        priv_avg_prob = y_prob[priv_mask].mean()
        unpriv_avg_prob = y_prob[unpriv_mask].mean()

        return {
            "disparate_impact": float(di),
            "statistical_parity_difference": float(spd),
            "equal_opportunity_difference": float(eod_tpr),
            "equalized_odds_difference": float(equalized_odds_diff),
            "predictive_parity_difference": float(ppd),
            "privileged_favorable_rate": float(priv_favorable_rate),
            "unprivileged_favorable_rate": float(unpriv_favorable_rate),
            "privileged_avg_prob": float(priv_avg_prob),
            "unprivileged_avg_prob": float(unpriv_avg_prob),
        }

    def _check_thresholds(self, metrics: dict) -> dict:
        """Check which metrics violate their configured thresholds."""
        flags = {}
        for metric_key, config in self.metric_thresholds.items():
            threshold = config["threshold"]
            value = metrics.get(metric_key)
            if value is None:
                continue

            if metric_key == "disparate_impact":
                flags[metric_key] = value < threshold  # Below 0.8 = violation
            else:
                flags[metric_key] = abs(value) > threshold  # Above threshold = violation

        return flags

    def _generate_recommendations(self, attr_key: str, metrics: dict, flags: dict) -> list[str]:
        """Generate actionable recommendations based on violations."""
        recs = []
        if flags.get("disparate_impact"):
            recs.append(
                f"Disparate impact ratio ({metrics['disparate_impact']:.2f}) is below "
                f"the four-fifths threshold. Consider reweighting training data or "
                f"applying post-processing calibration across groups."
            )
        if flags.get("equal_opportunity_difference"):
            recs.append(
                "Significant difference in true positive rates between groups. "
                "Consider threshold adjustment or equalized odds constraints during training."
            )
        if flags.get("statistical_parity_difference"):
            recs.append(
                "Selection rates differ significantly between groups. "
                "Review feature engineering for proxy discrimination."
            )
        if not recs:
            recs.append("No significant fairness violations detected for this attribute.")
        return recs

    def _generate_summary(self, audit_results: dict) -> str:
        """Generate human-readable audit summary."""
        lines = ["=" * 60, "FAIRNESS AUDIT SUMMARY", "=" * 60]
        for attr_key, result in audit_results.items():
            status = "FAIL" if result["has_violations"] else "PASS"
            lines.append(f"\n{result['label']}: [{status}]")
            lines.append(
                f"  Groups: {result['privileged_count']} privileged, "
                f"{result['unprivileged_count']} unprivileged"
            )
            di = result["metrics"]["disparate_impact"]
            lines.append(f"  Disparate Impact: {di:.3f} {'< 0.80' if di < 0.8 else '>= 0.80'}")

            if result["has_violations"]:
                violated = [k for k, v in result["flags"].items() if v]
                lines.append(f"  Violated metrics: {', '.join(violated)}")

        return "\n".join(lines)

    @staticmethod
    def _true_positive_rate(y_true, y_pred, positive_label=1):
        mask = y_true == positive_label
        if mask.sum() == 0:
            return 0.0
        return (y_pred[mask] == positive_label).mean()

    @staticmethod
    def _false_positive_rate(y_true, y_pred, positive_label=1):
        mask = y_true != positive_label
        if mask.sum() == 0:
            return 0.0
        return (y_pred[mask] == positive_label).mean()

    @staticmethod
    def _positive_predictive_value(y_true, y_pred, positive_label=1):
        mask = y_pred == positive_label
        if mask.sum() == 0:
            return 0.0
        return (y_true[mask] == positive_label).mean()
