"""Counterfactual explanations for actionable credit recourse.

This is the key differentiator of this project. Instead of just saying
"you were denied," we tell applicants exactly what changes would flip
the decision — grounded in DiCE (Diverse Counterfactual Explanations).
"""

from typing import Optional

import numpy as np
import pandas as pd
import dice_ml

from src.utils.logger import get_logger

logger = get_logger("explainability.counterfactual")

# Human-readable feature labels for output
FEATURE_LABELS = {
    "LIMIT_BAL": "credit limit",
    "AGE": "age",
    "BILL_AMT1": "most recent bill amount",
    "PAY_AMT1": "most recent payment amount",
    "PAY_0": "most recent payment status",
    "DEBT_TO_INCOME_PROXY": "debt-to-income ratio",
    "PAYMENT_RATIO": "payment-to-bill ratio",
    "UTILIZATION_RATE": "credit utilization rate",
    "MONTHS_DELINQUENT": "months with late payments",
    "AVG_PAYMENT_DELAY": "average payment delay",
    "PAYMENT_TREND": "payment trend",
    "MAX_CONSEC_DELINQUENT": "max consecutive late payments",
    "BALANCE_VOLATILITY": "balance volatility",
}

# Features that the applicant can realistically change
ACTIONABLE_FEATURES = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "BILL_AMT1",
    "PAYMENT_RATIO",
    "DEBT_TO_INCOME_PROXY",
    "UTILIZATION_RATE",
    "MONTHS_DELINQUENT",
    "AVG_PAYMENT_DELAY",
    "PAYMENT_TREND",
]

# Features that cannot be changed (immutable)
IMMUTABLE_FEATURES = ["SEX", "EDUCATION", "MARRIAGE", "AGE"]


class CounterfactualExplainer:
    """Generate actionable counterfactual explanations using DiCE."""

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        continuous_features: list[str],
        categorical_features: Optional[list[str]] = None,
        target_col: str = "DEFAULT",
        model_name: str = "model",
    ):
        self.model = model
        self.model_name = model_name
        self.target_col = target_col
        self.feature_names = list(X_train.columns)
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features or []

        # Build DiCE data object
        # DiCE needs the full dataframe including target for its data interface
        self.dice_data = dice_ml.Data(
            dataframe=X_train,
            continuous_features=continuous_features,
            outcome_name=target_col,
        )

        # Build DiCE model object
        self.dice_model = dice_ml.Model(
            model=model,
            backend="sklearn",
            model_type="classifier",
        )

        # Create DiCE explainer
        self.dice_exp = dice_ml.Dice(
            self.dice_data,
            self.dice_model,
            method="random",  # 'random' is most robust; 'genetic' for diversity
        )

        logger.info(f"Counterfactual explainer initialized for {model_name}")

    def generate_counterfactuals(
        self,
        instance: pd.DataFrame,
        num_cfs: int = 3,
        desired_class: int = 0,  # 0 = no default (approved)
        features_to_vary: Optional[list[str]] = None,
    ) -> dict:
        """Generate counterfactual explanations for a single applicant.

        Args:
            instance: Single applicant profile (1-row DataFrame, no target column).
            num_cfs: Number of diverse counterfactuals to generate.
            desired_class: Target outcome (0 = approved).
            features_to_vary: Which features can change. Defaults to ACTIONABLE_FEATURES.

        Returns:
            dict with counterfactuals, changes needed, and natural language summary.
        """
        if features_to_vary is None:
            features_to_vary = [f for f in ACTIONABLE_FEATURES if f in self.feature_names]

        # Ensure instance is single row without target
        if self.target_col in instance.columns:
            instance = instance.drop(columns=[self.target_col])
        if len(instance) > 1:
            instance = instance.iloc[[0]]

        original_prob = self.model.predict_proba(instance)[0, 1]
        original_pred = int(original_prob >= 0.5)

        try:
            dice_result = self.dice_exp.generate_counterfactuals(
                query_instances=instance,
                total_CFs=num_cfs,
                desired_class=desired_class,
                features_to_vary=features_to_vary,
                permitted_range=None,
            )

            cf_df = dice_result.cf_examples_list[0].final_cfs_df

            if cf_df is None or len(cf_df) == 0:
                return {
                    "status": "no_counterfactuals_found",
                    "original_prediction": original_pred,
                    "original_default_prob": float(original_prob),
                    "message": "No actionable changes found to flip the decision.",
                }

            # Remove target column from counterfactuals if present
            if self.target_col in cf_df.columns:
                cf_df = cf_df.drop(columns=[self.target_col])

            # Compute changes for each counterfactual
            counterfactuals = []
            for idx in range(len(cf_df)):
                cf_row = cf_df.iloc[idx]
                changes = self._compute_changes(instance.iloc[0], cf_row)
                cf_prob = self.model.predict_proba(cf_df.iloc[[idx]])[0, 1]

                counterfactuals.append(
                    {
                        "counterfactual_values": cf_row.to_dict(),
                        "changes": changes,
                        "new_default_prob": float(cf_prob),
                        "natural_language": self._changes_to_text(changes, cf_prob),
                    }
                )

            return {
                "status": "success",
                "original_prediction": original_pred,
                "original_default_prob": float(original_prob),
                "num_counterfactuals": len(counterfactuals),
                "counterfactuals": counterfactuals,
                "summary": self._generate_summary(counterfactuals),
            }

        except Exception as e:
            logger.error(f"Counterfactual generation failed: {e}")
            return {
                "status": "error",
                "original_prediction": original_pred,
                "original_default_prob": float(original_prob),
                "message": f"Could not generate counterfactuals: {str(e)}",
            }

    def _compute_changes(self, original: pd.Series, counterfactual: pd.Series) -> list[dict]:
        """Compute the differences between original and counterfactual."""
        changes = []
        for feat in self.feature_names:
            if feat == self.target_col:
                continue
            orig_val = original[feat]
            cf_val = counterfactual[feat]

            if not np.isclose(orig_val, cf_val, rtol=1e-3):
                change = {
                    "feature": feat,
                    "label": FEATURE_LABELS.get(feat, feat),
                    "original": float(orig_val),
                    "counterfactual": float(cf_val),
                    "direction": "increase" if cf_val > orig_val else "decrease",
                    "magnitude": float(abs(cf_val - orig_val)),
                    "pct_change": float(abs(cf_val - orig_val) / max(abs(orig_val), 1e-6) * 100),
                }
                changes.append(change)

        # Sort by magnitude of percentage change
        changes.sort(key=lambda x: x["pct_change"], reverse=True)
        return changes

    def _changes_to_text(self, changes: list[dict], new_prob: float) -> str:
        """Convert changes to a human-readable counterfactual statement."""
        if not changes:
            return "No changes needed."

        parts = []
        for c in changes[:4]:  # Top 4 most impactful changes
            label = c["label"]
            direction = c["direction"]

            if c["pct_change"] < 100:
                parts.append(f"your {label} {direction}d by {c['pct_change']:.0f}%")
            else:
                parts.append(
                    f"your {label} changed from {c['original']:.0f} to {c['counterfactual']:.0f}"
                )

        changes_text = ", ".join(parts[:-1])
        if len(parts) > 1:
            changes_text += f", and {parts[-1]}"
        else:
            changes_text = parts[0]

        return (
            f"If {changes_text}, your estimated default probability would drop "
            f"to {new_prob:.1%} (likely approved)."
        )

    def _generate_summary(self, counterfactuals: list[dict]) -> str:
        """Generate an overall summary of the most common changes across counterfactuals."""
        all_features_changed = {}
        for cf in counterfactuals:
            for change in cf["changes"]:
                feat = change["feature"]
                if feat not in all_features_changed:
                    all_features_changed[feat] = {
                        "label": change["label"],
                        "count": 0,
                        "avg_pct": 0,
                        "direction": change["direction"],
                    }
                all_features_changed[feat]["count"] += 1
                all_features_changed[feat]["avg_pct"] += change["pct_change"]

        # Average percentages
        for feat in all_features_changed:
            count = all_features_changed[feat]["count"]
            all_features_changed[feat]["avg_pct"] /= count

        # Sort by frequency then magnitude
        ranked = sorted(
            all_features_changed.items(),
            key=lambda x: (x[1]["count"], x[1]["avg_pct"]),
            reverse=True,
        )

        summary_parts = []
        for feat, info in ranked[:3]:
            summary_parts.append(
                f"{info['direction'].capitalize()} {info['label']} "
                f"(appeared in {info['count']}/{len(counterfactuals)} scenarios, "
                f"~{info['avg_pct']:.0f}% change)"
            )

        return "Most impactful changes to improve approval odds:\n" + "\n".join(
            f"  {i+1}. {part}" for i, part in enumerate(summary_parts)
        )
