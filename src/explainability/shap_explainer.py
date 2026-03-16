"""SHAP-based global and local explanations for credit decisions."""

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT

logger = get_logger("explainability.shap")

MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"


class SHAPExplainer:
    """Generate SHAP explanations for credit default models."""

    def __init__(self, model, X_background: pd.DataFrame, model_name: str = "model"):
        """
        Args:
            model: Trained classifier with predict_proba method.
            X_background: Background dataset for SHAP (typically training sample).
            model_name: Name for labeling outputs.
        """
        self.model = model
        self.model_name = model_name
        self.feature_names = list(X_background.columns)

        # Use TreeExplainer for tree-based models, KernelExplainer as fallback
        try:
            self.explainer = shap.TreeExplainer(model)
            self.explainer_type = "tree"
            logger.info(f"Using TreeExplainer for {model_name}")
        except Exception:
            background = shap.sample(X_background, min(100, len(X_background)))
            self.explainer = shap.KernelExplainer(model.predict_proba, background)
            self.explainer_type = "kernel"
            logger.info(f"Using KernelExplainer for {model_name}")

    def global_importance(self, X: pd.DataFrame, max_display: int = 15) -> pd.DataFrame:
        """Compute global feature importance via mean |SHAP values|."""
        shap_values = self._get_shap_values(X)

        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap", ascending=False)

        logger.info(f"Top 5 features: {list(importance.head(5)['feature'])}")
        return importance

    def local_explanation(self, instance: pd.DataFrame) -> dict:
        """Generate a local SHAP explanation for a single applicant.

        Returns:
            dict with keys: base_value, shap_values, feature_values, feature_names,
                            prediction_prob, top_positive, top_negative
        """
        if len(instance.shape) == 1:
            instance = instance.to_frame().T

        shap_values = self._get_shap_values(instance)
        # Ensure we have a 1D array for a single instance
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]
        shap_values = np.ravel(shap_values)

        base_value = self._get_base_value()
        prob = self.model.predict_proba(instance)[0, 1]

        # Identify top contributing features
        feature_values = np.ravel(instance.values[0])
        contributions = pd.DataFrame(
            {
                "feature": self.feature_names,
                "shap_value": shap_values,
                "feature_value": feature_values,
            }
        ).sort_values("shap_value", key=abs, ascending=False)

        top_positive = contributions[contributions["shap_value"] > 0].head(5)
        top_negative = contributions[contributions["shap_value"] < 0].head(5)

        return {
            "base_value": float(base_value),
            "shap_values": shap_values,
            "feature_values": instance.values[0],
            "feature_names": self.feature_names,
            "prediction_prob": float(prob),
            "top_risk_factors": top_positive.to_dict("records"),
            "top_protective_factors": top_negative.to_dict("records"),
            "all_contributions": contributions.to_dict("records"),
        }

    def plot_global_summary(self, X: pd.DataFrame, save_path: Optional[Path] = None):
        """Generate SHAP summary beeswarm plot."""
        shap_values = self._get_shap_values(X)

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        plt.title(f"SHAP Feature Importance — {self.model_name}", fontsize=14)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved global summary plot to {save_path}")
        plt.close()

    def plot_waterfall(self, instance: pd.DataFrame, save_path: Optional[Path] = None):
        """Generate SHAP waterfall plot for a single instance."""
        if len(instance.shape) == 1:
            instance = instance.to_frame().T

        shap_values = self._get_shap_values(instance)
        base_value = self._get_base_value()

        explanation = shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=instance.values[0],
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved waterfall plot to {save_path}")
        plt.close()

    def _get_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Extract SHAP values, handling different explainer types."""
        sv = self.explainer.shap_values(X)
        # For binary classification, TreeExplainer may return list of [class0, class1]
        if isinstance(sv, list):
            return np.array(sv[1])  # class 1 = default
        sv = np.array(sv)
        # Shape (samples, features, classes) — take class 1
        if sv.ndim == 3:
            return sv[:, :, 1]
        if hasattr(sv, "values"):
            return sv.values
        return sv

    def _get_base_value(self) -> float:
        """Extract base value from explainer."""
        bv = self.explainer.expected_value
        if isinstance(bv, (list, np.ndarray)):
            return float(bv[1]) if len(bv) > 1 else float(bv[0])
        return float(bv)
