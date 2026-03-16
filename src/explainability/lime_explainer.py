"""LIME-based instance-level explanations for credit decisions."""

from typing import Optional

import numpy as np
import pandas as pd
import lime.lime_tabular
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger("explainability.lime")


class LIMEExplainer:
    """Generate LIME explanations for individual credit decisions."""

    def __init__(
        self,
        model,
        X_train: pd.DataFrame,
        feature_names: list[str],
        categorical_features: Optional[list[int]] = None,
        model_name: str = "model",
    ):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []

        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            categorical_features=self.categorical_features,
            class_names=["No Default", "Default"],
            mode="classification",
            discretize_continuous=True,
        )
        logger.info(f"LIME explainer initialized for {model_name}")

    def explain_instance(
        self,
        instance: pd.DataFrame,
        num_features: int = 10,
    ) -> dict:
        """Generate LIME explanation for a single applicant.

        Returns:
            dict with keys: prediction_prob, intercept, feature_explanations, top_reasons
        """
        if len(instance.shape) == 2:
            instance_arr = instance.values[0]
        else:
            instance_arr = instance.values

        explanation = self.explainer.explain_instance(
            instance_arr,
            self.model.predict_proba,
            num_features=num_features,
            top_labels=2,
        )

        # Extract feature contributions for default class (class 1)
        # Fall back to whatever label is available
        available_labels = explanation.available_labels()
        label = 1 if 1 in available_labels else available_labels[0]
        feature_weights = explanation.as_list(label=label)

        prob = self.model.predict_proba(instance_arr.reshape(1, -1))[0]

        return {
            "prediction_prob": {
                "no_default": float(prob[0]),
                "default": float(prob[1]),
            },
            "intercept": float(explanation.intercept[label]),
            "feature_explanations": [
                {"rule": feat, "weight": float(weight)}
                for feat, weight in feature_weights
            ],
            "top_risk_factors": [
                {"rule": feat, "weight": float(weight)}
                for feat, weight in feature_weights
                if weight > 0
            ],
            "top_protective_factors": [
                {"rule": feat, "weight": float(weight)}
                for feat, weight in feature_weights
                if weight < 0
            ],
            "local_prediction": float(explanation.local_pred[0])
            if hasattr(explanation, "local_pred")
            else None,
            "r_squared": float(explanation.score) if hasattr(explanation, "score") else None,
        }

    def plot_explanation(
        self,
        instance: pd.DataFrame,
        num_features: int = 10,
        save_path=None,
    ):
        """Plot LIME explanation as horizontal bar chart."""
        if len(instance.shape) == 2:
            instance_arr = instance.values[0]
        else:
            instance_arr = instance.values

        explanation = self.explainer.explain_instance(
            instance_arr,
            self.model.predict_proba,
            num_features=num_features,
        )

        fig = explanation.as_pyplot_figure(label=1)
        fig.set_size_inches(10, 6)
        plt.title(f"LIME Explanation — {self.model_name}", fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved LIME plot to {save_path}")
        plt.close()
