"""Orchestrator for all explainability methods.

Runs SHAP, LIME, and DiCE counterfactual explanations in one call
and returns a unified explanation report.
"""

import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.counterfactual import CounterfactualExplainer
from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT
from src.data.preprocess import get_feature_columns

logger = get_logger("explainability.orchestrator")

MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def load_best_model():
    """Load the best trained model."""
    best_name = (MODELS_DIR / "best_model.txt").read_text().strip()
    with open(MODELS_DIR / f"{best_name}.pkl", "rb") as f:
        model = pickle.load(f)
    return model, best_name


def load_train_data():
    """Load training data."""
    return pd.read_csv(SPLITS_DIR / "train.csv")


def get_categorical_feature_indices(feature_cols: list[str]) -> list[int]:
    """Return indices of categorical features."""
    categorical = {"SEX", "EDUCATION", "MARRIAGE"}
    return [i for i, col in enumerate(feature_cols) if col in categorical]


def run_all_explanations(
    instance: pd.DataFrame,
    model=None,
    train_df=None,
    run_shap: bool = True,
    run_lime: bool = True,
    run_counterfactual: bool = True,
    num_counterfactuals: int = 3,
) -> dict:
    """Run all explanation methods on a single applicant.

    Args:
        instance: Single-row DataFrame with feature values (no target).
        model: Trained model. Loaded automatically if None.
        train_df: Training data. Loaded automatically if None.
        run_shap: Whether to run SHAP explanation.
        run_lime: Whether to run LIME explanation.
        run_counterfactual: Whether to run DiCE counterfactual.
        num_counterfactuals: Number of counterfactuals to generate.

    Returns:
        dict with keys: prediction, shap, lime, counterfactual
    """
    if model is None:
        model, model_name = load_best_model()
    else:
        model_name = type(model).__name__

    if train_df is None:
        train_df = load_train_data()

    feature_cols = get_feature_columns(train_df)
    X_train = train_df[feature_cols]

    # Ensure instance has correct columns
    if "DEFAULT" in instance.columns:
        instance = instance.drop(columns=["DEFAULT"])
    instance = instance[feature_cols]

    # Prediction
    prob = model.predict_proba(instance)[0, 1]
    pred = int(prob >= 0.5)

    results = {
        "prediction": {
            "model": model_name,
            "default_probability": float(prob),
            "decision": "DENIED" if pred == 1 else "APPROVED",
            "confidence": float(abs(prob - 0.5) * 200),
        }
    }

    # SHAP
    if run_shap:
        logger.info("Running SHAP explanation...")
        shap_exp = SHAPExplainer(model, X_train, model_name=model_name)
        results["shap"] = shap_exp.local_explanation(instance)
        logger.info("SHAP complete.")

    # LIME
    if run_lime:
        logger.info("Running LIME explanation...")
        cat_indices = get_categorical_feature_indices(feature_cols)
        lime_exp = LIMEExplainer(
            model, X_train, feature_cols,
            categorical_features=cat_indices,
            model_name=model_name,
        )
        results["lime"] = lime_exp.explain_instance(instance)
        logger.info("LIME complete.")

    # Counterfactual
    if run_counterfactual:
        logger.info("Running counterfactual explanation...")
        # DiCE needs training data WITH target column
        continuous = [c for c in feature_cols if c not in {"SEX", "EDUCATION", "MARRIAGE"}]
        categorical = [c for c in feature_cols if c in {"SEX", "EDUCATION", "MARRIAGE"}]

        try:
            cf_exp = CounterfactualExplainer(
                model=model,
                X_train=train_df[feature_cols + ["DEFAULT"]],
                continuous_features=continuous,
                categorical_features=categorical,
                target_col="DEFAULT",
                model_name=model_name,
            )
            results["counterfactual"] = cf_exp.generate_counterfactuals(
                instance=instance,
                num_cfs=num_counterfactuals,
                desired_class=0,
            )
            logger.info("Counterfactual complete.")
        except Exception as e:
            logger.error(f"Counterfactual failed: {e}")
            results["counterfactual"] = {
                "status": "error",
                "message": str(e),
            }

    return results


def print_explanation_report(results: dict) -> None:
    """Print a human-readable explanation report."""
    pred = results["prediction"]
    print("\n" + "=" * 60)
    print("XAI CREDIT LENS — EXPLANATION REPORT")
    print("=" * 60)
    print(f"\nDecision: {pred['decision']}")
    print(f"Default Probability: {pred['default_probability']:.1%}")
    print(f"Confidence: {pred['confidence']:.0f}%")
    print(f"Model: {pred['model']}")

    if "shap" in results:
        shap_res = results["shap"]
        print("\n--- SHAP Explanation ---")
        print("Top Risk Factors:")
        for f in shap_res.get("top_risk_factors", [])[:3]:
            print(f"  • {f['feature']}: {f['feature_value']:.2f} (SHAP: +{f['shap_value']:.3f})")
        print("Top Protective Factors:")
        for f in shap_res.get("top_protective_factors", [])[:3]:
            print(f"  • {f['feature']}: {f['feature_value']:.2f} (SHAP: {f['shap_value']:.3f})")

    if "lime" in results:
        lime_res = results["lime"]
        print("\n--- LIME Explanation ---")
        print(f"Local model R²: {lime_res.get('r_squared', 'N/A')}")
        print("Top factors:")
        for f in lime_res.get("feature_explanations", [])[:5]:
            print(f"  • {f['rule']} (weight: {f['weight']:+.3f})")

    if "counterfactual" in results:
        cf_res = results["counterfactual"]
        print("\n--- Counterfactual Explanation ---")
        if cf_res.get("status") == "success":
            print(cf_res.get("summary", ""))
            for i, cf in enumerate(cf_res.get("counterfactuals", []), 1):
                print(f"\n  Scenario {i}: {cf['natural_language']}")
        else:
            print(f"  {cf_res.get('message', 'No counterfactuals generated.')}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Demo: explain a random test applicant
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "splits" / "test.csv")
    feature_cols = get_feature_columns(test_df)

    # Pick a defaulter to explain
    defaulters = test_df[test_df["DEFAULT"] == 1]
    sample = defaulters.sample(1, random_state=42)

    print(f"Explaining applicant (index {sample.index[0]})...")
    results = run_all_explanations(sample[feature_cols])
    print_explanation_report(results)
