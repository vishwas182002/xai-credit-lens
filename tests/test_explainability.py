"""Tests for explainability modules (SHAP, LIME, Counterfactual)."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.explainability.shap_explainer import SHAPExplainer
from src.explainability.lime_explainer import LIMEExplainer
from src.data.preprocess import get_feature_columns
from src.utils.config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


@pytest.fixture
def model_and_data():
    """Load trained model and test data."""
    best_path = MODELS_DIR / "best_model.txt"
    if not best_path.exists():
        pytest.skip("Models not trained yet.")

    name = best_path.read_text().strip()
    with open(MODELS_DIR / f"{name}.pkl", "rb") as f:
        model = pickle.load(f)

    train = pd.read_csv(SPLITS_DIR / "train.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    feature_cols = get_feature_columns(train)

    return model, name, train[feature_cols], test[feature_cols]


class TestSHAPExplainer:
    def test_initialization(self, model_and_data):
        model, name, X_train, X_test = model_and_data
        explainer = SHAPExplainer(model, X_train, model_name=name)
        assert explainer.explainer is not None
        assert len(explainer.feature_names) == X_train.shape[1]

    def test_local_explanation_keys(self, model_and_data):
        model, name, X_train, X_test = model_and_data
        explainer = SHAPExplainer(model, X_train, model_name=name)
        result = explainer.local_explanation(X_test.iloc[[0]])

        expected_keys = {
            "base_value", "shap_values", "feature_values", "feature_names",
            "prediction_prob", "top_risk_factors", "top_protective_factors",
            "all_contributions",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_shap_values_shape(self, model_and_data):
        model, name, X_train, X_test = model_and_data
        explainer = SHAPExplainer(model, X_train, model_name=name)
        result = explainer.local_explanation(X_test.iloc[[0]])

        assert len(result["shap_values"]) == X_train.shape[1]

    def test_prediction_prob_range(self, model_and_data):
        model, name, X_train, X_test = model_and_data
        explainer = SHAPExplainer(model, X_train, model_name=name)
        result = explainer.local_explanation(X_test.iloc[[0]])

        assert 0 <= result["prediction_prob"] <= 1

    def test_global_importance(self, model_and_data):
        model, name, X_train, X_test = model_and_data
        explainer = SHAPExplainer(model, X_train, model_name=name)
        importance = explainer.global_importance(X_test.head(50))

        assert "feature" in importance.columns
        assert "mean_abs_shap" in importance.columns
        assert len(importance) == X_train.shape[1]
        assert importance["mean_abs_shap"].min() >= 0


class TestLIMEExplainer:
    def test_initialization(self, model_and_data):
        model, name, X_train, X_test = model_and_data
        feature_cols = list(X_train.columns)
        explainer = LIMEExplainer(model, X_train, feature_cols, model_name=name)
        assert explainer.explainer is not None

    def test_explanation_keys(self, model_and_data):
        model, name, X_train, X_test = model_and_data
        feature_cols = list(X_train.columns)
        explainer = LIMEExplainer(model, X_train, feature_cols, model_name=name)
        result = explainer.explain_instance(X_test.iloc[[0]])

        assert "prediction_prob" in result
        assert "feature_explanations" in result
        assert "top_risk_factors" in result
        assert "top_protective_factors" in result

    def test_prediction_probs_sum_to_one(self, model_and_data):
        model, name, X_train, X_test = model_and_data
        feature_cols = list(X_train.columns)
        explainer = LIMEExplainer(model, X_train, feature_cols, model_name=name)
        result = explainer.explain_instance(X_test.iloc[[0]])

        prob_sum = result["prediction_prob"]["no_default"] + result["prediction_prob"]["default"]
        assert abs(prob_sum - 1.0) < 0.01
