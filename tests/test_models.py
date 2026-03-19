"""Tests for model training and evaluation."""

import pickle
import json

import numpy as np
import pandas as pd
import pytest

from src.models.train import compute_metrics, select_best_model
from src.utils.config import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing metrics."""
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.35, 0.55])
    y_pred = (y_prob >= 0.5).astype(int)
    return y_true, y_pred, y_prob


class TestComputeMetrics:
    def test_returns_all_metrics(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        expected_keys = {"roc_auc", "avg_precision", "f1", "precision", "recall", "brier_score"}
        assert set(metrics.keys()) == expected_keys

    def test_metrics_in_valid_range(self, sample_predictions):
        y_true, y_pred, y_prob = sample_predictions
        metrics = compute_metrics(y_true, y_pred, y_prob)

        for key in ["roc_auc", "avg_precision", "f1", "precision", "recall"]:
            assert 0 <= metrics[key] <= 1, f"{key} out of range: {metrics[key]}"

        assert 0 <= metrics["brier_score"] <= 1

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        y_pred = np.array([0, 0, 1, 1])
        metrics = compute_metrics(y_true, y_pred, y_prob)

        assert metrics["roc_auc"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["brier_score"] == 0.0


class TestSelectBestModel:
    def test_selects_highest_auc(self):
        results = {
            "model_a": {"metrics": {"roc_auc": 0.75}},
            "model_b": {"metrics": {"roc_auc": 0.80}},
            "model_c": {"metrics": {"roc_auc": 0.78}},
        }
        best = select_best_model(results, metric="roc_auc")
        assert best == "model_b"

    def test_selects_by_f1(self):
        results = {
            "model_a": {"metrics": {"f1": 0.65}},
            "model_b": {"metrics": {"f1": 0.60}},
        }
        best = select_best_model(results, metric="f1")
        assert best == "model_a"


class TestSavedModels:
    def test_best_model_file_exists(self):
        path = MODELS_DIR / "best_model.txt"
        if not path.exists():
            pytest.skip("Models not trained yet.")
        name = path.read_text().strip()
        assert name in ["xgboost", "lightgbm", "random_forest"]

    def test_model_pickle_loadable(self):
        best_path = MODELS_DIR / "best_model.txt"
        if not best_path.exists():
            pytest.skip("Models not trained yet.")
        name = best_path.read_text().strip()
        model_path = MODELS_DIR / f"{name}.pkl"
        assert model_path.exists()

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        assert hasattr(model, "predict_proba")

    def test_metrics_json_valid(self):
        path = MODELS_DIR / "test_metrics.json"
        if not path.exists():
            pytest.skip("Models not trained yet.")
        with open(path) as f:
            metrics = json.load(f)
        assert "roc_auc" in metrics
        assert 0 < metrics["roc_auc"] < 1
