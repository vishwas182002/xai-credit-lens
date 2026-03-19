"""Tests for fairness audit module."""

import numpy as np
import pandas as pd
import pytest

from src.fairness.metrics import FairnessAuditor


@pytest.fixture
def auditor():
    return FairnessAuditor()


@pytest.fixture
def sample_data():
    """Generate sample data with known fairness properties."""
    np.random.seed(42)
    n = 1000

    # Create data where predictions are roughly fair
    y_true = np.random.binomial(1, 0.22, n)
    y_prob = np.clip(y_true * 0.6 + np.random.normal(0.3, 0.15, n), 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)

    sensitive = pd.DataFrame({
        "SEX": np.random.choice([1, 2], n),
        "EDUCATION": np.random.choice([1, 2, 3, 4], n),
        "MARRIAGE": np.random.choice([1, 2, 3], n),
        "AGE": np.random.randint(21, 70, n),
    })

    return y_true, y_pred, y_prob, sensitive


class TestFairnessAuditor:
    def test_audit_returns_all_attributes(self, auditor, sample_data):
        y_true, y_pred, y_prob, sensitive = sample_data
        report = auditor.run_full_audit(y_true, y_pred, y_prob, sensitive)

        assert "audit_results" in report
        assert "overall_status" in report
        assert report["overall_status"] in ["PASS", "FAIL"]

    def test_audit_has_all_protected_attributes(self, auditor, sample_data):
        y_true, y_pred, y_prob, sensitive = sample_data
        report = auditor.run_full_audit(y_true, y_pred, y_prob, sensitive)

        expected_attrs = {"sex", "education", "age", "marriage"}
        assert expected_attrs == set(report["audit_results"].keys())

    def test_disparate_impact_range(self, auditor, sample_data):
        y_true, y_pred, y_prob, sensitive = sample_data
        report = auditor.run_full_audit(y_true, y_pred, y_prob, sensitive)

        for attr_key, result in report["audit_results"].items():
            di = result["metrics"]["disparate_impact"]
            assert 0 <= di <= 2, f"DI out of range for {attr_key}: {di}"

    def test_metric_keys_present(self, auditor, sample_data):
        y_true, y_pred, y_prob, sensitive = sample_data
        report = auditor.run_full_audit(y_true, y_pred, y_prob, sensitive)

        expected_metrics = {
            "disparate_impact",
            "statistical_parity_difference",
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "predictive_parity_difference",
        }

        for attr_key, result in report["audit_results"].items():
            assert expected_metrics.issubset(set(result["metrics"].keys()))

    def test_flags_match_thresholds(self, auditor, sample_data):
        y_true, y_pred, y_prob, sensitive = sample_data
        report = auditor.run_full_audit(y_true, y_pred, y_prob, sensitive)

        for attr_key, result in report["audit_results"].items():
            di = result["metrics"]["disparate_impact"]
            di_flag = result["flags"].get("disparate_impact", False)
            if di < 0.8:
                assert di_flag, f"DI={di} should be flagged for {attr_key}"

    def test_recommendations_non_empty(self, auditor, sample_data):
        y_true, y_pred, y_prob, sensitive = sample_data
        report = auditor.run_full_audit(y_true, y_pred, y_prob, sensitive)

        for attr_key, result in report["audit_results"].items():
            assert len(result["recommendations"]) > 0

    def test_perfect_fairness(self, auditor):
        """When predictions are identical across groups, DI should be ~1."""
        n = 500
        y_true = np.zeros(n, dtype=int)
        y_pred = np.zeros(n, dtype=int)
        y_prob = np.full(n, 0.3)

        sensitive = pd.DataFrame({
            "SEX": np.repeat([1, 2], n // 2),
            "EDUCATION": np.repeat([1, 2, 3, 4], n // 4),
            "MARRIAGE": np.tile([1, 2, 3], n // 3 + 1)[:n],
            "AGE": np.concatenate([np.full(n // 2, 35), np.full(n // 2, 25)]),
        })

        report = auditor.run_full_audit(y_true, y_pred, y_prob, sensitive)

        for attr_key, result in report["audit_results"].items():
            di = result["metrics"]["disparate_impact"]
            assert abs(di - 1.0) < 0.01, f"Expected DI ~1.0, got {di} for {attr_key}"
