"""Tests for the data pipeline."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from src.data.preprocess import clean_data, engineer_features, split_data, get_feature_columns
from src.utils.config import PROJECT_ROOT

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "credit_default.csv"


@pytest.fixture
def raw_df():
    """Load raw dataset."""
    if not RAW_PATH.exists():
        pytest.skip("Raw data not found. Run download_data.py first.")
    return pd.read_csv(RAW_PATH)


@pytest.fixture
def clean_df(raw_df):
    """Clean raw dataset."""
    return clean_data(raw_df)


@pytest.fixture
def engineered_df(clean_df):
    """Dataset with engineered features."""
    return engineer_features(clean_df)


class TestCleanData:
    def test_no_missing_values(self, clean_df):
        assert clean_df.isnull().sum().sum() == 0

    def test_education_values(self, clean_df):
        """EDUCATION should only have values 1-4 after cleaning."""
        assert set(clean_df["EDUCATION"].unique()).issubset({1, 2, 3, 4})

    def test_marriage_values(self, clean_df):
        """MARRIAGE should only have values 1-3 after cleaning."""
        assert set(clean_df["MARRIAGE"].unique()).issubset({1, 2, 3})

    def test_row_count_preserved(self, raw_df, clean_df):
        """Cleaning should not drop rows."""
        assert len(clean_df) == len(raw_df)

    def test_id_column_removed(self, clean_df):
        assert "ID" not in clean_df.columns


class TestFeatureEngineering:
    def test_new_features_added(self, engineered_df):
        expected = [
            "DEBT_TO_INCOME_PROXY", "PAYMENT_RATIO", "UTILIZATION_RATE",
            "MONTHS_DELINQUENT", "AVG_PAYMENT_DELAY", "PAYMENT_TREND",
            "MAX_CONSEC_DELINQUENT", "BALANCE_VOLATILITY",
        ]
        for feat in expected:
            assert feat in engineered_df.columns, f"Missing feature: {feat}"

    def test_no_nan_in_engineered(self, engineered_df):
        eng_cols = [
            "DEBT_TO_INCOME_PROXY", "PAYMENT_RATIO", "UTILIZATION_RATE",
            "MONTHS_DELINQUENT", "AVG_PAYMENT_DELAY", "PAYMENT_TREND",
            "MAX_CONSEC_DELINQUENT", "BALANCE_VOLATILITY",
        ]
        for col in eng_cols:
            assert engineered_df[col].isnull().sum() == 0, f"NaN in {col}"

    def test_months_delinquent_range(self, engineered_df):
        """Should be between 0 and 6."""
        assert engineered_df["MONTHS_DELINQUENT"].min() >= 0
        assert engineered_df["MONTHS_DELINQUENT"].max() <= 6

    def test_max_consec_delinquent_range(self, engineered_df):
        """Should be between 0 and 6."""
        assert engineered_df["MAX_CONSEC_DELINQUENT"].min() >= 0
        assert engineered_df["MAX_CONSEC_DELINQUENT"].max() <= 6

    def test_utilization_rate_clipped(self, engineered_df):
        assert engineered_df["UTILIZATION_RATE"].max() <= 5

    def test_payment_ratio_clipped(self, engineered_df):
        assert engineered_df["PAYMENT_RATIO"].max() <= 5


class TestSplitData:
    def test_split_sizes(self, engineered_df):
        train, val, test = split_data(engineered_df)
        total = len(train) + len(val) + len(test)
        assert total == len(engineered_df)

    def test_stratification(self, engineered_df):
        """Default rates should be similar across splits."""
        train, val, test = split_data(engineered_df)
        train_rate = train["DEFAULT"].mean()
        val_rate = val["DEFAULT"].mean()
        test_rate = test["DEFAULT"].mean()

        assert abs(train_rate - val_rate) < 0.02
        assert abs(train_rate - test_rate) < 0.02

    def test_no_data_leakage(self, engineered_df):
        """No overlapping indices between splits."""
        train, val, test = split_data(engineered_df)
        train_idx = set(train.index)
        val_idx = set(val.index)
        test_idx = set(test.index)

        assert len(train_idx & val_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(val_idx & test_idx) == 0

    def test_feature_columns(self, engineered_df):
        feature_cols = get_feature_columns(engineered_df)
        assert "DEFAULT" not in feature_cols
        assert len(feature_cols) > 20
