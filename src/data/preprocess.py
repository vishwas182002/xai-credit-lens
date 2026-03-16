"""Data preprocessing and feature engineering for credit default prediction."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT, get_model_config

logger = get_logger("data.preprocess")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Columns containing payment status history
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
PAY_AMT_COLS = ["PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]


def load_raw_data() -> pd.DataFrame:
    """Load the raw CSV dataset."""
    path = RAW_DIR / "credit_default.csv"
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}. Run download_data.py first.")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw data: fix known data quality issues."""
    df = df.copy()

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Fix EDUCATION: merge undocumented categories (0, 5, 6) into 4 (Other)
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})

    # Fix MARRIAGE: merge undocumented category 0 into 3 (Other)
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

    # Clip PAY_X columns: values below -1 are all "paid duly", normalize to -1
    for col in PAY_COLS:
        df[col] = df[col].clip(lower=-2)

    logger.info(f"Cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that improve model performance and interpretability."""
    df = df.copy()

    # Debt-to-income proxy: average bill relative to credit limit
    avg_bill = df[BILL_COLS].mean(axis=1)
    df["DEBT_TO_INCOME_PROXY"] = (avg_bill / df["LIMIT_BAL"].replace(0, np.nan)).fillna(0)

    # Payment ratio: how much of the bill is being paid
    total_bill = df[BILL_COLS].sum(axis=1).replace(0, np.nan)
    total_pay = df[PAY_AMT_COLS].sum(axis=1)
    df["PAYMENT_RATIO"] = (total_pay / total_bill).fillna(0).clip(0, 5)

    # Credit utilization rate
    df["UTILIZATION_RATE"] = (avg_bill / df["LIMIT_BAL"].replace(0, np.nan)).fillna(0).clip(0, 5)

    # Number of months with delinquency (PAY > 0 means late)
    df["MONTHS_DELINQUENT"] = (df[PAY_COLS] > 0).sum(axis=1)

    # Average payment delay severity
    df["AVG_PAYMENT_DELAY"] = df[PAY_COLS].mean(axis=1)

    # Payment trend: slope of payment amounts over 6 months (positive = increasing payments)
    pay_values = df[PAY_AMT_COLS].values
    x = np.arange(6)
    x_centered = x - x.mean()
    slopes = (pay_values * x_centered).sum(axis=1) / (x_centered**2).sum()
    df["PAYMENT_TREND"] = slopes

    # Max consecutive months delinquent
    def max_consecutive_delinquent(row):
        vals = [row[c] > 0 for c in PAY_COLS]
        max_consec = 0
        current = 0
        for v in vals:
            if v:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        return max_consec

    df["MAX_CONSEC_DELINQUENT"] = df.apply(max_consecutive_delinquent, axis=1)

    # Balance volatility: std of bill amounts normalized by limit
    df["BALANCE_VOLATILITY"] = (
        df[BILL_COLS].std(axis=1) / df["LIMIT_BAL"].replace(0, np.nan)
    ).fillna(0)

    logger.info(f"Engineered features: added {9} new columns")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str = "DEFAULT",
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train/val/test with stratification."""
    # First split: train+val vs test
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )

    # Second split: train vs val
    adjusted_val_size = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val,
        test_size=adjusted_val_size,
        random_state=random_state,
        stratify=train_val[target_col],
    )

    logger.info(
        f"Split sizes — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}"
    )
    logger.info(
        f"Default rates — Train: {train[target_col].mean():.2%}, "
        f"Val: {val[target_col].mean():.2%}, Test: {test[target_col].mean():.2%}"
    )
    return train, val, test


def get_feature_columns(df: pd.DataFrame, target_col: str = "DEFAULT") -> list[str]:
    """Return all feature columns (excluding target and any IDs)."""
    exclude = {target_col, "ID"}
    return [c for c in df.columns if c not in exclude]


def run_preprocessing_pipeline() -> None:
    """Execute the full preprocessing pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    # Load and clean
    df = load_raw_data()
    df = clean_data(df)

    # Engineer features
    df = engineer_features(df)

    # Save processed
    df.to_csv(PROCESSED_DIR / "credit_default_processed.csv", index=False)
    logger.info(f"Saved processed data to {PROCESSED_DIR}")

    # Split
    train, val, test = split_data(df)
    train.to_csv(SPLITS_DIR / "train.csv", index=False)
    val.to_csv(SPLITS_DIR / "val.csv", index=False)
    test.to_csv(SPLITS_DIR / "test.csv", index=False)
    logger.info(f"Saved splits to {SPLITS_DIR}")

    # Save feature metadata
    feature_cols = get_feature_columns(df)
    feature_meta = pd.DataFrame(
        {
            "feature": feature_cols,
            "dtype": [str(df[c].dtype) for c in feature_cols],
            "nunique": [df[c].nunique() for c in feature_cols],
            "null_pct": [df[c].isnull().mean() for c in feature_cols],
        }
    )
    feature_meta.to_csv(PROCESSED_DIR / "feature_metadata.csv", index=False)
    logger.info("Preprocessing pipeline complete.")


if __name__ == "__main__":
    run_preprocessing_pipeline()
