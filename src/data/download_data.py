"""Download the UCI Default of Credit Card Clients dataset."""

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT

logger = get_logger("data.download")

RAW_DIR = PROJECT_ROOT / "data" / "raw"


def download_dataset() -> pd.DataFrame:
    """Fetch the dataset from UCI ML Repository and save to disk."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / "credit_default.csv"

    if output_path.exists():
        logger.info(f"Dataset already exists at {output_path}")
        return pd.read_csv(output_path)

    logger.info("Downloading UCI Default of Credit Card Clients dataset (ID: 350)...")
    dataset = fetch_ucirepo(id=350)

    # Combine features and target
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)

    # The UCI API returns generic names (X1..X23, Y). Map to real column names.
    column_map = {
        "X1": "LIMIT_BAL",
        "X2": "SEX",
        "X3": "EDUCATION",
        "X4": "MARRIAGE",
        "X5": "AGE",
        "X6": "PAY_0",
        "X7": "PAY_2",
        "X8": "PAY_3",
        "X9": "PAY_4",
        "X10": "PAY_5",
        "X11": "PAY_6",
        "X12": "BILL_AMT1",
        "X13": "BILL_AMT2",
        "X14": "BILL_AMT3",
        "X15": "BILL_AMT4",
        "X16": "BILL_AMT5",
        "X17": "BILL_AMT6",
        "X18": "PAY_AMT1",
        "X19": "PAY_AMT2",
        "X20": "PAY_AMT3",
        "X21": "PAY_AMT4",
        "X22": "PAY_AMT5",
        "X23": "PAY_AMT6",
        "Y": "DEFAULT",
    }
    df = df.rename(columns=column_map)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} records to {output_path}")
    logger.info(f"Features: {list(df.columns)}")
    logger.info(f"Default rate: {df['DEFAULT'].mean():.2%}")

    return df


if __name__ == "__main__":
    download_dataset()
