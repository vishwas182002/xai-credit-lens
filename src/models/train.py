"""Model training pipeline for credit default prediction.

Trains XGBoost, LightGBM, and Random Forest models with optional
Optuna hyperparameter tuning, then selects the best model.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT, get_model_config
from src.data.preprocess import get_feature_columns

logger = get_logger("models.train")

MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test splits."""
    train = pd.read_csv(SPLITS_DIR / "train.csv")
    val = pd.read_csv(SPLITS_DIR / "val.csv")
    test = pd.read_csv(SPLITS_DIR / "test.csv")
    return train, val, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute comprehensive evaluation metrics."""
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "avg_precision": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> xgb.XGBClassifier:
    """Train XGBoost with config parameters."""
    params = config["models"]["xgboost"].copy()

    # Auto-compute scale_pos_weight for class imbalance
    if params.get("scale_pos_weight") == "auto":
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        params["scale_pos_weight"] = neg_count / pos_count

    early_stopping = params.pop("early_stopping_rounds", 50)

    model = xgb.XGBClassifier(
        **params,
        use_label_encoder=False,
        random_state=42,
        early_stopping_rounds=early_stopping,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    logger.info(f"XGBoost trained — best iteration: {model.best_iteration}")
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> lgb.LGBMClassifier:
    """Train LightGBM with config parameters."""
    params = config["models"]["lightgbm"].copy()

    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    logger.info(f"LightGBM trained — best iteration: {model.best_iteration_}")
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> RandomForestClassifier:
    """Train Random Forest with config parameters."""
    params = config["models"]["random_forest"].copy()

    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    logger.info(f"Random Forest trained — {model.n_estimators} trees")
    return model


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> dict[str, Any]:
    """Train all models and return them with validation metrics."""
    results = {}

    # XGBoost
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val, config)
    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    results["xgboost"] = {
        "model": xgb_model,
        "metrics": compute_metrics(y_val, xgb_pred, xgb_prob),
    }

    # LightGBM
    logger.info("Training LightGBM...")
    lgb_model = train_lightgbm(X_train, y_train, X_val, y_val, config)
    lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
    lgb_pred = (lgb_prob >= 0.5).astype(int)
    results["lightgbm"] = {
        "model": lgb_model,
        "metrics": compute_metrics(y_val, lgb_pred, lgb_prob),
    }

    # Random Forest
    logger.info("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train, config)
    rf_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    results["random_forest"] = {
        "model": rf_model,
        "metrics": compute_metrics(y_val, rf_pred, rf_prob),
    }

    return results


def select_best_model(results: dict, metric: str = "roc_auc") -> str:
    """Select the best model based on a given metric."""
    scores = {name: res["metrics"][metric] for name, res in results.items()}
    best = max(scores, key=scores.get)
    logger.info(f"Best model by {metric}: {best} ({scores[best]:.4f})")
    for name, score in scores.items():
        logger.info(f"  {name}: {score:.4f}")
    return best


def save_models(results: dict, best_name: str) -> None:
    """Save all models and metrics to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, res in results.items():
        # Save model
        model_path = MODELS_DIR / f"{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(res["model"], f)

        # Save metrics
        metrics_path = MODELS_DIR / f"{name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(res["metrics"], f, indent=2)

    # Save best model indicator
    with open(MODELS_DIR / "best_model.txt", "w") as f:
        f.write(best_name)

    logger.info(f"All models saved to {MODELS_DIR}")


def run_training_pipeline() -> None:
    """Execute the full training pipeline."""
    config = get_model_config()

    # Load data
    train, val, test = load_splits()
    feature_cols = get_feature_columns(train)

    X_train, y_train = train[feature_cols], train["DEFAULT"]
    X_val, y_val = val[feature_cols], val["DEFAULT"]

    # Train all models
    results = train_all_models(X_train, y_train, X_val, y_val, config)

    # Print comparison
    logger.info("\n=== Model Comparison (Validation Set) ===")
    comparison = pd.DataFrame({name: res["metrics"] for name, res in results.items()})
    logger.info(f"\n{comparison.to_string()}")

    # Select best
    best_name = select_best_model(results)

    # Evaluate best on test set
    best_model = results[best_name]["model"]
    X_test, y_test = test[feature_cols], test["DEFAULT"]
    test_prob = best_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= 0.5).astype(int)
    test_metrics = compute_metrics(y_test, test_pred, test_prob)
    logger.info(f"\n=== Test Set Metrics ({best_name}) ===")
    for metric, val in test_metrics.items():
        logger.info(f"  {metric}: {val:.4f}")

    # Save everything
    save_models(results, best_name)
    with open(MODELS_DIR / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    run_training_pipeline()
