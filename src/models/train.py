"""Model training pipeline for credit default prediction.

Trains XGBoost, LightGBM, and Random Forest models with Optuna
hyperparameter tuning and Stratified K-Fold cross-validation,
then selects the best model.
"""

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    brier_score_loss,
)
import xgboost as xgb
import lightgbm as lgb

from src.utils.logger import get_logger
from src.utils.config import PROJECT_ROOT, get_model_config
from src.data.preprocess import get_feature_columns

logger = get_logger("models.train")

MODELS_DIR = PROJECT_ROOT / "models"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Suppress Optuna logs (too noisy)
optuna.logging.set_verbosity(optuna.logging.WARNING)


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


# ---------------------------------------------------------------------------
# Optuna objective functions
# ---------------------------------------------------------------------------

def xgboost_objective(trial, X, y, cv):
    """Optuna objective for XGBoost hyperparameter search."""
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "scale_pos_weight": neg_count / pos_count,
        "eval_metric": "auc",
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def lightgbm_objective(trial, X, y, cv):
    """Optuna objective for LightGBM hyperparameter search."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "is_unbalance": True,
        "random_state": 42,
        "verbose": -1,
    }

    model = lgb.LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def random_forest_objective(trial, X, y, cv):
    """Optuna objective for Random Forest hyperparameter search."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 5, 25),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 15),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.8]),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ---------------------------------------------------------------------------
# Training with Optuna
# ---------------------------------------------------------------------------

def tune_and_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
) -> dict[str, Any]:
    """Tune all models with Optuna + Stratified K-Fold, then train final models."""
    tuning_config = config.get("tuning", {})
    n_trials = tuning_config.get("n_trials", 50)
    cv_folds = tuning_config.get("cv_folds", 5)
    timeout = tuning_config.get("timeout", 3600)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Combine train + val for cross-validation tuning, then retrain on train only
    X_tune = pd.concat([X_train, X_val], axis=0)
    y_tune = pd.concat([y_train, y_val], axis=0)

    results = {}

    # --- XGBoost ---
    logger.info(f"Tuning XGBoost ({n_trials} trials, {cv_folds}-fold CV)...")
    xgb_study = optuna.create_study(direction="maximize", study_name="xgboost")
    xgb_study.optimize(
        lambda trial: xgboost_objective(trial, X_tune, y_tune, cv),
        n_trials=n_trials,
        timeout=timeout,
    )
    logger.info(f"XGBoost best CV AUC: {xgb_study.best_value:.4f}")

    xgb_params = xgb_study.best_params.copy()
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    xgb_params["scale_pos_weight"] = neg_count / pos_count
    xgb_params["eval_metric"] = "auc"
    xgb_params["random_state"] = 42
    xgb_params["early_stopping_rounds"] = 50

    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_prob = xgb_model.predict_proba(X_val)[:, 1]
    xgb_pred = (xgb_prob >= 0.5).astype(int)
    results["xgboost"] = {
        "model": xgb_model,
        "metrics": compute_metrics(y_val, xgb_pred, xgb_prob),
        "best_params": xgb_study.best_params,
        "cv_auc": xgb_study.best_value,
    }

    # --- LightGBM ---
    logger.info(f"Tuning LightGBM ({n_trials} trials, {cv_folds}-fold CV)...")
    lgb_study = optuna.create_study(direction="maximize", study_name="lightgbm")
    lgb_study.optimize(
        lambda trial: lightgbm_objective(trial, X_tune, y_tune, cv),
        n_trials=n_trials,
        timeout=timeout,
    )
    logger.info(f"LightGBM best CV AUC: {lgb_study.best_value:.4f}")

    lgb_params = lgb_study.best_params.copy()
    lgb_params["is_unbalance"] = True
    lgb_params["random_state"] = 42
    lgb_params["verbose"] = -1

    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    lgb_prob = lgb_model.predict_proba(X_val)[:, 1]
    lgb_pred = (lgb_prob >= 0.5).astype(int)
    results["lightgbm"] = {
        "model": lgb_model,
        "metrics": compute_metrics(y_val, lgb_pred, lgb_prob),
        "best_params": lgb_study.best_params,
        "cv_auc": lgb_study.best_value,
    }

    # --- Random Forest ---
    logger.info(f"Tuning Random Forest ({n_trials} trials, {cv_folds}-fold CV)...")
    rf_study = optuna.create_study(direction="maximize", study_name="random_forest")
    rf_study.optimize(
        lambda trial: random_forest_objective(trial, X_tune, y_tune, cv),
        n_trials=n_trials,
        timeout=timeout,
    )
    logger.info(f"Random Forest best CV AUC: {rf_study.best_value:.4f}")

    rf_params = rf_study.best_params.copy()
    rf_params["class_weight"] = "balanced"
    rf_params["random_state"] = 42
    rf_params["n_jobs"] = -1

    rf_model = RandomForestClassifier(**rf_params)
    rf_model.fit(X_train, y_train)
    rf_prob = rf_model.predict_proba(X_val)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    results["random_forest"] = {
        "model": rf_model,
        "metrics": compute_metrics(y_val, rf_pred, rf_prob),
        "best_params": rf_study.best_params,
        "cv_auc": rf_study.best_value,
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
    """Save all models, metrics, and tuning results to disk."""
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

        # Save best hyperparameters
        if "best_params" in res:
            params_path = MODELS_DIR / f"{name}_best_params.json"
            with open(params_path, "w") as f:
                json.dump(res["best_params"], f, indent=2)

    # Save best model indicator
    with open(MODELS_DIR / "best_model.txt", "w") as f:
        f.write(best_name)

    logger.info(f"All models saved to {MODELS_DIR}")


def run_training_pipeline() -> None:
    """Execute the full training pipeline with Optuna tuning."""
    config = get_model_config()

    # Load data
    train, val, test = load_splits()
    feature_cols = get_feature_columns(train)

    X_train, y_train = train[feature_cols], train["DEFAULT"]
    X_val, y_val = val[feature_cols], val["DEFAULT"]

    # Tune and train all models
    results = tune_and_train(X_train, y_train, X_val, y_val, config)

    # Print comparison
    logger.info("\n=== Model Comparison (Validation Set) ===")
    comparison = pd.DataFrame({name: res["metrics"] for name, res in results.items()})
    logger.info(f"\n{comparison.to_string()}")

    # Print CV scores
    logger.info("\n=== Cross-Validation AUC (5-Fold) ===")
    for name, res in results.items():
        logger.info(f"  {name}: {res['cv_auc']:.4f}")

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
