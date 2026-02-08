"""Training utilities for ignition probability model."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from config.settings import get_config, OUTPUT_DIR, PROCESSED_DATA_DIR
from .models import IgnitionModel, get_model

logger = logging.getLogger(__name__)


def train_ignition_model(
    training_data: Union[pd.DataFrame, Path],
    model_type: str = "random_forest",
    feature_columns: Optional[List[str]] = None,
    target_column: str = "label",
    test_size: float = 0.2,
    output_dir: Optional[Path] = None,
    **model_kwargs,
) -> Tuple[IgnitionModel, Dict]:
    """
    Train ignition probability model.

    Parameters
    ----------
    training_data : DataFrame or Path
        Training data with features and labels
    model_type : str
        Model type ("random_forest", "xgboost", "lightgbm")
    feature_columns : list, optional
        Feature column names (auto-detected if not provided)
    target_column : str
        Target column name
    test_size : float
        Fraction of data for testing
    output_dir : Path, optional
        Directory to save model and results
    **model_kwargs
        Additional model parameters

    Returns
    -------
    tuple
        (trained model, evaluation metrics dict)
    """
    output_dir = output_dir or OUTPUT_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data if path provided
    if isinstance(training_data, Path):
        training_data = pd.read_parquet(training_data)

    logger.info(f"Training data: {len(training_data)} samples")

    # Identify feature columns
    if feature_columns is None:
        exclude_cols = {target_column, "geometry", "date", "x", "y"}
        feature_columns = [c for c in training_data.columns if c not in exclude_cols]

    logger.info(f"Using {len(feature_columns)} features")

    # Prepare data
    X = training_data[feature_columns].copy()
    y = training_data[target_column].copy()

    # Handle missing values
    X = X.fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Train positive rate: {y_train.mean():.3f}")
    logger.info(f"Test positive rate: {y_test.mean():.3f}")

    # Create and train model
    model = get_model(model_type, **model_kwargs)
    model.fit(X_train, y_train)

    # Evaluate
    metrics = evaluate_ignition_model(model, X_test, y_test)

    # Log results
    logger.info(f"Test AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"Test AUC-PR: {metrics['auc_pr']:.4f}")
    logger.info(f"Test Recall: {metrics['recall']:.4f}")
    logger.info(f"Test Precision: {metrics['precision']:.4f}")

    # Save model
    model_path = output_dir / f"ignition_model_{model_type}.joblib"
    model.save(model_path)

    # Save feature importance
    importance = model.get_feature_importance()
    importance_df = pd.DataFrame(
        sorted(importance.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "importance"],
    )
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    return model, metrics


def evaluate_ignition_model(
    model: IgnitionModel,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    threshold: float = 0.5,
) -> Dict:
    """
    Evaluate ignition model performance.

    Parameters
    ----------
    model : IgnitionModel
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    threshold : float
        Classification threshold

    Returns
    -------
    dict
        Evaluation metrics
    """
    # Predictions
    y_proba = model.predict_proba(X_test)
    y_pred = (y_proba >= threshold).astype(int)

    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    # Metrics
    metrics = {}

    # AUC-ROC
    metrics["auc_roc"] = roc_auc_score(y_test, y_proba)

    # AUC-PR (better for imbalanced data)
    metrics["auc_pr"] = average_precision_score(y_test, y_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics["true_negatives"] = int(tn)
    metrics["false_positives"] = int(fp)
    metrics["false_negatives"] = int(fn)
    metrics["true_positives"] = int(tp)

    # Derived metrics
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = (
            2 * metrics["precision"] * metrics["recall"] /
            (metrics["precision"] + metrics["recall"])
        )
    else:
        metrics["f1"] = 0.0

    metrics["accuracy"] = (tp + tn) / len(y_test)

    # Optimal threshold (maximizing F1)
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    best_idx = np.argmax(f1_scores)
    metrics["optimal_threshold"] = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    metrics["optimal_f1"] = float(f1_scores[best_idx])

    return metrics


def cross_validate_model(
    training_data: Union[pd.DataFrame, Path],
    model_type: str = "random_forest",
    feature_columns: Optional[List[str]] = None,
    target_column: str = "label",
    n_folds: int = 5,
    **model_kwargs,
) -> Tuple[List[Dict], Dict]:
    """
    Cross-validate ignition model.

    Parameters
    ----------
    training_data : DataFrame or Path
        Training data
    model_type : str
        Model type
    feature_columns : list, optional
        Feature columns
    target_column : str
        Target column
    n_folds : int
        Number of CV folds
    **model_kwargs
        Model parameters

    Returns
    -------
    tuple
        (list of fold metrics, aggregated metrics)
    """
    if isinstance(training_data, Path):
        training_data = pd.read_parquet(training_data)

    if feature_columns is None:
        exclude_cols = {target_column, "geometry", "date", "x", "y"}
        feature_columns = [c for c in training_data.columns if c not in exclude_cols]

    X = training_data[feature_columns].fillna(0)
    y = training_data[target_column]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        logger.info(f"Training fold {fold + 1}/{n_folds}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = get_model(model_type, **model_kwargs)
        model.fit(X_train, y_train)

        metrics = evaluate_ignition_model(model, X_val, y_val)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)

        logger.info(f"  Fold {fold + 1} AUC: {metrics['auc_roc']:.4f}")

    # Aggregate metrics
    agg_metrics = {}
    numeric_keys = ["auc_roc", "auc_pr", "precision", "recall", "f1", "accuracy"]

    for key in numeric_keys:
        values = [m[key] for m in fold_metrics]
        agg_metrics[f"{key}_mean"] = np.mean(values)
        agg_metrics[f"{key}_std"] = np.std(values)

    logger.info(f"Cross-validation AUC: {agg_metrics['auc_roc_mean']:.4f} +/- {agg_metrics['auc_roc_std']:.4f}")

    return fold_metrics, agg_metrics


def train_with_temporal_split(
    training_data: Union[pd.DataFrame, Path],
    model_type: str = "random_forest",
    feature_columns: Optional[List[str]] = None,
    target_column: str = "label",
    date_column: str = "date",
    train_years: Tuple[int, int] = (2010, 2017),
    test_years: Tuple[int, int] = (2018, 2022),
    **model_kwargs,
) -> Tuple[IgnitionModel, Dict]:
    """
    Train model with temporal train/test split.

    Uses data from train_years for training and test_years for evaluation.
    This ensures no temporal leakage.

    Parameters
    ----------
    training_data : DataFrame or Path
        Training data with date column
    model_type : str
        Model type
    feature_columns : list, optional
        Feature columns
    target_column : str
        Target column
    date_column : str
        Date column name
    train_years : tuple
        (start_year, end_year) for training
    test_years : tuple
        (start_year, end_year) for testing

    Returns
    -------
    tuple
        (trained model, evaluation metrics)
    """
    if isinstance(training_data, Path):
        training_data = pd.read_parquet(training_data)

    # Convert date column
    training_data[date_column] = pd.to_datetime(training_data[date_column])
    training_data["year"] = training_data[date_column].dt.year

    # Split by year
    train_mask = (
        (training_data["year"] >= train_years[0]) &
        (training_data["year"] <= train_years[1])
    )
    test_mask = (
        (training_data["year"] >= test_years[0]) &
        (training_data["year"] <= test_years[1])
    )

    train_data = training_data[train_mask]
    test_data = training_data[test_mask]

    logger.info(f"Training on years {train_years}: {len(train_data)} samples")
    logger.info(f"Testing on years {test_years}: {len(test_data)} samples")

    if feature_columns is None:
        exclude_cols = {target_column, date_column, "geometry", "x", "y", "year"}
        feature_columns = [c for c in training_data.columns if c not in exclude_cols]

    X_train = train_data[feature_columns].fillna(0)
    y_train = train_data[target_column]
    X_test = test_data[feature_columns].fillna(0)
    y_test = test_data[target_column]

    model = get_model(model_type, **model_kwargs)
    model.fit(X_train, y_train)

    metrics = evaluate_ignition_model(model, X_test, y_test)

    logger.info(f"Temporal split AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"Temporal split Recall: {metrics['recall']:.4f}")

    return model, metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Optional[Path] = None,
) -> None:
    """Plot ROC curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Ignition Model")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved ROC curve to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    output_path: Optional[Path] = None,
) -> None:
    """Plot precision-recall curve."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    auc_pr = average_precision_score(y_true, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"AUC-PR = {auc_pr:.3f}")
    plt.axhline(y=y_true.mean(), color="k", linestyle="--", label="Baseline")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - Ignition Model")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved PR curve to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_feature_importance(
    model: IgnitionModel,
    top_n: int = 20,
    output_path: Optional[Path] = None,
) -> None:
    """Plot feature importance."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    importance = model.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]

    features, values = zip(*sorted_importance)

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(features)), values)
    plt.yticks(range(len(features)), features)
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances - Ignition Model")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved feature importance plot to {output_path}")
    else:
        plt.show()

    plt.close()
