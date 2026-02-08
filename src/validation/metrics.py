"""Validation metrics for burn probability model."""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)

logger = logging.getLogger(__name__)


def compute_discrimination_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict:
    """
    Compute discrimination metrics for binary predictions.

    Parameters
    ----------
    y_true : ndarray
        True binary outcomes (0 or 1)
    y_prob : ndarray
        Predicted probabilities
    threshold : float
        Classification threshold

    Returns
    -------
    dict
        Discrimination metrics
    """
    # Filter out invalid values
    valid = ~np.isnan(y_prob)
    y_true = np.asarray(y_true)[valid]
    y_prob = np.asarray(y_prob)[valid]

    if len(y_true) == 0 or y_true.sum() == 0:
        return {
            "auc_roc": 0.5,
            "auc_pr": 0.0,
            "brier_score": 1.0,
            "recall": 0.0,
            "precision": 0.0,
        }

    metrics = {}

    # AUC-ROC
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = 0.5

    # AUC-PR (better for imbalanced data)
    try:
        metrics["auc_pr"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["auc_pr"] = 0.0

    # Brier score (calibration measure)
    metrics["brier_score"] = float(brier_score_loss(y_true, y_prob))

    # Classification metrics at threshold
    y_pred = (y_prob >= threshold).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()

    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    metrics["accuracy"] = (tp + tn) / len(y_true)

    if metrics["precision"] + metrics["recall"] > 0:
        metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (
            metrics["precision"] + metrics["recall"]
        )
    else:
        metrics["f1"] = 0.0

    return metrics


def compute_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Compute calibration curve data.

    Parameters
    ----------
    y_true : ndarray
        True binary outcomes
    y_prob : ndarray
        Predicted probabilities
    n_bins : int
        Number of bins

    Returns
    -------
    dict
        Calibration curve data and metrics
    """
    from sklearn.calibration import calibration_curve

    # Filter invalid
    valid = ~np.isnan(y_prob) & ~np.isnan(y_true)
    y_true = np.asarray(y_true)[valid]
    y_prob = np.asarray(y_prob)[valid]

    if len(y_true) == 0:
        return {
            "prob_true": np.array([]),
            "prob_pred": np.array([]),
            "expected_calibration_error": 1.0,
        }

    # Compute calibration curve
    try:
        prob_true, prob_pred = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform"
        )
    except ValueError:
        return {
            "prob_true": np.array([]),
            "prob_pred": np.array([]),
            "expected_calibration_error": 1.0,
        }

    # Compute ECE
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_sums = np.bincount(bin_indices, weights=y_true.astype(float), minlength=n_bins)
    bin_prob_sums = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)

    nonzero = bin_counts > 0
    observed = np.zeros(n_bins)
    predicted = np.zeros(n_bins)

    observed[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
    predicted[nonzero] = bin_prob_sums[nonzero] / bin_counts[nonzero]

    ece = np.sum(bin_counts * np.abs(observed - predicted)) / len(y_prob)

    return {
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_observed": observed,
        "bin_predicted": predicted,
        "bin_counts": bin_counts,
        "expected_calibration_error": float(ece),
    }


def compute_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Compute and optionally plot reliability diagram.

    Parameters
    ----------
    y_true : ndarray
        True outcomes
    y_prob : ndarray
        Predicted probabilities
    n_bins : int
        Number of bins
    output_path : str, optional
        Path to save plot

    Returns
    -------
    dict
        Reliability diagram data
    """
    cal = compute_calibration_curve(y_true, y_prob, n_bins)

    if output_path:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={"height_ratios": [3, 1]})

            # Reliability diagram
            ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
            ax1.plot(cal["prob_pred"], cal["prob_true"], "o-", label="Model")
            ax1.set_xlabel("Mean predicted probability")
            ax1.set_ylabel("Fraction of positives")
            ax1.set_title(f"Reliability Diagram (ECE={cal['expected_calibration_error']:.3f})")
            ax1.legend()
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.grid(True, alpha=0.3)

            # Histogram
            ax2.bar(
                np.linspace(0.05, 0.95, n_bins),
                cal["bin_counts"],
                width=0.08,
                alpha=0.7,
            )
            ax2.set_xlabel("Mean predicted probability")
            ax2.set_ylabel("Count")
            ax2.set_xlim(0, 1)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved reliability diagram to {output_path}")

        except ImportError:
            logger.warning("matplotlib not available for plotting")

    return cal


def compute_skill_scores(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    climatology: Optional[float] = None,
) -> Dict:
    """
    Compute probabilistic skill scores.

    Parameters
    ----------
    y_true : ndarray
        True outcomes
    y_prob : ndarray
        Predicted probabilities
    climatology : float, optional
        Climatological (baseline) probability

    Returns
    -------
    dict
        Skill scores
    """
    # Filter invalid
    valid = ~np.isnan(y_prob) & ~np.isnan(y_true)
    y_true = np.asarray(y_true)[valid]
    y_prob = np.asarray(y_prob)[valid]

    if len(y_true) == 0:
        return {}

    # Climatology (base rate)
    if climatology is None:
        climatology = y_true.mean()

    # Brier Skill Score
    bs = brier_score_loss(y_true, y_prob)
    bs_clim = brier_score_loss(y_true, np.full_like(y_prob, climatology))

    if bs_clim > 0:
        bss = 1 - bs / bs_clim
    else:
        bss = 0.0

    # Decompose Brier score
    # Reliability, Resolution, Uncertainty
    n = len(y_true)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_sums = np.bincount(bin_indices, weights=y_true.astype(float), minlength=n_bins)
    bin_prob_sums = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)

    nonzero = bin_counts > 0
    observed = np.zeros(n_bins)
    predicted = np.zeros(n_bins)

    observed[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
    predicted[nonzero] = bin_prob_sums[nonzero] / bin_counts[nonzero]

    # Reliability
    reliability = np.sum(bin_counts * (predicted - observed) ** 2) / n

    # Resolution
    resolution = np.sum(bin_counts * (observed - climatology) ** 2) / n

    # Uncertainty
    uncertainty = climatology * (1 - climatology)

    return {
        "brier_score": float(bs),
        "brier_skill_score": float(bss),
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "climatology": float(climatology),
    }


def compute_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Compute ROC curve data.

    Parameters
    ----------
    y_true : ndarray
        True outcomes
    y_prob : ndarray
        Predicted probabilities
    output_path : str, optional
        Path to save plot

    Returns
    -------
    dict
        ROC curve data
    """
    # Filter invalid
    valid = ~np.isnan(y_prob) & ~np.isnan(y_true)
    y_true = np.asarray(y_true)[valid]
    y_prob = np.asarray(y_prob)[valid]

    if len(y_true) == 0 or y_true.sum() == 0:
        return {"fpr": np.array([0, 1]), "tpr": np.array([0, 1]), "auc": 0.5}

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    if output_path:
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f"Model (AUC = {auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", label="Random")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

        except ImportError:
            pass

    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(auc),
    }
