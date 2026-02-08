"""Hindcast validation for burn probability model."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import geopandas as gpd

from config.settings import get_config, OUTPUT_DIR

logger = logging.getLogger(__name__)


def run_hindcast_validation(
    predictions_dir: Path,
    fire_perimeters_gdf: gpd.GeoDataFrame,
    parcels_gdf: gpd.GeoDataFrame,
    years: Tuple[int, int],
    holdout_fraction: float = 0.30,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run hindcast validation across multiple years.

    For each year, compares predicted burn probability against
    30% holdout fires that were not used in training.

    Parameters
    ----------
    predictions_dir : Path
        Directory containing burn probability predictions by year
    fire_perimeters_gdf : GeoDataFrame
        All fire perimeters
    parcels_gdf : GeoDataFrame
        Parcel polygons
    years : tuple
        (start_year, end_year) inclusive
    holdout_fraction : float
        Fraction of fires to hold out per year
    output_dir : Path, optional
        Output directory

    Returns
    -------
    DataFrame
        Validation metrics for each year
    """
    output_dir = output_dir or OUTPUT_DIR / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    from .fire_holdout import split_fires_by_year

    # Split fires into train/holdout
    train_fires, holdout_fires = split_fires_by_year(
        fire_perimeters_gdf,
        holdout_fraction=holdout_fraction,
    )

    results = []

    for year in range(years[0], years[1] + 1):
        logger.info(f"Validating year {year}")

        # Load predictions for this year
        pred_path = predictions_dir / f"burn_probability_{year}.tif"
        if not pred_path.exists():
            logger.warning(f"No predictions found for {year}")
            continue

        # Get holdout fires for this year
        year_holdout = holdout_fires[holdout_fires["year_"] == year]
        if year_holdout.empty:
            logger.warning(f"No holdout fires for {year}")
            continue

        # Validate
        metrics = validate_year(
            burn_prob_path=pred_path,
            holdout_fires=year_holdout,
            parcels_gdf=parcels_gdf,
        )
        metrics["year"] = year
        metrics["n_holdout_fires"] = len(year_holdout)

        results.append(metrics)

    # Combine results
    results_df = pd.DataFrame(results)

    # Compute aggregate metrics
    agg_metrics = {
        "auc_roc_mean": results_df["auc_roc"].mean(),
        "auc_roc_std": results_df["auc_roc"].std(),
        "calibration_error_mean": results_df["calibration_error"].mean(),
        "recall_mean": results_df["recall"].mean(),
    }

    logger.info(f"Aggregate AUC: {agg_metrics['auc_roc_mean']:.4f} +/- {agg_metrics['auc_roc_std']:.4f}")

    # Save results
    results_df.to_csv(output_dir / "hindcast_results.csv", index=False)

    return results_df


def validate_year(
    burn_prob_path: Path,
    holdout_fires: gpd.GeoDataFrame,
    parcels_gdf: gpd.GeoDataFrame,
) -> Dict:
    """
    Validate predictions for a single year.

    Parameters
    ----------
    burn_prob_path : Path
        Path to burn probability raster
    holdout_fires : GeoDataFrame
        Holdout fire perimeters for validation
    parcels_gdf : GeoDataFrame
        Parcel polygons

    Returns
    -------
    dict
        Validation metrics
    """
    from .metrics import (
        compute_discrimination_metrics,
        compute_calibration_curve,
    )
    from src.integration.parcel_aggregation import aggregate_to_parcels

    # Aggregate burn probability to parcels
    parcels_with_prob = aggregate_to_parcels(burn_prob_path, parcels_gdf)

    # Identify parcels in holdout fires
    if holdout_fires.crs != parcels_gdf.crs:
        holdout_fires = holdout_fires.to_crs(parcels_gdf.crs)

    burned_parcels = gpd.sjoin(
        parcels_gdf[["apn", "geometry"]],
        holdout_fires[["geometry"]],
        how="inner",
        predicate="intersects",
    )["apn"].unique()

    parcels_with_prob["actually_burned"] = parcels_with_prob["apn"].isin(burned_parcels)

    # Compute metrics
    y_true = parcels_with_prob["actually_burned"].values.astype(int)
    y_prob = parcels_with_prob["burn_prob_mean"].values

    # Handle NaN
    valid_mask = ~np.isnan(y_prob)
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    # Discrimination metrics
    discrimination = compute_discrimination_metrics(y_true, y_prob)

    # Calibration metrics
    calibration = compute_calibration_curve(y_true, y_prob)

    metrics = {
        **discrimination,
        "calibration_error": calibration["expected_calibration_error"],
        "n_parcels": len(y_true),
        "n_burned": y_true.sum(),
        "burn_rate": y_true.mean(),
    }

    return metrics


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict:
    """
    Compute calibration metrics.

    Parameters
    ----------
    y_true : ndarray
        True binary outcomes
    y_prob : ndarray
        Predicted probabilities
    n_bins : int
        Number of bins for calibration

    Returns
    -------
    dict
        Calibration metrics
    """
    from sklearn.calibration import calibration_curve

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    bin_sums = np.bincount(bin_indices, weights=y_true, minlength=n_bins)
    bin_prob_sums = np.bincount(bin_indices, weights=y_prob, minlength=n_bins)

    nonzero = bin_counts > 0
    observed = np.zeros(n_bins)
    predicted = np.zeros(n_bins)

    observed[nonzero] = bin_sums[nonzero] / bin_counts[nonzero]
    predicted[nonzero] = bin_prob_sums[nonzero] / bin_counts[nonzero]

    ece = np.sum(bin_counts * np.abs(observed - predicted)) / len(y_prob)

    # Maximum Calibration Error
    mce = np.max(np.abs(observed[nonzero] - predicted[nonzero]))

    return {
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(mce),
        "prob_true": prob_true,
        "prob_pred": prob_pred,
        "bin_counts": bin_counts,
    }


def generate_validation_report(
    results_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """
    Generate validation report with plots and tables.

    Parameters
    ----------
    results_df : DataFrame
        Hindcast validation results
    output_dir : Path
        Output directory

    Returns
    -------
    Path
        Path to report file
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return output_dir / "validation_report.txt"

    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    # AUC by year
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results_df["year"], results_df["auc_roc"])
    ax.axhline(y=0.85, color="r", linestyle="--", label="Target (0.85)")
    ax.set_xlabel("Year")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Burn Probability Model Performance by Year")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.savefig(report_dir / "auc_by_year.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Calibration error by year
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results_df["year"], results_df["calibration_error"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("Calibration Error by Year")
    plt.savefig(report_dir / "calibration_by_year.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Summary statistics
    summary = results_df.describe()
    summary.to_csv(report_dir / "summary_statistics.csv")

    # Text report
    with open(report_dir / "validation_report.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BURN PROBABILITY MODEL VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write("AGGREGATE METRICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean AUC-ROC: {results_df['auc_roc'].mean():.4f}\n")
        f.write(f"Std AUC-ROC: {results_df['auc_roc'].std():.4f}\n")
        f.write(f"Mean Calibration Error: {results_df['calibration_error'].mean():.4f}\n")
        f.write(f"Mean Recall: {results_df['recall'].mean():.4f}\n\n")

        f.write("PER-YEAR RESULTS\n")
        f.write("-" * 40 + "\n")
        for _, row in results_df.iterrows():
            f.write(
                f"Year {int(row['year'])}: "
                f"AUC={row['auc_roc']:.4f}, "
                f"Cal Error={row['calibration_error']:.4f}, "
                f"N fires={int(row.get('n_holdout_fires', 0))}\n"
            )

        f.write("\n" + "=" * 60 + "\n")

    return report_dir / "validation_report.txt"
