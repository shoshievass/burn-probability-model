#!/usr/bin/env python3
"""Validate burn probability results against holdout fires."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAW_DATA_DIR, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_results(
    county: str = "Sonoma",
    year: int = 2020,
    predictions_dir: Path = None,
    output_dir: Path = None,
):
    """
    Validate burn probability predictions against holdout fires.

    Parameters
    ----------
    county : str
        County name
    year : int
        Year to validate
    predictions_dir : Path
        Directory containing predictions
    output_dir : Path
        Output directory for validation results
    """
    predictions_dir = predictions_dir or OUTPUT_DIR / "monte_carlo" / county.lower() / str(year)
    output_dir = output_dir or OUTPUT_DIR / "validation" / county.lower() / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BURN PROBABILITY VALIDATION")
    logger.info("=" * 60)
    logger.info(f"County: {county}")
    logger.info(f"Year: {year}")
    logger.info(f"Predictions: {predictions_dir}")

    # Load burn probability
    burn_prob_path = predictions_dir / "burn_probability.tif"
    if not burn_prob_path.exists():
        raise FileNotFoundError(f"Burn probability not found: {burn_prob_path}")

    # Load fire perimeters
    fire_perimeters_path = RAW_DATA_DIR / "fire_history" / "fire_perimeters.parquet"
    if not fire_perimeters_path.exists():
        raise FileNotFoundError(f"Fire perimeters not found: {fire_perimeters_path}")

    import geopandas as gpd

    fire_perimeters = gpd.read_parquet(fire_perimeters_path)
    logger.info(f"Loaded {len(fire_perimeters)} fire perimeters")

    # Get holdout fires for validation
    from src.validation.fire_holdout import split_fires_by_year

    _, holdout_fires = split_fires_by_year(fire_perimeters, holdout_frac=0.30)

    # Filter to year
    year_col = "year_" if "year_" in holdout_fires.columns else "year"
    year_holdout = holdout_fires[holdout_fires[year_col] == year]
    logger.info(f"Holdout fires for {year}: {len(year_holdout)}")

    if len(year_holdout) == 0:
        logger.warning(f"No holdout fires for {year}")
        return

    # Load parcels
    parcels_path = predictions_dir / "parcels_burn_probability.parquet"

    if parcels_path.exists():
        parcels = gpd.read_parquet(parcels_path)
    else:
        # Load raw parcels and aggregate
        raw_parcels_path = RAW_DATA_DIR / "parcels" / f"{county.lower()}_parcels.parquet"
        if not raw_parcels_path.exists():
            logger.error("No parcels found for validation")
            return

        parcels = gpd.read_parquet(raw_parcels_path)
        from src.integration.parcel_aggregation import aggregate_to_parcels

        parcels = aggregate_to_parcels(burn_prob_path, parcels)

    logger.info(f"Loaded {len(parcels)} parcels")

    # Add validation data
    from src.integration.parcel_aggregation import add_validation_data

    parcels = add_validation_data(parcels, year_holdout, year)
    n_burned = parcels["actually_burned"].sum()
    logger.info(f"Parcels actually burned: {n_burned}")

    # Compute validation metrics
    from src.validation.metrics import (
        compute_discrimination_metrics,
        compute_calibration_curve,
        compute_skill_scores,
        compute_roc_curve,
        compute_reliability_diagram,
    )

    import numpy as np

    y_true = parcels["actually_burned"].values.astype(int)
    y_prob = parcels["burn_prob_mean"].values

    # Filter valid
    valid = ~np.isnan(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    logger.info("=" * 60)
    logger.info("DISCRIMINATION METRICS")
    logger.info("=" * 60)

    disc_metrics = compute_discrimination_metrics(y_true, y_prob)
    for name, value in disc_metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    logger.info("=" * 60)
    logger.info("CALIBRATION METRICS")
    logger.info("=" * 60)

    cal_metrics = compute_calibration_curve(y_true, y_prob)
    logger.info(f"  Expected Calibration Error: {cal_metrics['expected_calibration_error']:.4f}")

    logger.info("=" * 60)
    logger.info("SKILL SCORES")
    logger.info("=" * 60)

    skill = compute_skill_scores(y_true, y_prob)
    for name, value in skill.items():
        logger.info(f"  {name}: {value:.4f}")

    # Generate plots
    logger.info("=" * 60)
    logger.info("Generating plots...")

    compute_roc_curve(y_true, y_prob, output_path=output_dir / "roc_curve.png")
    compute_reliability_diagram(y_true, y_prob, output_path=output_dir / "reliability_diagram.png")

    # Save metrics
    import pandas as pd

    all_metrics = {
        **disc_metrics,
        "expected_calibration_error": cal_metrics["expected_calibration_error"],
        **skill,
        "year": year,
        "n_parcels": len(y_true),
        "n_burned": n_burned,
        "burn_rate": y_true.mean(),
    }

    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False)

    # Check against targets
    from config.settings import get_config

    config = get_config()

    logger.info("=" * 60)
    logger.info("TARGET CHECK")
    logger.info("=" * 60)

    auc_target = config.validation.burn_prob_auc_target
    if disc_metrics["auc_roc"] >= auc_target:
        logger.info(f"AUC-ROC: {disc_metrics['auc_roc']:.4f} >= {auc_target} - PASS")
    else:
        logger.warning(f"AUC-ROC: {disc_metrics['auc_roc']:.4f} < {auc_target} - FAIL")

    # Calibration check
    if cal_metrics["expected_calibration_error"] < 0.05:
        logger.info(f"Calibration Error: {cal_metrics['expected_calibration_error']:.4f} < 0.05 - PASS")
    else:
        logger.warning(f"Calibration Error: {cal_metrics['expected_calibration_error']:.4f} >= 0.05 - NEEDS CALIBRATION")

    logger.info("=" * 60)
    logger.info(f"Validation results saved to: {output_dir}")
    logger.info("Validation complete!")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Validate burn probability results"
    )
    parser.add_argument(
        "--county",
        type=str,
        default="Sonoma",
        help="County name",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2020,
        help="Year to validate",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Directory containing predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    validate_results(
        county=args.county,
        year=args.year,
        predictions_dir=args.predictions_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
