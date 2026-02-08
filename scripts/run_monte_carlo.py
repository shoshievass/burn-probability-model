#!/usr/bin/env python3
"""Run Monte Carlo burn probability simulation."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_monte_carlo(
    county: str = "Sonoma",
    year: int = 2020,
    iterations: int = 1000,
    cores: int = 28,
    extreme_fraction: float = 0.15,
    output_dir: Path = None,
    calibrate: bool = False,
    calibration_years: str = "2015-2022",
):
    """
    Run Monte Carlo burn probability simulation.

    Parameters
    ----------
    county : str
        County name
    year : int
        Prediction year
    iterations : int
        Number of Monte Carlo iterations
    cores : int
        Number of CPU cores
    extreme_fraction : float
        Fraction of extreme weather scenarios
    output_dir : Path
        Output directory
    """
    output_dir = output_dir or OUTPUT_DIR / "monte_carlo" / county.lower() / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MONTE CARLO BURN PROBABILITY SIMULATION")
    logger.info("=" * 60)
    logger.info(f"County: {county}")
    logger.info(f"Year: {year}")
    logger.info(f"Iterations: {iterations}")
    logger.info(f"Cores: {cores}")
    logger.info(f"Extreme weather fraction: {extreme_fraction}")
    logger.info(f"Output: {output_dir}")

    # Load landscape file
    landscape_path = PROCESSED_DATA_DIR / "landscape.lcp"
    if not landscape_path.exists():
        # Try to find alternative
        landscape_path = PROCESSED_DATA_DIR / "landscape.tif"

    if not landscape_path.exists():
        logger.warning("No landscape file found. Creating synthetic landscape...")
        from scripts.test_flammap import create_synthetic_landscape
        landscape_path = create_synthetic_landscape(output_dir / "landscape")

    logger.info(f"Using landscape: {landscape_path}")

    # Load ignition model if available
    ignition_model = None
    model_path = OUTPUT_DIR / "models" / "ignition_model_random_forest.joblib"

    if model_path.exists():
        logger.info(f"Loading ignition model: {model_path}")
        from src.ignition.models import RandomForestIgnition
        ignition_model = RandomForestIgnition.load(model_path)
    else:
        logger.warning("No ignition model found - using default ignition rates")

    # Load GridMET data if available
    import xarray as xr

    gridmet_ds = None
    gridmet_dir = RAW_DATA_DIR / "weather" / "gridmet"
    gridmet_files = list(gridmet_dir.glob("*.nc"))

    if gridmet_files:
        logger.info("Loading GridMET weather data...")
        gridmet_ds = xr.open_mfdataset(gridmet_files)
    else:
        logger.warning("No GridMET data found - using default weather scenarios")

    # Run Monte Carlo
    logger.info("=" * 60)
    logger.info("Starting Monte Carlo simulation...")
    logger.info("=" * 60)

    from src.integration.monte_carlo import (
        MonteCarloEngine,
        MonteCarloConfig,
        run_parallel_monte_carlo,
    )

    if cores > 1 and iterations >= 100:
        # Parallel execution
        logger.info(f"Running in parallel with {cores} workers...")
        result = run_parallel_monte_carlo(
            landscape_path=landscape_path,
            n_iterations=iterations,
            n_workers=cores,
            output_dir=output_dir,
            extreme_weather_fraction=extreme_fraction,
        )
    else:
        # Single-threaded execution
        config = MonteCarloConfig(
            n_iterations=iterations,
            n_cores=1,
            extreme_weather_fraction=extreme_fraction,
        )

        engine = MonteCarloEngine(
            landscape_path=landscape_path,
            ignition_model=ignition_model,
            gridmet_ds=gridmet_ds,
            config=config,
        )

        def progress_callback(frac):
            logger.info(f"Progress: {frac * 100:.0f}%")

        result = engine.run(progress_callback=progress_callback)

    # Calibrate if requested
    if calibrate:
        logger.info("=" * 60)
        logger.info("CALIBRATING PREDICTIONS")
        logger.info("=" * 60)

        # Load historical fire perimeters for calibration
        fire_perimeters_path = RAW_DATA_DIR / "fire_history" / "fire_perimeters.parquet"
        if fire_perimeters_path.exists():
            import geopandas as gpd
            from shapely.geometry import box
            from src.integration.monte_carlo import calibrate_predictions

            fires = gpd.read_parquet(fire_perimeters_path)

            # Filter to calibration years
            cal_start, cal_end = map(int, calibration_years.split("-"))
            year_col = "year_" if "year_" in fires.columns else "year"
            cal_fires = fires[(fires[year_col] >= cal_start) & (fires[year_col] <= cal_end)]

            # Filter to study area
            study_bounds = box(*result.bounds)
            cal_fires = cal_fires[cal_fires.geometry.intersects(study_bounds)]

            n_cal_years = cal_end - cal_start + 1
            logger.info(f"Calibration fires ({cal_start}-{cal_end}, {n_cal_years} years): {len(cal_fires)}")

            if len(cal_fires) > 0:
                result = calibrate_predictions(result, cal_fires, n_years=n_cal_years)
                logger.info(f"Calibration factor: {result.calibration_factor:.3f}")
            else:
                logger.warning("No fires in study area for calibration")
        else:
            logger.warning("No fire perimeters found for calibration")

    # Save burn probability raster
    output_path = output_dir / "burn_probability.tif"
    result.save(output_path)

    # Print summary statistics
    burn_prob = result.burn_probability

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Iterations completed: {result.n_iterations}")
    logger.info(f"Grid size: {burn_prob.shape}")
    logger.info(f"Mean burn probability: {burn_prob.mean():.6f}")
    logger.info(f"Max burn probability: {burn_prob.max():.6f}")
    logger.info(f"Cells with P > 0.01: {(burn_prob > 0.01).sum()}")
    logger.info(f"Cells with P > 0.10: {(burn_prob > 0.10).sum()}")

    # Aggregate to parcels if available
    parcels_path = RAW_DATA_DIR / "parcels" / f"{county.lower()}_parcels.parquet"

    if parcels_path.exists():
        logger.info("Aggregating to parcels...")
        import geopandas as gpd
        from src.integration.parcel_aggregation import aggregate_to_parcels

        parcels = gpd.read_parquet(parcels_path)
        parcels_with_prob = aggregate_to_parcels(
            burn_probability_path=output_path,
            parcels_gdf=parcels,
            output_path=output_dir / "parcels_burn_probability.parquet",
        )

        logger.info(f"Aggregated to {len(parcels_with_prob)} parcels")
        logger.info(f"Mean parcel burn probability: {parcels_with_prob['burn_prob_mean'].mean():.6f}")
        logger.info(f"High risk parcels (P > 0.10): {(parcels_with_prob['burn_prob_mean'] > 0.10).sum()}")

        # Save CSV summary
        summary_cols = ["apn", "burn_prob_mean", "burn_prob_max", "burn_prob_std"]
        available_cols = [c for c in summary_cols if c in parcels_with_prob.columns]
        parcels_with_prob[available_cols].to_csv(
            output_dir / "parcels_summary.csv", index=False
        )

    logger.info("=" * 60)
    logger.info(f"Output saved to: {output_dir}")
    logger.info("Monte Carlo simulation complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo burn probability simulation"
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
        help="Prediction year",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of Monte Carlo iterations",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=28,
        help="Number of CPU cores",
    )
    parser.add_argument(
        "--extreme-fraction",
        type=float,
        default=0.15,
        help="Fraction of extreme weather scenarios",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate predictions using historical fire data",
    )
    parser.add_argument(
        "--calibration-years",
        type=str,
        default="2015-2022",
        help="Years to use for calibration (e.g., 2015-2022)",
    )

    args = parser.parse_args()

    run_monte_carlo(
        county=args.county,
        year=args.year,
        iterations=args.iterations,
        cores=args.cores,
        extreme_fraction=args.extreme_fraction,
        output_dir=args.output_dir,
        calibrate=args.calibrate,
        calibration_years=args.calibration_years,
    )


if __name__ == "__main__":
    main()
