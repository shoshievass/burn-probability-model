#!/usr/bin/env python3
"""Run conditional Monte Carlo simulation using real ignition events."""

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


def extract_ignition_points(fire_perimeters, bounds, landscape_path, crs="EPSG:3310"):
    """
    Extract ignition points from fire perimeters, ensuring they're in burnable cells.

    Uses the centroid of the fire perimeter, but checks that it's in a burnable
    fuel model. If not, samples a random point from within the fire perimeter
    that is burnable.
    """
    import geopandas as gpd
    import rasterio
    from rasterio.transform import rowcol
    from shapely.geometry import box, Point
    import numpy as np

    # Ensure correct CRS
    if fire_perimeters.crs != crs:
        fire_perimeters = fire_perimeters.to_crs(crs)

    # Filter to study area
    study_area = box(*bounds)
    fires_in_area = fire_perimeters[fire_perimeters.geometry.intersects(study_area)]

    # Load fuel model to check burnability
    with rasterio.open(landscape_path) as src:
        fuel = src.read(4)  # Fuel model band
        transform = src.transform

    def is_burnable(x, y):
        """Check if location is in a burnable fuel model (Scott/Burgan 101-204)."""
        try:
            row, col = rowcol(transform, x, y)
            if 0 <= row < fuel.shape[0] and 0 <= col < fuel.shape[1]:
                fuel_val = fuel[row, col]
                return 101 <= fuel_val <= 204
        except:
            pass
        return False

    def find_burnable_point(geometry, max_attempts=50):
        """Find a burnable point within the geometry."""
        # First try centroid
        centroid = geometry.centroid
        if is_burnable(centroid.x, centroid.y):
            return (centroid.x, centroid.y)

        # Sample random points within geometry
        minx, miny, maxx, maxy = geometry.bounds
        for _ in range(max_attempts):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            point = Point(x, y)
            if geometry.contains(point) and is_burnable(x, y):
                return (x, y)

        return None  # No burnable point found

    # Extract ignition points
    ignition_points = []
    fire_ids = []

    for idx, row in fires_in_area.iterrows():
        point = find_burnable_point(row.geometry)
        if point is not None:
            ignition_points.append(point)
            fire_ids.append(idx)

    return ignition_points, fire_ids


def run_conditional_monte_carlo(
    county: str = "Sonoma",
    start_year: int = 2015,
    end_year: int = 2022,
    weather_samples: int = 100,
    cores: int = 28,
    extreme_fraction: float = 0.15,
    output_dir: Path = None,
    use_holdout: bool = True,
):
    """
    Run conditional Monte Carlo using real ignition events.

    Parameters
    ----------
    county : str
        County name
    start_year, end_year : int
        Year range for ignition events
    weather_samples : int
        Number of weather scenarios per ignition
    cores : int
        Number of CPU cores
    extreme_fraction : float
        Fraction of extreme weather scenarios
    output_dir : Path
        Output directory
    use_holdout : bool
        If True, only use holdout fires (for validation)
    """
    import geopandas as gpd
    import numpy as np
    import rasterio

    output_dir = output_dir or OUTPUT_DIR / "conditional_mc" / county.lower() / f"{start_year}-{end_year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CONDITIONAL MONTE CARLO SIMULATION")
    logger.info("=" * 60)
    logger.info(f"County: {county}")
    logger.info(f"Years: {start_year}-{end_year}")
    logger.info(f"Weather samples per ignition: {weather_samples}")
    logger.info(f"Extreme weather fraction: {extreme_fraction}")
    logger.info(f"Use holdout fires only: {use_holdout}")
    logger.info(f"Output: {output_dir}")

    # Load landscape
    landscape_path = PROCESSED_DATA_DIR / "landscape.tif"
    if not landscape_path.exists():
        logger.error(f"Landscape not found: {landscape_path}")
        return

    with rasterio.open(landscape_path) as src:
        bounds = src.bounds
        crs = src.crs

    logger.info(f"Study area bounds: {bounds}")

    # Load fire perimeters
    fire_perimeters_path = RAW_DATA_DIR / "fire_history" / "fire_perimeters.parquet"
    if not fire_perimeters_path.exists():
        logger.error(f"Fire perimeters not found: {fire_perimeters_path}")
        return

    fires = gpd.read_parquet(fire_perimeters_path)
    logger.info(f"Loaded {len(fires)} fire perimeters")

    # Filter to year range
    year_col = "year_" if "year_" in fires.columns else "year"
    fires = fires[(fires[year_col] >= start_year) & (fires[year_col] <= end_year)]
    logger.info(f"Fires in {start_year}-{end_year}: {len(fires)}")

    # Optionally use holdout fires only
    if use_holdout:
        from src.validation.fire_holdout import split_fires_by_year
        _, holdout_fires = split_fires_by_year(fires, holdout_frac=0.30)
        fires = holdout_fires
        logger.info(f"Using holdout fires only: {len(fires)}")

    # Extract ignition points (filtered to burnable locations)
    ignition_points, fire_ids = extract_ignition_points(
        fires,
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top),
        landscape_path=landscape_path,
    )
    logger.info(f"Burnable ignition points in study area: {len(ignition_points)}")

    if len(ignition_points) == 0:
        logger.error("No ignition points in study area!")
        return

    # Initialize Monte Carlo engine
    from src.integration.monte_carlo import MonteCarloEngine, MonteCarloConfig

    config = MonteCarloConfig(
        n_cores=cores,
        extreme_weather_fraction=extreme_fraction,
    )

    engine = MonteCarloEngine(
        landscape_path=landscape_path,
        config=config,
    )

    # Run conditional simulation
    logger.info("=" * 60)
    logger.info("Running conditional Monte Carlo...")
    logger.info(f"Total simulations: {len(ignition_points)} × {weather_samples} = {len(ignition_points) * weather_samples}")
    logger.info("=" * 60)

    def progress_callback(frac):
        logger.info(f"Progress: {frac * 100:.0f}%")

    result = engine.run_conditional(
        ignition_points=ignition_points,
        n_weather_samples=weather_samples,
        progress_callback=progress_callback,
    )

    # Save burn probability raster
    output_path = output_dir / "burn_probability_conditional.tif"
    result.save(output_path)

    # Print summary statistics
    burn_prob = result.burn_probability

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Ignition events: {len(ignition_points)}")
    logger.info(f"Weather samples per ignition: {weather_samples}")
    logger.info(f"Total simulations: {result.n_iterations}")
    logger.info(f"Grid size: {burn_prob.shape}")
    logger.info(f"Mean burn probability: {burn_prob.mean():.6f}")
    logger.info(f"Max burn probability: {burn_prob.max():.6f}")
    logger.info(f"Cells with P > 0.01: {(burn_prob > 0.01).sum()}")
    logger.info(f"Cells with P > 0.10: {(burn_prob > 0.10).sum()}")

    # Aggregate to parcels
    parcels_path = RAW_DATA_DIR / "parcels" / f"{county.lower()}_parcels.parquet"

    if parcels_path.exists():
        logger.info("Aggregating to parcels...")
        import geopandas as gpd
        from src.integration.parcel_aggregation import aggregate_to_parcels

        parcels = gpd.read_parquet(parcels_path)
        parcels_with_prob = aggregate_to_parcels(
            burn_probability_path=output_path,
            parcels_gdf=parcels,
            output_path=output_dir / "parcels_burn_probability_conditional.parquet",
        )

        logger.info(f"Aggregated to {len(parcels_with_prob)} parcels")
        mean_prob = parcels_with_prob['burn_prob_mean'].mean()
        logger.info(f"Mean parcel burn probability: {mean_prob:.6f}")

        # Save CSV summary
        summary_cols = ["apn", "burn_prob_mean", "burn_prob_max", "burn_prob_std"]
        available_cols = [c for c in summary_cols if c in parcels_with_prob.columns]
        parcels_with_prob[available_cols].to_csv(
            output_dir / "parcels_summary_conditional.csv", index=False
        )

    logger.info("=" * 60)
    logger.info(f"Output saved to: {output_dir}")
    logger.info("Conditional Monte Carlo complete!")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run conditional Monte Carlo using real ignition events"
    )
    parser.add_argument(
        "--county",
        type=str,
        default="Sonoma",
        help="County name",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2015,
        help="Start year for ignitions",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2022,
        help="End year for ignitions",
    )
    parser.add_argument(
        "--weather-samples",
        type=int,
        default=100,
        help="Number of weather scenarios per ignition",
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
        "--all-fires",
        action="store_true",
        help="Use all fires instead of holdout only",
    )

    args = parser.parse_args()

    run_conditional_monte_carlo(
        county=args.county,
        start_year=args.start_year,
        end_year=args.end_year,
        weather_samples=args.weather_samples,
        cores=args.cores,
        extreme_fraction=args.extreme_fraction,
        output_dir=args.output_dir,
        use_holdout=not args.all_fires,
    )


if __name__ == "__main__":
    main()
