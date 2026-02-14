#!/usr/bin/env python3
"""
Run Monte Carlo burn probability simulation for a single tile.

This script is called by the SLURM job array, once per tile.
Each tile is processed independently and can run in parallel.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import xarray as xr
from rasterio.transform import rowcol

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# California tile grid (100km x 100km tiles)
TILE_GRID = {
    "n_cols": 8,
    "n_rows": 7,
    "tile_size_m": 100000,  # 100 km
    "buffer_m": 10000,  # 10 km overlap
    "origin_x": -400000,  # Western edge in EPSG:3310
    "origin_y": -650000,  # Southern edge in EPSG:3310
}


def get_tile_bounds(tile_id: int) -> tuple:
    """Get bounds for a tile ID (1-indexed)."""
    tile_idx = tile_id - 1
    col = tile_idx % TILE_GRID["n_cols"]
    row = tile_idx // TILE_GRID["n_cols"]

    size = TILE_GRID["tile_size_m"]
    buffer = TILE_GRID["buffer_m"]
    origin_x = TILE_GRID["origin_x"]
    origin_y = TILE_GRID["origin_y"]

    minx = origin_x + col * size - buffer
    miny = origin_y + row * size - buffer
    maxx = origin_x + (col + 1) * size + buffer
    maxy = origin_y + (row + 1) * size + buffer

    return (minx, miny, maxx, maxy)


def extract_ignition_points(fires_gdf, landscape_path):
    """
    Extract burnable ignition points from fire perimeters.

    Uses fire centroid if burnable, otherwise samples within perimeter.
    """
    with rasterio.open(landscape_path) as src:
        fuel = src.read(4)  # Fuel model band
        transform = src.transform
        bounds = src.bounds

    def is_burnable(x, y):
        """Check if location has burnable fuel (codes 101-204)."""
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
        centroid = geometry.centroid
        if is_burnable(centroid.x, centroid.y):
            return (centroid.x, centroid.y)

        minx, miny, maxx, maxy = geometry.bounds
        for _ in range(max_attempts):
            x = np.random.uniform(minx, maxx)
            y = np.random.uniform(miny, maxy)
            from shapely.geometry import Point
            if geometry.contains(Point(x, y)) and is_burnable(x, y):
                return (x, y)
        return None

    ignition_points = []
    fire_ids = []

    for idx, row in fires_gdf.iterrows():
        point = find_burnable_point(row.geometry)
        if point is not None:
            ignition_points.append(point)
            fire_ids.append(idx)

    return ignition_points, fire_ids


def run_tile_simulation(
    tile_id: str,
    landscape_path: Path = None,
    fires_path: Path = None,
    output_path: Path = None,
    gridmet_path: Path = None,
    weather_samples: int = 100,
    simulation_hours: int = 24,
    extreme_fraction: float = 0.15,
    cores: int = 16,
    seed: int = 42,
):
    """
    Run Monte Carlo simulation for a single tile.

    Parameters
    ----------
    tile_id : str or int
        Tile identifier
    landscape_path : Path, optional
        Path to tile landscape raster (auto-generated if None)
    fires_path : Path, optional
        Path to tile fire perimeters (auto-loaded if None)
    output_path : Path, optional
        Path for output burn probability raster
    gridmet_path : Path, optional
        Path to GridMET weather data (NetCDF). If provided, samples weather
        from the empirical historical distribution instead of using defaults.
    weather_samples : int
        Number of weather scenarios per ignition
    simulation_hours : int
        Fire simulation duration in hours
    extreme_fraction : float
        Fraction of extreme weather samples
    cores : int
        Number of CPU cores to use
    seed : int
        Random seed for reproducibility
    """
    np.random.seed(seed + int(tile_id) if isinstance(tile_id, str) else seed + tile_id)

    tile_id_int = int(tile_id) if isinstance(tile_id, str) else tile_id
    tile_id_str = f"{tile_id_int:03d}"

    logger.info("=" * 60)
    logger.info(f"TILE {tile_id_str} SIMULATION")
    logger.info("=" * 60)

    # Set default paths if not provided
    if landscape_path is None:
        landscape_path = PROCESSED_DATA_DIR / "tiles" / f"tile_{tile_id_str}_landscape.tif"
    if fires_path is None:
        fires_path = PROCESSED_DATA_DIR / "tiles" / f"tile_{tile_id_str}_fires.parquet"
    if output_path is None:
        output_path = OUTPUT_DIR / "tiles" / f"tile_{tile_id_str}_burn_probability.tif"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check inputs exist
    if not Path(landscape_path).exists():
        logger.error(f"Landscape not found: {landscape_path}")
        return

    logger.info(f"Landscape: {landscape_path}")
    logger.info(f"Fires: {fires_path}")
    logger.info(f"Weather samples: {weather_samples}")
    logger.info(f"Simulation hours: {simulation_hours}")

    # Load fire perimeters
    if not Path(fires_path).exists():
        logger.warning(f"No fires file found: {fires_path}")
        logger.info("Creating empty output (no ignitions in tile)")
        with rasterio.open(landscape_path) as src:
            profile = src.profile
            profile.update(count=1, dtype='float32')
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(np.zeros((src.height, src.width), dtype='float32'), 1)
        return

    fires = gpd.read_parquet(fires_path)
    logger.info(f"Loaded {len(fires)} fire perimeters")

    if len(fires) == 0:
        logger.info("No fires in tile, creating empty output")
        with rasterio.open(landscape_path) as src:
            profile = src.profile
            profile.update(count=1, dtype='float32')
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(np.zeros((src.height, src.width), dtype='float32'), 1)
        return

    # Extract ignition points
    ignition_points, fire_ids = extract_ignition_points(fires, landscape_path)
    logger.info(f"Extracted {len(ignition_points)} burnable ignition points")

    if len(ignition_points) == 0:
        logger.warning("No burnable ignition points found")
        with rasterio.open(landscape_path) as src:
            profile = src.profile
            profile.update(count=1, dtype='float32')
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(np.zeros((src.height, src.width), dtype='float32'), 1)
        return

    # Load GridMET weather data for empirical sampling
    gridmet_ds = None
    if gridmet_path is None:
        # Try default location
        default_gridmet = RAW_DATA_DIR / "weather" / "gridmet" / "gridmet_2010_2024.nc"
        if default_gridmet.exists():
            gridmet_path = default_gridmet

    if gridmet_path is not None and Path(gridmet_path).exists():
        logger.info(f"Loading GridMET weather data: {gridmet_path}")
        gridmet_ds = xr.open_dataset(gridmet_path)
        logger.info("Weather sampling: EMPIRICAL (from historical GridMET data)")
    else:
        logger.warning("GridMET data not found - using default weather scenarios")
        logger.warning("For empirical weather sampling, provide --gridmet path")

    # Initialize Monte Carlo engine
    from src.integration.monte_carlo import MonteCarloEngine, MonteCarloConfig

    config = MonteCarloConfig(
        n_cores=cores,
        extreme_weather_fraction=extreme_fraction,
        simulation_duration_minutes=simulation_hours * 60,
    )

    engine = MonteCarloEngine(
        landscape_path=landscape_path,
        gridmet_ds=gridmet_ds,
        config=config,
    )

    # Run conditional simulation
    total_sims = len(ignition_points) * weather_samples
    logger.info(f"Running {len(ignition_points)} ignitions × {weather_samples} weather = {total_sims} simulations")

    def progress_callback(frac):
        pct = int(frac * 100)
        if pct % 10 == 0:
            logger.info(f"Progress: {pct}%")

    result = engine.run_conditional(
        ignition_points=ignition_points,
        n_weather_samples=weather_samples,
        progress_callback=progress_callback,
    )

    # Save output
    result.save(output_path)

    # Log summary
    burn_prob = result.burn_probability
    logger.info("=" * 60)
    logger.info(f"TILE {tile_id_str} COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total simulations: {result.n_iterations}")
    logger.info(f"Mean burn probability: {burn_prob.mean():.6f}")
    logger.info(f"Max burn probability: {burn_prob.max():.6f}")
    logger.info(f"Cells with P > 0.01: {(burn_prob > 0.01).sum():,}")
    logger.info(f"Output saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Monte Carlo simulation for a single tile"
    )
    parser.add_argument(
        "--tile-id",
        type=str,
        required=True,
        help="Tile identifier (e.g., 001 or 1)",
    )
    parser.add_argument(
        "--landscape",
        type=Path,
        default=None,
        help="Path to tile landscape raster",
    )
    parser.add_argument(
        "--fires",
        type=Path,
        default=None,
        help="Path to tile fire perimeters",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for output burn probability raster",
    )
    parser.add_argument(
        "--gridmet",
        type=Path,
        default=None,
        help="Path to GridMET weather NetCDF for empirical weather sampling",
    )
    parser.add_argument(
        "--weather-samples",
        type=int,
        default=100,
        help="Weather scenarios per ignition",
    )
    parser.add_argument(
        "--simulation-hours",
        type=int,
        default=24,
        help="Fire simulation duration (hours)",
    )
    parser.add_argument(
        "--extreme-fraction",
        type=float,
        default=0.15,
        help="Fraction of extreme weather samples",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=16,
        help="Number of CPU cores",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    run_tile_simulation(
        tile_id=args.tile_id,
        landscape_path=args.landscape,
        fires_path=args.fires,
        output_path=args.output,
        gridmet_path=args.gridmet,
        weather_samples=args.weather_samples,
        simulation_hours=args.simulation_hours,
        extreme_fraction=args.extreme_fraction,
        cores=args.cores,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
