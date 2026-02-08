#!/usr/bin/env python3
"""Test FlamMap fire spread integration."""

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


def test_flammap(
    ignition_lat: float = 38.5,
    ignition_lon: float = -122.8,
    wind_speed: float = 10.0,
    wind_direction: float = 270.0,
    duration_minutes: int = 480,
    output_dir: Path = None,
):
    """
    Test FlamMap fire spread simulation.

    Parameters
    ----------
    ignition_lat, ignition_lon : float
        Ignition point (WGS84)
    wind_speed : float
        Wind speed in m/s
    wind_direction : float
        Wind direction (degrees from north)
    duration_minutes : int
        Simulation duration
    output_dir : Path
        Output directory
    """
    output_dir = output_dir or OUTPUT_DIR / "flammap_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Testing FlamMap integration")
    logger.info(f"Ignition point: ({ignition_lat}, {ignition_lon})")
    logger.info(f"Wind: {wind_speed} m/s from {wind_direction} degrees")
    logger.info(f"Duration: {duration_minutes} minutes")

    # Convert ignition point to California Albers
    import pyproj

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", "EPSG:3310", always_xy=True
    )
    ignition_x, ignition_y = transformer.transform(ignition_lon, ignition_lat)
    logger.info(f"Ignition point (EPSG:3310): ({ignition_x:.0f}, {ignition_y:.0f})")

    # Check for landscape file
    landscape_path = PROCESSED_DATA_DIR / "landscape.lcp"

    if not landscape_path.exists():
        logger.info("Creating landscape file from source data...")
        landscape_path = create_test_landscape(output_dir)

    if not landscape_path.exists():
        logger.error("Could not create landscape file. Check that terrain/LANDFIRE data exists.")
        return

    # Run fire spread simulation
    logger.info("Running fire spread simulation...")
    from src.spread.flammap_wrapper import run_basic_fire_spread

    result = run_basic_fire_spread(
        landscape_path=landscape_path,
        ignition_point=(ignition_x, ignition_y),
        duration_minutes=duration_minutes,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
    )

    # Save results
    burned_area = result["burned_area"]
    n_burned = burned_area.sum()
    total_cells = burned_area.size

    logger.info(f"Simulation complete!")
    logger.info(f"Burned cells: {n_burned} / {total_cells} ({100 * n_burned / total_cells:.2f}%)")

    # Save burned area raster
    import rasterio
    import numpy as np

    output_raster = output_dir / "burned_area.tif"

    profile = {
        "driver": "GTiff",
        "dtype": np.uint8,
        "width": burned_area.shape[1],
        "height": burned_area.shape[0],
        "count": 1,
        "crs": "EPSG:3310",
        "transform": result.get("transform"),
        "nodata": 255,
    }

    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(burned_area.astype(np.uint8), 1)

    logger.info(f"Saved burned area to: {output_raster}")

    # Compute fire statistics
    if "arrival_time" in result:
        arrival = result["arrival_time"]
        valid_arrival = arrival[~np.isnan(arrival)]
        if len(valid_arrival) > 0:
            logger.info(f"Mean arrival time: {np.mean(valid_arrival):.0f} minutes")
            logger.info(f"Max spread distance: {np.max(valid_arrival):.0f} minutes")

    logger.info("=" * 50)
    logger.info("FlamMap integration test complete!")


def create_test_landscape(output_dir: Path) -> Path:
    """Create a test landscape file from available data."""
    from src.spread.landscape_builder import LandscapeBuilder
    import numpy as np
    import rasterio

    logger.info("Creating test landscape...")

    # Check for DEM
    dem_path = RAW_DATA_DIR / "terrain" / "dem.tif"
    if not dem_path.exists():
        logger.warning(f"DEM not found at {dem_path}")
        # Create synthetic landscape for testing
        return create_synthetic_landscape(output_dir)

    # Load DEM to get bounds
    with rasterio.open(dem_path) as src:
        bounds = src.bounds
        resolution = int(abs(src.transform[0]))

    # Use a subset for testing
    minx = bounds.left
    miny = bounds.bottom
    maxx = min(bounds.right, minx + 20000)  # 20km x 20km
    maxy = min(bounds.top, miny + 20000)

    test_bounds = (minx, miny, maxx, maxy)

    logger.info(f"Creating landscape for bounds: {test_bounds}")

    # Check for fuel model
    fuel_path = None
    for pattern in ["*FBFM40*.tif", "*FBFM13*.tif"]:
        files = list((RAW_DATA_DIR / "landfire").glob(pattern))
        if files:
            fuel_path = files[0]
            break

    if fuel_path is None:
        logger.warning("No fuel model found, using synthetic")
        return create_synthetic_landscape(output_dir)

    # Build landscape
    from src.spread.landscape_builder import build_landscape_from_sources

    slope_path = RAW_DATA_DIR / "terrain" / "slope.tif"
    aspect_path = RAW_DATA_DIR / "terrain" / "aspect.tif"

    if not slope_path.exists() or not aspect_path.exists():
        logger.info("Computing terrain derivatives...")
        from src.data_acquisition.terrain import compute_all_terrain_derivatives
        compute_all_terrain_derivatives(dem_path)

    output_path = output_dir / "landscape.lcp"

    try:
        landscape_path = build_landscape_from_sources(
            dem_path=dem_path,
            fuel_model_path=fuel_path,
            bounds=test_bounds,
            output_path=output_path,
            resolution=resolution,
        )
        return landscape_path
    except Exception as e:
        logger.error(f"Failed to build landscape: {e}")
        return create_synthetic_landscape(output_dir)


def create_synthetic_landscape(output_dir: Path) -> Path:
    """Create a synthetic landscape for testing."""
    import numpy as np
    import rasterio

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Creating synthetic landscape for testing...")

    # Create a 100x100 cell landscape at 270m resolution
    nrows, ncols = 100, 100
    resolution = 270

    # Bounds (arbitrary but in California Albers range)
    minx, miny = -200000, 50000
    maxx = minx + ncols * resolution
    maxy = miny + nrows * resolution

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, ncols, nrows)

    # Create layers
    np.random.seed(42)

    # Elevation: 500-1500m with some variation
    elevation = 500 + 1000 * np.random.rand(nrows, ncols)

    # Slope: 0-30 degrees
    slope = 30 * np.random.rand(nrows, ncols)

    # Aspect: 0-360 degrees
    aspect = 360 * np.random.rand(nrows, ncols)

    # Fuel model: mostly Anderson model 4 (chaparral) with some grass
    fuel_model = np.random.choice([1, 4, 5, 6], size=(nrows, ncols), p=[0.1, 0.6, 0.2, 0.1])

    # Save as individual rasters (simpler than LCP for testing)
    for name, data in [
        ("elevation", elevation),
        ("slope", slope),
        ("aspect", aspect),
        ("fuel_model", fuel_model),
    ]:
        output_path = output_dir / f"{name}.tif"
        profile = {
            "driver": "GTiff",
            "dtype": data.dtype,
            "width": ncols,
            "height": nrows,
            "count": 1,
            "crs": "EPSG:3310",
            "transform": transform,
            "nodata": -9999,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(data, 1)

    # Create a combined "landscape" file (simplified)
    landscape_path = output_dir / "landscape.tif"
    profile = {
        "driver": "GTiff",
        "dtype": np.float32,
        "width": ncols,
        "height": nrows,
        "count": 4,
        "crs": "EPSG:3310",
        "transform": transform,
        "nodata": -9999,
    }

    with rasterio.open(landscape_path, "w", **profile) as dst:
        dst.write(elevation.astype(np.float32), 1)
        dst.write(slope.astype(np.float32), 2)
        dst.write(aspect.astype(np.float32), 3)
        dst.write(fuel_model.astype(np.float32), 4)

    logger.info(f"Created synthetic landscape: {landscape_path}")
    return landscape_path


def main():
    parser = argparse.ArgumentParser(
        description="Test FlamMap fire spread integration"
    )
    parser.add_argument(
        "--ignition-point",
        type=str,
        default="38.5,-122.8",
        help="Ignition point as lat,lon",
    )
    parser.add_argument(
        "--wind-speed",
        type=float,
        default=10.0,
        help="Wind speed (m/s)",
    )
    parser.add_argument(
        "--wind-direction",
        type=float,
        default=270.0,
        help="Wind direction (degrees from north)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=480,
        help="Simulation duration (minutes)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Parse ignition point
    lat, lon = map(float, args.ignition_point.split(","))

    test_flammap(
        ignition_lat=lat,
        ignition_lon=lon,
        wind_speed=args.wind_speed,
        wind_direction=args.wind_direction,
        duration_minutes=args.duration,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
