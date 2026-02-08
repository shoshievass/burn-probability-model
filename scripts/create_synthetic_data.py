#!/usr/bin/env python3
"""Create synthetic data for testing the burn probability pipeline."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

# Sonoma County approximate bounds in EPSG:3310
BOUNDS_ALBERS = (-245000, -15000, -180000, 50000)  # minx, miny, maxx, maxy
RESOLUTION = 270  # meters


def create_synthetic_dem():
    """Create synthetic DEM with realistic terrain."""
    print("Creating synthetic DEM...")

    minx, miny, maxx, maxy = BOUNDS_ALBERS
    ncols = int((maxx - minx) / RESOLUTION)
    nrows = int((maxy - miny) / RESOLUTION)

    transform = from_bounds(minx, miny, maxx, maxy, ncols, nrows)

    # Create elevation with mountain ridges
    np.random.seed(42)
    x = np.linspace(0, 1, ncols)
    y = np.linspace(0, 1, nrows)
    X, Y = np.meshgrid(x, y)

    # Base elevation with ridges
    elevation = (
        200 +  # Base
        400 * np.sin(X * 4 * np.pi) * np.cos(Y * 3 * np.pi) +  # Ridges
        300 * Y +  # North-south gradient
        100 * np.random.rand(nrows, ncols)  # Noise
    )

    output_dir = RAW_DATA_DIR / "terrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": np.float32,
        "width": ncols,
        "height": nrows,
        "count": 1,
        "crs": "EPSG:3310",
        "transform": transform,
        "nodata": -9999,
    }

    with rasterio.open(output_dir / "dem.tif", "w", **profile) as dst:
        dst.write(elevation.astype(np.float32), 1)

    # Compute slope
    dy, dx = np.gradient(elevation, RESOLUTION)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    with rasterio.open(output_dir / "slope.tif", "w", **profile) as dst:
        dst.write(slope.astype(np.float32), 1)

    # Compute aspect
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = np.where(aspect < 0, aspect + 360, aspect)

    with rasterio.open(output_dir / "aspect.tif", "w", **profile) as dst:
        dst.write(aspect.astype(np.float32), 1)

    # TPI
    from scipy.ndimage import uniform_filter
    mean_elev = uniform_filter(elevation, size=5)
    tpi = elevation - mean_elev

    with rasterio.open(output_dir / "tpi.tif", "w", **profile) as dst:
        dst.write(tpi.astype(np.float32), 1)

    print(f"  Created DEM: {nrows}x{ncols} at {RESOLUTION}m")
    return output_dir


def create_synthetic_fuel_model():
    """Create synthetic fuel model raster."""
    print("Creating synthetic fuel model...")

    minx, miny, maxx, maxy = BOUNDS_ALBERS
    ncols = int((maxx - minx) / RESOLUTION)
    nrows = int((maxy - miny) / RESOLUTION)

    transform = from_bounds(minx, miny, maxx, maxy, ncols, nrows)

    # Create fuel model distribution
    # Mix of chaparral (4), grass (1,3), timber (8,9,10)
    np.random.seed(43)

    # Base on elevation/position
    x = np.linspace(0, 1, ncols)
    y = np.linspace(0, 1, nrows)
    X, Y = np.meshgrid(x, y)

    fuel = np.zeros((nrows, ncols), dtype=np.int16)

    # Low elevation: grass (1, 3)
    low_mask = Y < 0.3
    fuel[low_mask] = np.random.choice([1, 3], size=low_mask.sum())

    # Mid elevation: chaparral (4, 5, 6)
    mid_mask = (Y >= 0.3) & (Y < 0.7)
    fuel[mid_mask] = np.random.choice([4, 5, 6], size=mid_mask.sum(), p=[0.6, 0.2, 0.2])

    # High elevation: timber (8, 9, 10)
    high_mask = Y >= 0.7
    fuel[high_mask] = np.random.choice([8, 9, 10], size=high_mask.sum())

    # Add some non-burnable (urban, water) patches
    non_burnable = np.random.rand(nrows, ncols) < 0.05
    fuel[non_burnable] = np.random.choice([91, 98], size=non_burnable.sum())

    output_dir = RAW_DATA_DIR / "landfire"
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "dtype": np.int16,
        "width": ncols,
        "height": nrows,
        "count": 1,
        "crs": "EPSG:3310",
        "transform": transform,
        "nodata": -1,
    }

    with rasterio.open(output_dir / "FBFM40.tif", "w", **profile) as dst:
        dst.write(fuel, 1)

    # Canopy cover (0-100%)
    canopy = np.where(
        np.isin(fuel, [8, 9, 10]),  # Timber
        np.random.randint(40, 90, size=(nrows, ncols)),
        np.where(
            np.isin(fuel, [4, 5, 6]),  # Chaparral
            np.random.randint(20, 60, size=(nrows, ncols)),
            np.random.randint(0, 20, size=(nrows, ncols))  # Grass
        )
    ).astype(np.int16)

    with rasterio.open(output_dir / "CC.tif", "w", **profile) as dst:
        dst.write(canopy, 1)

    print(f"  Created fuel model: {nrows}x{ncols}")
    return output_dir


def create_synthetic_fires():
    """Create synthetic fire perimeters and ignition points."""
    print("Creating synthetic fire history...")

    minx, miny, maxx, maxy = BOUNDS_ALBERS

    np.random.seed(44)

    fires = []
    for year in range(2010, 2024):
        # 5-15 fires per year
        n_fires = np.random.randint(5, 15)

        for i in range(n_fires):
            # Random center point
            cx = np.random.uniform(minx + 10000, maxx - 10000)
            cy = np.random.uniform(miny + 10000, maxy - 10000)

            # Random size (acres)
            size = np.random.exponential(500)

            # Convert to approximate radius
            radius = np.sqrt(size * 4047) / np.pi  # acres to m^2 to radius

            # Create irregular polygon
            n_points = np.random.randint(8, 15)
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            angles += np.random.rand(n_points) * 0.3
            radii = radius * (0.7 + 0.6 * np.random.rand(n_points))

            points = [(cx + r * np.cos(a), cy + r * np.sin(a))
                     for a, r in zip(angles, radii)]

            polygon = Polygon(points)

            fires.append({
                "fire_id": f"{year}_{i:03d}",
                "year_": year,
                "gis_acres": size,
                "fire_name": f"FIRE_{year}_{i}",
                "alarm_date": pd.Timestamp(f"{year}-{np.random.randint(5,11):02d}-{np.random.randint(1,28):02d}"),
                "cause": np.random.choice(["Lightning", "Human", "Unknown"]),
                "geometry": polygon,
            })

    fires_gdf = gpd.GeoDataFrame(fires, crs="EPSG:3310")

    output_dir = RAW_DATA_DIR / "fire_history"
    output_dir.mkdir(parents=True, exist_ok=True)

    fires_gdf.to_parquet(output_dir / "fire_perimeters.parquet")

    # Create ignition points (centroids)
    ignitions = fires_gdf.copy()
    ignitions["geometry"] = ignitions.geometry.centroid
    ignitions.to_parquet(output_dir / "ignition_points.parquet")

    print(f"  Created {len(fires_gdf)} fire perimeters")
    return fires_gdf


def create_synthetic_parcels():
    """Create synthetic parcel boundaries."""
    print("Creating synthetic parcels...")

    minx, miny, maxx, maxy = BOUNDS_ALBERS

    # Create grid of parcels (roughly 500m x 500m)
    parcel_size = 500

    parcels = []
    apn = 0

    for x in np.arange(minx, maxx, parcel_size):
        for y in np.arange(miny, maxy, parcel_size):
            # Add some variation to parcel sizes
            w = parcel_size * (0.8 + 0.4 * np.random.rand())
            h = parcel_size * (0.8 + 0.4 * np.random.rand())

            parcels.append({
                "apn": f"097-{apn // 1000:03d}-{apn % 1000:03d}",
                "geometry": box(x, y, x + w, y + h),
            })
            apn += 1

    parcels_gdf = gpd.GeoDataFrame(parcels, crs="EPSG:3310")

    output_dir = RAW_DATA_DIR / "parcels"
    output_dir.mkdir(parents=True, exist_ok=True)

    parcels_gdf.to_parquet(output_dir / "sonoma_parcels.parquet")

    print(f"  Created {len(parcels_gdf)} parcels")
    return parcels_gdf


def create_synthetic_infrastructure():
    """Create synthetic roads and power lines."""
    print("Creating synthetic infrastructure...")

    from shapely.geometry import LineString

    minx, miny, maxx, maxy = BOUNDS_ALBERS

    # Roads - grid pattern
    roads = []
    road_spacing = 5000  # 5km

    for x in np.arange(minx, maxx, road_spacing):
        roads.append({
            "road_id": len(roads),
            "geometry": LineString([(x, miny), (x, maxy)]),
        })

    for y in np.arange(miny, maxy, road_spacing):
        roads.append({
            "road_id": len(roads),
            "geometry": LineString([(minx, y), (maxx, y)]),
        })

    roads_gdf = gpd.GeoDataFrame(roads, crs="EPSG:3310")

    output_dir = RAW_DATA_DIR / "infrastructure"
    output_dir.mkdir(parents=True, exist_ok=True)

    roads_gdf.to_parquet(output_dir / "roads.parquet")

    # Power lines - fewer, diagonal
    power_lines = []
    for i in range(10):
        x1 = minx + (maxx - minx) * np.random.rand()
        y1 = miny
        x2 = x1 + (maxx - minx) * 0.3 * (np.random.rand() - 0.5)
        y2 = maxy

        power_lines.append({
            "line_id": i,
            "geometry": LineString([(x1, y1), (x2, y2)]),
        })

    power_gdf = gpd.GeoDataFrame(power_lines, crs="EPSG:3310")
    power_gdf.to_parquet(output_dir / "power_lines.parquet")

    print(f"  Created {len(roads_gdf)} road segments, {len(power_gdf)} power lines")
    return roads_gdf, power_gdf


def main():
    print("=" * 60)
    print("CREATING SYNTHETIC TEST DATA")
    print("=" * 60)

    create_synthetic_dem()
    create_synthetic_fuel_model()
    create_synthetic_fires()
    create_synthetic_parcels()
    create_synthetic_infrastructure()

    print("=" * 60)
    print("Synthetic data creation complete!")
    print(f"Data saved to: {RAW_DATA_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
