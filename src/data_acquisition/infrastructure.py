"""Download infrastructure data (roads, power lines)."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import requests
import geopandas as gpd
import numpy as np
from scipy.ndimage import distance_transform_edt
import rasterio
from rasterio import features

from config.settings import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# Data source URLs
ROADS_URL = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Roads/MapServer/0"
POWER_LINES_URL = (
    "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/"
    "Electric_Power_Transmission_Lines/FeatureServer/0"
)


def download_roads(
    bounds: Tuple[float, float, float, float],
    output_dir: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Download road network from Census TIGER.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path, optional
        Directory to save data

    Returns
    -------
    GeoDataFrame
        Road line geometries
    """
    output_dir = output_dir or RAW_DATA_DIR / "infrastructure"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "roads.parquet"

    if output_file.exists():
        logger.info(f"Loading existing roads from {output_file}")
        return gpd.read_parquet(output_file)

    logger.info("Downloading roads from Census TIGER...")

    params = {
        "where": "1=1",
        "geometry": f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outFields": "MTFCC,FULLNAME",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }

    all_features = []
    offset = 0
    chunk_size = 2000

    while True:
        params["resultOffset"] = offset
        params["resultRecordCount"] = chunk_size

        try:
            response = requests.get(
                f"{ROADS_URL}/query", params=params, timeout=120
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to download roads: {e}")
            break

        features_list = data.get("features", [])
        if not features_list:
            break

        all_features.extend(features_list)
        logger.info(f"Downloaded {len(all_features)} road segments...")

        if len(features_list) < chunk_size:
            break
        offset += chunk_size

    if not all_features:
        logger.warning("No roads found")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3310")

    gdf.to_parquet(output_file)
    logger.info(f"Saved {len(gdf)} roads to {output_file}")

    return gdf


def download_power_lines(
    bounds: Tuple[float, float, float, float],
    output_dir: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Download power transmission lines from HIFLD.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path, optional
        Directory to save data

    Returns
    -------
    GeoDataFrame
        Power line geometries
    """
    output_dir = output_dir or RAW_DATA_DIR / "infrastructure"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "power_lines.parquet"

    if output_file.exists():
        logger.info(f"Loading existing power lines from {output_file}")
        return gpd.read_parquet(output_file)

    logger.info("Downloading power lines from HIFLD...")

    params = {
        "where": "1=1",
        "geometry": f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "inSR": "4326",
        "outFields": "VOLTAGE,OWNER,STATUS",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }

    all_features = []
    offset = 0
    chunk_size = 2000

    while True:
        params["resultOffset"] = offset
        params["resultRecordCount"] = chunk_size

        try:
            response = requests.get(
                f"{POWER_LINES_URL}/query", params=params, timeout=120
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to download power lines: {e}")
            break

        features_list = data.get("features", [])
        if not features_list:
            break

        all_features.extend(features_list)
        logger.info(f"Downloaded {len(all_features)} power line segments...")

        if len(features_list) < chunk_size:
            break
        offset += chunk_size

    if not all_features:
        logger.warning("No power lines found")
        return gpd.GeoDataFrame()

    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    gdf = gdf.to_crs("EPSG:3310")

    gdf.to_parquet(output_file)
    logger.info(f"Saved {len(gdf)} power lines to {output_file}")

    return gdf


def compute_distance_raster(
    gdf: gpd.GeoDataFrame,
    template_path: Path,
    output_path: Path,
) -> Path:
    """
    Compute distance-to-feature raster.

    Parameters
    ----------
    gdf : GeoDataFrame
        Features to compute distance to
    template_path : Path
        Template raster for extent/resolution
    output_path : Path
        Output raster path

    Returns
    -------
    Path
        Path to distance raster
    """
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        height = src.height
        width = src.width

    # Rasterize features (1 where feature present, 0 elsewhere)
    shapes = [(geom, 1) for geom in gdf.geometry]

    if not shapes:
        # No features - return max distance
        distance = np.full((height, width), np.inf, dtype=np.float32)
    else:
        rasterized = features.rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # Compute distance transform
        # Distance is in pixels, convert to meters using cell size
        cell_size = abs(transform[0])
        distance = distance_transform_edt(rasterized == 0) * cell_size

    # Cap at reasonable max (50 km)
    distance = np.minimum(distance, 50000).astype(np.float32)

    # Write output
    profile.update(dtype=np.float32, nodata=-9999)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(distance, 1)

    logger.info(f"Computed distance raster: {output_path}")
    return output_path


def compute_distance_to_roads(
    bounds: Tuple[float, float, float, float],
    template_path: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Compute distance-to-roads raster."""
    output_dir = output_dir or RAW_DATA_DIR / "infrastructure"
    output_path = output_dir / "distance_to_roads.tif"

    roads = download_roads(bounds, output_dir)
    return compute_distance_raster(roads, template_path, output_path)


def compute_distance_to_power_lines(
    bounds: Tuple[float, float, float, float],
    template_path: Path,
    output_dir: Optional[Path] = None,
) -> Path:
    """Compute distance-to-power-lines raster."""
    output_dir = output_dir or RAW_DATA_DIR / "infrastructure"
    output_path = output_dir / "distance_to_power_lines.tif"

    power_lines = download_power_lines(bounds, output_dir)
    return compute_distance_raster(power_lines, template_path, output_path)


def load_roads(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load previously downloaded roads."""
    path = path or RAW_DATA_DIR / "infrastructure" / "roads.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Roads not found at {path}")
    return gpd.read_parquet(path)


def load_power_lines(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load previously downloaded power lines."""
    path = path or RAW_DATA_DIR / "infrastructure" / "power_lines.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Power lines not found at {path}")
    return gpd.read_parquet(path)
