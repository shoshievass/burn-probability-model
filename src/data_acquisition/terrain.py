"""Download terrain data from USGS 3DEP."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import requests
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

from config.settings import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# USGS 3DEP Elevation Service
DEM_SERVICE_URL = (
    "https://elevation.nationalmap.gov/arcgis/rest/services/"
    "3DEPElevation/ImageServer"
)


def download_dem(
    bounds: Tuple[float, float, float, float],
    output_dir: Optional[Path] = None,
    resolution: int = 30,  # meters
    crs: str = "EPSG:3310",  # California Albers
) -> Path:
    """
    Download DEM from USGS 3DEP Elevation Service.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path, optional
        Directory to save data
    resolution : int
        Output resolution in meters
    crs : str
        Output CRS (default California Albers)

    Returns
    -------
    Path
        Path to downloaded DEM
    """
    output_dir = output_dir or RAW_DATA_DIR / "terrain"
    output_dir.mkdir(parents=True, exist_ok=True)

    minx, miny, maxx, maxy = bounds

    logger.info(f"Downloading DEM for bounds {bounds}...")

    # Calculate image size based on resolution
    # Approximate degrees to meters at California latitude
    lat_center = (miny + maxy) / 2
    deg_to_m_lat = 111000  # ~111 km per degree latitude
    deg_to_m_lon = 111000 * np.cos(np.radians(lat_center))

    width_m = (maxx - minx) * deg_to_m_lon
    height_m = (maxy - miny) * deg_to_m_lat

    # Image dimensions (cap at 8000 pixels for API limits)
    width_px = min(int(width_m / resolution), 8000)
    height_px = min(int(height_m / resolution), 8000)

    # Export image endpoint
    export_url = f"{DEM_SERVICE_URL}/exportImage"

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "4326",
        "size": f"{width_px},{height_px}",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    try:
        response = requests.get(export_url, params=params, timeout=300)
        response.raise_for_status()

        # Save raw DEM
        raw_path = output_dir / "dem_raw.tif"
        with open(raw_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded raw DEM to {raw_path}")

        # Reproject to target CRS
        output_path = output_dir / "dem.tif"
        reproject_raster(raw_path, output_path, crs, resolution)

        return output_path

    except requests.RequestException as e:
        logger.error(f"Failed to download DEM: {e}")
        raise


def reproject_raster(
    src_path: Path,
    dst_path: Path,
    dst_crs: str,
    resolution: Optional[int] = None,
) -> Path:
    """
    Reproject a raster to a new CRS.

    Parameters
    ----------
    src_path : Path
        Source raster path
    dst_path : Path
        Destination raster path
    dst_crs : str
        Target CRS
    resolution : int, optional
        Target resolution in CRS units

    Returns
    -------
    Path
        Path to reprojected raster
    """
    with rasterio.open(src_path) as src:
        # Calculate transform
        if resolution:
            transform, width, height = calculate_default_transform(
                src.crs,
                dst_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=(resolution, resolution),
            )
        else:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

        # Update profile
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
        )

        # Reproject
        with rasterio.open(dst_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

    logger.info(f"Reprojected raster to {dst_path}")
    return dst_path


def compute_slope(dem_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Compute slope from DEM.

    Parameters
    ----------
    dem_path : Path
        Path to DEM raster
    output_path : Path, optional
        Output path for slope raster

    Returns
    -------
    Path
        Path to slope raster
    """
    output_path = output_path or dem_path.parent / "slope.tif"

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        cell_size = src.transform[0]  # Assumes square cells

    # Handle nodata
    nodata = profile.get("nodata", -9999)
    mask = dem == nodata

    # Compute gradient
    dy, dx = np.gradient(dem, cell_size)

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope[mask] = nodata

    # Write output
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(slope, 1)

    logger.info(f"Computed slope, saved to {output_path}")
    return output_path


def compute_aspect(dem_path: Path, output_path: Optional[Path] = None) -> Path:
    """
    Compute aspect from DEM.

    Parameters
    ----------
    dem_path : Path
        Path to DEM raster
    output_path : Path, optional
        Output path for aspect raster

    Returns
    -------
    Path
        Path to aspect raster
    """
    output_path = output_path or dem_path.parent / "aspect.tif"

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        cell_size = src.transform[0]

    # Handle nodata
    nodata = profile.get("nodata", -9999)
    mask = dem == nodata

    # Compute gradient
    dy, dx = np.gradient(dem, cell_size)

    # Aspect in degrees (0-360, clockwise from north)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    aspect[mask] = nodata

    # Write output
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(aspect, 1)

    logger.info(f"Computed aspect, saved to {output_path}")
    return output_path


def compute_tpi(
    dem_path: Path,
    output_path: Optional[Path] = None,
    window_size: int = 500,  # meters
) -> Path:
    """
    Compute Topographic Position Index (TPI).

    TPI = elevation - mean elevation in neighborhood
    Positive = ridges/peaks, Negative = valleys

    Parameters
    ----------
    dem_path : Path
        Path to DEM raster
    output_path : Path, optional
        Output path for TPI raster
    window_size : int
        Neighborhood size in meters

    Returns
    -------
    Path
        Path to TPI raster
    """
    from scipy.ndimage import uniform_filter

    output_path = output_path or dem_path.parent / "tpi.tif"

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        cell_size = abs(src.transform[0])

    # Handle nodata
    nodata = profile.get("nodata", -9999)
    mask = dem == nodata
    dem[mask] = np.nan

    # Window size in pixels
    window_px = int(window_size / cell_size)
    if window_px < 3:
        window_px = 3

    # Compute mean elevation in neighborhood
    mean_elev = uniform_filter(dem, size=window_px, mode="nearest")

    # TPI = local elevation - neighborhood mean
    tpi = dem - mean_elev
    tpi[mask] = nodata

    # Write output
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(tpi, 1)

    logger.info(f"Computed TPI (window={window_size}m), saved to {output_path}")
    return output_path


def compute_all_terrain_derivatives(
    dem_path: Path,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Compute all terrain derivatives from DEM.

    Parameters
    ----------
    dem_path : Path
        Path to DEM raster
    output_dir : Path, optional
        Output directory

    Returns
    -------
    dict
        Mapping of derivative name to file path
    """
    output_dir = output_dir or dem_path.parent

    derivatives = {}

    derivatives["slope"] = compute_slope(dem_path, output_dir / "slope.tif")
    derivatives["aspect"] = compute_aspect(dem_path, output_dir / "aspect.tif")
    derivatives["tpi"] = compute_tpi(dem_path, output_dir / "tpi.tif")

    logger.info(f"Computed all terrain derivatives in {output_dir}")
    return derivatives


def load_dem(path: Optional[Path] = None) -> Tuple[np.ndarray, dict]:
    """Load DEM raster."""
    path = path or RAW_DATA_DIR / "terrain" / "dem.tif"

    with rasterio.open(path) as src:
        data = src.read(1)
        profile = src.profile.copy()

    return data, profile
