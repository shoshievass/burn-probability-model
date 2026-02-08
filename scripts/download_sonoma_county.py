#!/usr/bin/env python3
"""
Download expanded terrain and LANDFIRE data for full Sonoma County.

Uses USGS 3DEP and LANDFIRE REST APIs to download data covering:
- Sonoma County bounds: -123.6 to -122.2 longitude, 38.1 to 38.9 latitude
- Resolution: 270m (EPSG:3310 California Albers)
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, Optional
import requests
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Sonoma County bounds (WGS84)
SONOMA_BOUNDS = (-123.6, 38.1, -122.2, 38.9)

# API endpoints
USGS_DEM_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
LANDFIRE_BASE_URL = "https://lfps.usgs.gov/arcgis/rest/services/Landfire_LF230"

# LANDFIRE product codes
LANDFIRE_PRODUCTS = {
    "FBFM40": "FBFM40",  # 40 Scott & Burgan fuel models
    "CC": "CC",          # Canopy Cover
    "CH": "CH",          # Canopy Height
}

# Target resolution and CRS
TARGET_RESOLUTION = 270  # meters
TARGET_CRS = "EPSG:3310"  # California Albers


def download_usgs_dem(
    bounds: Tuple[float, float, float, float],
    output_dir: Path,
    image_size: Tuple[int, int] = (1000, 800),
) -> Path:
    """
    Download DEM from USGS 3DEP Elevation Service.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path
        Directory to save data
    image_size : tuple
        (width, height) in pixels

    Returns
    -------
    Path
        Path to downloaded and reprojected DEM
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = bounds

    logger.info(f"Downloading USGS 3DEP DEM for bounds {bounds}")
    logger.info(f"Image size: {image_size}")

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "4326",
        "size": f"{image_size[0]},{image_size[1]}",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "F32",
        "noData": "-9999",
        "interpolation": "RSP_BilinearInterpolation",
        "f": "image",
    }

    try:
        logger.info(f"Requesting from {USGS_DEM_URL}")
        response = requests.get(USGS_DEM_URL, params=params, timeout=300)
        response.raise_for_status()

        # Check if we got an image or an error
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type and 'tiff' not in content_type:
            logger.error(f"Unexpected response type: {content_type}")
            logger.error(f"Response: {response.text[:500]}")
            raise ValueError(f"Expected image, got {content_type}")

        # Save raw DEM
        raw_path = output_dir / "dem_raw.tif"
        with open(raw_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded raw DEM to {raw_path} ({len(response.content)} bytes)")

        # Reproject to California Albers at target resolution
        output_path = output_dir / "dem.tif"
        reproject_raster(raw_path, output_path, TARGET_CRS, TARGET_RESOLUTION)

        # Compute terrain derivatives
        compute_terrain_derivatives(output_path, output_dir)

        return output_path

    except requests.RequestException as e:
        logger.error(f"Failed to download DEM: {e}")
        raise


def download_landfire_product(
    product: str,
    bounds: Tuple[float, float, float, float],
    output_dir: Path,
    image_size: Tuple[int, int] = (1000, 800),
) -> Path:
    """
    Download LANDFIRE product via ImageServer exportImage.

    Parameters
    ----------
    product : str
        Product code (FBFM40, CC, CH)
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path
        Output directory
    image_size : tuple
        (width, height) in pixels

    Returns
    -------
    Path
        Path to downloaded and reprojected raster
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    minx, miny, maxx, maxy = bounds

    # LANDFIRE service URL
    service_url = f"{LANDFIRE_BASE_URL}/US_230{product}/ImageServer/exportImage"

    logger.info(f"Downloading LANDFIRE {product} for bounds {bounds}")
    logger.info(f"Service URL: {service_url}")

    params = {
        "bbox": f"{minx},{miny},{maxx},{maxy}",
        "bboxSR": "4326",
        "size": f"{image_size[0]},{image_size[1]}",
        "imageSR": "4326",
        "format": "tiff",
        "pixelType": "U8" if product == "FBFM40" else "U8",  # Fuel models and percentages
        "noData": "-128" if product == "FBFM40" else "255",
        "interpolation": "RSP_NearestNeighbor",  # Use nearest neighbor for categorical data
        "f": "image",
    }

    try:
        logger.info(f"Requesting {product}...")
        response = requests.get(service_url, params=params, timeout=300)
        response.raise_for_status()

        # Check response type
        content_type = response.headers.get('content-type', '')
        if 'image' not in content_type and 'tiff' not in content_type:
            logger.error(f"Unexpected response type: {content_type}")
            logger.error(f"Response: {response.text[:500]}")
            raise ValueError(f"Expected image, got {content_type}")

        # Save raw file
        raw_path = output_dir / f"{product}_raw.tif"
        with open(raw_path, "wb") as f:
            f.write(response.content)

        logger.info(f"Downloaded raw {product} to {raw_path} ({len(response.content)} bytes)")

        # Reproject to California Albers
        output_path = output_dir / f"{product}.tif"

        # Use nearest neighbor for categorical data (fuel models)
        resampling = Resampling.nearest if product == "FBFM40" else Resampling.bilinear
        reproject_raster(raw_path, output_path, TARGET_CRS, TARGET_RESOLUTION, resampling)

        return output_path

    except requests.RequestException as e:
        logger.error(f"Failed to download {product}: {e}")
        raise


def reproject_raster(
    src_path: Path,
    dst_path: Path,
    dst_crs: str,
    resolution: int,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """
    Reproject a raster to target CRS and resolution.

    Parameters
    ----------
    src_path : Path
        Source raster path
    dst_path : Path
        Destination raster path
    dst_crs : str
        Target CRS
    resolution : int
        Target resolution in meters
    resampling : Resampling
        Resampling method

    Returns
    -------
    Path
        Path to reprojected raster
    """
    with rasterio.open(src_path) as src:
        # Calculate new transform
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=(resolution, resolution),
        )

        # Update profile
        profile = src.profile.copy()
        profile.update(
            crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
        )

        logger.info(f"Reprojecting to {dst_crs} at {resolution}m resolution")
        logger.info(f"Output size: {width} x {height} pixels")

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
                    resampling=resampling,
                )

    logger.info(f"Saved reprojected raster to {dst_path}")
    return dst_path


def compute_terrain_derivatives(dem_path: Path, output_dir: Path) -> dict:
    """
    Compute slope, aspect, and TPI from DEM.

    Parameters
    ----------
    dem_path : Path
        Path to DEM raster
    output_dir : Path
        Output directory

    Returns
    -------
    dict
        Mapping of derivative name to file path
    """
    from scipy.ndimage import uniform_filter

    logger.info("Computing terrain derivatives...")

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        cell_size = abs(src.transform[0])

    nodata = profile.get("nodata", -9999)
    mask = (dem == nodata) | np.isnan(dem)

    # Slope
    dy, dx = np.gradient(dem, cell_size)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope[mask] = nodata

    slope_path = output_dir / "slope.tif"
    with rasterio.open(slope_path, "w", **profile) as dst:
        dst.write(slope, 1)
    logger.info(f"Saved slope to {slope_path}")

    # Aspect
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect = np.where(aspect < 0, aspect + 360, aspect)
    aspect[mask] = nodata

    aspect_path = output_dir / "aspect.tif"
    with rasterio.open(aspect_path, "w", **profile) as dst:
        dst.write(aspect, 1)
    logger.info(f"Saved aspect to {aspect_path}")

    # TPI (Topographic Position Index)
    dem_clean = dem.copy()
    dem_clean[mask] = np.nan

    window_m = 500  # meters
    window_px = max(3, int(window_m / cell_size))

    mean_elev = uniform_filter(dem_clean, size=window_px, mode="nearest")
    tpi = dem - mean_elev
    tpi[mask] = nodata

    tpi_path = output_dir / "tpi.tif"
    with rasterio.open(tpi_path, "w", **profile) as dst:
        dst.write(tpi, 1)
    logger.info(f"Saved TPI to {tpi_path}")

    return {
        "slope": slope_path,
        "aspect": aspect_path,
        "tpi": tpi_path,
    }


def create_combined_landscape(
    terrain_dir: Path,
    landfire_dir: Path,
    output_path: Path,
) -> Path:
    """
    Create combined landscape file with all bands.

    Combines:
    - Band 1: Elevation (DEM)
    - Band 2: Slope
    - Band 3: Aspect
    - Band 4: Fuel Model (FBFM40)
    - Band 5: Canopy Cover (CC)
    - Band 6: Canopy Height (CH)

    Parameters
    ----------
    terrain_dir : Path
        Directory with terrain rasters
    landfire_dir : Path
        Directory with LANDFIRE rasters
    output_path : Path
        Output file path

    Returns
    -------
    Path
        Path to combined landscape file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Creating combined landscape file...")

    # Define input files
    input_files = {
        "elevation": terrain_dir / "dem.tif",
        "slope": terrain_dir / "slope.tif",
        "aspect": terrain_dir / "aspect.tif",
        "fuel_model": landfire_dir / "FBFM40.tif",
        "canopy_cover": landfire_dir / "CC.tif",
        "canopy_height": landfire_dir / "CH.tif",
    }

    # Check all files exist
    for name, path in input_files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    # Read reference raster (DEM) for profile
    with rasterio.open(input_files["elevation"]) as src:
        ref_profile = src.profile.copy()
        ref_transform = src.transform
        ref_bounds = src.bounds
        ref_crs = src.crs
        ref_width = src.width
        ref_height = src.height

    logger.info(f"Reference grid: {ref_width} x {ref_height} pixels")
    logger.info(f"Bounds: {ref_bounds}")
    logger.info(f"CRS: {ref_crs}")

    # Update profile for multi-band output
    ref_profile.update(
        count=6,
        dtype="float32",
        compress="lzw",
    )

    # Band descriptions
    band_descriptions = [
        "Elevation (m)",
        "Slope (degrees)",
        "Aspect (degrees)",
        "Fuel Model (FBFM40)",
        "Canopy Cover (%)",
        "Canopy Height (m)",
    ]

    # Read and align all rasters
    bands = []
    for name in ["elevation", "slope", "aspect", "fuel_model", "canopy_cover", "canopy_height"]:
        src_path = input_files[name]

        with rasterio.open(src_path) as src:
            # Check if reprojection needed
            if src.crs != ref_crs or src.transform != ref_transform or \
               src.width != ref_width or src.height != ref_height:
                # Reproject to match reference
                data = np.empty((ref_height, ref_width), dtype=np.float32)
                resampling = Resampling.nearest if name == "fuel_model" else Resampling.bilinear

                reproject(
                    source=rasterio.band(src, 1),
                    destination=data,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling,
                )
            else:
                data = src.read(1).astype(np.float32)

        bands.append(data)
        logger.info(f"Added band: {name} (min={data.min():.1f}, max={data.max():.1f})")

    # Stack and write
    stacked = np.stack(bands, axis=0)

    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(stacked)
        for i, desc in enumerate(band_descriptions, 1):
            dst.set_band_description(i, desc)

    logger.info(f"Saved combined landscape to {output_path}")
    logger.info(f"Shape: {stacked.shape}")

    return output_path


def main():
    """Download all data for Sonoma County."""

    logger.info("=" * 60)
    logger.info("Downloading expanded data for Sonoma County")
    logger.info("=" * 60)
    logger.info(f"Bounds (WGS84): {SONOMA_BOUNDS}")
    logger.info(f"Target CRS: {TARGET_CRS}")
    logger.info(f"Target resolution: {TARGET_RESOLUTION}m")

    terrain_dir = RAW_DATA_DIR / "terrain"
    landfire_dir = RAW_DATA_DIR / "landfire"

    # Image size for full county at ~270m resolution
    # County is about 1.4 degrees x 0.8 degrees
    # At 38 degrees latitude, 1 degree longitude ~ 87km
    # Width: 1.4 * 87 = 122 km -> 122000 / 270 ~ 450 pixels
    # Height: 0.8 * 111 = 89 km -> 89000 / 270 ~ 330 pixels
    # Use larger size to ensure good coverage
    image_size = (1000, 800)

    # Step 1: Download USGS DEM
    logger.info("")
    logger.info("Step 1: Downloading USGS 3DEP DEM...")
    logger.info("-" * 40)

    try:
        dem_path = download_usgs_dem(
            bounds=SONOMA_BOUNDS,
            output_dir=terrain_dir,
            image_size=image_size,
        )
        logger.info(f"DEM downloaded: {dem_path}")
    except Exception as e:
        logger.error(f"Failed to download DEM: {e}")
        raise

    # Step 2: Download LANDFIRE products
    logger.info("")
    logger.info("Step 2: Downloading LANDFIRE products...")
    logger.info("-" * 40)

    landfire_paths = {}
    for product in LANDFIRE_PRODUCTS:
        try:
            path = download_landfire_product(
                product=product,
                bounds=SONOMA_BOUNDS,
                output_dir=landfire_dir,
                image_size=image_size,
            )
            landfire_paths[product] = path
            logger.info(f"{product} downloaded: {path}")
        except Exception as e:
            logger.error(f"Failed to download {product}: {e}")
            raise

    # Step 3: Create combined landscape file
    logger.info("")
    logger.info("Step 3: Creating combined landscape file...")
    logger.info("-" * 40)

    landscape_path = PROCESSED_DATA_DIR / "landscape.tif"

    try:
        create_combined_landscape(
            terrain_dir=terrain_dir,
            landfire_dir=landfire_dir,
            output_path=landscape_path,
        )
        logger.info(f"Combined landscape: {landscape_path}")
    except Exception as e:
        logger.error(f"Failed to create landscape: {e}")
        raise

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)

    # Print file info
    for name, path in [
        ("DEM", terrain_dir / "dem.tif"),
        ("Slope", terrain_dir / "slope.tif"),
        ("Aspect", terrain_dir / "aspect.tif"),
        ("FBFM40", landfire_dir / "FBFM40.tif"),
        ("CC", landfire_dir / "CC.tif"),
        ("CH", landfire_dir / "CH.tif"),
        ("Landscape", landscape_path),
    ]:
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            with rasterio.open(path) as src:
                logger.info(f"  {name}: {src.width}x{src.height} px, {size_mb:.2f} MB")
        else:
            logger.warning(f"  {name}: NOT FOUND")


if __name__ == "__main__":
    main()
