"""Raster alignment and resampling utilities."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
)
from rasterio.mask import mask
from rasterio.merge import merge
import geopandas as gpd

from config.settings import PROCESSED_DATA_DIR, get_config

logger = logging.getLogger(__name__)


def align_rasters(
    raster_paths: List[Path],
    output_dir: Optional[Path] = None,
    target_crs: str = "EPSG:3310",
    target_resolution: int = 270,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resampling: Resampling = Resampling.bilinear,
) -> Dict[str, Path]:
    """
    Align multiple rasters to common grid.

    Parameters
    ----------
    raster_paths : list of Path
        Input raster files
    output_dir : Path, optional
        Output directory for aligned rasters
    target_crs : str
        Target coordinate reference system
    target_resolution : int
        Target resolution in meters
    bounds : tuple, optional
        (minx, miny, maxx, maxy) in target CRS
    resampling : Resampling
        Resampling method

    Returns
    -------
    dict
        Mapping of raster name to aligned file path
    """
    output_dir = output_dir or PROCESSED_DATA_DIR / "aligned"
    output_dir.mkdir(parents=True, exist_ok=True)

    aligned = {}

    # Determine common bounds if not provided
    if bounds is None:
        bounds = _compute_common_bounds(raster_paths, target_crs)

    logger.info(f"Aligning {len(raster_paths)} rasters to {target_resolution}m grid")

    for raster_path in raster_paths:
        name = raster_path.stem
        output_path = output_dir / f"{name}_aligned.tif"

        resample_raster(
            src_path=raster_path,
            dst_path=output_path,
            target_crs=target_crs,
            target_resolution=target_resolution,
            bounds=bounds,
            resampling=resampling,
        )

        aligned[name] = output_path

    return aligned


def resample_raster(
    src_path: Path,
    dst_path: Path,
    target_crs: str = "EPSG:3310",
    target_resolution: Optional[int] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """
    Resample a raster to target CRS and resolution.

    Parameters
    ----------
    src_path : Path
        Source raster path
    dst_path : Path
        Destination raster path
    target_crs : str
        Target CRS
    target_resolution : int, optional
        Target resolution in CRS units
    bounds : tuple, optional
        Target bounds in target CRS
    resampling : Resampling
        Resampling method

    Returns
    -------
    Path
        Path to resampled raster
    """
    with rasterio.open(src_path) as src:
        # Calculate transform
        if target_resolution and bounds:
            # Use exact bounds and resolution
            minx, miny, maxx, maxy = bounds
            width = int((maxx - minx) / target_resolution)
            height = int((maxy - miny) / target_resolution)

            transform = rasterio.transform.from_bounds(
                minx, miny, maxx, maxy, width, height
            )
        elif target_resolution:
            transform, width, height = calculate_default_transform(
                src.crs,
                target_crs,
                src.width,
                src.height,
                *src.bounds,
                resolution=(target_resolution, target_resolution),
            )
        else:
            transform, width, height = calculate_default_transform(
                src.crs, target_crs, src.width, src.height, *src.bounds
            )

        # Update profile
        profile = src.profile.copy()
        profile.update(
            crs=target_crs,
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
                    dst_crs=target_crs,
                    resampling=resampling,
                )

    logger.info(f"Resampled {src_path.name} -> {dst_path}")
    return dst_path


def create_aligned_stack(
    raster_dict: Dict[str, Path],
    output_path: Path,
    band_names: Optional[List[str]] = None,
) -> Path:
    """
    Create a multi-band raster stack from aligned rasters.

    Parameters
    ----------
    raster_dict : dict
        Mapping of band name to raster path
    output_path : Path
        Output stack path
    band_names : list, optional
        Order of bands in output

    Returns
    -------
    Path
        Path to output stack
    """
    band_names = band_names or list(raster_dict.keys())

    # Get profile from first raster
    first_path = raster_dict[band_names[0]]
    with rasterio.open(first_path) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width

    # Update for multi-band
    profile.update(count=len(band_names))

    # Write stack
    with rasterio.open(output_path, "w", **profile) as dst:
        for i, name in enumerate(band_names, 1):
            with rasterio.open(raster_dict[name]) as src:
                data = src.read(1)
                dst.write(data, i)
                dst.set_band_description(i, name)

    logger.info(f"Created raster stack with {len(band_names)} bands: {output_path}")
    return output_path


def clip_raster_to_boundary(
    raster_path: Path,
    boundary_gdf: gpd.GeoDataFrame,
    output_path: Path,
    all_touched: bool = False,
) -> Path:
    """
    Clip raster to vector boundary.

    Parameters
    ----------
    raster_path : Path
        Input raster path
    boundary_gdf : GeoDataFrame
        Boundary polygon(s) for clipping
    output_path : Path
        Output clipped raster path
    all_touched : bool
        Include all pixels touched by boundary

    Returns
    -------
    Path
        Path to clipped raster
    """
    with rasterio.open(raster_path) as src:
        # Reproject boundary if needed
        if boundary_gdf.crs != src.crs:
            boundary_gdf = boundary_gdf.to_crs(src.crs)

        # Clip
        out_image, out_transform = mask(
            src,
            boundary_gdf.geometry,
            crop=True,
            all_touched=all_touched,
        )

        # Update profile
        profile = src.profile.copy()
        profile.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
        )

    # Write output
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out_image)

    logger.info(f"Clipped {raster_path.name} to boundary: {output_path}")
    return output_path


def _compute_common_bounds(
    raster_paths: List[Path],
    target_crs: str,
) -> Tuple[float, float, float, float]:
    """Compute intersection of all raster bounds in target CRS."""
    from rasterio.warp import transform_bounds

    all_bounds = []

    for path in raster_paths:
        with rasterio.open(path) as src:
            bounds = transform_bounds(src.crs, target_crs, *src.bounds)
            all_bounds.append(bounds)

    # Compute intersection
    minx = max(b[0] for b in all_bounds)
    miny = max(b[1] for b in all_bounds)
    maxx = min(b[2] for b in all_bounds)
    maxy = min(b[3] for b in all_bounds)

    return (minx, miny, maxx, maxy)


def create_reference_grid(
    bounds: Tuple[float, float, float, float],
    resolution: int,
    crs: str,
    output_path: Path,
) -> Path:
    """
    Create a reference grid raster.

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in CRS units
    resolution : int
        Cell size in CRS units
    crs : str
        Coordinate reference system
    output_path : Path
        Output raster path

    Returns
    -------
    Path
        Path to reference grid
    """
    minx, miny, maxx, maxy = bounds

    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    # Create empty raster
    profile = {
        "driver": "GTiff",
        "dtype": np.uint8,
        "width": width,
        "height": height,
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": 0,
    }

    data = np.ones((height, width), dtype=np.uint8)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)

    logger.info(f"Created reference grid: {width}x{height} at {resolution}m")
    return output_path


def get_raster_info(raster_path: Path) -> dict:
    """Get metadata from a raster file."""
    with rasterio.open(raster_path) as src:
        return {
            "bounds": src.bounds,
            "crs": str(src.crs),
            "width": src.width,
            "height": src.height,
            "resolution": (src.transform[0], -src.transform[4]),
            "count": src.count,
            "dtype": str(src.dtypes[0]),
            "nodata": src.nodata,
        }
