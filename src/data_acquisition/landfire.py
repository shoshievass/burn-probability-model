"""Download LANDFIRE fuel and vegetation data."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
import requests
import tempfile
import zipfile
import rasterio
from rasterio.merge import merge
import numpy as np

from config.settings import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# LANDFIRE Product Service URL
LFPS_URL = "https://lfps.usgs.gov/arcgis/rest/services/LandfireProductService/GPServer"
LANDFIRE_DOWNLOAD_URL = "https://landfire.gov/bulk/downloadfile.php"

# Product codes
LANDFIRE_PRODUCTS = {
    "FBFM13": "13 Anderson Fire Behavior Fuel Models",
    "FBFM40": "40 Scott & Burgan Fire Behavior Fuel Models",
    "CC": "Canopy Cover",
    "CH": "Canopy Height",
    "CBH": "Canopy Base Height",
    "CBD": "Canopy Bulk Density",
    "ASP": "Aspect",
    "SLP": "Slope",
    "ELEV": "Elevation",
}


def download_landfire_products(
    products: Optional[List[str]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    output_dir: Optional[Path] = None,
    version: str = "230",  # LANDFIRE 2023
) -> dict:
    """
    Download LANDFIRE products for a region.

    Note: Due to LANDFIRE's authentication requirements, this function
    provides download instructions and handles locally available data.
    For automated download, use the LANDFIRE Data Access Tool (LFDAT).

    Parameters
    ----------
    products : list of str, optional
        Product codes to download. Default: all fuel-related products
    bounds : tuple, optional
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path, optional
        Directory to save data
    version : str
        LANDFIRE version (e.g., "230" for 2023)

    Returns
    -------
    dict
        Mapping of product code to file path
    """
    output_dir = output_dir or RAW_DATA_DIR / "landfire"
    output_dir.mkdir(parents=True, exist_ok=True)

    if products is None:
        products = ["FBFM40", "CC", "CH", "CBH", "CBD"]

    # Default to California bounds
    if bounds is None:
        bounds = (-124.5, 32.5, -114.0, 42.1)

    logger.info(f"Preparing LANDFIRE download for products: {products}")

    # Check if data already exists
    existing = {}
    for product in products:
        pattern = f"*{product}*.tif"
        files = list(output_dir.glob(pattern))
        if files:
            existing[product] = files[0]
            logger.info(f"Found existing {product} at {files[0]}")

    # For products not found, provide download instructions
    missing = [p for p in products if p not in existing]

    if missing:
        logger.info(
            f"\nMissing LANDFIRE products: {missing}\n"
            f"To download, use one of these methods:\n"
            f"1. LANDFIRE Data Download Tool: https://landfire.gov/lf_remap.php\n"
            f"2. LANDFIRE Bulk Downloader\n"
            f"3. Direct API call (see download_landfire_via_api)\n"
            f"\nBounds (WGS84): {bounds}\n"
            f"Version: LF{version}\n"
        )

        # Attempt API download for missing products
        for product in missing:
            try:
                filepath = download_landfire_via_api(
                    product=product,
                    bounds=bounds,
                    output_dir=output_dir,
                    version=version,
                )
                if filepath:
                    existing[product] = filepath
            except Exception as e:
                logger.warning(f"API download failed for {product}: {e}")

    return existing


def download_landfire_via_api(
    product: str,
    bounds: Tuple[float, float, float, float],
    output_dir: Path,
    version: str = "230",
) -> Optional[Path]:
    """
    Download LANDFIRE product via REST API.

    This uses the LANDFIRE Product Service (LFPS) REST endpoint.

    Parameters
    ----------
    product : str
        Product code (e.g., "FBFM40")
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path
        Output directory
    version : str
        LANDFIRE version

    Returns
    -------
    Path or None
        Path to downloaded file, or None if failed
    """
    minx, miny, maxx, maxy = bounds

    # LFPS submit job endpoint
    submit_url = f"{LFPS_URL}/submitJob"

    # Build request parameters
    params = {
        "Layer_List": product,
        "Area_of_Interest": f"{minx},{miny},{maxx},{maxy}",
        "Output_Projection": "3310",  # California Albers
        "Output_Format": "GeoTIFF",
        "f": "json",
    }

    logger.info(f"Submitting LANDFIRE download job for {product}...")

    try:
        # Submit job
        response = requests.get(submit_url, params=params, timeout=60)
        response.raise_for_status()
        job_info = response.json()

        job_id = job_info.get("jobId")
        if not job_id:
            logger.error(f"No job ID returned: {job_info}")
            return None

        # Check job status
        status_url = f"{LFPS_URL}/jobs/{job_id}"
        import time

        for _ in range(60):  # Wait up to 10 minutes
            status_response = requests.get(status_url, params={"f": "json"}, timeout=30)
            status_data = status_response.json()

            status = status_data.get("jobStatus")
            if status == "esriJobSucceeded":
                break
            elif status in ("esriJobFailed", "esriJobCancelled"):
                logger.error(f"Job failed: {status_data}")
                return None

            logger.info(f"Job status: {status}, waiting...")
            time.sleep(10)
        else:
            logger.error("Job timed out")
            return None

        # Get results
        results = status_data.get("results", {})
        output_file_info = results.get("Output_File", {})

        if output_file_info:
            download_url = output_file_info.get("paramUrl")
            if download_url:
                # Download the file
                download_response = requests.get(download_url, timeout=300)
                download_response.raise_for_status()

                output_path = output_dir / f"{product}_LF{version}.tif"

                # Handle zip file
                if download_url.endswith(".zip"):
                    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                        tmp.write(download_response.content)
                        tmp_path = Path(tmp.name)

                    with zipfile.ZipFile(tmp_path, "r") as zf:
                        for name in zf.namelist():
                            if name.endswith(".tif"):
                                zf.extract(name, output_dir)
                                extracted = output_dir / name
                                extracted.rename(output_path)
                                break

                    tmp_path.unlink()
                else:
                    with open(output_path, "wb") as f:
                        f.write(download_response.content)

                logger.info(f"Downloaded {product} to {output_path}")
                return output_path

    except requests.RequestException as e:
        logger.error(f"Download request failed: {e}")

    return None


def load_landfire_raster(
    product: str,
    path: Optional[Path] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Load a LANDFIRE raster product.

    Parameters
    ----------
    product : str
        Product code (e.g., "FBFM40")
    path : Path, optional
        Path to LANDFIRE directory or file

    Returns
    -------
    tuple
        (data array, rasterio profile)
    """
    path = path or RAW_DATA_DIR / "landfire"

    if path.is_dir():
        files = list(path.glob(f"*{product}*.tif"))
        if not files:
            raise FileNotFoundError(f"No {product} file found in {path}")
        filepath = files[0]
    else:
        filepath = path

    with rasterio.open(filepath) as src:
        data = src.read(1)
        profile = src.profile.copy()

    return data, profile


def get_fuel_model_at_point(
    lat: float,
    lon: float,
    fuel_raster_path: Optional[Path] = None,
) -> int:
    """
    Get fuel model code at a specific point.

    Parameters
    ----------
    lat, lon : float
        Point coordinates (WGS84)
    fuel_raster_path : Path, optional
        Path to FBFM raster

    Returns
    -------
    int
        Fuel model code
    """
    import pyproj

    # Load fuel model raster
    data, profile = load_landfire_raster("FBFM40", fuel_raster_path)

    # Convert coordinates to raster CRS
    src_crs = pyproj.CRS("EPSG:4326")
    dst_crs = pyproj.CRS(profile["crs"])
    transformer = pyproj.Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    x, y = transformer.transform(lon, lat)

    # Get pixel coordinates
    transform = profile["transform"]
    col = int((x - transform.c) / transform.a)
    row = int((y - transform.f) / transform.e)

    # Check bounds
    if 0 <= row < data.shape[0] and 0 <= col < data.shape[1]:
        return int(data[row, col])
    else:
        return -1  # Out of bounds


def create_fuel_mosaic(
    input_files: List[Path],
    output_path: Path,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Path:
    """
    Mosaic multiple LANDFIRE tiles into a single raster.

    Parameters
    ----------
    input_files : list of Path
        Input raster files
    output_path : Path
        Output mosaic path
    bounds : tuple, optional
        Clip to bounds after mosaicking

    Returns
    -------
    Path
        Path to output mosaic
    """
    # Open all source files
    sources = [rasterio.open(f) for f in input_files]

    # Merge
    mosaic, transform = merge(sources)

    # Close sources
    for src in sources:
        src.close()

    # Get profile from first source
    with rasterio.open(input_files[0]) as src:
        profile = src.profile.copy()

    # Update profile
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
    )

    # Write output
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    logger.info(f"Created mosaic at {output_path}")

    return output_path
