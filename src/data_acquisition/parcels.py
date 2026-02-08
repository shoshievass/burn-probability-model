"""Download parcel boundary data."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import requests
import geopandas as gpd

from config.settings import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# ArcGIS Feature Service URLs for county parcels
PARCEL_SERVICES = {
    "Sonoma": "https://services1.arcgis.com/P5Mv5GY5S66M8Z1Q/arcgis/rest/services/Parcels/FeatureServer/0",
    "San Diego": "https://services1.arcgis.com/1vIhDJwtG5eNmiqX/arcgis/rest/services/Parcels/FeatureServer/0",
    "Los Angeles": "https://services5.arcgis.com/FuNOhvlZYBa4n7q4/arcgis/rest/services/Parcels/FeatureServer/0",
    "Butte": "https://services.arcgis.com/WSiCwE0dTnOiTHLQ/arcgis/rest/services/Parcels/FeatureServer/0",
}


def download_county_parcels(
    county: str,
    output_dir: Optional[Path] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> gpd.GeoDataFrame:
    """
    Download parcel boundaries for a California county.

    Parameters
    ----------
    county : str
        County name (e.g., "Sonoma")
    output_dir : Path, optional
        Directory to save data
    bounds : tuple, optional
        (minx, miny, maxx, maxy) in WGS84 to filter

    Returns
    -------
    GeoDataFrame
        Parcel polygons with APN and attributes
    """
    output_dir = output_dir or RAW_DATA_DIR / "parcels"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{county.lower().replace(' ', '_')}_parcels.parquet"

    # Check if already downloaded
    if output_file.exists():
        logger.info(f"Loading existing parcels from {output_file}")
        return gpd.read_parquet(output_file)

    logger.info(f"Downloading parcels for {county} County...")

    # Try county-specific service first
    if county in PARCEL_SERVICES:
        gdf = _download_from_arcgis(PARCEL_SERVICES[county], bounds)
    else:
        # Fall back to statewide parcel search
        gdf = _download_statewide_parcels(county, bounds)

    if gdf is None or gdf.empty:
        logger.warning(f"No parcels found for {county}")
        return gpd.GeoDataFrame()

    # Standardize column names
    gdf = _standardize_parcel_columns(gdf)

    # Convert to California Albers
    gdf = gdf.to_crs("EPSG:3310")

    # Save
    gdf.to_parquet(output_file)
    logger.info(f"Saved {len(gdf)} parcels to {output_file}")

    return gdf


def _download_from_arcgis(
    service_url: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """Download parcels from ArcGIS Feature Service."""
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }

    if bounds:
        params["geometry"] = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"
        params["geometryType"] = "esriGeometryEnvelope"
        params["spatialRel"] = "esriSpatialRelIntersects"
        params["inSR"] = "4326"

    # Download in chunks
    all_features = []
    offset = 0
    chunk_size = 2000

    while True:
        params["resultOffset"] = offset
        params["resultRecordCount"] = chunk_size

        try:
            response = requests.get(
                f"{service_url}/query", params=params, timeout=120
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to download parcels: {e}")
            break

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        logger.info(f"Downloaded {len(all_features)} parcels...")

        if len(features) < chunk_size:
            break
        offset += chunk_size

    if all_features:
        return gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    return None


def _download_statewide_parcels(
    county: str,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[gpd.GeoDataFrame]:
    """
    Download parcels from statewide parcel service.

    Note: Statewide parcel data may require special access or download.
    This function provides a fallback method.
    """
    # California Statewide Parcel Map - requires data.ca.gov or similar source
    logger.info(f"Attempting statewide parcel download for {county}...")

    # Try data.ca.gov parcels API
    statewide_url = (
        "https://gis.data.ca.gov/datasets/"
        "California-Statewide-Parcel-Boundaries/FeatureServer/0"
    )

    params = {
        "where": f"COUNTY = '{county.upper()}'",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }

    if bounds:
        params["geometry"] = f"{bounds[0]},{bounds[1]},{bounds[2]},{bounds[3]}"
        params["geometryType"] = "esriGeometryEnvelope"
        params["spatialRel"] = "esriSpatialRelIntersects"
        params["inSR"] = "4326"

    try:
        response = requests.get(f"{statewide_url}/query", params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        features = data.get("features", [])
        if features:
            return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    except requests.RequestException as e:
        logger.warning(f"Statewide parcel download failed: {e}")

    logger.warning(
        f"Could not automatically download parcels for {county}.\n"
        f"Please download manually from:\n"
        f"  - UCLA Geoportal: https://geodata.lib.berkeley.edu/\n"
        f"  - County GIS portal\n"
        f"  - ArcGIS Hub: https://hub.arcgis.com/"
    )

    return None


def _standardize_parcel_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Standardize parcel column names across different sources."""
    # Common APN column names
    apn_cols = ["APN", "apn", "PARCEL_NUM", "parcel_num", "PARCELNUMB", "PIN"]

    # Find and rename APN column
    for col in apn_cols:
        if col in gdf.columns:
            gdf = gdf.rename(columns={col: "apn"})
            break
    else:
        # Create APN from index if not found
        gdf["apn"] = gdf.index.astype(str)

    # Keep only essential columns
    keep_cols = ["apn", "geometry"]

    # Add optional columns if present
    optional = {
        "SITUS_ADDR": "address",
        "situs_addr": "address",
        "ADDRESS": "address",
        "ACRES": "acres",
        "GIS_ACRES": "acres",
        "SHAPE_AREA": "area_sqm",
        "LAND_USE": "land_use",
        "ZONING": "zoning",
    }

    for old_name, new_name in optional.items():
        if old_name in gdf.columns:
            gdf = gdf.rename(columns={old_name: new_name})
            if new_name not in keep_cols:
                keep_cols.append(new_name)

    # Filter to keep columns
    existing_cols = [c for c in keep_cols if c in gdf.columns]
    return gdf[existing_cols]


def load_parcels(county: str, path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load previously downloaded parcels."""
    path = path or RAW_DATA_DIR / "parcels" / f"{county.lower().replace(' ', '_')}_parcels.parquet"

    if not path.exists():
        raise FileNotFoundError(f"Parcels not found at {path}")

    return gpd.read_parquet(path)


def get_parcel_count_by_county(parcels_dir: Optional[Path] = None) -> dict:
    """Get parcel counts for all downloaded counties."""
    parcels_dir = parcels_dir or RAW_DATA_DIR / "parcels"

    counts = {}
    for f in parcels_dir.glob("*_parcels.parquet"):
        county = f.stem.replace("_parcels", "").replace("_", " ").title()
        gdf = gpd.read_parquet(f)
        counts[county] = len(gdf)

    return counts
