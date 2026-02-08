"""Download fire history data from CAL FIRE and NIFC."""

import logging
from pathlib import Path
from typing import Optional, Tuple
import requests
import geopandas as gpd
from shapely.geometry import box

from config.settings import RAW_DATA_DIR, get_config

logger = logging.getLogger(__name__)

# API endpoints - use ArcGIS Hub download API
FIRE_PERIMETERS_URL = (
    "https://opendata.arcgis.com/api/v3/datasets/"
    "e3802d2abf8741a187e73a9db49d68fe_0/downloads/data"
)

FIRE_PERIMETERS_QUERY_URL = (
    "https://services1.arcgis.com/jUJYIo9tSA7EHvfZ/arcgis/rest/services/"
    "California_Fire_Perimeters_all/FeatureServer/0/query"
)

IGNITION_POINTS_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Interagency_Perimeters/FeatureServer/0/query"
)

# Alternative NIFC ignition points
NIFC_IGNITION_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "Historic_GeoMAC_Perimeters_All/FeatureServer/0/query"
)


def download_fire_perimeters(
    output_dir: Optional[Path] = None,
    years: Optional[Tuple[int, int]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    county: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Download California fire perimeters from CAL FIRE FRAP.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save downloaded data
    years : tuple of int, optional
        (start_year, end_year) inclusive range to filter
    bounds : tuple, optional
        (minx, miny, maxx, maxy) in WGS84 to filter spatially
    county : str, optional
        County name to filter (e.g., "Sonoma")

    Returns
    -------
    GeoDataFrame
        Fire perimeter polygons with attributes
    """
    output_dir = output_dir or RAW_DATA_DIR / "fire_history"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading fire perimeters from CAL FIRE FRAP...")

    # Try direct GeoJSON download first
    geojson_url = (
        "https://opendata.arcgis.com/api/v3/datasets/"
        "e3802d2abf8741a187e73a9db49d68fe_0/downloads/data"
    )

    all_features = []

    try:
        # Try downloading as GeoJSON
        params = {"format": "geojson", "spatialRefId": "4326"}
        response = requests.get(geojson_url, params=params, timeout=300, allow_redirects=True)

        if response.status_code == 200:
            data = response.json()
            all_features = data.get("features", [])
            logger.info(f"Downloaded {len(all_features)} fire perimeters via GeoJSON")
        else:
            raise requests.RequestException(f"Status {response.status_code}")

    except Exception as e:
        logger.warning(f"GeoJSON download failed: {e}, trying query API...")

        # Fall back to query API
        query_url = FIRE_PERIMETERS_QUERY_URL
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "outSR": "4326",
            "resultRecordCount": 2000,
        }

        offset = 0
        while True:
            params["resultOffset"] = offset

            try:
                response = requests.get(query_url, params=params, timeout=120)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e2:
                logger.error(f"Query API also failed: {e2}")
                break

            features = data.get("features", [])
            if not features:
                break

            all_features.extend(features)
            logger.info(f"Downloaded {len(all_features)} fire perimeters...")

            if len(features) < 2000:
                break
            offset += 2000

    if not all_features:
        logger.warning("No fire perimeters found - creating empty GeoDataFrame")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")

    # Clean up column names
    gdf.columns = [c.lower() for c in gdf.columns]

    # Filter by year if specified (post-download filtering)
    if years:
        year_col = None
        for col in ["year_", "year", "fire_year"]:
            if col in gdf.columns:
                year_col = col
                break
        if year_col:
            gdf = gdf[(gdf[year_col] >= years[0]) & (gdf[year_col] <= years[1])]
            logger.info(f"Filtered to {len(gdf)} fires in years {years}")

    # Filter by county if specified
    if county:
        # Load county boundaries for filtering
        county_gdf = _get_county_boundary(county)
        if county_gdf is not None:
            gdf = gpd.sjoin(gdf, county_gdf, predicate="intersects")
            gdf = gdf.drop(columns=["index_right"], errors="ignore")

    # Convert to California Albers
    gdf = gdf.to_crs("EPSG:3310")

    # Save to file
    output_file = output_dir / "fire_perimeters.parquet"
    gdf.to_parquet(output_file)
    logger.info(f"Saved {len(gdf)} fire perimeters to {output_file}")

    return gdf


def download_ignition_points(
    output_dir: Optional[Path] = None,
    years: Optional[Tuple[int, int]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    state: str = "CA",
) -> gpd.GeoDataFrame:
    """
    Download fire ignition points from NIFC WFIGS.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save downloaded data
    years : tuple of int, optional
        (start_year, end_year) inclusive range
    bounds : tuple, optional
        (minx, miny, maxx, maxy) in WGS84
    state : str
        State code to filter (default "CA")

    Returns
    -------
    GeoDataFrame
        Point locations of fire ignitions
    """
    output_dir = output_dir or RAW_DATA_DIR / "fire_history"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading ignition points from NIFC WFIGS...")

    # Use the fire perimeters centroid as proxy for ignition if direct points unavailable
    # WFIGS provides perimeters, we extract centroids for ignition locations

    where_clauses = [f"POOState = '{state}'"]

    if years:
        where_clauses.append(
            f"FireDiscoveryDateTime >= DATE '{years[0]}-01-01' AND "
            f"FireDiscoveryDateTime <= DATE '{years[1]}-12-31'"
        )

    params = {
        "where": " AND ".join(where_clauses),
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

    # Try primary source first
    url = IGNITION_POINTS_URL

    while True:
        params["resultOffset"] = offset
        params["resultRecordCount"] = chunk_size

        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.warning(f"Primary source failed: {e}, trying alternative...")
            # Fall back to extracting from perimeters
            break

        features = data.get("features", [])
        if not features:
            break

        all_features.extend(features)
        logger.info(f"Downloaded {len(all_features)} ignition records...")

        if len(features) < chunk_size:
            break
        offset += chunk_size

    if all_features:
        gdf = gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")
    else:
        # Fall back: use perimeter centroids
        logger.info("Using fire perimeter centroids as ignition points...")
        perimeters = download_fire_perimeters(
            output_dir=output_dir, years=years, bounds=bounds
        )
        if perimeters.empty:
            return gpd.GeoDataFrame()

        gdf = perimeters.copy()
        gdf["geometry"] = gdf.geometry.centroid
        gdf = gdf.to_crs("EPSG:4326")

    # Clean up column names
    gdf.columns = [c.lower() for c in gdf.columns]

    # Convert to California Albers
    gdf = gdf.to_crs("EPSG:3310")

    # Save to file
    output_file = output_dir / "ignition_points.parquet"
    gdf.to_parquet(output_file)
    logger.info(f"Saved {len(gdf)} ignition points to {output_file}")

    return gdf


def _get_county_boundary(county: str) -> Optional[gpd.GeoDataFrame]:
    """Get county boundary from Census TIGER."""
    url = (
        "https://services.arcgis.com/P3ePLMYs2RVChkJx/arcgis/rest/services/"
        "USA_Counties/FeatureServer/0/query"
    )

    params = {
        "where": f"NAME = '{county}' AND STATE_NAME = 'California'",
        "outFields": "NAME,STATE_NAME",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()

        if data.get("features"):
            return gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")
    except requests.RequestException as e:
        logger.warning(f"Could not download county boundary: {e}")

    return None


def load_fire_perimeters(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load previously downloaded fire perimeters."""
    path = path or RAW_DATA_DIR / "fire_history" / "fire_perimeters.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Fire perimeters not found at {path}")
    return gpd.read_parquet(path)


def load_ignition_points(path: Optional[Path] = None) -> gpd.GeoDataFrame:
    """Load previously downloaded ignition points."""
    path = path or RAW_DATA_DIR / "fire_history" / "ignition_points.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Ignition points not found at {path}")
    return gpd.read_parquet(path)
