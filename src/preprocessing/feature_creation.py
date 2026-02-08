"""Feature engineering for ignition model."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import xarray as xr
from scipy.ndimage import distance_transform_edt

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, get_config

logger = logging.getLogger(__name__)


def create_static_features(
    bounds: Tuple[float, float, float, float],
    resolution: int = 270,
    output_dir: Optional[Path] = None,
) -> Dict[str, Path]:
    """
    Create static feature rasters for ignition model.

    Features:
    - Elevation, slope, aspect, TPI
    - Distance to roads
    - Distance to power lines
    - Fuel model
    - Canopy cover, height, bulk density
    - WUI proximity
    - Population density

    Parameters
    ----------
    bounds : tuple
        (minx, miny, maxx, maxy) in EPSG:3310
    resolution : int
        Output resolution in meters
    output_dir : Path, optional
        Output directory

    Returns
    -------
    dict
        Mapping of feature name to raster path
    """
    output_dir = output_dir or PROCESSED_DATA_DIR / "features" / "static"
    output_dir.mkdir(parents=True, exist_ok=True)

    features = {}

    # Create reference grid
    ref_path = output_dir / "reference_grid.tif"
    _create_reference_grid(bounds, resolution, ref_path)

    # Terrain features (from processed terrain data)
    terrain_dir = RAW_DATA_DIR / "terrain"
    if (terrain_dir / "dem.tif").exists():
        from src.preprocessing.raster_alignment import resample_raster

        for name in ["dem", "slope", "aspect", "tpi"]:
            src_path = terrain_dir / f"{name}.tif"
            if src_path.exists():
                dst_path = output_dir / f"{name}.tif"
                resample_raster(
                    src_path, dst_path,
                    target_crs="EPSG:3310",
                    target_resolution=resolution,
                    bounds=bounds,
                )
                features[name] = dst_path

    # LANDFIRE features
    landfire_dir = RAW_DATA_DIR / "landfire"
    for name in ["FBFM40", "CC", "CH", "CBD"]:
        pattern = f"*{name}*.tif"
        files = list(landfire_dir.glob(pattern))
        if files:
            from src.preprocessing.raster_alignment import resample_raster

            dst_path = output_dir / f"{name.lower()}.tif"
            resample_raster(
                files[0], dst_path,
                target_crs="EPSG:3310",
                target_resolution=resolution,
                bounds=bounds,
            )
            features[name.lower()] = dst_path

    # Distance features
    infra_dir = RAW_DATA_DIR / "infrastructure"
    if (infra_dir / "distance_to_roads.tif").exists():
        features["dist_roads"] = infra_dir / "distance_to_roads.tif"
    if (infra_dir / "distance_to_power_lines.tif").exists():
        features["dist_power"] = infra_dir / "distance_to_power_lines.tif"

    logger.info(f"Created {len(features)} static feature rasters")
    return features


def create_dynamic_features(
    date: datetime,
    gridmet_ds: xr.Dataset,
    bounds: Tuple[float, float, float, float],
    resolution: int = 270,
    output_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Create dynamic (daily) feature arrays for a specific date.

    Features:
    - Maximum temperature
    - Minimum relative humidity
    - Wind speed
    - ERC (Energy Release Component)
    - Fuel moisture (100-hr)
    - Days since rain
    - Day of year
    - Day of week

    Parameters
    ----------
    date : datetime
        Date for features
    gridmet_ds : xr.Dataset
        GridMET dataset
    bounds : tuple
        (minx, miny, maxx, maxy) in EPSG:3310
    resolution : int
        Output resolution
    output_dir : Path, optional
        Output directory

    Returns
    -------
    dict
        Mapping of feature name to numpy array
    """
    date_str = date.strftime("%Y-%m-%d")

    # Select date from GridMET
    try:
        day_data = gridmet_ds.sel(day=date_str, method="nearest")
    except KeyError:
        logger.warning(f"Date {date_str} not in GridMET data")
        return {}

    features = {}

    # Weather variables
    var_mapping = {
        "tmmx": "temp_max",
        "rmin": "rh_min",
        "vs": "wind_speed",
        "erc": "erc",
        "fm100": "fuel_moisture_100hr",
        "pr": "precipitation",
    }

    for gridmet_var, feature_name in var_mapping.items():
        if gridmet_var in day_data.data_vars:
            data = day_data[gridmet_var].values
            features[feature_name] = data

    # Compute days since rain (requires time series)
    if "pr" in gridmet_ds.data_vars:
        features["days_since_rain"] = _compute_days_since_rain(
            gridmet_ds, date
        )

    # Temporal features
    features["day_of_year"] = np.full_like(
        features.get("temp_max", np.zeros((10, 10))),
        date.timetuple().tm_yday,
    )
    features["day_of_week"] = np.full_like(
        features.get("temp_max", np.zeros((10, 10))),
        date.weekday(),
    )

    # Season (0=winter, 1=spring, 2=summer, 3=fall)
    month = date.month
    if month in [12, 1, 2]:
        season = 0
    elif month in [3, 4, 5]:
        season = 1
    elif month in [6, 7, 8]:
        season = 2
    else:
        season = 3

    features["season"] = np.full_like(
        features.get("temp_max", np.zeros((10, 10))),
        season,
    )

    return features


def create_training_dataset(
    ignition_points: gpd.GeoDataFrame,
    fire_perimeters: gpd.GeoDataFrame,
    static_features: Dict[str, Path],
    gridmet_ds: xr.Dataset,
    years: Tuple[int, int],
    negative_ratio: int = 4,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create training dataset for ignition model.

    Parameters
    ----------
    ignition_points : GeoDataFrame
        Historical fire ignition points
    fire_perimeters : GeoDataFrame
        Fire perimeters (for excluding from negatives)
    static_features : dict
        Static feature rasters
    gridmet_ds : xr.Dataset
        GridMET weather data
    years : tuple
        (start_year, end_year) for training data
    negative_ratio : int
        Ratio of negative to positive samples
    output_path : Path, optional
        Path to save training data

    Returns
    -------
    DataFrame
        Training dataset with features and labels
    """
    output_path = output_path or PROCESSED_DATA_DIR / "training_data.parquet"

    logger.info(f"Creating training dataset for years {years}")

    # Filter ignition points to training years
    ignitions = ignition_points[
        (ignition_points["year_"] >= years[0]) &
        (ignition_points["year_"] <= years[1])
    ].copy()

    if ignitions.empty:
        # Try alternative year column names
        for col in ["year", "YEAR_", "fire_year"]:
            if col in ignition_points.columns:
                ignitions = ignition_points[
                    (ignition_points[col] >= years[0]) &
                    (ignition_points[col] <= years[1])
                ].copy()
                break

    logger.info(f"Found {len(ignitions)} ignition points in training period")

    # Extract features for positive samples
    positive_samples = []

    for idx, row in ignitions.iterrows():
        # Get date
        date = _extract_date(row)
        if date is None:
            continue

        # Get location
        point = row.geometry

        # Extract static features
        sample = {"label": 1, "geometry": point, "date": date}

        for name, path in static_features.items():
            value = _extract_raster_value(path, point)
            sample[name] = value

        # Extract dynamic features
        dynamic = create_dynamic_features(
            date=date,
            gridmet_ds=gridmet_ds,
            bounds=None,  # Will be derived from point
            resolution=270,
        )

        for name, arr in dynamic.items():
            # Get value at point location from GridMET
            sample[name] = _extract_gridmet_value(gridmet_ds, name, date, point)

        positive_samples.append(sample)

    logger.info(f"Extracted features for {len(positive_samples)} positive samples")

    # Generate negative samples
    negative_samples = _generate_negative_samples(
        positive_samples=positive_samples,
        fire_perimeters=fire_perimeters,
        static_features=static_features,
        gridmet_ds=gridmet_ds,
        ratio=negative_ratio,
    )

    # Combine
    all_samples = positive_samples + negative_samples

    # Convert to DataFrame
    df = pd.DataFrame(all_samples)

    # Drop geometry for training
    if "geometry" in df.columns:
        df = df.drop(columns=["geometry"])

    # Save
    df.to_parquet(output_path)
    logger.info(f"Saved training dataset ({len(df)} samples) to {output_path}")

    return df


def _generate_negative_samples(
    positive_samples: List[dict],
    fire_perimeters: gpd.GeoDataFrame,
    static_features: Dict[str, Path],
    gridmet_ds: xr.Dataset,
    ratio: int = 4,
) -> List[dict]:
    """Generate negative samples matched to positive sample dates."""
    from shapely.geometry import Point

    negative_samples = []

    # Get bounds from static features
    first_path = list(static_features.values())[0]
    with rasterio.open(first_path) as src:
        bounds = src.bounds
        transform = src.transform
        width = src.width
        height = src.height

    # Create burned area mask
    burned_mask = _create_burned_mask(fire_perimeters, first_path)

    # Get dates from positive samples
    dates = [s["date"] for s in positive_samples if s.get("date")]
    unique_dates = list(set(dates))

    logger.info(f"Generating {ratio}x negative samples from {len(unique_dates)} unique dates")

    n_per_date = int(len(positive_samples) * ratio / len(unique_dates))

    for date in unique_dates:
        # Sample random unburned locations
        sampled = 0
        attempts = 0
        max_attempts = n_per_date * 10

        while sampled < n_per_date and attempts < max_attempts:
            # Random pixel
            row = np.random.randint(0, height)
            col = np.random.randint(0, width)

            # Check if burned
            if burned_mask[row, col]:
                attempts += 1
                continue

            # Convert to coordinates
            x = transform.c + col * transform.a + transform.a / 2
            y = transform.f + row * transform.e + transform.e / 2
            point = Point(x, y)

            # Extract features
            sample = {"label": 0, "geometry": point, "date": date}

            for name, path in static_features.items():
                value = _extract_raster_value(path, point)
                sample[name] = value

                # Skip if nodata
                if value is None or np.isnan(value):
                    break
            else:
                # Extract dynamic features
                for name in ["temp_max", "rh_min", "wind_speed", "erc"]:
                    sample[name] = _extract_gridmet_value(gridmet_ds, name, date, point)

                negative_samples.append(sample)
                sampled += 1

            attempts += 1

    logger.info(f"Generated {len(negative_samples)} negative samples")
    return negative_samples


def _create_burned_mask(
    fire_perimeters: gpd.GeoDataFrame,
    template_path: Path,
) -> np.ndarray:
    """Create mask of burned areas from fire perimeters."""
    with rasterio.open(template_path) as src:
        height = src.height
        width = src.width
        transform = src.transform

    if fire_perimeters.empty:
        return np.zeros((height, width), dtype=bool)

    # Reproject if needed
    if fire_perimeters.crs != "EPSG:3310":
        fire_perimeters = fire_perimeters.to_crs("EPSG:3310")

    # Rasterize
    shapes = [(geom, 1) for geom in fire_perimeters.geometry]

    burned = rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
    )

    return burned.astype(bool)


def _extract_raster_value(raster_path: Path, point) -> Optional[float]:
    """Extract raster value at point location."""
    import pyproj
    from shapely.geometry import Point

    with rasterio.open(raster_path) as src:
        # Transform point if needed
        if src.crs.to_string() != "EPSG:3310":
            transformer = pyproj.Transformer.from_crs(
                "EPSG:3310", src.crs, always_xy=True
            )
            x, y = transformer.transform(point.x, point.y)
        else:
            x, y = point.x, point.y

        try:
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                value = src.read(1)[row, col]
                nodata = src.nodata
                if nodata is not None and value == nodata:
                    return None
                return float(value)
        except Exception:
            pass

    return None


def _extract_gridmet_value(
    ds: xr.Dataset,
    var: str,
    date: datetime,
    point,
) -> Optional[float]:
    """Extract GridMET value at point and date."""
    # GridMET variable mapping
    gridmet_vars = {
        "temp_max": "tmmx",
        "rh_min": "rmin",
        "wind_speed": "vs",
        "erc": "erc",
        "fuel_moisture_100hr": "fm100",
    }

    gridmet_var = gridmet_vars.get(var, var)
    if gridmet_var not in ds.data_vars:
        return None

    try:
        # Get point coordinates (GridMET is in WGS84)
        import pyproj

        transformer = pyproj.Transformer.from_crs(
            "EPSG:3310", "EPSG:4326", always_xy=True
        )
        lon, lat = transformer.transform(point.x, point.y)

        # GridMET uses 0-360 longitude
        if lon < 0:
            lon += 360

        value = ds[gridmet_var].sel(
            day=date.strftime("%Y-%m-%d"),
            lat=lat,
            lon=lon,
            method="nearest",
        ).values

        return float(value)
    except Exception:
        return None


def _extract_date(row) -> Optional[datetime]:
    """Extract date from fire record."""
    # Try different date column names
    date_cols = ["alarm_date", "ALARM_DATE", "discovery_date", "fire_date", "date"]

    for col in date_cols:
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            if isinstance(val, datetime):
                return val
            try:
                return pd.to_datetime(val)
            except Exception:
                continue

    return None


def _compute_days_since_rain(ds: xr.Dataset, date: datetime, threshold: float = 2.54) -> np.ndarray:
    """Compute days since last significant rain (>= 0.1 inch = 2.54mm)."""
    if "pr" not in ds.data_vars:
        return np.zeros((10, 10))

    # Look back up to 90 days
    end_date = date
    start_date = date - timedelta(days=90)

    precip = ds["pr"].sel(day=slice(start_date, end_date))

    # Find last day with significant rain
    rain_days = precip >= threshold

    # Compute days since rain for each cell
    days_since = np.zeros_like(precip.isel(day=0).values)

    for i in range(len(precip.day) - 1, -1, -1):
        mask = rain_days.isel(day=i).values
        days_since = np.where(mask, 0, days_since + 1)

    return days_since


def _create_reference_grid(
    bounds: Tuple[float, float, float, float],
    resolution: int,
    output_path: Path,
) -> Path:
    """Create reference grid raster."""
    minx, miny, maxx, maxy = bounds

    width = int((maxx - minx) / resolution)
    height = int((maxy - miny) / resolution)

    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, width, height)

    profile = {
        "driver": "GTiff",
        "dtype": np.uint8,
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:3310",
        "transform": transform,
        "nodata": 0,
    }

    data = np.ones((height, width), dtype=np.uint8)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data, 1)

    return output_path
