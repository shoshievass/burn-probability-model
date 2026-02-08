"""Feature engineering for ignition probability model."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np
import pandas as pd
import rasterio
import xarray as xr

from config.settings import get_config, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class IgnitionFeatureEngineer:
    """
    Feature engineering pipeline for ignition model.

    Computes static and dynamic features for ignition probability prediction.
    """

    # Static feature names
    STATIC_FEATURES = [
        "elevation",
        "slope",
        "aspect",
        "tpi",
        "fuel_model",
        "canopy_cover",
        "canopy_height",
        "canopy_bulk_density",
        "dist_roads",
        "dist_power_lines",
        "population_density",
        "wui_class",
    ]

    # Dynamic feature names
    DYNAMIC_FEATURES = [
        "temp_max",
        "temp_min",
        "rh_min",
        "rh_max",
        "wind_speed",
        "wind_direction",
        "erc",
        "fuel_moisture_100hr",
        "fuel_moisture_1000hr",
        "precipitation",
        "days_since_rain",
    ]

    # Temporal features
    TEMPORAL_FEATURES = [
        "day_of_year",
        "day_of_week",
        "month",
        "season",
        "is_weekend",
    ]

    def __init__(
        self,
        static_rasters: Optional[Dict[str, Path]] = None,
        gridmet_ds: Optional[xr.Dataset] = None,
    ):
        """
        Initialize feature engineer.

        Parameters
        ----------
        static_rasters : dict, optional
            Mapping of feature name to raster path
        gridmet_ds : xr.Dataset, optional
            GridMET weather dataset
        """
        self.static_rasters = static_rasters or {}
        self.gridmet_ds = gridmet_ds
        self.config = get_config()

    def compute_features_for_point(
        self,
        x: float,
        y: float,
        date: datetime,
        crs: str = "EPSG:3310",
    ) -> Dict[str, float]:
        """
        Compute all features for a single point.

        Parameters
        ----------
        x, y : float
            Point coordinates in specified CRS
        date : datetime
            Date for dynamic features
        crs : str
            Coordinate reference system of input point

        Returns
        -------
        dict
            Feature name to value mapping
        """
        features = {}

        # Static features from rasters
        for name, path in self.static_rasters.items():
            if path.exists():
                value = self._extract_raster_value(path, x, y, crs)
                features[name] = value

        # Dynamic features from GridMET
        if self.gridmet_ds is not None:
            dynamic = self._extract_gridmet_features(x, y, date, crs)
            features.update(dynamic)

        # Temporal features
        temporal = self._compute_temporal_features(date)
        features.update(temporal)

        # Derived features
        derived = self._compute_derived_features(features)
        features.update(derived)

        return features

    def compute_features_for_grid(
        self,
        date: datetime,
        bounds: Tuple[float, float, float, float],
        resolution: int = 270,
    ) -> Dict[str, np.ndarray]:
        """
        Compute features for entire grid.

        Parameters
        ----------
        date : datetime
            Date for dynamic features
        bounds : tuple
            (minx, miny, maxx, maxy) in EPSG:3310
        resolution : int
            Grid resolution in meters

        Returns
        -------
        dict
            Feature name to 2D array mapping
        """
        minx, miny, maxx, maxy = bounds
        ncols = int((maxx - minx) / resolution)
        nrows = int((maxy - miny) / resolution)

        features = {}

        # Static features
        for name, path in self.static_rasters.items():
            if path.exists():
                with rasterio.open(path) as src:
                    # Resample to target grid if needed
                    data = self._resample_to_grid(
                        src, bounds, nrows, ncols
                    )
                    features[name] = data

        # Dynamic features
        if self.gridmet_ds is not None:
            dynamic = self._compute_gridmet_grid(date, bounds, nrows, ncols)
            features.update(dynamic)

        # Temporal features (constant across grid)
        temporal = self._compute_temporal_features(date)
        for name, value in temporal.items():
            features[name] = np.full((nrows, ncols), value)

        return features

    def _extract_raster_value(
        self,
        raster_path: Path,
        x: float,
        y: float,
        crs: str,
    ) -> Optional[float]:
        """Extract value from raster at point location."""
        import pyproj

        with rasterio.open(raster_path) as src:
            # Transform coordinates if needed
            if str(src.crs) != crs:
                transformer = pyproj.Transformer.from_crs(
                    crs, src.crs, always_xy=True
                )
                x, y = transformer.transform(x, y)

            try:
                row, col = src.index(x, y)
                if 0 <= row < src.height and 0 <= col < src.width:
                    value = float(src.read(1)[row, col])
                    if src.nodata is not None and value == src.nodata:
                        return np.nan
                    return value
            except Exception:
                pass

        return np.nan

    def _extract_gridmet_features(
        self,
        x: float,
        y: float,
        date: datetime,
        crs: str,
    ) -> Dict[str, float]:
        """Extract GridMET features for point and date."""
        import pyproj

        # Transform to WGS84 for GridMET
        transformer = pyproj.Transformer.from_crs(
            crs, "EPSG:4326", always_xy=True
        )
        lon, lat = transformer.transform(x, y)

        # GridMET uses 0-360 longitude
        if lon < 0:
            lon += 360

        features = {}
        date_str = date.strftime("%Y-%m-%d")

        # Variable mapping
        var_map = {
            "tmmx": "temp_max",
            "tmmn": "temp_min",
            "rmax": "rh_max",
            "rmin": "rh_min",
            "vs": "wind_speed",
            "th": "wind_direction",
            "erc": "erc",
            "fm100": "fuel_moisture_100hr",
            "fm1000": "fuel_moisture_1000hr",
            "pr": "precipitation",
        }

        for gridmet_var, feature_name in var_map.items():
            if gridmet_var in self.gridmet_ds.data_vars:
                try:
                    value = self.gridmet_ds[gridmet_var].sel(
                        day=date_str,
                        lat=lat,
                        lon=lon,
                        method="nearest",
                    ).values
                    features[feature_name] = float(value)
                except Exception:
                    features[feature_name] = np.nan

        # Days since rain
        features["days_since_rain"] = self._compute_days_since_rain(
            lat, lon, date
        )

        return features

    def _compute_temporal_features(self, date: datetime) -> Dict[str, float]:
        """Compute temporal features from date."""
        return {
            "day_of_year": date.timetuple().tm_yday,
            "day_of_week": date.weekday(),
            "month": date.month,
            "season": self._get_season(date.month),
            "is_weekend": 1.0 if date.weekday() >= 5 else 0.0,
        }

    def _compute_derived_features(
        self, features: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute derived features from base features."""
        derived = {}

        # Aspect categories (N, NE, E, SE, S, SW, W, NW, flat)
        aspect = features.get("aspect", np.nan)
        if not np.isnan(aspect):
            derived["aspect_sin"] = np.sin(np.radians(aspect))
            derived["aspect_cos"] = np.cos(np.radians(aspect))

        # Vapor pressure deficit proxy
        temp_max = features.get("temp_max", np.nan)
        rh_min = features.get("rh_min", np.nan)
        if not np.isnan(temp_max) and not np.isnan(rh_min):
            # Simplified VPD
            temp_c = temp_max - 273.15 if temp_max > 100 else temp_max
            es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
            derived["vpd"] = es * (1 - rh_min / 100)

        # Fire weather index (simplified)
        wind = features.get("wind_speed", np.nan)
        if not np.isnan(temp_max) and not np.isnan(rh_min) and not np.isnan(wind):
            # Higher values = more fire-prone
            temp_c = temp_max - 273.15 if temp_max > 100 else temp_max
            derived["fire_weather_index"] = (
                max(0, temp_c - 10) * (100 - rh_min) * (1 + wind / 10) / 1000
            )

        return derived

    def _compute_days_since_rain(
        self,
        lat: float,
        lon: float,
        date: datetime,
        threshold: float = 2.54,  # 0.1 inch in mm
    ) -> float:
        """Compute days since last significant rain."""
        if "pr" not in self.gridmet_ds.data_vars:
            return np.nan

        # Look back 90 days
        from datetime import timedelta
        start_date = date - timedelta(days=90)

        try:
            precip = self.gridmet_ds["pr"].sel(
                day=slice(start_date, date),
                lat=lat,
                lon=lon,
                method="nearest",
            )

            # Find last day with significant rain
            rain_days = precip >= threshold
            if rain_days.any():
                last_rain_idx = np.where(rain_days.values)[0][-1]
                return len(precip) - 1 - last_rain_idx
            else:
                return 90.0

        except Exception:
            return np.nan

    def _get_season(self, month: int) -> int:
        """Get season from month (0=winter, 1=spring, 2=summer, 3=fall)."""
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3

    def _resample_to_grid(
        self,
        src,
        bounds: Tuple[float, float, float, float],
        nrows: int,
        ncols: int,
    ) -> np.ndarray:
        """Resample raster data to target grid."""
        from rasterio.warp import reproject, Resampling

        minx, miny, maxx, maxy = bounds
        transform = rasterio.transform.from_bounds(
            minx, miny, maxx, maxy, ncols, nrows
        )

        dst = np.empty((nrows, ncols), dtype=np.float32)

        reproject(
            source=src.read(1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs="EPSG:3310",
            resampling=Resampling.bilinear,
        )

        return dst

    def _compute_gridmet_grid(
        self,
        date: datetime,
        bounds: Tuple[float, float, float, float],
        nrows: int,
        ncols: int,
    ) -> Dict[str, np.ndarray]:
        """Compute GridMET features for entire grid."""
        import pyproj
        from scipy.interpolate import griddata

        # Transform bounds to WGS84
        transformer = pyproj.Transformer.from_crs(
            "EPSG:3310", "EPSG:4326", always_xy=True
        )

        minx, miny, maxx, maxy = bounds
        lon_min, lat_min = transformer.transform(minx, miny)
        lon_max, lat_max = transformer.transform(maxx, maxy)

        # Adjust longitude for GridMET
        if lon_min < 0:
            lon_min += 360
        if lon_max < 0:
            lon_max += 360

        date_str = date.strftime("%Y-%m-%d")

        features = {}
        var_map = {
            "tmmx": "temp_max",
            "rmin": "rh_min",
            "vs": "wind_speed",
            "erc": "erc",
        }

        for gridmet_var, feature_name in var_map.items():
            if gridmet_var in self.gridmet_ds.data_vars:
                try:
                    data = self.gridmet_ds[gridmet_var].sel(
                        day=date_str,
                        lat=slice(lat_max, lat_min),
                        lon=slice(lon_min, lon_max),
                    ).values

                    # Resample to target grid
                    from scipy.ndimage import zoom
                    zoom_factors = (nrows / data.shape[0], ncols / data.shape[1])
                    features[feature_name] = zoom(data, zoom_factors, order=1)

                except Exception as e:
                    logger.warning(f"Failed to get {gridmet_var}: {e}")
                    features[feature_name] = np.full((nrows, ncols), np.nan)

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of all feature names."""
        return (
            self.STATIC_FEATURES +
            self.DYNAMIC_FEATURES +
            self.TEMPORAL_FEATURES +
            ["aspect_sin", "aspect_cos", "vpd", "fire_weather_index"]
        )


def compute_all_features(
    points_gdf,
    static_rasters: Dict[str, Path],
    gridmet_ds: xr.Dataset,
    date_column: str = "date",
) -> pd.DataFrame:
    """
    Compute features for all points in a GeoDataFrame.

    Parameters
    ----------
    points_gdf : GeoDataFrame
        Points with geometry and date
    static_rasters : dict
        Static feature rasters
    gridmet_ds : xr.Dataset
        GridMET data
    date_column : str
        Name of date column

    Returns
    -------
    DataFrame
        Features for all points
    """
    engineer = IgnitionFeatureEngineer(
        static_rasters=static_rasters,
        gridmet_ds=gridmet_ds,
    )

    records = []

    for idx, row in points_gdf.iterrows():
        point = row.geometry
        date = row[date_column] if date_column in row.index else datetime.now()

        if pd.isna(date):
            continue

        features = engineer.compute_features_for_point(
            x=point.x,
            y=point.y,
            date=pd.to_datetime(date),
        )

        records.append(features)

    return pd.DataFrame(records)
