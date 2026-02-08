"""Download weather data from GridMET and RTMA."""

import logging
from pathlib import Path
from typing import Optional, Tuple, List
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 - needed for rio accessor

from config.settings import RAW_DATA_DIR

logger = logging.getLogger(__name__)

# GridMET THREDDS server
GRIDMET_BASE_URL = "http://thredds.northwestknowledge.net:8080/thredds/dodsC"

# Variable mappings
GRIDMET_VARIABLES = {
    "tmmx": "daily_maximum_temperature",
    "tmmn": "daily_minimum_temperature",
    "rmax": "daily_maximum_relative_humidity",
    "rmin": "daily_minimum_relative_humidity",
    "vs": "daily_mean_wind_speed",
    "pr": "daily_precipitation_amount",
    "erc": "energy_release_component-g",
    "fm100": "dead_fuel_moisture_100hr",
    "fm1000": "dead_fuel_moisture_1000hr",
    "bi": "burning_index_g",
    "th": "wind_from_direction",  # wind direction
}


def download_gridmet(
    variables: Optional[List[str]] = None,
    years: Optional[Tuple[int, int]] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    output_dir: Optional[Path] = None,
) -> xr.Dataset:
    """
    Download GridMET meteorological data.

    Parameters
    ----------
    variables : list of str, optional
        Variables to download. Default: ["tmmx", "rmin", "vs", "erc", "fm100"]
    years : tuple of int, optional
        (start_year, end_year) inclusive
    bounds : tuple, optional
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path, optional
        Directory to save data

    Returns
    -------
    xr.Dataset
        GridMET data for specified variables and region
    """
    output_dir = output_dir or RAW_DATA_DIR / "weather" / "gridmet"
    output_dir.mkdir(parents=True, exist_ok=True)

    if variables is None:
        variables = ["tmmx", "rmin", "vs", "erc", "fm100", "pr"]

    if years is None:
        years = (2010, 2024)

    # Default to California bounds
    if bounds is None:
        bounds = (-124.5, 32.5, -114.0, 42.1)

    logger.info(f"Downloading GridMET data for {variables}, years {years}")

    datasets = []

    for var in variables:
        logger.info(f"Downloading {var}...")
        ds = download_gridmet_variable(
            variable=var,
            years=years,
            bounds=bounds,
            output_dir=output_dir,
        )
        if ds is not None:
            datasets.append(ds)

    if not datasets:
        raise RuntimeError("No GridMET data downloaded")

    # Merge all variables
    combined = xr.merge(datasets)

    # Save combined dataset
    output_file = output_dir / f"gridmet_{years[0]}_{years[1]}.nc"
    combined.to_netcdf(output_file)
    logger.info(f"Saved combined GridMET data to {output_file}")

    return combined


def download_gridmet_variable(
    variable: str,
    years: Tuple[int, int],
    bounds: Tuple[float, float, float, float],
    output_dir: Optional[Path] = None,
) -> Optional[xr.Dataset]:
    """
    Download a single GridMET variable.

    Parameters
    ----------
    variable : str
        Variable name (e.g., "tmmx", "vs", "erc")
    years : tuple of int
        (start_year, end_year) inclusive
    bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    output_dir : Path, optional
        Directory to save data

    Returns
    -------
    xr.Dataset or None
        Dataset for the variable, or None if download failed
    """
    output_dir = output_dir or RAW_DATA_DIR / "weather" / "gridmet"
    output_dir.mkdir(parents=True, exist_ok=True)

    # GridMET THREDDS URL pattern
    url = f"{GRIDMET_BASE_URL}/agg_met_{variable}_1979_CurrentYear_CONUS.nc"

    try:
        # Open remote dataset
        ds = xr.open_dataset(url, engine="netcdf4")

        # Subset by time
        start_date = f"{years[0]}-01-01"
        end_date = f"{years[1]}-12-31"
        ds = ds.sel(day=slice(start_date, end_date))

        # Subset by space (GridMET uses lon/lat)
        # GridMET longitude is in 0-360 format, convert bounds
        minx, miny, maxx, maxy = bounds
        if minx < 0:
            minx += 360
        if maxx < 0:
            maxx += 360

        ds = ds.sel(lon=slice(minx, maxx), lat=slice(maxy, miny))  # lat is descending

        # Rename coordinates back to standard
        if "lon" in ds.coords:
            ds = ds.assign_coords(lon=(ds.lon - 360).where(ds.lon > 180, ds.lon))

        # Save to file
        output_file = output_dir / f"{variable}_{years[0]}_{years[1]}.nc"
        ds.to_netcdf(output_file)
        logger.info(f"Saved {variable} to {output_file}")

        return ds

    except Exception as e:
        logger.error(f"Failed to download {variable}: {e}")
        return None


def download_gridmet_for_county(
    county_bounds: Tuple[float, float, float, float],
    years: Tuple[int, int],
    output_dir: Optional[Path] = None,
    buffer_deg: float = 0.1,
) -> xr.Dataset:
    """
    Download GridMET data for a specific county with buffer.

    Parameters
    ----------
    county_bounds : tuple
        (minx, miny, maxx, maxy) in WGS84
    years : tuple
        (start_year, end_year)
    output_dir : Path, optional
        Output directory
    buffer_deg : float
        Buffer in degrees around county bounds

    Returns
    -------
    xr.Dataset
        GridMET data for county region
    """
    minx, miny, maxx, maxy = county_bounds
    buffered_bounds = (
        minx - buffer_deg,
        miny - buffer_deg,
        maxx + buffer_deg,
        maxy + buffer_deg,
    )

    return download_gridmet(
        variables=["tmmx", "tmmn", "rmin", "rmax", "vs", "th", "erc", "fm100", "pr"],
        years=years,
        bounds=buffered_bounds,
        output_dir=output_dir,
    )


def load_gridmet(path: Optional[Path] = None) -> xr.Dataset:
    """Load previously downloaded GridMET data."""
    path = path or RAW_DATA_DIR / "weather" / "gridmet"

    if path.is_dir():
        # Find the combined file
        nc_files = list(path.glob("gridmet_*.nc"))
        if nc_files:
            return xr.open_dataset(nc_files[0])

        # Or load individual files
        all_files = list(path.glob("*.nc"))
        if all_files:
            return xr.open_mfdataset(all_files)

    elif path.is_file():
        return xr.open_dataset(path)

    raise FileNotFoundError(f"GridMET data not found at {path}")


def extract_weather_for_date(
    ds: xr.Dataset,
    date: datetime,
    lat: float,
    lon: float,
) -> dict:
    """
    Extract weather values for a specific date and location.

    Parameters
    ----------
    ds : xr.Dataset
        GridMET dataset
    date : datetime
        Date to extract
    lat, lon : float
        Location coordinates (WGS84)

    Returns
    -------
    dict
        Weather variables for the date/location
    """
    # Select nearest grid cell and time
    point = ds.sel(day=date.strftime("%Y-%m-%d"), method="nearest")
    point = point.sel(lat=lat, lon=lon, method="nearest")

    return {var: float(point[var].values) for var in point.data_vars}


def compute_fire_weather_index(
    temp_max: float,
    rh_min: float,
    wind_speed: float,
    precip: float,
) -> float:
    """
    Compute simplified fire weather index.

    This is a simplified version - for production use NFDRS indices from GridMET.

    Parameters
    ----------
    temp_max : float
        Maximum temperature (K)
    rh_min : float
        Minimum relative humidity (%)
    wind_speed : float
        Wind speed (m/s)
    precip : float
        Precipitation (mm)

    Returns
    -------
    float
        Fire weather index (0-100)
    """
    # Convert temp to Celsius
    temp_c = temp_max - 273.15

    # Simple index based on temperature, humidity, wind
    # Higher temp, lower humidity, higher wind = higher index
    temp_factor = min(max((temp_c - 10) / 30, 0), 1)  # 10-40C range
    rh_factor = 1 - min(max(rh_min / 100, 0), 1)  # Invert RH
    wind_factor = min(wind_speed / 15, 1)  # Cap at 15 m/s

    # Reduce if precipitation
    precip_factor = 1 if precip < 1 else 0.5 if precip < 5 else 0.2

    index = (temp_factor * 0.3 + rh_factor * 0.4 + wind_factor * 0.3) * precip_factor
    return index * 100


def identify_santa_ana_days(
    ds: xr.Dataset,
    wind_threshold: float = 10.0,  # m/s
    rh_threshold: float = 20.0,  # %
) -> np.ndarray:
    """
    Identify Santa Ana wind event days.

    Santa Ana conditions:
    - High winds (>10 m/s)
    - Low humidity (<20%)
    - Offshore wind direction (NE to E, roughly 45-135 degrees)

    Parameters
    ----------
    ds : xr.Dataset
        GridMET dataset with wind speed, direction, humidity
    wind_threshold : float
        Minimum wind speed (m/s)
    rh_threshold : float
        Maximum relative humidity (%)

    Returns
    -------
    np.ndarray
        Boolean mask of Santa Ana days
    """
    # Get Southern California region (where Santa Ana occurs)
    so_cal = ds.sel(lat=slice(35, 32), lon=slice(-120, -114))

    # Compute regional mean
    wind_speed = so_cal["vs"].mean(dim=["lat", "lon"])
    rh_min = so_cal["rmin"].mean(dim=["lat", "lon"])

    # Check conditions
    high_wind = wind_speed > wind_threshold
    low_humidity = rh_min < rh_threshold

    # Wind direction check if available
    if "th" in ds.data_vars:
        wind_dir = so_cal["th"].mean(dim=["lat", "lon"])
        # Offshore: 45-135 degrees (NE to SE)
        offshore = (wind_dir > 45) & (wind_dir < 135)
        santa_ana = high_wind & low_humidity & offshore
    else:
        santa_ana = high_wind & low_humidity

    return santa_ana.values
