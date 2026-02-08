"""Weather stream generation for fire spread simulations."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
import xarray as xr

from config.settings import get_config

logger = logging.getLogger(__name__)


@dataclass
class WeatherScenario:
    """Weather scenario for fire simulation."""
    date: datetime
    temp_max_f: float  # Fahrenheit
    temp_min_f: float
    rh_max: float  # Percent
    rh_min: float
    wind_speed_mph: float
    wind_direction: float  # Degrees from north
    precipitation: float  # Inches
    erc: float  # Energy Release Component
    fuel_moisture_1hr: float
    fuel_moisture_10hr: float
    fuel_moisture_100hr: float
    is_extreme: bool = False  # Santa Ana/Diablo wind event


@dataclass
class WeatherStream:
    """Multi-day weather stream for simulation."""
    scenarios: List[WeatherScenario]

    def to_flammap_format(self, output_path: Path) -> Path:
        """Write weather stream to FlamMap format."""
        with open(output_path, "w") as f:
            f.write("WEATHER\n")
            f.write("ENGLISH\n")

            for sc in self.scenarios:
                # Month Day Precip MinTemp MaxTemp MinHum MaxHum Elev PST SST
                month = sc.date.month
                day = sc.date.day
                f.write(
                    f"{month} {day} {sc.precipitation:.2f} "
                    f"{sc.temp_min_f:.1f} {sc.temp_max_f:.1f} "
                    f"{sc.rh_min:.0f} {sc.rh_max:.0f} "
                    f"1000 1400 1900\n"  # Default elevation, peak sun, sunset
                )

        logger.info(f"Wrote weather stream to {output_path}")
        return output_path


def create_weather_stream(
    gridmet_ds: xr.Dataset,
    start_date: datetime,
    n_days: int = 7,
    lat: float = 38.5,
    lon: float = -122.5,
) -> WeatherStream:
    """
    Create weather stream from GridMET data.

    Parameters
    ----------
    gridmet_ds : xr.Dataset
        GridMET dataset
    start_date : datetime
        Start date for stream
    n_days : int
        Number of days
    lat, lon : float
        Location coordinates (WGS84)

    Returns
    -------
    WeatherStream
        Weather stream for simulation
    """
    scenarios = []

    # Adjust longitude for GridMET (0-360)
    if lon < 0:
        lon += 360

    for day_offset in range(n_days):
        date = start_date + timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")

        try:
            # Extract values
            point = gridmet_ds.sel(
                day=date_str,
                lat=lat,
                lon=lon,
                method="nearest",
            )

            # Temperature (K to F)
            temp_max_k = float(point["tmmx"].values) if "tmmx" in point else 300
            temp_min_k = float(point["tmmn"].values) if "tmmn" in point else 280
            temp_max_f = (temp_max_k - 273.15) * 9 / 5 + 32
            temp_min_f = (temp_min_k - 273.15) * 9 / 5 + 32

            # Humidity
            rh_max = float(point["rmax"].values) if "rmax" in point else 60
            rh_min = float(point["rmin"].values) if "rmin" in point else 20

            # Wind (m/s to mph)
            wind_speed = float(point["vs"].values) if "vs" in point else 5
            wind_speed_mph = wind_speed * 2.237

            # Wind direction
            wind_dir = float(point["th"].values) if "th" in point else 270

            # Precipitation (mm to inches)
            precip = float(point["pr"].values) if "pr" in point else 0
            precip_in = precip / 25.4

            # ERC
            erc = float(point["erc"].values) if "erc" in point else 50

            # Fuel moisture
            fm100 = float(point["fm100"].values) if "fm100" in point else 10

            # Estimate 1hr and 10hr from 100hr
            fm_1hr = max(2, fm100 - 4)
            fm_10hr = max(3, fm100 - 2)

            # Check for extreme conditions
            is_extreme = (
                wind_speed_mph > 25 and
                rh_min < 15 and
                temp_max_f > 85
            )

            scenario = WeatherScenario(
                date=date,
                temp_max_f=temp_max_f,
                temp_min_f=temp_min_f,
                rh_max=rh_max,
                rh_min=rh_min,
                wind_speed_mph=wind_speed_mph,
                wind_direction=wind_dir,
                precipitation=precip_in,
                erc=erc,
                fuel_moisture_1hr=fm_1hr,
                fuel_moisture_10hr=fm_10hr,
                fuel_moisture_100hr=fm100,
                is_extreme=is_extreme,
            )

        except Exception as e:
            logger.warning(f"Could not get weather for {date_str}: {e}")
            # Use default scenario
            scenario = _get_default_scenario(date)

        scenarios.append(scenario)

    return WeatherStream(scenarios=scenarios)


def sample_weather_scenario(
    gridmet_ds: xr.Dataset,
    lat: float,
    lon: float,
    sample_extreme: bool = False,
    season: Optional[str] = None,
    random_state: Optional[int] = None,
) -> WeatherScenario:
    """
    Sample a random weather scenario from historical data.

    Parameters
    ----------
    gridmet_ds : xr.Dataset
        GridMET historical data
    lat, lon : float
        Location (WGS84)
    sample_extreme : bool
        If True, sample from extreme fire weather days
    season : str, optional
        Restrict to season ("fire" for June-Oct, "santa_ana" for Sep-Dec)
    random_state : int, optional
        Random seed

    Returns
    -------
    WeatherScenario
        Sampled weather scenario
    """
    rng = np.random.default_rng(random_state)

    # Adjust longitude for GridMET
    if lon < 0:
        lon += 360

    # Get time series at location
    try:
        point_data = gridmet_ds.sel(lat=lat, lon=lon, method="nearest")
    except Exception:
        # Return default if location not in data
        return _get_default_scenario(datetime.now())

    # Filter by season if specified
    if season == "fire":
        # Fire season: June - October
        mask = (point_data["day"].dt.month >= 6) & (point_data["day"].dt.month <= 10)
        point_data = point_data.sel(day=mask)
    elif season == "santa_ana":
        # Santa Ana season: September - December
        mask = (point_data["day"].dt.month >= 9) | (point_data["day"].dt.month <= 1)
        point_data = point_data.sel(day=mask)

    if sample_extreme:
        # Filter to extreme fire weather days
        # High wind, low humidity, high temperature
        if "vs" in point_data and "rmin" in point_data and "tmmx" in point_data:
            wind = point_data["vs"]
            rh = point_data["rmin"]
            temp = point_data["tmmx"]

            # 95th percentile thresholds
            wind_thresh = float(wind.quantile(0.95))
            rh_thresh = float(rh.quantile(0.05))  # Low humidity
            temp_thresh = float(temp.quantile(0.90))

            extreme_mask = (
                (wind > wind_thresh) &
                (rh < rh_thresh) &
                (temp > temp_thresh)
            )

            if extreme_mask.any():
                point_data = point_data.sel(day=extreme_mask)

    # Sample random day
    n_days = len(point_data["day"])
    if n_days == 0:
        return _get_default_scenario(datetime.now())

    day_idx = rng.integers(0, n_days)
    sampled_day = point_data.isel(day=day_idx)
    sampled_date = pd.to_datetime(sampled_day["day"].values)

    # Extract values
    temp_max_k = float(sampled_day["tmmx"].values) if "tmmx" in sampled_day else 300
    temp_min_k = float(sampled_day["tmmn"].values) if "tmmn" in sampled_day else 280
    temp_max_f = (temp_max_k - 273.15) * 9 / 5 + 32
    temp_min_f = (temp_min_k - 273.15) * 9 / 5 + 32

    rh_max = float(sampled_day["rmax"].values) if "rmax" in sampled_day else 60
    rh_min = float(sampled_day["rmin"].values) if "rmin" in sampled_day else 20

    wind_speed = float(sampled_day["vs"].values) if "vs" in sampled_day else 5
    wind_speed_mph = wind_speed * 2.237

    wind_dir = float(sampled_day["th"].values) if "th" in sampled_day else 270

    precip = float(sampled_day["pr"].values) if "pr" in sampled_day else 0
    precip_in = precip / 25.4

    erc = float(sampled_day["erc"].values) if "erc" in sampled_day else 50

    fm100 = float(sampled_day["fm100"].values) if "fm100" in sampled_day else 10
    fm_1hr = max(2, fm100 - 4)
    fm_10hr = max(3, fm100 - 2)

    is_extreme = sample_extreme or (
        wind_speed_mph > 25 and rh_min < 15 and temp_max_f > 85
    )

    return WeatherScenario(
        date=sampled_date,
        temp_max_f=temp_max_f,
        temp_min_f=temp_min_f,
        rh_max=rh_max,
        rh_min=rh_min,
        wind_speed_mph=wind_speed_mph,
        wind_direction=wind_dir,
        precipitation=precip_in,
        erc=erc,
        fuel_moisture_1hr=fm_1hr,
        fuel_moisture_10hr=fm_10hr,
        fuel_moisture_100hr=fm100,
        is_extreme=is_extreme,
    )


def _get_default_scenario(date: datetime) -> WeatherScenario:
    """Get default fire weather scenario."""
    return WeatherScenario(
        date=date,
        temp_max_f=90,
        temp_min_f=60,
        rh_max=50,
        rh_min=15,
        wind_speed_mph=15,
        wind_direction=270,
        precipitation=0,
        erc=70,
        fuel_moisture_1hr=4,
        fuel_moisture_10hr=6,
        fuel_moisture_100hr=10,
        is_extreme=False,
    )


def get_fuel_moisture_from_weather(scenario: WeatherScenario) -> Dict[str, float]:
    """
    Convert weather scenario to fuel moisture file values.

    Parameters
    ----------
    scenario : WeatherScenario
        Weather scenario

    Returns
    -------
    dict
        Fuel moisture values by size class
    """
    return {
        "1hr": scenario.fuel_moisture_1hr,
        "10hr": scenario.fuel_moisture_10hr,
        "100hr": scenario.fuel_moisture_100hr,
        "live_herb": 60 if scenario.is_extreme else 80,
        "live_woody": 80 if scenario.is_extreme else 100,
    }


# Need pandas for date conversion
import pandas as pd
