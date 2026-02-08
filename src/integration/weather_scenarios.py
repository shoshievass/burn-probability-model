"""Weather scenario generation for Monte Carlo simulations."""

import logging
from pathlib import Path
from typing import Optional, List, Tuple, Iterator
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
    temp_max: float  # Celsius
    temp_min: float
    rh_max: float  # Percent
    rh_min: float
    wind_speed: float  # m/s
    wind_direction: float  # Degrees from north
    precipitation: float  # mm
    erc: float  # Energy Release Component
    fuel_moisture_100hr: float  # Percent
    is_extreme: bool = False


class WeatherScenarioGenerator:
    """
    Generator for weather scenarios from historical data.

    Samples weather conditions for Monte Carlo fire simulations.
    """

    # Fire season months
    FIRE_SEASON = [6, 7, 8, 9, 10]  # June - October

    # Santa Ana / Diablo wind months
    EXTREME_MONTHS = [9, 10, 11, 12]  # September - December

    # Thresholds for extreme fire weather
    EXTREME_THRESHOLDS = {
        "wind_speed": 10.0,  # m/s
        "rh_min": 15.0,  # %
        "temp_max": 35.0,  # C (95F)
    }

    def __init__(
        self,
        gridmet_ds: xr.Dataset,
        lat: float,
        lon: float,
    ):
        """
        Initialize weather generator.

        Parameters
        ----------
        gridmet_ds : xr.Dataset
            GridMET historical data
        lat, lon : float
            Location (WGS84)
        """
        self.gridmet_ds = gridmet_ds
        self.lat = lat
        self.lon = lon if lon > 0 else lon + 360  # GridMET uses 0-360

        # Extract time series at location
        self.point_data = gridmet_ds.sel(
            lat=lat, lon=self.lon, method="nearest"
        )

        # Build indices for seasonal sampling
        self._build_indices()

    def _build_indices(self):
        """Build indices for efficient sampling."""
        dates = self.point_data["day"].values

        # Fire season days
        months = [np.datetime64(d, "M").astype(int) % 12 + 1 for d in dates]
        months = np.array(months)

        self.fire_season_mask = np.isin(months, self.FIRE_SEASON)
        self.extreme_month_mask = np.isin(months, self.EXTREME_MONTHS)

        # Identify extreme fire weather days
        self.extreme_weather_mask = self._identify_extreme_days()

        logger.info(
            f"Found {self.fire_season_mask.sum()} fire season days, "
            f"{self.extreme_weather_mask.sum()} extreme weather days"
        )

    def _identify_extreme_days(self) -> np.ndarray:
        """Identify extreme fire weather days."""
        mask = np.zeros(len(self.point_data["day"]), dtype=bool)

        if "vs" in self.point_data:
            wind = self.point_data["vs"].values
            mask |= wind > self.EXTREME_THRESHOLDS["wind_speed"]

        if "rmin" in self.point_data:
            rh = self.point_data["rmin"].values
            mask |= rh < self.EXTREME_THRESHOLDS["rh_min"]

        if "tmmx" in self.point_data:
            temp = self.point_data["tmmx"].values - 273.15  # K to C
            mask |= temp > self.EXTREME_THRESHOLDS["temp_max"]

        # Must be in extreme weather season
        mask &= self.extreme_month_mask

        return mask

    def sample_scenario(
        self,
        extreme: bool = False,
        random_state: Optional[np.random.Generator] = None,
    ) -> WeatherScenario:
        """
        Sample a single weather scenario.

        Parameters
        ----------
        extreme : bool
            Sample from extreme weather days
        random_state : Generator, optional
            Random number generator

        Returns
        -------
        WeatherScenario
            Sampled weather conditions
        """
        rng = random_state or np.random.default_rng()

        # Select sampling pool
        if extreme:
            mask = self.extreme_weather_mask
            if not mask.any():
                mask = self.extreme_month_mask
        else:
            mask = self.fire_season_mask

        # Get valid indices
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            valid_indices = np.arange(len(self.point_data["day"]))

        # Random selection
        idx = rng.choice(valid_indices)
        day_data = self.point_data.isel(day=idx)

        # Extract values
        return self._extract_scenario(day_data, extreme)

    def sample_scenarios(
        self,
        n_scenarios: int,
        extreme_fraction: float = 0.15,
        random_state: Optional[int] = None,
    ) -> List[WeatherScenario]:
        """
        Sample multiple weather scenarios.

        Parameters
        ----------
        n_scenarios : int
            Number of scenarios to sample
        extreme_fraction : float
            Fraction of extreme weather scenarios
        random_state : int, optional
            Random seed

        Returns
        -------
        list of WeatherScenario
            Sampled scenarios
        """
        rng = np.random.default_rng(random_state)

        n_extreme = int(n_scenarios * extreme_fraction)
        n_normal = n_scenarios - n_extreme

        scenarios = []

        # Sample extreme scenarios
        for _ in range(n_extreme):
            scenarios.append(self.sample_scenario(extreme=True, random_state=rng))

        # Sample normal fire season scenarios
        for _ in range(n_normal):
            scenarios.append(self.sample_scenario(extreme=False, random_state=rng))

        # Shuffle
        rng.shuffle(scenarios)

        return scenarios

    def _extract_scenario(
        self,
        day_data: xr.Dataset,
        is_extreme: bool,
    ) -> WeatherScenario:
        """Extract scenario from day data."""
        import pandas as pd

        date = pd.to_datetime(day_data["day"].values)

        # Temperature
        temp_max = float(day_data["tmmx"].values) - 273.15 if "tmmx" in day_data else 30
        temp_min = float(day_data["tmmn"].values) - 273.15 if "tmmn" in day_data else 15

        # Humidity
        rh_max = float(day_data["rmax"].values) if "rmax" in day_data else 60
        rh_min = float(day_data["rmin"].values) if "rmin" in day_data else 20

        # Wind
        wind_speed = float(day_data["vs"].values) if "vs" in day_data else 5
        wind_dir = float(day_data["th"].values) if "th" in day_data else 270

        # Precipitation
        precip = float(day_data["pr"].values) if "pr" in day_data else 0

        # Fire indices
        erc = float(day_data["erc"].values) if "erc" in day_data else 50
        fm100 = float(day_data["fm100"].values) if "fm100" in day_data else 10

        return WeatherScenario(
            date=date,
            temp_max=temp_max,
            temp_min=temp_min,
            rh_max=rh_max,
            rh_min=rh_min,
            wind_speed=wind_speed,
            wind_direction=wind_dir,
            precipitation=precip,
            erc=erc,
            fuel_moisture_100hr=fm100,
            is_extreme=is_extreme,
        )

    def get_percentile_scenario(
        self,
        percentile: float,
        variable: str = "erc",
    ) -> WeatherScenario:
        """
        Get scenario at specified percentile of a variable.

        Parameters
        ----------
        percentile : float
            Percentile (0-100)
        variable : str
            Variable to compute percentile for

        Returns
        -------
        WeatherScenario
            Scenario at percentile
        """
        if variable not in self.point_data:
            raise ValueError(f"Variable {variable} not in dataset")

        values = self.point_data[variable].values
        mask = self.fire_season_mask

        # Compute percentile
        threshold = np.nanpercentile(values[mask], percentile)

        # Find closest day
        diffs = np.abs(values - threshold)
        diffs[~mask] = np.inf
        idx = np.argmin(diffs)

        day_data = self.point_data.isel(day=idx)

        return self._extract_scenario(day_data, is_extreme=percentile > 90)


def generate_fire_season_scenarios(
    gridmet_ds: xr.Dataset,
    lat: float,
    lon: float,
    n_scenarios: int = 1000,
    extreme_fraction: float = 0.15,
    random_state: Optional[int] = None,
) -> List[WeatherScenario]:
    """
    Generate weather scenarios for fire season.

    Parameters
    ----------
    gridmet_ds : xr.Dataset
        Historical GridMET data
    lat, lon : float
        Location (WGS84)
    n_scenarios : int
        Number of scenarios
    extreme_fraction : float
        Fraction of extreme weather
    random_state : int, optional
        Random seed

    Returns
    -------
    list of WeatherScenario
        Generated scenarios
    """
    generator = WeatherScenarioGenerator(gridmet_ds, lat, lon)

    return generator.sample_scenarios(
        n_scenarios=n_scenarios,
        extreme_fraction=extreme_fraction,
        random_state=random_state,
    )


def analyze_weather_distribution(
    gridmet_ds: xr.Dataset,
    lat: float,
    lon: float,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Analyze historical weather distribution.

    Parameters
    ----------
    gridmet_ds : xr.Dataset
        Historical data
    lat, lon : float
        Location
    output_dir : Path, optional
        Directory for plots

    Returns
    -------
    dict
        Distribution statistics
    """
    generator = WeatherScenarioGenerator(gridmet_ds, lat, lon)

    stats = {}

    # Fire season statistics
    fire_mask = generator.fire_season_mask

    for var in ["tmmx", "rmin", "vs", "erc"]:
        if var in generator.point_data:
            values = generator.point_data[var].values[fire_mask]
            values = values[~np.isnan(values)]

            if var == "tmmx":
                values = values - 273.15  # K to C

            stats[var] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p50": float(np.percentile(values, 50)),
                "p90": float(np.percentile(values, 90)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
            }

    # Extreme weather frequency
    extreme_mask = generator.extreme_weather_mask
    stats["extreme_day_fraction"] = float(extreme_mask.sum() / fire_mask.sum())

    # Santa Ana / high wind days
    if "vs" in generator.point_data:
        wind = generator.point_data["vs"].values
        high_wind = wind > 15  # m/s
        stats["high_wind_days"] = int(high_wind[generator.extreme_month_mask].sum())

    return stats
