"""Data acquisition module for downloading and managing source data."""

from .fire_history import download_fire_perimeters, download_ignition_points
from .weather import download_gridmet, download_gridmet_variable
from .landfire import download_landfire_products
from .terrain import download_dem
from .parcels import download_county_parcels
from .infrastructure import download_roads, download_power_lines

__all__ = [
    "download_fire_perimeters",
    "download_ignition_points",
    "download_gridmet",
    "download_gridmet_variable",
    "download_landfire_products",
    "download_dem",
    "download_county_parcels",
    "download_roads",
    "download_power_lines",
]
