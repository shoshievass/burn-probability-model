"""Prepare landscape files for FlamMap."""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import struct
import numpy as np
import rasterio

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class LandscapeFile:
    """
    FlamMap Landscape File (.LCP) builder.

    LCP format specification based on FlamMap/FARSITE documentation.
    Contains terrain and fuel data in a single binary file.
    """

    # LCP file constants
    CROWN_FUELS = 21  # Include crown fuel layers
    GROUND_FUELS = 22  # Include ground fuel layers

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: int,
        crs: str = "EPSG:3310",
    ):
        """
        Initialize landscape file builder.

        Parameters
        ----------
        bounds : tuple
            (minx, miny, maxx, maxy) in CRS units
        resolution : int
            Cell size in CRS units (meters)
        crs : str
            Coordinate reference system
        """
        self.bounds = bounds
        self.resolution = resolution
        self.crs = crs

        minx, miny, maxx, maxy = bounds
        self.ncols = int((maxx - minx) / resolution)
        self.nrows = int((maxy - miny) / resolution)

        # Layer data
        self.layers = {}

    def add_layer(self, name: str, data: np.ndarray) -> None:
        """Add a data layer."""
        if data.shape != (self.nrows, self.ncols):
            raise ValueError(
                f"Layer {name} shape {data.shape} doesn't match "
                f"expected ({self.nrows}, {self.ncols})"
            )
        self.layers[name] = data

    def write(self, output_path: Path) -> Path:
        """
        Write LCP file.

        Parameters
        ----------
        output_path : Path
            Output file path

        Returns
        -------
        Path
            Path to written file
        """
        # Validate required layers
        required = ["elevation", "slope", "aspect", "fuel_model"]
        missing = [r for r in required if r not in self.layers]
        if missing:
            raise ValueError(f"Missing required layers: {missing}")

        logger.info(f"Writing LCP file: {output_path}")

        with open(output_path, "wb") as f:
            # Write header
            self._write_header(f)

            # Write layers in order
            layer_order = [
                "elevation", "slope", "aspect", "fuel_model",
                "canopy_cover", "canopy_height", "canopy_base_height",
                "canopy_bulk_density",
            ]

            for layer_name in layer_order:
                if layer_name in self.layers:
                    self._write_layer(f, self.layers[layer_name])

        logger.info(f"Wrote LCP file with {len(self.layers)} layers")
        return output_path

    def _write_header(self, f) -> None:
        """Write LCP file header."""
        minx, miny, maxx, maxy = self.bounds

        # Header structure (7316 bytes total for FlamMap 5.0+)
        # See FlamMap documentation for full specification

        # Crown fuels present (1 = yes, includes CC, CH, CBH, CBD)
        crown_fuels = 1 if "canopy_cover" in self.layers else 0

        # Ground fuels present (not used in basic version)
        ground_fuels = 0

        # Latitude (for solar calculations)
        lat = (miny + maxy) / 2
        # Convert from Albers to approximate WGS84 latitude
        import pyproj
        transformer = pyproj.Transformer.from_crs(
            self.crs, "EPSG:4326", always_xy=True
        )
        _, lat_wgs84 = transformer.transform(
            (minx + maxx) / 2, (miny + maxy) / 2
        )

        # Units: 0 = Metric, 1 = English
        units = 0

        # Header values
        header = struct.pack(
            "<"  # Little endian
            "i"  # Crown fuels (0 or 21)
            "i"  # Ground fuels (0 or 22)
            "i"  # Latitude (degrees * 100)
            "d"  # Lo East (minx)
            "d"  # Hi East (maxx)
            "d"  # Lo North (miny)
            "d"  # Hi North (maxy)
            "i"  # Lo Elev (from data)
            "i"  # Hi Elev (from data)
            "i"  # Number of values per elevation unit
            "i"  # Elev units (0=metric)
            "i"  # Lo Slope
            "i"  # Hi Slope
            "i"  # Number of values per slope unit
            "i"  # Slope units (0=degrees)
            "i"  # Lo Aspect
            "i"  # Hi Aspect
            "i"  # Number of values per aspect unit
            "i"  # Aspect units (0=grass, 1=aspect cats)
            "i"  # Lo Fuel
            "i"  # Hi Fuel
            "i"  # Number fuel values per unit
            "i"  # Fuel units (0=models, 1=custom)
            "i"  # Lo Cover
            "i"  # Hi Cover
            "i"  # Number cover values per unit
            "i"  # Cover units (0=percent)
            "i"  # Lo Height
            "i"  # Hi Height
            "i"  # Number height values per unit
            "i"  # Height units (1=meters, 2=feet)
            "i"  # Lo Base
            "i"  # Hi Base
            "i"  # Number base values per unit
            "i"  # Base units (1=meters)
            "i"  # Lo Density
            "i"  # Hi Density
            "i"  # Number density values per unit
            "i"  # Density units (1=kg/m3)
            "i"  # Lo Duff
            "i"  # Hi Duff
            "i"  # Number duff values per unit
            "i"  # Duff units
            "i"  # Lo Woody
            "i"  # Hi Woody
            "i"  # Number woody values per unit
            "i"  # Woody units
            "i"  # Number of rows
            "i"  # Number of cols
            "d"  # East UTM (minx)
            "d"  # West UTM (maxx - not used?)
            "d"  # North UTM (maxy)
            "d"  # South UTM (miny)
            "i"  # Grid units (0=meters)
            "d"  # X resolution
            "d"  # Y resolution
            "h"  # Elev file code
            "h"  # Slope file code
            "h"  # Aspect file code
            "h"  # Fuel file code
            "h"  # Cover file code
            "h"  # Height file code
            "h"  # Base file code
            "h"  # Density file code
            "h"  # Duff file code
            "h"  # Woody file code
            "256s"  # Description
            ,
            crown_fuels * 21 if crown_fuels else 0,  # Crown fuels indicator
            ground_fuels * 22 if ground_fuels else 0,  # Ground fuels indicator
            int(lat_wgs84 * 100),  # Latitude
            minx, maxx, miny, maxy,  # Bounds
            int(self.layers["elevation"].min()) if "elevation" in self.layers else 0,
            int(self.layers["elevation"].max()) if "elevation" in self.layers else 1000,
            1,  # Elev resolution
            0,  # Elev units (metric)
            0, 90, 1, 0,  # Slope range/units
            0, 360, 1, 0,  # Aspect range/units
            1, 13, 1, 0,  # Fuel model range/units (Anderson 13)
            0, 100, 1, 0,  # Cover range/units
            0, 100, 10, 1,  # Height range/units (meters * 10)
            0, 50, 10, 1,  # Base height range/units
            0, 50, 100, 1,  # Bulk density range/units (kg/m3 * 100)
            0, 0, 1, 0,  # Duff (not used)
            0, 0, 1, 0,  # Woody (not used)
            self.nrows, self.ncols,
            minx, maxx, maxy, miny,
            0,  # Grid units (meters)
            float(self.resolution), float(self.resolution),
            1, 1, 1, 1,  # File present codes (elev, slope, aspect, fuel)
            1 if crown_fuels else 0,  # Cover
            1 if crown_fuels else 0,  # Height
            1 if crown_fuels else 0,  # Base
            1 if crown_fuels else 0,  # Density
            0, 0,  # Duff, Woody
            b"Burn Probability Model Landscape File\0".ljust(256, b"\0"),
        )

        f.write(header)

    def _write_layer(self, f, data: np.ndarray) -> None:
        """Write a single layer as 16-bit integers."""
        # FlamMap expects row-major order, top to bottom
        # Flip if necessary (rasters are usually stored bottom-up)
        data_flipped = np.flipud(data)

        # Convert to int16
        data_int = data_flipped.astype(np.int16)

        # Write as binary
        f.write(data_int.tobytes())


def create_landscape_file(
    dem_path: Path,
    slope_path: Path,
    aspect_path: Path,
    fuel_model_path: Path,
    canopy_cover_path: Optional[Path] = None,
    canopy_height_path: Optional[Path] = None,
    canopy_base_height_path: Optional[Path] = None,
    canopy_bulk_density_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Path:
    """
    Create FlamMap landscape file from component rasters.

    Parameters
    ----------
    dem_path : Path
        Elevation raster (meters)
    slope_path : Path
        Slope raster (degrees)
    aspect_path : Path
        Aspect raster (degrees)
    fuel_model_path : Path
        Fuel model raster (Anderson 13 or Scott/Burgan 40)
    canopy_cover_path : Path, optional
        Canopy cover raster (percent)
    canopy_height_path : Path, optional
        Canopy height raster (meters)
    canopy_base_height_path : Path, optional
        Canopy base height raster (meters)
    canopy_bulk_density_path : Path, optional
        Canopy bulk density raster (kg/m3)
    output_path : Path, optional
        Output LCP file path
    bounds : tuple, optional
        Clip to bounds

    Returns
    -------
    Path
        Path to created LCP file
    """
    output_path = output_path or PROCESSED_DATA_DIR / "landscape.lcp"

    # Read reference raster
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        profile = src.profile
        transform = src.transform
        crs = str(src.crs)

        if bounds is None:
            bounds = src.bounds

    resolution = abs(transform[0])

    # Initialize landscape file
    lcp = LandscapeFile(
        bounds=(bounds.left, bounds.bottom, bounds.right, bounds.top)
        if hasattr(bounds, 'left') else bounds,
        resolution=int(resolution),
        crs=crs,
    )

    # Add required layers
    lcp.add_layer("elevation", dem)

    with rasterio.open(slope_path) as src:
        lcp.add_layer("slope", src.read(1))

    with rasterio.open(aspect_path) as src:
        lcp.add_layer("aspect", src.read(1))

    with rasterio.open(fuel_model_path) as src:
        lcp.add_layer("fuel_model", src.read(1))

    # Add optional crown fuel layers
    if canopy_cover_path and canopy_cover_path.exists():
        with rasterio.open(canopy_cover_path) as src:
            lcp.add_layer("canopy_cover", src.read(1))

    if canopy_height_path and canopy_height_path.exists():
        with rasterio.open(canopy_height_path) as src:
            # Convert to decimeters for storage
            lcp.add_layer("canopy_height", (src.read(1) * 10).astype(np.int16))

    if canopy_base_height_path and canopy_base_height_path.exists():
        with rasterio.open(canopy_base_height_path) as src:
            lcp.add_layer("canopy_base_height", (src.read(1) * 10).astype(np.int16))

    if canopy_bulk_density_path and canopy_bulk_density_path.exists():
        with rasterio.open(canopy_bulk_density_path) as src:
            # Convert to kg/m3 * 100 for storage
            lcp.add_layer("canopy_bulk_density", (src.read(1) * 100).astype(np.int16))

    # Write LCP file
    lcp.write(output_path)

    return output_path


def prepare_flammap_inputs(
    landscape_path: Path,
    weather_data: Dict,
    wind_speed: float,
    wind_direction: float,
    fuel_moisture: Dict[str, float],
    output_dir: Path,
) -> Dict[str, Path]:
    """
    Prepare all input files for FlamMap simulation.

    Parameters
    ----------
    landscape_path : Path
        Path to LCP file
    weather_data : dict
        Weather conditions
    wind_speed : float
        Wind speed (m/s)
    wind_direction : float
        Wind direction (degrees from north)
    fuel_moisture : dict
        Fuel moisture values by size class
    output_dir : Path
        Output directory

    Returns
    -------
    dict
        Mapping of input type to file path
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = {"landscape": landscape_path}

    # Weather file (.wtr)
    weather_path = output_dir / "weather.wtr"
    _write_weather_file(weather_path, weather_data)
    inputs["weather"] = weather_path

    # Wind file (.wnd)
    wind_path = output_dir / "wind.wnd"
    _write_wind_file(wind_path, wind_speed, wind_direction)
    inputs["wind"] = wind_path

    # Fuel moisture file (.fms)
    fms_path = output_dir / "moisture.fms"
    _write_fuel_moisture_file(fms_path, fuel_moisture)
    inputs["fuel_moisture"] = fms_path

    return inputs


def _write_weather_file(path: Path, weather: Dict) -> None:
    """Write FlamMap weather file."""
    # Simple weather file format
    # Month Day Precip MinTemp MaxTemp MinHum MaxHum Elev PST SST
    with open(path, "w") as f:
        f.write("WEATHER\n")
        f.write("ENGLISH\n")  # or METRIC
        f.write(f"{weather.get('month', 8)} ")  # Month
        f.write(f"{weather.get('day', 15)} ")  # Day
        f.write(f"{weather.get('precip', 0.0):.2f} ")  # Precip
        f.write(f"{weather.get('temp_min', 60):.1f} ")  # Min temp F
        f.write(f"{weather.get('temp_max', 95):.1f} ")  # Max temp F
        f.write(f"{weather.get('rh_min', 15):.0f} ")  # Min RH
        f.write(f"{weather.get('rh_max', 40):.0f} ")  # Max RH
        f.write(f"{weather.get('elevation', 1000):.0f} ")  # Elev
        f.write(f"{weather.get('pst', 1600):.0f} ")  # Peak sun time
        f.write(f"{weather.get('sst', 1800):.0f}\n")  # Sunset time


def _write_wind_file(path: Path, speed: float, direction: float) -> None:
    """Write FlamMap wind file."""
    # Convert m/s to mph for FlamMap
    speed_mph = speed * 2.237

    with open(path, "w") as f:
        f.write("WINDS\n")
        f.write("ENGLISH\n")
        # Month Day Hour Speed Direction CloudCover
        f.write(f"8 15 1200 {speed_mph:.1f} {direction:.0f} 0\n")


def _write_fuel_moisture_file(path: Path, moisture: Dict[str, float]) -> None:
    """Write FlamMap fuel moisture file."""
    with open(path, "w") as f:
        f.write("FUEL_MOISTURES\n")
        # Fuel model followed by moisture values
        # 1hr 10hr 100hr LiveHerb LiveWoody
        for model in range(1, 14):  # Anderson 13 models
            f.write(f"{model} ")
            f.write(f"{moisture.get('1hr', 4):.0f} ")
            f.write(f"{moisture.get('10hr', 5):.0f} ")
            f.write(f"{moisture.get('100hr', 8):.0f} ")
            f.write(f"{moisture.get('live_herb', 60):.0f} ")
            f.write(f"{moisture.get('live_woody', 90):.0f}\n")
