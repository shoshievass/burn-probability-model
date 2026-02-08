"""Build landscape files for FlamMap from source data."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR

logger = logging.getLogger(__name__)


class LandscapeBuilder:
    """
    Builder for FlamMap landscape files.

    Creates .LCP files from various source datasets.
    """

    # Required layers for landscape file
    REQUIRED_LAYERS = ["elevation", "slope", "aspect", "fuel_model"]

    # Optional crown fuel layers
    CROWN_LAYERS = ["canopy_cover", "canopy_height", "canopy_base_height", "canopy_bulk_density"]

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        resolution: int = 30,
        crs: str = "EPSG:3310",
    ):
        """
        Initialize landscape builder.

        Parameters
        ----------
        bounds : tuple
            (minx, miny, maxx, maxy) in target CRS
        resolution : int
            Output resolution in meters
        crs : str
            Target coordinate reference system
        """
        self.bounds = bounds
        self.resolution = resolution
        self.crs = crs

        minx, miny, maxx, maxy = bounds
        self.ncols = int((maxx - minx) / resolution)
        self.nrows = int((maxy - miny) / resolution)

        # Transform for output grid
        self.transform = rasterio.transform.from_bounds(
            minx, miny, maxx, maxy, self.ncols, self.nrows
        )

        # Layer storage
        self.layers = {}

    def add_dem(self, dem_path: Path) -> "LandscapeBuilder":
        """
        Add DEM and compute terrain derivatives.

        Parameters
        ----------
        dem_path : Path
            Path to DEM raster

        Returns
        -------
        self
        """
        logger.info(f"Adding DEM from {dem_path}")

        # Load and resample DEM
        elevation = self._load_and_resample(dem_path)
        self.layers["elevation"] = elevation

        # Compute slope and aspect
        slope, aspect = self._compute_terrain_derivatives(elevation)
        self.layers["slope"] = slope
        self.layers["aspect"] = aspect

        return self

    def add_fuel_model(
        self,
        fuel_path: Path,
        model_type: str = "anderson13",
    ) -> "LandscapeBuilder":
        """
        Add fuel model layer.

        Parameters
        ----------
        fuel_path : Path
            Path to fuel model raster
        model_type : str
            "anderson13" or "scott_burgan40"

        Returns
        -------
        self
        """
        logger.info(f"Adding fuel model from {fuel_path}")

        fuel_model = self._load_and_resample(fuel_path, Resampling.nearest)

        # Convert Scott/Burgan to Anderson 13 if needed
        if model_type == "scott_burgan40":
            fuel_model = self._convert_sb40_to_anderson13(fuel_model)

        self.layers["fuel_model"] = fuel_model.astype(np.int16)

        return self

    def add_canopy_cover(self, path: Path) -> "LandscapeBuilder":
        """Add canopy cover layer (0-100%)."""
        logger.info(f"Adding canopy cover from {path}")
        data = self._load_and_resample(path)
        self.layers["canopy_cover"] = np.clip(data, 0, 100).astype(np.int16)
        return self

    def add_canopy_height(self, path: Path, units: str = "meters") -> "LandscapeBuilder":
        """Add canopy height layer."""
        logger.info(f"Adding canopy height from {path}")
        data = self._load_and_resample(path)

        # Convert to decimeters for storage
        if units == "meters":
            data = data * 10
        elif units == "feet":
            data = data * 3.048  # feet to decimeters

        self.layers["canopy_height"] = data.astype(np.int16)
        return self

    def add_canopy_base_height(self, path: Path, units: str = "meters") -> "LandscapeBuilder":
        """Add canopy base height layer."""
        logger.info(f"Adding canopy base height from {path}")
        data = self._load_and_resample(path)

        if units == "meters":
            data = data * 10  # to decimeters

        self.layers["canopy_base_height"] = data.astype(np.int16)
        return self

    def add_canopy_bulk_density(self, path: Path, units: str = "kg/m3") -> "LandscapeBuilder":
        """Add canopy bulk density layer."""
        logger.info(f"Adding canopy bulk density from {path}")
        data = self._load_and_resample(path)

        # Convert to kg/m3 * 100 for storage
        if units == "kg/m3":
            data = data * 100

        self.layers["canopy_bulk_density"] = data.astype(np.int16)
        return self

    def build(self, output_path: Path) -> Path:
        """
        Build the landscape file.

        Parameters
        ----------
        output_path : Path
            Output LCP file path

        Returns
        -------
        Path
            Path to created file
        """
        # Validate required layers
        missing = [l for l in self.REQUIRED_LAYERS if l not in self.layers]
        if missing:
            raise ValueError(f"Missing required layers: {missing}")

        logger.info(f"Building landscape file: {output_path}")

        # Check for crown fuel layers
        has_crown = all(l in self.layers for l in self.CROWN_LAYERS)

        # Write LCP file
        self._write_lcp(output_path, has_crown)

        logger.info(f"Created landscape file: {self.nrows}x{self.ncols} at {self.resolution}m")

        return output_path

    def _load_and_resample(
        self,
        path: Path,
        resampling: Resampling = Resampling.bilinear,
    ) -> np.ndarray:
        """Load raster and resample to target grid."""
        with rasterio.open(path) as src:
            # Create output array
            dst = np.empty((self.nrows, self.ncols), dtype=np.float32)

            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.transform,
                dst_crs=self.crs,
                resampling=resampling,
            )

            # Handle nodata
            if src.nodata is not None:
                dst[dst == src.nodata] = np.nan

        return dst

    def _compute_terrain_derivatives(
        self,
        elevation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute slope and aspect from elevation."""
        # Gradient
        dy, dx = np.gradient(elevation, self.resolution)

        # Slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
        slope = np.clip(slope, 0, 89)  # Cap at 89 degrees

        # Aspect in degrees (0-360, clockwise from north)
        aspect = np.degrees(np.arctan2(-dx, dy))
        aspect = np.where(aspect < 0, aspect + 360, aspect)

        # Handle flat areas
        flat_mask = slope < 0.5
        aspect[flat_mask] = -1  # FlamMap convention for flat

        return slope.astype(np.int16), aspect.astype(np.int16)

    def _convert_sb40_to_anderson13(self, fuel: np.ndarray) -> np.ndarray:
        """
        Convert Scott/Burgan 40 fuel models to Anderson 13.

        Simplified crosswalk - actual conversion depends on local conditions.
        """
        # Mapping (approximate)
        sb40_to_a13 = {
            # Non-burnable
            91: 91, 92: 92, 93: 93, 98: 98, 99: 99,
            # Grass (GR)
            101: 1, 102: 1, 103: 3, 104: 3, 105: 3, 106: 3, 107: 3, 108: 3, 109: 3,
            # Grass-Shrub (GS)
            121: 2, 122: 2, 123: 2, 124: 2,
            # Shrub (SH)
            141: 5, 142: 5, 143: 6, 144: 6, 145: 4, 146: 4, 147: 4, 148: 4, 149: 4,
            # Timber-Understory (TU)
            161: 10, 162: 10, 163: 10, 164: 10, 165: 10,
            # Timber Litter (TL)
            181: 8, 182: 8, 183: 8, 184: 9, 185: 8, 186: 8, 187: 8, 188: 9, 189: 8,
            # Slash (SB)
            201: 11, 202: 12, 203: 13, 204: 13,
        }

        result = np.zeros_like(fuel)
        for sb40, a13 in sb40_to_a13.items():
            result[fuel == sb40] = a13

        # Default unmapped values to closest
        unmapped = (result == 0) & (fuel > 0) & (fuel < 90)
        result[unmapped] = 8  # Default to closed timber litter

        return result

    def _write_lcp(self, path: Path, include_crown: bool = True) -> None:
        """Write LCP binary file."""
        import struct
        import pyproj

        minx, miny, maxx, maxy = self.bounds

        # Get approximate WGS84 latitude
        transformer = pyproj.Transformer.from_crs(
            self.crs, "EPSG:4326", always_xy=True
        )
        _, lat = transformer.transform((minx + maxx) / 2, (miny + maxy) / 2)

        with open(path, "wb") as f:
            # Write header (simplified - see full spec for complete header)
            header = struct.pack(
                "<iiddddiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiddddiddhhhhhhhhhhh256s",
                21 if include_crown else 0,  # Crown fuels present
                0,  # Ground fuels
                lat,  # Latitude
                minx, maxx, miny, maxy,  # Bounds
                int(np.nanmin(self.layers["elevation"])),
                int(np.nanmax(self.layers["elevation"])),
                1, 0,  # Elev resolution, units
                0, 90, 1, 0,  # Slope
                0, 360, 1, 0,  # Aspect
                1, 13, 1, 0,  # Fuel
                0, 100, 1, 0,  # Cover
                0, 500, 10, 1,  # Height (decimeters)
                0, 200, 10, 1,  # Base height
                0, 100, 100, 1,  # Bulk density
                0, 0, 1, 0,  # Duff
                0, 0, 1, 0,  # Woody
                self.nrows, self.ncols,
                minx, maxx, maxy, miny,
                0,  # Grid units (meters)
                float(self.resolution), float(self.resolution),
                1, 1, 1, 1,  # Required layers present
                1 if include_crown else 0,  # Cover
                1 if include_crown else 0,  # Height
                1 if include_crown else 0,  # Base
                1 if include_crown else 0,  # Density
                0, 0,  # Duff, Woody
                b"Burn Probability Model\0".ljust(256, b"\0"),
            )
            f.write(header)

            # Write layers
            for layer_name in ["elevation", "slope", "aspect", "fuel_model"]:
                data = np.flipud(self.layers[layer_name])
                f.write(data.astype(np.int16).tobytes())

            if include_crown:
                for layer_name in self.CROWN_LAYERS:
                    if layer_name in self.layers:
                        data = np.flipud(self.layers[layer_name])
                        f.write(data.astype(np.int16).tobytes())
                    else:
                        # Write zeros
                        zeros = np.zeros((self.nrows, self.ncols), dtype=np.int16)
                        f.write(zeros.tobytes())


def build_landscape_from_sources(
    dem_path: Path,
    fuel_model_path: Path,
    bounds: Tuple[float, float, float, float],
    output_path: Path,
    resolution: int = 30,
    canopy_cover_path: Optional[Path] = None,
    canopy_height_path: Optional[Path] = None,
    canopy_base_height_path: Optional[Path] = None,
    canopy_bulk_density_path: Optional[Path] = None,
) -> Path:
    """
    Build landscape file from source rasters.

    Parameters
    ----------
    dem_path : Path
        DEM raster
    fuel_model_path : Path
        Fuel model raster
    bounds : tuple
        (minx, miny, maxx, maxy)
    output_path : Path
        Output LCP path
    resolution : int
        Output resolution
    canopy_cover_path : Path, optional
        Canopy cover raster
    canopy_height_path : Path, optional
        Canopy height raster
    canopy_base_height_path : Path, optional
        Canopy base height raster
    canopy_bulk_density_path : Path, optional
        Canopy bulk density raster

    Returns
    -------
    Path
        Path to created LCP file
    """
    builder = LandscapeBuilder(bounds=bounds, resolution=resolution)

    builder.add_dem(dem_path)
    builder.add_fuel_model(fuel_model_path)

    if canopy_cover_path and canopy_cover_path.exists():
        builder.add_canopy_cover(canopy_cover_path)

    if canopy_height_path and canopy_height_path.exists():
        builder.add_canopy_height(canopy_height_path)

    if canopy_base_height_path and canopy_base_height_path.exists():
        builder.add_canopy_base_height(canopy_base_height_path)

    if canopy_bulk_density_path and canopy_bulk_density_path.exists():
        builder.add_canopy_bulk_density(canopy_bulk_density_path)

    return builder.build(output_path)
