"""Python wrapper for FlamMap fire spread simulation."""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import shutil
import numpy as np
import rasterio

from config.settings import get_config, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class FlamMapConfig:
    """Configuration for FlamMap simulation."""
    landscape_file: Path
    weather_file: Optional[Path] = None
    wind_file: Optional[Path] = None
    fuel_moisture_file: Optional[Path] = None

    # Simulation parameters
    simulation_time: int = 480  # minutes
    timestep: int = 30  # minutes

    # Ignition
    ignition_file: Optional[Path] = None  # Shapefile or coordinates

    # Output options
    output_flames: bool = True
    output_spread_rate: bool = False
    output_intensity: bool = False
    output_arrival_time: bool = True

    # Crown fire
    crown_fire_method: str = "Finney"  # or "ScottReinhardt"

    # Spotting
    spot_probability: float = 0.0
    spot_delay: int = 0  # minutes


class FlamMapRunner:
    """
    Python wrapper for running FlamMap simulations.

    FlamMap is a USFS fire behavior mapping and analysis program.
    This wrapper provides a Python interface to the command-line version.
    """

    def __init__(
        self,
        flammap_path: Optional[str] = None,
        temp_dir: Optional[Path] = None,
    ):
        """
        Initialize FlamMap runner.

        Parameters
        ----------
        flammap_path : str, optional
            Path to FlamMap executable
        temp_dir : Path, optional
            Temporary directory for input/output files
        """
        self.config = get_config()
        self.flammap_path = flammap_path or self.config.spread.flammap_path
        self.temp_dir = temp_dir or Path(tempfile.mkdtemp(prefix="flammap_"))

        # Check if FlamMap is available
        self._check_flammap()

    def _check_flammap(self) -> bool:
        """Check if FlamMap executable is available."""
        if not Path(self.flammap_path).exists():
            logger.warning(
                f"FlamMap not found at {self.flammap_path}. "
                "Fire spread simulations will use fallback method."
            )
            return False
        return True

    def run_simulation(
        self,
        config: FlamMapConfig,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """
        Run FlamMap simulation.

        Parameters
        ----------
        config : FlamMapConfig
            Simulation configuration
        output_dir : Path, optional
            Directory for output files

        Returns
        -------
        dict
            Mapping of output type to file path
        """
        output_dir = output_dir or self.temp_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create FlamMap input file
        input_file = self._create_input_file(config, output_dir)

        # Run FlamMap
        if Path(self.flammap_path).exists():
            result = self._run_flammap_cli(input_file)

            if result.returncode != 0:
                logger.error(f"FlamMap failed: {result.stderr}")
                return {}
        else:
            # Use fallback spread model
            logger.info("Using fallback fire spread model")
            return self._run_fallback_spread(config, output_dir)

        # Collect outputs
        outputs = {}
        for output_type in ["flames", "arrival_time", "spread_rate", "intensity"]:
            output_file = output_dir / f"{output_type}.tif"
            if output_file.exists():
                outputs[output_type] = output_file

        return outputs

    def _create_input_file(
        self,
        config: FlamMapConfig,
        output_dir: Path,
    ) -> Path:
        """Create FlamMap input/command file."""
        input_path = self.temp_dir / "flammap_inputs.txt"

        with open(input_path, "w") as f:
            # Landscape file
            f.write(f"LANDSCAPE_FILE: {config.landscape_file}\n")

            # Weather
            if config.weather_file:
                f.write(f"WEATHER_FILE: {config.weather_file}\n")

            # Wind
            if config.wind_file:
                f.write(f"WIND_FILE: {config.wind_file}\n")

            # Fuel moisture
            if config.fuel_moisture_file:
                f.write(f"FUEL_MOISTURE_FILE: {config.fuel_moisture_file}\n")

            # Ignition
            if config.ignition_file:
                f.write(f"IGNITION_FILE: {config.ignition_file}\n")

            # Simulation parameters
            f.write(f"SIMULATION_TIME: {config.simulation_time}\n")
            f.write(f"TIMESTEP: {config.timestep}\n")

            # Crown fire
            f.write(f"CROWN_FIRE_METHOD: {config.crown_fire_method}\n")

            # Spotting
            if config.spot_probability > 0:
                f.write(f"SPOT_PROBABILITY: {config.spot_probability}\n")
                f.write(f"SPOT_DELAY: {config.spot_delay}\n")

            # Outputs
            f.write(f"OUTPUT_DIRECTORY: {output_dir}\n")
            if config.output_flames:
                f.write("OUTPUT_FLAMES: YES\n")
            if config.output_arrival_time:
                f.write("OUTPUT_ARRIVAL_TIME: YES\n")
            if config.output_spread_rate:
                f.write("OUTPUT_SPREAD_RATE: YES\n")
            if config.output_intensity:
                f.write("OUTPUT_INTENSITY: YES\n")

        return input_path

    def _run_flammap_cli(self, input_file: Path) -> subprocess.CompletedProcess:
        """Run FlamMap command-line interface."""
        cmd = [self.flammap_path, "-i", str(input_file)]

        logger.info(f"Running FlamMap: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )

        return result

    def _run_fallback_spread(
        self,
        config: FlamMapConfig,
        output_dir: Path,
    ) -> Dict[str, Path]:
        """
        Run simplified fire spread model as fallback.

        Uses basic Rothermel spread rate calculations.
        """
        logger.info("Running fallback fire spread simulation")

        # Load landscape data
        with rasterio.open(config.landscape_file) as src:
            # For LCP files, read bands in order
            # Band order: elev, slope, aspect, fuel, cc, ch, cbh, cbd
            profile = src.profile
            transform = src.transform

        # Simple spread simulation based on fuel and terrain
        spread_result = run_basic_fire_spread(
            landscape_path=config.landscape_file,
            ignition_file=config.ignition_file,
            duration_minutes=config.simulation_time,
            timestep_minutes=config.timestep,
            wind_speed=10.0,  # Default
            wind_direction=270.0,  # Default west wind
        )

        # Save output
        output_file = output_dir / "flames.tif"
        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(spread_result["burned_area"].astype(np.uint8), 1)

        return {"flames": output_file}

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)


def run_flammap_simulation(
    landscape_path: Path,
    ignition_point: Tuple[float, float],
    weather: Dict,
    wind_speed: float,
    wind_direction: float,
    fuel_moisture: Dict[str, float],
    duration_minutes: int = 480,
    output_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    """
    Run a single FlamMap fire spread simulation.

    Parameters
    ----------
    landscape_path : Path
        Path to LCP landscape file
    ignition_point : tuple
        (x, y) ignition coordinates in landscape CRS
    weather : dict
        Weather conditions
    wind_speed : float
        Wind speed in m/s
    wind_direction : float
        Wind direction in degrees (from north)
    fuel_moisture : dict
        Fuel moisture by size class
    duration_minutes : int
        Simulation duration
    output_dir : Path, optional
        Output directory

    Returns
    -------
    dict
        Simulation results including burned_area raster
    """
    output_dir = output_dir or Path(tempfile.mkdtemp(prefix="flammap_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create ignition file
    ignition_path = output_dir / "ignition.shp"
    _create_ignition_shapefile(ignition_point, ignition_path, landscape_path)

    # Create weather/wind/moisture files
    from src.preprocessing.landscape_prep import prepare_flammap_inputs
    inputs = prepare_flammap_inputs(
        landscape_path=landscape_path,
        weather_data=weather,
        wind_speed=wind_speed,
        wind_direction=wind_direction,
        fuel_moisture=fuel_moisture,
        output_dir=output_dir,
    )

    # Create config
    config = FlamMapConfig(
        landscape_file=landscape_path,
        weather_file=inputs.get("weather"),
        wind_file=inputs.get("wind"),
        fuel_moisture_file=inputs.get("fuel_moisture"),
        ignition_file=ignition_path,
        simulation_time=duration_minutes,
    )

    # Run simulation
    runner = FlamMapRunner(temp_dir=output_dir)
    outputs = runner.run_simulation(config, output_dir)

    # Load results
    results = {}
    if "flames" in outputs:
        with rasterio.open(outputs["flames"]) as src:
            results["burned_area"] = src.read(1)
            results["transform"] = src.transform
            results["crs"] = src.crs

    return results


def run_basic_fire_spread(
    landscape_path: Path,
    ignition_file: Optional[Path] = None,
    ignition_point: Optional[Tuple[float, float]] = None,
    duration_minutes: int = 480,
    timestep_minutes: int = 30,
    wind_speed: float = 10.0,
    wind_direction: float = 270.0,
) -> Dict[str, np.ndarray]:
    """
    Run basic fire spread simulation without FlamMap.

    Uses simplified Rothermel spread rate calculations with elliptical fire
    spread based on wind direction, terrain slope effects, and fuel-based
    spread rates.

    Parameters
    ----------
    landscape_path : Path
        Path to landscape file (LCP or GeoTiff stack)
    ignition_file : Path, optional
        Ignition shapefile
    ignition_point : tuple, optional
        (x, y) ignition coordinates
    duration_minutes : int
        Simulation duration
    timestep_minutes : int
        Time step
    wind_speed : float
        Wind speed (m/s)
    wind_direction : float
        Wind from direction (degrees, 0=N, 90=E, 180=S, 270=W)

    Returns
    -------
    dict
        Simulation results
    """
    logger.info("Running basic fire spread simulation")

    # Load landscape
    with rasterio.open(landscape_path) as src:
        if src.count >= 4:
            elevation = src.read(1).astype(float)
            slope = src.read(2).astype(float)
            aspect = src.read(3).astype(float)
            fuel_model = src.read(4).astype(int)
        else:
            # Single band - assume fuel model
            fuel_model = src.read(1).astype(int)
            elevation = np.zeros_like(fuel_model, dtype=float)
            slope = np.zeros_like(fuel_model, dtype=float)
            aspect = np.zeros_like(fuel_model, dtype=float)

        transform = src.transform
        height, width = fuel_model.shape
        cell_size = abs(transform[0])

    # Initialize burned area and arrival time
    burned = np.zeros((height, width), dtype=bool)
    arrival_time = np.full((height, width), np.inf)

    # Set ignition
    ignition_cells = []
    if ignition_point:
        col = int((ignition_point[0] - transform.c) / transform.a)
        row = int((ignition_point[1] - transform.f) / transform.e)
        if 0 <= row < height and 0 <= col < width:
            burned[row, col] = True
            arrival_time[row, col] = 0
            ignition_cells.append((row, col))
    elif ignition_file and Path(ignition_file).exists():
        import geopandas as gpd
        ignitions = gpd.read_file(ignition_file)
        for _, row_data in ignitions.iterrows():
            point = row_data.geometry
            col = int((point.x - transform.c) / transform.a)
            row_idx = int((point.y - transform.f) / transform.e)
            if 0 <= row_idx < height and 0 <= col < width:
                burned[row_idx, col] = True
                arrival_time[row_idx, col] = 0
                ignition_cells.append((row_idx, col))
    else:
        # Default ignition in center
        row_idx, col = height // 2, width // 2
        burned[row_idx, col] = True
        arrival_time[row_idx, col] = 0
        ignition_cells.append((row_idx, col))

    # Compute base spread rate from fuel model (m/min)
    base_ros = _get_base_spread_rates(fuel_model)

    # Determine burnable mask
    burnable = (
        ((fuel_model >= 1) & (fuel_model <= 13)) |  # Anderson 13
        ((fuel_model >= 101) & (fuel_model <= 204))  # Scott/Burgan 40
    )
    non_burnable = np.isin(fuel_model, [91, 92, 93, 98, 99, -9999, 0])
    burnable = burnable & ~non_burnable

    # Wind direction: convert from "wind from" to "wind to" direction
    # Wind blowing FROM 270 (west) means fire spreads TO the east
    wind_to_deg = (wind_direction + 180) % 360
    wind_to_rad = np.radians(wind_to_deg)

    # Compute elliptical spread parameters based on wind speed
    # Length-to-breadth ratio increases with wind speed (Anderson 1983)
    # LB = 1 + 0.25 * wind_speed (approximate relationship)
    length_to_breadth = 1.0 + 0.25 * wind_speed
    length_to_breadth = min(length_to_breadth, 8.0)  # Cap at realistic max

    # Eccentricity of ellipse
    if length_to_breadth > 1:
        eccentricity = np.sqrt(1 - 1 / (length_to_breadth ** 2))
    else:
        eccentricity = 0

    # 8-connected neighbor offsets (row, col) and their angles from center
    # Angles: 0=N, 45=NE, 90=E, 135=SE, 180=S, 225=SW, 270=W, 315=NW
    neighbors = [
        (-1, 0, 0),      # N
        (-1, 1, 45),     # NE
        (0, 1, 90),      # E
        (1, 1, 135),     # SE
        (1, 0, 180),     # S
        (1, -1, 225),    # SW
        (0, -1, 270),    # W
        (-1, -1, 315),   # NW
    ]

    # Distance to each neighbor in cells
    neighbor_distances = [
        cell_size,                    # N
        cell_size * np.sqrt(2),       # NE
        cell_size,                    # E
        cell_size * np.sqrt(2),       # SE
        cell_size,                    # S
        cell_size * np.sqrt(2),       # SW
        cell_size,                    # W
        cell_size * np.sqrt(2),       # NW
    ]

    # Precompute slope effect for uphill/downhill spread
    # Slope in degrees, aspect in degrees (direction slope faces)
    slope_rad = np.radians(np.clip(slope, 0, 60))  # Cap extreme slopes

    # Time stepping - use smaller internal timestep for accuracy
    # but more iterations for longer fire spread
    internal_timestep = min(timestep_minutes, 5)  # 5 min max internal step
    n_steps = int(duration_minutes / internal_timestep)
    time_per_step = internal_timestep  # minutes

    logger.info(f"Running {n_steps} timesteps of {time_per_step} min each")

    for step in range(n_steps):
        current_time = step * time_per_step

        # Find active fire front (burned cells that might spread)
        from scipy.ndimage import binary_dilation
        kernel = np.ones((3, 3), dtype=bool)
        potential_spread = binary_dilation(burned, structure=kernel) & ~burned & burnable

        # Get coordinates of potential spread cells
        spread_candidates = np.argwhere(potential_spread)

        if len(spread_candidates) == 0:
            continue

        for target_row, target_col in spread_candidates:
            # Check each neighbor that is already burning
            min_arrival = np.inf

            for dr, dc, neighbor_angle in neighbors:
                src_row = target_row - dr
                src_col = target_col - dc

                if not (0 <= src_row < height and 0 <= src_col < width):
                    continue
                if not burned[src_row, src_col]:
                    continue

                # Calculate spread direction angle (from source to target)
                spread_angle_deg = neighbor_angle
                spread_angle_rad = np.radians(spread_angle_deg)

                # Distance to this neighbor
                idx = neighbors.index((dr, dc, neighbor_angle))
                distance = neighbor_distances[idx]

                # Base spread rate from fuel model at target cell
                ros_base = base_ros[target_row, target_col]
                if ros_base <= 0:
                    continue

                # 1. WIND EFFECT - Elliptical spread shape
                # Calculate angle difference between spread direction and wind direction
                angle_diff = spread_angle_rad - wind_to_rad
                # Use ellipse equation: r = a * (1 - e) / (1 - e * cos(theta))
                # where a = semi-major axis, e = eccentricity
                # This gives faster spread downwind, slower upwind
                if eccentricity > 0:
                    # Ellipse factor: 1.0 at downwind, reduced at flanks/upwind
                    cos_angle = np.cos(angle_diff)
                    # Headfire (downwind) gets full LB multiplier
                    # Backfire (upwind) gets 1/LB multiplier
                    # Flanks get 1.0
                    wind_factor = (1 + eccentricity * cos_angle) / (1 - eccentricity ** 2)
                    wind_factor = max(wind_factor, 0.1)  # Minimum spread rate
                else:
                    wind_factor = 1.0

                # Additional wind speed multiplier on base rate
                wind_speed_factor = 1.0 + 0.15 * wind_speed

                # 2. SLOPE EFFECT - Fires spread faster uphill
                # Calculate if we're spreading uphill or downhill
                target_elev = elevation[target_row, target_col]
                src_elev = elevation[src_row, src_col]
                elev_diff = target_elev - src_elev

                # Slope at target cell
                target_slope = slope_rad[target_row, target_col]

                # Check if spread direction aligns with upslope
                # Aspect is direction the slope faces (downhill direction)
                target_aspect = aspect[target_row, target_col]
                upslope_direction = (target_aspect + 180) % 360

                # Angle between spread direction and upslope
                slope_alignment = np.cos(np.radians(spread_angle_deg - upslope_direction))

                # Slope factor: Rothermel uses phi_s = 5.275 * beta^(-0.3) * tan(slope)^2
                # Simplified: faster uphill, slower downhill
                if slope_alignment > 0:  # Spreading uphill
                    # Uphill acceleration: tan^2 relationship
                    slope_factor = 1.0 + 5.0 * (np.tan(target_slope) ** 2) * slope_alignment
                else:  # Spreading downhill or across slope
                    # Slight reduction for downhill spread
                    slope_factor = 1.0 + 0.5 * slope_alignment

                slope_factor = np.clip(slope_factor, 0.2, 10.0)

                # 3. Combined spread rate (m/min)
                ros_effective = ros_base * wind_factor * wind_speed_factor * slope_factor

                # Time to spread this distance
                if ros_effective > 0:
                    travel_time = distance / ros_effective  # minutes
                    potential_arrival = arrival_time[src_row, src_col] + travel_time

                    if potential_arrival < min_arrival:
                        min_arrival = potential_arrival

            # Check if fire arrives at this cell during this timestep
            if min_arrival <= current_time + time_per_step:
                if min_arrival < arrival_time[target_row, target_col]:
                    arrival_time[target_row, target_col] = min_arrival
                    burned[target_row, target_col] = True

    # Set arrival time to NaN for unburned cells
    arrival_time[~burned] = np.nan

    logger.info(f"Fire spread complete: {burned.sum()} cells burned")

    return {
        "burned_area": burned.astype(np.uint8),
        "arrival_time": arrival_time,
        "transform": transform,
    }


def _get_base_spread_rates(fuel_model: np.ndarray) -> np.ndarray:
    """
    Get base spread rates for each fuel model.

    Parameters
    ----------
    fuel_model : ndarray
        Fuel model codes

    Returns
    -------
    ndarray
        Base spread rate in m/min for each cell
    """
    # Base spread rates for fuel models (m/min at no wind/slope)
    # From Rothermel 1972 (Anderson 13) and Scott/Burgan 2005 (FBFM40)
    base_ros = {
        # Anderson 13 models
        1: 5.0,   # Short grass
        2: 2.5,   # Timber grass
        3: 15.0,  # Tall grass
        4: 8.0,   # Chaparral
        5: 2.0,   # Short brush
        6: 2.5,   # Dormant brush
        7: 3.5,   # Southern rough
        8: 0.5,   # Closed timber litter
        9: 1.5,   # Hardwood litter
        10: 2.5,  # Timber understory
        11: 1.5,  # Light logging slash
        12: 3.0,  # Medium logging slash
        13: 4.0,  # Heavy logging slash
        # Scott/Burgan 40 models
        101: 5.0, 102: 8.0, 103: 10.0, 104: 15.0, 105: 12.0,
        106: 10.0, 107: 18.0, 108: 20.0, 109: 25.0,  # GR1-9
        121: 3.0, 122: 5.0, 123: 6.0, 124: 8.0,  # GS1-4
        141: 3.0, 142: 4.0, 143: 5.0, 144: 6.0, 145: 8.0,
        146: 4.0, 147: 6.0, 148: 4.0, 149: 6.0,  # SH1-9
        161: 2.0, 162: 3.0, 163: 4.0, 164: 3.0, 165: 5.0,  # TU1-5
        181: 0.5, 182: 0.8, 183: 1.0, 184: 1.2, 185: 1.5,
        186: 1.8, 187: 2.0, 188: 2.5, 189: 3.0,  # TL1-9
        201: 3.0, 202: 4.0, 203: 5.0, 204: 6.0,  # SB1-4
    }

    # Initialize spread rate array
    ros = np.zeros_like(fuel_model, dtype=float)

    for model, rate in base_ros.items():
        ros[fuel_model == model] = rate

    return ros


def _compute_spread_rate(
    fuel_model: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
    wind_speed: float,
    wind_direction: float,
) -> np.ndarray:
    """
    Compute fire spread rate using simplified Rothermel equations.

    Parameters
    ----------
    fuel_model : ndarray
        Fuel model codes (Anderson 13)
    slope : ndarray
        Slope in degrees
    aspect : ndarray
        Aspect in degrees
    wind_speed : float
        Wind speed (m/s)
    wind_direction : float
        Wind from direction (degrees)

    Returns
    -------
    ndarray
        Spread rate in m/s
    """
    # Base spread rates for fuel models (m/min at no wind/slope)
    # From Rothermel 1972 (Anderson 13) and Scott/Burgan 2005 (FBFM40)
    base_ros = {
        # Anderson 13 models
        1: 5.0,   # Short grass
        2: 2.5,   # Timber grass
        3: 15.0,  # Tall grass
        4: 8.0,   # Chaparral
        5: 2.0,   # Short brush
        6: 2.5,   # Dormant brush
        7: 3.5,   # Southern rough
        8: 0.5,   # Closed timber litter
        9: 1.5,   # Hardwood litter
        10: 2.5,  # Timber understory
        11: 1.5,  # Light logging slash
        12: 3.0,  # Medium logging slash
        13: 4.0,  # Heavy logging slash
        # Scott/Burgan 40 models (GR = grass, GS = grass-shrub, SH = shrub, TU = timber-understory, TL = timber-litter, SB = slash)
        101: 5.0, 102: 8.0, 103: 10.0, 104: 15.0, 105: 12.0, 106: 10.0, 107: 18.0, 108: 20.0, 109: 25.0,  # GR1-9
        121: 3.0, 122: 5.0, 123: 6.0, 124: 8.0,  # GS1-4
        141: 3.0, 142: 4.0, 143: 5.0, 144: 6.0, 145: 8.0, 146: 4.0, 147: 6.0, 148: 4.0, 149: 6.0,  # SH1-9
        161: 2.0, 162: 3.0, 163: 4.0, 164: 3.0, 165: 5.0,  # TU1-5
        181: 0.5, 182: 0.8, 183: 1.0, 184: 1.2, 185: 1.5, 186: 1.8, 187: 2.0, 188: 2.5, 189: 3.0,  # TL1-9
        201: 3.0, 202: 4.0, 203: 5.0, 204: 6.0,  # SB1-4
    }

    # Initialize spread rate
    ros = np.zeros_like(slope, dtype=float)

    for model, rate in base_ros.items():
        ros[fuel_model == model] = rate

    # Wind factor (simplified)
    wind_factor = 1 + 0.1 * wind_speed  # Increases with wind

    # Slope factor (simplified)
    # Upslope spread is faster
    slope_rad = np.radians(slope)
    slope_factor = 1 + 2 * np.tan(slope_rad)
    slope_factor = np.clip(slope_factor, 1, 10)

    # Apply factors
    ros = ros * wind_factor * slope_factor

    # Convert to m/s
    ros = ros / 60

    return ros


def _create_ignition_shapefile(
    point: Tuple[float, float],
    output_path: Path,
    template_path: Path,
) -> Path:
    """Create ignition point shapefile."""
    import geopandas as gpd
    from shapely.geometry import Point

    # Get CRS from template
    with rasterio.open(template_path) as src:
        crs = src.crs

    # Create point
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[Point(point[0], point[1])],
        crs=crs,
    )

    gdf.to_file(output_path)
    return output_path
