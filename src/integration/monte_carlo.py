"""Monte Carlo engine for burn probability estimation."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass, field
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import tempfile

from config.settings import get_config, OUTPUT_DIR

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo burn probability simulation."""
    burn_counts: np.ndarray  # Number of times each cell burned
    n_iterations: int
    bounds: Tuple[float, float, float, float]
    resolution: int
    transform: object  # rasterio transform
    calibration_factor: float = 1.0  # Multiplier to calibrate predictions

    @property
    def burn_probability(self) -> np.ndarray:
        """Compute burn probability from counts."""
        raw_prob = self.burn_counts / self.n_iterations
        return np.clip(raw_prob * self.calibration_factor, 0, 1)

    @property
    def raw_burn_probability(self) -> np.ndarray:
        """Uncalibrated burn probability."""
        return self.burn_counts / self.n_iterations

    def save(self, output_path: Path) -> Path:
        """Save burn probability raster."""
        import rasterio

        prob = self.burn_probability

        profile = {
            "driver": "GTiff",
            "dtype": np.float32,
            "width": prob.shape[1],
            "height": prob.shape[0],
            "count": 1,
            "crs": "EPSG:3310",
            "transform": self.transform,
            "nodata": -9999,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(prob.astype(np.float32), 1)

        logger.info(f"Saved burn probability to {output_path}")
        return output_path


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""
    n_iterations: int = 1000
    n_cores: int = 28
    chunk_size: int = 50

    # Weather sampling
    extreme_weather_fraction: float = 0.15
    fire_season_months: Tuple[int, ...] = (6, 7, 8, 9, 10)
    santa_ana_months: Tuple[int, ...] = (9, 10, 11, 12)

    # Ignition parameters
    daily_ignition_rate: float = 0.001  # Ignitions per km² per day during fire season
    use_ignition_model: bool = True

    # Spread simulation
    simulation_duration_minutes: int = 480  # 8 hours per fire
    max_fires_per_iteration: int = 10

    # Calibration
    calibration_factor: float = 1.0  # Multiplier to match observed burn rates
    auto_calibrate: bool = False  # Compute calibration from historical data

    # Random seed
    random_seed: Optional[int] = 42


class MonteCarloEngine:
    """
    Monte Carlo engine for burn probability estimation.

    Combines:
    - Ignition probability model
    - Fire spread simulation (FlamMap)
    - Weather scenario sampling
    """

    def __init__(
        self,
        landscape_path: Path,
        ignition_model: Optional["IgnitionModel"] = None,
        gridmet_ds: Optional["xr.Dataset"] = None,
        config: Optional[MonteCarloConfig] = None,
    ):
        """
        Initialize Monte Carlo engine.

        Parameters
        ----------
        landscape_path : Path
            Path to FlamMap landscape file
        ignition_model : IgnitionModel, optional
            Trained ignition probability model
        gridmet_ds : xr.Dataset, optional
            GridMET historical weather data
        config : MonteCarloConfig, optional
            Simulation configuration
        """
        self.landscape_path = landscape_path
        self.ignition_model = ignition_model
        self.gridmet_ds = gridmet_ds
        self.config = config or MonteCarloConfig()

        # Load landscape metadata
        self._load_landscape_info()

        # Initialize random state
        self.rng = np.random.default_rng(self.config.random_seed)

    def _load_landscape_info(self):
        """Load landscape file metadata."""
        import rasterio

        # Try to open as raster first
        try:
            with rasterio.open(self.landscape_path) as src:
                self.bounds = src.bounds
                self.transform = src.transform
                self.nrows = src.height
                self.ncols = src.width
                self.resolution = abs(src.transform[0])
        except Exception:
            # Fallback for LCP files
            self.bounds = (-200000, -50000, -100000, 50000)  # Default
            self.resolution = 270
            self.ncols = int((self.bounds[2] - self.bounds[0]) / self.resolution)
            self.nrows = int((self.bounds[3] - self.bounds[1]) / self.resolution)
            import rasterio
            self.transform = rasterio.transform.from_bounds(
                *self.bounds, self.ncols, self.nrows
            )

    def run(
        self,
        n_iterations: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        n_iterations : int, optional
            Number of iterations (overrides config)
        progress_callback : callable, optional
            Function called with progress updates

        Returns
        -------
        MonteCarloResult
            Simulation results with burn counts
        """
        n_iterations = n_iterations or self.config.n_iterations

        logger.info(f"Starting Monte Carlo simulation: {n_iterations} iterations")

        # Initialize burn counts
        burn_counts = np.zeros((self.nrows, self.ncols), dtype=np.int32)

        # Determine sampling mode
        use_empirical = self.gridmet_ds is not None
        if use_empirical:
            logger.info("Weather sampling: EMPIRICAL (pure random from fire season days)")
        else:
            # Only use artificial extreme/normal split when no empirical data
            n_extreme = int(n_iterations * self.config.extreme_weather_fraction)
            logger.info(f"Weather sampling: DEFAULT (artificial {self.config.extreme_weather_fraction:.0%} extreme)")

        # Run iterations
        for i in range(n_iterations):
            if i % 100 == 0:
                logger.info(f"Iteration {i}/{n_iterations}")
                if progress_callback:
                    progress_callback(i / n_iterations)

            # Sample weather scenario
            # With empirical data: pure random sampling (no artificial extreme forcing)
            # Without empirical data: use artificial extreme/normal split
            if use_empirical:
                weather = self._sample_weather(extreme=False)  # Pure random from fire season
            else:
                is_extreme = i < n_extreme
                weather = self._sample_weather(extreme=is_extreme)

            # Compute ignition probability
            ignition_prob = self._compute_ignition_probability(weather)

            # Sample ignition locations
            ignition_points = self._sample_ignitions(ignition_prob)

            # Run fire spread for each ignition
            for point in ignition_points:
                burned = self._simulate_fire_spread(point, weather)
                burn_counts += burned

        logger.info(f"Completed {n_iterations} iterations")

        return MonteCarloResult(
            burn_counts=burn_counts,
            n_iterations=n_iterations,
            bounds=self.bounds,
            resolution=self.resolution,
            transform=self.transform,
            calibration_factor=self.config.calibration_factor,
        )

    def run_conditional(
        self,
        ignition_points: List[Tuple[float, float]],
        n_weather_samples: int = 100,
        progress_callback: Optional[Callable] = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation conditional on real ignition events.

        Instead of sampling ignition locations, uses fixed real ignition points
        and only varies weather conditions. This gives P(burn | actual ignitions).

        Parameters
        ----------
        ignition_points : list of (x, y) tuples
            Real ignition locations in projection coordinates (EPSG:3310)
        n_weather_samples : int
            Number of weather scenarios to sample per ignition
        progress_callback : callable, optional
            Function called with progress updates

        Returns
        -------
        MonteCarloResult
            Simulation results with burn counts
        """
        n_ignitions = len(ignition_points)
        total_iterations = n_ignitions * n_weather_samples

        logger.info(
            f"Starting conditional Monte Carlo: {n_ignitions} ignitions × "
            f"{n_weather_samples} weather samples = {total_iterations} simulations"
        )

        # Initialize burn counts
        burn_counts = np.zeros((self.nrows, self.ncols), dtype=np.int32)

        # Determine sampling mode
        use_empirical = self.gridmet_ds is not None
        if use_empirical:
            logger.info("Weather sampling: EMPIRICAL (pure random from fire season days)")
        else:
            # Only use artificial extreme/normal split when no empirical data
            n_extreme = int(n_weather_samples * self.config.extreme_weather_fraction)
            logger.info(f"Weather sampling: DEFAULT (artificial {self.config.extreme_weather_fraction:.0%} extreme)")

        from rasterio.transform import rowcol

        iteration = 0
        for i, point in enumerate(ignition_points):
            # Check if point is within bounds
            if not (self.bounds[0] <= point[0] <= self.bounds[2] and
                    self.bounds[1] <= point[1] <= self.bounds[3]):
                logger.debug(f"Ignition {i} outside bounds, skipping")
                continue

            # Convert (x, y) coordinates to (row, col) grid indices
            try:
                row, col = rowcol(self.transform, point[0], point[1])
                if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
                    logger.debug(f"Ignition {i} at ({point[0]:.0f}, {point[1]:.0f}) -> row={row}, col={col} outside grid")
                    continue
            except Exception as e:
                logger.debug(f"Ignition {i} coordinate conversion failed: {e}")
                continue

            # Run multiple weather scenarios for this ignition
            for w in range(n_weather_samples):
                # With empirical data: pure random sampling (no artificial extreme forcing)
                # Without empirical data: use artificial extreme/normal split
                if use_empirical:
                    weather = self._sample_weather(extreme=False)  # Pure random from fire season
                else:
                    is_extreme = w < n_extreme
                    weather = self._sample_weather(extreme=is_extreme)

                # Simulate fire spread from this ignition (expects row, col indices)
                burned = self._simulate_fire_spread((row, col), weather)
                burn_counts += burned

                iteration += 1
                if iteration % 500 == 0:
                    logger.info(f"Completed {iteration}/{total_iterations} simulations")
                    if progress_callback:
                        progress_callback(iteration / total_iterations)

        logger.info(f"Completed {iteration} total simulations")

        return MonteCarloResult(
            burn_counts=burn_counts,
            n_iterations=total_iterations,
            bounds=self.bounds,
            resolution=self.resolution,
            transform=self.transform,
            calibration_factor=self.config.calibration_factor,
        )

    def _sample_weather(self, extreme: bool = False) -> Dict:
        """
        Sample a weather scenario.

        Parameters
        ----------
        extreme : bool
            If True and no empirical data, use hardcoded extreme weather.
            If empirical data is available, this flag is ignored - we sample
            purely from the historical fire season distribution, so extreme
            events occur at their natural historical frequency.

        Returns
        -------
        dict
            Weather parameters for fire spread simulation
        """
        from src.spread.weather_streams import sample_weather_scenario

        if self.gridmet_ds is not None:
            # EMPIRICAL MODE: Sample random day from fire season
            # Extreme events will occur at their natural historical frequency
            # (no artificial forcing of extreme vs normal)

            # Convert center of bounds from Albers to WGS84
            import pyproj
            transformer = pyproj.Transformer.from_crs(
                "EPSG:3310", "EPSG:4326", always_xy=True
            )
            lon, lat = transformer.transform(
                (self.bounds[0] + self.bounds[2]) / 2,
                (self.bounds[1] + self.bounds[3]) / 2,
            )

            # Pure random sampling from fire season (June-October)
            # sample_extreme=False means no filtering for extreme conditions
            # The historical frequency of extreme events is preserved
            scenario = sample_weather_scenario(
                self.gridmet_ds,
                lat=lat,
                lon=lon,
                sample_extreme=False,  # Pure random, no extreme filtering
                season="fire",  # Fire season: June-October
            )

            return {
                "temp_max": scenario.temp_max_f,
                "temp_min": scenario.temp_min_f,
                "rh_min": scenario.rh_min,
                "rh_max": scenario.rh_max,
                "wind_speed": scenario.wind_speed_mph,
                "wind_direction": scenario.wind_direction,  # Actual historical direction
                "erc": scenario.erc,
                "is_extreme": scenario.is_extreme,  # Determined by actual conditions
            }
        else:
            # DEFAULT MODE: Hardcoded weather (only used when no GridMET data)
            # Uses artificial extreme/normal split with fixed values
            if extreme:
                return {
                    "temp_max": 100,
                    "temp_min": 70,
                    "rh_min": 8,
                    "rh_max": 25,
                    "wind_speed": 35,
                    "wind_direction": 45,  # NE wind (offshore) - hardcoded
                    "erc": 95,
                    "is_extreme": True,
                }
            else:
                return {
                    "temp_max": 90,
                    "temp_min": 60,
                    "rh_min": 20,
                    "rh_max": 50,
                    "wind_speed": 10,
                    "wind_direction": 270,  # W wind - hardcoded
                    "erc": 60,
                    "is_extreme": False,
                }

    def _compute_ignition_probability(self, weather: Dict) -> np.ndarray:
        """Compute ignition probability grid for weather conditions."""
        if self.ignition_model is not None:
            # Use trained model
            # Would need to prepare features grid here
            # For now, use simplified approach
            pass

        # Simplified ignition probability based on ERC and terrain
        base_prob = self.config.daily_ignition_rate * (self.resolution / 1000) ** 2

        # Adjust by ERC (normalized 0-100)
        erc_factor = weather.get("erc", 60) / 60  # Higher ERC = higher ignition

        # Adjust for extreme weather
        if weather.get("is_extreme", False):
            erc_factor *= 2

        # Create probability grid
        ignition_prob = np.full(
            (self.nrows, self.ncols),
            base_prob * erc_factor,
            dtype=np.float32,
        )

        return ignition_prob

    def _sample_ignitions(self, ignition_prob: np.ndarray) -> List[Tuple[int, int]]:
        """Sample ignition locations from probability grid."""
        # Use Poisson process
        n_ignitions = self.rng.poisson(ignition_prob.sum())
        n_ignitions = min(n_ignitions, self.config.max_fires_per_iteration)

        if n_ignitions == 0:
            return []

        # Sample locations weighted by probability
        flat_prob = ignition_prob.flatten()
        flat_prob = flat_prob / flat_prob.sum()  # Normalize

        indices = self.rng.choice(
            len(flat_prob),
            size=n_ignitions,
            replace=False,
            p=flat_prob,
        )

        # Convert to row, col
        rows = indices // self.ncols
        cols = indices % self.ncols

        return list(zip(rows, cols))

    def _simulate_fire_spread(
        self,
        ignition_point: Tuple[int, int],
        weather: Dict,
    ) -> np.ndarray:
        """
        Simulate fire spread from ignition point.

        Returns binary mask of burned cells.
        """
        from src.spread.flammap_wrapper import run_basic_fire_spread

        row, col = ignition_point

        # Convert to coordinates
        x = self.transform.c + col * self.transform.a + self.transform.a / 2
        y = self.transform.f + row * self.transform.e + self.transform.e / 2

        try:
            result = run_basic_fire_spread(
                landscape_path=self.landscape_path,
                ignition_point=(x, y),
                duration_minutes=self.config.simulation_duration_minutes,
                wind_speed=weather.get("wind_speed", 10) / 2.237,  # mph to m/s
                wind_direction=weather.get("wind_direction", 270),
            )

            return result["burned_area"]

        except Exception as e:
            logger.warning(f"Fire spread simulation failed: {e}")
            # Return minimal burn (just ignition cell)
            burned = np.zeros((self.nrows, self.ncols), dtype=np.uint8)
            burned[row, col] = 1
            return burned


def run_monte_carlo_simulation(
    landscape_path: Path,
    ignition_model: Optional["IgnitionModel"] = None,
    gridmet_ds: Optional["xr.Dataset"] = None,
    n_iterations: int = 1000,
    output_path: Optional[Path] = None,
    **kwargs,
) -> MonteCarloResult:
    """
    Run Monte Carlo burn probability simulation.

    Parameters
    ----------
    landscape_path : Path
        Path to landscape file
    ignition_model : IgnitionModel, optional
        Trained ignition model
    gridmet_ds : xr.Dataset, optional
        Historical weather data
    n_iterations : int
        Number of Monte Carlo iterations
    output_path : Path, optional
        Path to save results

    Returns
    -------
    MonteCarloResult
        Simulation results
    """
    config = MonteCarloConfig(n_iterations=n_iterations, **kwargs)

    engine = MonteCarloEngine(
        landscape_path=landscape_path,
        ignition_model=ignition_model,
        gridmet_ds=gridmet_ds,
        config=config,
    )

    result = engine.run()

    if output_path:
        result.save(output_path)

    return result


def run_parallel_monte_carlo(
    landscape_path: Path,
    n_iterations: int = 10000,
    n_workers: int = 28,
    output_dir: Optional[Path] = None,
    **kwargs,
) -> MonteCarloResult:
    """
    Run Monte Carlo simulation in parallel.

    Divides iterations across worker processes.

    Parameters
    ----------
    landscape_path : Path
        Path to landscape file
    n_iterations : int
        Total number of iterations
    n_workers : int
        Number of parallel workers
    output_dir : Path, optional
        Output directory

    Returns
    -------
    MonteCarloResult
        Combined results
    """
    output_dir = output_dir or OUTPUT_DIR / "monte_carlo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Divide iterations among workers
    iters_per_worker = n_iterations // n_workers
    remainder = n_iterations % n_workers

    worker_iterations = [
        iters_per_worker + (1 if i < remainder else 0)
        for i in range(n_workers)
    ]

    logger.info(
        f"Running parallel Monte Carlo: {n_iterations} iterations "
        f"across {n_workers} workers"
    )

    # Run in parallel
    results = []

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []

        for i, n_iters in enumerate(worker_iterations):
            # Each worker gets different random seed
            seed = kwargs.get("random_seed", 42)
            worker_seed = seed + i if seed else None

            future = executor.submit(
                _run_worker,
                landscape_path=landscape_path,
                n_iterations=n_iters,
                random_seed=worker_seed,
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Worker failed: {e}")

    # Combine results
    if not results:
        raise RuntimeError("All workers failed")

    combined_counts = sum(r.burn_counts for r in results)
    total_iterations = sum(r.n_iterations for r in results)

    # Get calibration factor from kwargs
    calibration_factor = kwargs.get("calibration_factor", 1.0)

    combined = MonteCarloResult(
        burn_counts=combined_counts,
        n_iterations=total_iterations,
        bounds=results[0].bounds,
        resolution=results[0].resolution,
        transform=results[0].transform,
        calibration_factor=calibration_factor,
    )

    # Save combined result
    output_path = output_dir / "burn_probability.tif"
    combined.save(output_path)

    return combined


def _run_worker(
    landscape_path: Path,
    n_iterations: int,
    random_seed: Optional[int] = None,
) -> MonteCarloResult:
    """Worker function for parallel Monte Carlo."""
    config = MonteCarloConfig(
        n_iterations=n_iterations,
        random_seed=random_seed,
    )

    engine = MonteCarloEngine(
        landscape_path=landscape_path,
        config=config,
    )

    return engine.run()


def compute_calibration_factor(
    predicted_prob: np.ndarray,
    fire_perimeters: "gpd.GeoDataFrame",
    transform: object,
    crs: str = "EPSG:3310",
    n_years: int = 1,
) -> float:
    """
    Compute calibration factor from historical fire data.

    The calibration factor adjusts predicted probabilities to match
    observed burn rates. Properly converts cumulative observed burn
    rate to annual rate before comparison.

    Parameters
    ----------
    predicted_prob : ndarray
        Raw predicted burn probabilities (annual, uncalibrated)
    fire_perimeters : GeoDataFrame
        Historical fire perimeters for calibration period
    transform : rasterio transform
        Transform for the prediction raster
    crs : str
        Coordinate reference system
    n_years : int
        Number of years of fire data (for cumulative to annual conversion)

    Returns
    -------
    float
        Calibration factor (multiply predictions by this)
    """
    import rasterio.features

    # Compute observed burn fraction
    height, width = predicted_prob.shape

    # Rasterize fire perimeters
    if len(fire_perimeters) > 0:
        from shapely.ops import unary_union
        burned_area = unary_union(fire_perimeters.geometry.values)

        # Create burned mask
        burned_mask = rasterio.features.rasterize(
            [(burned_area, 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        observed_cumulative = burned_mask.mean()
    else:
        observed_cumulative = 0.0

    # Convert cumulative burn rate to annual burn rate
    # P_annual = 1 - (1 - P_cumulative)^(1/n_years)
    if observed_cumulative > 0 and observed_cumulative < 1:
        observed_annual = 1 - (1 - observed_cumulative) ** (1 / n_years)
    else:
        observed_annual = observed_cumulative / n_years  # Fallback for edge cases

    # Predicted burn rate (already annual)
    predicted_annual = predicted_prob.mean()

    if predicted_annual <= 0:
        logger.warning("Predicted burn rate is zero, using calibration factor 1.0")
        return 1.0

    if observed_annual <= 0:
        logger.warning("No observed fires, using calibration factor 1.0")
        return 1.0

    # Calibration factor = observed_annual / predicted_annual
    calibration_factor = observed_annual / predicted_annual

    logger.info(
        f"Calibration ({n_years} years): "
        f"cumulative={observed_cumulative:.6f}, "
        f"annual_observed={observed_annual:.6f}, "
        f"annual_predicted={predicted_annual:.6f}, "
        f"factor={calibration_factor:.3f}"
    )

    return calibration_factor


def calibrate_predictions(
    result: MonteCarloResult,
    fire_perimeters: "gpd.GeoDataFrame",
    n_years: int = 1,
) -> MonteCarloResult:
    """
    Calibrate Monte Carlo results using historical fire data.

    Parameters
    ----------
    result : MonteCarloResult
        Uncalibrated simulation results
    fire_perimeters : GeoDataFrame
        Historical fire perimeters
    n_years : int
        Number of years of fire data (for cumulative to annual conversion)

    Returns
    -------
    MonteCarloResult
        Calibrated results
    """
    factor = compute_calibration_factor(
        predicted_prob=result.raw_burn_probability,
        fire_perimeters=fire_perimeters,
        transform=result.transform,
        n_years=n_years,
    )

    # Create new result with calibration factor
    return MonteCarloResult(
        burn_counts=result.burn_counts,
        n_iterations=result.n_iterations,
        bounds=result.bounds,
        resolution=result.resolution,
        transform=result.transform,
        calibration_factor=factor,
    )
