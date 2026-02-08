"""Global configuration for burn probability model."""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import os


# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = DATA_DIR / "output"
SRC_DIR = PROJECT_ROOT / "src"


@dataclass
class GridConfig:
    """Grid configuration for raster processing."""
    resolution: int = 270  # meters, matches FSim
    crs: str = "EPSG:3310"  # California Albers (meters)
    nodata: float = -9999.0


@dataclass
class IgnitionModelConfig:
    """Configuration for ignition probability model."""
    model_type: str = "random_forest"  # or "xgboost", "lightgbm"
    n_estimators: int = 500
    max_depth: int = 15
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    n_jobs: int = -1  # Use all cores
    random_state: int = 42

    # Training data
    negative_sample_ratio: int = 4  # 1:4 positive:negative
    training_years: tuple = (2010, 2014)  # Inclusive

    # Validation
    holdout_fraction: float = 0.30  # 30% within-year holdout

    # Feature groups to include
    include_static_features: bool = True
    include_dynamic_features: bool = True


@dataclass
class SpreadModelConfig:
    """Configuration for fire spread model (FlamMap)."""
    flammap_path: str = "/usr/local/bin/flammap"  # Adjust to actual install path
    landscape_resolution: int = 30  # LANDFIRE native resolution
    output_resolution: int = 270  # Upscale for Monte Carlo

    # Simulation parameters
    simulation_duration: int = 480  # minutes (8 hours)
    timestep: int = 30  # minutes
    spot_probability: float = 0.1

    # Crown fire settings
    enable_crown_fire: bool = True
    link_crowning: bool = True


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo integration."""
    n_iterations: int = 10000  # Production
    n_iterations_pilot: int = 1000  # Desktop testing
    n_iterations_quick: int = 100  # Quick validation

    # Parallel processing
    n_cores: int = 28  # Desktop
    chunk_size: int = 50  # Iterations per chunk

    # Tile settings for HPC
    tile_size_km: int = 100
    tile_buffer_km: int = 10  # Overlap to handle edge effects

    # Weather sampling
    sample_santa_ana_separately: bool = True
    santa_ana_fraction: float = 0.15  # 15% of iterations use extreme weather


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    holdout_fraction: float = 0.30  # 30% holdout fires
    random_seed: int = 42

    # Target metrics
    ignition_auc_target: float = 0.85
    ignition_recall_target: float = 0.70
    burn_prob_auc_target: float = 0.80

    # Calibration bins
    n_calibration_bins: int = 10


@dataclass
class PilotConfig:
    """Configuration for desktop pilot study."""
    county: str = "Sonoma"
    years: tuple = (2018, 2022)  # Inclusive
    training_years: tuple = (2010, 2017)  # For ignition model
    n_iterations: int = 1000

    # Bounding box for Sonoma County (approximate, in EPSG:4326)
    bounds_wgs84: tuple = (-123.5, 38.1, -122.3, 38.9)  # (minx, miny, maxx, maxy)


@dataclass
class HPCConfig:
    """Configuration for HPC cluster runs."""
    scheduler: str = "slurm"  # or "pbs"
    partition: str = "normal"
    nodes_per_job: int = 1
    cores_per_node: int = 32
    memory_gb: int = 128
    walltime_hours: int = 4

    # Job array settings
    n_tiles: int = 200  # Approximate for California
    max_concurrent_jobs: int = 50


@dataclass
class Config:
    """Main configuration container."""
    grid: GridConfig = field(default_factory=GridConfig)
    ignition: IgnitionModelConfig = field(default_factory=IgnitionModelConfig)
    spread: SpreadModelConfig = field(default_factory=SpreadModelConfig)
    monte_carlo: MonteCarloConfig = field(default_factory=MonteCarloConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    hpc: HPCConfig = field(default_factory=HPCConfig)

    # Years for analysis
    analysis_years: tuple = (2015, 2025)

    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()


def get_config() -> Config:
    """Get the global configuration."""
    return config


def load_pilot_config(county: str = "Sonoma") -> Config:
    """Load configuration for desktop pilot study."""
    cfg = Config()
    cfg.pilot.county = county
    cfg.monte_carlo.n_iterations = cfg.monte_carlo.n_iterations_pilot
    return cfg


def load_production_config() -> Config:
    """Load configuration for full production run."""
    cfg = Config()
    cfg.monte_carlo.n_iterations = cfg.monte_carlo.n_iterations
    return cfg
