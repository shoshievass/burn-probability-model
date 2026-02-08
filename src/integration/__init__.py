"""Monte Carlo integration module for burn probability estimation."""

from .monte_carlo import (
    MonteCarloEngine,
    run_monte_carlo_simulation,
    run_parallel_monte_carlo,
)
from .weather_scenarios import (
    WeatherScenarioGenerator,
    generate_fire_season_scenarios,
)
from .parcel_aggregation import (
    aggregate_to_parcels,
    compute_parcel_statistics,
)

__all__ = [
    "MonteCarloEngine",
    "run_monte_carlo_simulation",
    "run_parallel_monte_carlo",
    "WeatherScenarioGenerator",
    "generate_fire_season_scenarios",
    "aggregate_to_parcels",
    "compute_parcel_statistics",
]
