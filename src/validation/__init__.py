"""Validation module for burn probability model."""

from .hindcast import (
    run_hindcast_validation,
    validate_year,
    compute_calibration_metrics,
)
from .metrics import (
    compute_discrimination_metrics,
    compute_calibration_curve,
    compute_reliability_diagram,
)
from .fire_holdout import (
    split_fires_by_year,
    create_holdout_set,
)

__all__ = [
    "run_hindcast_validation",
    "validate_year",
    "compute_calibration_metrics",
    "compute_discrimination_metrics",
    "compute_calibration_curve",
    "compute_reliability_diagram",
    "split_fires_by_year",
    "create_holdout_set",
]
