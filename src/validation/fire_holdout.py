"""Fire holdout splitting for validation."""

import logging
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import geopandas as gpd

from config.settings import get_config

logger = logging.getLogger(__name__)


def split_fires_by_year(
    fires_gdf: gpd.GeoDataFrame,
    holdout_frac: float = 0.30,
    seed: int = 42,
    stratify_by_size: bool = True,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Hold out fraction of fires WITHIN each year.

    This ensures model is tested on fires from the same climate/weather
    regime while avoiding temporal leakage.

    Parameters
    ----------
    fires_gdf : GeoDataFrame
        Fire perimeters with year column
    holdout_frac : float
        Fraction to hold out (default 0.30)
    seed : int
        Random seed for reproducibility
    stratify_by_size : bool
        Stratify by fire size to ensure large fires in both sets

    Returns
    -------
    tuple of GeoDataFrame
        (training_fires, holdout_fires)
    """
    # Identify year column
    year_col = None
    for col in ["year_", "year", "YEAR_", "fire_year"]:
        if col in fires_gdf.columns:
            year_col = col
            break

    if year_col is None:
        raise ValueError("No year column found in fires GeoDataFrame")

    train_fires = []
    test_fires = []

    rng = np.random.default_rng(seed)

    for year in fires_gdf[year_col].unique():
        year_fires = fires_gdf[fires_gdf[year_col] == year].copy()

        if len(year_fires) == 0:
            continue

        n_holdout = max(1, int(len(year_fires) * holdout_frac))

        if stratify_by_size and "gis_acres" in year_fires.columns:
            # Stratified sampling by fire size
            test_idx = _stratified_sample(
                year_fires, n_holdout, "gis_acres", rng
            )
        else:
            # Random sampling
            test_idx = rng.choice(
                year_fires.index,
                size=min(n_holdout, len(year_fires)),
                replace=False,
            )

        year_train = year_fires.drop(test_idx)
        year_test = year_fires.loc[test_idx]

        train_fires.append(year_train)
        test_fires.append(year_test)

    train_gdf = pd.concat(train_fires, ignore_index=True)
    test_gdf = pd.concat(test_fires, ignore_index=True)

    # Preserve GeoDataFrame type
    train_gdf = gpd.GeoDataFrame(train_gdf, crs=fires_gdf.crs)
    test_gdf = gpd.GeoDataFrame(test_gdf, crs=fires_gdf.crs)

    logger.info(
        f"Split fires: {len(train_gdf)} training, {len(test_gdf)} holdout "
        f"({len(test_gdf) / len(fires_gdf) * 100:.1f}%)"
    )

    return train_gdf, test_gdf


def _stratified_sample(
    gdf: gpd.GeoDataFrame,
    n_samples: int,
    size_col: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stratified sampling by fire size."""
    # Define size classes
    sizes = gdf[size_col].values
    thresholds = [100, 1000, 10000]  # Small, medium, large, mega

    size_class = np.zeros(len(sizes), dtype=int)
    for i, threshold in enumerate(thresholds):
        size_class[sizes > threshold] = i + 1

    # Sample from each class
    selected = []
    classes = np.unique(size_class)

    # Allocate samples proportionally
    class_counts = [np.sum(size_class == c) for c in classes]
    total = sum(class_counts)

    for c, count in zip(classes, class_counts):
        n_class = max(1, int(n_samples * count / total))
        class_indices = gdf.index[size_class == c]

        if len(class_indices) > 0:
            sample = rng.choice(
                class_indices,
                size=min(n_class, len(class_indices)),
                replace=False,
            )
            selected.extend(sample)

    return np.array(selected)


def create_holdout_set(
    fires_gdf: gpd.GeoDataFrame,
    holdout_years: Tuple[int, int],
    holdout_frac: float = 0.30,
    seed: int = 42,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Create holdout set for specific years.

    Parameters
    ----------
    fires_gdf : GeoDataFrame
        All fire perimeters
    holdout_years : tuple
        (start_year, end_year) for holdout period
    holdout_frac : float
        Fraction to hold out within each year
    seed : int
        Random seed

    Returns
    -------
    tuple
        (training_fires, holdout_fires)
    """
    # Identify year column
    year_col = None
    for col in ["year_", "year", "YEAR_"]:
        if col in fires_gdf.columns:
            year_col = col
            break

    if year_col is None:
        raise ValueError("No year column found")

    # Filter to holdout years
    holdout_mask = (
        (fires_gdf[year_col] >= holdout_years[0]) &
        (fires_gdf[year_col] <= holdout_years[1])
    )

    holdout_period_fires = fires_gdf[holdout_mask]
    training_period_fires = fires_gdf[~holdout_mask]

    # Split within holdout period
    if len(holdout_period_fires) > 0:
        _, holdout_fires = split_fires_by_year(
            holdout_period_fires,
            holdout_frac=holdout_frac,
            seed=seed,
        )
    else:
        holdout_fires = gpd.GeoDataFrame(columns=fires_gdf.columns, crs=fires_gdf.crs)

    return training_period_fires, holdout_fires


def compute_holdout_statistics(
    train_fires: gpd.GeoDataFrame,
    holdout_fires: gpd.GeoDataFrame,
) -> dict:
    """
    Compute statistics about the holdout split.

    Parameters
    ----------
    train_fires : GeoDataFrame
        Training fires
    holdout_fires : GeoDataFrame
        Holdout fires

    Returns
    -------
    dict
        Split statistics
    """
    stats = {
        "n_train": len(train_fires),
        "n_holdout": len(holdout_fires),
        "holdout_fraction": len(holdout_fires) / (len(train_fires) + len(holdout_fires)),
    }

    # Size statistics
    if "gis_acres" in train_fires.columns:
        stats["train_mean_size"] = train_fires["gis_acres"].mean()
        stats["holdout_mean_size"] = holdout_fires["gis_acres"].mean()
        stats["train_total_acres"] = train_fires["gis_acres"].sum()
        stats["holdout_total_acres"] = holdout_fires["gis_acres"].sum()

    # Year distribution
    year_col = None
    for col in ["year_", "year", "YEAR_"]:
        if col in train_fires.columns:
            year_col = col
            break

    if year_col:
        stats["train_years"] = sorted(train_fires[year_col].unique().tolist())
        stats["holdout_years"] = sorted(holdout_fires[year_col].unique().tolist())

    return stats


def validate_holdout_balance(
    train_fires: gpd.GeoDataFrame,
    holdout_fires: gpd.GeoDataFrame,
) -> bool:
    """
    Check that holdout set is balanced with training set.

    Parameters
    ----------
    train_fires : GeoDataFrame
        Training fires
    holdout_fires : GeoDataFrame
        Holdout fires

    Returns
    -------
    bool
        True if sets appear balanced
    """
    issues = []

    # Check size distribution
    if "gis_acres" in train_fires.columns and "gis_acres" in holdout_fires.columns:
        train_median = train_fires["gis_acres"].median()
        holdout_median = holdout_fires["gis_acres"].median()

        if abs(train_median - holdout_median) / train_median > 0.5:
            issues.append(
                f"Size distribution imbalance: train median={train_median:.1f}, "
                f"holdout median={holdout_median:.1f}"
            )

    # Check year coverage
    year_col = None
    for col in ["year_", "year", "YEAR_"]:
        if col in train_fires.columns:
            year_col = col
            break

    if year_col:
        train_years = set(train_fires[year_col].unique())
        holdout_years = set(holdout_fires[year_col].unique())

        missing_years = holdout_years - train_years
        if missing_years:
            issues.append(f"Holdout contains years not in training: {missing_years}")

    if issues:
        for issue in issues:
            logger.warning(f"Holdout balance issue: {issue}")
        return False

    return True
