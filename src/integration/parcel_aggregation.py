"""Aggregate burn probability raster to parcel polygons."""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from config.settings import get_config, OUTPUT_DIR

logger = logging.getLogger(__name__)


def aggregate_to_parcels(
    burn_probability_path: Path,
    parcels_gdf: gpd.GeoDataFrame,
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Aggregate burn probability raster to parcel polygons.

    Uses zonal statistics to compute burn probability per parcel.

    Parameters
    ----------
    burn_probability_path : Path
        Path to burn probability raster
    parcels_gdf : GeoDataFrame
        Parcel polygons with APN
    output_path : Path, optional
        Path to save results

    Returns
    -------
    GeoDataFrame
        Parcels with burn probability statistics
    """
    from rasterstats import zonal_stats

    logger.info(f"Aggregating burn probability to {len(parcels_gdf)} parcels")

    # Ensure parcels are in raster CRS
    with rasterio.open(burn_probability_path) as src:
        raster_crs = src.crs

    if parcels_gdf.crs != raster_crs:
        parcels_gdf = parcels_gdf.to_crs(raster_crs)

    # Compute zonal statistics
    stats = zonal_stats(
        parcels_gdf,
        burn_probability_path,
        stats=["mean", "max", "min", "std", "median", "count"],
        all_touched=True,  # Include partially covered cells
        nodata=-9999,
    )

    # Add statistics to parcels
    stats_df = pd.DataFrame(stats)
    stats_df.columns = [
        "burn_prob_mean",
        "burn_prob_max",
        "burn_prob_min",
        "burn_prob_std",
        "burn_prob_median",
        "burn_prob_count",
    ]

    result = parcels_gdf.copy()
    for col in stats_df.columns:
        result[col] = stats_df[col]

    # Handle missing values (parcels outside raster extent)
    result["burn_prob_mean"] = result["burn_prob_mean"].fillna(0)
    result["burn_prob_max"] = result["burn_prob_max"].fillna(0)

    logger.info(f"Computed burn probability for {(result['burn_prob_count'] > 0).sum()} parcels")

    if output_path:
        result.to_parquet(output_path)
        logger.info(f"Saved results to {output_path}")

    return result


def compute_parcel_statistics(
    parcels_gdf: gpd.GeoDataFrame,
    raster_paths: Dict[str, Path],
    output_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """
    Compute multiple raster statistics per parcel.

    Parameters
    ----------
    parcels_gdf : GeoDataFrame
        Parcel polygons
    raster_paths : dict
        Mapping of stat name to raster path
    output_path : Path, optional
        Output path

    Returns
    -------
    GeoDataFrame
        Parcels with all statistics
    """
    from rasterstats import zonal_stats

    result = parcels_gdf.copy()

    for name, raster_path in raster_paths.items():
        logger.info(f"Computing {name} statistics")

        # Ensure CRS match
        with rasterio.open(raster_path) as src:
            if result.crs != src.crs:
                result_projected = result.to_crs(src.crs)
            else:
                result_projected = result

        stats = zonal_stats(
            result_projected,
            raster_path,
            stats=["mean", "max"],
            all_touched=True,
            nodata=-9999,
        )

        result[f"{name}_mean"] = [s["mean"] for s in stats]
        result[f"{name}_max"] = [s["max"] for s in stats]

    if output_path:
        result.to_parquet(output_path)

    return result


def add_fire_hazard_zones(
    parcels_gdf: gpd.GeoDataFrame,
    fhsz_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Add Fire Hazard Severity Zone to parcels.

    Parameters
    ----------
    parcels_gdf : GeoDataFrame
        Parcel polygons
    fhsz_gdf : GeoDataFrame
        Fire Hazard Severity Zone polygons

    Returns
    -------
    GeoDataFrame
        Parcels with FHSZ column
    """
    # Spatial join
    result = gpd.sjoin(
        parcels_gdf,
        fhsz_gdf[["geometry", "HAZ_CLASS"]],
        how="left",
        predicate="intersects",
    )

    # Rename and clean
    result = result.rename(columns={"HAZ_CLASS": "fhsz"})
    result = result.drop(columns=["index_right"], errors="ignore")

    # Handle parcels in multiple zones (take highest hazard)
    hazard_order = {"Very High": 3, "High": 2, "Moderate": 1}
    result["fhsz_rank"] = result["fhsz"].map(hazard_order).fillna(0)

    # Keep row with highest hazard for each parcel
    result = result.sort_values("fhsz_rank", ascending=False)
    result = result.drop_duplicates(subset=["apn"], keep="first")
    result = result.drop(columns=["fhsz_rank"])

    return result


def add_validation_data(
    parcels_gdf: gpd.GeoDataFrame,
    fire_perimeters_gdf: gpd.GeoDataFrame,
    year: int,
) -> gpd.GeoDataFrame:
    """
    Add actual burn status for validation.

    Parameters
    ----------
    parcels_gdf : GeoDataFrame
        Parcel polygons with burn probability
    fire_perimeters_gdf : GeoDataFrame
        Fire perimeters with year
    year : int
        Year to check

    Returns
    -------
    GeoDataFrame
        Parcels with actual_burned column
    """
    # Filter fires to year
    year_fires = fire_perimeters_gdf[
        fire_perimeters_gdf["year_"] == year
    ] if "year_" in fire_perimeters_gdf.columns else fire_perimeters_gdf

    if year_fires.empty:
        parcels_gdf["actually_burned"] = False
        return parcels_gdf

    # Ensure CRS match
    if parcels_gdf.crs != year_fires.crs:
        year_fires = year_fires.to_crs(parcels_gdf.crs)

    # Spatial join to find burned parcels
    burned_parcels = gpd.sjoin(
        parcels_gdf[["apn", "geometry"]],
        year_fires[["geometry"]],
        how="inner",
        predicate="intersects",
    )

    # Add burned flag
    parcels_gdf["actually_burned"] = parcels_gdf["apn"].isin(burned_parcels["apn"])

    n_burned = parcels_gdf["actually_burned"].sum()
    logger.info(f"Year {year}: {n_burned} parcels actually burned")

    return parcels_gdf


def create_output_table(
    parcels_gdf: gpd.GeoDataFrame,
    year: int,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create final output table in schema format.

    Parameters
    ----------
    parcels_gdf : GeoDataFrame
        Parcels with all computed fields
    year : int
        Prediction year
    output_path : Path, optional
        Output path

    Returns
    -------
    DataFrame
        Output table (without geometry for CSV)
    """
    # Select and rename columns
    output_cols = {
        "apn": "apn",
        "burn_prob_mean": "burn_prob_mean",
        "burn_prob_max": "burn_prob_max",
        "burn_prob_std": "burn_prob_std",
        "fhsz": "fhsz",
        "actually_burned": "actually_burned",
    }

    # Add optional columns if present
    optional = ["county", "address", "acres", "dominant_fuel", "wui_class"]
    for col in optional:
        if col in parcels_gdf.columns:
            output_cols[col] = col

    available_cols = {k: v for k, v in output_cols.items() if k in parcels_gdf.columns}

    output = parcels_gdf[list(available_cols.keys())].rename(columns=available_cols)
    output["year"] = year

    # Reorder columns
    col_order = ["apn", "year", "burn_prob_mean", "burn_prob_max", "burn_prob_std"]
    col_order += [c for c in output.columns if c not in col_order]
    output = output[col_order]

    if output_path:
        if output_path.suffix == ".csv":
            output.to_csv(output_path, index=False)
        else:
            output.to_parquet(output_path)
        logger.info(f"Saved output table to {output_path}")

    return output


def compute_percentile_burn_probability(
    burn_counts: np.ndarray,
    n_iterations: int,
    percentiles: List[float] = [5, 50, 95],
) -> Dict[str, np.ndarray]:
    """
    Compute percentile-based burn probability estimates.

    Uses beta distribution to estimate uncertainty.

    Parameters
    ----------
    burn_counts : ndarray
        Number of times each cell burned
    n_iterations : int
        Total number of iterations
    percentiles : list
        Percentiles to compute

    Returns
    -------
    dict
        Mapping of percentile name to array
    """
    from scipy import stats

    # Use beta distribution for binomial proportion
    # Jeffreys prior: Beta(0.5, 0.5)
    alpha = burn_counts + 0.5
    beta = (n_iterations - burn_counts) + 0.5

    results = {}

    for p in percentiles:
        results[f"p{p:02.0f}"] = stats.beta.ppf(p / 100, alpha, beta)

    return results
