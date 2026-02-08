#!/usr/bin/env python3
"""Download data for pilot county study."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import RAW_DATA_DIR, get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# County bounding boxes (WGS84)
COUNTY_BOUNDS = {
    "Sonoma": (-123.5, 38.1, -122.3, 38.9),
    "San Diego": (-117.6, 32.5, -116.1, 33.5),
    "Los Angeles": (-118.9, 33.7, -117.6, 34.8),
    "Butte": (-122.0, 39.3, -121.0, 40.2),
}


def download_pilot_data(
    county: str,
    years: tuple = (2010, 2024),
    output_dir: Path = None,
):
    """
    Download all data for pilot county.

    Parameters
    ----------
    county : str
        County name
    years : tuple
        (start_year, end_year) for data
    output_dir : Path
        Output directory
    """
    output_dir = output_dir or RAW_DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if county not in COUNTY_BOUNDS:
        available = ", ".join(COUNTY_BOUNDS.keys())
        raise ValueError(f"Unknown county: {county}. Available: {available}")

    bounds = COUNTY_BOUNDS[county]
    logger.info(f"Downloading data for {county} County")
    logger.info(f"Bounds (WGS84): {bounds}")
    logger.info(f"Years: {years}")

    # 1. Fire history
    logger.info("=" * 50)
    logger.info("Downloading fire history...")
    from src.data_acquisition.fire_history import (
        download_fire_perimeters,
        download_ignition_points,
    )

    perimeters = download_fire_perimeters(
        output_dir=output_dir / "fire_history",
        years=years,
        bounds=bounds,
        county=county,
    )
    logger.info(f"Downloaded {len(perimeters)} fire perimeters")

    ignitions = download_ignition_points(
        output_dir=output_dir / "fire_history",
        years=years,
        bounds=bounds,
    )
    logger.info(f"Downloaded {len(ignitions)} ignition points")

    # 2. Weather (GridMET)
    logger.info("=" * 50)
    logger.info("Downloading GridMET weather data...")
    from src.data_acquisition.weather import download_gridmet_for_county

    try:
        gridmet = download_gridmet_for_county(
            county_bounds=bounds,
            years=years,
            output_dir=output_dir / "weather" / "gridmet",
        )
        logger.info("Downloaded GridMET data")
    except Exception as e:
        logger.warning(f"GridMET download failed: {e}")
        logger.info("You may need to download GridMET data manually")

    # 3. Terrain (DEM)
    logger.info("=" * 50)
    logger.info("Downloading terrain data...")
    from src.data_acquisition.terrain import download_dem, compute_all_terrain_derivatives

    try:
        dem_path = download_dem(
            bounds=bounds,
            output_dir=output_dir / "terrain",
            resolution=30,
        )
        logger.info(f"Downloaded DEM: {dem_path}")

        # Compute derivatives
        derivatives = compute_all_terrain_derivatives(dem_path)
        logger.info(f"Computed terrain derivatives: {list(derivatives.keys())}")
    except Exception as e:
        logger.warning(f"DEM download failed: {e}")
        logger.info("You may need to download DEM data manually from USGS 3DEP")

    # 4. LANDFIRE
    logger.info("=" * 50)
    logger.info("Checking LANDFIRE data...")
    from src.data_acquisition.landfire import download_landfire_products

    landfire = download_landfire_products(
        products=["FBFM40", "CC", "CH", "CBH", "CBD"],
        bounds=bounds,
        output_dir=output_dir / "landfire",
    )
    if landfire:
        logger.info(f"Found/downloaded LANDFIRE products: {list(landfire.keys())}")
    else:
        logger.info(
            "LANDFIRE data needs manual download.\n"
            "Visit: https://landfire.gov/lf_remap.php\n"
            f"Download for bounds: {bounds}"
        )

    # 5. Infrastructure
    logger.info("=" * 50)
    logger.info("Downloading infrastructure data...")
    from src.data_acquisition.infrastructure import download_roads, download_power_lines

    try:
        roads = download_roads(
            bounds=bounds,
            output_dir=output_dir / "infrastructure",
        )
        logger.info(f"Downloaded {len(roads)} road segments")
    except Exception as e:
        logger.warning(f"Roads download failed: {e}")

    try:
        power_lines = download_power_lines(
            bounds=bounds,
            output_dir=output_dir / "infrastructure",
        )
        logger.info(f"Downloaded {len(power_lines)} power line segments")
    except Exception as e:
        logger.warning(f"Power lines download failed: {e}")

    # 6. Parcels
    logger.info("=" * 50)
    logger.info("Downloading parcel boundaries...")
    from src.data_acquisition.parcels import download_county_parcels

    try:
        parcels = download_county_parcels(
            county=county,
            output_dir=output_dir / "parcels",
            bounds=bounds,
        )
        logger.info(f"Downloaded {len(parcels)} parcels")
    except Exception as e:
        logger.warning(f"Parcel download failed: {e}")
        logger.info("You may need to download parcel data from county GIS portal")

    # Summary
    logger.info("=" * 50)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("=" * 50)

    data_summary = {
        "Fire History": output_dir / "fire_history",
        "Weather": output_dir / "weather",
        "Terrain": output_dir / "terrain",
        "LANDFIRE": output_dir / "landfire",
        "Infrastructure": output_dir / "infrastructure",
        "Parcels": output_dir / "parcels",
    }

    for name, path in data_summary.items():
        if path.exists():
            n_files = len(list(path.glob("**/*")))
            logger.info(f"  {name}: {n_files} files in {path}")
        else:
            logger.info(f"  {name}: NOT DOWNLOADED")

    logger.info("=" * 50)
    logger.info("Data download complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Download data for pilot county study"
    )
    parser.add_argument(
        "--county",
        type=str,
        default="Sonoma",
        choices=list(COUNTY_BOUNDS.keys()),
        help="County to download data for",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="2010-2024",
        help="Year range (e.g., 2010-2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/raw)",
    )

    args = parser.parse_args()

    # Parse year range
    start, end = args.years.split("-")
    years = (int(start), int(end))

    download_pilot_data(
        county=args.county,
        years=years,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
