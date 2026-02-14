#!/usr/bin/env python3
"""
Download GridMET historical weather data for fire simulations.

GridMET provides 4km daily weather data including:
- Temperature (min/max)
- Relative humidity (min/max)
- Wind speed and direction
- Precipitation
- Energy Release Component (ERC)
- Fuel moisture (100-hour)

Data source: https://www.climatologylab.org/gridmet.html
THREDDS: http://thredds.northwestknowledge.net/thredds/catalog.html
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_acquisition.weather import download_gridmet, download_gridmet_for_county

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# California bounding box (WGS84)
CALIFORNIA_BOUNDS = (-124.5, 32.5, -114.0, 42.1)

# Sonoma County bounding box
SONOMA_BOUNDS = (-123.6, 38.1, -122.3, 38.9)


def main():
    parser = argparse.ArgumentParser(
        description="Download GridMET historical weather data for fire simulations"
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["california", "sonoma", "custom"],
        default="california",
        help="Region to download (default: california)",
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Custom bounding box (required if region=custom)",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2010,
        help="Start year (default: 2010)",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2024,
        help="End year (default: 2024)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/raw/weather/gridmet)",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=["tmmx", "tmmn", "rmin", "rmax", "vs", "th", "erc", "fm100", "pr"],
        help="Variables to download",
    )

    args = parser.parse_args()

    # Determine bounds
    if args.region == "california":
        bounds = CALIFORNIA_BOUNDS
        logger.info("Downloading GridMET for California")
    elif args.region == "sonoma":
        bounds = SONOMA_BOUNDS
        logger.info("Downloading GridMET for Sonoma County")
    elif args.region == "custom":
        if args.bbox is None:
            parser.error("--bbox is required when region=custom")
        bounds = tuple(args.bbox)
        logger.info(f"Downloading GridMET for custom bounds: {bounds}")

    logger.info(f"Years: {args.start_year} - {args.end_year}")
    logger.info(f"Variables: {args.variables}")

    # Download
    ds = download_gridmet(
        variables=args.variables,
        years=(args.start_year, args.end_year),
        bounds=bounds,
        output_dir=args.output_dir,
    )

    # Summary
    logger.info("=" * 60)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Variables: {list(ds.data_vars)}")
    logger.info(f"Time range: {ds['day'].values[0]} to {ds['day'].values[-1]}")
    logger.info(f"Spatial extent: lat [{float(ds['lat'].min()):.2f}, {float(ds['lat'].max()):.2f}]")
    logger.info(f"               lon [{float(ds['lon'].min()):.2f}, {float(ds['lon'].max()):.2f}]")

    # Print usage hint
    print("\n" + "=" * 60)
    print("USAGE")
    print("=" * 60)
    print("To use empirical weather sampling in simulations, pass the GridMET file:")
    print()
    print("  python scripts/run_tile_simulation.py \\")
    print("      --tile-id 001 \\")
    print("      --gridmet data/raw/weather/gridmet/gridmet_2010_2024.nc")
    print()
    print("Or for cluster runs, ensure GridMET is downloaded before submitting jobs.")


if __name__ == "__main__":
    main()
