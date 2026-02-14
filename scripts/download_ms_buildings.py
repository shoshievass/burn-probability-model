#!/usr/bin/env python3
"""
Download Microsoft Building Footprints for a given region.

Microsoft Building Footprints are ML-detected building polygons from aerial imagery.
Data source: https://planetarycomputer.microsoft.com/dataset/ms-buildings

Features:
- Building footprint polygons
- Height estimates for each building
- Confidence scores
"""

import argparse
import gzip
import json
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box, shape

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset links URL
DATASET_LINKS_URL = "https://minedbuildings.z5.web.core.windows.net/global-buildings/dataset-links.csv"


def quadkey_to_bbox(quadkey: str) -> tuple:
    """Convert quadkey to bounding box (lon_min, lat_min, lon_max, lat_max)."""
    import math

    def tile_to_bbox(x, y, z):
        n = 2 ** z
        lon1 = x / n * 360.0 - 180.0
        lon2 = (x + 1) / n * 360.0 - 180.0
        lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        lat2 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        return (lon1, min(lat1, lat2), lon2, max(lat1, lat2))

    x, y = 0, 0
    for c in quadkey:
        x *= 2
        y *= 2
        if c == "1":
            x += 1
        elif c == "2":
            y += 1
        elif c == "3":
            x += 1
            y += 1
    return tile_to_bbox(x, y, len(quadkey))


def download_ms_buildings(
    bbox: tuple,
    output_path: Path,
    country: str = "UnitedStates",
) -> gpd.GeoDataFrame:
    """
    Download Microsoft Building Footprints for a bounding box.

    Parameters
    ----------
    bbox : tuple
        Bounding box as (lon_min, lat_min, lon_max, lat_max) in WGS84
    output_path : Path
        Output path for parquet file
    country : str
        Country name in dataset (default: UnitedStates)

    Returns
    -------
    GeoDataFrame
        Building footprints with height and confidence columns
    """
    logger.info(f"Downloading MS Buildings for bbox: {bbox}")
    logger.info(f"Country: {country}")

    # Get dataset links
    logger.info("Fetching dataset links...")
    response = requests.get(DATASET_LINKS_URL)
    response.raise_for_status()

    lines = response.text.strip().split("\n")
    links = []
    for line in lines[1:]:  # Skip header
        parts = line.split(",")
        if len(parts) >= 3:
            links.append({
                "location": parts[0],
                "quadkey": parts[1],
                "url": parts[2],
            })

    df = pd.DataFrame(links)
    country_links = df[df["location"] == country]
    logger.info(f"Found {len(country_links)} quadkey tiles for {country}")

    # Create bbox polygon
    bbox_poly = box(*bbox)

    # Find quadkeys that intersect bbox
    matching_quadkeys = []
    for _, row in country_links.iterrows():
        try:
            qk_bbox = quadkey_to_bbox(row["quadkey"])
            if box(*qk_bbox).intersects(bbox_poly):
                matching_quadkeys.append(row)
        except Exception:
            pass

    logger.info(f"Quadkeys intersecting bbox: {len(matching_quadkeys)}")

    if not matching_quadkeys:
        logger.warning("No quadkeys found for this bbox")
        return gpd.GeoDataFrame()

    # Download each quadkey
    all_buildings = []
    for i, row in enumerate(matching_quadkeys):
        url = row["url"]
        qk = row["quadkey"]
        logger.info(f"Downloading {i+1}/{len(matching_quadkeys)}: quadkey {qk}")

        try:
            response = requests.get(url, timeout=120)
            if not response.ok:
                logger.warning(f"  Failed: {response.status_code}")
                continue

            # Decompress if gzipped
            if url.endswith(".gz"):
                content = gzip.decompress(response.content).decode("utf-8")
            else:
                content = response.content.decode("utf-8")

            # Parse newline-delimited JSON
            features = []
            for line in content.strip().split("\n"):
                if line:
                    try:
                        features.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

            if not features:
                continue

            # Convert to GeoDataFrame
            geometries = []
            properties = []
            for f in features:
                try:
                    geom = shape(f["geometry"])
                    geometries.append(geom)
                    properties.append(f.get("properties", {}))
                except Exception:
                    pass

            if geometries:
                gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")

                # Filter to bbox
                gdf = gdf[gdf.intersects(bbox_poly)]

                if len(gdf) > 0:
                    all_buildings.append(gdf)
                    logger.info(f"  Got {len(gdf):,} buildings in bbox")

        except Exception as e:
            logger.warning(f"  Error: {e}")

    if not all_buildings:
        logger.warning("No buildings found!")
        return gpd.GeoDataFrame()

    # Combine all
    combined = pd.concat(all_buildings, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs="EPSG:4326")
    logger.info(f"Total buildings: {len(combined):,}")

    # Add ID column
    combined["building_id"] = [f"MS-{i:08d}" for i in range(len(combined))]

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path)
    logger.info(f"Saved to: {output_path}")

    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Download Microsoft Building Footprints"
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        required=True,
        metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"),
        help="Bounding box in WGS84 (lon_min, lat_min, lon_max, lat_max)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output parquet file path",
    )
    parser.add_argument(
        "--country",
        type=str,
        default="UnitedStates",
        help="Country name in dataset (default: UnitedStates)",
    )

    args = parser.parse_args()

    gdf = download_ms_buildings(
        bbox=tuple(args.bbox),
        output_path=args.output,
        country=args.country,
    )

    if len(gdf) > 0:
        print(f"\nDownloaded {len(gdf):,} buildings")
        print(f"Columns: {list(gdf.columns)}")
        if "height" in gdf.columns:
            print(f"Mean height: {gdf['height'].mean():.1f} m")
    else:
        print("No buildings downloaded")
        sys.exit(1)


if __name__ == "__main__":
    main()
