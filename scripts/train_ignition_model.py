#!/usr/bin/env python3
"""Train ignition probability model."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    get_config,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_ignition_model(
    county: str = "Sonoma",
    training_years: tuple = (2010, 2017),
    validation_years: tuple = (2018, 2022),
    model_type: str = "random_forest",
    n_estimators: int = 500,
    holdout_fraction: float = 0.30,
    output_dir: Path = None,
):
    """
    Train ignition probability model.

    Parameters
    ----------
    county : str
        County for training data
    training_years : tuple
        Years for model training
    validation_years : tuple
        Years for validation
    model_type : str
        Model type (random_forest, xgboost, lightgbm)
    n_estimators : int
        Number of trees/estimators
    holdout_fraction : float
        Fraction of fires to hold out per year
    output_dir : Path
        Output directory
    """
    output_dir = output_dir or OUTPUT_DIR / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training ignition model for {county} County")
    logger.info(f"Training years: {training_years}")
    logger.info(f"Validation years: {validation_years}")
    logger.info(f"Model type: {model_type}")

    # Load fire history
    logger.info("Loading fire history...")
    import geopandas as gpd

    fire_perimeters_path = RAW_DATA_DIR / "fire_history" / "fire_perimeters.parquet"
    ignition_points_path = RAW_DATA_DIR / "fire_history" / "ignition_points.parquet"

    if not fire_perimeters_path.exists():
        raise FileNotFoundError(
            f"Fire perimeters not found: {fire_perimeters_path}\n"
            "Run download_pilot_data.py first"
        )

    fire_perimeters = gpd.read_parquet(fire_perimeters_path)
    logger.info(f"Loaded {len(fire_perimeters)} fire perimeters")

    if ignition_points_path.exists():
        ignition_points = gpd.read_parquet(ignition_points_path)
        logger.info(f"Loaded {len(ignition_points)} ignition points")
    else:
        # Use fire perimeter centroids
        logger.info("Using fire perimeter centroids as ignition points")
        ignition_points = fire_perimeters.copy()
        ignition_points["geometry"] = ignition_points.geometry.centroid

    # Split fires for validation
    logger.info("Splitting fires into train/holdout...")
    from src.validation.fire_holdout import split_fires_by_year

    train_fires, holdout_fires = split_fires_by_year(
        fire_perimeters,
        holdout_frac=holdout_fraction,
    )
    logger.info(f"Training fires: {len(train_fires)}, Holdout fires: {len(holdout_fires)}")

    # Load GridMET data
    logger.info("Loading weather data...")
    import xarray as xr

    gridmet_dir = RAW_DATA_DIR / "weather" / "gridmet"
    gridmet_files = list(gridmet_dir.glob("*.nc"))

    if gridmet_files:
        # Open without dask chunking
        gridmet_ds = xr.open_mfdataset(gridmet_files, chunks=None)
        logger.info("Loaded GridMET data")
    else:
        logger.warning("No GridMET data found - using simplified features")
        gridmet_ds = None

    # Load static features
    logger.info("Loading static features...")
    static_features = {}

    terrain_dir = RAW_DATA_DIR / "terrain"
    for name in ["dem", "slope", "aspect", "tpi"]:
        path = terrain_dir / f"{name}.tif"
        if path.exists():
            static_features[name] = path

    landfire_dir = RAW_DATA_DIR / "landfire"
    for pattern in ["*FBFM40*.tif", "*CC*.tif", "*CH*.tif"]:
        for path in landfire_dir.glob(pattern):
            name = path.stem.split("_")[0].lower()
            static_features[name] = path

    logger.info(f"Found static features: {list(static_features.keys())}")

    # Create training dataset
    logger.info("Creating training dataset...")
    from src.preprocessing.feature_creation import create_training_dataset

    training_data_path = PROCESSED_DATA_DIR / "training_data.parquet"

    if training_data_path.exists():
        logger.info(f"Loading existing training data from {training_data_path}")
        import pandas as pd
        training_data = pd.read_parquet(training_data_path)
    else:
        training_data = create_training_dataset(
            ignition_points=ignition_points,
            fire_perimeters=train_fires,
            static_features=static_features,
            gridmet_ds=gridmet_ds,
            years=training_years,
            negative_ratio=4,
            output_path=training_data_path,
        )

    logger.info(f"Training data: {len(training_data)} samples")
    logger.info(f"Positive rate: {training_data['label'].mean():.3f}")

    # Train model
    logger.info(f"Training {model_type} model...")
    from src.ignition.train import train_ignition_model as train_model

    model, metrics = train_model(
        training_data=training_data,
        model_type=model_type,
        output_dir=output_dir,
        n_estimators=n_estimators,
    )

    # Print results
    logger.info("=" * 50)
    logger.info("TRAINING RESULTS")
    logger.info("=" * 50)
    logger.info(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    logger.info(f"AUC-PR: {metrics['auc_pr']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"F1: {metrics['f1']:.4f}")

    # Check target
    config = get_config()
    auc_target = config.validation.ignition_auc_target
    recall_target = config.validation.ignition_recall_target

    if metrics["auc_roc"] >= auc_target:
        logger.info(f"AUC target ({auc_target}) ACHIEVED")
    else:
        logger.warning(f"AUC target ({auc_target}) NOT achieved")

    if metrics["recall"] >= recall_target:
        logger.info(f"Recall target ({recall_target}) ACHIEVED")
    else:
        logger.warning(f"Recall target ({recall_target}) NOT achieved")

    # Feature importance
    logger.info("=" * 50)
    logger.info("TOP 10 FEATURES")
    logger.info("=" * 50)
    importance = model.get_feature_importance()
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for name, value in sorted_importance[:10]:
        logger.info(f"  {name}: {value:.4f}")

    # Save plots
    from src.ignition.train import (
        plot_roc_curve,
        plot_precision_recall_curve,
        plot_feature_importance,
    )

    # Get feature columns (numeric only)
    feature_cols = [c for c in training_data.columns
                   if c not in ["label", "date", "geometry"]
                   and training_data[c].dtype in ["float64", "float32", "int64", "int32", "int16"]]

    y_test = training_data["label"].values
    y_prob = model.predict_proba(training_data[feature_cols].fillna(0))

    plot_roc_curve(y_test, y_prob, output_dir / "roc_curve.png")
    plot_precision_recall_curve(y_test, y_prob, output_dir / "pr_curve.png")
    plot_feature_importance(model, top_n=20, output_path=output_dir / "feature_importance.png")

    logger.info("=" * 50)
    logger.info(f"Model saved to: {output_dir}")
    logger.info("Training complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Train ignition probability model"
    )
    parser.add_argument(
        "--county",
        type=str,
        default="Sonoma",
        help="County for training",
    )
    parser.add_argument(
        "--training-years",
        type=str,
        default="2010-2017",
        help="Training year range",
    )
    parser.add_argument(
        "--validation-years",
        type=str,
        default="2018-2022",
        help="Validation year range",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost", "lightgbm"],
        help="Model type",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=500,
        help="Number of trees",
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=0.30,
        help="Fraction of fires to hold out",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )

    args = parser.parse_args()

    # Parse year ranges
    train_start, train_end = args.training_years.split("-")
    val_start, val_end = args.validation_years.split("-")

    train_ignition_model(
        county=args.county,
        training_years=(int(train_start), int(train_end)),
        validation_years=(int(val_start), int(val_end)),
        model_type=args.model_type,
        n_estimators=args.n_estimators,
        holdout_fraction=args.holdout_fraction,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
