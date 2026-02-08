"""Tests for validation module."""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFireHoldout:
    """Tests for fire holdout splitting."""

    def create_test_fires(self, n_fires: int = 100):
        """Create synthetic fire GeoDataFrame."""
        np.random.seed(42)

        fires = gpd.GeoDataFrame({
            "fire_id": range(n_fires),
            "year_": np.random.randint(2015, 2023, n_fires),
            "gis_acres": np.random.exponential(1000, n_fires),
            "geometry": [Point(np.random.rand() * 100, np.random.rand() * 100)
                         for _ in range(n_fires)],
        }, crs="EPSG:3310")

        return fires

    def test_split_fires_by_year(self):
        """Test within-year fire splitting."""
        from src.validation.fire_holdout import split_fires_by_year

        fires = self.create_test_fires(200)

        train, holdout = split_fires_by_year(fires, holdout_frac=0.30)

        # Check no overlap
        train_ids = set(train["fire_id"])
        holdout_ids = set(holdout["fire_id"])
        assert len(train_ids & holdout_ids) == 0

        # Check all fires accounted for
        assert len(train) + len(holdout) == len(fires)

        # Check approximate holdout fraction
        holdout_frac = len(holdout) / len(fires)
        assert 0.25 < holdout_frac < 0.35

    def test_split_preserves_years(self):
        """Test that holdout has fires from each year."""
        from src.validation.fire_holdout import split_fires_by_year

        fires = self.create_test_fires(200)

        _, holdout = split_fires_by_year(fires, holdout_frac=0.30)

        train_years = set(fires["year_"].unique())
        holdout_years = set(holdout["year_"].unique())

        # Holdout should have fires from most years
        assert len(holdout_years) >= len(train_years) - 1

    def test_stratified_by_size(self):
        """Test stratification by fire size."""
        from src.validation.fire_holdout import split_fires_by_year

        fires = self.create_test_fires(200)

        train, holdout = split_fires_by_year(fires, stratify_by_size=True)

        # Check that large fires exist in both sets
        large_threshold = fires["gis_acres"].quantile(0.9)

        train_large = (train["gis_acres"] > large_threshold).sum()
        holdout_large = (holdout["gis_acres"] > large_threshold).sum()

        assert train_large > 0
        assert holdout_large > 0


class TestValidationMetrics:
    """Tests for validation metrics."""

    def test_discrimination_metrics(self):
        """Test discrimination metric computation."""
        from src.validation.metrics import compute_discrimination_metrics

        np.random.seed(42)

        # Perfect predictions
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        metrics = compute_discrimination_metrics(y_true, y_prob)

        assert metrics["auc_roc"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["precision"] == 1.0

        # Random predictions
        y_prob_random = np.random.rand(8)
        metrics_random = compute_discrimination_metrics(y_true, y_prob_random)

        assert 0 <= metrics_random["auc_roc"] <= 1

    def test_calibration_curve(self):
        """Test calibration curve computation."""
        from src.validation.metrics import compute_calibration_curve

        np.random.seed(42)
        n = 1000

        # Well-calibrated predictions
        y_true = np.random.rand(n) < 0.3
        y_prob = np.where(y_true, 0.7, 0.1) + np.random.rand(n) * 0.2 - 0.1
        y_prob = np.clip(y_prob, 0, 1)

        cal = compute_calibration_curve(y_true.astype(int), y_prob)

        assert "expected_calibration_error" in cal
        assert "prob_true" in cal
        assert "prob_pred" in cal

        # ECE should be reasonable
        assert cal["expected_calibration_error"] < 0.5

    def test_skill_scores(self):
        """Test skill score computation."""
        from src.validation.metrics import compute_skill_scores

        np.random.seed(42)
        n = 1000

        y_true = (np.random.rand(n) < 0.2).astype(int)
        y_prob = y_true * 0.7 + (1 - y_true) * 0.1 + np.random.rand(n) * 0.2 - 0.1
        y_prob = np.clip(y_prob, 0, 1)

        scores = compute_skill_scores(y_true, y_prob)

        assert "brier_score" in scores
        assert "brier_skill_score" in scores
        assert "reliability" in scores
        assert "resolution" in scores

        # BSS should be positive for good model
        assert scores["brier_skill_score"] > 0

    def test_handles_edge_cases(self):
        """Test metrics handle edge cases."""
        from src.validation.metrics import compute_discrimination_metrics

        # All zeros
        y_true = np.zeros(10)
        y_prob = np.random.rand(10)

        metrics = compute_discrimination_metrics(y_true, y_prob)
        assert metrics["auc_roc"] == 0.5  # Default for no positives

        # All ones
        y_true = np.ones(10)
        metrics = compute_discrimination_metrics(y_true, y_prob)
        assert "recall" in metrics

    def test_handles_nan(self):
        """Test metrics handle NaN values."""
        from src.validation.metrics import compute_discrimination_metrics

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, np.nan, 0.8, 0.3, 0.9])

        metrics = compute_discrimination_metrics(y_true, y_prob)

        # Should compute on valid values
        assert 0 <= metrics["auc_roc"] <= 1


class TestHindcastValidation:
    """Tests for hindcast validation."""

    def test_validate_year(self):
        """Test single year validation."""
        from src.validation.hindcast import compute_calibration_metrics

        np.random.seed(42)

        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        metrics = compute_calibration_metrics(y_true, y_prob)

        assert "expected_calibration_error" in metrics
        assert "max_calibration_error" in metrics
        assert "prob_true" in metrics
