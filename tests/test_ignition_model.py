"""Tests for ignition probability model."""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestIgnitionModels:
    """Tests for ignition probability models."""

    def create_test_data(self, n_samples: int = 1000):
        """Create synthetic test data."""
        np.random.seed(42)

        # Features
        X = pd.DataFrame({
            "elevation": np.random.rand(n_samples) * 2000,
            "slope": np.random.rand(n_samples) * 45,
            "aspect": np.random.rand(n_samples) * 360,
            "fuel_model": np.random.randint(1, 14, n_samples),
            "temp_max": np.random.rand(n_samples) * 20 + 25,
            "rh_min": np.random.rand(n_samples) * 40 + 10,
            "wind_speed": np.random.rand(n_samples) * 15,
            "erc": np.random.rand(n_samples) * 100,
        })

        # Labels (correlated with some features)
        prob = (
            0.2 * (X["temp_max"] - 25) / 20 +
            0.3 * (100 - X["rh_min"]) / 100 +
            0.2 * X["wind_speed"] / 15 +
            0.3 * X["erc"] / 100
        )
        prob = np.clip(prob, 0, 1)
        y = (np.random.rand(n_samples) < prob * 0.3).astype(int)

        return X, y

    def test_random_forest_fit_predict(self):
        """Test Random Forest model training and prediction."""
        from src.ignition.models import RandomForestIgnition

        X, y = self.create_test_data()

        model = RandomForestIgnition(n_estimators=10, max_depth=5)
        model.fit(X, y)

        assert model.is_fitted

        # Predict
        proba = model.predict_proba(X)
        assert proba.shape == (len(X),)
        assert proba.min() >= 0
        assert proba.max() <= 1

        # Binary prediction
        pred = model.predict(X)
        assert set(pred).issubset({0, 1})

    def test_random_forest_feature_importance(self):
        """Test feature importance extraction."""
        from src.ignition.models import RandomForestIgnition

        X, y = self.create_test_data()

        model = RandomForestIgnition(n_estimators=10)
        model.fit(X, y)

        importance = model.get_feature_importance()

        assert len(importance) == X.shape[1]
        assert all(v >= 0 for v in importance.values())
        # Sum should be approximately 1
        assert 0.99 < sum(importance.values()) < 1.01

    def test_model_save_load(self):
        """Test model serialization."""
        from src.ignition.models import RandomForestIgnition

        X, y = self.create_test_data()

        model = RandomForestIgnition(n_estimators=10)
        model.fit(X, y)

        original_proba = model.predict_proba(X[:10])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            model.save(path)

            loaded = RandomForestIgnition.load(path)

        loaded_proba = loaded.predict_proba(X[:10])

        np.testing.assert_array_almost_equal(original_proba, loaded_proba)

    def test_handles_missing_values(self):
        """Test model handles NaN values."""
        from src.ignition.models import RandomForestIgnition

        X, y = self.create_test_data(100)

        # Add some NaN values
        X_with_nan = X.copy()
        X_with_nan.iloc[10, 0] = np.nan
        X_with_nan.iloc[20, 3] = np.nan

        model = RandomForestIgnition(n_estimators=10)
        model.fit(X_with_nan, y)

        # Should not raise
        proba = model.predict_proba(X_with_nan)
        assert len(proba) == len(X_with_nan)

    def test_get_model_factory(self):
        """Test model factory function."""
        from src.ignition.models import get_model

        rf = get_model("random_forest", n_estimators=10)
        assert rf.__class__.__name__ == "RandomForestIgnition"

        with pytest.raises(ValueError):
            get_model("unknown_model")


class TestIgnitionTraining:
    """Tests for ignition model training utilities."""

    def test_train_ignition_model(self):
        """Test full training pipeline."""
        from src.ignition.train import train_ignition_model
        import pandas as pd
        import numpy as np

        # Create test data
        np.random.seed(42)
        n = 500

        data = pd.DataFrame({
            "elevation": np.random.rand(n) * 2000,
            "slope": np.random.rand(n) * 45,
            "temp_max": np.random.rand(n) * 20 + 25,
            "rh_min": np.random.rand(n) * 40 + 10,
            "label": np.random.randint(0, 2, n),
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            model, metrics = train_ignition_model(
                training_data=data,
                model_type="random_forest",
                output_dir=Path(tmpdir),
                n_estimators=10,
            )

            assert model is not None
            assert "auc_roc" in metrics
            assert 0 <= metrics["auc_roc"] <= 1

    def test_evaluate_ignition_model(self):
        """Test model evaluation."""
        from src.ignition.train import evaluate_ignition_model
        from src.ignition.models import RandomForestIgnition
        import numpy as np

        # Create and fit simple model
        np.random.seed(42)
        X = np.random.rand(100, 4)
        y = (X[:, 0] + X[:, 1] > 1).astype(int)

        model = RandomForestIgnition(n_estimators=10)
        model.fit(X, y)

        metrics = evaluate_ignition_model(model, X, y)

        assert "auc_roc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "optimal_threshold" in metrics


class TestFeatureEngineering:
    """Tests for feature engineering."""

    def test_temporal_features(self):
        """Test temporal feature computation."""
        from src.ignition.feature_engineering import IgnitionFeatureEngineer
        from datetime import datetime

        engineer = IgnitionFeatureEngineer()

        # Test summer day
        summer_date = datetime(2020, 7, 15)
        features = engineer._compute_temporal_features(summer_date)

        assert features["day_of_year"] == 197
        assert features["month"] == 7
        assert features["season"] == 2  # Summer
        assert features["is_weekend"] == 0.0  # Wednesday

        # Test weekend
        weekend_date = datetime(2020, 7, 18)  # Saturday
        features = engineer._compute_temporal_features(weekend_date)
        assert features["is_weekend"] == 1.0

    def test_derived_features(self):
        """Test derived feature computation."""
        from src.ignition.feature_engineering import IgnitionFeatureEngineer

        engineer = IgnitionFeatureEngineer()

        features = {
            "aspect": 90,  # East
            "temp_max": 35,  # Celsius
            "rh_min": 15,
            "wind_speed": 10,
        }

        derived = engineer._compute_derived_features(features)

        assert "aspect_sin" in derived
        assert "aspect_cos" in derived
        assert "vpd" in derived
        assert "fire_weather_index" in derived

        # Check aspect sin/cos
        assert abs(derived["aspect_sin"] - 1.0) < 0.01  # sin(90) = 1
        assert abs(derived["aspect_cos"]) < 0.01  # cos(90) = 0
