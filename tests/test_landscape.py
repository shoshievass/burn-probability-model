"""Tests for landscape file generation."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestLandscapeBuilder:
    """Tests for LandscapeBuilder class."""

    def test_landscape_builder_init(self):
        """Test LandscapeBuilder initialization."""
        from src.spread.landscape_builder import LandscapeBuilder

        bounds = (-200000, 0, -100000, 100000)
        builder = LandscapeBuilder(bounds=bounds, resolution=270)

        assert builder.ncols == 370  # (100000) / 270
        assert builder.nrows == 370
        assert builder.resolution == 270

    def test_add_layers(self):
        """Test adding layers to landscape."""
        from src.spread.landscape_builder import LandscapeBuilder

        bounds = (0, 0, 27000, 27000)  # 100x100 cells at 270m
        builder = LandscapeBuilder(bounds=bounds, resolution=270)

        # Add required layers
        elevation = np.random.rand(100, 100) * 1000 + 500
        slope = np.random.rand(100, 100) * 45
        aspect = np.random.rand(100, 100) * 360
        fuel = np.random.randint(1, 14, size=(100, 100))

        builder.add_layer("elevation", elevation)
        builder.add_layer("slope", slope)
        builder.add_layer("aspect", aspect)
        builder.add_layer("fuel_model", fuel)

        assert "elevation" in builder.layers
        assert "slope" in builder.layers
        assert "aspect" in builder.layers
        assert "fuel_model" in builder.layers

    def test_layer_shape_validation(self):
        """Test that layers with wrong shape are rejected."""
        from src.spread.landscape_builder import LandscapeBuilder

        bounds = (0, 0, 27000, 27000)
        builder = LandscapeBuilder(bounds=bounds, resolution=270)

        # Wrong shape
        wrong_shape = np.random.rand(50, 50)

        with pytest.raises(ValueError):
            builder.add_layer("elevation", wrong_shape)

    def test_build_requires_layers(self):
        """Test that build fails without required layers."""
        from src.spread.landscape_builder import LandscapeBuilder

        bounds = (0, 0, 27000, 27000)
        builder = LandscapeBuilder(bounds=bounds, resolution=270)

        # Add only some layers
        elevation = np.random.rand(100, 100) * 1000
        builder.add_layer("elevation", elevation)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.lcp"

            with pytest.raises(ValueError) as excinfo:
                builder.build(output_path)

            assert "Missing required layers" in str(excinfo.value)

    def test_build_lcp_file(self):
        """Test building complete LCP file."""
        from src.spread.landscape_builder import LandscapeBuilder

        bounds = (0, 0, 27000, 27000)
        builder = LandscapeBuilder(bounds=bounds, resolution=270)

        # Add all required layers
        nrows, ncols = 100, 100
        builder.add_layer("elevation", np.random.rand(nrows, ncols) * 1000 + 500)
        builder.add_layer("slope", np.random.rand(nrows, ncols) * 45)
        builder.add_layer("aspect", np.random.rand(nrows, ncols) * 360)
        builder.add_layer("fuel_model", np.random.randint(1, 14, size=(nrows, ncols)))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test.lcp"
            result = builder.build(output_path)

            assert result.exists()
            assert result.stat().st_size > 0


class TestTerrainDerivatives:
    """Tests for terrain derivative computation."""

    def test_compute_slope(self):
        """Test slope computation from DEM."""
        from src.data_acquisition.terrain import compute_slope

        with tempfile.TemporaryDirectory() as tmpdir:
            import rasterio

            # Create test DEM with known slope
            dem_path = Path(tmpdir) / "dem.tif"
            nrows, ncols = 100, 100
            resolution = 30

            # Constant slope in x direction
            x = np.arange(ncols) * resolution
            y = np.arange(nrows) * resolution
            X, Y = np.meshgrid(x, y)
            elevation = X * 0.1  # 10% slope = ~5.7 degrees

            transform = rasterio.transform.from_bounds(0, 0, ncols * resolution, nrows * resolution, ncols, nrows)

            with rasterio.open(
                dem_path, "w",
                driver="GTiff",
                height=nrows,
                width=ncols,
                count=1,
                dtype=np.float32,
                crs="EPSG:3310",
                transform=transform,
            ) as dst:
                dst.write(elevation.astype(np.float32), 1)

            # Compute slope
            slope_path = compute_slope(dem_path)

            # Check result
            with rasterio.open(slope_path) as src:
                slope = src.read(1)

            # Should be approximately 5.7 degrees (arctan(0.1))
            expected_slope = np.degrees(np.arctan(0.1))
            # Allow some error at edges due to gradient calculation
            center_slope = slope[10:-10, 10:-10]
            assert np.abs(center_slope.mean() - expected_slope) < 1.0

    def test_compute_aspect(self):
        """Test aspect computation from DEM."""
        from src.data_acquisition.terrain import compute_aspect

        with tempfile.TemporaryDirectory() as tmpdir:
            import rasterio

            dem_path = Path(tmpdir) / "dem.tif"
            nrows, ncols = 100, 100
            resolution = 30

            # Slope facing east (aspect = 90)
            x = np.arange(ncols) * resolution
            y = np.arange(nrows) * resolution
            X, Y = np.meshgrid(x, y)
            elevation = -X * 0.1  # Decreasing to east

            transform = rasterio.transform.from_bounds(0, 0, ncols * resolution, nrows * resolution, ncols, nrows)

            with rasterio.open(
                dem_path, "w",
                driver="GTiff",
                height=nrows,
                width=ncols,
                count=1,
                dtype=np.float32,
                crs="EPSG:3310",
                transform=transform,
            ) as dst:
                dst.write(elevation.astype(np.float32), 1)

            # Compute aspect
            aspect_path = compute_aspect(dem_path)

            with rasterio.open(aspect_path) as src:
                aspect = src.read(1)

            # Should be approximately 90 degrees (east)
            center_aspect = aspect[10:-10, 10:-10]
            mean_aspect = center_aspect.mean()
            assert 80 < mean_aspect < 100 or 260 < mean_aspect < 280


class TestFuelModels:
    """Tests for fuel model handling."""

    def test_fuel_model_conversion(self):
        """Test Scott/Burgan to Anderson 13 conversion."""
        from src.spread.landscape_builder import LandscapeBuilder

        bounds = (0, 0, 2700, 2700)
        builder = LandscapeBuilder(bounds=bounds, resolution=270)

        # Create Scott/Burgan fuel array
        sb40_fuel = np.array([
            [101, 102, 141, 142],
            [161, 181, 182, 201],
            [91, 98, 99, 93],
            [121, 122, 145, 146],
        ])

        # Convert
        anderson = builder._convert_sb40_to_anderson13(sb40_fuel)

        # Check non-burnable codes preserved
        assert anderson[2, 0] == 91
        assert anderson[2, 1] == 98
        assert anderson[2, 2] == 99
        assert anderson[2, 3] == 93

        # Check burnable models converted
        assert 1 <= anderson[0, 0] <= 13 or anderson[0, 0] in [91, 92, 93, 98, 99]
