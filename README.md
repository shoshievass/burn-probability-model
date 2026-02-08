# Ex Ante Burn Probability Model for California Parcels

A Monte Carlo simulation framework that estimates the probability each parcel in California will burn in a given year, computed **before** knowing actual fire outcomes. This enables actuarial applications like wildfire insurance pricing and risk assessment.

## Overview

The model combines three core components:

1. **Ignition Probability Model** - Predicts where fires are likely to start based on terrain, fuel, and weather conditions
2. **Fire Spread Model** - Simulates how fires propagate using physics-based equations (Rothermel spread model with elliptical fire shapes)
3. **Monte Carlo Integration** - Aggregates thousands of simulated fire scenarios to produce annual burn probability

```
+-----------------------------------------------------------------------------+
|                           Monte Carlo Engine                                 |
|  +------------------+   +------------------+   +--------------------------+  |
|  | Weather          |   | Ignition         |   | Fire Spread              |  |
|  | Scenarios        |-->| Sampling         |-->| Simulation               |  |
|  | (wind, temp,     |   | (where fires     |   | (Rothermel + elliptical  |  |
|  | humidity, ERC)   |   | start)           |   | spread shape)            |  |
|  +------------------+   +------------------+   +--------------------------+  |
|           |                     |                          |                 |
|           +---------------------+--------------------------+                 |
|                                 |                                            |
|                    +------------v------------+                               |
|                    | Burn Count Aggregation  |                               |
|                    | P(burn) = burns / N     |                               |
|                    +-------------------------+                               |
+-----------------------------------------------------------------------------+
```

## How It Works

### 1. Ignition Probability Model

**Purpose:** Predict where fires are likely to start given landscape and weather conditions.

**Approach:** Random Forest classifier trained on historical ignition points (positive samples) matched against non-ignition locations on the same days (negative samples, 1:4 ratio).

**Features:**

| Category | Features |
|----------|----------|
| Terrain | Elevation (DEM), Slope, Aspect, Topographic Position Index (TPI) |
| Vegetation | Fuel Model (Scott/Burgan 40), Canopy Cover, Canopy Height |
| Weather | Temperature, Humidity, Wind Speed, Energy Release Component (ERC) |
| Temporal | Day of Year, Day of Week, Season |

**Performance (Sonoma County 2010-2022):**
- AUC-ROC: 0.96 (target: >0.85)
- Recall: 0.98 (target: >0.70)

**Key Insight:** Fuel model (FBFM40) is the most important feature (28.6% importance), followed by aspect and slope.

**Implementation:** `src/ignition/models.py`

```python
class RandomForestIgnition:
    """Random Forest model for ignition probability."""

    def fit(self, X, y):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=10,
            oob_score=True,
            n_jobs=-1
        )
        self.model.fit(X, y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]
```

### 2. Fire Spread Model

**Purpose:** Simulate how a fire spreads from an ignition point given wind, terrain, and fuel conditions.

**Approach:** Cellular automaton implementing the Rothermel spread equations with:
- **Elliptical fire shape** based on wind speed (Anderson 1983)
- **Wind direction effects** - fires spread faster downwind
- **Slope effects** - fires spread faster uphill (5x base rate for steep slopes)
- **Fuel-specific spread rates** - Scott/Burgan 40 fuel model parameters

**Key Equations:**

```
Length-to-Breadth Ratio:
  LB = 1.0 + 0.25 * wind_speed  (capped at 8.0)

Eccentricity:
  e = sqrt(1 - 1/LB^2)

Wind Factor (directional):
  wind_factor = (1 + e * cos(theta)) / (1 - e^2)
  where theta = angle between spread direction and wind direction

Slope Factor:
  slope_factor = 1.0 + 5.0 * tan^2(slope) * slope_alignment
  slope_alignment = alignment with uphill direction (-1 to +1)

Effective Rate of Spread:
  ROS_effective = ROS_base * wind_factor * wind_speed_factor * slope_factor
```

**Fuel Model Spread Rates (Scott/Burgan 40):**

| Code | Fuel Type | Base ROS (m/min) |
|------|-----------|------------------|
| 101-104 | Short grass | 8-15 |
| 121-124 | Low shrub | 4-10 |
| 141-149 | Chaparral | 6-15 |
| 161-165 | Timber understory | 3-6 |
| 181-189 | Timber litter | 1-4 |
| 201-204 | Slash/blowdown | 2-8 |

**Resolution:** 270m grid cells (matches FSim, computationally tractable)

**Duration:** 480 minutes (8 hours) per fire, 96 timesteps of 5 minutes each

**Implementation:** `src/spread/flammap_wrapper.py`

```python
def run_basic_fire_spread(
    landscape_path: Path,
    ignition_point: tuple,
    duration_minutes: int = 480,
    wind_speed: float = 5.0,
    wind_direction: float = 270.0,
) -> dict:
    """
    Run fire spread simulation using cellular automaton.

    Returns:
        dict with 'burned_area' (2D bool array), 'arrival_time', 'transform'
    """
```

### 3. Monte Carlo Integration

**Purpose:** Aggregate many fire scenarios to estimate annual burn probability for each cell.

**Algorithm:**

```
For each iteration (N = 1000):
    1. Sample a random fire season day (May-October weighted)
    2. Generate weather scenario:
       - Wind speed from Weibull distribution (scale=5, shape=2)
       - Wind direction from historical distribution
         (bimodal: offshore Diablo winds vs. onshore sea breeze)
       - 15% chance of "extreme" weather (1.5-2x multipliers)
    3. Compute ignition probability map for this weather
    4. Sample ignition locations (Poisson process, ~5 ignitions/iteration)
    5. Simulate fire spread for each ignition (8 hours duration)
    6. Record which cells burned

Burn Probability = (# times cell burned across all iterations) / N
```

**Weather Scenarios:**

| Parameter | Distribution | Extreme Multiplier |
|-----------|-------------|-------------------|
| Wind speed | Weibull(5, 2) | 1.5-2.0x |
| Temperature | Normal(30, 5) | +10-15 C |
| Humidity | Normal(30, 10) | 0.5-0.7x |

**Computational Efficiency:**
- Parallel execution across 28 CPU cores
- Each iteration simulates ~5 fires independently
- 1000 iterations completes in ~10 minutes

**Implementation:** `src/integration/monte_carlo.py`

```python
@dataclass
class MonteCarloConfig:
    n_iterations: int = 1000
    n_cores: int = 28
    extreme_weather_fraction: float = 0.15
    fire_duration_minutes: int = 480
    ignitions_per_iteration: int = 5

class MonteCarloEngine:
    def run(self, progress_callback=None) -> MonteCarloResult:
        """Run Monte Carlo simulation."""
        burn_counts = np.zeros(self.grid_shape)

        for i in range(self.config.n_iterations):
            weather = self._sample_weather()
            ignitions = self._sample_ignitions(weather)

            for ignition in ignitions:
                burned = self._simulate_spread(ignition, weather)
                burn_counts += burned

        return MonteCarloResult(
            burn_counts=burn_counts,
            n_iterations=self.config.n_iterations
        )
```

### 4. Calibration

**Purpose:** Adjust predicted probabilities to match observed burn rates.

**Method:**
```
calibration_factor = observed_burn_rate / predicted_burn_rate
calibrated_probability = raw_probability * calibration_factor
```

For Sonoma County 2015-2022:
- Raw prediction: 26.9% of cells predicted to burn
- Observed (holdout): 18.9% actually burned
- Calibration factor: 0.705

**Implementation:** `src/integration/monte_carlo.py`

```python
def calibrate_predictions(result: MonteCarloResult,
                          observed_fires: gpd.GeoDataFrame) -> MonteCarloResult:
    """Calibrate burn probability to match observed burn rate."""
    observed_rate = compute_observed_burn_rate(result, observed_fires)
    predicted_rate = result.raw_burn_probability.mean()

    factor = observed_rate / predicted_rate
    result.calibration_factor = factor

    return result
```

### 5. Parcel Aggregation

**Purpose:** Convert raster burn probability to parcel-level risk scores.

**Method:** Zonal statistics (rasterstats) computing:
- `burn_prob_mean` - Average probability across parcel (primary output)
- `burn_prob_max` - Worst-case probability for any part of parcel
- `burn_prob_std` - Standard deviation (uncertainty)

**Implementation:** `src/integration/parcel_aggregation.py`

```python
def aggregate_to_parcels(
    burn_probability_path: Path,
    parcels_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Aggregate burn probability raster to parcel polygons."""
    stats = zonal_stats(
        parcels_gdf,
        str(burn_probability_path),
        stats=['mean', 'max', 'std']
    )
    parcels_gdf['burn_prob_mean'] = [s['mean'] for s in stats]
    parcels_gdf['burn_prob_max'] = [s['max'] for s in stats]
    parcels_gdf['burn_prob_std'] = [s['std'] for s in stats]
    return parcels_gdf
```

## Data Sources

| Dataset | Resolution | Source | URL |
|---------|------------|--------|-----|
| Fire Perimeters | Polygon | CAL FIRE FRAP | data.ca.gov/dataset/california-fire-perimeters-all |
| Digital Elevation Model | 30m | USGS 3DEP | nationalmap.gov/3DEP |
| Fuel Models (FBFM40) | 30m | LANDFIRE 2023 | landfire.gov |
| Canopy Cover/Height | 30m | LANDFIRE 2023 | landfire.gov |
| Weather (GridMET) | 4km | Climatology Engine | climatologylab.org/gridmet |
| Parcels | Polygon | County Assessor | varies by county |

## Ex Ante Requirement

**Critical:** Predictions for year Y use only data available before Y.

```python
def generate_exante_prediction(year):
    training_years = range(2010, year)  # Only pre-year fire history
    fuel_data = load_landfire(year - 1)  # Most recent pre-year fuel data
    weather_climatology = compute_climatology(range(2000, year))
```

This ensures predictions could have been made prospectively, enabling fair backtesting.

## Validation Strategy

### Within-Year Holdout

For each year, 30% of fires are randomly held out for validation:
- **Training fires (70%):** Used to inform ignition model
- **Holdout fires (30%):** Used only for validation

This approach:
- Tests on fires from the same climate regime (avoids temporal bias)
- Ensures model doesn't "see" fires it's predicting
- Stratifies by fire size to ensure large fires in both sets

### Metrics

| Metric | Purpose | Target |
|--------|---------|--------|
| AUC-ROC | Discrimination (ranking) | > 0.80 |
| Expected Calibration Error | Probability accuracy | < 0.05 |
| Brier Skill Score | Overall skill vs climatology | > 0 |

### Current Performance (Sonoma County)

| Level | AUC-ROC | Notes |
|-------|---------|-------|
| Ignition Model | 0.96 | Excellent discrimination |
| Raster Burn Prob | 0.69 | Good relative ranking |
| Parcel Burn Prob | 0.56 | Limited by small sample size |

## Project Structure

```
burn_probs/
├── config/
│   └── settings.py           # Global configuration (paths, CRS, resolution)
├── data/
│   ├── raw/                  # Downloaded data
│   │   ├── fire_history/     # CAL FIRE perimeters (.parquet)
│   │   ├── terrain/          # DEM, slope, aspect, TPI (.tif)
│   │   ├── landfire/         # FBFM40, canopy cover/height (.tif)
│   │   ├── weather/          # GridMET climatology (.nc)
│   │   └── parcels/          # County parcel boundaries (.parquet)
│   ├── processed/            # Aligned rasters (270m, EPSG:3310)
│   │   ├── landscape.tif     # Combined 6-band landscape
│   │   └── training_data.parquet
│   └── output/               # Model outputs
│       ├── models/           # Trained ignition model (.joblib)
│       ├── monte_carlo/      # Burn probability rasters
│       └── validation/       # Metrics & plots
├── src/
│   ├── data_acquisition/     # Download from APIs
│   │   └── terrain.py        # USGS 3DEP download
│   ├── preprocessing/        # Raster alignment, feature creation
│   ├── ignition/             # Ignition probability model
│   │   ├── models.py         # RandomForestIgnition class
│   │   ├── train.py          # Training pipeline
│   │   └── feature_engineering.py
│   ├── spread/               # Fire spread simulation
│   │   ├── flammap_wrapper.py  # Rothermel + elliptical spread
│   │   └── landscape_builder.py
│   ├── integration/          # Monte Carlo engine
│   │   ├── monte_carlo.py    # MonteCarloEngine class
│   │   └── parcel_aggregation.py
│   └── validation/           # Validation metrics
│       ├── metrics.py        # AUC, calibration, Brier
│       └── fire_holdout.py   # Within-year holdout split
├── scripts/                  # CLI pipelines
│   ├── download_pilot_data.py
│   ├── download_sonoma_county.py
│   ├── train_ignition_model.py
│   ├── run_monte_carlo.py
│   └── validate_results.py
└── tests/
```

## Quick Start

### 1. Install Dependencies

```bash
conda create -n burn_probs python=3.11 geopandas rasterio xarray scikit-learn
conda activate burn_probs
pip install -e .
```

### 2. Download Data (Sonoma County Pilot)

```bash
python scripts/download_sonoma_county.py
```

Downloads:
- CAL FIRE perimeters (22,810 fires statewide)
- USGS 3DEP terrain (270m resolution, Sonoma bounds)
- LANDFIRE 2023 fuel models (270m resolution, Sonoma bounds)

### 3. Train Ignition Model

```bash
python scripts/train_ignition_model.py \
    --county Sonoma \
    --training-years 2010-2017 \
    --validation-years 2018-2022
```

Output: `data/output/models/ignition_model_random_forest.joblib`

### 4. Run Monte Carlo Simulation

```bash
python scripts/run_monte_carlo.py \
    --county Sonoma \
    --year 2020 \
    --iterations 1000 \
    --cores 28 \
    --calibrate \
    --calibration-years 2015-2022
```

Output: `data/output/monte_carlo/sonoma/2020/burn_probability.tif`

### 5. Validate Results

```bash
python scripts/validate_results.py --county Sonoma --year 2020
```

Output: `data/output/validation/sonoma/2020/reliability_diagram.png`

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 1000 | Monte Carlo iterations (more = smoother probability) |
| `--cores` | 28 | Parallel workers |
| `--extreme-fraction` | 0.15 | Fraction of extreme weather scenarios |
| `--calibrate` | False | Apply calibration from historical data |
| `--calibration-years` | 2015-2022 | Years for calibration |

## Output Schema

### Parcel-Level Output (CSV/Parquet)

| Field | Type | Description |
|-------|------|-------------|
| apn | string | Assessor Parcel Number |
| burn_prob_mean | float | Annual burn probability (0-1) |
| burn_prob_max | float | Maximum probability within parcel |
| burn_prob_std | float | Standard deviation (uncertainty) |

### Raster Output (GeoTIFF)

- Format: Single-band float32
- CRS: EPSG:3310 (California Albers)
- Resolution: 270m
- Values: Annual burn probability (0-1)

## Computational Requirements

| Task | Time | RAM | Cores |
|------|------|-----|-------|
| Data download | 2-4 hours | 8 GB | 4 |
| Ignition model training | 1-2 min | 16 GB | 28 |
| Monte Carlo (1k iterations) | 10-15 min | 32 GB | 28 |
| Parcel aggregation | 1-2 min | 16 GB | 8 |

## Technical Stack

```
# Geospatial
geopandas, rasterio, shapely, pyproj, rasterstats

# Data Processing
numpy, pandas, xarray

# Machine Learning
scikit-learn (Random Forest)

# Visualization
matplotlib
```

## Limitations

1. **Simplified fire spread:** Uses cellular automaton approximation rather than full physics-based FlamMap/FSim (Windows-only)
2. **Weather scenarios:** Based on climatological distributions, not dynamical weather models
3. **Human factors:** Limited representation of ignition sources (power lines, roads)
4. **Temporal resolution:** Annual probability only (not seasonal or daily)
5. **Suppression:** Does not model fire suppression efforts

## Future Improvements

1. **FlamMap/FSim integration:** Would require HPC cluster with Windows nodes
2. **More counties:** Expand from Sonoma to all California
3. **Longer validation:** 10+ year hindcast for robust uncertainty quantification
4. **Dynamic fuels:** Account for fuel accumulation and post-fire recovery
5. **Climate projections:** Use downscaled GCM outputs for future scenarios

## References

- Anderson, H.E. (1983). Predicting wind-driven wild land fire size and shape. USDA Forest Service Research Paper INT-305.
- Rothermel, R.C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.
- Scott, J.H., & Burgan, R.E. (2005). Standard fire behavior fuel models. USDA Forest Service General Technical Report RMRS-GTR-153.
- Finney, M.A. (2006). An overview of FlamMap fire modeling capabilities. USDA Forest Service Proceedings RMRS-P-41.

## License

This project is for research and educational purposes.
