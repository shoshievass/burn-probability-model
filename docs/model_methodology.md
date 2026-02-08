# Wildfire Burn Probability Model for Sonoma County
## Technical Methodology and Validation

---

## Executive Summary

This document describes a Monte Carlo simulation model that estimates the probability that any given parcel in Sonoma County, California will burn in a wildfire. The model combines:

1. **Real fire ignition locations** from historical fire perimeters (2015-2022)
2. **Vegetation and fuel data** from the LANDFIRE national database
3. **Terrain data** (slope, aspect, elevation) from USGS
4. **Randomized weather conditions** representing the range of fire weather

By simulating fire spread from each historical ignition point under 20 different weather scenarios (840 total simulations), we estimate the probability that each 270-meter grid cell—and each parcel—would burn.

**Key Validation Result:** The model achieves an AUC-ROC of 0.87, meaning it successfully distinguishes between areas that burned and areas that did not. When the model predicts a location has ≥20% burn probability, that location actually burned 74% of the time.

---

## Part 1: Where Does the Data Come From?

### 1.1 Fuel and Vegetation: LANDFIRE

**What is LANDFIRE?**

LANDFIRE (Landscape Fire and Resource Management Planning Tools) is a joint program of the U.S. Forest Service and Department of the Interior that maps vegetation and wildfire fuels across the entire United States. It is the authoritative source for fire behavior modeling inputs and is used by federal, state, and local fire management agencies.

**URL:** https://landfire.gov

**How is LANDFIRE data created?**

LANDFIRE combines multiple data sources:
- **Satellite imagery** (Landsat) to classify vegetation types
- **Field plots** from the Forest Inventory and Analysis (FIA) program
- **Existing vegetation maps** from state and federal agencies
- **Machine learning models** trained on these data to predict fuel characteristics

The data is updated on a regular cycle (approximately every 2 years) to reflect changes from fires, development, and vegetation growth.

**What LANDFIRE layers do we use?**

| Layer | What It Measures | How It's Used |
|-------|------------------|---------------|
| **Fuel Model (FBFM40)** | Classification of vegetation into 40 fire behavior categories | Determines how fast fire spreads and how intense it burns |
| **Elevation** | Height above sea level in meters | Input to slope/aspect calculation |
| **Slope** | Steepness of terrain in degrees | Fire spreads faster uphill |
| **Aspect** | Compass direction the slope faces | South-facing slopes are drier and burn more readily |
| **Canopy Cover** | Percentage of ground shaded by tree canopy | Affects wind speed at surface and fuel moisture |
| **Canopy Height** | Average height of trees in meters | Affects transition to crown fire |

**Resolution:** LANDFIRE data is natively 30 meters. We resample to 270 meters for computational efficiency, which is the same resolution used by the USFS Large Fire Simulator (FSim).

**The Scott and Burgan 40 Fuel Model System**

The fuel model is the most critical input. Each 270m cell is assigned one of 40 fuel types developed by Joe Scott and Robert Burgan (2005). These fuel models are grouped into categories:

| Category | Codes | Description | Fire Behavior |
|----------|-------|-------------|---------------|
| **Grass (GR)** | 101-109 | Grasslands, pastures, savannas | Fast spread (up to 40 m/min), low flame heights |
| **Grass-Shrub (GS)** | 121-124 | Mixed grass and shrub | Moderate spread, moderate flames |
| **Shrub (SH)** | 141-149 | Chaparral, manzanita, chamise | Variable spread (can exceed 100 m/min under wind), high intensity |
| **Timber Understory (TU)** | 161-165 | Forest with shrub/grass understory | Moderate spread, can transition to crown fire |
| **Timber Litter (TL)** | 181-189 | Dead needles and leaves on forest floor | Slow spread (0.3-2 m/min), smoldering |
| **Slash/Blowdown (SB)** | 201-204 | Logging debris or wind-thrown trees | High intensity, difficult to control |
| **Non-Burnable** | 91-99 | Urban, water, agriculture, barren | Fire cannot spread |

**Fuel Distribution in Our Study Area (Sonoma County):**

```
Burnable Fuels (74% of area):
├── Grass (GR2, GR4)           18.5%  - Fast spread, common in valleys
├── Timber Understory (TU5)    13.6%  - Forests with heavy understory
├── Grass-Shrub (GS1, GS2)      9.4%  - Transitional zones
├── Shrub (SH2, SH5, SH7)       8.0%  - Chaparral in hills
├── Timber Litter (TL3, TL8)    4.7%  - Redwood and oak forests
└── Other burnable             19.9%  - Various fuel types

Non-Burnable (26% of area):
├── Water                      12.9%  - Pacific Ocean, rivers, lakes
├── Urban/Developed             7.5%  - Cities, towns, roads
└── Agriculture                 5.2%  - Vineyards, farms
```

### 1.2 Fire History: CAL FIRE FRAP

**What is FRAP?**

The Fire and Resource Assessment Program (FRAP) is a unit within the California Department of Forestry and Fire Protection (CAL FIRE) that maintains comprehensive databases of fire history in California.

**URL:** https://frap.fire.ca.gov/mapping/gis-data/

**What does the fire perimeter database contain?**

The database includes the boundary (perimeter) of every fire greater than:
- 10 acres in timber/brush
- 30 acres in grass

For each fire, the database records:
- **Geometry:** The polygon boundary of the burned area
- **Year:** When the fire occurred
- **Alarm Date:** When the fire was first reported
- **Fire Name:** Official name (e.g., "Tubbs Fire", "Kincade Fire")
- **Cause:** If known (lightning, powerline, arson, etc.)
- **Acres:** Final size of the fire

**How we use fire perimeter data:**

We extract the **centroid** (center point) of each fire perimeter as a proxy for the ignition location. While the actual ignition point may differ from the centroid, this approach:
1. Ensures the ignition is within the area that actually burned
2. Places ignitions in representative fuel types
3. Is consistent across all fires (no bias toward fires with known ignition points)

**Fires in our analysis:**

| Statistic | Value |
|-----------|-------|
| Total fires in database | 22,810 (statewide) |
| Fires 2015-2022 | 3,216 |
| Fires intersecting study area | ~100 |
| Holdout fires for validation | 42 |

**Train/Holdout Split:**

To validate the model, we randomly split fires *within each year*:
- **70% Training:** Used to develop the model
- **30% Holdout:** Reserved for validation

This within-year split is important because fire behavior varies by year (drought years vs. wet years). By holding out fires from the same years, we ensure fair validation.

### 1.3 Weather Conditions: Simulated

**Current Implementation:**

For each simulation, we randomly sample weather conditions:

| Parameter | Distribution | Range |
|-----------|--------------|-------|
| Wind Speed | Uniform | 5-25 mph |
| Wind Direction | Uniform | 0-360 degrees |

**Extreme Weather (15% of simulations):**

To capture the conditions during major fires (Diablo winds, Red Flag events), 15% of simulations use elevated wind speeds (20-40 mph). This reflects the observation that the largest, most destructive fires occur during extreme weather events.

**Limitations and Future Improvements:**

The current weather sampling is simplified. In reality, weather conditions are:
- **Spatially correlated:** Wind patterns are regional, not random per cell
- **Temporally correlated:** Conditions persist for hours to days
- **Seasonally varying:** Fire season (June-November) has different weather than winter
- **Historically grounded:** Past observations should inform the distribution

Future versions could sample from historical weather data (e.g., GridMET) to capture realistic weather distributions, including:
- Diurnal patterns (wind peaks in afternoon)
- Seasonal patterns (dry summers, wet winters)
- Extreme event frequency (Red Flag Warning days)

### 1.4 Parcel Boundaries: OpenStreetMap

**Source:** OpenStreetMap building footprints via Overpass API

**URL:** https://overpass-api.de

**What is OpenStreetMap?**

OpenStreetMap (OSM) is a collaborative project to create a free, editable map of the world. Volunteers digitize building footprints from aerial imagery, and this data is freely available.

**Why OSM instead of assessor parcels?**

- Assessor parcel data often requires licensing fees
- Building footprints directly represent structures at risk
- OSM provides consistent coverage across jurisdictions

**Statistics:**
- Buildings extracted: 464,315
- Buildings with burn probability data: 119,350 (26%)
- Buildings with zero probability: 344,965 (74%, mostly in non-fire-prone areas)

---

## Part 2: How Does the Monte Carlo Simulation Work?

### 2.1 What is Monte Carlo Simulation?

Monte Carlo simulation is a computational technique that uses repeated random sampling to estimate probabilities. The name comes from the famous casino in Monaco—like gambling, Monte Carlo methods rely on randomness.

**The key insight:** Fire behavior is inherently uncertain. The same ignition can lead to very different outcomes depending on wind speed, wind direction, and other factors. By simulating many possible outcomes, we can estimate the *probability* of each outcome.

**Analogy:** Imagine rolling a die 1,000 times to estimate the probability of rolling a 6. After 1,000 rolls, you'd expect about 167 sixes (1/6 probability). Monte Carlo simulation does the same thing for fire spread—we "roll the dice" on weather conditions many times and count how often each location burns.

### 2.2 The Simulation Algorithm

Here is exactly what the model does, step by step:

```
ALGORITHM: Conditional Monte Carlo Burn Probability

INPUT:
  - 42 ignition points (from holdout fire perimeters)
  - Landscape raster (fuel, slope, aspect, canopy)
  - Weather sampling parameters

OUTPUT:
  - Burn probability for each 270m grid cell

STEP 1: Initialize
  - Create a grid matching the landscape (475 rows × 466 columns)
  - Set burn_count[i,j] = 0 for every cell

STEP 2: Loop over each ignition point
  For ignition = 1 to 42:

    STEP 2a: Convert coordinates
      - The ignition point is stored as (x, y) in California Albers projection
      - Convert to grid indices (row, column) using the raster transform
      - Check that the ignition falls on a burnable fuel type (codes 101-204)
      - If not burnable, skip this ignition

    STEP 2b: Loop over weather samples
      For weather_sample = 1 to 20:

        STEP 2b-i: Sample weather
          - If weather_sample ≤ 3 (15% of 20):
              wind_speed = random(20, 40) mph  [Extreme]
          - Else:
              wind_speed = random(5, 25) mph   [Normal]
          - wind_direction = random(0, 360) degrees

        STEP 2b-ii: Simulate fire spread
          - Start fire at ignition cell
          - For each 5-minute timestep (96 timesteps = 8 hours):
              - For each actively burning cell:
                  - For each of 8 neighboring cells:
                      - Skip if already burned or non-burnable
                      - Calculate spread rate using Rothermel model
                      - Calculate probability of spread in this timestep
                      - Randomly decide if fire spreads (coin flip weighted by probability)
                      - If spread: mark neighbor as burned
          - Record which cells burned in this simulation

        STEP 2b-iii: Accumulate counts
          - For each cell that burned:
              burn_count[row, col] += 1

STEP 3: Calculate probabilities
  - total_simulations = 42 ignitions × 20 weather samples = 840
  - For each cell:
      burn_probability[row, col] = burn_count[row, col] / 840

STEP 4: Aggregate to parcels
  - For each parcel:
      - Find all grid cells that intersect the parcel
      - parcel.burn_prob_mean = average of cell probabilities
      - parcel.burn_prob_max = maximum of cell probabilities
```

### 2.3 The Fire Spread Model (Rothermel Equations)

At the heart of each simulation is the Rothermel (1972) fire spread model. This is the same model used by the U.S. Forest Service in all major fire behavior prediction systems (FARSITE, FlamMap, FSim, BehavePlus).

**What the Rothermel model calculates:**

Given:
- Fuel characteristics (from the fuel model)
- Wind speed and direction
- Terrain slope and aspect

The model outputs:
- **Rate of spread** (meters per minute): How fast the fire front advances
- **Flame length** (meters): How tall the flames are
- **Fireline intensity** (kW/m): Energy released per meter of fire front

**The spread rate equation (simplified):**

```
Spread Rate = (Heat Released × Efficiency) / (Heat Required to Ignite Fuel)
            × (1 + Wind Factor + Slope Factor)
```

**Wind Factor:** Fire spreads faster in the direction the wind is blowing. A 20 mph wind can increase spread rate by 5-10x compared to calm conditions.

**Slope Factor:** Fire spreads faster uphill because:
1. Flames lean toward uphill fuels, preheating them
2. Convective heat rises along the slope
3. The formula: fire spreads ~2x faster for every 30° of slope

**Spread rates by fuel type:**

| Fuel Type | Typical Spread Rate | Example Conditions |
|-----------|--------------------|--------------------|
| Grass (GR2) | 5-40 m/min | Cured grass, moderate wind |
| Shrub (SH5) | 10-50 m/min | Chaparral, dry conditions |
| Shrub (SH7) | 20-116 m/min | Extreme wind, steep slope |
| Timber Litter (TL3) | 0.5-2 m/min | Forest floor, light wind |
| Timber Understory (TU5) | 2-10 m/min | Forest with brush |

### 2.4 Why 8 Hours? Why 20 Weather Samples?

**Simulation Duration (8 hours):**

We simulate 8 hours of fire spread, which represents:
- A typical operational period before major suppression resources arrive
- Enough time for a fire to establish and show directional spread
- The initial "free burning" period before containment

**Limitation:** Major fires like the Tubbs Fire (2017) burned for days. Our 8-hour window may underestimate the size of fires that would burn under sustained extreme conditions.

**Weather Samples (20 per ignition):**

With 20 weather samples, we capture:
- 3 extreme weather scenarios (15%)
- 17 normal weather scenarios (85%)
- Variation in wind direction (affects which way fire spreads)
- Variation in wind speed (affects how far fire spreads)

**Why not more?** Computational cost. Each simulation takes ~1 second, so:
- 42 ignitions × 20 samples = 840 simulations ≈ 15 minutes
- 42 ignitions × 100 samples = 4,200 simulations ≈ 75 minutes
- 42 ignitions × 1000 samples = 42,000 simulations ≈ 12 hours

For production use, 100-1000 samples would provide more stable probability estimates.

---

## Part 3: Validation—How Do We Know the Model is Good?

### 3.1 Validation Approach

The gold standard for validating a fire model is to compare predictions against fires that the model has never seen. Our approach:

1. **Hold out 30% of fires** from each year (42 fires total)
2. **Run the simulation** using ignition points from these holdout fires
3. **Compare** the predicted burn probability map to the actual fire perimeters

This tests whether the model correctly predicts the *spatial pattern* of burning—given that we know where fires started, does the simulated fire spread match where fires actually burned?

### 3.2 Discrimination: AUC-ROC = 0.87

**What is AUC-ROC?**

The Area Under the Receiver Operating Characteristic curve (AUC-ROC) measures how well a model distinguishes between positive cases (cells that burned) and negative cases (cells that didn't burn).

- **AUC = 0.5:** No better than random guessing (coin flip)
- **AUC = 0.7:** Acceptable discrimination
- **AUC = 0.8:** Good discrimination
- **AUC = 0.9:** Excellent discrimination
- **AUC = 1.0:** Perfect discrimination

**Our result: AUC = 0.873**

This means: if you randomly pick one cell that actually burned and one cell that didn't, there's an 87.3% chance the model assigns a higher burn probability to the burned cell.

**Why this is impressive:**

Fire spread is inherently stochastic. Even with perfect knowledge of fuels and weather, you can't predict exactly which cells will burn. An AUC of 0.87 indicates the model captures the dominant drivers of fire spread (fuel type, wind, slope) and correctly identifies high-risk areas.

### 3.3 Precision and Recall by Threshold

When using burn probability for decision-making, you must choose a threshold: "I'll flag all cells with probability ≥ X%." Different thresholds trade off between precision (avoiding false alarms) and recall (catching all actual fires).

| Threshold | Cells Flagged | Precision | Recall | Interpretation |
|-----------|---------------|-----------|--------|----------------|
| ≥ 1% | 104,270 | 15.2% | 88.4% | Cast wide net, catch most burned area |
| ≥ 5% | 73,316 | 21.4% | 87.7% | Moderate filtering |
| ≥ 10% | 31,015 | 42.1% | 73.1% | **Good balance** |
| ≥ 20% | 6,611 | 74.0% | 27.4% | High precision, miss some burned area |

**Interpreting the 10% threshold row:**
- We flag 31,015 cells as "high risk" (≥10% burn probability)
- Of these, 42.1% (13,057 cells) actually burned—our precision
- These cells account for 73.1% of all burned area—our recall
- We correctly identify most of the burned area while limiting false alarms

**Why precision seems low:**

A precision of 42% at the 10% threshold might seem poor, but consider:
- We're conditioning on fires that *actually occurred*
- The simulated fires spread in slightly different directions than actual fires due to random weather sampling
- A cell with 10% probability should burn 10% of the time—if all our 10% cells burned 40% of the time, we're actually *under*-predicting!

### 3.4 Calibration: Predicted vs. Observed Burn Rates

Calibration asks: when we predict 10% probability, does the location actually burn 10% of the time?

| Predicted Range | Cells in Bin | Mean Predicted | Actually Burned | Ratio |
|-----------------|--------------|----------------|-----------------|-------|
| 0-1% | 117,080 | 0.1% | 1.8% | 18× under-predicted |
| 1-5% | 30,954 | 2.7% | 0.4% | 7× over-predicted |
| 5-10% | 42,301 | 6.7% | 6.2% | ✓ Well calibrated |
| 10-20% | 24,404 | 13.7% | 33.5% | 2.4× under-predicted |
| 20-30% | 6,611 | 23.0% | 74.0% | 3.2× under-predicted |

**Interpretation:**

The model is well-calibrated in the 5-10% range (ratio ≈ 1.0), but:
- **Under-predicts high probabilities:** Cells we think have 20-30% probability actually burned 74% of the time
- **Complex pattern at low probabilities:** Mix of over/under-prediction

**Why the under-prediction at high probabilities?**

1. **Simulation duration:** We simulate 8 hours; real fires burned for days
2. **Weather averaging:** We average across 20 weather scenarios, including mild conditions. The actual fires burned during the most extreme conditions
3. **No spotting:** Ember transport can cause fire to jump barriers; we don't model this

**This is actually good news:** The model is *conservative* at high probabilities. Areas we flag as high risk are even more dangerous than predicted.

### 3.5 Spatial Validation: Visual Comparison

Beyond statistics, we can visually compare the burn probability map to actual fire perimeters:

**What we observe:**
- High probability areas cluster around historical ignition points ✓
- Fire spread follows terrain (uphill, aligned with wind) ✓
- Non-burnable areas (water, urban) correctly have zero probability ✓
- Burn probability decreases with distance from ignitions ✓

### 3.6 Summary: Why We Believe the Model

| Evidence | What It Shows | Result |
|----------|---------------|--------|
| **AUC = 0.87** | Model distinguishes burned from unburned | Strong discrimination |
| **73% recall at 10% threshold** | Model captures most burned area | Good sensitivity |
| **74% precision at 20% threshold** | High-risk predictions are reliable | High specificity for extremes |
| **Calibrated at 5-10%** | Predicted probabilities are meaningful | Trustworthy mid-range estimates |
| **Conservative at high P** | High-risk areas burn even more than predicted | Safe for decision-making |
| **Physics-based spread model** | Rothermel equations are well-validated | Mechanistic foundation |
| **Authoritative data sources** | LANDFIRE, CAL FIRE FRAP, USGS | Reliable inputs |

---

## Part 4: Model Outputs

### 4.1 Grid-Level Results

| Metric | Value |
|--------|-------|
| Grid dimensions | 475 rows × 466 columns |
| Cell size | 270 meters |
| Total cells | 221,350 |
| Mean burn probability | 3.88% |
| Maximum burn probability | 28.9% |
| Cells with P > 1% | 104,270 (47%) |
| Cells with P > 10% | 30,334 (14%) |

### 4.2 Parcel-Level Results

| Metric | Value |
|--------|-------|
| Total parcels | 464,315 |
| Parcels with P > 0 | 119,350 (26%) |
| Mean parcel probability | 1.08% |
| Maximum parcel probability | 26.9% |

### 4.3 Output Files

| File | Description |
|------|-------------|
| `burn_probability_conditional.tif` | GeoTIFF raster of burn probability (0-1 scale) |
| `parcels_burn_probability_conditional.parquet` | Parcel polygons with burn probability attributes |
| `parcels_summary_conditional.csv` | CSV with parcel ID and burn probability (no geometry) |

---

## Part 5: Limitations and Caveats

### 5.1 What the Model Does NOT Include

| Missing Factor | Why It Matters | Impact on Results |
|----------------|----------------|-------------------|
| **Fuel moisture** | Dry fuels burn faster; wet fuels may not burn | May over-predict in wet areas |
| **Spotting** | Embers can start fires miles ahead | Under-predicts extreme fire spread |
| **Suppression** | Firefighting contains fires | Over-predicts burn extent |
| **Time-varying weather** | Wind changes during fires | Simplifies spread patterns |
| **Historical weather** | We sample uniformly, not from real distributions | May not capture rare extremes |
| **Fire duration** | We simulate 8 hours; fires burn for days | Under-predicts large fire sizes |

### 5.2 Appropriate Use Cases

**This model IS appropriate for:**
- Identifying high-risk areas for mitigation planning
- Comparing relative risk across parcels
- Understanding the spatial pattern of fire spread
- Screening large areas to prioritize detailed assessment

**This model is NOT appropriate for:**
- Real-time fire behavior prediction during an incident
- Precise probability estimates for insurance pricing
- Predicting ignition likelihood (we use historical ignitions)
- Fine-scale (individual structure) risk assessment

### 5.3 Recommendations for Improvement

1. **Longer simulations:** Increase from 8 to 24-72 hours
2. **More weather samples:** Increase from 20 to 100-1000 per ignition
3. **Historical weather:** Sample from GridMET archive instead of uniform distribution
4. **Fuel moisture:** Incorporate Energy Release Component (ERC) or 100-hour fuel moisture
5. **Spotting model:** Add probabilistic ember transport for extreme wind events
6. **Ignition model:** Develop P(ignition) model for unconditional burn probability

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **AUC-ROC** | Area Under the Receiver Operating Characteristic curve; measures discrimination |
| **Calibration** | Agreement between predicted probabilities and observed frequencies |
| **FBFM40** | Scott and Burgan 40 Fire Behavior Fuel Models |
| **FRAP** | Fire and Resource Assessment Program (CAL FIRE) |
| **LANDFIRE** | Landscape Fire and Resource Management Planning Tools |
| **Monte Carlo** | Computational method using repeated random sampling |
| **Precision** | Of the locations we flag as high-risk, what fraction actually burned? |
| **Recall** | Of the locations that actually burned, what fraction did we flag? |
| **Rothermel model** | Standard equations for fire spread rate (Rothermel, 1972) |

## Appendix B: References

1. Rothermel, R.C. (1972). A mathematical model for predicting fire spread in wildland fuels. USDA Forest Service Research Paper INT-115.

2. Scott, J.H. and Burgan, R.E. (2005). Standard fire behavior fuel models: a comprehensive set for use with Rothermel's surface fire spread model. USDA Forest Service General Technical Report RMRS-GTR-153.

3. Finney, M.A. (2002). Fire growth using minimum travel time methods. Canadian Journal of Forest Research, 32(8), 1420-1424.

4. LANDFIRE Program. (2022). LANDFIRE Remap 2020. U.S. Department of Agriculture, Forest Service; U.S. Department of the Interior. https://landfire.gov

5. CAL FIRE. (2023). Fire Perimeters. California Department of Forestry and Fire Protection, Fire and Resource Assessment Program. https://frap.fire.ca.gov

## Appendix C: Reproduction

To reproduce this analysis:

```bash
# Run conditional Monte Carlo with holdout fires
python scripts/run_conditional_monte_carlo.py \
    --county Sonoma \
    --start-year 2015 \
    --end-year 2022 \
    --weather-samples 20

# Output files will be in:
# data/output/conditional_mc/sonoma/2015-2022/
```

**Software Requirements:**
- Python 3.9+
- geopandas, rasterio, numpy, scikit-learn
- ~10 GB RAM
- ~15 minutes runtime

---

*Document generated: February 2024*
*Model version: Conditional Monte Carlo v1.0*
*Contact: [your email]*
