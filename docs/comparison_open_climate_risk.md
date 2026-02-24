# Comparison: burn_probs vs. Open Climate Risk (CarbonPlan)

This document provides a detailed comparison between the **burn_probs** model (this repository) and CarbonPlan's **Open Climate Risk (OCR)** platform for wildfire risk assessment. Both aim to estimate wildfire risk at the building/parcel level, but they take fundamentally different approaches.

---

## Executive Summary

| Dimension | burn_probs | FSim (Riley et al. 2025) | Open Climate Risk (OCR) |
|-----------|-----------|--------------------------|------------------------|
| **Core approach** | Independent Monte Carlo simulation | Federal Monte Carlo simulation (gold standard) | Post-processing of FSim + WRC federal products |
| **Fire spread** | Rothermel surface fire (cellular automaton) | Rothermel surface + crown fire (MTT algorithm) | Wind-informed spatial blurring (no fire physics) |
| **Burn probability** | Self-generated | Self-generated | Ingested from Riley et al. 2025 (FSim) |
| **Fire intensity** | Not output | 6 flame-length probability classes | Inherited via cRPS (Scott et al. 2024) |
| **Geographic scope** | California (Sonoma pilot) | CONUS (134 FPUs) | CONUS (156M buildings) |
| **Resolution** | 270m | 270m nationally, 90-120m regional | 30m raster (BP upscaled from 270m) |
| **Building data** | OSM via Overpass API | N/A (raster only) | Overture Maps Foundation |
| **Risk metric** | Annual burn probability (0-1) | BP + flame-length probabilities | RPS = BP x cRPS (categorical 0-10 scale) |
| **Weather** | GridMET (4km daily, empirical sampling) | Synthetic autoregressive ERC (10k-50k years) | CONUS404 (4km hourly, wind direction only) |
| **Ignition modeling** | Random Forest (all fires) | Logistic regression on ERC (large fires only) | None |
| **Suppression** | None | Statistical perimeter-trimming | N/A |
| **Crown fire** | None | Van Wagner + Rothermel crown models | N/A |
| **Calibration** | Scalar factor (observed/predicted) | Iterative (fire size dist. + mean BP) | None (inherits FSim) |
| **Validation** | Holdout fire backtesting, AUC, calibration plots | Moran et al. 2025 (57-80% burned area in top 20% BP) | Cross-dataset concordance, historical benchmarking |

---

## 1. Philosophical Approach

### burn_probs: Simulation from First Principles

burn_probs is a **process-based model** that simulates individual fire events from ignition through spread. It samples weather conditions, generates ignition locations, runs physics-based fire spread (Rothermel equations on a cellular automaton), and accumulates thousands of simulated fires to derive burn probability. This is analogous to the USFS's FSim (Fire Simulator) approach but implemented independently.

Key implication: the model can be re-parameterized for novel conditions (different fuel treatments, climate scenarios, land use changes) without depending on external datasets for its core predictions.

### OCR: Assembly and Enhancement of Federal Products

OCR takes a **data integration approach**. It ingests the burn probability raster already produced by the USFS (Riley et al. 2025, which itself was generated using FSim), then enhances it with directional wind-informed spreading and multiplies by conditional risk values (Scott et al. 2024) to produce a final risk score. OCR's primary innovation is the **wind-adjusted blurring** that spreads burn probability into non-burnable (developed) areas where the USFS product assigns zero probability.

Key implication: OCR inherits the quality and assumptions of its upstream federal datasets. It cannot independently model new fire scenarios but can rapidly scale to CONUS by leveraging existing products.

---

## 2. FSim: The Upstream Model Behind OCR's Burn Probability

Since OCR does not generate its own burn probability, understanding the comparison requires understanding **FSim** (the Fire Simulator), the USFS model that produces OCR's primary input. FSim was developed by Mark Finney at the Missoula Fire Sciences Laboratory (Finney et al. 2011) and is the most widely used large-fire simulation system in the United States.

### 2.1 FSim Architecture

FSim is a stochastic Monte Carlo simulator -- the same class of model as burn_probs. It simulates tens of thousands of hypothetical fire seasons to produce probabilistic burn probability and fire intensity maps. The system has four core modules:

1. **Synthetic Weather Generation** -- produces statistically realistic daily weather sequences
2. **Large-Fire Ignition** -- stochastically places ignitions based on historical fire-weather relationships
3. **Fire Growth** -- spreads fires across the landscape using the Minimum Travel Time (MTT) algorithm
4. **Suppression** -- statistically approximates containment effects on fire duration and size

### 2.2 FSim Weather Generation

FSim generates synthetic weather using an **autoregressive time-series model** of the Energy Release Component (ERC), a fire danger index that serves as a proxy for fuel moisture. The process:

1. Analyze historical weather records from RAWS (Remote Automatic Weather Stations) or gridded data over ~15-20 year periods
2. Characterize the **seasonal trend** in ERC, **autocorrelation** of daily residuals, and **daily standard deviation**
3. Stochastically generate artificial time series of daily ERC values that preserve the statistical properties of the historical record
4. Pair ERC values with correlated wind speed and direction distributions from historical records

This produces **10,000 to 50,000 synthetic "years"** of daily weather per simulation unit, far exceeding the historical record length. The synthetic weather preserves temporal autocorrelation (multi-day drought sequences, sustained wind events) while exploring the full statistical range of possible conditions.

**Comparison to burn_probs:** burn_probs samples actual historical days from GridMET (preserving exact covariance but limited to ~2,250 observed fire-season days across 15 years). FSim generates synthetic weather that can explore conditions not yet observed in the historical record, at the cost of assuming stationarity in weather statistics.

### 2.3 FSim Ignition Modeling

FSim models daily large-fire ignition probability using a **logistic regression** between historical large fire ignitions and ERC values for each simulation unit. Key details:

- Only the largest ~3-5% of fires (by size) are modeled -- FSim is explicitly a **large fire** simulator
- Fires are only ignited and spread on days when ERC meets or exceeds the **80th percentile** for that region
- Ignition locations are placed stochastically across the landscape based on historical fire occurrence density
- There is **no contagion among fires** -- each fire is simulated independently

**Comparison to burn_probs:** burn_probs uses a Random Forest classifier trained on actual ignition locations with landscape and weather features (AUC 0.96), generating spatially explicit ignition probability grids. FSim uses a simpler logistic regression on ERC alone for ignition *timing*, with spatial placement based on historical fire occurrence density. burn_probs models all fires; FSim focuses on large fires only. However, FSim's ignition model is calibrated across decades of data within each Fire Planning Unit (FPU), while burn_probs' Random Forest has been validated only for Sonoma County.

### 2.4 FSim Fire Growth: Minimum Travel Time (MTT) Algorithm

FSim uses the **Minimum Travel Time** algorithm (Finney 2002) rather than a cellular automaton:

- **Principle:** Fire growth follows Huygens' wavelet principle -- the fire perimeter advances as a series of expanding wavelets from each point on the edge
- **Spread model:** Rothermel (1972) surface fire spread equations, same as burn_probs, plus crown fire models (Van Wagner 1977, Rothermel 1991)
- **Algorithm:** MTT searches for pathways with minimum fire travel time from ignition points through the landscape, propagating fire across cell corners at user-specified resolution
- **Weather during growth:** Held constant within each fire growth period but changes daily across the fire's lifetime
- **Fire duration:** Fires burn for **multiple days** until ERC drops below a suppression-triggering threshold or the suppression model terminates the fire
- **Resolution:** 270m nationally, 90-120m for regional analyses

**Comparison to burn_probs:** Both use Rothermel surface spread equations, but:

| Aspect | burn_probs | FSim |
|--------|-----------|------|
| **Growth algorithm** | Cellular automaton (8-connected grid) | Minimum Travel Time (Huygens wavelets) |
| **Duration** | Fixed 8 hours per fire | Multi-day (until suppressed or weather changes) |
| **Weather variability** | Constant during 8-hr simulation | Changes daily during fire lifetime |
| **Crown fire** | Not modeled | Van Wagner/Rothermel crown fire models |
| **Fire size range** | Limited by 8-hr cap | Produces realistic large-fire size distributions |
| **Spotting** | Not modeled | Not modeled in base FSim |

The MTT algorithm is more computationally sophisticated than a cellular automaton and produces smoother, more realistic fire perimeters. The multi-day burn duration is critical: real large fires in California (Camp Fire, Tubbs Fire, Dixie Fire) burned for days to weeks. burn_probs' 8-hour cap means it systematically underestimates the size and probability of large fire events.

### 2.5 FSim Suppression Modeling

FSim includes a **statistical suppression model** that approximates containment:

- Determines the daily proportion of fire containment as a function of the elapsed proportion of fire duration and a regression-based suppression factor
- If ERC drops below a threshold for several consecutive days, fire growth is assumed to slow/stop in directions where suppression can occur ("perimeter trimming")
- Containment typically progresses from the fire's tail (upwind/downslope edge) toward the head
- The net effect is that simulated fires are smaller than they would be without suppression, producing fire size distributions that better match historical observations

**Comparison to burn_probs:** burn_probs does not model suppression at all. This means burn_probs over-predicts fire extent for any given ignition (all else equal), while FSim's suppression model helps calibrate fire sizes to match observed distributions.

### 2.6 FSim Calibration

FSim calibration is a structured iterative process:

1. Run simulation for a given region (Fire Planning Unit or "pyrome")
2. Compare modeled **fire size distribution** against the historical distribution using ordinary least squares regression over the prior ~15 years
3. Compare modeled **mean annual burn probability** against historical burn rates
4. Adjust inputs (ignition rates, suppression parameters) until:
   - The **slope of the fire size distribution** (log-log) matches historical data
   - The **average burn probability** falls within an acceptable range of the historical reference value
5. Repeat until convergence

Moran et al. (2025) benchmarked FSim calibration performance against 2020-2023 California wildfire activity and found: 56.7-79.8% of actually burned area occurred in the top 20% of predicted burn probability, with mean BP values in burned areas 238-349% greater than in unburned areas.

**Comparison to burn_probs:** burn_probs applies a simpler **single scalar calibration factor** (observed_annual / predicted_annual). FSim calibrates both the fire size distribution shape and the mean burn probability, a more rigorous approach that ensures not just the average but also the extremes are well-represented.

### 2.7 FSim Outputs Used by OCR

Riley et al. (2025) applied FSim across CONUS using:
- **LANDFIRE 2020** fuel and vegetation data
- **Circa 2011 climate** (2004-2018 weather records) and **circa 2047 projected climate** (CMIP5 ensemble of 6 GCMs, shifted monthly temperature/precipitation/humidity)
- **134 Fire Planning Units** (original) or individual pyromes with 30km buffers (updated methodology)
- **Minimum 10,000 iterations** per simulation unit
- **270m resolution**

The outputs consumed by OCR are:
- **Annual burn probability** (BP) -- probability that each 270m pixel burns in a given year
- **Conditional flame-length probabilities** (FLP) -- for each pixel, the probability of burning in each of 6 flame-length classes (used by Scott et al. 2024 for cRPS, not directly by OCR)

---

## 3. WRC and cRPS: How OCR Computes Structural Risk

OCR's risk metric (RPS) depends on the **Wildfire Risk to Communities** (WRC) project's conditional risk to potential structures (cRPS) from Scott et al. (2024). Understanding how cRPS is derived completes the picture of OCR's methodology.

### 3.1 Flame-Length Probability Classes

FSim produces conditional flame-length probabilities (FLP) for each pixel across **6 fire intensity levels (FILs)**:

| Class | Flame Length | Interpretation |
|-------|-------------|----------------|
| FIL1 | < 2 ft | Low intensity, surface fire |
| FIL2 | 2-4 ft | Moderate surface fire |
| FIL3 | 4-6 ft | High surface fire |
| FIL4 | 6-8 ft | Very high, potential crown fire initiation |
| FIL5 | 8-12 ft | Crown fire likely |
| FIL6 | > 12 ft | Extreme crown fire |

These are conditional probabilities: given that a pixel burns, what is the probability it burns at each intensity? They sum to 1.0 for each pixel.

### 3.2 Structure Response Function

The WRC project applies a **response function** representing the expected proportional loss to a hypothetical structure at each flame-length class. This is based on expert judgment and empirical structure loss data:

- At FIL1 (< 2 ft): minimal structural damage
- At FIL6 (> 12 ft): near-complete structural loss
- Values range from 0 (no damage) to -100 (complete loss)

The response function encodes the physical reality that low-intensity grass fires rarely destroy structures, while high-intensity crown fires almost certainly do.

### 3.3 cRPS Calculation

For each pixel:

```
cRPS = SUM over i=1..6 of: FLP_i * ResponseFunction_i
```

This gives the **expected percentage loss to a hypothetical structure, conditional on fire occurring**. It was computed at 270m resolution (matching FSim's BP output) by the WRC project, then Scott et al. (2024) produced a 30m version.

### 3.4 RPS Calculation (What OCR Outputs)

OCR computes the final risk metric:

```
RPS = BP_wind_adjusted * cRPS
```

Where `BP_wind_adjusted` is the Riley et al. burn probability after OCR's directional wind blurring. This gives the **expected annual percentage loss** to a hypothetical structure at each location, combining the probability of fire (BP) with the severity of consequences (cRPS).

### 3.5 Exposure Types

The WRC project also classifies each building's exposure:
- **Direct exposure:** Structure is adjacent to wildland vegetation and could be reached by flame contact
- **Indirect exposure:** Structure is in a developed area but could be reached by embers or home-to-home ignition
- **Not exposed:** Structure is sufficiently distant from both direct and indirect ignition sources

OCR's wind-informed blurring effectively extends risk into indirectly exposed areas that FSim's raw BP would assign zero probability.

### 3.6 Comparison to burn_probs

burn_probs does not compute fire intensity, flame length, or structural loss. It outputs only burn probability. To achieve equivalent actuarial utility, burn_probs would need to:
1. Output flame-length or fireline intensity per simulation (feasible from Rothermel equations but not currently implemented)
2. Apply a structure response function to convert intensity to expected loss
3. Combine probability and loss into an expected annual loss metric

This is a significant methodological gap, though the underlying physics (Rothermel equations) already compute the flame length -- it is simply not captured in the current output.

---

## 4. Burn Probability Estimation: Three-Way Comparison

The burn probability in OCR ultimately originates from FSim. Comparing all three systems side-by-side clarifies where burn_probs and OCR's upstream source (FSim) agree and diverge:

| Component | burn_probs | FSim (Riley et al. 2025) | OCR |
|-----------|-----------|--------------------------|-----|
| **Approach** | Independent Monte Carlo simulation | Monte Carlo simulation (federal standard) | Post-processing of FSim output |
| **Spread model** | Rothermel surface fire (cellular automaton) | Rothermel surface + crown fire (MTT algorithm) | No fire simulation; directional spatial blurring |
| **Weather** | GridMET daily samples (~2,250 days) | Synthetic autoregressive ERC time series (10k-50k years) | CONUS404 hourly (wind direction analysis only) |
| **Ignition** | Random Forest on landscape + weather features | Logistic regression on ERC (large fires only, >80th percentile ERC) | None |
| **Fire duration** | Fixed 8 hours | Multi-day (weather-dependent, with suppression) | N/A |
| **Suppression** | None | Statistical perimeter-trimming model | N/A |
| **Crown fire** | None | Van Wagner/Rothermel crown fire models | N/A |
| **Calibration** | Single scalar factor (observed / predicted annual rate) | Iterative: match fire size distribution slope + mean annual BP | None (inherits FSim calibration) |
| **Iterations** | 1,000-10,000 | 10,000-50,000 per simulation unit | N/A |
| **Resolution** | 270m (EPSG:3310) | 270m nationally, 90-120m regional (EPSG:4326) | 30m (EPSG:4326, upscaled from 270m BP) |
| **Scope** | California (Sonoma pilot) | CONUS (134 FPUs / pyromes) | CONUS (799 spatial chunks) |
| **Fire intensity output** | None | 6 flame-length probability classes | Inherited via cRPS |

### burn_probs Method

1. Sample weather scenario (from GridMET historical fire-season days or hardcoded defaults)
2. Compute ignition probability grid (base rate x ERC factor)
3. Sample ignition locations via Poisson process
4. Simulate fire spread for 8 hours (96 x 5-min timesteps) using Rothermel spread equations on an 8-connected cellular automaton
5. Accumulate burned cell counts across 1,000-10,000 iterations
6. Divide by number of iterations to get raw burn probability
7. Apply calibration factor (predicted vs. observed annual burn rate from historical fire perimeters)

### FSim Method (Upstream of OCR)

1. Generate 10,000-50,000 synthetic years of daily weather per simulation unit via autoregressive ERC time-series model
2. For each simulated day where ERC >= 80th percentile, stochastically place large-fire ignitions based on logistic regression of historical fire occurrence
3. Grow fires using Minimum Travel Time algorithm with Rothermel surface and crown fire models; weather changes daily during multi-day fire growth
4. Apply statistical suppression model (perimeter trimming based on ERC drop and elapsed fire duration)
5. Accumulate burn counts and flame-length tallies across all simulated years
6. Calibrate iteratively: adjust ignition rates and suppression parameters until fire size distribution slope and mean annual BP match historical observations

### OCR Method

1. Ingest Riley et al. 2025 burn probability raster (270m, from FSim)
2. Upscale from 270m to 30m resolution
3. Identify non-burnable pixels (developed areas where BP = 0)
4. Apply wind-informed directional blurring to spread BP into non-burnable areas (up to ~1.5 km, 3 iterations)
5. Wind directions from fire-weather analysis of CONUS404 hourly data (99th percentile FFWI conditions)
6. Eight elliptical blurring kernels (one per cardinal/ordinal direction) weighted by local fire-weather wind frequency

### Why This Matters

burn_probs and FSim are methodological siblings -- both are Monte Carlo fire simulators built on Rothermel spread equations. The key differences are in sophistication: FSim has synthetic weather generation (exploring weather space far beyond the historical record), multi-day fire duration, crown fire modeling, suppression modeling, and rigorous multi-metric calibration. burn_probs is simpler but independently controllable and transparent.

OCR is a fundamentally different kind of system. It does not simulate fires at all; it spatially redistributes the FSim-derived burn probability using wind-informed filters. OCR's contribution is extending risk estimates into developed areas where FSim assigns zero probability (since FSim treats urban fuel types as non-burnable), and combining BP with cRPS for a more actuarially useful risk metric.

burn_probs' 270m resolution matches FSim's national resolution. OCR's 30m output resolution comes from the 30m cRPS layer (Scott et al. 2024), not from a finer burn probability simulation -- the underlying BP is still 270m, upscaled.

---

## 5. Risk Metric

### burn_probs: Raw Burn Probability

Output is annual burn probability (0-1 continuous), aggregated to parcels via zonal statistics (mean, max, min, std, median). Optional Bayesian confidence intervals via beta distribution (Jeffreys prior). The metric answers: *"What is the probability this location burns in a given year?"*

### OCR: Risk to Potential Structures (RPS)

Output is **RPS = BP x cRPS**, where:
- **BP** = wind-adjusted burn probability (from Riley et al. 2025 + wind blurring)
- **cRPS** = conditional risk to potential structures (from Scott et al. 2024), representing expected net value change to a hypothetical structure *given* that burning occurs

The metric answers: *"What is the expected annual loss to a hypothetical structure at this location?"* This is a more actuarially relevant metric since it incorporates fire intensity and structure vulnerability, not just likelihood of burning. RPS is then converted to a categorical 0-10 score using percentile-based bins (75th, 82.5th, 88th, 92.5th, 96th, 98th, 99.5th, 99.9th, 99.99th percentiles).

### Why This Matters

burn_probs provides a single-dimension estimate (probability of burning) while OCR provides a two-dimensional estimate (probability x consequence). For insurance pricing, the OCR approach is theoretically more useful because a high-probability, low-intensity grass fire is very different from a low-probability, high-intensity crown fire for structure loss. However, burn_probs does output the dominant fuel model per parcel, which could serve as a proxy for fire intensity.

---

## 6. Datasets Compared

### 4.1 Burn Probability Baseline

| Attribute | burn_probs | OCR |
|-----------|-----------|-----|
| **Source** | Self-generated via Monte Carlo simulation | Riley et al. 2025 (USFS FSim) |
| **Native resolution** | 270m | 270m (upscaled to 30m) |
| **Time periods** | Current conditions (calibrated 2015-2022) | ~2011 climate conditions; ~2047 projected |
| **Fuel models** | LANDFIRE FBFM40 (Scott & Burgan 40) | Embedded in FSim (same LANDFIRE origin) |

Both ultimately trace back to LANDFIRE fuel models and Rothermel-family spread equations, but burn_probs runs its own simulation while OCR uses the USFS's pre-computed result.

### 4.2 Weather / Climate Data

| Attribute | burn_probs | OCR |
|-----------|-----------|-----|
| **Dataset** | GridMET (U. Idaho) | CONUS404 / Rasmussen et al. 2023 (NCAR-USGS) |
| **Resolution** | 4km daily | 4km hourly |
| **Variables used** | Temp, RH, wind speed/dir, ERC, fuel moisture | Wind speed, wind direction (for FFWI calculation) |
| **Time period** | 1979-present (fire season June-Oct) | 1979-2022 |
| **Usage** | Drives fire spread simulation directly (wind, moisture affect spread rate) | Identifies fire-weather conditions (99th percentile FFWI) to determine directional wind patterns for blurring |

GridMET is an observational gridded product; CONUS404 is a dynamically downscaled reanalysis. Both are 4km but CONUS404 has hourly temporal resolution enabling identification of specific fire-weather episodes. burn_probs uses weather to directly modulate fire spread rates in each simulation. OCR uses weather only to characterize the *direction* fire would spread under extreme conditions, not to drive a physical spread model.

### 4.3 Conditional Risk / Fire Intensity

| Attribute | burn_probs | OCR |
|-----------|-----------|-----|
| **Fire intensity** | Implicit in fuel model spread rates (no explicit flame length/intensity output) | Scott et al. 2024 cRPS at 30m (explicit conditional loss values) |
| **Structure vulnerability** | Not modeled | Implicit in cRPS (hypothetical "potential structure") |

This is a significant gap in burn_probs. The model computes whether a cell burns but not the fire intensity or expected structural damage. OCR leverages the 30m cRPS product which encodes the expected percentage loss to a structure conditional on burning, accounting for flame length, fire type (surface vs. crown), and ember exposure.

### 4.4 Building/Parcel Data

| Attribute | burn_probs | OCR |
|-----------|-----------|-----|
| **Source** | OpenStreetMap via Overpass API | Overture Maps Foundation |
| **Coverage** | Sonoma County: 464k buildings | CONUS: 156M buildings |
| **Method** | Centroid-based zonal statistics from 270m raster | Centroid sampling from 30m raster |
| **Underlying data** | OSM contributors | Composite of OSM, Esri, Microsoft ML Footprints, Google Open Buildings, USGS 3DEP |

Overture Maps is a superset that includes OSM plus several other sources, likely providing more complete coverage especially in rural areas where OSM may have gaps.

### 4.5 Fire History / Perimeters

| Attribute | burn_probs | OCR |
|-----------|-----------|-----|
| **Source** | CAL FIRE FRAP (data.ca.gov) | US historical fire perimeters (for benchmarking only) |
| **Usage** | Training ignition model, calibrating burn probability, validation | Benchmarking BP estimates against 70+ years of burn history |
| **Time period** | 2010-2022 (training); 2015-2022 (validation) | Full historical record |

burn_probs uses fire history for three purposes (ignition model training, calibration, validation). OCR uses it only for post-hoc benchmarking since it doesn't generate its own burn probability.

### 4.6 Terrain

| Attribute | burn_probs | OCR |
|-----------|-----------|-----|
| **Source** | USGS 3DEP (10m native) | Not directly used (embedded in upstream FSim/cRPS) |
| **Products** | DEM, slope, aspect, TPI | N/A |
| **Usage** | Slope/aspect affect fire spread rates; features for ignition model | Terrain effects already captured in Riley et al. burn probability |

### 4.7 Fuel Models

| Attribute | burn_probs | OCR |
|-----------|-----------|-----|
| **Source** | LANDFIRE FBFM40, canopy cover/height/base height/bulk density | Not directly used (embedded in upstream FSim/cRPS) |
| **Usage** | Drives Rothermel spread rates; determines burnability; features for ignition model | Fuel effects already captured in Riley et al. burn probability and Scott et al. cRPS |

---

## 7. Ignition Modeling

### burn_probs: Machine Learning Ignition Model

- **Model:** Random Forest classifier (500 trees, max depth 15)
- **Training data:** Actual ignition centroids (positive) + random non-fire locations (negative, 1:4 ratio)
- **Features:** Terrain (elevation, slope, aspect, TPI), vegetation (fuel model, canopy), weather (temp, RH, wind, ERC), temporal (day of year, season)
- **Performance:** AUC-ROC 0.96, top feature = fuel model (28.6%)
- **Usage:** Generates ignition probability grids for Monte Carlo sampling

### FSim (Upstream of OCR): Logistic Regression on ERC

- **Model:** Logistic regression relating daily large-fire ignition probability to Energy Release Component (ERC)
- **Scope:** Only large fires (largest ~3-5% by size for each simulation unit)
- **Threshold:** Fires only ignited on days with ERC >= 80th percentile for the region
- **Spatial placement:** Based on historical fire occurrence density within each Fire Planning Unit
- **No contagion:** Each fire is independent; fire size emerges from weather and landscape, not fire-to-fire interaction

### OCR: No Ignition Model

OCR does not model ignitions. The ignition process is already embedded in the Riley et al. 2025 burn probability product (which was generated using FSim).

### Why This Matters

| Aspect | burn_probs | FSim | OCR |
|--------|-----------|------|-----|
| **Model type** | Random Forest (500 trees) | Logistic regression on ERC | None |
| **Fire types** | All fires | Large fires only (top 3-5%) | N/A |
| **Features** | 10+ (terrain, fuel, weather, temporal) | ERC only (for timing); historical density (for location) | N/A |
| **Spatial precision** | Grid-cell level probability | Regional density-based placement | N/A |
| **Interpretability** | Feature importance (fuel model = 28.6%) | Simple ERC threshold | N/A |

burn_probs' explicit ignition model allows investigation of ignition drivers and scenario analysis (e.g., "What if ignitions near power lines doubled?"). FSim's simpler approach is well-validated at national scale but cannot isolate individual ignition drivers. OCR inherits FSim's ignition model as a black box.

---

## 8. Fire Spread Modeling

### burn_probs: Rothermel Cellular Automaton

- **Physics:** Rothermel (1972) surface fire spread equations
- **Algorithm:** Cellular automaton on 270m grid, 8-connected neighbors
- **Duration:** 8 hours per simulation (96 x 5-min timesteps)
- **Modifiers:** Wind (elliptical shape, LB ratio up to 8x), slope (up to 10x uphill), fuel model (base rate lookup)
- **Crown fire:** Not modeled
- **Non-burnable:** Fuel codes 91-99 (urban, water, agriculture, barren)
- **Limitations:** No spotting/ember transport, no suppression, fixed weather during each 8-hr simulation

### FSim (Upstream of OCR): Minimum Travel Time with Crown Fire

- **Physics:** Rothermel (1972) surface fire + Van Wagner (1977) crown fire initiation + Rothermel (1991) crown fire spread
- **Algorithm:** Minimum Travel Time (Finney 2002) -- Huygens' wavelet principle, searching for pathways with minimum fire travel time across cell corners
- **Duration:** Multi-day; fires burn until suppression model terminates them or weather conditions (ERC) drop below thresholds
- **Weather:** Changes daily during fire lifetime (not held constant)
- **Suppression:** Statistical perimeter-trimming model based on elapsed duration and ERC
- **Spotting:** Not modeled in base FSim
- **Resolution:** 270m nationally, 90-120m for regional analyses

### OCR: Wind-Informed Directional Blurring

- **Method:** Applies 8 elliptical blurring kernels (one per cardinal/ordinal direction) to the Riley et al. BP raster
- **Purpose:** Spread burn probability from burnable into non-burnable (developed) areas
- **Distance:** Up to ~1.5 km (3 iterations)
- **Wind weighting:** Kernel weights derived from 99th percentile FFWI fire-weather wind direction frequencies per 4km pixel
- **Innovation:** Directional spreading (fire reaches developed areas preferentially from the upwind wildland direction) rather than uniform circular blurring
- **Not a fire model:** No physics-based spread; purely a spatial redistribution of existing BP

### Why This Matters

Three very different approaches to fire spread sit in this comparison:

1. **FSim** is the gold standard for large-fire simulation in the US, with multi-day fire growth, crown fire physics, daily weather variation, and suppression modeling. Its MTT algorithm produces smoother, more realistic fire perimeters than cellular automata.

2. **burn_probs** implements the same core physics (Rothermel surface spread) but in a simplified form: fixed 8-hour duration, no crown fire, no suppression, constant weather. This means burn_probs systematically underestimates large fire events (California's most destructive fires burned for days to weeks) but is fully transparent and independently controllable.

3. **OCR** does not simulate fire at all. It spatially redistributes FSim's burn probability using wind-informed filters. This is computationally cheap and addresses a real limitation (extending risk into developed areas), but cannot capture terrain-driven fire behavior.

The 8-hour simulation cap in burn_probs is arguably its most significant limitation relative to FSim. Historical California megafires (Camp, Tubbs, Dixie) burned for days to weeks under sustained adverse weather. FSim's multi-day simulation with daily weather changes and suppression modeling captures this; burn_probs' fixed 8-hour window cannot.

However, OCR's directional blurring addresses a limitation that affects *both* burn_probs and FSim: structures in developed areas adjacent to wildlands face real fire risk even though both models treat urban fuels as non-burnable. OCR spreads this risk directionally; burn_probs partially addresses it through coarse resolution (270m cells at the WUI boundary contain both burnable and non-burnable land).

---

## 9. Calibration

### burn_probs

- **Method:** Compare predicted annual burn rate against observed historical burn rate from fire perimeters
- **Formula:** `calibration_factor = observed_annual / predicted_annual`
- **Observed conversion:** `annual = 1 - (1 - cumulative)^(1/n_years)` (converts multi-year cumulative to annual)
- **Application:** Multiply all raw probabilities by calibration factor
- **Example (Sonoma 2015-2022):** Factor = 0.705 (model slightly over-predicts)

### FSim (Upstream of OCR)

- **Method:** Iterative multi-metric calibration per simulation unit (Fire Planning Unit or pyrome)
- **Targets:**
  1. **Fire size distribution slope** (log-log) must match historical distribution via OLS regression over prior ~15 years
  2. **Mean annual burn probability** must fall within acceptable range of historical reference value
- **Adjustable parameters:** Ignition rates, suppression coefficients, fire duration parameters
- **Validation (Moran et al. 2025):** 56.7-79.8% of burned area fell in top 20% of predicted BP; mean BP in burned areas was 238-349% greater than in unburned areas; Kolmogorov-Smirnov tests confirmed significant separation (p < 0.01)

### OCR

OCR does not perform its own calibration. The Riley et al. 2025 burn probability is taken as-is (calibrated within the FSim framework). OCR's wind-adjusted spreading redistributes BP spatially but aims to conserve total probability mass within each processing region.

### Calibration Comparison

| Aspect | burn_probs | FSim | OCR |
|--------|-----------|------|-----|
| **Approach** | Single scalar factor | Iterative multi-metric | None (inherits FSim) |
| **What is calibrated** | Mean annual BP only | Fire size distribution slope + mean annual BP | N/A |
| **Reference data** | CAL FIRE FRAP perimeters (8 years) | Historical fire records (~15 years) per FPU | N/A |
| **Rigor** | Simple but effective for mean adjustment | Ensures both average and extremes are realistic | N/A |
| **Limitation** | Does not calibrate fire size distribution | Sensitive to FPU boundary definitions | Inherits upstream limitations |

---

## 10. Evaluation and Validation

### burn_probs: Fire Holdout Backtesting

**Strategy:**
- Within-year holdout: 70% fires for model input, 30% for validation
- Conditional Monte Carlo: Fix real ignition points, vary weather
- Tests whether simulated spread matches actual fire perimeters

**Metrics:**
| Metric | Value | Notes |
|--------|-------|-------|
| AUC-ROC (ignition model) | 0.96 | Excellent discrimination |
| AUC-ROC (holdout fires) | 0.87 | Good spatial prediction |
| Calibration | Well-calibrated 5-10%; conservative at high P | Under-predicts extreme fires (8-hr simulation vs. multi-day real fires) |
| Brier Score | Reported | Lower is better |
| Precision-Recall | Threshold-dependent | 10% threshold: 42% precision, 73% recall |

**Strengths:** Direct validation against actual fire outcomes. Can test "did the model predict burning where fires actually burned?"

### OCR: Cross-Dataset Concordance and Historical Benchmarking

**Strategy 1: Cross-dataset comparison**
- Compare OCR risk scores against Scott et al. 2024 (WRC) and CAL FIRE Fire Hazard Severity Zones
- Metric: Kendall's Tau concordance at census-tract level
- California results (13.7M buildings):
  - OCR vs. Scott RPS: Tau = 0.299
  - OCR vs. CAL FIRE: Tau = 0.151
  - Scott vs. CAL FIRE: Tau = 0.157

**Strategy 2: Historical burn probability benchmarking**
- Compare BP estimates against 70+ years of fire perimeter data
- Adapted from Moran et al. 2025 (Nature Scientific Reports)
- Examines CDF separation between historically burned vs. unburned pixels

**Strategy 3: Building-level CONUS comparison**
- 156M buildings compared between OCR and WRC
- Average census-tract correlation: 0.79
- Median bias: 0.00019
- Key divergence drivers: (1) wind effect (directional vs. uniform spreading), (2) development effect (WRC reduces BP in developed areas)

**Strengths:** Validates against independent datasets at massive scale. Identifies systematic biases.

### Why This Matters

The validation approaches are complementary. burn_probs validates against *ground truth* (did fire actually burn here?) while OCR validates against *expert consensus* (do other models agree?). burn_probs can directly measure predictive accuracy but only at its limited geographic scope. OCR can assess consistency across CONUS but cannot directly measure prediction accuracy since it doesn't simulate individual fires.

Notably, the moderate Kendall's Tau values in OCR's California comparison (0.15-0.30) suggest meaningful disagreement between all fire risk datasets, highlighting how much uncertainty remains in this domain regardless of approach.

---

## 11. Computational Architecture

### burn_probs

- **Parallelization:** Python multiprocessing across CPU cores (28 typical)
- **Scaling unit:** Monte Carlo iterations divided across workers
- **Infrastructure:** Single machine (desktop/server), 32GB RAM
- **Processing time:** ~3 hours for Sonoma County pilot
- **Statewide:** ~200 HPC tiles, ~800 core-hours total

### OCR

- **Parallelization:** Coiled/Dask distributed computing
- **Scaling unit:** 799 spatial regions (6000x4500 pixels each) at 30m
- **Infrastructure:** Cloud-based (S3-backed Icechunk storage)
- **Output format:** Zarr (raster) + GeoParquet (buildings)
- **Fault tolerance:** Region-level; failed chunks reprocess independently

OCR is designed for cloud-native CONUS-scale processing from the ground up. burn_probs is designed for single-machine execution with optional HPC tiling for statewide runs.

---

## 12. Output Comparison

### burn_probs Output Schema (per parcel)

| Field | Type | Description |
|-------|------|-------------|
| `burn_prob_mean` | float | Primary metric: annual burn probability (0-1) |
| `burn_prob_max` | float | Worst-case cell in parcel |
| `burn_prob_min` | float | Best-case cell in parcel |
| `burn_prob_std` | float | Uncertainty within parcel |
| `burn_prob_median` | float | Robust central tendency |
| `burn_prob_p05` | float | 5th percentile (optional) |
| `burn_prob_p95` | float | 95th percentile (optional) |
| `dominant_fuel` | string | Fuel model code |
| `fhsz` | string | Fire Hazard Severity Zone |
| `wui_class` | string | Wildland-Urban Interface classification |

### OCR Output Schema (per building)

| Field | Type | Description |
|-------|------|-------------|
| `rps_2011` | float | Annual RPS for ~2011 climate (%) |
| `rps_2047` | float | Annual RPS for ~2047 projected climate (%) |
| `bp_2011` | float | Wind-adjusted burn probability (2011) |
| `bp_2047` | float | Wind-adjusted burn probability (2047) |
| `rps_scott` | float | Scott et al. 2024 RPS reference |
| `crps_scott` | float | Conditional risk from Scott et al. 2024 |
| `bp_2011_riley` | float | Original Riley et al. BP (before wind adjustment) |
| `bp_2047_riley` | float | Riley et al. projected BP |
| `score` | int | Categorical 0-10 risk score |

OCR provides both current and projected climate scenarios, plus the original upstream values for comparison. burn_probs provides richer per-parcel statistics (min/max/std/percentiles) and contextual attributes (fuel model, FHSZ, WUI class).

---

## 13. Strengths and Limitations Summary

### burn_probs Strengths
- **Mechanistic transparency:** Every component (ignition, spread, weather) is explicitly modeled and inspectable
- **Scenario flexibility:** Can test fuel treatments, climate projections, or altered ignition patterns by changing inputs
- **Direct validation:** Tests against actual fire outcomes, not just dataset agreement
- **Uncertainty quantification:** Bayesian confidence intervals from Monte Carlo sampling
- **Ignition modeling:** Explicit ML model for where fires start, with feature importance
- **Independence:** Not dependent on external federal products; can be updated on any schedule

### burn_probs Limitations (Relative to FSim)
- **Simpler fire growth:** Cellular automaton vs. FSim's MTT algorithm (less realistic perimeters)
- **No crown fire:** Surface fire only, while FSim models crown fire initiation and spread
- **8-hour simulation cap:** FSim simulates multi-day fires; burn_probs' cap systematically underestimates large fire events
- **No suppression modeling:** Over-predicts fire extent; FSim's perimeter-trimming produces more realistic fire sizes
- **Simpler weather sampling:** Empirical day sampling (~2,250 days) vs. FSim's 10k-50k synthetic years exploring the full weather distribution
- **Simpler calibration:** Single scalar factor vs. FSim's iterative fire-size-distribution matching
- **No fire intensity output:** Does not produce flame-length probabilities needed for structural risk assessment
- **Limited geographic scope:** California only (Sonoma pilot); no CONUS coverage
- **No future climate projections:** Current conditions only (FSim/OCR have circa 2047 projections)

### OCR Strengths
- **Leverages best available federal science:** FSim + WRC represent decades of USFS research and calibration
- **CONUS-scale coverage:** 156M buildings, 799 processing regions
- **30m resolution:** Finer spatial detail for building-level risk
- **Risk metric (RPS):** Combines probability and consequence via cRPS, more actuarially relevant than BP alone
- **Climate projections:** Both ~2011 and ~2047 scenarios from Riley et al.
- **Wind-informed spreading:** Directional spreading into developed areas is a meaningful innovation over raw FSim
- **Cloud-native architecture:** Scalable, fault-tolerant, modern data formats (Icechunk, GeoParquet)
- **Transparency:** Open source, open data, reproducible

### OCR Limitations
- **Fully dependent on upstream products:** Cannot independently model fire behavior; inherits all FSim assumptions and limitations
- **No fire simulation:** Wind blurring is a spatial heuristic, not a physical model; cannot capture terrain-driven spread dynamics
- **No ignition model:** Cannot investigate what drives fire starts or test ignition scenarios
- **No independent calibration:** If FSim's calibration is wrong for a region, OCR inherits that error
- **No direct validation against fire outcomes:** Cannot test "did the model predict where fires actually burned?" -- validates only against other models
- **Static blurring distance:** ~1.5 km may underestimate spread in extreme wind events (e.g., Santa Ana winds pushing fire >5 km into developed areas) or overestimate in calm conditions
- **No uncertainty quantification:** Single point estimates, no confidence intervals
- **BP resolution mismatch:** Advertises 30m resolution, but the underlying burn probability is 270m upscaled; true 30m detail comes only from the cRPS layer

---

## 14. Complementary Use Cases

The two approaches are more complementary than competitive:

| Use case | Better fit |
|----------|-----------|
| CONUS-wide screening of building risk | OCR |
| Local/regional deep-dive with validation | burn_probs |
| Insurance pricing (probability x severity) | OCR (has cRPS) |
| Fuel treatment scenario analysis | burn_probs |
| Climate change projections | OCR (has 2047 scenario) |
| Understanding fire behavior drivers | burn_probs |
| Rapid deployment to new regions | OCR |
| Regulatory/actuarial defensibility | burn_probs (mechanistic model + direct validation) |
| WUI risk where FSim assigns zero BP | OCR (wind-adjusted spreading) |

A combined approach might use OCR for initial CONUS screening and burn_probs for detailed local analysis where mechanistic understanding and scenario testing are needed.

---

## 15. Data Source Lineage

All three systems share a common ancestry through LANDFIRE and Rothermel, but diverge in how they use these foundations:

```
LANDFIRE 2020 Fuel & Vegetation (USGS/USFS)
    |
    +---> LANDFIRE FBFM40 + Canopy layers
    |         |
    |         +---> FSim (USFS, Finney et al. 2011)
    |         |       |--- Synthetic weather (autoregressive ERC, 10k-50k years)
    |         |       |--- Logistic regression ignition (ERC >= 80th percentile)
    |         |       |--- MTT fire growth (Rothermel surface + crown fire)
    |         |       |--- Statistical suppression model
    |         |       |--- Iterative calibration (fire size dist + mean BP)
    |         |       |
    |         |       +---> Riley et al. 2025: BP + FLP rasters (270m, CONUS)
    |         |                 |
    |         |                 +---> BP (270m) ---------> OCR wind-adjusted BP (30m)
    |         |                 |                               |
    |         |                 +---> FLP (6 classes) ---> Scott et al. 2024
    |         |                                              |--- Structure response function
    |         |                                              |--- cRPS (30m) ---> OCR
    |         |                                              |                     |
    |         |                                              +---> RPS = BP x cRPS |
    |         |                                                    (OCR final output)
    |         |
    |         +---> burn_probs (independent implementation)
    |                 |--- GridMET weather sampling (empirical, ~2,250 fire-season days)
    |                 |--- Random Forest ignition model (10+ features, all fires)
    |                 |--- Cellular automaton fire growth (Rothermel surface only)
    |                 |--- No suppression, no crown fire
    |                 |--- Scalar calibration (observed / predicted annual rate)
    |                 |
    |                 +---> Annual burn probability (270m, California)
    |
    +---> USGS 3DEP terrain ---> burn_probs (slope/aspect in spread + ignition features)
                                 FSim (embedded in landscape file)

Weather Sources:
  RAWS stations ---> FireFamilyPlus ---> FSim (synthetic weather generation)
  GridMET (U. Idaho, 4km daily) -------> burn_probs (direct spread simulation input)
  CONUS404 (NCAR, 4km hourly) ---------> OCR (fire-weather wind direction for blurring)

Building Sources:
  OSM (via Overpass API) ---------------> burn_probs (464k Sonoma parcels)
  Overture Maps Foundation -------------> OCR (156M CONUS buildings)

Validation Sources:
  CAL FIRE FRAP perimeters -------------> burn_probs (training, calibration, holdout testing)
  US historical fire perimeters --------> OCR (benchmarking BP), FSim (calibration target)
  Scott et al. 2024 / CAL FIRE --------> OCR (cross-dataset concordance)
  Moran et al. 2025 -------------------> FSim (benchmarked 2020-2023 CA fires)
```

### Key References

- **Finney, M.A. et al. (2011).** A simulation of probabilistic wildfire risk components for the continental United States. *Stochastic Environmental Research and Risk Assessment*, 25: 973-1000. (FSim foundational paper)
- **Finney, M.A. (2002).** Fire growth using minimum travel time methods. *Canadian Journal of Forest Research*, 32(8): 1420-1424. (MTT algorithm)
- **Riley, K.L. et al. (2025).** Spatial datasets of probabilistic wildfire risk components for the conterminous United States (270m). USFS Research Data Archive. DOI: 10.2737/RDS-2025-0006
- **Scott, J.H. et al. (2024).** Wildfire Risk to Communities: Spatial datasets of landscape-wide wildfire risk components for the United States, 2nd edition. USFS Research Data Archive. DOI: 10.2737/RDS-2020-0016-2
- **Moran, C.J. et al. (2025).** Benchmarking performance of annual burn probability modeling against subsequent wildfire activity in California. *Nature Scientific Reports*. DOI: 10.1038/s41598-025-07968-6
- **Rasmussen, R. et al. (2023).** CONUS404: The NCAR-USGS 4-km long-term regional hydroclimate reanalysis. DOI: 10.1175/BAMS-D-21-0326.1
- **Rothermel, R.C. (1972).** A mathematical model for predicting fire spread in wildland fuels. USFS Research Paper INT-115.
- **Van Wagner, C.E. (1977).** Conditions for the start and spread of crown fire. *Canadian Journal of Forest Research*, 7(1): 23-34.
