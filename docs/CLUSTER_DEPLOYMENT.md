# California Statewide Burn Probability Model
## SLURM Cluster Deployment Guide

This guide explains how to run the burn probability model for all of California at 30m resolution using a SLURM-based HPC cluster.

---

## Overview

### Scale of the Problem

| Parameter | Value |
|-----------|-------|
| Geographic extent | California (~424,000 km²) |
| Resolution | 30 meters |
| Total grid cells | ~470 million |
| Tiles (100km × 100km) | 56 tiles |
| Fire ignitions (2015-2022) | ~3,200 statewide |
| Weather samples per ignition | 100 |
| Total simulations | ~320,000 |
| Parcels/buildings | ~15 million |

### Computational Requirements

| Resource | Per Tile | Total |
|----------|----------|-------|
| CPU time | 2-4 hours | 100-200 node-hours |
| Memory | 32-64 GB | - |
| Storage (input) | 50 GB | 50 GB (shared) |
| Storage (output) | 2 GB | 120 GB |
| Recommended nodes | 1 | 28-56 concurrent |

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 0: Setup                                                 │
│  - Clone repository                                             │
│  - Create conda environment                                     │
│  - Configure paths                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: Data Download (1 job, ~2 hours)                       │
│  - Download LANDFIRE fuel, canopy, topography                   │
│  - Download fire perimeters from CAL FIRE                       │
│  - Download building footprints                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: Preprocessing (1 job, ~1 hour)                        │
│  - Mosaic LANDFIRE tiles                                        │
│  - Create California-wide landscape raster                      │
│  - Generate tile boundaries with buffers                        │
│  - Split fire perimeters by tile                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: Monte Carlo Simulation (56 jobs, ~3 hours each)       │
│  - SLURM job array: one job per tile                            │
│  - Each tile: extract ignitions, run simulations, save results  │
│  - Embarrassingly parallel                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: Merge & Aggregate (1 job, ~2 hours)                   │
│  - Merge tile rasters (handle overlaps)                         │
│  - Aggregate to parcels statewide                               │
│  - Generate summary statistics                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 5: Validation (1 job, ~30 min)                           │
│  - Compute AUC, calibration metrics                             │
│  - Generate validation report                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_ORG/burn_probs.git
cd burn_probs

# 2. Set up environment (see detailed instructions below)
./cluster/setup_environment.sh

# 3. Configure paths for your cluster
cp cluster/config.template.yaml cluster/config.yaml
vim cluster/config.yaml  # Edit paths

# 4. Submit the full pipeline
./cluster/submit_pipeline.sh

# 5. Monitor progress
./cluster/monitor_jobs.sh
```

---

## Detailed Instructions

### Step 0: Clone and Setup

#### 0.1 Clone the Repository

```bash
# On the cluster login node
cd /scratch/$USER  # or your project directory
git clone https://github.com/YOUR_ORG/burn_probs.git
cd burn_probs
```

#### 0.2 Create Conda Environment

```bash
# Load conda module (cluster-specific)
module load anaconda3  # or: module load miniconda

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate burn_probs
```

If `environment.yml` doesn't exist, create it:

```bash
conda create -n burn_probs python=3.11 \
    geopandas rasterio xarray numpy pandas \
    scikit-learn pyproj shapely fiona \
    dask distributed joblib tqdm pyyaml \
    -c conda-forge

conda activate burn_probs
pip install rasterstats
```

#### 0.3 Configure Cluster Paths

Copy and edit the configuration file:

```bash
cp cluster/config.template.yaml cluster/config.yaml
```

Edit `cluster/config.yaml`:

```yaml
# Cluster configuration
cluster:
  name: "your_cluster_name"
  partition: "normal"           # SLURM partition name
  account: "your_account"       # SLURM account/allocation
  email: "you@university.edu"   # For job notifications

# Storage paths
paths:
  project_root: "/scratch/users/$USER/burn_probs"
  data_raw: "/scratch/users/$USER/burn_probs/data/raw"
  data_processed: "/scratch/users/$USER/burn_probs/data/processed"
  output: "/scratch/users/$USER/burn_probs/data/output"
  logs: "/scratch/users/$USER/burn_probs/logs"

# Model parameters
model:
  resolution: 30                 # meters
  tile_size: 100000              # meters (100 km)
  buffer_size: 10000             # meters (10 km overlap)
  weather_samples: 100           # per ignition
  simulation_hours: 24           # fire duration
  extreme_weather_fraction: 0.15
  start_year: 2015
  end_year: 2022

# Resource allocation
resources:
  cpus_per_task: 16
  memory_gb: 64
  time_limit: "06:00:00"
```

#### 0.4 Create Directory Structure

```bash
./cluster/setup_directories.sh
```

This creates:
```
burn_probs/
├── data/
│   ├── raw/
│   │   ├── landfire/
│   │   ├── fire_history/
│   │   └── buildings/
│   ├── processed/
│   │   ├── landscape/
│   │   └── tiles/
│   └── output/
│       ├── tiles/
│       ├── merged/
│       └── validation/
├── logs/
│   ├── download/
│   ├── preprocess/
│   ├── simulate/
│   ├── merge/
│   └── validate/
└── cluster/
```

---

### Step 1: Download Data

#### 1.1 Submit Download Job

```bash
sbatch cluster/jobs/01_download_data.slurm
```

This downloads:
- LANDFIRE fuel model, canopy, topography (30m, ~30 GB)
- CAL FIRE fire perimeters (~500 MB)
- Building footprints from OSM (~5 GB)

#### 1.2 Monitor Download Progress

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/download/download_$(date +%Y%m%d).log
```

#### 1.3 Verify Downloads

```bash
python scripts/verify_downloads.py
```

Expected output:
```
✓ LANDFIRE fuel model: 47.2 GB
✓ LANDFIRE canopy: 12.3 GB
✓ LANDFIRE topography: 8.1 GB
✓ Fire perimeters: 487 MB (22,810 fires)
✓ Building footprints: 4.8 GB (14.2M buildings)
All downloads complete.
```

---

### Step 2: Preprocess Data

#### 2.1 Submit Preprocessing Job

```bash
sbatch cluster/jobs/02_preprocess.slurm
```

This step:
1. Mosaics LANDFIRE tiles into a single California raster
2. Creates the 6-band landscape file at 30m
3. Generates 56 tile boundaries (100km × 100km with 10km buffer)
4. Clips fire perimeters to each tile
5. Splits fires into 70% train / 30% holdout

#### 2.2 Verify Preprocessing

```bash
python scripts/verify_preprocessing.py
```

Expected output:
```
✓ Landscape raster: 15,234 × 8,891 pixels (30m)
✓ Tile definitions: 56 tiles
✓ Fire perimeters split: 2,269 train, 947 holdout
✓ Ignition points extracted: 947 burnable locations
Ready for simulation.
```

---

### Step 3: Run Monte Carlo Simulations

This is the main computational step, running as a SLURM job array.

#### 3.1 Submit Simulation Job Array

```bash
sbatch cluster/jobs/03_simulate.slurm
```

This submits 56 jobs (one per tile) that run in parallel based on cluster availability.

#### 3.2 Monitor Progress

```bash
# Overall status
squeue -u $USER

# Detailed progress
./cluster/monitor_simulations.sh

# Example output:
# Tile 01/56: COMPLETED (2.3 hours, 1,247 simulations)
# Tile 02/56: RUNNING   (1.5 hours elapsed, 67% complete)
# Tile 03/56: PENDING
# ...
# Overall: 12/56 complete, 8 running, 36 pending
```

#### 3.3 Handle Failed Jobs

If any tiles fail:

```bash
# Check which tiles failed
./cluster/check_failures.sh

# Resubmit failed tiles only
./cluster/resubmit_failed.sh
```

---

### Step 4: Merge Results

After all simulation jobs complete:

#### 4.1 Submit Merge Job

```bash
sbatch cluster/jobs/04_merge.slurm
```

This step:
1. Merges 56 tile rasters into a single California raster
2. Handles buffer overlaps (average values in overlap zones)
3. Aggregates burn probability to all ~15M parcels
4. Generates county-level summaries

#### 4.2 Verify Merge

```bash
python scripts/verify_merge.py
```

Expected output:
```
✓ Merged raster: 15,234 × 8,891 pixels
✓ Coverage: 100% of California
✓ No gaps between tiles
✓ Parcel aggregation: 14,892,341 parcels
✓ Output size: 2.3 GB (raster) + 45 GB (parcels)
```

---

### Step 5: Validation

#### 5.1 Submit Validation Job

```bash
sbatch cluster/jobs/05_validate.slurm
```

This computes:
- Statewide AUC-ROC
- Calibration by probability bin
- Per-county validation metrics
- Generates validation report (HTML + PDF)

#### 5.2 View Results

```bash
# Validation summary
cat data/output/validation/summary.txt

# Full report
firefox data/output/validation/validation_report.html
```

---

## File Descriptions

### SLURM Job Scripts

| File | Purpose |
|------|---------|
| `cluster/jobs/01_download_data.slurm` | Download all input data |
| `cluster/jobs/02_preprocess.slurm` | Create landscape, define tiles |
| `cluster/jobs/03_simulate.slurm` | Job array for Monte Carlo (56 tasks) |
| `cluster/jobs/04_merge.slurm` | Merge tiles, aggregate to parcels |
| `cluster/jobs/05_validate.slurm` | Compute validation metrics |

### Python Scripts

| File | Purpose |
|------|---------|
| `scripts/download_landfire.py` | Download LANDFIRE data |
| `scripts/download_fire_perimeters.py` | Download CAL FIRE data |
| `scripts/download_buildings.py` | Download OSM buildings |
| `scripts/create_landscape.py` | Build 6-band landscape raster |
| `scripts/create_tiles.py` | Generate tile boundaries |
| `scripts/run_tile_simulation.py` | Run MC for single tile |
| `scripts/merge_tiles.py` | Merge tile outputs |
| `scripts/aggregate_parcels.py` | Aggregate to parcels |
| `scripts/validate_statewide.py` | Compute validation metrics |

### Helper Scripts

| File | Purpose |
|------|---------|
| `cluster/setup_environment.sh` | Create conda environment |
| `cluster/setup_directories.sh` | Create directory structure |
| `cluster/submit_pipeline.sh` | Submit all jobs with dependencies |
| `cluster/monitor_jobs.sh` | Check job status |
| `cluster/monitor_simulations.sh` | Detailed simulation progress |
| `cluster/check_failures.sh` | List failed jobs |
| `cluster/resubmit_failed.sh` | Resubmit failed jobs |

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory

```
slurmstepd: error: Detected 1 oom-kill event(s)
```

**Solution:** Increase memory in config.yaml:
```yaml
resources:
  memory_gb: 128  # Increase from 64
```

Then resubmit failed jobs.

#### 2. Job Timeout

```
TIMEOUT: Job exceeded time limit
```

**Solution:** Increase time limit or reduce tile size:
```yaml
resources:
  time_limit: "12:00:00"  # Increase from 6 hours
```

#### 3. Missing Input Files

```
FileNotFoundError: landscape_tile_23.tif not found
```

**Solution:** Check preprocessing completed:
```bash
ls data/processed/tiles/
python scripts/verify_preprocessing.py
```

#### 4. SLURM Account Issues

```
sbatch: error: Batch job submission failed: Invalid account
```

**Solution:** Check your account name:
```bash
sacctmgr show associations user=$USER
```
Update `cluster/config.yaml` with correct account.

### Getting Help

1. Check logs: `logs/<stage>/<jobid>.out` and `logs/<stage>/<jobid>.err`
2. Run verification scripts to identify issues
3. For cluster-specific issues, contact your HPC support team

---

## Output Files

After successful completion:

```
data/output/
├── merged/
│   ├── burn_probability_california_30m.tif      # 2.3 GB - main output
│   ├── burn_probability_california_30m.vrt      # Virtual raster (for quick viewing)
│   └── metadata.json                            # Processing metadata
├── parcels/
│   ├── california_parcels_burn_prob.parquet     # 45 GB - all parcels
│   ├── california_parcels_summary.csv           # 1.2 GB - summary stats
│   └── by_county/                               # County-level files
│       ├── alameda_parcels.parquet
│       ├── alpine_parcels.parquet
│       └── ...
├── validation/
│   ├── summary.txt                              # Quick stats
│   ├── validation_report.html                   # Full report
│   ├── calibration_plot.png                     # Calibration curve
│   └── roc_curve.png                            # ROC curve
└── tiles/                                       # Intermediate tile outputs
    ├── tile_001.tif
    ├── tile_002.tif
    └── ...
```

---

## Estimated Timeline

| Stage | Duration | Notes |
|-------|----------|-------|
| Setup | 30 min | One-time setup |
| Download | 2-3 hours | Depends on network |
| Preprocess | 1 hour | Single node |
| Simulate | 3-4 hours | With 56 concurrent nodes |
| Merge | 2 hours | Single node, I/O intensive |
| Validate | 30 min | Single node |
| **Total** | **8-10 hours** | Wall clock time |

With limited node availability, simulation stage may take longer (up to 24 hours if only 8-10 nodes available).

---

## Scaling Considerations

### Adjusting Resolution

To run at different resolutions, modify `cluster/config.yaml`:

```yaml
model:
  resolution: 90   # 90m instead of 30m = 9× faster
```

### Adjusting Weather Samples

For faster runs (at cost of precision):
```yaml
model:
  weather_samples: 20   # Instead of 100
```

### Regional Runs

To run a single county instead of statewide:
```bash
python scripts/run_tile_simulation.py --county "Los Angeles"
```

---

## Citation

If you use this pipeline, please cite:

```bibtex
@software{burn_probs,
  title = {California Wildfire Burn Probability Model},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/YOUR_ORG/burn_probs}
}
```

---

*Last updated: February 2024*
