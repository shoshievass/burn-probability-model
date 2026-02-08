#!/bin/bash
#SBATCH --job-name=burn_download
#SBATCH --output=logs/download/%x_%j.out
#SBATCH --error=logs/download/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Stage 1: Download Data
# ==============================================================================
# Downloads all required input data:
#   - LANDFIRE fuel model, canopy, topography (30m)
#   - CAL FIRE fire perimeters
#   - OpenStreetMap building footprints
# ==============================================================================

set -euo pipefail

echo "=========================================="
echo "BURN PROBABILITY MODEL - DATA DOWNLOAD"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Started: $(date)"
echo ""

# Load configuration
source cluster/load_config.sh

# Activate conda environment
module load anaconda3
conda activate burn_probs

# Create directories
mkdir -p "${DATA_RAW}/landfire"
mkdir -p "${DATA_RAW}/fire_history"
mkdir -p "${DATA_RAW}/buildings"

# ==============================================================================
# Download LANDFIRE data
# ==============================================================================
echo "[1/3] Downloading LANDFIRE data..."

python scripts/download_landfire.py \
    --output-dir "${DATA_RAW}/landfire" \
    --region california \
    --products FBFM40 CC CH ELEV SLP ASP \
    --resolution 30

echo "LANDFIRE download complete."

# ==============================================================================
# Download fire perimeters
# ==============================================================================
echo "[2/3] Downloading fire perimeters..."

python scripts/download_fire_perimeters.py \
    --output "${DATA_RAW}/fire_history/fire_perimeters.parquet" \
    --years 2000-2023

echo "Fire perimeters download complete."

# ==============================================================================
# Download building footprints
# ==============================================================================
echo "[3/3] Downloading building footprints..."

python scripts/download_buildings.py \
    --output-dir "${DATA_RAW}/buildings" \
    --region california \
    --source osm

echo "Building footprints download complete."

# ==============================================================================
# Verify downloads
# ==============================================================================
echo ""
echo "Verifying downloads..."
python scripts/verify_downloads.py

echo ""
echo "=========================================="
echo "DATA DOWNLOAD COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
