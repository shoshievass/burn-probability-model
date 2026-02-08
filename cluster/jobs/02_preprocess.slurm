#!/bin/bash
#SBATCH --job-name=burn_preprocess
#SBATCH --output=logs/preprocess/%x_%j.out
#SBATCH --error=logs/preprocess/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Stage 2: Preprocess Data
# ==============================================================================
# Creates processed data ready for simulation:
#   - Mosaics LANDFIRE tiles into single California raster
#   - Creates 6-band landscape file
#   - Generates tile boundaries (100km × 100km with 10km buffer)
#   - Extracts ignition points from fire perimeters
#   - Splits fires into train/holdout sets
# ==============================================================================

set -euo pipefail

echo "=========================================="
echo "BURN PROBABILITY MODEL - PREPROCESSING"
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
mkdir -p "${DATA_PROCESSED}/landscape"
mkdir -p "${DATA_PROCESSED}/tiles"
mkdir -p "${DATA_PROCESSED}/fires"

# ==============================================================================
# Create statewide landscape raster
# ==============================================================================
echo "[1/4] Creating California landscape raster..."

python scripts/create_landscape.py \
    --landfire-dir "${DATA_RAW}/landfire" \
    --output "${DATA_PROCESSED}/landscape/california_landscape_30m.tif" \
    --resolution ${RESOLUTION} \
    --crs "EPSG:3310"

echo "Landscape raster complete."

# ==============================================================================
# Generate tile definitions
# ==============================================================================
echo "[2/4] Generating tile boundaries..."

python scripts/create_tiles.py \
    --landscape "${DATA_PROCESSED}/landscape/california_landscape_30m.tif" \
    --tile-size ${TILE_SIZE} \
    --buffer-size ${BUFFER_SIZE} \
    --output-dir "${DATA_PROCESSED}/tiles"

N_TILES=$(ls -1 ${DATA_PROCESSED}/tiles/tile_*.gpkg 2>/dev/null | wc -l)
echo "Generated ${N_TILES} tiles."

# ==============================================================================
# Process fire perimeters
# ==============================================================================
echo "[3/4] Processing fire perimeters..."

python scripts/process_fires.py \
    --input "${DATA_RAW}/fire_history/fire_perimeters.parquet" \
    --landscape "${DATA_PROCESSED}/landscape/california_landscape_30m.tif" \
    --output-dir "${DATA_PROCESSED}/fires" \
    --start-year ${START_YEAR} \
    --end-year ${END_YEAR} \
    --holdout-fraction ${HOLDOUT_FRACTION} \
    --seed ${RANDOM_SEED}

echo "Fire processing complete."

# ==============================================================================
# Create tile-specific fire lists
# ==============================================================================
echo "[4/4] Assigning fires to tiles..."

python scripts/assign_fires_to_tiles.py \
    --fires "${DATA_PROCESSED}/fires/holdout_fires.parquet" \
    --tiles-dir "${DATA_PROCESSED}/tiles" \
    --output-dir "${DATA_PROCESSED}/tiles"

echo "Fire assignment complete."

# ==============================================================================
# Verify preprocessing
# ==============================================================================
echo ""
echo "Verifying preprocessing..."
python scripts/verify_preprocessing.py

echo ""
echo "=========================================="
echo "PREPROCESSING COMPLETE"
echo "Finished: $(date)"
echo "=========================================="
