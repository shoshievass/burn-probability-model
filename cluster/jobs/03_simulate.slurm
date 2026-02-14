#!/bin/bash
#SBATCH --job-name=burn_simulate
#SBATCH --output=logs/simulate/%x_%A_%a.out
#SBATCH --error=logs/simulate/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --mail-type=END,FAIL
#SBATCH --array=1-56%20

# ==============================================================================
# Stage 3: Monte Carlo Simulation (Job Array)
# ==============================================================================
# Runs Monte Carlo burn probability simulation for each tile.
# This is a SLURM job array - each array task processes one tile.
#
# Array task ID corresponds to tile number (1-56).
# The %20 limits concurrent tasks to 20 (adjust based on cluster policy).
# ==============================================================================

set -euo pipefail

# Get tile number from array task ID
TILE_ID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})

echo "=========================================="
echo "BURN PROBABILITY MODEL - SIMULATION"
echo "=========================================="
echo "Job Array ID: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Tile: ${TILE_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Started: $(date)"
echo ""

# Load configuration
source cluster/load_config.sh

# Activate conda environment
module load anaconda3
conda activate burn_probs

# Define paths for this tile
TILE_DIR="${DATA_PROCESSED}/tiles"
TILE_LANDSCAPE="${TILE_DIR}/tile_${TILE_ID}_landscape.tif"
TILE_FIRES="${TILE_DIR}/tile_${TILE_ID}_fires.parquet"
OUTPUT_DIR="${OUTPUT}/tiles"
TILE_OUTPUT="${OUTPUT_DIR}/tile_${TILE_ID}_burn_probability.tif"

# GridMET weather data for empirical sampling
GRIDMET_PATH="${DATA_RAW}/weather/gridmet/gridmet_2010_2024.nc"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Check if tile has already been processed
if [[ -f "${TILE_OUTPUT}" ]]; then
    echo "Tile ${TILE_ID} already processed. Skipping."
    exit 0
fi

# Check if tile has fires
if [[ ! -f "${TILE_FIRES}" ]]; then
    echo "No fires in tile ${TILE_ID}. Creating empty output."
    python scripts/create_empty_tile.py \
        --landscape "${TILE_LANDSCAPE}" \
        --output "${TILE_OUTPUT}"
    exit 0
fi

# ==============================================================================
# Run Monte Carlo simulation for this tile
# ==============================================================================
echo "Running Monte Carlo simulation..."
echo "  Landscape: ${TILE_LANDSCAPE}"
echo "  Fires: ${TILE_FIRES}"
echo "  GridMET: ${GRIDMET_PATH}"
echo "  Weather samples: ${WEATHER_SAMPLES}"
echo "  Simulation hours: ${SIMULATION_HOURS}"
echo "  Extreme fraction: ${EXTREME_WEATHER_FRACTION}"
echo ""

# Check if GridMET data exists for empirical weather sampling
if [[ -f "${GRIDMET_PATH}" ]]; then
    echo "Using EMPIRICAL weather sampling from historical GridMET data"
    GRIDMET_ARG="--gridmet ${GRIDMET_PATH}"
else
    echo "WARNING: GridMET data not found. Using DEFAULT weather scenarios."
    echo "         For empirical sampling, run: scripts/download_gridmet.py"
    GRIDMET_ARG=""
fi
echo ""

START_TIME=$(date +%s)

python scripts/run_tile_simulation.py \
    --tile-id ${TILE_ID} \
    --landscape "${TILE_LANDSCAPE}" \
    --fires "${TILE_FIRES}" \
    --output "${TILE_OUTPUT}" \
    ${GRIDMET_ARG} \
    --weather-samples ${WEATHER_SAMPLES} \
    --simulation-hours ${SIMULATION_HOURS} \
    --extreme-fraction ${EXTREME_WEATHER_FRACTION} \
    --cores ${SLURM_CPUS_PER_TASK} \
    --seed ${RANDOM_SEED}

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# ==============================================================================
# Log completion
# ==============================================================================
echo ""
echo "=========================================="
echo "TILE ${TILE_ID} COMPLETE"
echo "Elapsed time: ${ELAPSED} seconds ($(( ELAPSED / 60 )) minutes)"
echo "Output: ${TILE_OUTPUT}"
echo "Finished: $(date)"
echo "=========================================="

# Write completion marker
echo "${SLURM_ARRAY_TASK_ID},${ELAPSED},$(date -Iseconds)" >> "${OUTPUT_DIR}/completed_tiles.csv"
