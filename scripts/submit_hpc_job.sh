#!/bin/bash
#SBATCH --job-name=burn_prob
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=1-200%50
#SBATCH --output=logs/burn_prob_%A_%a.out
#SBATCH --error=logs/burn_prob_%A_%a.err

# ============================================================================
# HPC Job Script for Burn Probability Monte Carlo Simulation
# ============================================================================
#
# This script runs burn probability simulations across California tiles.
# It's designed for SLURM-based HPC clusters.
#
# Usage:
#   sbatch submit_hpc_job.sh [year]
#
# The --array parameter controls parallel tile processing.
# Adjust %50 to control max concurrent jobs.
# ============================================================================

set -e

# Load required modules (adjust for your cluster)
module load python/3.11 2>/dev/null || true
module load gdal/3.6 2>/dev/null || true
module load proj/9.0 2>/dev/null || true

# Configuration
YEAR=${1:-2020}
N_ITERATIONS=10000
EXTREME_FRACTION=0.15

# Paths (adjust for your cluster)
PROJECT_DIR="${HOME}/burn_probs"
DATA_DIR="${PROJECT_DIR}/data"
OUTPUT_DIR="${DATA_DIR}/output/hpc/${YEAR}"
LOG_DIR="${PROJECT_DIR}/logs"

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Tile ID from SLURM array
TILE_ID=${SLURM_ARRAY_TASK_ID}

echo "============================================"
echo "Burn Probability HPC Job"
echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Year: ${YEAR}"
echo "Tile: ${TILE_ID}"
echo "Iterations: ${N_ITERATIONS}"
echo "============================================"

# Activate conda environment
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate burn_probs

# Navigate to project
cd "${PROJECT_DIR}"

# Run tile simulation
python -u scripts/run_tile_simulation.py \
    --tile-id ${TILE_ID} \
    --year ${YEAR} \
    --iterations ${N_ITERATIONS} \
    --extreme-fraction ${EXTREME_FRACTION} \
    --output-dir "${OUTPUT_DIR}/tiles"

echo "============================================"
echo "Tile ${TILE_ID} complete!"
echo "============================================"
