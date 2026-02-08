#!/bin/bash
#SBATCH --job-name=burn_merge
#SBATCH --output=logs/merge/%x_%j.out
#SBATCH --error=logs/merge/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Stage 4: Merge Results
# ==============================================================================
# Merges tile outputs into statewide products:
#   - Merges 56 tile rasters into single California raster
#   - Handles buffer overlaps (averages values)
#   - Aggregates burn probability to parcels
#   - Generates county-level summaries
# ==============================================================================

set -euo pipefail

echo "=========================================="
echo "BURN PROBABILITY MODEL - MERGE"
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

# Create output directories
mkdir -p "${OUTPUT}/merged"
mkdir -p "${OUTPUT}/parcels"
mkdir -p "${OUTPUT}/parcels/by_county"

# ==============================================================================
# Check all tiles completed
# ==============================================================================
echo "[1/4] Checking tile completion..."

N_EXPECTED=56
N_COMPLETED=$(ls -1 ${OUTPUT}/tiles/tile_*_burn_probability.tif 2>/dev/null | wc -l)

if [[ ${N_COMPLETED} -lt ${N_EXPECTED} ]]; then
    echo "ERROR: Only ${N_COMPLETED}/${N_EXPECTED} tiles completed."
    echo "Run: ./cluster/check_failures.sh to identify missing tiles."
    exit 1
fi

echo "All ${N_COMPLETED} tiles complete."

# ==============================================================================
# Merge tile rasters
# ==============================================================================
echo "[2/4] Merging tile rasters..."

python scripts/merge_tiles.py \
    --tiles-dir "${OUTPUT}/tiles" \
    --output "${OUTPUT}/merged/burn_probability_california_30m.tif" \
    --buffer-size ${BUFFER_SIZE} \
    --method "average"

echo "Raster merge complete."

# ==============================================================================
# Aggregate to parcels
# ==============================================================================
echo "[3/4] Aggregating to parcels..."

python scripts/aggregate_parcels_statewide.py \
    --raster "${OUTPUT}/merged/burn_probability_california_30m.tif" \
    --buildings-dir "${DATA_RAW}/buildings" \
    --output "${OUTPUT}/parcels/california_parcels_burn_prob.parquet" \
    --summary "${OUTPUT}/parcels/california_parcels_summary.csv" \
    --by-county "${OUTPUT}/parcels/by_county" \
    --cores ${SLURM_CPUS_PER_TASK}

echo "Parcel aggregation complete."

# ==============================================================================
# Generate statistics
# ==============================================================================
echo "[4/4] Generating summary statistics..."

python scripts/generate_statistics.py \
    --raster "${OUTPUT}/merged/burn_probability_california_30m.tif" \
    --parcels "${OUTPUT}/parcels/california_parcels_burn_prob.parquet" \
    --output "${OUTPUT}/merged/statistics.json"

echo "Statistics complete."

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=========================================="
echo "MERGE COMPLETE"
echo ""
cat "${OUTPUT}/merged/statistics.json"
echo ""
echo "Output files:"
echo "  Raster: ${OUTPUT}/merged/burn_probability_california_30m.tif"
echo "  Parcels: ${OUTPUT}/parcels/california_parcels_burn_prob.parquet"
echo ""
echo "Finished: $(date)"
echo "=========================================="
