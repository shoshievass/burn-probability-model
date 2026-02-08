#!/bin/bash
#SBATCH --job-name=burn_validate
#SBATCH --output=logs/validate/%x_%j.out
#SBATCH --error=logs/validate/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# Stage 5: Validation
# ==============================================================================
# Validates model against holdout fires:
#   - Computes statewide AUC-ROC
#   - Computes calibration by probability bin
#   - Generates per-county metrics
#   - Creates validation report
# ==============================================================================

set -euo pipefail

echo "=========================================="
echo "BURN PROBABILITY MODEL - VALIDATION"
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

# Create output directory
mkdir -p "${OUTPUT}/validation"

# ==============================================================================
# Run validation
# ==============================================================================
echo "Running validation against holdout fires..."

python scripts/validate_statewide.py \
    --raster "${OUTPUT}/merged/burn_probability_california_30m.tif" \
    --holdout-fires "${DATA_PROCESSED}/fires/holdout_fires.parquet" \
    --output-dir "${OUTPUT}/validation" \
    --generate-report

# ==============================================================================
# Print summary
# ==============================================================================
echo ""
echo "=========================================="
echo "VALIDATION RESULTS"
echo "=========================================="
cat "${OUTPUT}/validation/summary.txt"
echo ""
echo "Full report: ${OUTPUT}/validation/validation_report.html"
echo ""
echo "Finished: $(date)"
echo "=========================================="
