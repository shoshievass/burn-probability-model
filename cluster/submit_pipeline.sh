#!/bin/bash
# ==============================================================================
# Submit Full Pipeline
# ==============================================================================
# Submits all pipeline stages with proper dependencies.
# Each stage waits for the previous stage to complete before starting.
# ==============================================================================

set -euo pipefail

echo "=========================================="
echo "BURN PROBABILITY MODEL - PIPELINE SUBMISSION"
echo "=========================================="
echo ""

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_config.sh"

# Create log directories
mkdir -p "${LOGS}/download"
mkdir -p "${LOGS}/preprocess"
mkdir -p "${LOGS}/simulate"
mkdir -p "${LOGS}/merge"
mkdir -p "${LOGS}/validate"

# Change to project root
cd "${PROJECT_ROOT}"

# ==============================================================================
# Parse arguments
# ==============================================================================
SKIP_DOWNLOAD=false
SKIP_PREPROCESS=false
START_STAGE=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-preprocess)
            SKIP_PREPROCESS=true
            shift
            ;;
        --start-stage)
            START_STAGE=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ==============================================================================
# Submit jobs with dependencies
# ==============================================================================

# Add account to sbatch if specified
SBATCH_OPTS=""
if [[ -n "${CLUSTER_ACCOUNT:-}" ]]; then
    SBATCH_OPTS="--account=${CLUSTER_ACCOUNT}"
fi

if [[ -n "${CLUSTER_PARTITION:-}" ]]; then
    SBATCH_OPTS="${SBATCH_OPTS} --partition=${CLUSTER_PARTITION}"
fi

LAST_JOB_ID=""

# Stage 1: Download
if [[ ${START_STAGE} -le 1 ]] && [[ ${SKIP_DOWNLOAD} == false ]]; then
    echo "Submitting Stage 1: Download..."
    JOB1=$(sbatch ${SBATCH_OPTS} cluster/jobs/01_download_data.slurm | awk '{print $4}')
    echo "  Job ID: ${JOB1}"
    LAST_JOB_ID=${JOB1}
else
    echo "Skipping Stage 1: Download"
fi

# Stage 2: Preprocess
if [[ ${START_STAGE} -le 2 ]] && [[ ${SKIP_PREPROCESS} == false ]]; then
    echo "Submitting Stage 2: Preprocess..."
    if [[ -n "${LAST_JOB_ID}" ]]; then
        JOB2=$(sbatch ${SBATCH_OPTS} --dependency=afterok:${LAST_JOB_ID} cluster/jobs/02_preprocess.slurm | awk '{print $4}')
    else
        JOB2=$(sbatch ${SBATCH_OPTS} cluster/jobs/02_preprocess.slurm | awk '{print $4}')
    fi
    echo "  Job ID: ${JOB2}"
    LAST_JOB_ID=${JOB2}
else
    echo "Skipping Stage 2: Preprocess"
fi

# Stage 3: Simulate (job array)
if [[ ${START_STAGE} -le 3 ]]; then
    echo "Submitting Stage 3: Simulate (56 tiles)..."
    if [[ -n "${LAST_JOB_ID}" ]]; then
        JOB3=$(sbatch ${SBATCH_OPTS} --dependency=afterok:${LAST_JOB_ID} cluster/jobs/03_simulate.slurm | awk '{print $4}')
    else
        JOB3=$(sbatch ${SBATCH_OPTS} cluster/jobs/03_simulate.slurm | awk '{print $4}')
    fi
    echo "  Job Array ID: ${JOB3}"
    LAST_JOB_ID=${JOB3}
fi

# Stage 4: Merge
if [[ ${START_STAGE} -le 4 ]]; then
    echo "Submitting Stage 4: Merge..."
    JOB4=$(sbatch ${SBATCH_OPTS} --dependency=afterok:${LAST_JOB_ID} cluster/jobs/04_merge.slurm | awk '{print $4}')
    echo "  Job ID: ${JOB4}"
    LAST_JOB_ID=${JOB4}
fi

# Stage 5: Validate
if [[ ${START_STAGE} -le 5 ]]; then
    echo "Submitting Stage 5: Validate..."
    JOB5=$(sbatch ${SBATCH_OPTS} --dependency=afterok:${LAST_JOB_ID} cluster/jobs/05_validate.slurm | awk '{print $4}')
    echo "  Job ID: ${JOB5}"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=========================================="
echo "PIPELINE SUBMITTED"
echo "=========================================="
echo ""
echo "Monitor progress with:"
echo "  squeue -u \$USER"
echo "  ./cluster/monitor_jobs.sh"
echo ""
echo "View logs in: ${LOGS}/"
echo ""
