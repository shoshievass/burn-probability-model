#!/bin/bash
# ==============================================================================
# Monitor Pipeline Jobs
# ==============================================================================
# Shows status of all pipeline jobs with a nice summary.
# ==============================================================================

echo "=========================================="
echo "BURN PROBABILITY MODEL - JOB STATUS"
echo "=========================================="
echo "Time: $(date)"
echo ""

# Get all jobs for this user
JOBS=$(squeue -u $USER --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R" 2>/dev/null)

if [[ -z "$JOBS" ]]; then
    echo "No active jobs found."
else
    echo "Active Jobs:"
    echo "$JOBS"
fi

echo ""
echo "=========================================="
echo "SIMULATION PROGRESS"
echo "=========================================="

# Check for completed tiles
TILES_DIR="${OUTPUT:-data/output}/tiles"
if [[ -d "$TILES_DIR" ]]; then
    N_COMPLETED=$(ls -1 ${TILES_DIR}/tile_*_burn_probability.tif 2>/dev/null | wc -l)
    N_EXPECTED=56
    PCT=$(echo "scale=1; $N_COMPLETED * 100 / $N_EXPECTED" | bc)

    echo "Tiles completed: ${N_COMPLETED}/${N_EXPECTED} (${PCT}%)"

    # Show progress bar
    FILLED=$((N_COMPLETED * 40 / N_EXPECTED))
    EMPTY=$((40 - FILLED))
    printf "["
    printf "%${FILLED}s" | tr ' ' '='
    printf "%${EMPTY}s" | tr ' ' '-'
    printf "]\n"
else
    echo "Output directory not found: ${TILES_DIR}"
fi

echo ""
echo "=========================================="
echo "RECENT LOG ENTRIES"
echo "=========================================="

# Show last few log entries from simulation jobs
LATEST_LOG=$(ls -t logs/simulate/*.out 2>/dev/null | head -1)
if [[ -n "$LATEST_LOG" ]]; then
    echo "Latest simulation log: ${LATEST_LOG}"
    tail -5 "$LATEST_LOG"
else
    echo "No simulation logs found yet."
fi

echo ""
echo "=========================================="
