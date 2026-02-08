#!/bin/bash
# ==============================================================================
# Load configuration from config.yaml
# ==============================================================================
# This script is sourced by SLURM job scripts to load configuration variables.
# It parses the YAML config file and exports shell variables.
# ==============================================================================

# Find the cluster config file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.yaml"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: Configuration file not found: ${CONFIG_FILE}"
    echo "Please copy config.template.yaml to config.yaml and edit it."
    exit 1
fi

# Parse YAML using Python (more reliable than bash parsing)
eval $(python3 << EOF
import yaml
import os

with open("${CONFIG_FILE}") as f:
    config = yaml.safe_load(f)

# Expand $USER in paths
user = os.environ.get("USER", "")

def expand_path(path):
    return path.replace("\$USER", user)

# Export paths
paths = config.get("paths", {})
print(f'export PROJECT_ROOT="{expand_path(paths.get("project_root", ""))}"')
print(f'export DATA_RAW="{expand_path(paths.get("data_raw", ""))}"')
print(f'export DATA_PROCESSED="{expand_path(paths.get("data_processed", ""))}"')
print(f'export OUTPUT="{expand_path(paths.get("output", ""))}"')
print(f'export LOGS="{expand_path(paths.get("logs", ""))}"')

# Export model parameters
model = config.get("model", {})
print(f'export RESOLUTION={model.get("resolution", 30)}')
print(f'export TILE_SIZE={model.get("tile_size", 100000)}')
print(f'export BUFFER_SIZE={model.get("buffer_size", 10000)}')
print(f'export WEATHER_SAMPLES={model.get("weather_samples", 100)}')
print(f'export SIMULATION_HOURS={model.get("simulation_hours", 24)}')
print(f'export EXTREME_WEATHER_FRACTION={model.get("extreme_weather_fraction", 0.15)}')
print(f'export START_YEAR={model.get("start_year", 2015)}')
print(f'export END_YEAR={model.get("end_year", 2022)}')
print(f'export HOLDOUT_FRACTION={model.get("holdout_fraction", 0.30)}')
print(f'export RANDOM_SEED={model.get("random_seed", 42)}')

# Export cluster settings
cluster = config.get("cluster", {})
print(f'export CLUSTER_PARTITION="{cluster.get("partition", "normal")}"')
print(f'export CLUSTER_ACCOUNT="{cluster.get("account", "")}"')
EOF
)

# Verify variables were set
if [[ -z "${PROJECT_ROOT:-}" ]]; then
    echo "ERROR: Failed to parse configuration file."
    exit 1
fi

echo "Configuration loaded from ${CONFIG_FILE}"
