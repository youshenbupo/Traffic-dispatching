#!/bin/bash
# Convenience script to activate environment and set GPU restriction.
# Usage: source scripts/setup_env.sh [GPU_IDS]

CONDA_BASE="/opt/miniforge3"
ENV_NAME="light_rl"
GPU_IDS="${1:-2,3}"

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export SUMO_HOME="$(python -c 'import sumo; print(sumo.SUMO_HOME)' 2>/dev/null || echo '')"

echo "Activated conda env: $ENV_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "SUMO_HOME=$SUMO_HOME"
