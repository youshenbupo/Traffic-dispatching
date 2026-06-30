#!/bin/bash
# Run a suite of MARL robustness/transfer experiments sequentially.
set -e
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl
export CUDA_VISIBLE_DEVICES=2,3

EXPS=(
    "configs/mappo_grid4x4_obsdropout_p0.1.yaml mappo_grid4x4_obsdropout_p0.1"
    "configs/mappo_grid4x4_obsdropout_p0.3.yaml mappo_grid4x4_obsdropout_p0.3"
    "configs/mappo_grid4x4_weak_combined.yaml mappo_grid4x4_weak_combined"
    "configs/mappo_grid4x4_demand_spike.yaml mappo_grid4x4_demand_spike"
    "configs/mappo_hangzhou4x4_transfer_grid.yaml mappo_hangzhou4x4_transfer_grid --pretrained_model logs/mappo_grid4x4_obsdropout/best_model/mappo.pth"
)

for entry in "${EXPS[@]}"; do
    read -r config exp_name extra <<< "$entry"
    echo "===================================================================="
    echo "Running $exp_name"
    echo "===================================================================="
    python scripts/train.py --config "$config" --exp_name "$exp_name" $extra
    echo ""
done

echo "Suite complete."
