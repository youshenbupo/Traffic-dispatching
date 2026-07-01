#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
gpu="${2:?gpu required}"
cd /data/users/jyh/light_RL
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl

mkdir -p results/hangzhou_fg_pdd_v2

python scripts/eval.py \
    --config configs/mappo_hangzhou4x4_robust_eval.yaml \
    --model_path "logs/mappo_hangzhou4x4_base30_s${seed}/final_model" \
    --gpus "$gpu" --num_episodes 5 --seed_start 52000 \
    --output "results/hangzhou_fg_pdd_v2/base30_s${seed}.json" \
    > "logs/hz_base30_metric_eval_s${seed}.log" 2>&1

python scripts/eval.py \
    --config configs/fg_pdd_hangzhou4x4_eval.yaml \
    --model_path "logs/fg_pdd_hangzhou4x4_s${seed}/final_model" \
    --gpus "$gpu" --num_episodes 5 --seed_start 52000 \
    --output "results/hangzhou_fg_pdd_v2/fg_s${seed}.json" \
    > "logs/hz_fg_metric_eval_s${seed}.log" 2>&1

echo "Hangzhou metric reevaluation completed for seed ${seed}."
