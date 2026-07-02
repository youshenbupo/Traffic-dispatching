#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
gpu="${2:?gpu required}"
cd /data/users/jyh/light_RL
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl

mkdir -p results/hangzhou_failure_suite

for condition in clean p01 p05 burst correlated; do
    case "$condition" in
        clean) seed_start=51000 ;;
        p01) seed_start=53000 ;;
        p05) seed_start=55000 ;;
        burst) seed_start=57000 ;;
        correlated) seed_start=59000 ;;
    esac

    python scripts/eval.py \
        --config "configs/mappo_hangzhou4x4_eval_${condition}.yaml" \
        --model_path "logs/mappo_hangzhou4x4_base30_s${seed}/final_model" \
        --gpus "$gpu" --num_episodes 5 --seed_start "$seed_start" \
        --output "results/hangzhou_failure_suite/base_s${seed}_${condition}.json" \
        > "logs/hz_suite_base_s${seed}_${condition}.log" 2>&1

    python scripts/eval.py \
        --config "configs/fg_pdd_hangzhou4x4_eval_${condition}.yaml" \
        --model_path "logs/fg_pdd_hangzhou4x4_s${seed}/final_model" \
        --gpus "$gpu" --num_episodes 5 --seed_start "$seed_start" \
        --output "results/hangzhou_failure_suite/fg_s${seed}_${condition}.json" \
        > "logs/hz_suite_fg_s${seed}_${condition}.log" 2>&1
done

echo "Hangzhou failure suite completed for seed ${seed}."
