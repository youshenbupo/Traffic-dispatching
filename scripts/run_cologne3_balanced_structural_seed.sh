#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
gpu="${2:?gpu required}"
cd /data/users/jyh/light_RL
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl
mkdir -p results/cologne3_balanced_structural

experiment="fg_pdd_cologne3_balanced_structural_s${seed}"
python scripts/train.py \
    --config configs/fg_pdd_cologne3_balanced_structural.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$experiment" \
    > "logs/cologne_balanced_structural_train_s${seed}.log" 2>&1

python scripts/eval.py \
    --config configs/fg_pdd_cologne3_balanced_structural_eval.yaml \
    --model_path "logs/${experiment}/best_model" \
    --gpus "$gpu" --num_episodes 5 --seed_start 62000 \
    --output "results/cologne3_balanced_structural/s${seed}.json" \
    > "logs/cologne_balanced_structural_eval_s${seed}.log" 2>&1

echo "Balanced structural adapter completed: seed ${seed}."
