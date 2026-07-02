#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
gpu="${2:?gpu required}"
cd /data/users/jyh/light_RL
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl
mkdir -p results/cologne3_structural

experiment="fg_pdd_cologne3_structural_s${seed}"
python scripts/train.py \
    --config configs/fg_pdd_cologne3_structural.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$experiment" \
    > "logs/cologne_structural_train_s${seed}.log" 2>&1

python scripts/eval.py \
    --config configs/fg_pdd_cologne3_structural_eval.yaml \
    --model_path "logs/${experiment}/final_model" \
    --gpus "$gpu" --num_episodes 5 --seed_start 62000 \
    --output "results/cologne3_structural/s${seed}.json" \
    > "logs/cologne_structural_eval_s${seed}.log" 2>&1

echo "Cologne3 structural adapter completed: seed ${seed}."
