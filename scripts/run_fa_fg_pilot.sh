#!/usr/bin/env bash
set -euo pipefail

cd /data/users/jyh/light_RL
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl

gpu="${1:-0}"
mkdir -p results/fa_fg_pdd

for seed in 42 43 44; do
    for method in burst_fg fa_fg; do
        if [[ "$method" == "fa_fg" ]]; then
            train_config="configs/fa_fg_pdd_grid4x4.yaml"
            burst_eval_config="configs/fa_fg_pdd_grid4x4_eval_burst.yaml"
            iid_eval_config="configs/fa_fg_pdd_grid4x4_eval.yaml"
            clean_eval_config="configs/fa_fg_pdd_grid4x4_eval_clean.yaml"
        else
            train_config="configs/burst_fg_pdd_grid4x4.yaml"
            burst_eval_config="configs/fg_pdd_grid4x4_eval_burst.yaml"
            iid_eval_config="configs/fg_pdd_grid4x4_eval.yaml"
            clean_eval_config="configs/fg_pdd_grid4x4_eval_clean.yaml"
        fi
        experiment="${method}_pdd_grid4x4_s${seed}"
        python scripts/train.py \
            --config "$train_config" \
            --gpus "$gpu" --seed "$seed" --exp_name "$experiment" \
            > "logs/${method}_s${seed}.log" 2>&1

        python scripts/eval.py \
            --config "$burst_eval_config" \
            --model_path "logs/${experiment}/final_model" --gpus "$gpu" \
            --num_episodes 5 --seed_start 37000 \
            --output "results/fa_fg_pdd/${method}_s${seed}_burst.json" \
            > "logs/${method}_burst_eval_s${seed}.log" 2>&1

        python scripts/eval.py \
            --config "$iid_eval_config" \
            --model_path "logs/${experiment}/final_model" --gpus "$gpu" \
            --num_episodes 5 --seed_start 34000 \
            --output "results/fa_fg_pdd/${method}_s${seed}_iid30.json" \
            > "logs/${method}_iid_eval_s${seed}.log" 2>&1

        python scripts/eval.py \
            --config "$clean_eval_config" \
            --model_path "logs/${experiment}/final_model" --gpus "$gpu" \
            --num_episodes 5 --seed_start 31000 \
            --output "results/fa_fg_pdd/${method}_s${seed}_clean.json" \
            > "logs/${method}_clean_eval_s${seed}.log" 2>&1
    done
done

echo "Failure-age FG-PDD pilot completed."
