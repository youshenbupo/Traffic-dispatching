#!/usr/bin/env bash
set -euo pipefail

cd /data/users/jyh/light_RL
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl

gpu="${1:-0}"
mkdir -p results/fa_fg_pdd

for seed in 42 43 44; do
    experiment="fa_context_fg_pdd_grid4x4_s${seed}"
    python scripts/train.py \
        --config configs/fa_fg_pdd_grid4x4.yaml \
        --gpus "$gpu" --seed "$seed" --exp_name "$experiment" \
        > "logs/fa_context_s${seed}.log" 2>&1

    for condition in burst iid30 clean; do
        case "$condition" in
            burst)
                config="configs/fa_fg_pdd_grid4x4_eval_burst.yaml"
                seed_start=37000
                ;;
            iid30)
                config="configs/fa_fg_pdd_grid4x4_eval.yaml"
                seed_start=34000
                ;;
            clean)
                config="configs/fa_fg_pdd_grid4x4_eval_clean.yaml"
                seed_start=31000
                ;;
        esac
        python scripts/eval.py \
            --config "$config" \
            --model_path "logs/${experiment}/final_model" --gpus "$gpu" \
            --num_episodes 5 --seed_start "$seed_start" \
            --output "results/fa_fg_pdd/fa_context_s${seed}_${condition}.json" \
            > "logs/fa_context_${condition}_eval_s${seed}.log" 2>&1
    done
done

echo "Corrected failure-context FG-PDD rerun completed."
