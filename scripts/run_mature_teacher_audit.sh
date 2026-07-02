#!/usr/bin/env bash
set -euo pipefail

scenario="${1:?scenario required}"
seed="${2:?seed required}"
gpu="${3:?gpu required}"
cd /data/users/jyh/light_RL
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl
mkdir -p results/mature_teacher_audit

case "$scenario" in
    hangzhou)
        config="configs/mappo_hangzhou4x4_robust_eval.yaml"
        model_prefix="mappo_hangzhou4x4_base30_s${seed}"
        seed_start=64000
        ;;
    cologne3)
        config="configs/mappo_cologne3_eval.yaml"
        model_prefix="mappo_cologne3_base30_s${seed}"
        seed_start=65000
        ;;
    *)
        echo "Unknown scenario: $scenario" >&2
        exit 2
        ;;
esac

for checkpoint in best_model final_model; do
    python scripts/eval.py \
        --config "$config" \
        --model_path "logs/${model_prefix}/${checkpoint}" \
        --gpus "$gpu" --num_episodes 5 --seed_start "$seed_start" \
        --output \
        "results/mature_teacher_audit/${scenario}30_s${seed}_${checkpoint}.json" \
        > "logs/audit_${scenario}30_s${seed}_${checkpoint}.log" 2>&1
done

echo "Mature teacher audit completed: ${scenario} seed ${seed}."
