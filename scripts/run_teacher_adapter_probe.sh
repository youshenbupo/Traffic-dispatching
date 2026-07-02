#!/usr/bin/env bash
set -euo pipefail

scenario="${1:?scenario required}"
seed="${2:?seed required}"
gpu="${3:?gpu required}"
cd /data/users/jyh/light_RL
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl
mkdir -p results/teacher_adapter_probe

case "$scenario" in
    hangzhou)
        teacher_config="configs/mappo_hangzhou4x4_robust_eval.yaml"
        student_config="configs/fg_pdd_hangzhou4x4_fixed_teacher_eval.yaml"
        student_path="logs/fg_pdd_hangzhou_fixed_s${seed}/final_model"
        ;;
    cologne3)
        teacher_config="configs/mappo_cologne3_eval.yaml"
        student_config="configs/fg_pdd_cologne3_fixed_teacher_eval.yaml"
        student_path="logs/fg_pdd_cologne3_fixed_s${seed}/final_model"
        ;;
    *)
        echo "Unknown scenario: $scenario" >&2
        exit 2
        ;;
esac

python scripts/probe_teacher_adapter_effect.py \
    --teacher_config "$teacher_config" \
    --student_config "$student_config" \
    --teacher_path "logs/selected_teachers/${scenario}" \
    --student_path "$student_path" \
    --gpus "$gpu" --num_episodes 1 --seed_start 63000 \
    --output "results/teacher_adapter_probe/${scenario}_s${seed}.json"
