#!/usr/bin/env bash
set -euo pipefail

scenario="${1:?scenario required}"
seed="${2:?seed required}"
gpu="${3:?gpu required}"
cd /data/users/jyh/light_RL

while [[ ! -f "logs/selected_teachers/${scenario}/mappo.pth" ]]; do
    sleep 60
done

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl
mkdir -p results/fixed_teacher_study

case "$scenario" in
    hangzhou)
        train_config="configs/fg_pdd_hangzhou4x4_fixed_teacher.yaml"
        eval_config="configs/fg_pdd_hangzhou4x4_fixed_teacher_eval.yaml"
        experiment="fg_pdd_hangzhou_fixed_s${seed}"
        seed_start=52000
        ;;
    cologne3)
        train_config="configs/fg_pdd_cologne3_fixed_teacher.yaml"
        eval_config="configs/fg_pdd_cologne3_fixed_teacher_eval.yaml"
        experiment="fg_pdd_cologne3_fixed_s${seed}"
        seed_start=62000
        ;;
    *)
        echo "Unknown scenario: $scenario" >&2
        exit 2
        ;;
esac

python scripts/train.py \
    --config "$train_config" --gpus "$gpu" --seed "$seed" \
    --exp_name "$experiment" \
    > "logs/fixed_${scenario}_train_s${seed}.log" 2>&1

python scripts/eval.py \
    --config "$eval_config" \
    --model_path "logs/${experiment}/final_model" \
    --gpus "$gpu" --num_episodes 5 --seed_start "$seed_start" \
    --output "results/fixed_teacher_study/${scenario}_s${seed}.json" \
    > "logs/fixed_${scenario}_eval_s${seed}.log" 2>&1

echo "Fixed-teacher adapter completed: ${scenario} seed ${seed}."
