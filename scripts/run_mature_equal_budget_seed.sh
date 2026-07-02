#!/usr/bin/env bash
set -euo pipefail

scenario="${1:?scenario required}"
seed="${2:?seed required}"
gpu="${3:?gpu required}"
cd /data/users/jyh/light_RL

while [[ ! -f "logs/selected_teachers_mature/${scenario}/mappo.pth" ]]; do
    sleep 60
done

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl
mkdir -p results/mature_equal_budget

case "$scenario" in
    hangzhou)
        student_config="configs/fg_pdd_hangzhou4x4_mature_teacher.yaml"
        student_eval="configs/fg_pdd_hangzhou4x4_mature_teacher_eval.yaml"
        baseline_config="configs/mappo_hangzhou4x4_robust_pilot.yaml"
        baseline_eval="configs/mappo_hangzhou4x4_robust_eval.yaml"
        base30="mappo_hangzhou4x4_base30_s${seed}"
        seed_start=52000
        ;;
    cologne3)
        student_config="configs/fg_pdd_cologne3_mature_teacher.yaml"
        student_eval="configs/fg_pdd_cologne3_mature_teacher_eval.yaml"
        baseline_config="configs/mappo_cologne3.yaml"
        baseline_eval="configs/mappo_cologne3_eval.yaml"
        base30="mappo_cologne3_base30_s${seed}"
        seed_start=62000
        ;;
    *)
        echo "Unknown scenario: $scenario" >&2
        exit 2
        ;;
esac

student="mature_fg_${scenario}_s${seed}"
baseline="mappo_${scenario}_base40_s${seed}"

python scripts/train.py \
    --config "$student_config" --gpus "$gpu" --seed "$seed" \
    --exp_name "$student" \
    > "logs/mature_fg_${scenario}_train_s${seed}.log" 2>&1

python scripts/train.py \
    --config "$baseline_config" --gpus "$gpu" --seed "$seed" \
    --exp_name "$baseline" \
    --pretrained_model "logs/${base30}/final_model" \
    > "logs/base40_${scenario}_train_s${seed}.log" 2>&1

python scripts/eval.py \
    --config "$student_eval" \
    --model_path "logs/${student}/final_model" \
    --gpus "$gpu" --num_episodes 5 --seed_start "$seed_start" \
    --output "results/mature_equal_budget/fg_${scenario}_s${seed}.json" \
    > "logs/mature_fg_${scenario}_eval_s${seed}.log" 2>&1

python scripts/eval.py \
    --config "$baseline_eval" \
    --model_path "logs/${baseline}/final_model" \
    --gpus "$gpu" --num_episodes 5 --seed_start "$seed_start" \
    --output "results/mature_equal_budget/base40_${scenario}_s${seed}.json" \
    > "logs/base40_${scenario}_eval_s${seed}.log" 2>&1

echo "Mature equal-budget study completed: ${scenario} seed ${seed}."
