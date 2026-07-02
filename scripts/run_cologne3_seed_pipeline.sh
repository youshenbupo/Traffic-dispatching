#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
gpu="${2:?gpu required}"
cd /data/users/jyh/light_RL

# Preserve experiment priority and avoid competing with the long Hangzhou jobs.
while systemctl --user is-active --quiet hzpipe45 \
   || systemctl --user is-active --quiet hzpipe46 \
   || systemctl --user is-active --quiet hzsuite42 \
   || systemctl --user is-active --quiet hzsuite43 \
   || systemctl --user is-active --quiet hzsuite44; do
    sleep 60
done

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl

teacher="mappo_cologne3_teacher_s${seed}"
continued="mappo_cologne3_continue_s${seed}"
baseline="mappo_cologne3_base30_s${seed}"
student="fg_pdd_cologne3_s${seed}"

python scripts/train.py \
    --config configs/mappo_cologne3.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$teacher" \
    > "logs/cologne_teacher_s${seed}.log" 2>&1

python scripts/train.py \
    --config configs/mappo_cologne3.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$continued" \
    --pretrained_model "logs/${teacher}/final_model" \
    > "logs/cologne_continue_s${seed}.log" 2>&1

python scripts/train.py \
    --config configs/fg_pdd_cologne3.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$student" \
    > "logs/cologne_fg_s${seed}.log" 2>&1

python scripts/train.py \
    --config configs/mappo_cologne3.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$baseline" \
    --pretrained_model "logs/${continued}/final_model" \
    > "logs/cologne_base30_s${seed}.log" 2>&1

mkdir -p results/cologne3_fg_pdd
python scripts/eval.py \
    --config configs/mappo_cologne3_eval.yaml \
    --model_path "logs/${baseline}/final_model" --gpus "$gpu" \
    --num_episodes 5 --seed_start 62000 \
    --output "results/cologne3_fg_pdd/base30_s${seed}.json" \
    > "logs/cologne_base_eval_s${seed}.log" 2>&1

python scripts/eval.py \
    --config configs/fg_pdd_cologne3_eval.yaml \
    --model_path "logs/${student}/final_model" --gpus "$gpu" \
    --num_episodes 5 --seed_start 62000 \
    --output "results/cologne3_fg_pdd/fg_s${seed}.json" \
    > "logs/cologne_fg_eval_s${seed}.log" 2>&1

echo "Cologne3 seed ${seed} pipeline completed."
