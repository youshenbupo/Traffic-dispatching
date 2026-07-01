#!/usr/bin/env bash
set -euo pipefail

seed="${1:?seed required}"
gpu="${2:?gpu required}"
root="/data/users/jyh/light_RL"
cd "$root"

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate light_rl

teacher="mappo_hangzhou4x4_norm_teacher_s${seed}"
continued="mappo_hangzhou4x4_continue_s${seed}"
baseline="mappo_hangzhou4x4_base30_s${seed}"
student="fg_pdd_hangzhou4x4_s${seed}"
teacher_model="logs/${teacher}/final_model/mappo.pth"

# The initial teacher may still be attached to the launching SSH client.
# Wait up to three hours for its atomic final checkpoint.
for _ in $(seq 1 360); do
    [[ -f "$teacher_model" ]] && break
    if ! pgrep -f "scripts/train.py --config configs/mappo_hangzhou4x4_robust_pilot.yaml --gpus ${gpu} --seed ${seed} --exp_name ${teacher}" >/dev/null; then
        echo "Teacher process disappeared; restarting it inside the server pipeline."
        python scripts/train.py \
            --config configs/mappo_hangzhou4x4_robust_pilot.yaml \
            --gpus "$gpu" --seed "$seed" --exp_name "$teacher" \
            > "logs/hz_norm_teacher_s${seed}.log" 2>&1
    fi
    sleep 30
done
[[ -f "$teacher_model" ]] || {
    echo "Teacher checkpoint not found: $teacher_model" >&2
    exit 2
}

python scripts/train.py \
    --config configs/mappo_hangzhou4x4_robust_pilot.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$continued" \
    --pretrained_model "logs/${teacher}/final_model" \
    > "logs/hz_continue_s${seed}.log" 2>&1

python scripts/train.py \
    --config configs/fg_pdd_hangzhou4x4.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$student" \
    > "logs/hz_fg_s${seed}.log" 2>&1

python scripts/train.py \
    --config configs/mappo_hangzhou4x4_robust_pilot.yaml \
    --gpus "$gpu" --seed "$seed" --exp_name "$baseline" \
    --pretrained_model "logs/${continued}/final_model" \
    > "logs/hz_base30_s${seed}.log" 2>&1

mkdir -p results/hangzhou_fg_pdd
python scripts/eval.py \
    --config configs/mappo_hangzhou4x4_robust_eval.yaml \
    --model_path "logs/${baseline}/final_model" --gpus "$gpu" \
    --num_episodes 5 --seed_start 52000 \
    --output "results/hangzhou_fg_pdd/base30_s${seed}.csv" \
    > "logs/hz_base30_eval_s${seed}.log" 2>&1

python scripts/eval.py \
    --config configs/fg_pdd_hangzhou4x4_eval.yaml \
    --model_path "logs/${student}/final_model" --gpus "$gpu" \
    --num_episodes 5 --seed_start 52000 \
    --output "results/hangzhou_fg_pdd/fg_s${seed}.csv" \
    > "logs/hz_fg_eval_s${seed}.log" 2>&1

echo "Hangzhou seed ${seed} pipeline completed."
