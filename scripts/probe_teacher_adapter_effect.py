"""Paired action probe isolating a failure adapter from its selected teacher."""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from scripts.eval import make_agent, set_gpus
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.trainers.online_trainer import _agent_act
from src.utils.config import load_config, merge_config


def act(agent, observations, masks, adjacency, observation_masks):
    return _agent_act(
        agent,
        observations,
        masks,
        adjacency,
        explore=False,
        obs_mask=observation_masks,
        clean_obs=observations,
    )[0]


def sensor_indices(env, agent_id):
    lane_count = len(env.incoming_lanes[agent_id])
    lane_slots = (
        env.max_incoming_lanes
        if env.pad_heterogeneous_spaces
        else lane_count
    )
    return np.concatenate([
        np.arange(0, lane_count),
        np.arange(lane_slots, lane_slots + lane_count),
        np.arange(2 * lane_slots, 2 * lane_slots + lane_count),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_config", required=True)
    parser.add_argument("--student_config", required=True)
    parser.add_argument("--teacher_path", required=True)
    parser.add_argument("--student_path", required=True)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--seed_start", type=int, default=63000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    teacher_config = load_config(args.teacher_config)
    student_config = load_config(args.student_config)
    if args.gpus is not None:
        teacher_config = merge_config(
            teacher_config, {"training": {"gpus": args.gpus}}
        )
        student_config = merge_config(
            student_config, {"training": {"gpus": args.gpus}}
        )
        set_gpus(args.gpus)

    env = SUMOMultiAgentEnv(student_config)
    env.reset(seed=student_config.get("seed", 42))
    teacher = make_agent(env, teacher_config)
    student = make_agent(env, student_config)
    teacher.load(args.teacher_path)
    student.load(args.student_path)

    counts = {
        "clean_teacher_student_mismatch": 0,
        "teacher_dropout_flip": 0,
        "student_dropout_flip": 0,
        "dropout_teacher_student_mismatch": 0,
        "trials": 0,
    }
    for episode in range(args.num_episodes):
        _, _ = env.reset(seed=args.seed_start + episode)
        done = False
        while not done:
            masks = env.get_legal_actions()
            adjacency = env.get_adjacency(
                mode=student_config.get("dynamic_graph", {}).get(
                    "graph_type", "static"
                ),
                k=student_config.get("dynamic_graph", {}).get(
                    "neighbor_k", 1
                ),
            )
            clean = {
                aid: value.copy()
                for aid, value in env.last_clean_observations.items()
            }
            clean_masks = {
                aid: np.ones_like(value, dtype=np.float32)
                for aid, value in clean.items()
            }
            dropped = {aid: value.copy() for aid, value in clean.items()}
            dropped_masks = {
                aid: value.copy() for aid, value in clean_masks.items()
            }
            for aid in env.agent_ids:
                indices = sensor_indices(env, aid)
                dropped[aid][indices] = 0.0
                dropped_masks[aid][indices] = 0.0

            teacher_clean = act(
                teacher, clean, masks, adjacency, clean_masks
            )
            student_clean = act(
                student, clean, masks, adjacency, clean_masks
            )
            teacher_dropped = act(
                teacher, dropped, masks, adjacency, dropped_masks
            )
            student_dropped = act(
                student, dropped, masks, adjacency, dropped_masks
            )

            for aid in env.agent_ids:
                if np.asarray(masks[aid]).sum() <= 1:
                    continue
                counts["trials"] += 1
                counts["clean_teacher_student_mismatch"] += int(
                    teacher_clean[aid] != student_clean[aid]
                )
                counts["teacher_dropout_flip"] += int(
                    teacher_dropped[aid] != teacher_clean[aid]
                )
                counts["student_dropout_flip"] += int(
                    student_dropped[aid] != student_clean[aid]
                )
                counts["dropout_teacher_student_mismatch"] += int(
                    student_dropped[aid] != teacher_dropped[aid]
                )

            _, _, terminated, truncated, _ = env.step(teacher_clean)
            done = terminated or truncated

    trials = max(counts["trials"], 1)
    result = {
        key + "_rate": value / trials
        for key, value in counts.items()
        if key != "trials"
    }
    result["trials"] = counts["trials"]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    main()
