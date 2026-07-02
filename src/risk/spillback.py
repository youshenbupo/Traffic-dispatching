"""Semantic traffic tokens and privileged future-spillback targets."""
from typing import Dict, Sequence

import numpy as np


def build_phase_lane_matrix(
    incoming_lanes: Sequence[str],
    phases: Sequence[str],
    controlled_links,
    max_lanes: int,
    max_phases: int,
) -> np.ndarray:
    """Map each padded incoming lane to phases that can serve it."""
    matrix = np.zeros((max_lanes, max_phases), dtype=np.float32)
    lane_indices = {lane: index for index, lane in enumerate(incoming_lanes)}
    for signal_index, links in enumerate(controlled_links):
        served_lanes = {
            link[0] for link in links
            if link and link[0] in lane_indices
        }
        for phase_index, state in enumerate(phases[:max_phases]):
            if signal_index >= len(state) or state[signal_index] not in "Gg":
                continue
            for lane in served_lanes:
                matrix[lane_indices[lane], phase_index] = 1.0
    return matrix


def semantic_lane_tokens(
    observation: np.ndarray,
    observation_mask: np.ndarray,
    phase_lane_matrix: np.ndarray,
    lane_presence: np.ndarray,
) -> np.ndarray:
    """Encode padded lane measurements with phase-service semantics."""
    max_lanes, max_phases = phase_lane_matrix.shape
    expected = 3 * max_lanes + max_phases
    if observation.shape[-1] != expected:
        raise ValueError(
            f"Expected observation width {expected}, got {observation.shape[-1]}"
        )
    values = observation[:3 * max_lanes].reshape(3, max_lanes).T
    masks = observation_mask[:3 * max_lanes].reshape(3, max_lanes).T
    phase_state = observation[3 * max_lanes:expected]
    active_service = phase_lane_matrix @ phase_state
    tokens = np.concatenate([
        values,
        masks,
        phase_lane_matrix,
        active_service[:, None],
        lane_presence[:max_lanes, None],
    ], axis=-1)
    return tokens.astype(np.float32)


def future_spillback_targets(
    queue_length: Sequence[float],
    active_vehicles: Sequence[float],
    throughput: Sequence[float],
    horizon: int = 120,
    queue_mean_threshold: float = 0.5,
    queue_peak_threshold: float = 0.75,
    active_growth_threshold: float = 50.0,
) -> Dict[str, np.ndarray]:
    """Create privileged labels using only traffic outcomes after each step."""
    queue = np.asarray(queue_length, dtype=np.float32)
    active = np.asarray(active_vehicles, dtype=np.float32)
    departed = np.asarray(throughput, dtype=np.float32)
    if not (len(queue) == len(active) == len(departed)):
        raise ValueError("Traffic target series must have equal lengths")
    if horizon <= 0:
        raise ValueError("horizon must be positive")

    size = len(queue)
    future_mean = np.zeros(size, dtype=np.float32)
    future_peak = np.zeros(size, dtype=np.float32)
    active_growth = np.zeros(size, dtype=np.float32)
    future_departures = np.zeros(size, dtype=np.float32)
    valid = np.zeros(size, dtype=np.float32)
    for step in range(size):
        end = min(size, step + horizon + 1)
        if step + 1 >= end:
            continue
        future_queue = queue[step + 1:end]
        future_active = active[step + 1:end]
        future_mean[step] = float(np.mean(future_queue))
        future_peak[step] = float(np.max(future_queue))
        active_growth[step] = float(np.max(future_active) - active[step])
        future_departures[step] = float(np.sum(departed[step + 1:end]))
        valid[step] = float(end == step + horizon + 1)

    spillback = (
        (future_mean >= queue_mean_threshold)
        & (future_peak >= queue_peak_threshold)
        & (active_growth >= active_growth_threshold)
    ).astype(np.float32)
    return {
        "spillback_risk": spillback,
        "future_queue_mean": future_mean,
        "future_queue_peak": future_peak,
        "future_active_growth": active_growth,
        "future_departures": future_departures,
        "valid": valid,
    }
