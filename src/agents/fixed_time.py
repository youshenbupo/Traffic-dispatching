"""Fixed-time traffic signal controller."""
from typing import Dict


class FixedTimeController:
    """Cyclic fixed-time controller."""

    def __init__(self, phases: Dict[str, list], cycle_time: float = 60.0, yellow_time: float = 3.0):
        self.phases = phases
        self.cycle_time = cycle_time
        self.yellow_time = yellow_time
        self.elapsed = {tl_id: 0.0 for tl_id in phases}

    def reset(self):
        self.elapsed = {tl_id: 0.0 for tl_id in self.phases}

    def act(self, obs: Dict, info: Dict = None) -> Dict[str, int]:
        actions = {}
        n_phases = {tl_id: len(phases) for tl_id, phases in self.phases.items()}
        green_time = (self.cycle_time - self.yellow_time * max(1, max(n_phases.values()))) / max(1, max(n_phases.values()))
        for tl_id in self.phases:
            self.elapsed[tl_id] += 5  # assuming delta_time = 5
            phase_idx = int((self.elapsed[tl_id] % self.cycle_time) / (green_time + self.yellow_time))
            phase_idx = min(phase_idx, n_phases[tl_id] - 1)
            actions[tl_id] = phase_idx
        return actions
