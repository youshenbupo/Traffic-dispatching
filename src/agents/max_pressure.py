"""Max-pressure traffic signal controller."""
from typing import Dict, List
import numpy as np


class MaxPressureController:
    """Select phase that maximizes pressure (upstream queue - downstream queue)."""

    def __init__(self, phases: Dict[str, list], incoming_lanes: Dict[str, List[str]],
                 outgoing_lanes: Dict[str, List[str]], yellow_time: float = 3.0,
                 min_green: float = 7.0):
        self.phases = phases
        self.incoming_lanes = incoming_lanes
        self.outgoing_lanes = outgoing_lanes
        self.yellow_time = yellow_time
        self.min_green = min_green
        self.last_switch = {tl_id: 0.0 for tl_id in phases}
        self.current_phase = {tl_id: 0 for tl_id in phases}
        self.time = 0.0

    def reset(self):
        self.last_switch = {tl_id: 0.0 for tl_id in self.phases}
        self.current_phase = {tl_id: 0 for tl_id in self.phases}
        self.time = 0.0

    def act(self, obs: Dict[str, np.ndarray], info: Dict = None,
            env=None) -> Dict[str, int]:
        self.time += 5  # assume delta_time=5
        actions = {}
        for tl_id in self.phases:
            elapsed = self.time - self.last_switch[tl_id]
            if elapsed < self.min_green:
                actions[tl_id] = self.current_phase[tl_id]
                continue

            best_phase = self.current_phase[tl_id]
            best_pressure = -1e9
            for p_idx, state in enumerate(self.phases[tl_id]):
                pressure = self._compute_phase_pressure(tl_id, state, env)
                if pressure > best_pressure:
                    best_pressure = pressure
                    best_phase = p_idx

            if best_phase != self.current_phase[tl_id]:
                self.current_phase[tl_id] = best_phase
                self.last_switch[tl_id] = self.time
            actions[tl_id] = self.current_phase[tl_id]
        return actions

    def _compute_phase_pressure(self, tl_id: str, state: str, env) -> float:
        """Compute pressure for a phase state string.

        Pressure is sum over green movements of (upstream queue - downstream queue).
        """
        lanes = self.incoming_lanes[tl_id]
        pressure = 0.0
        for lane_idx, lane in enumerate(lanes):
            if lane_idx >= len(state):
                continue
            signal = state[lane_idx]
            if signal.lower() == "g":
                q_up = self._queue(lane, env)
                q_down = 0.0
                # Find downstream lane
                links = env.sumo.lane.getLinks(lane)
                for link in links:
                    q_down += self._queue(link[0], env)
                pressure += (q_up - q_down)
        return pressure

    def _queue(self, lane: str, env) -> float:
        try:
            vehs = env.sumo.lane.getLastStepVehicleIDs(lane)
            return float(sum(1 for v in vehs if env.sumo.vehicle.getSpeed(v) < 0.1))
        except Exception:
            return 0.0
