"""Evaluation metrics for traffic signal control."""
import numpy as np
from typing import Dict, List, Any


class MetricsTracker:
    """Track and compute TSC metrics from a SUMO simulation episode."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_data = []
        self.travel_times = []
        self.waiting_times = []
        self.queue_lengths = []
        self.throughputs = []
        self.rewards = []
        self.communication_rates = []
        self.mean_reliabilities = []
        self.observation_qualities = []

    def record_step(self, info: Dict[str, Any]):
        self.step_data.append(info)
        if "reward" in info:
            self.rewards.append(info["reward"])
        if "queue_length" in info:
            self.queue_lengths.append(info["queue_length"])
        if "throughput" in info:
            self.throughputs.append(info["throughput"])
        if "waiting_time" in info:
            self.step_data[-1]["waiting_time"] = info["waiting_time"]
        if "communication_rate" in info:
            self.communication_rates.append(info["communication_rate"])
        if "mean_reliability" in info:
            self.mean_reliabilities.append(info["mean_reliability"])
        if "observation_quality" in info:
            self.observation_qualities.append(info["observation_quality"])

    def finalize(self, arrived_vehicles: List[Dict[str, Any]]):
        """Finalize episode metrics using arrived vehicle data."""
        for v in arrived_vehicles:
            self.travel_times.append(v.get("travel_time", 0.0))
            self.waiting_times.append(v.get("waiting_time", 0.0))

    def aggregate(self) -> Dict[str, float]:
        result = {
            "average_travel_time": float(np.mean(self.travel_times)) if self.travel_times else 0.0,
            "average_waiting_time": float(np.mean(self.waiting_times)) if self.waiting_times else 0.0,
            "max_waiting_time": float(np.max(self.waiting_times)) if self.waiting_times else 0.0,
            "queue_length": float(np.mean(self.queue_lengths)) if self.queue_lengths else 0.0,
            "throughput": float(np.sum(self.throughputs)),
            "episode_reward": float(np.sum(self.rewards)),
            "episode_length": len(self.step_data),
        }
        if self.communication_rates:
            result["communication_rate"] = float(np.mean(self.communication_rates))
        if self.mean_reliabilities:
            result["mean_reliability"] = float(np.mean(self.mean_reliabilities))
        if self.observation_qualities:
            result["observation_quality"] = float(np.mean(self.observation_qualities))
        return result

    @staticmethod
    def compute_from_env(env) -> Dict[str, float]:
        """Compute current step metrics directly from environment."""
        total_queue = 0.0
        total_wait = 0.0
        n_lanes = 0
        for tl_id in env.tls_ids:
            for lane in env.incoming_lanes[tl_id]:
                vehs = env.sumo.lane.getLastStepVehicleIDs(lane)
                total_queue += sum(1 for v in vehs if env.sumo.vehicle.getSpeed(v) < 0.1)
                total_wait += env.sumo.lane.getWaitingTime(lane)
                n_lanes += 1
        return {
            "queue_length": total_queue / max(1, n_lanes),
            "waiting_time": total_wait / max(1, n_lanes),
            # Count completed trips, not lane occupancy. The environment
            # accumulates arrivals across all internal SUMO steps.
            "throughput": float(env.last_step_arrived_count),
            "observation_quality": float(np.mean(
                list(env.last_observation_quality.values())
            )) if env.last_observation_quality else 1.0,
        }

    @staticmethod
    def compute_travel_time_arrived(simulation) -> List[Dict[str, Any]]:
        """Extract arrived vehicle travel times from a SUMO-like simulation object."""
        arrived = []
        # To be implemented per simulation backend (SUMO/CityFlow)
        return arrived
