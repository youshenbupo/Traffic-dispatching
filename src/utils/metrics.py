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
        self.reconstruction_errors = []
        self.reconstruction_confidences = []
        self.decision_confidences = []
        self.active_vehicle_counts = []
        self.departed_vehicle_counts = []
        self.step_durations = []

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
        if "reconstruction_error" in info:
            self.reconstruction_errors.append(info["reconstruction_error"])
        if "reconstruction_confidence" in info:
            self.reconstruction_confidences.append(info["reconstruction_confidence"])
        if "decision_confidence" in info:
            self.decision_confidences.append(info["decision_confidence"])
        if "active_vehicles" in info:
            self.active_vehicle_counts.append(info["active_vehicles"])
            self.step_durations.append(info.get("step_duration", 1.0))
        if "departed_vehicles" in info:
            self.departed_vehicle_counts.append(info["departed_vehicles"])

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
        if self.active_vehicle_counts:
            total_time_spent = float(np.sum(
                np.asarray(self.active_vehicle_counts)
                * np.asarray(self.step_durations)
            ))
            departed = float(max(self.departed_vehicle_counts, default=0.0))
            result["total_time_spent"] = total_time_spent
            result["departed_vehicles"] = departed
            result["time_spent_per_departed_vehicle"] = (
                total_time_spent / max(departed, 1.0)
            )
            result["completion_rate"] = (
                result["throughput"] / max(departed, 1.0)
            )
        if self.communication_rates:
            result["communication_rate"] = float(np.mean(self.communication_rates))
        if self.mean_reliabilities:
            result["mean_reliability"] = float(np.mean(self.mean_reliabilities))
        if self.observation_qualities:
            result["observation_quality"] = float(np.mean(self.observation_qualities))
        if self.reconstruction_errors:
            result["reconstruction_error"] = float(np.mean(self.reconstruction_errors))
        if self.reconstruction_confidences:
            result["reconstruction_confidence"] = float(
                np.mean(self.reconstruction_confidences)
            )
        if self.decision_confidences:
            result["decision_confidence"] = float(
                np.mean(self.decision_confidences)
            )
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
            "active_vehicles": float(len(env.vehicle_subscriptions)),
            "departed_vehicles": float(
                len(env.arrived_vehicle_info)
                + len(env.vehicle_subscriptions)
            ),
            "step_duration": float(env.delta_time),
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
