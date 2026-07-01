"""SUMO-based multi-agent traffic signal control environment."""
import os
import sys
import copy
import random
import numpy as np
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

# SUMO imports
try:
    import traci
    import sumolib
except ImportError:
    pass


class SUMOMultiAgentEnv:
    """Multi-agent TSC environment using SUMO and TraCI.

    Each traffic-light-controlled intersection is one agent.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.env_cfg = config["env"]
        self.scenario_dir = self.env_cfg["scenario_dir"]
        self.config_file = os.path.join(self.scenario_dir, self.env_cfg["config_file"])
        self.route_file = os.path.join(self.scenario_dir, self.env_cfg["route_file"])
        self.temp_route_file: Optional[str] = None

        self.use_gui = self.env_cfg.get("use_gui", False)
        self.num_seconds = self.env_cfg.get("num_seconds", 3600)
        self.delta_time = self.env_cfg.get("delta_time", 5)
        self.yellow_time = self.env_cfg.get("yellow_time", 3)
        self.min_green = self.env_cfg.get("min_green", 7)
        self.max_green = self.env_cfg.get("max_green", 60)
        self.reward_type = self.env_cfg.get("reward_type", "delay_queue")
        self.obs_type = self.env_cfg.get("observation_type", "queue_wait_phase_flow")
        self.demand_jitter = float(self.env_cfg.get("demand_jitter", 0.0))
        self.queue_obs_scale = float(
            self.env_cfg.get("queue_obs_scale", 1.0)
        )
        self.wait_obs_scale = float(
            self.env_cfg.get("wait_obs_scale", 1.0)
        )
        self.flow_obs_scale = float(
            self.env_cfg.get("flow_obs_scale", 1.0)
        )
        self.reward_scale = float(self.env_cfg.get("reward_scale", 1.0))
        self.sumo_seed = int(config.get("seed", 42))

        self.sumo = None
        self.vehicle_subscriptions = {}
        self.arrived_vehicle_info: List[Dict[str, Any]] = []
        self.last_step_arrived_count = 0

        # Populated on reset
        self.tls_ids: List[str] = []
        self.agent_ids: List[str] = []
        self.num_agents: int = 0
        self.phases: Dict[str, List[str]] = {}
        self.incoming_lanes: Dict[str, List[str]] = {}
        self.outgoing_lanes: Dict[str, List[str]] = {}
        self.current_phase: Dict[str, int] = {}
        self.phase_duration: Dict[str, int] = {}
        self.pending_phase: Dict[str, Optional[int]] = {}
        self.transition_remaining: Dict[str, int] = {}
        self.last_step_vehicles: Dict[str, List[str]] = defaultdict(list)

        # Robustness perturbations (optional)
        self.perturbations = config.get("robustness", {}).get("perturbations", [])
        self.sensor_noise_std = 0.0
        self.obs_dropout_prob = 0.0
        self.agent_obs_dropout_prob = 0.0
        self.burst_failure_start_prob = 0.0
        self.burst_failure_recovery_prob = 1.0
        self.burst_failure_state: Dict[str, bool] = {}
        self.correlated_failure_prob = 0.0
        self.correlated_failure_fraction = 0.0
        self.temporal_observation_fallback = False
        self.cached_valid_observations: Dict[str, np.ndarray] = {}
        self.demand_scale = 1.0
        self.directional_demand_source = None
        self.directional_demand_scale = 1.0
        self.last_observation_quality: Dict[str, float] = {}
        self.last_clean_observations: Dict[str, np.ndarray] = {}
        self.last_observation_masks: Dict[str, np.ndarray] = {}
        self.failure_age: Dict[str, int] = {}
        self.last_failure_age: Dict[str, float] = {}
        self._parse_perturbations()
        self.demand_spike_seed_offset = 0

    def _parse_perturbations(self):
        for p in self.perturbations:
            if p["type"] == "sensor_noise":
                self.sensor_noise_std = p.get("std", 0.0)
            elif p["type"] == "observation_dropout":
                self.obs_dropout_prob = p.get("prob", 0.0)
            elif p["type"] == "agent_observation_dropout":
                self.agent_obs_dropout_prob = p.get("prob", 0.0)
            elif p["type"] == "burst_agent_observation_dropout":
                self.burst_failure_start_prob = p.get("start_prob", 0.05)
                self.burst_failure_recovery_prob = p.get(
                    "recovery_prob", 0.12
                )
            elif p["type"] == "correlated_agent_observation_dropout":
                self.correlated_failure_prob = p.get("prob", 0.6)
                self.correlated_failure_fraction = p.get("fraction", 0.5)
            elif p["type"] == "temporal_observation_fallback":
                self.temporal_observation_fallback = p.get("enabled", True)
            elif p["type"] == "demand_spike":
                self.demand_scale = p.get("scale", 1.0)
            elif p["type"] == "directional_demand_spike":
                self.directional_demand_source = p.get("source", "left")
                self.directional_demand_scale = p.get("scale", 2.0)

    def _create_demand_route_file(self, seed: int) -> Optional[str]:
        """Create a temporary route file with scaled departure times.

        demand_scale > 1 compresses departures (higher density).
        demand_scale < 1 stretches them (lower density).
        For demand_scale > 1 we additionally duplicate a fraction of vehicles
        to better approximate a true demand increase.
        """
        if (
            self.demand_scale == 1.0
            and self.demand_jitter <= 0
            and self.directional_demand_scale == 1.0
        ):
            return None
        if not os.path.exists(self.route_file):
            return None
        try:
            tree = ET.parse(self.route_file)
        except Exception:
            return None
        root = tree.getroot()
        rng = np.random.RandomState(seed)
        vehicles = root.findall('vehicle')
        inv_scale = 1.0 / self.demand_scale
        for v in vehicles:
            depart = float(v.get('depart', 0.0)) * inv_scale
            if self.demand_jitter > 0:
                depart += rng.uniform(-self.demand_jitter, self.demand_jitter)
                depart = max(0.0, depart)
            v.set('depart', f"{depart:.2f}")
        # Optional: duplicate a fraction of vehicles to better approximate
        # a true demand increase.  We keep duplication modest to avoid
        # overloading the network.
        if self.demand_scale > 1.0:
            dup_prob = min(0.3, self.demand_scale - 1.0)
            additions = []
            for v in vehicles:
                if rng.rand() < dup_prob:
                    v2 = copy.deepcopy(v)
                    depart = float(v2.get('depart', 0.0)) + 0.05
                    v2.set('id', f"{v2.get('id')}_dup{seed}")
                    v2.set('depart', f"{depart:.2f}")
                    additions.append(v2)
            for v2 in additions:
                root.append(v2)
        if self.directional_demand_scale > 1.0:
            directional_additions = []
            extra_copies = int(self.directional_demand_scale - 1.0)
            fractional_copy = self.directional_demand_scale - 1.0 - extra_copies
            for v in vehicles:
                route = v.find("route")
                edges = route.get("edges", "").split() if route is not None else []
                if not edges or not edges[0].startswith(
                    self.directional_demand_source
                ):
                    continue
                copies = extra_copies + int(rng.rand() < fractional_copy)
                for copy_id in range(copies):
                    v2 = copy.deepcopy(v)
                    depart = float(v2.get("depart", 0.0)) + 0.05 * (
                        copy_id + 1
                    )
                    v2.set(
                        "id",
                        (
                            f"{v2.get('id')}_dir"
                            f"{self.directional_demand_source}{copy_id}"
                        ),
                    )
                    v2.set("depart", f"{depart:.2f}")
                    directional_additions.append(v2)
            for v2 in directional_additions:
                root.append(v2)
        os.makedirs(self.cfg.get("log_dir", "./logs"), exist_ok=True)
        temp_path = os.path.join(
            self.cfg.get("log_dir", "./logs"),
            (
                f"temp_demand_{self.demand_scale:.2f}_seed{seed}"
                f"_pid{os.getpid()}.rou.xml"
            ),
        )
        tree.write(temp_path, encoding="UTF-8", xml_declaration=True)
        return temp_path

    def _start_sumo(self):
        if self.sumo is not None:
            return
        if "SUMO_HOME" not in os.environ:
            try:
                import sumo
                os.environ["SUMO_HOME"] = sumo.SUMO_HOME
            except Exception:
                os.environ["SUMO_HOME"] = os.path.dirname(os.path.dirname(traci.__file__))
        sumo_binary = "sumo-gui" if self.use_gui else "sumo"
        cmd = [
            sumo_binary,
            "-c", self.config_file,
            "--no-step-log", "true",
            "--no-warnings", "true",
            "--duration-log.disable", "true",
            "--time-to-teleport", "-1",
            "--seed", str(self.sumo_seed),
        ]
        if self.temp_route_file and os.path.exists(self.temp_route_file):
            cmd.extend(["--route-files", self.temp_route_file])
        traci.start(cmd, label=self.cfg.get("experiment_name", "default"))
        self.sumo = traci.getConnection(self.cfg.get("experiment_name", "default"))

    def _build_topology(self):
        self.tls_ids = list(self.sumo.trafficlight.getIDList())
        self.agent_ids = self.tls_ids
        self.num_agents = len(self.tls_ids)

        for tl_id in self.tls_ids:
            logic = self.sumo.trafficlight.getAllProgramLogics(tl_id)[0]
            # The policy selects stable green phases only. Yellow phases belong
            # to the environment's safety transition, not the action space.
            green_phases = [
                p.state for p in logic.phases
                if ("G" in p.state or "g" in p.state) and "y" not in p.state
            ]
            if not green_phases:
                raise ValueError(f"Traffic light {tl_id} has no controllable green phase")
            self.phases[tl_id] = green_phases
            lanes = self.sumo.trafficlight.getControlledLanes(tl_id)
            self.incoming_lanes[tl_id] = list(dict.fromkeys(lanes))
            self.outgoing_lanes[tl_id] = self._get_outgoing_lanes(tl_id)
            self.current_phase[tl_id] = 0
            self.phase_duration[tl_id] = 0
            self.pending_phase[tl_id] = None
            self.transition_remaining[tl_id] = 0
            self.sumo.trafficlight.setRedYellowGreenState(tl_id, green_phases[0])

    def _get_outgoing_lanes(self, tl_id: str) -> List[str]:
        """Infer outgoing lanes from incoming lanes via connections."""
        outgoing = []
        for lane in self.incoming_lanes[tl_id]:
            conns = self.sumo.lane.getLinks(lane)
            for conn in conns:
                if conn[0] not in outgoing:
                    outgoing.append(conn[0])
        return outgoing

    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.sumo_seed = int(seed)
        if self.sumo is not None:
            try:
                self.sumo.close()
            except Exception:
                pass
        self.sumo = None
        self.demand_spike_seed_offset += 1
        self.temp_route_file = self._create_demand_route_file(
            seed if seed is not None else self.demand_spike_seed_offset
        )
        self._start_sumo()
        self._build_topology()
        self.vehicle_subscriptions = {}
        self.arrived_vehicle_info = []
        self.last_step_arrived_count = 0
        self.cached_valid_observations = {}
        self.burst_failure_state = {
            tl_id: False for tl_id in self.tls_ids
        }
        self.failure_age = {tl_id: 0 for tl_id in self.tls_ids}
        self.last_failure_age = {tl_id: 0.0 for tl_id in self.tls_ids}
        obs = self._get_observations()
        info = {"num_agents": self.num_agents, "tls_ids": self.tls_ids}
        return obs, info

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        bool,
        bool,
        Dict[str, Any]
    ]:
        # Apply actions (phase selection)
        for tl_id, action in actions.items():
            self._apply_action(tl_id, action)

        # Step simulation for delta_time
        arrivals_before = len(self.arrived_vehicle_info)
        for _ in range(self.delta_time):
            self.sumo.simulationStep()
            self._update_phase_durations()
            self._update_vehicle_subscriptions()
        self.last_step_arrived_count = len(self.arrived_vehicle_info) - arrivals_before

        obs = self._get_observations()
        rewards = self._compute_rewards()
        terminated = self.sumo.simulation.getMinExpectedNumber() <= 0 and self.sumo.simulation.getTime() >= self.num_seconds
        truncated = self.sumo.simulation.getTime() >= self.num_seconds
        info = self._get_step_info()
        return obs, rewards, terminated, truncated, info

    def _apply_action(self, tl_id: str, action: int):
        """Switch tl_id to phase action if legal."""
        if action < 0 or action >= len(self.phases[tl_id]):
            raise ValueError(f"Invalid action {action} for traffic light {tl_id}")
        if self.pending_phase[tl_id] is not None:
            return
        if action == self.current_phase[tl_id]:
            return
        # Action legality: minimum green elapsed
        if self.phase_duration[tl_id] < self.min_green:
            return
        if self.yellow_time <= 0:
            self.sumo.trafficlight.setRedYellowGreenState(tl_id, self.phases[tl_id][action])
            self.current_phase[tl_id] = action
            self.phase_duration[tl_id] = 0
            return

        current = self.phases[tl_id][self.current_phase[tl_id]]
        target = self.phases[tl_id][action]
        yellow = "".join(
            "y" if c in "Gg" and t in "rR" else
            c if c in "Gg" and t in "Gg" else
            "r"
            for c, t in zip(current, target)
        )
        self.sumo.trafficlight.setRedYellowGreenState(tl_id, yellow)
        self.pending_phase[tl_id] = action
        self.transition_remaining[tl_id] = self.yellow_time
        self.phase_duration[tl_id] = 0

    def _update_phase_durations(self):
        for tl_id in self.tls_ids:
            if self.pending_phase[tl_id] is not None:
                self.transition_remaining[tl_id] -= 1
                if self.transition_remaining[tl_id] <= 0:
                    action = self.pending_phase[tl_id]
                    self.sumo.trafficlight.setRedYellowGreenState(
                        tl_id, self.phases[tl_id][action]
                    )
                    self.current_phase[tl_id] = action
                    self.pending_phase[tl_id] = None
                    self.transition_remaining[tl_id] = 0
                    self.phase_duration[tl_id] = 0
                continue
            # This method is called after every one-second SUMO simulation step.
            self.phase_duration[tl_id] += 1

    def _update_vehicle_subscriptions(self):
        # Arrived vehicles are removed from SUMO immediately. Persist their last
        # cached subscription result before pruning the active-vehicle cache.
        now = float(self.sumo.simulation.getTime())
        for v in self.sumo.simulation.getArrivedIDList():
            cached = self.vehicle_subscriptions.pop(v, None)
            if cached is None:
                continue
            depart_time = float(cached.get("depart_time", now))
            self.arrived_vehicle_info.append({
                "id": v,
                "travel_time": max(0.0, now - depart_time),
                "time_loss": float(cached.get(traci.constants.VAR_TIMELOSS, 0.0)),
                "waiting_time": float(cached.get(
                    traci.constants.VAR_ACCUMULATED_WAITING_TIME, 0.0
                )),
            })

        for v in self.sumo.simulation.getDepartedIDList():
            self.sumo.vehicle.subscribe(v, [
                traci.constants.VAR_TIMELOSS,
                traci.constants.VAR_ACCUMULATED_WAITING_TIME,
            ])
            self.vehicle_subscriptions[v] = {"depart_time": now}
        for v in list(self.vehicle_subscriptions.keys()):
            try:
                depart_time = self.vehicle_subscriptions[v]["depart_time"]
                result = self.sumo.vehicle.getSubscriptionResults(v) or {}
                result["depart_time"] = depart_time
                self.vehicle_subscriptions[v] = result
            except Exception:
                self.vehicle_subscriptions.pop(v, None)

    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs = {}
        self.last_observation_quality = {}
        self.last_clean_observations = {}
        self.last_observation_masks = {}
        self.last_failure_age = {}
        correlated_failures = set()
        if (
            self.correlated_failure_prob > 0
            and np.random.rand() < self.correlated_failure_prob
            and self.tls_ids
        ):
            anchor = self.tls_ids[np.random.randint(len(self.tls_ids))]
            anchor_position = self.sumo.junction.getPosition(anchor)
            ordered = sorted(
                self.tls_ids,
                key=lambda candidate: (
                    (
                        self.sumo.junction.getPosition(candidate)[0]
                        - anchor_position[0]
                    ) ** 2
                    + (
                        self.sumo.junction.getPosition(candidate)[1]
                        - anchor_position[1]
                    ) ** 2
                ),
            )
            count = max(
                1,
                int(round(
                    len(ordered) * self.correlated_failure_fraction
                )),
            )
            correlated_failures = set(ordered[:count])
        for tl_id in self.tls_ids:
            lanes = self.incoming_lanes[tl_id]
            queues = []
            waits = []
            flows = []
            for lane in lanes:
                # Queue: vehicles with speed < 0.1 m/s
                vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
                q = sum(1 for v in vehs if self.sumo.vehicle.getSpeed(v) < 0.1)
                queues.append(q)
                # Waiting time (accumulated)
                waits.append(self.sumo.lane.getWaitingTime(lane))
                # Flow: vehicle count
                flows.append(self.sumo.lane.getLastStepVehicleNumber(lane))

            clean_queues = list(queues)
            clean_waits = list(waits)

            # Add sensor noise if configured
            if self.sensor_noise_std > 0:
                queues = [max(0, q + np.random.normal(0, self.sensor_noise_std)) for q in queues]
                waits = [max(0, w + np.random.normal(0, self.sensor_noise_std * 5)) for w in waits]

            phase_onehot = np.zeros(len(self.phases[tl_id]), dtype=np.float32)
            phase_onehot[self.current_phase[tl_id]] = 1.0

            clean_feat = np.concatenate([
                np.array(clean_queues, dtype=np.float32),
                np.array(clean_waits, dtype=np.float32),
                np.array(flows, dtype=np.float32),
                phase_onehot,
            ])
            feat = np.concatenate([
                np.array(queues, dtype=np.float32),
                np.array(waits, dtype=np.float32),
                np.array(flows, dtype=np.float32),
                phase_onehot,
            ])
            feature_scale = np.concatenate([
                np.full(
                    len(queues), self.queue_obs_scale, dtype=np.float32
                ),
                np.full(
                    len(waits), self.wait_obs_scale, dtype=np.float32
                ),
                np.full(
                    len(flows), self.flow_obs_scale, dtype=np.float32
                ),
                np.ones_like(phase_onehot),
            ])
            clean_feat = clean_feat / np.maximum(feature_scale, 1e-6)
            feat = feat / np.maximum(feature_scale, 1e-6)

            # Apply observation dropout on lane-based features only
            quality = 1.0
            observation_mask = np.ones_like(feat, dtype=np.float32)
            n_lane_features = len(queues) + len(waits) + len(flows)
            if self.sensor_noise_std > 0:
                observation_mask[:len(queues) + len(waits)] = 0.0
            if self.obs_dropout_prob > 0:
                mask = np.random.rand(n_lane_features) > self.obs_dropout_prob
                feat[:n_lane_features] *= mask.astype(np.float32)
                observation_mask[:n_lane_features] *= mask.astype(np.float32)
                quality *= float(mask.mean())
            burst_failed = self.burst_failure_state.get(tl_id, False)
            if burst_failed:
                burst_failed = not (
                    np.random.rand() < self.burst_failure_recovery_prob
                )
            elif self.burst_failure_start_prob > 0:
                burst_failed = (
                    np.random.rand() < self.burst_failure_start_prob
                )
            self.burst_failure_state[tl_id] = burst_failed
            agent_failed = (
                (
                    self.agent_obs_dropout_prob > 0
                    and np.random.rand() < self.agent_obs_dropout_prob
                )
                or burst_failed
                or tl_id in correlated_failures
            )
            if agent_failed:
                feat[:n_lane_features] = 0.0
                observation_mask[:n_lane_features] = 0.0
                quality = 0.0

            # Failure age is causal metadata: the number of consecutive
            # control steps for which at least one sensor feature was absent.
            # It lets a policy distinguish a fresh loss from a stale outage.
            has_missing_sensor = bool(
                np.any(observation_mask[:n_lane_features] == 0.0)
            )
            age = self.failure_age.get(tl_id, 0) + 1 if has_missing_sensor else 0
            self.failure_age[tl_id] = age
            self.last_failure_age[tl_id] = float(age)

            # Last-observation carry-forward is a causal temporal baseline:
            # it uses only measurements seen before the current failure and
            # keeps the mask at zero so no feature is mislabeled as observed.
            if self.temporal_observation_fallback:
                previous = self.cached_valid_observations.get(tl_id)
                missing = observation_mask[:n_lane_features] == 0.0
                if previous is not None:
                    feat[:n_lane_features][missing] = previous[missing]
                current_cache = (
                    previous.copy()
                    if previous is not None
                    else np.zeros(n_lane_features, dtype=np.float32)
                )
                valid = ~missing
                current_cache[valid] = feat[:n_lane_features][valid]
                self.cached_valid_observations[tl_id] = current_cache

            obs[tl_id] = feat
            self.last_clean_observations[tl_id] = clean_feat
            self.last_observation_masks[tl_id] = observation_mask
            self.last_observation_quality[tl_id] = quality
        return obs

    def _compute_rewards(self) -> Dict[str, float]:
        rewards = {}
        global_delay = 0.0
        global_queue = 0.0
        total_incoming_lanes = 0
        for tl_id in self.tls_ids:
            lanes = self.incoming_lanes[tl_id]
            total_incoming_lanes += len(lanes)
            for lane in lanes:
                delay = self.sumo.lane.getWaitingTime(lane)
                vehs = self.sumo.lane.getLastStepVehicleIDs(lane)
                queue = sum(1 for v in vehs if self.sumo.vehicle.getSpeed(v) < 0.1)
                global_delay += delay
                global_queue += queue
        arrived = self.last_step_arrived_count

        # Normalize to avoid huge magnitudes (scale to roughly [-10, 0])
        norm_delay = global_delay / max(1, total_incoming_lanes) * 0.01
        norm_queue = global_queue / max(1, total_incoming_lanes) * 0.05
        norm_throughput = arrived / max(1, len(self.tls_ids)) * 0.5

        for tl_id in self.tls_ids:
            if self.reward_type == "delay":
                rewards[tl_id] = -norm_delay
            elif self.reward_type == "queue":
                rewards[tl_id] = -norm_queue
            elif self.reward_type == "delay_queue":
                rewards[tl_id] = -(norm_delay + 2.0 * norm_queue)
            elif self.reward_type == "local_delay_queue":
                lanes = self.incoming_lanes[tl_id]
                delay = sum(self.sumo.lane.getWaitingTime(l) for l in lanes) / max(1, len(lanes)) * 0.01
                queue = sum(1 for l in lanes for v in self.sumo.lane.getLastStepVehicleIDs(l)
                            if self.sumo.vehicle.getSpeed(v) < 0.1) / max(1, len(lanes)) * 0.05
                rewards[tl_id] = -(delay + 2.0 * queue)
            elif self.reward_type == "delay_queue_throughput":
                rewards[tl_id] = -(norm_delay + 2.0 * norm_queue) + norm_throughput
            else:
                rewards[tl_id] = -norm_delay
            rewards[tl_id] *= self.reward_scale
        return rewards

    def _get_step_info(self) -> Dict[str, Any]:
        info = {
            "time": self.sumo.simulation.getTime(),
            "vehicles": self.sumo.simulation.getDepartedNumber(),
            "arrived": self.last_step_arrived_count,
        }
        return info

    def get_legal_actions(self) -> Dict[str, np.ndarray]:
        masks = {}
        for tl_id in self.tls_ids:
            mask = np.ones(len(self.phases[tl_id]), dtype=bool)
            if self.pending_phase[tl_id] is not None:
                mask[:] = False
                mask[self.current_phase[tl_id]] = True
                masks[tl_id] = mask
                continue
            if self.phase_duration[tl_id] < self.min_green:
                mask[:] = False
                mask[self.current_phase[tl_id]] = True
            # Enforce max green: must switch if exceeded
            if self.phase_duration[tl_id] >= self.max_green:
                mask[self.current_phase[tl_id]] = False
            masks[tl_id] = mask
        return masks

    def get_adjacency(self, mode: str = "static", k: int = 2) -> np.ndarray:
        """Compute adjacency matrix for dynamic communication graph.

        Args:
            mode: 'static' (grid neighbor), 'distance', or 'flow'.
            k: number of nearest neighbors or distance threshold factor.
        """
        n = self.num_agents
        adj = np.zeros((n, n), dtype=np.float32)
        if mode == "static":
            positions = {tl_id: self.sumo.junction.getPosition(tl_id) for tl_id in self.tls_ids}
            for i, a in enumerate(self.tls_ids):
                for j, b in enumerate(self.tls_ids):
                    if i == j:
                        adj[i, j] = 1.0
                        continue
                    dist = np.linalg.norm(np.array(positions[a]) - np.array(positions[b]))
                    if dist <= k * self.cfg.get("network", {}).get("edge_length", 200):
                        adj[i, j] = 1.0
        elif mode == "flow":
            # Edge weights based on number of vehicles traveling between intersections
            for i, tl_id in enumerate(self.tls_ids):
                adj[i, i] = 1.0
                outgoing = self.outgoing_lanes[tl_id]
                for lane in outgoing:
                    # Find downstream tl controlling this lane
                    for j, other in enumerate(self.tls_ids):
                        if i != j and lane in self.incoming_lanes[other]:
                            flow = self.sumo.lane.getLastStepVehicleNumber(lane)
                            adj[i, j] = max(adj[i, j], min(1.0, flow / 10.0))
        return adj

    def get_arrived_vehicle_info(self) -> List[Dict[str, Any]]:
        """Return metrics for every vehicle that arrived in the current episode."""
        return list(self.arrived_vehicle_info)

    def close(self):
        if self.sumo is not None:
            self.sumo.close()
            self.sumo = None

    def __del__(self):
        self.close()
