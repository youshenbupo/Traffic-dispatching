"""Online MARL trainer."""
import os
import numpy as np
from typing import Dict, Any
from tqdm import tqdm

from src.utils.metrics import MetricsTracker
from src.utils.csv_logger import CSVLogger


def _agent_act(
    agent, obs, masks, adj, explore=True, obs_mask=None, clean_obs=None,
    failure_age=None,
):
    """Call agent.act and normalize return to (actions, log_probs, values)."""
    import inspect
    sig = inspect.signature(agent.act)
    kwargs = {"explore": explore}
    if "adj" in sig.parameters:
        kwargs["adj"] = adj
    if "obs_mask" in sig.parameters:
        kwargs["obs_mask"] = obs_mask
    if "clean_obs" in sig.parameters:
        kwargs["clean_obs"] = clean_obs
    if "failure_age" in sig.parameters:
        kwargs["failure_age"] = failure_age
    result = agent.act(obs, masks, **kwargs)
    if isinstance(result, tuple):
        actions, log_probs, values = result
        return actions, log_probs, values
    else:
        actions = result
        return actions, None, None


class OnlineTrainer:
    def __init__(self, env, agent, config, logger):
        self.env = env
        self.agent = agent
        self.cfg = config
        self.logger = logger
        self.num_episodes = config.get("training", {}).get("num_episodes", 200)
        self.update_interval = config.get("training", {}).get("update_interval", 100)
        self.eval_interval = config.get("training", {}).get("eval_interval", 20)
        self.save_interval = config.get("training", {}).get("save_interval", 50)
        graph_cfg = config.get("dynamic_graph", {})
        self.graph_type = graph_cfg.get("graph_type", "static")
        self.neighbor_k = graph_cfg.get("neighbor_k", 1)
        self.log_dir = config.get("log_dir", "./logs")
        self.result_dir = config.get("result_dir", "./results")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        self.csv_logger = CSVLogger(self.result_dir, "training_metrics.csv")

    def run(self):
        best_metric = float("inf")
        for ep in tqdm(range(1, self.num_episodes + 1), desc="Training"):
            obs, info = self.env.reset(seed=self.cfg.get("seed", 42) + ep)
            episode_reward = 0.0
            step_count = 0
            done = False
            train_tracker = MetricsTracker()

            while not done:
                masks = self.env.get_legal_actions()
                clean_obs = {
                    aid: value.copy()
                    for aid, value in self.env.last_clean_observations.items()
                }
                obs_mask = {
                    aid: value.copy()
                    for aid, value in self.env.last_observation_masks.items()
                }
                failure_age = dict(
                    getattr(self.env, "last_failure_age", {})
                )
                adj = self.env.get_adjacency(
                    mode=self.graph_type, k=self.neighbor_k
                )
                actions, log_probs, values = _agent_act(
                    self.agent, obs, masks, adj, explore=True,
                    obs_mask=obs_mask, clean_obs=clean_obs,
                    failure_age=failure_age,
                )

                if hasattr(self.agent, "store_transition") and log_probs is not None:
                    import inspect
                    transition = {
                        "obs": obs,
                        "action": actions,
                        "reward": {aid: 0.0 for aid in self.env.agent_ids},
                        "value": values,
                        "log_prob": log_probs,
                        "mask": masks,
                        "done": False,
                    }
                    parameters = inspect.signature(
                        self.agent.store_transition
                    ).parameters
                    if "adj" in parameters:
                        transition["adj"] = adj
                    if "clean_obs" in parameters:
                        transition["clean_obs"] = clean_obs
                    if "obs_mask" in parameters:
                        transition["obs_mask"] = obs_mask
                    if "failure_age" in parameters:
                        transition["failure_age"] = failure_age
                    self.agent.store_transition(**transition)

                next_obs, rewards, terminated, truncated, info_step = self.env.step(actions)
                done = terminated or truncated
                episode_reward += sum(rewards.values())
                step_count += 1
                step_metrics = MetricsTracker.compute_from_env(self.env)
                step_metrics.update(getattr(self.agent, "last_comm_stats", {}))
                train_tracker.record_step({"reward": sum(rewards.values()), **step_metrics})

                # Update stored reward with actual reward
                if hasattr(self.agent, "buffer") and len(self.agent.buffer) > 0 and log_probs is not None:
                    self.agent.buffer[-1]["reward"] = rewards
                    self.agent.buffer[-1]["done"] = float(done)

                # DQN/QMIX store in replay buffer
                if hasattr(self.agent, "store"):
                    self.agent.store(obs, actions, rewards, next_obs, float(done))

                obs = next_obs

                # Replay-buffer algorithms may update within an episode.
                # On-policy PPO/MAPPO must keep a complete rollout so GAE does
                # not incorrectly bootstrap a truncated chunk with value zero.
                if log_probs is None and step_count % self.update_interval == 0:
                    loss_info = self.agent.update()
                    if loss_info:
                        self.logger.info(f"Episode {ep} step {step_count} update: {loss_info}")

            # End of episode
            if hasattr(self.agent, "buffer") and len(self.agent.buffer) > 0 and log_probs is not None:
                self.agent.buffer[-1]["done"] = 1.0
            loss_info = self.agent.update()
            arrived = self.env.get_arrived_vehicle_info()
            train_tracker.finalize(arrived)
            train_metrics = train_tracker.aggregate()
            self.logger.info(f"Episode {ep} finished: steps={step_count}, reward={episode_reward:.2f}, loss={loss_info}")
            self.csv_logger.log({
                "episode": ep,
                "phase": "train",
                "episode_reward": episode_reward,
                "episode_length": step_count,
                **loss_info,
                **train_metrics,
            })

            if ep % self.eval_interval == 0:
                eval_metrics = self.evaluate()
                eval_metrics["episode"] = ep
                eval_metrics["phase"] = "eval"
                self.csv_logger.log(eval_metrics)
                self.logger.info(f"Episode {ep} eval: {eval_metrics}")
                if eval_metrics.get("average_travel_time", float("inf")) < best_metric:
                    best_metric = eval_metrics["average_travel_time"]
                    self.agent.save(os.path.join(self.log_dir, "best_model"))

            if ep % self.save_interval == 0:
                self.agent.save(os.path.join(self.log_dir, f"checkpoint_{ep}"))

        # Save final model for downstream transfer / evaluation
        self.agent.save(os.path.join(self.log_dir, "final_model"))
        self.env.close()

    def evaluate(self, num_episodes: int = None) -> Dict[str, float]:
        num_episodes = num_episodes or self.cfg.get("training", {}).get("num_eval_episodes", 5)
        metrics_list = []
        for ep in range(num_episodes):
            obs, info = self.env.reset(seed=10000 + ep)
            tracker = MetricsTracker()
            done = False
            while not done:
                masks = self.env.get_legal_actions()
                adj = self.env.get_adjacency(
                    mode=self.graph_type, k=self.neighbor_k
                )
                obs_mask = {
                    aid: value.copy()
                    for aid, value in self.env.last_observation_masks.items()
                }
                clean_obs = {
                    aid: value.copy()
                    for aid, value in self.env.last_clean_observations.items()
                }
                failure_age = dict(
                    getattr(self.env, "last_failure_age", {})
                )
                actions, _, _ = _agent_act(
                    self.agent, obs, masks, adj, explore=False,
                    obs_mask=obs_mask, clean_obs=clean_obs,
                    failure_age=failure_age,
                )
                next_obs, rewards, terminated, truncated, info_step = self.env.step(actions)
                done = terminated or truncated
                step_metrics = MetricsTracker.compute_from_env(self.env)
                step_metrics.update(getattr(self.agent, "last_comm_stats", {}))
                tracker.record_step({"reward": sum(rewards.values()), **step_metrics})
                obs = next_obs
            arrived = self.env.get_arrived_vehicle_info()
            tracker.finalize(arrived)
            metrics_list.append(tracker.aggregate())

        # Average metrics
        result = {}
        for key in metrics_list[0]:
            result[key] = float(np.mean([m[key] for m in metrics_list]))
        return result
