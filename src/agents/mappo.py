"""Multi-Agent PPO (MAPPO) with parameter sharing and centralized critic."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

from src.networks.base import MLPActor, MLPCritic, ResidualCommActor
from src.networks.gat import GATCommLayer
from src.networks.comm import MeanPoolCommLayer, ReliabilityAwareCommLayer


class MAPPOAgent:
    """Parameter-shared MAPPO for homogeneous intersections."""

    def __init__(self, obs_dim: int, action_dim: int, agent_ids: List[str], config: Dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.cfg = config
        self.agent_cfg = config.get("agent", {})
        self.lr = self.agent_cfg.get("lr", 3e-4)
        self.gamma = self.agent_cfg.get("gamma", 0.99)
        self.gae_lambda = self.agent_cfg.get("gae_lambda", 0.95)
        self.clip_param = self.agent_cfg.get("clip_param", 0.2)
        self.entropy_coef = self.agent_cfg.get("entropy_coef", 0.01)
        self.value_loss_coef = self.agent_cfg.get("value_loss_coef", 0.5)
        self.max_grad_norm = self.agent_cfg.get("max_grad_norm", 0.5)
        self.hidden_dim = self.agent_cfg.get("hidden_dim", 128)
        self.ppo_epochs = 4
        self.batch_size = config.get("training", {}).get("batch_size", 256)

        self.use_gat = config.get("dynamic_graph", {}).get("enabled", False)
        self.gat_heads = config.get("dynamic_graph", {}).get("gat_heads", 4)
        self.comm_type = config.get("dynamic_graph", {}).get("comm_type", "gat")
        self.gated_comm = config.get("dynamic_graph", {}).get("gated_comm", False)
        self.comm_target = config.get("dynamic_graph", {}).get("comm_target", "both")
        graph_cfg = config.get("dynamic_graph", {})
        self.comm_cost_coef = graph_cfg.get("comm_cost_coef", 0.01)
        self.reliability_coef = graph_cfg.get("reliability_coef", 0.1)
        self.reconstruction_coef = graph_cfg.get("reconstruction_coef", 0.0)
        self.counterfactual_coef = graph_cfg.get("counterfactual_coef", 0.1)
        self.counterfactual_samples = graph_cfg.get("counterfactual_samples", 2)
        self.counterfactual_temperature = graph_cfg.get("counterfactual_temperature", 1.0)
        self.counterfactual_margin = graph_cfg.get("counterfactual_margin", 0.05)
        self.comm_warmup_updates = graph_cfg.get("warmup_updates", 5)
        self.update_count = 0
        self.last_comm_stats = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() and config.get("training", {}).get("use_gpu", True) else "cpu")

        if self.use_gat:
            if self.comm_type == "racc":
                self.comm = ReliabilityAwareCommLayer(
                    obs_dim,
                    self.hidden_dim,
                    dropout=graph_cfg.get("dropout", 0.0),
                    gate_threshold=graph_cfg.get("gate_threshold", 0.5),
                ).to(self.device)
            elif self.comm_type == "mean_pool":
                self.comm = MeanPoolCommLayer(obs_dim, self.hidden_dim).to(self.device)
            else:
                self.comm = GATCommLayer(obs_dim, self.hidden_dim, heads=self.gat_heads).to(self.device)
            # Actor receives comm-enhanced features only when targeting 'both'
            if self.comm_type == "racc":
                self.actor = ResidualCommActor(
                    obs_dim, self.hidden_dim, action_dim, self.hidden_dim
                ).to(self.device)
                self.critic = MLPCritic(
                    (obs_dim + self.hidden_dim) * self.n_agents, self.hidden_dim
                ).to(self.device)
            elif self.comm_target == "critic":
                self.actor = MLPActor(obs_dim, action_dim, self.hidden_dim).to(self.device)
                self.critic = MLPCritic(self.hidden_dim * self.n_agents, self.hidden_dim).to(self.device)
            else:
                self.actor = MLPActor(obs_dim + self.hidden_dim, action_dim, self.hidden_dim).to(self.device)
                self.critic = MLPCritic(self.hidden_dim * self.n_agents, self.hidden_dim).to(self.device)
            if self.gated_comm and self.comm_type != "racc":
                self.comm_gate = nn.Linear(obs_dim, self.hidden_dim).to(self.device)
            else:
                self.comm_gate = None
        else:
            self.comm = None
            self.comm_gate = None
            self.actor = MLPActor(obs_dim, action_dim, self.hidden_dim).to(self.device)
            self.critic = MLPCritic(obs_dim * self.n_agents, self.hidden_dim).to(self.device)

        params = list(self.actor.parameters()) + list(self.critic.parameters())
        if self.comm is not None:
            params += list(self.comm.parameters())
        if self.comm_gate is not None:
            params += list(self.comm_gate.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        self.buffer = []

    def act(self, obs: Dict[str, np.ndarray], masks: Dict[str, np.ndarray] = None,
            explore: bool = True, adj: np.ndarray = None,
            obs_mask: Dict[str, np.ndarray] = None,
            clean_obs: Dict[str, np.ndarray] = None) -> Dict[str, int]:
        obs_tensor = torch.FloatTensor(np.stack([obs[aid] for aid in self.agent_ids])).to(self.device)
        if obs_mask is None:
            obs_mask_tensor = torch.ones_like(obs_tensor)
        else:
            obs_mask_tensor = torch.FloatTensor(
                np.stack([obs_mask[aid] for aid in self.agent_ids])
            ).to(self.device)
        mask_tensor = None
        if masks is not None:
            mask_tensor = torch.BoolTensor(np.stack([masks[aid] for aid in self.agent_ids])).to(self.device)

        with torch.no_grad():
            if self.comm is not None and adj is not None:
                adj_t = torch.FloatTensor(adj).unsqueeze(0).to(self.device)
                if self.comm_type == "racc":
                    hard_comm = self.update_count >= self.comm_warmup_updates
                    comm_feats, details = self.comm(
                        obs_tensor.unsqueeze(0), adj_t,
                        return_details=True, hard=hard_comm,
                    )
                    comm_feats = comm_feats.squeeze(0)
                    candidates = details["candidate_mask"].sum().clamp(min=1.0)
                    self.last_comm_stats = {
                        "communication_rate": float(
                            (details["hard_gate"].sum() / candidates).item()
                        ),
                        "mean_reliability": float(
                            details["reliability"].mean().item()
                        ),
                    }
                else:
                    comm_feats = self.comm(obs_tensor.unsqueeze(0), adj_t).squeeze(0)
                if self.comm_gate is not None:
                    gate = torch.sigmoid(self.comm_gate(obs_tensor))
                    comm_feats = gate * comm_feats
                if self.comm_type == "racc":
                    reconstructed = self.comm.reconstruct(
                        comm_feats, obs_tensor, obs_mask_tensor
                    )
                    if clean_obs is not None:
                        clean_tensor = torch.FloatTensor(
                            np.stack([
                                clean_obs[aid] for aid in self.agent_ids
                            ])
                        ).to(self.device)
                        self.last_comm_stats["reconstruction_error"] = float(
                            self.comm.reconstruction_loss(
                                reconstructed,
                                clean_tensor,
                                obs_mask_tensor,
                            ).item()
                        )
                    global_state = torch.cat(
                        [reconstructed, comm_feats], dim=-1
                    ).reshape(1, -1)
                    actor_input = torch.cat(
                        [reconstructed, comm_feats], dim=-1
                    )
                else:
                    global_state = comm_feats.reshape(1, -1)
                if self.comm_type != "racc" and self.comm_target == "critic":
                    actor_input = obs_tensor
                elif self.comm_type != "racc":
                    actor_input = torch.cat([obs_tensor, comm_feats], dim=-1)
            else:
                global_state = obs_tensor.reshape(1, -1)
                actor_input = obs_tensor

            dist = self.actor(actor_input, mask_tensor)
            value = self.critic(global_state).squeeze(0)

            if explore:
                action_tensor = dist.sample()
            else:
                action_tensor = dist.probs.argmax(dim=-1)
            log_prob = dist.log_prob(action_tensor)

        actions = {aid: int(action_tensor[i].item()) for i, aid in enumerate(self.agent_ids)}
        log_probs = {aid: float(log_prob[i].item()) for i, aid in enumerate(self.agent_ids)}
        values = {aid: float(value.item()) for aid in self.agent_ids}  # shared value

        return actions, log_probs, values

    def store_transition(
        self, obs, action, reward, value, log_prob, mask, adj, done,
        clean_obs=None, obs_mask=None,
    ):
        self.buffer.append({
            "obs": obs,
            "action": action,
            "reward": reward,
            "value": value,
            "log_prob": log_prob,
            "mask": mask,
            "adj": adj,
            "done": done,
            "clean_obs": clean_obs if clean_obs is not None else obs,
            "obs_mask": obs_mask,
        })

    def _compute_advantages(self, rewards, values, dones):
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0
                next_non_terminal = 1.0 - dones[t]
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        return returns, advantages

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.batch_size:
            return {"mean_mappo_loss": 0.0}

        # Stack data: [T, n_agents, ...]
        obs = torch.FloatTensor(np.stack([np.stack([t["obs"][aid] for aid in self.agent_ids]) for t in self.buffer])).to(self.device)
        actions = torch.LongTensor(np.stack([[t["action"][aid] for aid in self.agent_ids] for t in self.buffer])).to(self.device)
        old_log_probs = torch.FloatTensor(np.stack([[t["log_prob"][aid] for aid in self.agent_ids] for t in self.buffer])).to(self.device)
        masks = torch.BoolTensor(np.stack([np.stack([t["mask"][aid] for aid in self.agent_ids]) for t in self.buffer])).to(self.device)
        clean_obs = torch.FloatTensor(np.stack([
            np.stack([t.get("clean_obs", t["obs"])[aid] for aid in self.agent_ids])
            for t in self.buffer
        ])).to(self.device)
        obs_masks = torch.FloatTensor(np.stack([
            np.stack([
                (
                    t["obs_mask"][aid]
                    if t.get("obs_mask") is not None
                    else np.ones_like(t["obs"][aid], dtype=np.float32)
                )
                for aid in self.agent_ids
            ])
            for t in self.buffer
        ])).to(self.device)

        rewards = np.array([np.mean([t["reward"][aid] for aid in self.agent_ids]) for t in self.buffer])
        values = np.array([np.mean([t["value"][aid] for aid in self.agent_ids]) for t in self.buffer])
        dones = np.array([t["done"] for t in self.buffer])
        returns, advantages = self._compute_advantages(rewards, values, dones)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = []
        auxiliary_metrics = []
        for _ in range(self.ppo_epochs):
            # Forward
            T, N, _ = obs.shape
            masks_flat = masks.reshape(T * N, -1)
            comm_details = None

            if self.comm is not None:
                # Process each timestep with GAT over agents
                # Use per-timestep adjacency instead of only the first one
                adj = torch.FloatTensor(np.stack([t["adj"] for t in self.buffer])).to(self.device)
                if self.comm_type == "racc":
                    hard_comm = self.update_count >= self.comm_warmup_updates
                    comm_out, comm_details = self.comm(
                        obs, adj, return_details=True, hard=hard_comm
                    )
                else:
                    comm_out = self.comm(obs, adj)  # [T, N, hidden_dim]
                if self.comm_gate is not None:
                    gate = torch.sigmoid(self.comm_gate(obs))
                    comm_out = gate * comm_out
                if self.comm_type == "racc":
                    reconstructed = self.comm.reconstruct(
                        comm_out, obs, obs_masks
                    )
                    global_state = torch.cat(
                        [reconstructed, comm_out], dim=-1
                    ).reshape(T, -1)
                    actor_input = torch.cat(
                        [reconstructed, comm_out], dim=-1
                    )
                else:
                    global_state = comm_out.reshape(T, -1)
                if self.comm_type != "racc" and self.comm_target == "critic":
                    actor_input = obs
                elif self.comm_type != "racc":
                    actor_input = torch.cat([obs, comm_out], dim=-1)
            else:
                global_state = obs.reshape(T, -1)
                actor_input = obs
            values_pred = self.critic(global_state).squeeze(-1)

            dist = self.actor(actor_input.reshape(T * N, -1), masks_flat)
            log_probs = dist.log_prob(actions.reshape(T * N)).reshape(T, N)

            # Parameter sharing turns each (time, agent) pair into a PPO
            # sample. Multiplying all agents' likelihood ratios causes the
            # ratio to explode/vanish exponentially with network size.
            ratio = torch.exp(log_probs - old_log_probs)
            agent_advantages = advantages.unsqueeze(-1).expand_as(ratio)
            surr1 = ratio * agent_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * agent_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, returns)
            entropy_loss = -dist.entropy().reshape(T, N).mean()
            loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss

            comm_cost = torch.zeros((), device=self.device)
            reliability_loss = torch.zeros((), device=self.device)
            reconstruction_loss = torch.zeros((), device=self.device)
            counterfactual_loss = torch.zeros((), device=self.device)
            if self.comm is not None and self.comm_type == "racc" and comm_details is not None:
                candidates = comm_details["candidate_mask"].sum().clamp(min=1.0)
                comm_cost = comm_details["gate_prob"].sum() / candidates
                reliability_loss = self.comm.reliability_loss(obs)
                reconstruction_loss = self.comm.reconstruction_loss(
                    reconstructed, clean_obs, obs_masks
                )

                # Remove sampled senders and use the centralized critic's value
                # change as a detached target for their average outgoing gate.
                terms = []
                sender_ids = torch.randperm(N, device=self.device)[
                    :min(self.counterfactual_samples, N)
                ]
                for sender_id in sender_ids:
                    cf_adj = adj.clone()
                    cf_adj[:, :, sender_id] = 0.0
                    cf_comm, _ = self.comm(
                        obs, cf_adj, return_details=True, hard=hard_comm
                    )
                    cf_reconstructed = self.comm.reconstruct(
                        cf_comm, obs, obs_masks
                    )
                    cf_state = torch.cat(
                        [cf_reconstructed, cf_comm], dim=-1
                    ).reshape(T, -1)
                    cf_value = self.critic(cf_state).squeeze(-1)
                    target = torch.sigmoid(
                        (
                            values_pred.detach()
                            - cf_value.detach()
                            - self.counterfactual_margin
                        )
                        / max(self.counterfactual_temperature, 1e-6)
                    )
                    sender_gate = comm_details["gate_prob"][:, :, sender_id]
                    sender_mask = comm_details["candidate_mask"][:, :, sender_id]
                    valid = sender_mask.sum(dim=1) > 0
                    if valid.any():
                        predicted = (
                            (sender_gate * sender_mask).sum(dim=1)
                            / sender_mask.sum(dim=1).clamp(min=1.0)
                        )
                        terms.append(F.binary_cross_entropy(
                            predicted[valid].clamp(1e-6, 1 - 1e-6),
                            target[valid],
                        ))
                if terms:
                    counterfactual_loss = torch.stack(terms).mean()

                loss = (
                    loss
                    + (
                        0.0 if self.update_count < self.comm_warmup_updates
                        else self.comm_cost_coef
                    ) * comm_cost
                    + self.reliability_coef * reliability_loss
                    + self.reconstruction_coef * reconstruction_loss
                    + self.counterfactual_coef * counterfactual_loss
                )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.max_grad_norm)
            self.optimizer.step()
            total_loss.append(loss.item())
            auxiliary_metrics.append({
                "comm_rate": float(comm_cost.detach().item()),
                "reliability_loss": float(reliability_loss.detach().item()),
                "reconstruction_loss": float(reconstruction_loss.detach().item()),
                "counterfactual_loss": float(counterfactual_loss.detach().item()),
            })

        self.buffer = []
        self.update_count += 1
        result = {"mean_mappo_loss": float(np.mean(total_loss))}
        if self.comm is not None and self.comm_type == "racc" and auxiliary_metrics:
            for key in auxiliary_metrics[0]:
                result[key] = float(np.mean([m[key] for m in auxiliary_metrics]))
        return result

    def behavior_clone(self, dataset: List[Dict], epochs: int = 50, batch_size: int = 256) -> Dict[str, float]:
        """Behavior cloning pretraining from an offline dataset.

        dataset: list of transitions with 'obs' and 'action' dicts.
        """
        if not dataset:
            return {"bc_loss": 0.0}
        # Flatten dataset to per-agent (obs, action, mask) tuples
        pairs = []
        for transition in dataset:
            obs = transition["obs"]
            action = transition["action"]
            # Reconstruct mask from action legality: assume all taken actions were legal
            for aid in self.agent_ids:
                pairs.append((obs[aid], action[aid]))
        if not pairs:
            return {"bc_loss": 0.0}

        obs_arr = np.stack([p[0] for p in pairs])
        act_arr = np.array([p[1] for p in pairs])

        dataset_size = len(pairs)
        losses = []
        for epoch in range(epochs):
            indices = np.random.permutation(dataset_size)
            epoch_losses = []
            for start in range(0, dataset_size, batch_size):
                end = min(start + batch_size, dataset_size)
                batch_idx = indices[start:end]
                obs_batch = torch.FloatTensor(obs_arr[batch_idx]).to(self.device)
                act_batch = torch.LongTensor(act_arr[batch_idx]).to(self.device)

                # No masks: assume all actions legal in offline data
                if self.comm is not None and self.comm_target != "critic":
                    # Offline samples may not contain synchronized graph
                    # neighborhoods. Pretrain the guaranteed local fallback.
                    obs_batch = torch.cat([
                        obs_batch,
                        torch.zeros(
                            obs_batch.shape[0], self.hidden_dim,
                            device=self.device,
                        ),
                    ], dim=-1)
                dist = self.actor(obs_batch)
                loss = -dist.log_prob(act_batch).mean()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], self.max_grad_norm)
                self.optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(np.mean(epoch_losses))
        return {"bc_loss": float(np.mean(losses))}

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "comm": self.comm.state_dict() if self.comm else None,
            "update_count": self.update_count,
        }, os.path.join(path, "mappo.pth"))

    def load(self, path: str):
        ckpt = torch.load(os.path.join(path, "mappo.pth"), map_location=self.device)
        self._load_state_dict_flexible(self.actor, ckpt["actor"], "actor")
        self._load_state_dict_flexible(self.critic, ckpt["critic"], "critic")
        if self.comm and ckpt.get("comm"):
            self._load_state_dict_flexible(self.comm, ckpt["comm"], "comm")
        self.update_count = int(ckpt.get("update_count", self.comm_warmup_updates))

    @staticmethod
    def _load_state_dict_flexible(model, state_dict, name):
        """Load compatible parameters and warn about mismatches.

        Allows transfer learning across observation/action dimensions.
        """
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state_dict.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                skipped.append(k)
        model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[transfer] {name}: skipped {len(skipped)} mismatched params: {skipped}")

    def freeze_actor(self):
        for p in self.actor.parameters():
            p.requires_grad = False

    def unfreeze_actor(self):
        for p in self.actor.parameters():
            p.requires_grad = True

    def freeze_critic(self):
        for p in self.critic.parameters():
            p.requires_grad = False

    def unfreeze_critic(self):
        for p in self.critic.parameters():
            p.requires_grad = True
