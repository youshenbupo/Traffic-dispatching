"""Pre-train MAPPO actor via behavior cloning on an offline dataset."""
import os
import sys
import argparse
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.envs.sumo_env import SUMOMultiAgentEnv
from src.agents import MAPPOAgent


def pretrain(config, dataset_path: str, output_path: str, epochs: int = 50, batch_size: int = 256):
    env = SUMOMultiAgentEnv(config)
    env.reset(seed=config.get("seed", 42))
    agent = MAPPOAgent(
        env._get_observations()[env.agent_ids[0]].shape[0],
        len(env.phases[env.agent_ids[0]]),
        env.agent_ids,
        config,
    )

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # Flatten dataset to (obs, action) pairs for all agents
    obs_list, action_list = [], []
    for transition in dataset:
        for aid in agent.agent_ids:
            obs_list.append(transition["obs"][aid])
            action_list.append(transition["action"][aid])

    obs = torch.FloatTensor(np.stack(obs_list)).to(agent.device)
    actions = torch.LongTensor(action_list).to(agent.device)

    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    n = len(obs_list)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_obs = obs[idx]
            batch_actions = actions[idx]

            # Get actor input shape right (depends on comm)
            if agent.comm is not None:
                # Use zero communication for BC pretraining
                comm_zero = torch.zeros(batch_obs.size(0), agent.hidden_dim, device=agent.device)
                actor_input = torch.cat([batch_obs, comm_zero], dim=-1)
            else:
                actor_input = batch_obs

            logits = agent.actor.net(actor_input)
            loss = criterion(logits, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(dim=-1) == batch_actions).sum().item()

        acc = correct / n
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss / n:.4f}, acc={acc:.4f}")

    torch.save({"actor": agent.actor.state_dict()}, output_path)
    print(f"Saved pretrained actor to {output_path}")
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()

    config = load_config(args.config)
    pretrain(config, args.dataset, args.output, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
