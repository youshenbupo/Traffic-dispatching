"""Offline-to-online MARL trainer."""
import os
import pickle
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm


class OfflineTrainer:
    """Train agent from offline dataset, optionally fine-tune online."""

    def __init__(self, env, agent, config, logger):
        self.env = env
        self.agent = agent
        self.cfg = config
        self.logger = logger
        self.offline_cfg = config.get("offline", {})
        self.dataset_path = self.offline_cfg.get("dataset_path")
        self.offline_epochs = self.offline_cfg.get("offline_epochs", 100)
        self.online_finetune = self.offline_cfg.get("online_finetune", True)
        self.num_online_episodes = config.get("training", {}).get("num_episodes", 50)

    def load_dataset(self) -> List[Dict[str, Any]]:
        if self.dataset_path and os.path.exists(self.dataset_path):
            with open(self.dataset_path, "rb") as f:
                return pickle.load(f)
        else:
            self.logger.warning("No offline dataset found. Please run generate_offline_dataset.py first.")
            return []

    def offline_train(self):
        dataset = self.load_dataset()
        if not dataset:
            return
        # For MAPPO, use behavior cloning on actor if available
        if hasattr(self.agent, "behavior_clone"):
            loss_info = self.agent.behavior_clone(dataset, epochs=self.offline_epochs)
            self.logger.info(f"Offline BC pretraining: {loss_info}")
        else:
            # For DQN/QMIX, load transitions directly
            if hasattr(self.agent, "buffer"):
                self.agent.buffer.extend(dataset)
            for epoch in tqdm(range(1, self.offline_epochs + 1), desc="Offline training"):
                loss_info = self.agent.update()
                if epoch % 10 == 0:
                    self.logger.info(f"Offline epoch {epoch}: {loss_info}")

        self.agent.save(os.path.join(self.cfg.get("log_dir", "./logs"), "offline_model"))

    def online_finetune_run(self):
        from .online_trainer import OnlineTrainer
        online_trainer = OnlineTrainer(self.env, self.agent, self.cfg, self.logger)
        online_trainer.run()

    def run(self):
        self.offline_train()
        if self.online_finetune:
            self.online_finetune_run()
