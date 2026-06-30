# System Design

## Architecture

```
scripts/           Entry points
train.py           Online/offline training
eval.py            Model evaluation
run_demo.py        Baseline demo
generate_*.py      Data/network generation

src/
envs/sumo_env.py   SUMO multi-agent wrapper
agents/            RL agents and baselines
networks/          NN modules (MLP, GAT, QMIX mixer)
trainers/          Training loops
utils/             Config, logging, metrics

data/
networks/          SUMO network files
scenarios/         Complete SUMO scenarios
datasets/          Offline RL datasets
```

## Extension Interfaces

### Dynamic Communication Graph
- Enabled via `dynamic_graph.enabled: true`
- `MAPPOAgent` uses `GATCommLayer` when enabled
- `SUMOMultiAgentEnv.get_adjacency(mode='static'|'flow')` provides graph structure
- Future: learned adjacency matrix can be added as a new mode

### Robustness
- Perturbations configured under `robustness.perturbations`
- Currently supported: `sensor_noise`, `demand_spike`
- Extend by adding perturbation logic in `SUMOMultiAgentEnv._parse_perturbations`

### Offline RL
- Generate dataset with `scripts/generate_offline_dataset.py`
- Set `offline.enabled: true` and `offline.dataset_path`
- Current supports: DQN/QMIX replay buffer loading
- Future: add CQL/TD3+BC loss in agents
