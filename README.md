# Robust Offline-to-Online MARL with Dynamic Communication Graphs for Traffic Signal Control

Research codebase for the AAMAS/AAAI/IJCAI paper:
**"Robust Offline-to-Online Multi-Agent Reinforcement Learning with Dynamic Communication Graphs for Traffic Signal Control"**

## Project Structure

```
light_RL/
├── configs/               # YAML experiment configs
├── data/
│   ├── networks/          # SUMO networks (.net.xml)
│   ├── scenarios/         # SUMO scenario folders (net + route + config)
│   └── datasets/          # Offline RL datasets
├── docs/                  # Research notes and design docs
├── logs/                  # Training logs / checkpoints
├── results/               # Evaluation results / CSV metrics
├── scripts/               # Entry-point scripts
├── src/
│   ├── agents/            # RL agents and baselines
│   ├── envs/              # SUMO multi-agent wrappers
│   ├── networks/          # Neural network modules (GAT, QMIX mixer)
│   ├── trainers/          # Online/offline trainers
│   └── utils/             # Config, logging, metrics, CSV logger
├── tests/                 # Unit tests
├── tools/                 # External tools (SUMO binaries if needed)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Environment Setup

```bash
# Activate the pre-created environment and set GPUs 2,3
source scripts/setup_env.sh 2,3

# Or manually:
conda activate light_rl
export CUDA_VISIBLE_DEVICES=2,3
```

### 2. Verify Installation

```bash
python -c "import torch, traci; print(torch.__version__, torch.cuda.is_available())"
sumo --version
```

### 3. Generate SUMO Scenarios

```bash
# 3x3, 4x4, 5x5 grid networks
python scripts/generate_grid_network.py --rows 3 --cols 3 --out_dir data/scenarios/grid3x3 --demand 1500
python scripts/generate_grid_network.py --rows 4 --cols 4 --out_dir data/scenarios/grid4x4 --demand 2500
python scripts/generate_grid_network.py --rows 5 --cols 5 --out_dir data/scenarios/grid5x5 --demand 4000
```

### 4. Download Public Real-World Datasets

We use the LibSignal cross-simulator benchmark (Hangzhou, Jinan, synthetic grids, etc.).

```bash
python scripts/download_data.py --dataset libsignal --out_dir data/datasets
```

This performs a sparse git checkout of only the `data/raw_data` folder (~300 MB).
SUMO-ready scenarios include:

- `hangzhou_4x4_gudang_18041610_1h` (used by the example configs)
- `arterial4x4`, `grid4x4`
- `atlanta_1x5`
- `cologne1`, `cologne3`
- `ingolstadt21`
- `manhattan_28x7`
- Various `hangzhou_1x1_*` single-intersection scenarios
- `hangzhou_4x4_hetero` (uses `*_m.sumocfg` naming)

### 5. Run Baseline Demos

```bash
python scripts/run_demo.py --scenario data/scenarios/grid3x3 --controller fixed_time --steps 1000
python scripts/run_demo.py --scenario data/scenarios/grid3x3 --controller max_pressure --steps 1000

# Compare all baselines
python scripts/run_baselines.py --config configs/mappo_grid3x3.yaml --num_steps 1000
```

### 6. Train MAPPO (uses GPUs 2 and 3)

```bash
# Synthetic grid
python scripts/train.py --config configs/mappo_grid3x3.yaml --gpus 2,3

# Real-world Hangzhou 4x4
python scripts/train.py --config configs/mappo_hangzhou4x4.yaml --gpus 2,3
```

### 7. Train GAT-based MAPPO

```bash
# Synthetic grid
python scripts/train.py --config configs/test_gat.yaml --gpus 2,3

# Real-world Hangzhou 4x4
python scripts/train.py --config configs/gatmappo_hangzhou4x4.yaml --gpus 2,3
```

### 8. Generate Offline Dataset and Offline-to-Online Training

```bash
# Step 1: collect offline data with max-pressure
python scripts/generate_offline_dataset.py --config configs/mappo_grid3x3.yaml \
    --policy max_pressure --num_episodes 50 \
    --output data/datasets/grid3x3_offline.pkl

# Step 2: enable offline in config and train
python scripts/train.py --config configs/mappo_grid3x3.yaml --gpus 2,3
```

### 9. Evaluate Trained Model

```bash
python scripts/eval.py --config configs/mappo_grid3x3.yaml \
    --model_path logs/<exp_name>/best_model --gpus 2,3 --num_episodes 10
```

## Datasets

- **Synthetic grids**: generated under `data/scenarios/`.
- **Public real-world scenarios**: downloaded from [LibSignal](https://github.com/DaRL-LibSignal/LibSignal) into `data/datasets/libsignal/`.

## Implemented Components

### Environments
- SUMO multi-agent TSC wrapper (`SUMOMultiAgentEnv`)
- State: queue length, waiting time, current phase, traffic flow
- Action: phase selection
- Reward: normalized delay + queue + throughput
- Action masking for min/max green and phase legality

### Baselines
- Fixed-time control
- Max-pressure control
- DQN / Double DQN
- PPO
- MAPPO (parameter-shared, centralized critic)
- QMIX
- GAT-based MAPPO (CoLight-style dynamic communication)

### Extension Interfaces
- **Dynamic communication graph**: enable via `dynamic_graph.enabled`
- **Robustness perturbations**: configure under `robustness.perturbations`
- **Offline-to-online RL**: generate dataset + enable `offline.enabled`

## GPU Restriction

All scripts respect `CUDA_VISIBLE_DEVICES`. The recommended way is:

```bash
source scripts/setup_env.sh 2,3
```

This restricts PyTorch to logical GPUs 0,1 which map to physical GPUs 2,3.

## Citation

TBD
