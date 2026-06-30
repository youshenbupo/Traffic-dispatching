# Mid-term Analysis (1h40m into 5h cycle)

## Completed Experiments

### grid4x4 (synthetic 4×4 grid, 16 agents, 1800s simulation)

| Method | Final Queue | Final Reward | Notes |
|--------|-------------|--------------|-------|
| MaxPressure (500-step baseline) | 9.57 | -331k | Strong classical baseline |
| MAPPO | **0.54** | **-602** | Excellent convergence |
| GAT-MAPPO (static) | 7.92 | -249k | **Catastrophic failure** |
| GAT-MAPPO (flow) | 7.92 | -249k | Identical to static |

### hangzhou_4x4 (real-world, 16 agents, 1800s simulation)

| Method | Final Queue | Final Reward | Notes |
|--------|-------------|--------------|-------|
| MaxPressure (200-step baseline) | 1.49 | -12.6k | Very strong baseline |
| MAPPO | 3.38 | -116k | Mediocre, no clear learning |
| GAT-MAPPO (static, partial) | ~3.38 | -116k | Similar to MAPPO |

## Key Findings

1. **MAPPO alone is already very strong** on the synthetic grid, beating MaxPressure by a large margin.
2. **Current GAT implementation is actively harmful** on grid4x4. The policy collapses to high queue / low reward.
3. **Graph type (static vs. flow) does not matter** with the current GAT — results are identical.
4. **Real-world Hangzhou is much harder** for RL; MaxPressure is competitive and RL shows little improvement in 15 episodes.

## Hypothesis for GAT Failure

The original `GATCommLayer` has two weaknesses:
- No residual connection or layer normalization, causing unstable feature magnitudes.
- GAT features are used only by the centralized critic, not by the actor. The actor therefore cannot exploit inter-agent communication, while the critic's biased value estimates corrupt the GAE/advantage computation.

## Innovation Under Test

Implemented an improved GAT module:
- Residual connection + layer normalization.
- Actor receives concatenation of local obs and GAT communication features.
- Update uses per-timestep adjacency instead of reusing the first timestep.

Test config: `configs/gatmappo_grid4x4_static_v2.yaml`.
