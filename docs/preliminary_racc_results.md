# Preliminary corrected-baseline results

These are engineering checks, not paper results. They use grid3x3, one training
seed, short 300-second episodes, and five paired evaluation traffic seeds.

| Training | Method | Travel time | Waiting time | Queue | Throughput | Comm. rate |
|---|---|---:|---:|---:|---:|---:|
| 2 episodes | MAPPO | 99.32 | 25.03 | 0.143 | 33.6 | 0 |
| 2 episodes | RACC v0 | 112.18 | 31.18 | 0.185 | 33.0 | 0.071 |
| 10 episodes | MAPPO | 92.99 | 21.35 | 0.122 | 35.8 | 0 |
| 10 episodes | RACC v1 warm start | 121.44 | 34.12 | 0.221 | 31.0 | 0.956 |

## Diagnosis

- The corrected no-communication MAPPO is a strong baseline.
- A high initial gate threshold made v0 reject almost every message before the
  reliability and counterfactual heads learned.
- Soft warm-up plus a lower threshold made v1 converge to almost all edges.
- The original counterfactual target maps zero marginal value to probability
  0.5. That structurally encourages communication even when a message is not
  useful.

## v2 change

RACC v2 subtracts a positive utility margin before constructing the gate target:

`target = sigmoid((V(full) - V(without_sender) - margin) / temperature)`

Thus a message with no estimated benefit receives a target below 0.5. Formal
experiments should not start until v2 shows that communication rate responds
monotonically to the communication-cost coefficient and corrupted senders are
rejected more often than clean senders.

## Subsequent prototypes

- **v2:** positive counterfactual margin prevented the 0%/100% gate collapse,
  but remained worse than MAPPO.
- **v3:** introduced a zero-initialized residual communication actor.
- **v4:** added whole-agent corruption to reliability self-supervision. A
  single grid3x3 training seed looked very strong, including under agent
  observation dropout.
- **v5:** fixed a confound in the residual actor. Previously, its communication
  branch could produce a correction from local observations even with no
  accepted messages. The corrected branch is exactly zero with no messages.
- **v6:** trained both RACC and MAPPO with 20% agent observation dropout.

## Corrected same-weight ablation

On grid3x3 with five paired evaluation traffic seeds:

| Method | Travel | Waiting | Queue | Throughput | Comm. |
|---|---:|---:|---:|---:|---:|
| RACC v5 full | 90.04 | 17.81 | 0.0815 | 41.0 | 42.5% |
| RACC v5, same weights, no messages | 89.90 | 17.72 | 0.0797 | 41.2 | 0% |
| MAPPO | 95.89 | 23.10 | 0.1478 | 34.2 | 0% |

The improvement over MAPPO is therefore mostly an architecture/training effect,
not a communication effect.

With 20% failure during training and 30% failure at evaluation:

| Method | Travel | Waiting | Queue | Throughput |
|---|---:|---:|---:|---:|
| RACC v6 full | 85.49 | 12.89 | 0.0609 | 41.6 |
| RACC v6, same weights, no messages | 85.52 | 13.11 | 0.0672 | 40.6 |
| Robust MAPPO | 99.50 | 25.26 | 0.1881 | 31.8 |

Communication has a small benefit here, but still explains only a fraction of
the gap to MAPPO.

## Grid4x4 robust multi-seed pilot

Three independent training seeds, each evaluated on the same five traffic
seeds with 30% whole-intersection observation dropout:

| Method | Travel (mean ± train-seed SD) | Waiting | Queue | Throughput | Comm. |
|---|---:|---:|---:|---:|---:|
| Robust MAPPO | 94.56 ± 1.62 | 11.13 ± 1.32 | 0.0245 ± 0.0030 | 232.40 ± 0.20 | 0% |
| RACC full | 94.79 ± 3.12 | 10.65 ± 2.13 | 0.0233 ± 0.0055 | 232.73 ± 0.31 | 91.6% |
| RACC, same weights, no messages | 94.57 ± 4.00 | 10.49 ± 2.74 | 0.0230 ± 0.0067 | 232.93 ± 0.61 | 0% |

The learned communication is not consistently useful and is not sparse. The
counterfactual value target is therefore not yet a viable paper contribution.

## Decision

Do not launch the 100-episode study with the current gate. The next prototype
must give messages an identifiable task that cannot be solved by an extra local
network branch. The recommended direction is masked local-state reconstruction:

1. retain an uncorrupted observation target only during centralized training;
2. mask one agent's local sensors;
3. require accepted neighbor messages to reconstruct its missing lane state;
4. train the action residual from the reconstructed state;
5. report reconstruction error and same-weight no-message performance.

Only proceed to full experiments if communication beats the same-weight
no-message ablation across at least three training seeds.
