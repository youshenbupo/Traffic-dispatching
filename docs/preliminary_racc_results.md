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
