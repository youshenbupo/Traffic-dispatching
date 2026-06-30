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

## v7: masked local-state reconstruction

v7 retains clean observations only during centralized training. For missing
features, a bias-free decoder predicts the local state from accepted neighbor
messages. The policy consumes the reconstructed local state, while rejecting
all messages still gives an exact corrupted-local fallback.

### Grid3x3 admission test

Three training seeds, five paired evaluation traffic seeds, 30% whole-agent
observation dropout:

| Method | Travel | Waiting | Queue | Reconstruction error | Comm. |
|---|---:|---:|---:|---:|---:|
| v7 full | 84.20 | 12.99 | 0.0648 | 0.1374 | 36.6% |
| v7 same weights, no messages | 84.72 | 13.39 | 0.0663 | 0.1452 | 0% |

Communication improved travel time, waiting time, and reconstruction error in
all three training seeds, so v7 passed the small-scale admission test.

### Grid4x4 multi-seed result

Three training seeds, five paired evaluation traffic seeds, 30% whole-agent
observation dropout:

| Method | Travel | Waiting | Queue | Throughput | Reconstruction error | Comm. |
|---|---:|---:|---:|---:|---:|---:|
| Robust MAPPO | 98.11 | 11.57 | 0.0253 | 229.07 | — | 0% |
| v7 full | 96.17 | 9.80 | 0.0209 | 229.40 | 0.0545 | 57.4% |
| v7 same weights, no messages | 95.13 | 9.28 | 0.0198 | 229.87 | 0.0573 | 0% |

The decoder reduced reconstruction error in every seed, but full communication
made every control metric worse than the same-weight no-message policy in every
seed. The apparent advantage over MAPPO therefore still cannot be attributed
to communication.

## Updated decision

v7 demonstrates that neighbor messages contain predictive information, but
blindly substituting reconstructed values introduces enough bias to harm
control. Do not scale v7 to the full study.

The next candidate should use uncertainty-calibrated imputation:

1. predict both missing state and reconstruction uncertainty;
2. keep missing local features at their fallback values unless confidence is
   high;
3. expose reconstruction confidence to the policy;
4. penalize control sensitivity to low-confidence reconstructions;
5. repeat the same-weight full/no-message test before comparison with MAPPO.

## v8: uncertainty-calibrated imputation

v8 predicts a per-feature confidence together with each reconstructed value.
Both the imputed local state and the residual message feature are scaled by
confidence. Zero messages still produce zero confidence and the exact fallback.

### Grid3x3 admission test

Three training seeds and five paired evaluation seeds:

| Method | Travel | Waiting | Queue | Reconstruction error | Confidence | Comm. |
|---|---:|---:|---:|---:|---:|---:|
| v8 full | 95.08 | 21.68 | 0.1127 | 0.1722 | 0.483 | 51.2% |
| v8 same weights, no messages | 95.81 | 22.32 | 0.1186 | 0.1812 | 0 | 0% |

Travel time, waiting time, and reconstruction error improved in all three
training seeds, so v8 passed the small-scale test.

### Grid4x4 multi-seed result

| Method | Travel | Waiting | Queue | Throughput | Recon. error | Confidence | Comm. |
|---|---:|---:|---:|---:|---:|---:|---:|
| Robust MAPPO | 96.38 | 11.38 | 0.0245 | 225.53 | — | — | 0% |
| v8 full | 93.74 | 9.83 | 0.0209 | 226.47 | 0.0519 | 0.857 | 94.5% |
| v8 same weights, no messages | 93.24 | 9.26 | 0.0197 | 226.53 | 0.0541 | 0 | 0% |

Only one of three training seeds benefited from communication. On average,
full communication was worse than the same-weight fallback on every control
metric. Confidence also saturated at 0.857 and produced almost dense
communication.

## v8 decision

Reconstruction confidence is not control confidence. A prediction can have
small state error yet move the signal policy in a harmful direction. The next
gate target must be decision-aware: compare the imputed policy against a clean
observation teacher and open communication only when it reduces policy
divergence or improves a control-value estimate.

## v9: co-adapting decision teacher

v9 supervises communication with the KL divergence to the policy evaluated on
privileged clean observations. A soft-message counterfactual prevents hard
gates from losing all gradients. On grid4x4, three training seeds and five
common evaluation seeds at 30% whole-agent dropout gave:

| Method | Travel | Waiting | Queue | Throughput | Comm. |
|---|---:|---:|---:|---:|---:|
| v9 full | 99.96 | 13.01 | 0.0343 | 236.33 | 92.7% |
| v9 same weights, no messages | 99.91 | 12.98 | 0.0342 | 236.47 | 0% |

Raising the inference threshold from 0.35 to 0.50 reduced communication to
about 5% without materially changing control. The teacher and student
co-adapted, policy-divergence utility stayed near zero, and communication again
provided no reproducible benefit.

## Temporal fallback diagnostic

A causal last-valid-observation fallback was tested as an alternative source
of information. Against robust MAPPO with the same three training seeds,
training budget, and five evaluation seeds:

| Method | Travel | Waiting | Queue | Throughput |
|---|---:|---:|---:|---:|
| Robust MAPPO | 95.99 ± 1.88 | 11.32 ± 1.52 | 0.0254 ± 0.0038 | 236.60 ± 0.40 |
| Temporal fallback | 95.31 ± 2.41 | 11.31 ± 1.50 | 0.0262 ± 0.0044 | 236.67 ± 0.46 |

The fixed stale-state rule gives only a 0.7% travel-time improvement and makes
queueing slightly worse. Temporal information remains plausible, but it needs
a learned belief with staleness uncertainty rather than carry-forward.

## v10: frozen clean-policy teacher

v10 freezes a separately trained robust MAPPO teacher and initializes the
student local policy from it. The corrupted student is trained to approach the
teacher's clean-observation action distribution. To control for the extra ten
training episodes, the original robust MAPPO models were also continued for
the same budget.

| Method | Travel | Waiting | Queue | Throughput | Comm. |
|---|---:|---:|---:|---:|---:|
| Continued robust MAPPO | 93.80 ± 0.72 | 10.08 ± 0.68 | 0.0226 ± 0.0017 | 237.07 ± 0.50 | 0% |
| v10 full | **90.92 ± 0.71** | **8.24 ± 0.68** | **0.0183 ± 0.0020** | 237.40 ± 0.53 | 8.1% |
| v10 same weights, no messages | 91.08 ± 0.86 | 8.35 ± 0.83 | 0.0186 ± 0.0024 | **237.47 ± 0.42** | 0% |

Teacher anchoring improves travel time by 3.1%, waiting by 18.2%, and queueing
by about 19% over the equal-budget baseline. Communication itself adds only a
small mean improvement and changes behavior materially in one of three seeds.
The promising contribution is therefore privileged decision distillation for
robust TSC, not yet sparse communication. The next mandatory ablation must
separate teacher initialization, fixed-teacher distillation, reconstruction,
and communication during training.
