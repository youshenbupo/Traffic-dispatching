# RACC: Reliability-Aware Counterfactual Communication

## Paper hypothesis

Always-on neighborhood aggregation can reduce MARL performance when local
observations are already sufficient or when a neighbor is corrupted. RACC
tests whether agents can improve robustness by communicating only when a
message is both reliable and useful.

## Method

Each actor has an unconditional local path and a message path:

`policy_input_i = [local_observation_i, accepted_neighbor_messages_i]`

Rejecting every message therefore recovers a valid decentralized policy.
For candidate edge `j -> i`, RACC predicts:

- sender reliability from `observation_j`;
- pairwise utility from receiver/sender latent features;
- a binary execution gate using a straight-through estimator.

The training loss is:

`L = L_MAPPO + lambda_c L_cost + lambda_r L_reliability + lambda_cf L_counterfactual`

`L_cost` penalizes candidate-edge usage. `L_reliability` is self-supervised by
distinguishing nominal observations from synthetic missing/noisy observations.
`L_counterfactual` removes sampled senders and uses the centralized critic's
value change as a detached target for their outgoing gates.

## Primary claims to test

1. Always-on communication propagates corrupted observations and can be worse
   than no communication.
2. Reliability and counterfactual utility are complementary: each improves
   worst-case performance, while the full method also reduces bandwidth.
3. The local fallback prevents catastrophic degradation as corruption severity
   or network size increases.

## Required comparisons

- Fixed-Time and Max-Pressure.
- Parameter-shared MAPPO without communication.
- Always-on RACC message architecture.
- RACC without reliability supervision.
- RACC without counterfactual supervision.
- Full RACC.
- GAT/CoLight-style always-on communication.

Use at least five training seeds and paired evaluation seeds. Report mean,
standard deviation, 95% confidence interval, and per-seed results for travel
time, accumulated waiting time, queue length, completed-trip throughput,
communication rate, and inference cost.

## Evaluation conditions

- Nominal demand.
- Observation dropout at multiple severities.
- Gaussian sensor noise at multiple severities.
- Correlated intersection failures and communication delay (to implement).
- Demand shifts and incidents.
- Unseen network size/topology.

Training and evaluation perturbations must be configured separately. Model
selection must use validation seeds; final claims must use untouched test seeds.

## Reproducible configs

- `configs/mappo_grid4x4_paper.yaml`
- `configs/racc_grid4x4.yaml`
- `configs/racc_grid4x4_always_comm.yaml`
- `configs/racc_grid4x4_no_reliability.yaml`
- `configs/racc_grid4x4_no_counterfactual.yaml`
- `configs/*_eval_dropout.yaml`
- `configs/*_eval_noise.yaml`

The old checkpoints are not comparable: the corrected environment removes
yellow phases from the policy action space and enforces yellow transitions.

## Masked reconstruction prototype

The current experimental branch adds a bias-free decoder from accepted
neighbor messages to missing local features. Clean observations are privileged
training targets only. A feature mask ensures observed values are never
overwritten, and zero accepted messages produce exactly the corrupted local
observation.

This prototype improves reconstruction error but does not yet improve control
over the same-weight no-message ablation on grid4x4. Reconstruction uncertainty
must be calibrated before this mechanism is presented as the final method.

An uncertainty-calibrated v8 was also tested. It passed grid3x3 but became
overconfident and nearly dense on grid4x4. This establishes that calibration
against state reconstruction error alone is insufficient; the confidence
target must measure decision impact.
