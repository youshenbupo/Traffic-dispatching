# Experiment Log

This document is the running lab notebook for the traffic-light control paper
work. Every non-trivial run should record: purpose, command/config, produced
files, metrics, interpretation, and next action.

## GitHub synchronization note

Current reliable workflow:

1. Develop and run experiments on the server: `/data/users/jyh/light_RL`.
2. Export a bundle to the local machine when needed.
3. Push from the local clone `D:\RL-light\.github-push` to
   `https://github.com/youshenbupo/Traffic-dispatching.git`.

This works, but it is fragile when the local machine cannot reach GitHub.
Configuring GitHub SSH access directly on the server is recommended. After that,
the server can use a remote such as:

```bash
git remote set-url origin git@github.com:youshenbupo/Traffic-dispatching.git
git push origin main
```

The server does not strictly need GitHub SSH for experiments, but direct SSH
push from the server will make synchronization simpler and less dependent on
the local Windows network.

## 2026-07-03: 20-seed semantic spillback-risk data and first control probes

### Goal

Move beyond reactive queue shields and test whether a proactive semantic
spillback-risk model can support an AAMAS/期刊-level algorithmic contribution.

The target idea was:

- use lane-level semantic tokens;
- predict future spillback/congestion before gridlock;
- use the predicted risk to decide when a failed-observation controller should
  switch from observed policy actions to a temporal fallback.

### Code changes

Relevant commits on the server:

- `d979096` — add spillback risk evaluation diagnostics.
- `640cae4` — support multi-seed spillback risk validation.
- `2b38778` — add auxiliary spillback target training.
- `2391d27` — add risk-model controlled fallback probe.
- `803e131` — add persistent risk fallback hold.

As of this entry, the server is at `803e131`. GitHub was last successfully
pushed to `640cae4`; later pushes failed because the local machine could not
connect to `github.com:443`.

### Data collection

Collected observed-policy semantic traces for seeds `62000` through `62019`.
Long single-process/tmux runs were unstable, so traces were collected in stable
2-seed batches:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds <two-seed-batch> \
  --control_source observed \
  --include_semantic_tokens \
  --gpus 0 \
  --output results/oracle_recoverability/semantic_risk_batch_<start>_<end>.json
```

Output files:

- `results/oracle_recoverability/semantic_risk_batch_62000_62001.json`
- ...
- `results/oracle_recoverability/semantic_risk_batch_62018_62019.json`

Observed total-time-spent highlights:

| Seed | Observed total time |
|---:|---:|
| 62000 | 365255 |
| 62005 | 350335 |
| 62009 | 362575 |
| 62016 | 370915 |

These four seeds are the main observed-policy failure/gridlock cases in the
20-seed cohort.

### Dataset build: horizon 120

Command:

```bash
python scripts/build_spillback_dataset.py \
  --inputs 'results/oracle_recoverability/semantic_risk_batch_*.json' \
  --output results/oracle_recoverability/spillback_20seed_v1.npz \
  --history 60 \
  --horizon 120 \
  --stride 5
```

Result:

- samples: `2180`
- sequence shape: `(2180, 60, 3, 8, 12)`
- positive rate: `0.0486`

Per-seed positive labels:

| Seed | Samples | Positives | First positive step |
|---:|---:|---:|---:|
| 62000 | 109 | 26 | 419 |
| 62005 | 109 | 24 | 484 |
| 62009 | 109 | 26 | 419 |
| 62016 | 109 | 30 | 184 |
| other 16 seeds | 109 each | 0 | — |

Interpretation: the label aligns with observed bad seeds, but for several seeds
the first positive label is relatively late.

### Binary semantic risk predictor

Command:

```bash
python scripts/train_spillback_risk.py \
  --dataset results/oracle_recoverability/spillback_20seed_v1.npz \
  --output results/oracle_recoverability/spillback_20seed_model_v1.pth \
  --epochs 30 \
  --hidden_dim 64 \
  --batch_size 64 \
  --validation_seeds 62016,62017,62018,62019 \
  --gpus 0
```

Evaluation:

```bash
python scripts/evaluate_spillback_risk.py \
  --dataset results/oracle_recoverability/spillback_20seed_v1.npz \
  --model results/oracle_recoverability/spillback_20seed_model_v1.pth \
  --output results/oracle_recoverability/spillback_20seed_eval_v1.json \
  --gpus 0
```

Validation seeds: `62016,62017,62018,62019`.

Metrics:

- ROC-AUC: `0.8689`
- Average precision: `0.4128`
- BCE: `0.9270`
- threshold 0.5:
  - precision: `0.3210`
  - recall: `0.8667`
  - F1: `0.4685`

Per-seed alert behavior at threshold 0.5:

| Seed | First positive | First alert | Lead steps |
|---:|---:|---:|---:|
| 62016 | 184 | 234 | -50 |
| 62017 | — | 279 | false alert |
| 62018 | — | 414 | false alert |
| 62019 | — | — | no alert |

Interpretation: the model has ranking signal, but thresholded early warning is
not yet good enough. Low threshold can alert earlier but causes false alerts;
high threshold reduces false alerts but alerts too late.

### Auxiliary/multitask risk predictor

Added auxiliary prediction of:

- future queue mean;
- future queue peak;
- future active vehicle growth;
- future departures.

Command:

```bash
python scripts/train_spillback_risk.py \
  --dataset results/oracle_recoverability/spillback_20seed_v1.npz \
  --output results/oracle_recoverability/spillback_20seed_aux02_model_v1.pth \
  --epochs 30 \
  --hidden_dim 64 \
  --batch_size 64 \
  --validation_seeds 62016,62017,62018,62019 \
  --auxiliary_weight 0.2 \
  --gpus 0
```

Metrics:

- ROC-AUC: `0.8690`
- Average precision: `0.3663`

Interpretation: auxiliary targets did not improve the binary risk ranking.
However, the predicted future queue mean was useful for diagnosis: true bad
cases often predicted qmean around `2.4`, while recoverable false-alert cases
were much lower, typically around `0.4–1.0`.

### Long-horizon label experiment: horizon 240

Command:

```bash
python scripts/build_spillback_dataset.py \
  --inputs 'results/oracle_recoverability/semantic_risk_batch_*.json' \
  --output results/oracle_recoverability/spillback_20seed_h240_v1.npz \
  --history 60 \
  --horizon 240 \
  --stride 5
```

Result:

- samples: `1700`
- positive rate: `0.0900`
- `62016` first positive step moved earlier to `109`.

Training with auxiliary weight `0.2` gave poor validation discrimination:

- h240 validation ROC-AUC: `0.4970`
- h120 cross-evaluation ROC-AUC: `0.4875`

Interpretation: long-horizon labels provide earlier positives but are too hard
or noisy for the current model and data size. This path is not currently the
best use of time.

### Risk-model controlled fallback probes

Implemented `--control_source risk_model` in
`scripts/probe_dual_policy_risk.py`.

The policy uses observed actions by default and switches to temporal fallback
when risk triggers. Variants:

- binary risk threshold;
- risk + auxiliary qmean threshold;
- sticky fallback hold via `--risk_hold_steps`.

#### Baseline: temporal-only on bad4

Command:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62000,62005,62009,62016 \
  --control_source temporal \
  --gpus 0 \
  --output results/oracle_recoverability/temporal_bad4.json
```

Results:

| Seed | Observed | Temporal-only |
|---:|---:|---:|
| 62000 | 365255 | 209425 |
| 62005 | 350335 | 193670 |
| 62009 | 362575 | 193065 |
| 62016 | 370915 | 191040 |

Interpretation: temporal fallback itself is effective for all four bad seeds.
The remaining problem is deciding when/how to switch.

#### Risk binary gate

Command:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62000,62005,62009,62016 \
  --control_source risk_model \
  --risk_model_path results/oracle_recoverability/spillback_20seed_aux02_model_v1.pth \
  --risk_threshold 0.5 \
  --gpus 0 \
  --output results/oracle_recoverability/risk_bad_binary.json
```

Results:

| Seed | Observed | Risk binary gate | Delta vs observed |
|---:|---:|---:|---:|
| 62000 | 365255 | 368675 | +3420 |
| 62005 | 350335 | 191240 | -159095 |
| 62009 | 362575 | 351965 | -10610 |
| 62016 | 370915 | 696665 | +325750 |

Interpretation: can save one seed strongly, but unsafe because intermittent
switching can catastrophically worsen `62016`.

#### Risk + qmean recoverability gate

Command:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62000,62005,62009,62016 \
  --control_source risk_model \
  --risk_model_path results/oracle_recoverability/spillback_20seed_aux02_model_v1.pth \
  --risk_threshold 0.5 \
  --risk_aux_qmean_threshold 1.0 \
  --gpus 0 \
  --output results/oracle_recoverability/risk_bad_recover_q1.json
```

Results:

| Seed | Observed | Risk + qmean gate | Delta vs observed |
|---:|---:|---:|---:|
| 62000 | 365255 | 366950 | +1695 |
| 62005 | 350335 | 352670 | +2335 |
| 62009 | 362575 | 186155 | -176420 |
| 62016 | 370915 | 696665 | +325750 |

Interpretation: qmean gate can save a different seed (`62009`) but remains
unsafe for `62016`.

#### Sticky risk gate

Command:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62000,62005,62009,62016 \
  --control_source risk_model \
  --risk_model_path results/oracle_recoverability/spillback_20seed_aux02_model_v1.pth \
  --risk_threshold 0.5 \
  --risk_hold_steps -1 \
  --gpus 0 \
  --output results/oracle_recoverability/risk_bad_binary_sticky.json
```

Results:

| Seed | Observed | Sticky risk gate | Delta vs observed |
|---:|---:|---:|---:|
| 62000 | 365255 | 367980 | +2725 |
| 62005 | 350335 | 195105 | -155230 |
| 62009 | 362575 | 366340 | +3765 |
| 62016 | 370915 | 696665 | +325750 |

Interpretation: sticky fallback removes some policy oscillation but still does
not solve timing. It saves `62005` but fails on three bad seeds.

### Current conclusion

The project has moved from “can we detect congestion risk?” to a sharper
research question:

> Under observation failure, the key is not only predicting spillback risk, but
> predicting whether an intervention/fallback switch will improve the future
> trajectory.

This suggests a stronger and more publishable direction:

**Counterfactual intervention-benefit gating**.

Instead of training only a spillback-risk predictor, train a gate to estimate:

- if we keep observed actions, what happens?
- if we switch to temporal fallback now, what happens?
- is the predicted intervention benefit positive enough to switch?

### Next planned experiments

1. Run intervention timing sweeps on bad seeds.
   - Force switch to temporal fallback from fixed steps:
     `60, 90, 120, 150, 180, 210, 240, 270, 300, ...`.
   - Build per-seed “latest safe switch time” curves.

2. Construct counterfactual benefit labels.
   - Label examples by whether switching at the current state reduces future
     total time / queue relative to staying observed.

3. Train a benefit gate instead of a pure risk gate.
   - Candidate output: predicted improvement or binary beneficial-switch label.

4. Evaluate on all 20 seeds.
   - Must improve bad seeds without introducing new normal-seed collapses.

5. Only after benefit gate shows signal, expand to more seeds/maps.

## 2026-07-03: Fixed-step temporal intervention timing sweep

### Goal

Test the hypothesis that the previous risk gates failed because they switched
to temporal fallback at the wrong time or for the wrong duration. Instead of
using a learned gate, force a switch from observed actions to temporal fallback
at fixed episode steps and measure the counterfactual intervention outcome.

This experiment directly estimates the intervention window needed for a future
benefit-gate label.

### Code change

Added `temporal_after_step` to `scripts/probe_dual_policy_risk.py`.

Behavior:

- before `--temporal_start_step`: use observed actions;
- at/after `--temporal_start_step`: use temporal fallback actions;
- record `temporal_active` and `temporal_start_step` in the trace.

Commit:

- `832b331` — add fixed-step temporal intervention probe.

### Experiment command

For each start step in `0, 60, 120, 180, 240, 300, 360, 420, 480`:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62000,62005,62009,62016 \
  --control_source temporal_after_step \
  --temporal_start_step <STEP> \
  --gpus 0 \
  --output results/oracle_recoverability/fixed_temporal_start_<STEP>_bad4.json
```

Summary files:

- `results/oracle_recoverability/fixed_temporal_bad4_summary.csv`
- `results/oracle_recoverability/fixed_temporal_bad4_summary.json`

### Results

Total time spent:

| Seed | observed | temporal-only / step 0 | step 60 | step 120 | step 180 | step 240 | step 300 | step 360 | step 420 | step 480 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 62000 | 365255 | 209425 | 1030965 | 200145 | 197950 | 204385 | 346315 | 367745 | 367290 | 367515 |
| 62005 | 350335 | 193670 | 369710 | 196205 | 200095 | 204030 | 197985 | 359505 | 354005 | 352615 |
| 62009 | 362575 | 193065 | 186275 | 204065 | 200245 | 197330 | 189945 | 361455 | 363325 | 364155 |
| 62016 | 370915 | 191040 | 200220 | 189275 | 191510 | 696665 | 516680 | 452560 | 428350 | 412585 |

Using `total_time_spent <= 250000` as a rough rescued threshold:

| Seed | Best start step | Best total time | Latest rescued start step |
|---:|---:|---:|---:|
| 62000 | 180 | 197950 | 240 |
| 62005 | 0 | 193670 | 300 |
| 62009 | 60 | 186275 | 300 |
| 62016 | 120 | 189275 | 180 |

### Interpretation

The intervention window is real and strongly state-dependent.

Key observations:

1. `temporal-only` rescues all four bad seeds, confirming that temporal fallback
   is a useful intervention.
2. Switching too late fails:
   - `62016` is no longer rescued at step 240 and becomes catastrophic
     (`696665`).
   - `62000/62005/62009` mostly fail by step 360.
3. Switching at a bad intermediate moment can be worse than both baselines:
   - `62000` at step 60 becomes `1030965`, far worse than observed and
     temporal-only.
   - `62005` at step 60 also fails (`369710`).
4. Safe switch timing is not monotonic:
   - `62000` fails at step 60 but succeeds at steps 120/180/240.
   - This suggests phase/state compatibility matters, not merely “earlier is
     better.”

### Research implication

This supports a stronger paper direction:

> The control problem is a counterfactual intervention-timing problem, not a
> pure risk-detection problem.

A future gate should predict intervention benefit:

```text
benefit(s_t, t) =
  future_cost(observed from t) - future_cost(temporal fallback from t)
```

This benefit can be non-monotonic in time, so a useful model must learn both:

- whether the scene is at risk;
- whether switching now is safe/useful.

### Next planned experiments

1. Refine the timing sweep around critical windows:
   - `62000`: steps `30, 45, 60, 75, 90, 105, 120`.
   - `62016`: steps `180, 195, 210, 225, 240`.
2. Run fixed-step sweep on normal seeds to check whether temporal intervention
   harms otherwise healthy episodes.
3. Build a counterfactual benefit dataset from the fixed-step traces.
4. Train a first benefit classifier/regressor and compare it against risk-only
   gates.

## 2026-07-03: Refined intervention-window sweep

### Goal

Refine two critical windows discovered by the coarse fixed-step sweep:

- `62000`: coarse sweep showed a catastrophic result at step 60 but recovery at
  step 120.
- `62016`: coarse sweep showed success at step 180 but catastrophic failure at
  step 240.

The purpose was to test whether these are noisy outcomes or stable narrow
intervention windows.

### Commands

For `62000`:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62000 \
  --control_source temporal_after_step \
  --temporal_start_step <30|45|60|75|90|105|120> \
  --gpus 0 \
  --output results/oracle_recoverability/fixed_temporal_seed62000_start_<STEP>.json
```

For `62016`:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62016 \
  --control_source temporal_after_step \
  --temporal_start_step <180|195|210|225|240> \
  --gpus 0 \
  --output results/oracle_recoverability/fixed_temporal_seed62016_start_<STEP>.json
```

Summary files:

- `results/oracle_recoverability/fixed_temporal_refined_summary.csv`
- `results/oracle_recoverability/fixed_temporal_refined_summary.json`

### Results

`62000`:

| Start step | Total time spent | Outcome |
|---:|---:|---|
| 30 | 209425 | rescued |
| 45 | 209425 | rescued |
| 60 | 1030965 | catastrophic |
| 75 | 1030965 | catastrophic |
| 90 | 198630 | rescued |
| 105 | 197870 | rescued, best refined point |
| 120 | 200145 | rescued |

`62016`:

| Start step | Total time spent | Outcome |
|---:|---:|---|
| 180 | 191510 | rescued |
| 195 | 196395 | rescued |
| 210 | 187445 | rescued, best refined point |
| 225 | 710800 | catastrophic |
| 240 | 696665 | catastrophic |

### Interpretation

The refined sweep confirms that the fixed-step results are not random noise.

Important findings:

1. `62000` has a stable bad intervention window at steps `60–75`.
   - Switching at 30/45 is safe.
   - Switching at 60/75 is catastrophic.
   - Switching again at 90/105/120 is safe.
2. `62016` has a sharp cliff between 210 and 225.
   - Step 210 is the best refined point (`187445`).
   - Step 225 fails catastrophically (`710800`).
3. The intervention value is sharply non-monotonic.
   - This rules out a simple “switch earlier when risk is high” policy.
   - The gate must reason about state/phase compatibility and downstream
     recoverability.

### Research implication

This is strong evidence for the paper's central novelty:

> Failure recovery in MARL traffic control should be formulated as
> counterfactual intervention timing, not merely risk prediction or observation
> imputation.

The next model should learn a benefit surface:

```text
V_switch(s_t, phase_t, failure_t) - V_observed(s_t, phase_t, failure_t)
```

The label should penalize unsafe switch states such as:

- `62000` at steps 60/75;
- `62016` at steps 225/240.

### Next planned experiments

1. Run fixed-step temporal intervention on normal seeds at representative
   steps (`0, 120, 180, 240, 300`) to measure collateral damage.
2. Build a first intervention-benefit dataset:
   - positive: switch step leads to rescued outcome;
   - negative: switch step causes no benefit or catastrophic outcome.
3. Add phase/state features to the benefit model, because bad windows likely
   correspond to phase-incompatible fallback switches.
