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

## 2026-07-03: Normal-seed collateral-damage sweep

### Goal

Check whether fixed temporal intervention harms episodes that are already
healthy under the observed policy. This is necessary because a recovery method
must not rescue bad seeds by creating new failures on normal seeds.

### Seeds and baseline

Normal seeds:

| Seed | Observed total time |
|---:|---:|
| 62001 | 189435 |
| 62002 | 182580 |
| 62003 | 188810 |
| 62004 | 185095 |

### Command

For each start step in `0, 120, 180, 240, 300`:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62001,62002,62003,62004 \
  --control_source temporal_after_step \
  --temporal_start_step <STEP> \
  --gpus 0 \
  --output results/oracle_recoverability/fixed_temporal_start_<STEP>_normal4.json
```

Summary files:

- `results/oracle_recoverability/fixed_temporal_normal4_summary.csv`
- `results/oracle_recoverability/fixed_temporal_normal4_summary.json`

### Results

Total time spent:

| Seed | observed | step 0 | step 120 | step 180 | step 240 | step 300 |
|---:|---:|---:|---:|---:|---:|---:|
| 62001 | 189435 | 188635 | 376480 | 199965 | 195150 | 194980 |
| 62002 | 182580 | 202760 | 194130 | 201105 | 196935 | 190135 |
| 62003 | 188810 | 189000 | 190990 | 193905 | 200005 | 194420 |
| 62004 | 185095 | 191770 | 197915 | 202095 | 194765 | 192085 |

Step-level summary:

| Start step | Mean total | Mean delta vs observed | Max delta vs observed | Catastrophic count >300k |
|---:|---:|---:|---:|---:|
| 0 | 193041 | +6561 | +20180 | 0 |
| 120 | 239879 | +53399 | +187045 | 1 |
| 180 | 199268 | +12788 | +18525 | 0 |
| 240 | 196714 | +10234 | +14355 | 0 |
| 300 | 192905 | +6425 | +7555 | 0 |

### Interpretation

Fixed temporal intervention is not free on normal seeds.

Key findings:

1. Temporal-only is usually safe but slightly worse on average for normal seeds.
2. Step 120 causes a catastrophic normal-seed failure for `62001`
   (`376480`, +187045 vs observed).
3. Other fixed steps mostly cause moderate degradation of roughly 3–7%.
4. Therefore, a deployable recovery method must be selective:
   - switch when the original observed trajectory is likely to fail;
   - avoid phase/state combinations where switching itself is harmful.

### Research implication

The benefit-gate dataset must include both kinds of negatives:

1. bad-seed unsafe switch times, e.g. `62000@60/75`, `62016@225/240`;
2. normal-seed harmful switch times, e.g. `62001@120`.

This is stronger than a spillback-risk label because it learns the actual
decision boundary for intervention usefulness.

### Next planned experiments

1. Construct a small first-generation intervention-benefit dataset from:
   - bad4 fixed-step sweep;
   - refined critical-window sweep;
   - normal4 collateral-damage sweep.
2. Use semantic traces around each switch time as input and label by observed
   intervention outcome:
   - positive if `total_time_spent <= 250000` and improves over observed;
   - negative if it is worse than observed or catastrophic.
3. Train a lightweight benefit classifier and compare its selected switch
   decisions with risk-only gates.

## 2026-07-03: First intervention-benefit dataset and classifier

### Goal

Convert fixed-step intervention outcomes into a supervised dataset for learning
whether switching to temporal fallback at a candidate state is beneficial.

This is the first concrete step from risk prediction toward counterfactual
intervention-benefit prediction.

### Dataset builder

Added:

- `scripts/build_intervention_benefit_dataset.py`

Inputs:

- observed semantic traces:
  `results/oracle_recoverability/semantic_risk_batch_*.json`;
- fixed-step intervention outcomes:
  - `fixed_temporal_start_*_bad4.json`;
  - `fixed_temporal_seed62000_start_*.json`;
  - `fixed_temporal_seed62016_start_*.json`;
  - `fixed_temporal_start_*_normal4.json`.

Command:

```bash
python scripts/build_intervention_benefit_dataset.py \
  --semantic_inputs 'results/oracle_recoverability/semantic_risk_batch_*.json' \
  --intervention_inputs \
    'results/oracle_recoverability/fixed_temporal_start_*_bad4.json' \
    'results/oracle_recoverability/fixed_temporal_seed62000_start_*.json' \
    'results/oracle_recoverability/fixed_temporal_seed62016_start_*.json' \
    'results/oracle_recoverability/fixed_temporal_start_*_normal4.json' \
  --output results/oracle_recoverability/intervention_benefit_v1.npz \
  --history 60 \
  --observed_totals \
    62000:365255,62005:350335,62009:362575,62016:370915,\
62001:189435,62002:182580,62003:188810,62004:185095 \
  --rescue_threshold 250000 \
  --min_improvement 0
```

Dataset:

- samples: `54`
- shape: `(54, 60, 3, 8, 12)`
- positive rate: `0.3519`

Per-seed distribution:

| Seed | Samples | Positives | Notes |
|---:|---:|---:|---|
| 62000 | 11 | 5 | includes unsafe 60/75 windows |
| 62001 | 4 | 0 | normal seed; includes harmful 120 switch |
| 62002 | 4 | 0 | normal seed |
| 62003 | 4 | 0 | normal seed |
| 62004 | 4 | 0 | normal seed |
| 62005 | 8 | 4 | bad seed with broad rescue window |
| 62009 | 8 | 5 | bad seed with broad rescue window |
| 62016 | 11 | 5 | includes sharp 210/225 cliff |

Important labeled negatives:

- `62000@60`, `62000@75`: catastrophic harmful switches.
- `62016@225`, `62016@240`: catastrophic harmful switches.
- `62001@120`: normal-seed collateral-damage switch.

### First benefit classifier

Added:

- `scripts/train_intervention_benefit.py`

Command:

```bash
python scripts/train_intervention_benefit.py \
  --dataset results/oracle_recoverability/intervention_benefit_v1.npz \
  --output results/oracle_recoverability/intervention_benefit_model_v1.pth \
  --validation_seeds 62016,62001 \
  --epochs 100 \
  --hidden_dim 32 \
  --batch_size 16 \
  --gpus 0
```

Validation split:

- `62016`: key bad seed with sharp safe/unsafe cliff.
- `62001`: normal seed with collateral-damage negative.

Result:

| Epoch | Val AUC | Val AP | Val F1@0.5 | Mean prob |
|---:|---:|---:|---:|---:|
| 1 | 0.5200 | 0.4467 | 0.0000 | 0.4342 |
| 50 | 0.2400 | 0.2674 | 0.2000 | 0.4974 |
| 100 | 0.3400 | 0.2942 | 0.3077 | 0.5171 |

### Interpretation

The first benefit classifier is not usable yet.

Likely causes:

1. Dataset is very small (`54` samples).
2. Holding out `62016` is a hard generalization test because its cliff
   (`210 -> 225`) is not well represented by other seeds.
3. The model sees semantic history but not an explicit candidate switch-time
   feature or engineered phase-compatibility feature.
4. Labels are episode-level outcomes assigned to a single switch point, so the
   sample is high variance.

This is still useful because it tells us the next algorithmic requirement:

> Benefit prediction needs more counterfactual coverage and explicit
> switch-state/phase compatibility modeling.

### Next planned experiments

1. Expand fixed-step sweeps to more normal and bad seeds to increase benefit
   sample count.
2. Add explicit candidate-step/time and phase-compatibility features to the
   benefit model.
3. Consider a pairwise/ranking objective:
   - for the same seed, rank safe switch steps above unsafe switch steps;
   - this may be more sample-efficient than binary classification.
4. Only then re-test learned benefit-gated control.

## 2026-07-03: Graph Benefit Gate v1

### Goal

Test whether explicitly modeling the intersection graph improves
counterfactual intervention-benefit prediction.

Motivation:

- traffic intersections are graph-structured agents;
- a local fallback switch can affect upstream/downstream spillback;
- previous temporal-only benefit classifier did not generalize on the hard
  validation split.

### Model

Added:

- `GraphInterventionBenefitPredictor` in `src/risk/model.py`.

Architecture:

1. lane-level semantic encoder;
2. per-agent temporal GRU over the 60-step history;
3. candidate switch-step embedding;
4. two graph message-passing layers over the intersection adjacency matrix;
5. network-level benefit classifier.

Training script:

- `scripts/train_intervention_benefit.py --model_type graph`

Commit:

- `1ba5d9e` — add graph intervention benefit predictor.

### Command

```bash
python scripts/train_intervention_benefit.py \
  --dataset results/oracle_recoverability/intervention_benefit_v1.npz \
  --output results/oracle_recoverability/intervention_benefit_graph_v1.pth \
  --validation_seeds 62016,62001 \
  --model_type graph \
  --epochs 100 \
  --hidden_dim 32 \
  --batch_size 16 \
  --gpus 0
```

Validation split:

- `62016`: hard bad seed with sharp safe/unsafe cliff.
- `62001`: normal seed with a harmful intervention at step 120.

### Results

Compared with the previous temporal benefit classifier on the same split:

| Model | Final Val AUC | Final Val AP | Final F1@0.5 |
|---|---:|---:|---:|
| Temporal benefit classifier | 0.3400 | 0.2942 | 0.3077 |
| Graph Benefit Gate v1 | 0.8600 | 0.6595 | 0.7273 |

Training trajectory for Graph Benefit Gate v1:

| Epoch | Val AUC | Val AP | Val F1@0.5 |
|---:|---:|---:|---:|
| 10 | 0.6600 | 0.6330 | 0.5000 |
| 20 | 0.7800 | 0.5976 | 0.5000 |
| 60 | 0.8400 | 0.7226 | 0.6667 |
| 80 | 0.8600 | 0.7417 | 0.7692 |
| 100 | 0.8600 | 0.6595 | 0.7273 |

### Sample-level diagnosis

Important validation samples:

| Sample | Label | Benefit | Predicted probability | Diagnosis |
|---|---:|---:|---:|---|
| `62016@210` | 1 | +183470 | 0.488 | borderline false negative at threshold 0.5 |
| `62016@225` | 0 | -339885 | 0.478 | correctly below 0.5 but close |
| `62016@240` | 0 | -325750 | 0.395 | correctly lower |
| `62001@120` | 0 | -187045 | 0.673 | false positive |

Important training/cohort samples:

| Sample | Label | Benefit | Predicted probability | Diagnosis |
|---|---:|---:|---:|---|
| `62000@60` | 0 | -665710 | 0.609 | false positive |
| `62000@75` | 0 | -665710 | 0.604 | false positive |
| `62000@90` | 1 | +166625 | 0.738 | correct positive |
| `62000@105` | 1 | +167385 | 0.686 | correct positive |

### Interpretation

This is the first strong evidence that graph-structured benefit modeling is
better than the earlier temporal-only classifier. The improvement is large on
the hard split:

```text
AUC: 0.34 -> 0.86
F1@0.5: 0.31 -> 0.73
```

However, the sample-level diagnosis also shows the current graph model is not
deployable yet:

1. It still misclassifies `62001@120`, a normal-seed harmful intervention.
2. It still gives high probabilities to `62000@60/75`, the catastrophic bad
   intervention window.
3. It mostly learns a coarse timing trend but not enough phase/action
   compatibility.

### Research implication

The GNN direction is justified, but graph topology alone is not sufficient.

The next model should explicitly include:

- candidate switch step;
- current traffic phase;
- observed action;
- temporal fallback action;
- whether observed and temporal actions disagree;
- phase-lane service compatibility before and after the candidate switch.

This supports the evolving method:

> Graph-based Counterfactual Intervention-Benefit Gate with
> Phase/Action-Compatibility Features.

### Next planned experiments

1. Expand the benefit dataset with more fixed-step sweeps so the graph model is
   not trained on only 54 samples.
2. Add action disagreement and phase-compatibility features to the benefit
   dataset.
3. Re-train Graph Benefit Gate v2 and specifically test whether it fixes:
   - `62001@120`;
   - `62000@60/75`;
   - `62016@210/225` boundary.

## 2026-07-03: Graph Benefit Gate v2/v3 with switch-context features

### Goal

Continue the GNN direction by adding explicit switch-state context features.
Graph Benefit Gate v1 showed that topology helps, but it still misclassified
important phase/action-incompatible interventions:

- `62001@120`;
- `62000@60/75`;
- the `62016@210/225` boundary.

The hypothesis was that the model needs more than graph topology; it also needs
features describing whether the candidate switch point is compatible with the
current traffic/failure/action state.

### Code changes

Added context features to `scripts/build_intervention_benefit_dataset.py`:

- current failure fraction;
- historical failure fraction;
- current observed-vs-temporal disagreement fraction;
- historical disagreement fraction;
- current queue mean/max;
- queue on lanes currently served by the active phase;
- queue on lanes not currently served;
- active service fraction;
- unserved queue pressure;
- queue mean growth over the history window.

Updated `GraphInterventionBenefitPredictor` to accept these context features in
addition to:

- lane semantic history;
- failure history;
- candidate switch step;
- graph adjacency.

Commit:

- `45d446d` — add context features for graph benefit gate.

### Graph Benefit Gate v2: same 54-sample dataset with context

Dataset:

- `results/oracle_recoverability/intervention_benefit_v2_context.npz`
- samples: `54`
- positive rate: `0.3519`

Command:

```bash
python scripts/train_intervention_benefit.py \
  --dataset results/oracle_recoverability/intervention_benefit_v2_context.npz \
  --output results/oracle_recoverability/intervention_benefit_graph_v2_context.pth \
  --validation_seeds 62016,62001 \
  --model_type graph \
  --epochs 100 \
  --hidden_dim 32 \
  --batch_size 16 \
  --gpus 0
```

Result:

- epoch 1: AUC `0.8600`, AP `0.6595`, but all probabilities below 0.5.
- epoch 100: AUC `0.5600`, AP `0.4119`, F1@0.5 `0.6250`.

Interpretation:

Context features can help identify useful structure, but with only 54 samples
they are easy to overfit. The final model fixed `62000@60/75`, but generalized
poorly to held-out `62016` and `62001`.

### Normal12 collateral-damage expansion

To increase negative coverage, ran fixed-step temporal interventions on the
remaining 12 normal seeds:

```bash
python scripts/probe_dual_policy_risk.py \
  --config configs/mappo_cologne3_eval.yaml \
  --model_path logs/selected_teachers_mature/cologne3 \
  --seeds 62006,62007,62008,62010,62011,62012,62013,62014,62015,62017,62018,62019 \
  --control_source temporal_after_step \
  --temporal_start_step <120|180|240|300> \
  --gpus 3 \
  --output results/oracle_recoverability/fixed_temporal_start_<STEP>_normal12.json
```

Summary files:

- `results/oracle_recoverability/fixed_temporal_normal12_summary.csv`
- `results/oracle_recoverability/fixed_temporal_normal12_summary.json`

Results:

| Seed | step 120 | step 180 | step 240 | step 300 |
|---:|---:|---:|---:|---:|
| 62006 | 192210 | 191830 | 196765 | 193790 |
| 62007 | 477170 | 439955 | 206305 | 196230 |
| 62008 | 205865 | 199230 | 381595 | 350140 |
| 62010 | 201675 | 199135 | 198295 | 194345 |
| 62011 | 189280 | 199000 | 196540 | 369335 |
| 62012 | 203440 | 200415 | 199715 | 194620 |
| 62013 | 191845 | 954360 | 204340 | 195095 |
| 62014 | 195330 | 200165 | 198490 | 193685 |
| 62015 | 203380 | 366030 | 193555 | 195735 |
| 62017 | 190670 | 359040 | 195445 | 194640 |
| 62018 | 201335 | 193510 | 198770 | 194060 |
| 62019 | 196525 | 481740 | 196190 | 195090 |

Step-level summary:

| Step | Mean total | Mean delta vs observed | Max delta | Catastrophic count >300k |
|---:|---:|---:|---:|---:|
| 120 | 220727 | +33453 | +290240 | 1 |
| 180 | 332034 | +144760 | +765460 | 5 |
| 240 | 213834 | +26559 | +190020 | 1 |
| 300 | 222230 | +34956 | +181600 | 2 |

Interpretation:

Fixed temporal switching can seriously damage normal seeds. Step 180 is
especially dangerous in this cohort, with 5/12 catastrophic normal-seed
failures. This provides the missing negative coverage for learning a safer gate.

### Graph Benefit Gate v3: context + normal12 expansion

Dataset:

- `results/oracle_recoverability/intervention_benefit_v3_context_normal12.npz`
- samples: `102`
- positive rate: `0.1863`

Command:

```bash
python scripts/train_intervention_benefit.py \
  --dataset results/oracle_recoverability/intervention_benefit_v3_context_normal12.npz \
  --output results/oracle_recoverability/intervention_benefit_graph_v3_context_e100.pth \
  --validation_seeds 62016,62001 \
  --model_type graph \
  --epochs 100 \
  --hidden_dim 32 \
  --batch_size 16 \
  --gpus 3
```

Validation results:

| Epoch | Val AUC | Val AP | Val F1@0.5 |
|---:|---:|---:|---:|
| 20 | 0.9400 | 0.9250 | 0.0000 |
| 70 | 0.8200 | 0.7742 | 0.7273 |
| 90 | 0.9200 | 0.9111 | 0.7500 |
| 100 | 0.9400 | 0.9250 | 0.8889 |

This is the best benefit model so far.

### Sample-level diagnosis for v3

Key held-out samples:

| Sample | Label | Benefit | Probability | Diagnosis |
|---|---:|---:|---:|---|
| `62016@210` | 1 | +183470 | 0.521 | correct, but close |
| `62016@225` | 0 | -339885 | 0.322 | fixed |
| `62016@240` | 0 | -325750 | 0.287 | fixed |
| `62001@120` | 0 | -187045 | 0.035 | fixed |

Key previous false positives:

| Sample | Label | Benefit | Probability | Diagnosis |
|---|---:|---:|---:|---|
| `62000@60` | 0 | -665710 | 0.024 | fixed |
| `62000@75` | 0 | -665710 | 0.004 | fixed |
| `62000@90` | 1 | +166625 | 0.945 | correct |
| `62000@105` | 1 | +167385 | 0.897 | correct |

Remaining issues:

- `62016@195` is a false negative at threshold 0.5 (`0.190`) despite being
  beneficial.
- Some minor-harm normal interventions still receive high probabilities, e.g.
  `62019@120` has probability `0.847` but only modest harm (`-11660`).

### Interpretation

Adding normal12 negative coverage and switch-context features materially
improved the graph benefit model:

```text
Temporal benefit classifier: AUC 0.34
Graph Benefit Gate v1:      AUC 0.86
Graph Benefit Gate v3:      AUC 0.94, AP 0.925, F1@0.5 0.889
```

The model now correctly handles the main catastrophic cases that motivated the
method. Remaining mistakes are mostly threshold/calibration or minor-harm cases,
suggesting the next step should distinguish catastrophic harm from small
performance loss.

### Next planned experiments

1. Add benefit magnitude/regression or weighted classification:
   - severe negatives such as `62013@180` and `62000@60` should matter more
     than minor negatives such as `62019@120`.
2. Evaluate learned Graph Benefit Gate v3 as an actual control gate, not only
   as an offline classifier.
3. Calibrate a threshold that prioritizes avoiding catastrophic false positives
   while still rescuing bad seeds.
