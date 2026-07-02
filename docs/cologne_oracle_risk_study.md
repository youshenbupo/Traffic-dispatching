# Cologne3 Recoverability and Proactive-Risk Study

## Purpose

This study asks whether Cologne3 failures can be solved by recovering missing
observations, and whether a causal selector can safely choose between the
corrupted-observation teacher and a last-valid temporal fallback. All main
comparisons use the same mature teacher and SUMO seeds 62000--62019.

## Privileged clean-observation counterfactual

The clean condition gives the deployed teacher the simulator's
pre-perturbation observation while leaving the traffic demand and failure
schedule unchanged.

| Metric | Observed teacher | Clean input | Relative change |
|---|---:|---:|---:|
| Travel time | 65.890 | 69.208 | +5.0% |
| Waiting time | 9.820 | 10.858 | +10.6% |
| Queue length | 0.368 | 0.675 | +83.4% |
| Throughput | 2746.2 | 2679.0 | -2.5% |
| Total time spent | 222114.8 | 278343.8 | +25.3% |
| Time per departed vehicle | 79.62 | 108.73 | +36.6% |
| Completion rate | 0.9730 | 0.9588 | -1.5% |

Travel and waiting degradation is statistically detectable in the paired
sample, but the congestion metrics have high variance. The important
mechanism is a mode switch: observed input gridlocks on seeds 62000, 62005,
62009, and 62016, whereas clean input gridlocks on 62010, 62013, and 62017.
Therefore, clean state is not an upper bound for a fixed nonlinear policy.

A hindsight episode selector has substantial headroom: choosing the lower-TTS
outcome between observed and clean reduces queue by 28.2%, total time by
14.4%, and time per departed vehicle by 16.4% relative to observed control.
This is diagnostic only and is not a deployable result.

## Causal last-valid temporal fallback

The temporal candidate replaces a missing feature with its most recent valid
measurement and never uses future or clean measurements. It also remains
unsafe when used throughout an episode:

| Metric | Observed teacher | Temporal fallback | Relative change |
|---|---:|---:|---:|
| Travel time | 65.890 | 70.491 | +7.0% |
| Waiting time | 9.820 | 10.857 | +10.6% |
| Queue length | 0.368 | 0.567 | +54.2% |
| Throughput | 2746.2 | 2716.8 | -1.1% |
| Total time spent | 222114.8 | 262144.2 | +18.0% |
| Completion rate | 0.9730 | 0.9666 | -0.7% |

Temporal fallback repairs all four observed-input gridlocks but creates new
gridlocks on seeds 62007 and 62014. A hindsight observed/temporal selector
would reduce total time by 14.9%, again showing selection headroom but not a
safe deployable method.

## Reactive queue shield

A privileged causal shield used observed control until the preceding 60-step
mean queue exceeded 0.5, then latched to temporal control. On the four known
observed-input failures, it triggered at steps 273--554. It repaired none of
them and increased seed 62016 total time from 370915 to 519845. Detecting
congestion after sustained queue growth is too late.

## Can failure history predict gridlock?

The dual-policy probe records failed intersections, observed/temporal action
disagreement, queue, and active vehicles. A natural cohort of 32 new seeds
(62100--62131) contained four gridlocks. Models using only causal failure-mask
and action-disagreement summaries were tested at multiple prefixes.

- Four-fold cross-validation can find cohort-specific correlations.
- On the independent targeted 62000 cohort, multivariate prediction through
  step 480 has ROC AUC between 0.25 and 0.56.
- AUC rises to 0.875 at step 600, but this occurs after intervention is useful.
- A simple consecutive-failure threshold detects only two of four independent
  gridlocks and produces a false alarm.

Failure duration and candidate disagreement alone do not generalize as an
early-warning signal.

## Algorithm decision

The next method should predict *future spillback risk*, not reconstruct every
missing value and not react to current aggregate congestion:

1. Canonicalize heterogeneous lanes and phases into movement-semantic tokens.
2. Encode recent local observations, actions, masks, and failure ages.
3. Fuse only healthy-neighbor movement tokens during detected failures.
4. Train a privileged risk head to predict future queue growth, downstream
   saturation, and incomplete-trip risk over a fixed horizon.
5. Use risk and uncertainty to select between the nominal teacher and a
   temporal/neighbor belief candidate.
6. Hard-bypass all belief and gating modules under clean observations so the
   nominal action remains exact.

Admission criteria are zero clean mismatch, intervention before measured
queue divergence, improved tail risk and mean total time over teacher-only,
and replication across training and evaluation seeds. Generic imputation,
always-on temporal fallback, and reactive queue thresholds are rejected by
the current evidence.
