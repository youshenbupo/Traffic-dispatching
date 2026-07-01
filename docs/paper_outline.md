# Paper plan: Privileged Decision Distillation for Robust Multi-Agent TSC

## Working title

**Learning to Act Through Sensor Failures: Privileged Decision Distillation
for Robust Multi-Agent Traffic Signal Control**

Short method name: **DC-PDD** (Decision-Critical Privileged Decision
Distillation).

## Central claim

Robust traffic-signal policies should recover the *decision* made under clean
traffic observations instead of reconstructing every missing sensor value.
During training, a frozen clean-observation teacher supplies action-level
privileged supervision. During deployment, every intersection acts from its
corrupted local observation without privileged information and, in the core
method, without communication.

This claim is narrower and better supported than claiming that neighbor
communication is generally beneficial.

## Research questions

1. Does clean-policy decision distillation outperform equal-budget robust
   MAPPO under missing intersection observations?
2. Are improvements caused by teacher initialization, fixed-teacher
   distillation, reconstruction, or communication?
3. Does action-level supervision remain effective across failure severity,
   duration, topology, and demand shifts?
4. When does optional communication add value beyond the distilled local
   fallback?

## Method

Let the clean centralized-training observation of intersection \(i\) be
\(o_i\), its corrupted deployment observation be \(\tilde{o}_i\), and the
legal-action mask be \(m_i\).

A teacher \(\pi_T(a_i\mid o_i,m_i)\) is trained first and then frozen. We
measure whether a failure is decision-critical by applying the same teacher to
the clean and corrupted observations:

\[
c_i = D_{KL}\left[
\pi_T(\cdot\mid o_i,m_i)
\;\|\;
\pi_T(\cdot\mid\tilde{o}_i,m_i)
\right].
\]

The student \(\pi_\theta(a_i\mid\tilde{o}_i,m_i)\) minimizes the usual MAPPO
loss plus a criticality-weighted decision-distillation objective:

\[
\mathcal{L}_{PDD}
=
\frac{1}{\sum_i z_i w_i}
\sum_i z_i w_i
D_{KL}\left[
\pi_T(\cdot\mid o_i,m_i)
\;\|\;
\pi_\theta(\cdot\mid\tilde{o}_i,m_i)
\right],
\]

where \(z_i=1\) when at least one observation feature is unavailable. The
weight \(w_i\) is a clipped normalized function of \(c_i\), with a small floor
for coverage. The teacher is never used at deployment.

The optional communication extension estimates whether a neighbor message
reduces teacher-policy divergence. It must be reported separately from the
communication-free core and must beat the same-weight no-message ablation.

## Candidate contributions

1. A decision-critical privileged distillation objective that allocates
   supervision according to the control impact of sensor failure rather than
   state-reconstruction error.
2. A strict causal evaluation protocol that separates teacher initialization,
   training-time privileged supervision, reconstruction, and online
   communication.
3. A systematic study of transient, burst, and correlated sensor failures
   across synthetic and real traffic networks.
4. An empirical characterization of when communication improves decisions
   beyond a robust local fallback. This remains a secondary contribution until
   the effect is reproducible.

## Mandatory ablations

| ID | Teacher init | Frozen decision loss | Reconstruction | Train comm | Test comm |
|---|---:|---:|---:|---:|---:|
| A | no | no | no | no | no |
| B | yes | no | no | no | no |
| C (PDD) | yes | yes | no | no | no |
| D | no | yes | no | no | no |
| E | yes | yes | yes | no | no |
| F | yes | yes | yes | yes | no |
| G | yes | yes | yes | yes | yes |

A is equal-budget robust MAPPO. C is the proposed core method. F versus G is
the same-weight communication intervention.

## Evaluation protocol

- At least five training seeds and common paired evaluation seeds.
- Equal environment steps and equal teacher-pretraining accounting.
- Clean observations and dropout severity curves from 0% to 50%.
- IID feature loss, whole-intersection loss, burst failures, correlated
  regional failures, noise, and delayed observations.
- Grid3x3/grid4x4 plus Hangzhou and Jinan real-network scenarios.
- Baselines: fixed time, max pressure, MAPPO, robust MAPPO, CoLight/FGLight or
  another graph MARL baseline, and a recent missing-observation method.
- Report travel time, waiting time, queue, throughput, failure recovery time,
  inference cost, communication rate, mean/std, paired confidence intervals,
  and effect sizes.

## Current evidence and claim boundary

On grid4x4 with 30% whole-intersection dropout, three training seeds, and five
paired evaluation seeds:

| Method | Travel | Waiting | Queue | Throughput |
|---|---:|---:|---:|---:|
| Equal-budget continued robust MAPPO | 93.80 | 10.08 | 0.0226 | 237.07 |
| Teacher initialization only | 90.16 | 8.27 | 0.0185 | 237.40 |
| Uniform PDD | 90.59 | 8.16 | 0.0182 | 237.33 |
| Decision-critical PDD | 91.97 | **7.42** | **0.0156** | 236.80 |

Decision-critical weighting consistently improves all three congestion metrics
over the equal-budget baseline and substantially reduces seed variance. It
does not dominate initialization-only on travel time or throughput. This
supports a congestion-focused contribution but still requires five-seed,
failure-curve, and real-network validation. It does not support a strong
communication claim.

## Go/no-go criteria

Proceed with PDD-MAPPO as the main paper method only if:

1. C beats both A and B on travel, waiting, and queue in at least four of five
   seeds;
2. the advantage persists on at least one real network and across at least
   three failure severities;
3. gains are not explained by extra optimization steps or teacher
   initialization;
4. clean-condition performance does not materially regress.

Promote communication to a main contribution only if G beats F consistently
and gives a useful performance-bandwidth frontier. Otherwise report it as a
negative finding or omit it from the principal method.
