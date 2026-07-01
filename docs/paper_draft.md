# Decision-Critical Privileged Distillation for Failure-Robust Multi-Agent
# Traffic Signal Control

## Abstract

Multi-agent traffic signal control methods usually assume that every
intersection continuously observes reliable traffic measurements. Sensor
failures violate this assumption and can cause a decentralized policy to take
different signal actions from those it would choose under intact observations.
Existing robust approaches commonly randomize observations, reconstruct
missing state, or exchange neighbor messages. Our experiments show that lower
state-reconstruction error and denser communication do not necessarily improve
traffic control. We therefore propose Decision-Critical Privileged Decision
Distillation (DC-PDD), a centralized-training, decentralized-execution method
that uses a frozen clean-observation teacher only during training. DC-PDD
measures the control impact of a sensor failure by the divergence between the
teacher's clean- and corrupted-observation action distributions, and focuses
student distillation on failures that change the teacher's decision. At
deployment, each intersection acts using only its corrupted local observation
and requires neither privileged state nor inter-agent communication.
Preliminary grid4x4 experiments with whole-intersection observation failures
show that DC-PDD consistently reduces waiting and queueing relative to an
equal-budget robust MAPPO baseline, while exhibiting substantially lower
between-seed variance. These results motivate a broader evaluation across
failure processes, real traffic networks, and strong robust-control baselines.

## 1. Introduction

Deep multi-agent reinforcement learning has become a standard approach to
networked traffic signal control, where one policy controls each intersection
and centralized training supports decentralized execution. Most evaluations,
however, assume reliable lane queues, waiting times, and flows. Real detectors
can be unavailable, delayed, noisy, or jointly affected by regional failures.
The resulting observation corruption is not ordinary traffic stochasticity:
it directly changes the information available to the deployed controller.

A natural response is to reconstruct missing traffic state or to request
neighbor observations. Our controlled experiments reveal two limitations.
First, improved reconstruction accuracy may still move the policy toward a
worse signal action. Second, communication modules can improve performance
relative to a baseline while performing no better than the same trained
weights with all messages removed. This means that architecture and robust
training effects can easily be misattributed to communication.

We take a decision-centered view. A sensor failure matters when it changes the
action distribution of a competent clean-observation policy. During training,
DC-PDD evaluates a frozen teacher on both clean and corrupted observations.
Their policy divergence estimates decision criticality. A student acting on
corrupted local observations then receives stronger privileged supervision on
failures with high decision impact. This avoids treating every missing value
as equally important and avoids requiring state reconstruction at deployment.

Our current contributions are:

1. We formulate decision criticality as the clean-to-corrupted policy
   divergence of a frozen traffic-control teacher and use it to weight
   privileged policy distillation.
2. We provide a communication-free CTDE algorithm for intersection-level
   sensor failures: privileged observations are used only during training,
   while deployment remains local and decentralized.
3. We introduce strict causal ablations separating training budget, teacher
   initialization, uniform distillation, decision-critical weighting,
   reconstruction, and online communication.
4. We report a systematic negative result: state reconstruction and learned
   communication can improve auxiliary objectives without improving control
   over the same-weight no-message policy.

## 2. Related work and positioning

Robust traffic signal control under demand changes, incidents, and sensor
failures predates the current graph-MARL literature. Recent methods such as
RobustLight explicitly restore corrupted traffic observations, while graph
controllers such as FGLight and MacLight target coordination and scalability.
DC-PDD differs by transferring clean *decisions* rather than treating accurate
state recovery as the objective.

Privileged-information and teacher-student learning are established ideas in
partially observable reinforcement learning. Recent theory also shows that
naive expert distillation can fail under general observation aliasing.
Therefore, the paper must not claim privileged distillation itself as novel.
Its intended novelty is the decision-critical failure weighting, the
communication-free robust TSC formulation, and the controlled empirical
separation of decision recovery from state recovery.

Relevant sources:

- [Provable Partially Observable RL with Privileged Information
  (NeurIPS 2024)](https://openreview.net/forum?id=1rnKF0JtXK)
- [AERIAL: MARL under Stochastic Partial Observability
  (ICML 2023)](https://openreview.net/forum?id=23uOLxPd34)
- [RobustLight (ICML 2025)](https://icml.cc/virtual/2025/poster/44919)
- [FGLight (AAMAS 2025)](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2181.pdf)
- [MacLight (AAMAS 2025)](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p1263.pdf)

## 3. Method

### 3.1 Problem setting

Intersection \(i\) has clean observation \(o_i\), corrupted observation
\(\tilde{o}_i\), legal-action mask \(m_i\), and action \(a_i\). Clean
observations are simulator-only privileged information during training.
Deployment exposes only \(\tilde{o}_i\) and \(m_i\).

### 3.2 Frozen teacher

A teacher policy \(\pi_T\) is trained first and then frozen. The student is
optimized under the same corrupted-observation process used for robust MAPPO.
Teacher training cost and student training cost are reported separately.

### 3.3 Decision criticality

For a corrupted intersection, define

\[
c_i =
D_{KL}\!\left(
\pi_T(\cdot\mid o_i,m_i)
\;\|\;
\pi_T(\cdot\mid\tilde{o}_i,m_i)
\right).
\]

Unlike reconstruction error, \(c_i\) is zero when corruption does not change
the teacher's action distribution, even if many raw features are absent.

### 3.4 Weighted privileged decision loss

Let \(z_i\) indicate that at least one observation feature is missing. The
normalized clipped weight is

\[
w_i = \operatorname{clip}\left(
\epsilon + \frac{c_i}{\mathbb{E}[c\mid z=1]+\delta},
\;0,\;w_{\max}
\right).
\]

The student loss is

\[
\mathcal{L}
= \mathcal{L}_{MAPPO}
+ \lambda
\frac{\sum_i z_i w_i
D_{KL}\!\left(
\pi_T(\cdot\mid o_i,m_i)
\;\|\;
\pi_\theta(\cdot\mid\tilde{o}_i,m_i)
\right)}
{\sum_i z_i w_i}.
\]

The teacher and criticality computation are removed at deployment.

## 4. Experimental protocol

The main comparison uses equal student-environment steps and common evaluation
seeds. Every result reports at least five training seeds in the final paper.
Required conditions include clean observations, IID dropout, whole-agent
dropout, burst failures, correlated regional failures, noise, observation
delay, and demand shift. Required scenarios include grid3x3/grid4x4 and at
least Hangzhou and Jinan.

The mandatory method ablations are robust MAPPO, equal-budget continued
MAPPO, teacher initialization only, uniform PDD, DC-PDD, DC-PDD without
initialization, state reconstruction, training-time communication, and
same-weight inference-time communication removal.

## 5. Preliminary results

Grid4x4, 30% whole-intersection dropout, three training seeds, five common
evaluation seeds:

| Method | Travel | Waiting | Queue | Throughput |
|---|---:|---:|---:|---:|
| Equal-budget robust MAPPO | 93.80 ± 0.72 | 10.08 ± 0.68 | 0.0226 ± 0.0017 | 237.07 ± 0.50 |
| Teacher initialization only | 90.16 ± 2.14 | 8.27 ± 1.07 | 0.0185 ± 0.0023 | 237.40 ± 1.06 |
| Uniform PDD | 90.59 ± 1.18 | 8.16 ± 1.10 | 0.0182 ± 0.0029 | 237.33 ± 0.31 |
| DC-PDD | 91.97 ± 0.23 | **7.42 ± 0.12** | **0.0156 ± 0.0002** | 236.80 ± 0.35 |

DC-PDD reduces waiting by approximately 26.4% and queueing by approximately
31% relative to the equal-budget baseline, with improvements in all three
training seeds. Its travel time is better than the baseline but worse than
initialization-only and uniform PDD, exposing a clear mobility-congestion
tradeoff. This is preliminary evidence, not yet a final claim.

## 6. Limitations and next experiments

The current evidence uses only three training seeds, short training, and a
synthetic network. Teacher quality, clean-condition regression, and sensitivity
to the criticality clipping parameters remain untested. The next admission
test is five-seed validation on grid4x4 across a 0--50% failure curve, followed
by Hangzhou and Jinan. If the waiting/queue gains do not persist, DC-PDD should
not be presented as the final method.
