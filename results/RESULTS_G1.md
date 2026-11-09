# G1 — truth vs estimate (grid world)
8 Nov 2026 · code: gw_core.py, estimators.py, G1_truth_vs_estimate.py · seed 20261108

## Population specification (fixed before runs)
Two rooms joined by a door (internal wall at x = N//2, door at y = N//2); start distribution uniform
on the left column; goal bottom-right; episode ends at the goal and the next recorded state is drawn
from the start distribution (goal is never a recorded S). Slip: 0.05 everywhere, 0.30 in a zone of
the right room; a slip moves perpendicular (each side half). Policy: shortest-path greedy action with
probability 1−ε, uniform otherwise. p(s,a,s') = μ(s)π(a|s)T(s'|s,a), μ the stationary distribution of
the recording chain. Exact quantities from p; estimates from contiguous windows of one rollout
(temporal dependence of real operation). Estimators: plug-in, Miller–Madow, Chao–Shen; all
quantities from one coherent set of entropies.

Primary world 7×7 (43 states), ε = 0.2 and ε = 0 (deterministic). Stress world 27×27 (703 states),
ε = 0.4, chosen to reach HalfCheetah's sparsity.

## Regime match
HalfCheetah at W = 500: triplet K/W 0.69, Good–Turing coverage 0.41.
Stress world at W = 200: K/W 0.74, coverage 0.44 · W = 350: 0.65, 0.55 · W = 500: 0.59, 0.60.

## Results
**P is well conditioned in the sparse regime.** Stress world, coverage 0.44: relative bias of P
+1.8% (plug-in), +2.2% (MM), +1.4% (CS). Across coverage 0.26–0.90 the absolute bias of P stays
between 0.004 and 0.009 (truth 0.4275). Primary world: +0.016 at W = 50 falling to +0.001 at W = 20000.

**The components are not.** Stress world, coverage 0.44 (truth → plug-in bias):
H_b 1.71 → −1.07 · H_f 0.34 → −0.27 · I(A;S'|S) 1.10 → −0.55.
Chao–Shen reduces this (H_b −0.41, H_f −0.16, I(A;S'|S) −0.05) without removing it.

**Deterministic policy, exact symbols:** I(A;S'|S) = 0 and H(A|S) = 0 in every window and every
estimator. Non-zero values of these terms require a representation in which A is not a function of
the recorded S.

**Bounds:** no plug-in or MM window out of bound; one Chao–Shen window (ε = 0, primary world) out of
bound — reported, not clipped.

## Reading
The budget normalisation makes P a stable share in the sparsely sampled regime: numerator and denominator biases largely cancel. The individual conditional entropies carry
large negative bias there; they are read as deviations from each agent's own calibration, so G2
must show whether that bias is stable when occupancy changes.

## Estimator decision (H2)
Chao–Shen validated as the less biased estimator for the components at HalfCheetah-like coverage;
P is essentially estimator-independent. Chao–Shen is frozen for the HalfCheetah H2 run.
