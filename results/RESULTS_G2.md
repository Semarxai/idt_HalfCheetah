# G2 — occupancy vs mechanism (grid world)
8 Nov 2026 · code: G2_occupancy_vs_mechanism.py · seed 20261109 · plug-in estimates

Two constructions per case: (i) operational rollouts, occupancy free to change; (ii) one-step
sampling with the healthy d0(s,a) held fixed, s' from the healthy or perturbed kernel.
Worlds: primary 7×7 (ε 0.2, W 500); stress 27×27 (ε 0.4, W 200, HalfCheetah-like coverage).
d = estimated change / healthy window SD. Reward = goal rate per window.

## Standardised shifts d — primary world

| case | constr. | P | H_f | H_b | I(S;S') | I(A;S'∣S) | H(S) | reward |
|---|---|---|---|---|---|---|---|---|
| A visitation only (start moved) | i | 0.41 | 0.44 | −0.76 | −3.93 | 0.19 | −5.12 | −0.59 |
| A | ii | 0.04 | 0.01 | −0.04 | 0.00 | 0.06 | −0.03 | 0.08 |
| D action noise 10% (agent side) | i | −5.98 | 6.78 | 3.66 | −0.87 | −1.65 | 1.22 | −2.47 |
| D | ii (d0 fixed) | −5.34 | 6.85 | 3.24 | −1.72 | −1.64 | 0.07 | −0.55 |
| F reward only | i | −0.02 | 0.03 | 0.12 | −0.03 | 0.15 | 0.07 | −3.28 |
| B zone slip 0.30→0.60 | i | −0.06 | 0.24 | 0.13 | 0.21 | 0.00 | 0.29 | −0.13 |
| C zone slip one-sided | i | 0.12 | −0.31 | −0.06 | −0.23 | 0.22 | −0.22 | 0.26 |

## Standardised shifts d — stress world (HalfCheetah-like coverage)

| case | constr. | P | H_f | H_b | I(S;S') | I(A;S'∣S) | H(S) | reward |
|---|---|---|---|---|---|---|---|---|
| A visitation only | i | −0.71 | 0.53 | 1.43 | −1.89 | 1.39 | −1.93 | −0.10 |
| D action noise 10% | i | −2.12 | 3.01 | 1.16 | −0.44 | −0.35 | −0.21 | −0.56 |
| D | ii (d0 fixed) | −1.40 | 2.70 | 0.77 | −0.20 | −0.42 | 0.02 | −0.26 |
| F reward only | i | 0.00 | −0.07 | 0.10 | −0.05 | 0.14 | −0.01 | −4.18 |

Full tables, exact population changes and estimated changes: G2_results.csv.

## Findings
1. **Mechanism, not occupancy, drives the detection.** With occupancy held fixed (ii), the
   action-noise fault still moves P and H_f strongly (d −5.3 / +6.9 primary; −1.4 / +2.7 stress)
   while H(S) does not move (0.07; 0.02).
2. **P is the least occupancy-sensitive reading.** A pure visitation change moves the occupancy
   statistics most (H(S) −5.1, I(S;S') −3.9) and P least (0.41). In the sparse regime P −0.71 vs H(S)
   −1.93; there H_b and I(A;S'∣S) acquire visitation sensitivity through estimator bias (true change
   ≈ 0, estimated +0.12 / +0.11) — a property of the sparse record, stated as such.
3. **The loop changes while reward moves much less.** Action noise: P/H_f |d| 6–7 vs reward 2.5
   (primary rollout); 2–3 vs 0.6 (stress). With d0 fixed: 5–7 vs 0.55.
4. **The converse:** a reward-only change leaves every loop statistic flat (|d| ≤ 0.15) while reward
   moves (−3.3; −4.2). The two readings answer different questions.
5. **Sparse-regime attenuation.** Estimates move in the true direction but attenuated (action noise,
   stress world: ΔH_f estimated 0.09 vs true 0.38). Faults confined to rarely visited states (B, C)
   give small true changes and small shifts in every reading.
