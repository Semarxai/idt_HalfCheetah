# R2 — grid-world results replicated over 10 independent seeds (review point 2)
8 Nov 2026 · code gridworld/R2_replicate.py (unchanged G1–G3 functions, generator reseeded: seeds 101–110) ·
outputs gridworld/R2_G1_reps.csv, R2_G2_reps.csv, R2_G3_reps_g3a.csv, R2_G3_reps_g3b.csv

Values: mean ± SD over the 10 seeds (range in brackets where given).

## G1 — truth vs estimate
Stress world 27×27, ε 0.4, W 200 (triplet Good–Turing coverage 0.45; HalfCheetah 0.41):

| quantity | truth | plug-in relative bias | Miller–Madow | Chao–Shen |
|---|---|---|---|---|
| P | 0.4275 | +1.66% [1.61, 1.74] | +2.03% [1.98, 2.13] | +1.25% [1.16, 1.39] |
| H_f | 0.3358 | −80.0% [−80.7, −79.3] | −74.5% | −47.4% [−49.2, −45.4] |
| H_b | 1.7123 | −61.9% [−62.1, −61.7] | −52.6% | −23.1% [−23.6, −22.1] |
| I(A;S'∣S) | 1.0964 | −49.6% [−50.0, −49.0] | −37.7% | −4.3% [−5.4, −3.1] |

Same world, W 500 (coverage 0.60): P +2.14% (plug-in), +1.84% (CS); H_f −70.0% / −50.8%; H_b −48.4% / −26.7%.
Primary world 7×7, W 500: ε 0.2 — P +2.39% (plug-in), +0.94% (CS); ε 0 (deterministic) — I(A;S'∣S) = 0 in
every window, every estimator, every seed. SD of the bias across seeds ≤ 0.007 bits for every quantity.
Out-of-bound windows: 0 for every estimator across the 10 seeds (the single Chao–Shen window in the first G1
run was seed-specific).

## G2 — occupancy vs mechanism (standardised shift d)

| world | construction | case | P | H_f | H_b | I(S;S') | H(S) | reward |
|---|---|---|---|---|---|---|---|---|
| 7×7 | rollout | A visitation only | +0.42±0.11 | +0.36±0.09 | −0.80±0.18 | −3.63±0.18 | −4.58±0.33 | −0.55±0.11 |
| 7×7 | rollout | D action noise 10% | −5.78±0.20 | +6.57±0.32 | +3.62±0.19 | −0.79±0.08 | +1.12±0.14 | −2.57±0.17 |
| 7×7 | rollout | F reward only | −0.02±0.09 | +0.01±0.08 | +0.00±0.14 | +0.01±0.10 | +0.02±0.07 | −3.27±0.16 |
| 7×7 | d0 fixed | A visitation only | −0.03±0.09 | +0.02±0.11 | +0.03±0.07 | −0.07±0.11 | −0.05±0.11 | −0.07±0.13 |
| 7×7 | d0 fixed | D action noise 10% | −5.42±0.29 | +6.57±0.37 | +3.15±0.15 | −1.71±0.10 | −0.08±0.09 | −0.44±0.08 |
| 27×27 | rollout | A visitation only | −0.54±0.10 | +0.48±0.07 | +1.22±0.10 | −1.77±0.11 | −1.86±0.10 | −0.03±0.05 |
| 27×27 | rollout | D action noise 10% | −1.96±0.09 | +3.15±0.09 | +1.08±0.09 | −0.35±0.08 | −0.15±0.08 | −0.54±0.04 |
| 27×27 | rollout | F reward only | +0.04±0.07 | −0.05±0.05 | −0.03±0.07 | +0.00±0.07 | −0.00±0.07 | −4.05±0.09 |
| 27×27 | d0 fixed | D action noise 10% | −1.28±0.12 | +2.58±0.17 | +0.67±0.08 | −0.12±0.07 | +0.04±0.08 | −0.09±0.07 |

Cases B (zone slip 0.30→0.60) and C (one-sided slip) stay below |0.5| in every reading (full table in CSV).

## G3 — MMD and the accounting (standardised shift d)

| world | construction | case | MMD delta | MMD RBF | P | H_f | H(S) |
|---|---|---|---|---|---|---|---|
| 7×7 | rollout | visitation only | +28.50±3.59 | +18.63±3.99 | +0.48±0.13 | +0.31±0.12 | −4.72±0.30 |
| 7×7 | rollout | action noise 10% | +1.70±0.40 | +0.42±0.30 | −5.91±0.25 | +6.62±0.34 | +1.13±0.10 |
| 7×7 | d0 fixed | action noise 10% | +0.55±0.20 | +0.01±0.12 | −5.37±0.47 | +6.52±0.40 | +0.10±0.10 |
| 27×27 | rollout | visitation only | +1.52±0.51 | +2.37±0.70 | −0.55±0.09 | +0.49±0.12 | −1.86±0.11 |
| 27×27 | rollout | action noise 10% | +0.52±0.12 | +0.24±0.14 | −2.11±0.10 | +3.27±0.26 | −0.15±0.10 |
| 27×27 | d0 fixed | action noise 10% | +0.10±0.12 | +0.01±0.09 | −1.32±0.11 | +2.69±0.19 | −0.01±0.07 |

## Reading
Every G1–G3 finding replicates with small between-seed spread: P carries 1–2% bias where the components
carry 23–80%; a mechanism change moves P and H_f with visitation held fixed while H(S) and MMD do not; a
pure visitation change moves H(S) and MMD while P barely moves; a reward-only change leaves the loop
statistics flat.
