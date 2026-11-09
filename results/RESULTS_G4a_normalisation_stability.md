# G4a — estimation stability of the normalisations against exact truth
8 Nov 2026 · code gridworld/G4a_normalisation_stability.py · outputs gridworld/G4a_results.csv, G4a_summary.csv

Statistics: P = I/C (C = H(S)+H(A)+H(S')), I/min{H(S,A),H(S')}, I/H(S,A,S'), unnormalised I(S,A;S').
Worlds: 7×7 (ε 0.2; triplet coverage 0.68–0.97) and 27×27 (ε 0.4; coverage 0.26–0.81, HalfCheetah-like 0.44
at W 200). Estimators plug-in, Miller–Madow, Chao–Shen, each from one coherent set of entropies. 10 seeds
(101–110). Relative bias = (mean window estimate − truth)/truth; drift = range of the relative bias across
window lengths (i.e. across coverage); within-window SD relative to truth.

## Stress world (sparse regime)
Relative bias %, by window (coverage):

| estimator | statistic | W 100 (0.26) | 200 (0.44) | 350 (0.54) | 500 (0.61) | 1000 (0.72) | 2000 (0.81) | drift (pts) |
|---|---|---|---|---|---|---|---|---|
| plug-in | **P** | +0.83 | **+1.67** | +2.01 | +2.15 | +2.14 | +1.93 | **1.32** |
| plug-in | I/min | +3.71 | +3.50 | +3.26 | +3.10 | +2.69 | +2.24 | 1.46 |
| plug-in | I/H(S,A,S') | +16.72 | +15.00 | +13.20 | +12.02 | +9.53 | +7.17 | 9.55 |
| plug-in | raw I | −20.75 | −13.08 | −8.40 | −5.94 | −2.61 | −0.70 | 20.05 |
| Chao–Shen | **P** | +0.03 | **+1.25** | +1.59 | +1.86 | +1.96 | +1.76 | **1.93** |
| Chao–Shen | I/min | +0.90 | +1.96 | +2.16 | +2.26 | +2.17 | +1.88 | 1.36 |
| Chao–Shen | I/H(S,A,S') | +0.09 | +5.12 | +6.52 | +6.92 | +6.12 | +4.65 | 6.83 |
| Chao–Shen | raw I | −9.62 | −5.86 | −3.50 | −1.82 | +0.11 | +0.86 | 10.48 |

Within-window relative SD at W 200, plug-in: P 0.74%, I/min 0.48%, I/H(S,A,S') 1.88%, raw I 2.28%.
Miller–Madow: P +2.05%, I/min +3.27%, joint +12.90%, raw −8.24% at W 200.

## Primary world (well sampled)
Plug-in relative bias at W 500 (coverage 0.91): P +2.42%, I/min +2.63%, joint +6.79%, raw +1.16%;
drift across W 50–2000: P 3.21, I/min 4.29, joint 16.08, raw 6.01 points. Chao–Shen: P +0.98%, I/min +0.93%,
joint +1.25%, raw +0.53% at W 500.

## Reading
- P is among the well-conditioned normalisations: bias 1–2% and nearly constant across coverage in the sparse
  regime, against +12–17% for the joint normalisation and −6 to −21% for unnormalised I.
- The min-entropy normalisation is comparably stable (bias 2–4%, drift 1.5 points; lower window noise). P is
  therefore not uniquely the best-estimated normalisation.
