# H2 — coverage-adjusted (Chao–Shen) estimation on HalfCheetah
8 Nov 2026 · scripts halfcheetah/H2_chaoshen.py, H2_eval.py, H2_boot.py · outputs out/H2cs/, out/H2_trial_scores_cs.csv,
out/H2_eval_output.txt, out/H2_boot_*.csv, out/H2_boot_summary.txt

Estimator frozen before any HalfCheetah detection outcome was inspected, after grid-world validation
at HalfCheetah-like coverage (gridworld/RESULTS_G1.md). Entropies H(S), H(A), H(S'), H(S,A), H(S,A,S')
estimated per window with Chao–Shen; I, P, H_f, H_b, ΔH derived from that one set. Same windows,
calibration (0–290), evaluation (≥300), latched rule and nested protocol as H5.

## Healthy calibration levels (mean over 189 retained trials)
| | plug-in | Chao–Shen |
|---|---|---|
| P | 0.3337 | 0.3020 |
| I(S,A;S') bits | 6.090 | 5.841 |
| H_f bits | 0.560 | 1.383 |
| H_b bits | 1.091 | 2.233 |
| ΔH bits | −0.531 | −0.850 |

## Undefined windows
278 of 249,732 windows (0.11%, all trials incl. screened seeds) have zero interaction budget (one state held for
the whole window); P is undefined there and the window is skipped (pandas max/mean skip NaN).

## Bound checks (187,299 windows, all trials)
P > ½: 0 · H_f < 0: 62 (0.03%) · H_b < 0: 21 · I < 0: 25. Reported, not clipped.

## In-sample (descriptive)
±3σ latched: three-of-four 86.9% at 11/21 controls (plug-in 69.0% at 5/21); four-of-four 70.2% at
7/21 (plug-in 49.4% at 1/21). Single P at 5/21 controls: Chao–Shen 73.8% (z 3.40) vs plug-in 58.9% (z 3.36).

## Nested leave-one-agent-out (primary), 1,000 bootstrap resamples with selection inside each
ALPHA 0.25:

| method | held-out detection [95%] | held-out control alarms [95%] |
|---|---|---|
| **consensus, Chao–Shen** | **76.8%** [64.9, 86.9] | 6/21 [23.7%, 47.6%] |
| **P alone, Chao–Shen** | **74.4%** [60.1, 86.9] | 6/21 [19.0%, 33.3%] |
| consensus, plug-in | 72.0% [57.7, 82.1] | 8/21 [19.0%, 47.6%] |
| MMD | 52.4% [39.9, 75.6] | 6/21 [19.0%, 33.3%] |
| reward | 50.6% [35.7, 67.3] | 6/21 [19.0%, 33.5%] |
| H(S) | 50.0% [39.9, 58.9] | 6/21 [14.3%, 33.3%] |

Selected operating points: consensus Chao–Shen one-of-four at z ≈ 4.1 (15 folds) / 4.05 (6);
P Chao–Shen z ≈ 3.4.

Differences (percentage points, 95%):

| | detection | false alarms |
|---|---|---|
| consensus CS − reward | [+8.9, +41.7] | [−9.5, +23.8] |
| consensus CS − MMD | [+4.8, +37.5] | [−9.5, +23.8] |
| consensus CS − H(S) | [+13.1, +39.9] | [−9.5, +28.6] |
| consensus CS − consensus plug-in | [−9.5, +19.1] | [−19.2, +23.8] |
| P CS − reward | [+4.2, +41.1] | [−14.3, +9.5] |
| P CS − MMD | [0.0, +34.5] | [−9.5, +9.5] |
| P CS − H(S) | [+7.1, +39.3] | [−9.5, +14.3] |
| P CS − consensus plug-in | [−13.7, +18.5] | [−23.8, +9.5] |

ALPHA 0.10: consensus CS 65.5% at 5/21; P CS 64.3% at 3/21; consensus plug-in 61.3% at 2/21;
P plug-in 38.7% at 3/21; reward 35.1% at 3/21.

## Decision gate
Bias correction changes the results materially — upward. Per the plan, corrected results become
primary. With the better estimate of the framework's quantities, the held-out advantage holds over
reward, over the occupancy-only H(S) and now also over MMD, at equal held-out control-alarm counts
(6/21 each).
