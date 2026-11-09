# H5 (+ H11, H12) — nested leave-one-agent-out, all methods
8 Nov 2026 · script halfcheetah/H5_nested.py · trial scores out/H5_trial_scores.csv · curve out/H5_nested_curve.csv
· bootstrap out/H5_boot_*.csv

Protocol (declared before running): outer loop over the 21 agents; inner agents choose the operating
point — threshold z, and k for the consensus — maximising inner detection subject to inner control
false-alarm rate ≤ ALPHA (ties → larger z). The held-out agent's healthy windows (0–290) calibrate its
own statistics; its outcomes never enter the selection. Primary ALPHA = 0.25 (the level of the paper's
operating point). Same treatment for every method. Intervals: 1,000 agent-level bootstrap resamples,
nested selection repeated inside each resample (seed 20260917+chunk).

## Primary held-out result (ALPHA = 0.25)

| method | held-out detection [95%] | held-out control alarms [95%] | selected |
|---|---|---|---|
| three-of/four-of-four consensus | **72.0%** [59.5, 82.7] | 8/21 (38%) [19, 48] | 4-of-4 @ z 2.65 (15 folds), 3-of-4 @ 2.9 (6) |
| P alone | 59.5% [36.9, 79.2] | 6/21 [19, 33] | z ≈ 3.35 |
| MMD | 52.4% [39.3, 74.4] | 6/21 [19, 33] | z ≈ 7.9 |
| reward | 50.6% [35.1, 66.7] | 6/21 [19, 33] | z ≈ 2.9 |
| H(S) | 50.0% [39.9, 59.5] | 6/21 [14, 33] | z ≈ 3.4 |

Consensus − other (detection, percentage points, 95%):
reward [+1.8, +40.5] · H(S) [+10.7, +33.3] · MMD [−3.0, +35.1] · P alone [−3.6, +31.0].
Consensus − other (false alarms): all intervals include zero ([−9.5, +23.8] to [−9.5, +28.6]).

## Nested detection–false-alarm curve (held-out detection % / held-out control alarm %)

| ALPHA | consensus | reward | MMD | H(S) | P alone |
|---|---|---|---|---|---|
| 0.05 | 58.9 / 14.3 | 34.5 / 9.5 | 38.7 / 9.5 | 38.7 / 9.5 | 38.1 / 4.8 |
| 0.10 | **61.3 / 9.5** | 35.1 / 14.3 | 43.5 / 14.3 | 43.5 / 14.3 | 38.7 / 14.3 |
| 0.15 | 64.9 / 28.6 | 39.3 / 19.0 | 50.0 / 19.0 | 45.8 / 19.0 | 45.8 / 19.0 |
| 0.20 | 69.6 / 38.1 | 44.6 / 23.8 | 51.2 / 23.8 | 47.6 / 19.0 | 58.9 / 23.8 |
| 0.25 | 72.0 / 38.1 | 50.6 / 28.6 | 52.4 / 28.6 | 50.0 / 28.6 | 59.5 / 28.6 |
| 0.30 | 73.2 / 47.6 | 61.3 / 33.3 | 58.3 / 33.3 | 53.6 / 33.3 | 64.3 / 33.3 |
| 0.40 | 73.8 / 33.3 | 69.6 / 42.9 | 72.6 / 38.1 | 58.9 / 38.1 | 74.4 / 42.9 |
| 0.50 | 82.7 / 61.9 | 76.8 / 52.4 | 75.0 / 52.4 | 60.1 / 52.4 | 82.7 / 42.9 |

Held-out false-alarm rates are not monotone in ALPHA (21 controls). Reading the curve as held-out
(false alarm, detection) pairs: up to about one-third false alarms the consensus detects more at equal
or lower false-alarm rate — e.g. 61.3% at 9.5% vs reward 35.1%, MMD 43.5% at 14.3% — and the
methods converge above ~40% false alarms.

## Reading
- The held-out advantage over reward and over the occupancy-only H(S) is supported (intervals exclude
  zero); over MMD the point estimate is +19.6 points but the interval includes zero at 21 agents →
  complementarity and structural interpretation, not detector superiority (G3, H3).
- Held-out false alarms at the primary level are 8/21 (interval 19–48%) → exploratory telemetry, not a
  validated alarm system; the low-false-alarm part of the curve is where the consensus is strongest.
