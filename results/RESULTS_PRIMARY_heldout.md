# PRIMARY held-out comparison (main display) — includes H13 dynamics model and H14 reward + telemetry
8 Nov 2026 · scripts halfcheetah/PRIMARY_nested_table.py, analysis/H13_dynamics.py · outputs
out/PRIMARY_trial_scores.csv, out/PRIMARY_point.txt, out/PRIMARY_curve.txt, out/PRIMARY_nested_curve.csv,
out/PRIMARY_boot_*.csv, out/PRIMARY_bootsum.txt, out/H13/

Retained cohort: 21 agents (11 SAC, 10 PPO), 168 perturbation trials, 21 controls.
Protocol (H5): nested leave-one-agent-out; inner agents select the operating point maximising inner
detection with inner control false-alarm rate ≤ ALPHA; held-out agent calibrates its own statistics,
its outcomes never enter selection; 1,000 agent-level bootstrap resamples with the whole selection
repeated inside each. Primary ALPHA 0.25, declared in advance.

Methods
- k-of-four telemetry (Chao–Shen): k-of-{P, H_f, H_b, ΔH} at common z; k and z both selected (named "consensus" in earlier notes). Fixed three-of-four reported in RESULTS_R1_fixed_rule.md.
- P alone (Chao–Shen).
- reward + telemetry (H14, fixed rule): alarm when the consensus fires OR reward exceeds the same z.
- dynamics model (H13): per-trial MLP (2×64) fitted on the trial's own healthy steps 0–7,499 to predict
  S' from (S, A) in the shared z-scored 40-d units; window statistic = mean squared one-step error;
  calibration windows 150–290; one-sided, latched.
- MMD (H3), H(S), reward: as before.

## Primary (ALPHA 0.25)

| method | held-out detection [95%] | held-out control alarms [95%] | consensus − method, detection [95%] |
|---|---|---|---|
| reward + telemetry (fixed OR) | 77.4% [63.1, 86.9] | 6/21 [19.0, 47.6] | [−7.1, +9.5] |
| **k-of-four telemetry, nested-selected (Chao–Shen)** — one-of-four in all 21 folds | **76.8%** [64.3, 87.5] | 6/21 [23.8, 47.6] | — |
| P alone (Chao–Shen) | 74.4% [60.7, 86.9] | 6/21 [19.0, 33.3] | [−10.7, +14.9] |
| dynamics model | 61.3% [38.1, 78.0] | 6/21 [19.0, 33.3] | [−1.8, +38.7] |
| MMD | 52.4% [37.5, 75.0] | 6/21 [19.0, 33.3] | [+4.8, +38.1] |
| reward | 50.6% [34.5, 67.9] | 6/21 [19.0, 33.3] | [+7.7, +41.7] |
| H(S) | 50.0% [39.3, 58.9] | 6/21 [14.3, 33.3] | [+13.7, +39.9] |

All false-alarm differences include zero. Every method ends at 6/21 held-out control alarms.

## Nested curve (held-out detection % / held-out control alarm %)

| ALPHA | consensus CS | P CS | reward + telemetry | dynamics | MMD | H(S) | reward |
|---|---|---|---|---|---|---|---|
| 0.05 | 67.3 / 9.5 | 63.1 / 4.8 | 63.7 / 14.3 | 38.1 / 9.5 | 38.7 / 9.5 | 38.7 / 9.5 | 34.5 / 9.5 |
| 0.10 | 65.5 / 23.8 | 64.3 / 14.3 | 71.4 / 19.0 | 45.8 / 14.3 | 43.5 / 14.3 | 43.5 / 14.3 | 35.1 / 14.3 |
| 0.15 | 68.5 / 23.8 | 66.1 / 19.0 | 73.2 / 23.8 | 54.8 / 19.0 | 50.0 / 19.0 | 45.8 / 19.0 | 39.3 / 19.0 |
| 0.20 | 73.2 / 28.6 | 71.4 / 23.8 | 71.4 / 23.8 | 58.3 / 23.8 | 51.2 / 23.8 | 47.6 / 19.0 | 44.6 / 23.8 |
| 0.25 | 76.8 / 28.6 | 74.4 / 28.6 | 77.4 / 28.6 | 61.3 / 28.6 | 52.4 / 28.6 | 50.0 / 28.6 | 50.6 / 28.6 |
| 0.30 | 79.8 / 33.3 | 79.8 / 33.3 | 80.4 / 33.3 | 66.7 / 33.3 | 58.3 / 33.3 | 53.6 / 33.3 | 61.3 / 33.3 |
| 0.40 | 83.3 / 42.9 | 82.1 / 42.9 | 80.4 / 47.6 | 72.6 / 42.9 | 72.6 / 38.1 | 58.9 / 38.1 | 69.6 / 42.9 |
| 0.50 | 87.5 / 61.9 | 88.7 / 52.4 | 87.5 / 61.9 | 73.2 / 52.4 | 75.0 / 52.4 | 60.1 / 52.4 | 76.8 / 52.4 |

At 9.5% held-out false alarms (ALPHA 0.05): consensus 67.3%, reward 34.5%, dynamics 38.1%, MMD 38.7%.

## Held-out detection by condition (ALPHA 0.25, %)

| condition | consensus CS | P CS | reward + tel. | dynamics | MMD | H(S) | reward |
|---|---|---|---|---|---|---|---|
| action noise 1% | 57.1 | 47.6 | 52.4 | 42.9 | 47.6 | 28.6 | 47.6 |
| action noise 3% | 100.0 | 90.5 | 100.0 | 52.4 | 28.6 | 33.3 | 42.9 |
| action noise 4% | 95.2 | 100.0 | 95.2 | 76.2 | 52.4 | 57.1 | 47.6 |
| obs noise 1% | 85.7 | 90.5 | 85.7 | 66.7 | 47.6 | 57.1 | 42.9 |
| obs noise 3% | 100.0 | 90.5 | 100.0 | 76.2 | 66.7 | 95.2 | 85.7 |
| force 5 N | 61.9 | 52.4 | 61.9 | 47.6 | 23.8 | 38.1 | 38.1 |
| force 10 N | 42.9 | 52.4 | 47.6 | 61.9 | 61.9 | 28.6 | 38.1 |
| gravity 110% | 71.4 | 71.4 | 76.2 | 66.7 | 90.5 | 61.9 | 61.9 |

## Other numbers
- ALPHA 0.10: consensus 65.5% (5/21), P CS 64.3% (3/21), reward + telemetry 71.4% (4/21), dynamics
  45.8% (3/21), MMD 43.5% (3/21), H(S) 43.5% (3/21), reward 35.1% (3/21).
- Dynamics model in-sample: +3σ 73.2% at 9/21 controls; at 5/21 controls (z 4.15) 59.5%.

## Reading
- Telemetry exceeds reward, the occupancy-only H(S) and MMD held-out at equal false-alarm counts →
  report that bounded result without universalising.
- The predictive dynamics model is the strongest baseline (61.3%); the consensus is 15.5 points ahead,
  interval [−1.8, +38.7] → ahead but not conclusive at 21 agents; complementary on force 10 N, where the
  dynamics model is better.
- Adding reward to the telemetry adds little at this level (77.4% vs 76.8%) and helps at low false-
  alarm levels (71.4% vs 65.5% at ALPHA 0.10).
- Held-out false alarms 6/21 for every method → telemetry, not a validated alarm system.
