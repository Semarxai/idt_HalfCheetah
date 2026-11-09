# G4b — entry point × mechanism, all four cells (grid world)
8 Nov 2026 · code gridworld/G4b_entry_mechanism.py, G4b_eval.py · outputs gridworld/G4b_scores_7.csv,
G4b_scores_27.csv, G4b_alarm_rates.csv, G4b_shifts.csv

Protocol as G4d (50,000-step trials, onset at step 15,000, latched, thresholds matched to 5/21 controls,
21 seeds per condition, Chao–Shen readings). The four cells the HalfCheetah design left partly empty:

| | stochastic (zero-mean) | systematic (directional) |
|---|---|---|
| **agent side** | 10% of executed actions random (log keeps the commanded action) | sensor offset: in the left room the agent observes, acts on and logs the cell to its east |
| **environment side** | slip 0.05 → 0.15 everywhere | wind: in the right room, p 0.2 push one cell south |

## Alarm rates (21 trials each), thresholds matched to 5/21 controls

| world | condition | three-of-four | P alone | H(S) | MMD | reward |
|---|---|---|---|---|---|---|
| 7×7 W500 | agent stochastic | 100% | 100% | 90% | 95% | 100% |
| 7×7 W500 | agent systematic | 100% | 100% | 100% | 100% | **14%** |
| 7×7 W500 | env stochastic | 100% | 100% | 100% | 100% | 100% |
| 7×7 W500 | env systematic | 100% | 100% | 90% | 100% | 100% |
| 27×27 W200 | agent stochastic | **100%** | **100%** | 38% | 71% | 10% |
| 27×27 W200 | agent systematic | 67% | 62% | 76% | 71% | 33% |
| 27×27 W200 | env stochastic | **100%** | **100%** | 52% | 86% | 10% |
| 27×27 W200 | env systematic | 100% | 100% | 24% | 100% | 100% |

Thresholds 7×7 / 27×27: three-of-four 3.11 / 3.51; P 3.37 / 4.03; H(S) 3.19 / 3.71; MMD 5.51 / 4.74; reward 3.62 / 2.69.

## Mean signed shift (evaluation mean − calibration mean) / calibration SD

| world | condition | P | H_f | H_b | ΔH | state term | action term | H(S) | MMD | reward |
|---|---|---|---|---|---|---|---|---|---|---|
| 7×7 | control | −0.04 | 0.03 | 0.01 | 0.01 | −0.04 | 0.01 | −0.02 | −0.09 | −0.01 |
| 7×7 | agent stochastic | −5.81 | 6.31 | 3.81 | 0.21 | −2.08 | **−2.34** | 1.19 | 2.03 | −2.57 |
| 7×7 | agent systematic | −4.77 | 2.97 | 1.97 | 0.02 | −3.93 | **0.16** | −4.32 | 25.96 | −0.01 |
| 7×7 | env stochastic | −6.53 | 7.22 | 4.51 | 0.33 | −2.53 | **−2.60** | 1.72 | 4.41 | −3.17 |
| 7×7 | env systematic | −4.07 | 4.41 | 3.08 | −0.11 | −3.32 | **−0.03** | 1.11 | 11.35 | 2.79 |
| 27×27 | control | −0.03 | 0.02 | 0.01 | −0.01 | −0.02 | 0.01 | −0.03 | 0.00 | −0.01 |
| 27×27 | agent stochastic | −3.12 | 3.46 | 1.56 | 0.17 | −0.81 | **−1.02** | −0.14 | 0.48 | −0.53 |
| 27×27 | agent systematic | −0.74 | 0.61 | 0.22 | 0.08 | −0.25 | −0.16 | −0.47 | 0.42 | 0.05 |
| 27×27 | env stochastic | −3.91 | 4.38 | 2.09 | 0.10 | −1.08 | **−1.24** | −0.03 | 0.60 | −0.63 |
| 27×27 | env systematic | −2.24 | 3.06 | 1.60 | −0.14 | −0.92 | **−0.30** | 0.90 | 2.07 | 0.90 |

## Reading
1. **All four cells are detected** by the telemetry (P alone and three-of-four: 100% in seven of eight
   world × cell combinations; the sensor offset in the sparse world 62–67%). **P falls in every cell**, as the
   interpretation predicts.
2. **Reward misses what does not touch the outcome**: the sensor offset (14%, 33%) and, in the sparse world,
   both stochastic faults (10%).
3. **H(S) and MMD respond to where the agent goes**: strong on the sensor offset (MMD shift 26), weak on the
   stochastic faults in the sparse world (H(S) 38–52%).
4. **The decomposition separates the mechanism, not the entry point.** Stochastic faults (action noise, slip)
   lower both the state term and the action term; systematic faults (sensor offset, wind) lower the state
   term while the action term stays near zero. Agent-side and environment-side faults of the same mechanism
   look alike. This is consistent with the submission's finding that the entry point is not identifiable
   from the readings (held-out-condition recall at chance) and adds that the mechanism is.
5. Caveat: in HalfCheetah, with a deterministic policy and binned symbols, the action term is
   representation-dependent (H8, G1); the mechanism signature is shown here in exact-symbol records only.
