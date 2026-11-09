# E5 — histogram KL on the binned alphabet (HalfCheetah, 21 agents)

8 Nov 2026 · code halfcheetah/E5_kl.py (raw logs), E5_eval.py, E5_boot.py  · outputs out/E5/ (per-trial npz), out/E5_trial_scores.csv, out/E5_curve.csv, E5_by_condition.csv, E5_boot.csv

Design (fixed in the script): per 500-step window (stride 50), KL(window ‖ reference) in bits, add-½ smoothing over the union of symbols in the window and the reference; reference = steps 0–7,499 of the same run (as for MMD); published symbols (baseline.json, 3 bins). Two alphabets: the joint triplet (S, A, S′) and S alone. Trial score: one-sided z against calibration windows 150–290, max over evaluation windows ≥ 300. Nested leave-one-agent-out (H5 protocol).

## Held out, ALPHA 0.25 (168 fault trials; every method at 6/21 control alarms)

| method | detection | 95% interval |
|---|---|---|
| three-of-four (Chao–Shen) | 72.6% | [62.5, 82.1] |
| P alone (Chao–Shen) | 74.4% | [60.1, 86.9] |
| **KL, joint (S, A, S′)** | **72.6%** | [60.7, 83.9] |
| **KL, S only** | **70.2%** | [56.5, 81.5] |
| MMD (RBF kernel, median heuristic, standardised 40-d raw vectors) | 52.4% | [38.7, 74.4] |
| reward | 50.6% | [33.9, 67.3] |

Differences (1,000 agent-level resamples, selection inside): three-of-four − KL joint [−14.3, +14.3]; P − KL joint [−18.5, +18.5]; three-of-four − KL S [−13.7, +20.2]; P − KL S [−12.5, +20.8]; KL joint − reward [+7.7, +35.7]; KL joint − MMD [+1.2, +32.7].

## Detection–false-alarm curve (held out, %)

| ALPHA | three-of-four | P | KL joint | KL S | MMD | H(S) | reward |
|---|---|---|---|---|---|---|---|
| 0.05 | 66.7 | 63.1 | 59.5 | 52.4 | 38.7 | 38.7 | 34.5 |
| 0.10 | 67.3 | 64.3 | 66.7 | 60.1 | 43.5 | 43.5 | 35.1 |
| 0.15 | 67.9 | 66.1 | 70.2 | 66.1 | 50.0 | 45.8 | 39.3 |
| 0.20 | 69.0 | 71.4 | 70.8 | 67.3 | 51.2 | 47.6 | 44.6 |
| 0.25 | 72.6 | 74.4 | 72.6 | 70.2 | 52.4 | 50.0 | 50.6 |
| 0.30 | 74.4 | 79.8 | 73.8 | 70.8 | 58.3 | 53.6 | 61.3 |
| 0.40 | 79.8 | 82.1 | 76.8 | 77.4 | 72.6 | 58.9 | 69.6 |
| 0.50 | 83.3 | 88.7 | 83.9 | 85.7 | 75.0 | 60.1 | 76.8 |

## By fault condition (held out, ALPHA 0.25, %)

| condition | three-of-four | P | KL joint | KL S | MMD | reward |
|---|---|---|---|---|---|---|
| act_noise_01% | 52.4 | 47.6 | 38.1 | 57.1 | 47.6 | 47.6 |
| act_noise_03% | 95.2 | 90.5 | 81.0 | 76.2 | 28.6 | 42.9 |
| act_noise_04% | 95.2 | 100.0 | 95.2 | 76.2 | 52.4 | 47.6 |
| force_torso_10N | 47.6 | 52.4 | 52.4 | 52.4 | 61.9 | 38.1 |
| force_torso_5N | 52.4 | 52.4 | 42.9 | 33.3 | 23.8 | 38.1 |
| gravity_110% | 52.4 | 71.4 | 100.0 | 100.0 | 90.5 | 61.9 |
| obs_noise_01% | 90.5 | 90.5 | 81.0 | 76.2 | 47.6 | 42.9 |
| obs_noise_03% | 95.2 | 90.5 | 90.5 | 90.5 | 66.7 | 85.7 |

## Reading

1. **A smoothed histogram KL on the same binned alphabet detects as many faults as the telemetry** under the latched window detector (72.6% vs 72.6–74.4%; differences centred on zero, ±14–19 points). Unlike MMD with an RBF kernel on standardised raw vectors (52.4%), the KL row is a strong generic baseline on this record.
2. **The detection advantage of the telemetry over generic distribution-shift tests is therefore not general**: it holds against MMD, H(S) and the dynamics model, not against KL on the binned alphabet. Detection alone does not distinguish them.
3. **What the accounting adds is the reading, not the alarm rate**: KL says the window distribution moved; P, H_f, H_b, ΔH say which relationship in the loop moved (decomposition, H8; grid world G2–G4). By condition, KL is stronger on gravity (100% vs 52–71%) and action noise 4%; the telemetry on action noise 3% (91–95% vs 81%), observation noise 1% (91% vs 81%) and force 5 N (52% vs 43%).
4. KL on S alone (occupancy distribution) reaches 70.2% while the occupancy entropy H(S) reaches 50.0%: the faults shift where the agent goes as well as how the loop is organised. The grid world separates the two (G2: mechanism changes at fixed visitation move P, not H(S) or MMD).
