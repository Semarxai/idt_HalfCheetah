# R4 — MMD sensitivity to reference size and kernel bandwidth (review point 4)
8 Nov 2026 · scripts halfcheetah/R4_mmd_sensitivity.py, R4_eval.py · outputs out/R4/, out/R4_point.csv/.txt,
out/R4_boot_*.csv, out/R4_bootsum.txt

Grid: reference size {800, 2,000} steps (drawn from steps 0–7,499) × RBF bandwidth {0.5, 1, 2} × median
heuristic. Same input (40-d z-scored stream), windows, 300-step window sample, calibration (150–290),
evaluation (≥300), latched one-sided rule as H3. A fresh reference sample is drawn per configuration (the
"800, ×1" row therefore differs slightly from H3, which shows the sampling variability). Nested protocol H5,
1,000 bootstrap resamples with selection inside each.

| MMD configuration | ±3σ (in-sample) | in-sample at 5/21 | nested ALPHA 0.25 [95%] | nested ALPHA 0.10 | P CS − MMD, nested [95%] | three-of-four CS − MMD [95%] |
|---|---|---|---|---|---|---|
| H3: ref 800, bw ×1 | 96.4%, 18/21 | 51.8 | 52.4 (6/21) [39.9, 75.6] | 43.5 (3/21) | [+1.8, +35.7] | [−1.2, +32.1] |
| ref 800, bw ×0.5 | 95.2%, 20/21 | 49.4 | 52.4 (6/21) [38.7, 67.9] | 46.4 (2/21) | [+6.5, +38.1] | [+6.0, +32.8] |
| ref 800, bw ×1 | 94.0%, 20/21 | 50.0 | 51.8 (6/21) [35.7, 69.6] | 42.3 (3/21) | [+4.2, +39.9] | [+2.4, +35.7] |
| ref 800, bw ×2 | 93.5%, 20/21 | 50.6 | 51.8 (6/21) [34.5, 67.9] | 42.3 (3/21) | [+5.4, +39.9] | [+3.6, +36.9] |
| ref 2,000, bw ×0.5 | 96.4%, 19/21 | 52.4 | 53.0 (6/21) [40.5, 66.7] | 50.0 (3/21) | [+5.4, +35.7] | [+5.3, +31.0] |
| ref 2,000, bw ×1 | 95.8%, 19/21 | 57.1 | 57.7 (6/21) [42.3, 70.2] | 53.0 (3/21) | [+1.2, +32.2] | [+1.8, +28.6] |
| ref 2,000, bw ×2 | 94.6%, 19/21 | 57.1 | **61.3 (6/21)** [41.1, 72.6] | 48.8 (3/21) | [0.0, +33.3] | [0.0, +29.8] |

Reference: P alone CS 74.4% (6/21) [60.1, 86.9]; three-of-four CS 72.6% (6/21) [63.1, 82.1]; reward 50.6%
(6/21) [35.1, 68.5]. False-alarm differences: all intervals include zero ([−14.3, +19.0] at widest).

## Reading
- The best MMD setting found (2,000-step reference, bandwidth ×2) reaches 61.3% held-out — 9 points
  above the original H3 setting — and still trails P (74.4%) and the fixed three-of-four rule (72.6%).
- Against P, the interval excludes zero in five of six settings and touches zero in the best one
  ([0.0, +33.3]); against three-of-four the same pattern holds.
- At ±3σ every MMD setting alarms on 18–20 of 21 controls: a distribution distance on this stream needs a
  much wider threshold than the information readings.
- The comparison is therefore not an artefact of an under-tuned MMD: across the grid, the best MMD
  matches the dynamics model (61.3%) and remains below the coverage-adjusted telemetry.
