# H6 — seed accounting and results including screened seeds
8 Nov 2026 · scripts halfcheetah/H6_extra_seeds.py, H6_eval.py · outputs out/H6_extended_label_map.csv,
out/H6_trial_scores_all.csv, out/H6_eval_output.txt

## Accounting (from the archive)
- SAC: 11 seeds evaluated under the protocol, all retained (sac01, 03, 05, 07, 09, 10, 11, 13, 14, 18, 19).
- PPO retained (10): ppo02, 04, 08, 11, 13, 17, 27, 30, 33, 36. ppo36 is a second evaluation run of the same policy as
  seed 19 (raw dir ppo19_p, 2 Jan 2026), distinct from the screened ppo19 run (raw dir ppo19, 1 Jan).
- PPO screened, logs present: ppo15, ppo18, ppo19 (in trial_label_map.csv); ppo06, ppo20, ppo21,
  ppo35 (raw dirs ppo6, ppo20_p, ppo21, ppo35_p; mapped here with the map's file-name convention,
  *_00_percent = 1%; ppo35's extra obs_noise_04 run not used). FROZEN_CONFIG.md records the exclusions
  ppo15, ppo18, ppo19, ppo21.
- Reconciliation with the manuscript: **27 trained seeds = 11 SAC + 16 PPO; 28 logged runs**, because PPO
  seed 19 was evaluated twice with the same checkpoint (verified in H10b gate 1) — the first evaluation (ppo19) was screened, the second (ppo36) retained. PPO seeds:
  retained {2, 4, 8, 11, 13, 17, 19, 27, 30, 33}; screened {6, 15, 18, 20, 21, 35} plus the first seed-19 run.
  "16 PPO run, 10 retained, 6 excluded" holds at the level of seeds. All 28 logged runs are analysed below.
- Screening criterion: inspection of unperturbed return traces; no numerical criterion recorded.

## Results by cohort (nested leave-one-agent-out, ALPHA 0.25; in-sample ±3σ for reference)

| cohort | agents | trials / controls | consensus CS | P CS | consensus plug-in | reward | MMD | H(S) |
|---|---|---|---|---|---|---|---|---|
| retained (paper) | 21 (11 SAC, 10 PPO) | 168 / 21 | 76.8% · 6/21 | 74.4% · 6/21 | 72.0% · 8/21 | 50.6% · 6/21 | 52.4% · 6/21 | 50.0% · 6/21 |
| + ppo15, 18, 19 | 24 (11, 13) | 192 / 24 | 68.2% · 7/24 | 65.6% · 6/24 | 61.5% · 6/24 | 48.4% · 6/24 | 49.0% · 6/24 | 49.0% · 6/24 |
| all logged | 28 (11, 17) | 224 / 28 | 62.5% · 9/28 | 63.8% · 7/28 | 60.7% · 6/28 | 43.8% · 7/28 | 46.4% · 7/28 | 47.3% · 7/28 |

In-sample ±3σ (descriptive): retained — consensus 3-of-4 69.0% at 5/21, reward 44.0% at 5/21;
24 agents — 67.2% at 6/24, reward 42.7% at 5/24; 28 agents — 66.5% at 8/28, reward 44.6% at 8/28.

Reading: including every logged seed lowers detection for all methods and leaves the ordering
unchanged — telemetry ahead of MMD, H(S) and reward by 15–20 points at similar held-out false-alarm
counts. Control alarms rise equally for the monitor and for reward (in-sample 5/21 → 8/28 for both),
so screening did not favour the monitor's false-alarm count.

## Screened seeds' control trials (max |z| over evaluation windows)

| seed | consensus 3rd | consensus CS 1st | P CS | reward | MMD | calibration reward/step | CV |
|---|---|---|---|---|---|---|---|
| ppo15 | 2.96 | 4.59 | 3.69 | 2.44 | 2.84 | 2.08 | 0.03 |
| ppo18 | 3.06 | 6.82 | 5.21 | 2.28 | 5.36 | 2.00 | 0.05 |
| ppo19 | 2.94 | 3.54 | 2.67 | 1.85 | 4.84 | 4.13 | 0.02 |
| ppo06 | 4.53 | 9.20 | 9.20 | 6.67 | 23.51 | 7.33 | 0.16 |
| ppo20 | 2.99 | 3.82 | 3.61 | 3.03 | 3.59 | 2.11 | 0.03 |
| ppo21 | 3.12 | 5.34 | 4.44 | 16.48 | 691.30 | 5.32 | 0.05 |
| ppo35 | 2.82 | 5.94 | 3.31 | 2.42 | 4.98 | 2.04 | 0.04 |

Calibration-window reward level and CV do not separate screened from retained seeds (retained PPO
reward/step 1.96–7.06, CV 0.026–0.048; retained SAC 7.73–15.93, CV 0.034–0.074), except ppo06
(CV 0.16). ppo21's nominal run contains a large excursion after calibration (reward z 16.5, MMD z 691).
The screening was based on whole return traces, which the calibration windows do not capture; it
cannot be reconstructed numerically from the archive.


## H6b — ppo19 and ppo36 held out together (8 Nov)
Both runs are evaluations of one checkpoint (H10b gate 1). Holding them out as one group (script H6b_grouped.py, out/H6b_grouped.csv) gives results identical to the separate hold-out above for every method and cohort: 24 agents — three-of-four Chao–Shen 67.2% (5/24), P 65.6% (6/24), reward 48.4% (6/24); all 28 runs — three-of-four 65.2% (7/28), P 63.8% (7/28), reward 43.8% (7/28).
