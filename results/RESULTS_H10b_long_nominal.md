# H10b — long nominal runs, false-alarm target at the episode scale (HalfCheetah, 21 frozen agents)

8 Nov 2026 · code long_runs/ (policies.py, readings.py, runner.py, gate1_actions.py, gate2_compare.py, batch.sh, post_batch.py)

## Design (fixed before the runs)

- Agents: the 21 retained agents (SAC 01, 03, 05, 07, 09, 10, 11, 13, 14, 18, 19; PPO 02, 04, 08, 11, 13, 17, 27, 30, 33, 36). No training.
- One nominal run per agent: 1,000,000 steps = 1,000 episodes; written as 10 segments of 100 episodes; each segment starts from a reset seeded by 9,100,000 + 1000·(agent index) (×100 + segment). The archived evaluations used unseeded resets, so these seeds are new.
- Evaluation loop replicated from the archived code (app.py evaluate_and_log_*): gymnasium HalfCheetah-v4 (here gymnasium 1.4.0, MuJoCo 3.4.0), VecNormalize observation statistics of each agent (training off, clip 10, reward not normalised), deterministic policy, auto-reset observation logged as s_next of the last step, second reset starts the next episode. Policies evaluated in numpy from policy.pth (no torch).
- Readings: published pipeline (baseline.json, 3 bins, W 500, stride 50), Chao–Shen; vectorised integer symbols verified against analyze_trajectory on an archived control (max difference 3.6e-15 in all five entropies).
- Calibration: windows 0–290 (steps 0–15,000) of each run; monitoring from window 300 (985 episodes); latched alarm (first alert).
- Target: mean time to first false alarm ≥ 100 episodes. For each held-out agent, the other 20 agents' long runs set the smallest z meeting the target (censored runs count at 985 episodes, conservative); methods three-of-four telemetry, P alone, reward. Frozen thresholds then applied to the held-out agent's archived control and 8 fault trials.
- Batches (two workers each): B1 sac01, ppo02, sac03, ppo04 · B2 sac05, ppo11, sac07, ppo13 · B3 sac09, ppo17, sac10, ppo27 · B4 sac11, ppo30, sac13, ppo33 · B5 sac14, sac18, sac19, ppo36, ppo08.

## Gate 1 — the checkpoints reproduce the logged actions

Logged (normalised) observations of each archived control (first 5,000 steps) → policy in numpy → max |action − logged action|. PPO hidden activation: ReLU for the 256-unit networks (seeds 2, 4, 8, 11, 13), tanh for the 64-unit networks (17, 27, 30, 33, 19).

| agent | archived control | max abs difference |
|---|---|---|
| ppo02 | ppo_seed2_no_noise_20251226_110411 | 4.1e-07 |
| ppo04 | ppo_seed4_no_noise_20251226_191400 | 4.4e-07 |
| ppo08 | ppo_seed8_no_noise_20251226_190002 | 3.0e-07 |
| ppo11 | ppo_seed11_no_noise_20260101_150453 | 3.2e-07 |
| ppo13 | ppo_seed13_no_noise_20260101_175542 | 3.2e-07 |
| ppo17 | ppo_seed17_no_noise_20260101_203448 | 4.6e-07 |
| ppo27 | ppo_seed27_no_noise_20260102_215842 | 5.8e-07 |
| ppo30 | ppo_seed30_no_noise_20260102_220217 | 4.0e-07 |
| ppo33 | ppo_seed33_no_noise_20260102_221810 | 4.1e-07 |
| ppo36 | ppo_seed19_no_noise_20260102_223225 | 3.1e-07 |
| ppo19(screened) | ppo_seed19_no_noise_20260101_180703 | 3.1e-07 |
| sac01 | sac_seed1_no_noise_20251226_104913 | 9.5e-06 |
| sac03 | sac_seed3_no_noise_20251223_201715 | 1.0e-06 |
| sac05 | sac_seed5_no_noise_20251226_105937 | 3.5e-06 |
| sac07 | sac_seed7_no_noise_20251226_180114 | 8.4e-06 |
| sac09 | sac_seed9_no_noise_20251226_204046 | 3.3e-06 |
| sac10 | sac_seed10_no_noise_20260101_141459 | 4.1e-06 |
| sac11 | sac_seed11_no_noise_20260101_154004 | 2.3e-05 |
| sac13 | sac_seed13_no_noise_20260101_174239 | 8.9e-06 |
| sac14 | sac_seed14_no_noise_20260101_174624 | 4.9e-06 |
| sac18 | sac_seed18_no_noise_20260101_144353 | 6.3e-06 |
| sac19 | sac_seed19_no_noise_20260101_175009 | 3.3e-06 |

All 21 pass (PPO ≤ 5.8e-7, SAC ≤ 2.3e-5).

**Seed-accounting finding.** The one seed-19 PPO checkpoint (ppo_claude_seed19.zip, saved 29 Dec 2025) reproduces both the screened run ppo19 (1 Jan 2026) and the retained run ppo36 (2 Jan 2026) to 3.1e-7. The two runs are two evaluations of the same policy, not two training runs. RESULTS_H6_seeds.md ("PPO seed 19 trained twice") must be corrected: 27 seeds, 28 logged evaluation runs; seed 19 was evaluated twice, the first evaluation screened and the second retained.

## Gate 2 — fresh nominal runs match the archived controls

One new 50-episode nominal run per agent (seed 9,100,500 + 1000·index). Criterion declared before inspection: for P, H_f, H_b, ΔH (Chao–Shen) and mean reward, the new calibration-window mean lies within 1 archived calibration SD of the archived mean.

| agent | reward/step archived | new | ΔP | ΔH_f | ΔH_b | ΔΔH | Δreward | SD ratio P | SD ratio reward | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| ppo02 | 6.351 | 6.357 | +0.08 | -0.06 | +0.06 | -0.36 | +0.08 | 1.10 | 0.98 | yes |
| ppo04 | 7.058 | 7.066 | -0.06 | +0.01 | +0.03 | -0.04 | +0.04 | 1.09 | 0.89 | yes |
| ppo08 | 6.072 | 5.931 | +0.38 | -0.37 | -0.46 | +0.35 | -1.51 | 1.49 | 4.56 | **no** |
| ppo11 | 6.652 | 6.639 | -0.16 | +0.27 | +0.40 | -0.47 | -0.18 | 1.07 | 1.08 | yes |
| ppo13 | 6.998 | 6.985 | +0.44 | -0.26 | -0.16 | -0.21 | -0.09 | 1.13 | 1.03 | yes |
| ppo17 | 5.420 | 5.416 | +0.01 | +0.04 | +0.02 | +0.07 | -0.08 | 0.92 | 1.18 | yes |
| ppo27 | 2.075 | 2.077 | -0.31 | +0.38 | +0.26 | +0.01 | -0.20 | 0.97 | 1.03 | yes |
| ppo30 | 2.099 | 2.053 | +0.14 | +0.03 | +0.29 | -0.69 | -0.09 | 0.75 | 1.14 | yes |
| ppo33 | 1.969 | 1.936 | -0.27 | +0.28 | +0.25 | +0.01 | +0.31 | 1.12 | 1.20 | yes |
| ppo36 | 4.129 | 4.130 | -0.14 | +0.08 | +0.16 | -0.22 | -0.05 | 0.92 | 0.91 | yes |
| sac01 | 12.228 | 12.212 | -0.32 | +0.36 | +0.20 | +0.23 | -0.12 | 1.07 | 0.97 | yes |
| sac03 | 15.890 | 15.890 | -0.03 | +0.02 | +0.00 | +0.03 | +0.01 | 1.00 | 0.99 | yes |
| sac05 | 13.213 | 13.193 | -0.15 | +0.04 | +0.04 | -0.03 | +0.01 | 1.03 | 1.00 | yes |
| sac07 | 13.349 | 13.354 | +0.09 | -0.11 | -0.16 | +0.15 | +0.00 | 0.96 | 1.00 | yes |
| sac09 | 13.520 | 13.458 | +0.27 | -0.10 | -0.34 | +0.39 | -0.26 | 1.35 | 1.49 | yes |
| sac10 | 13.314 | 13.272 | -0.35 | +0.31 | +0.29 | -0.13 | -0.06 | 0.84 | 0.94 | yes |
| sac11 | 7.712 | 7.709 | +0.02 | -0.04 | +0.01 | -0.09 | -0.02 | 0.88 | 0.99 | yes |
| sac13 | 11.636 | 11.602 | +0.17 | -0.21 | -0.10 | -0.22 | -0.02 | 1.02 | 0.96 | yes |
| sac14 | 11.976 | 11.997 | -0.00 | -0.16 | -0.02 | -0.17 | +0.17 | 1.10 | 1.00 | yes |
| sac18 | 15.873 | 15.866 | +0.13 | -0.17 | -0.21 | +0.22 | +0.01 | 0.93 | 0.98 | yes |
| sac19 | 11.582 | 11.552 | +0.26 | -0.33 | -0.32 | +0.25 | -0.09 | 1.10 | 1.01 | yes |

(Δ = (new mean − archived mean) / archived calibration SD.) 20/21 pass; reward per step agrees within 0.0–2.3%.

**ppo08 fails on reward (−1.51 SD; SD ratio 4.6).** The policy is reproduced exactly (gate 1); the new run contains three degraded episodes (returns 2,708, 3,563, 3,964 against an archived range 5,507–6,253). Four further 50-episode runs of ppo08: 4, 4, 1, 1 episodes below 5,500 — 13 of 250 episodes (5%) in all. The archived control had none of 50 (probability ≈ 0.95^50 ≈ 8% at that rate). The difference is ppo08's own nominal variability, not a replication failure; this is the variability the long runs are meant to measure.

## Batch log

- Batch 1 (sac01, ppo02, sac03, ppo04): finished 8 Nov 15:18 UTC; 4 × 1,000,000 steps; readings_<agent>.csv.gz written; summaries readings/summary_sac01_ppo02.csv, summary_sac03_ppo04.csv. Reward/step sac01 12.19, ppo02 6.31, sac03 15.89, ppo04 7.01; episodes below 90% of the agent's median return: 3, 10, 0, 16 of 1,000; undefined-P windows 0, 26, 0, 0.

## Decision record — false-alarm target (8 Nov, after the code check, before the final comparison)
The target first proposed — mean time to first false alarm ≥ 100 episodes, censored runs counted at 985 — was found
unsuitable during a code check of the evaluation script on the first 10 completed agents (no bootstrap, not reported):
a few runs without any alarm (each counted at 985 episodes) bring the mean near 100 while most runs alarm within
1–8 episodes, so the criterion selects very low thresholds. Replaced, before the full 21-agent comparison was run, by:
**at most 25% of nominal runs alarm within 100 episodes after calibration.** For each detector (three-of-four, P alone,
reward) and held-out agent: the lowest z at which at most 5 of the other 20 nominal runs alarm within 100 episodes;
frozen and applied to the held-out nominal run and to its archived control and fault trials. The latched alarm is
unchanged. Reported: achieved held-out alarm rate within 100 episodes and the full time-to-first-alarm curve.

## Batch log (completed)

- Batch 2 (sac05, ppo11, sac07, ppo13), 3 (sac09, ppo17, sac10, ppo27), 4 (sac11, ppo30, sac13, ppo33), 5 (sac14, sac18, ppo08, sac19, ppo36): all 10 segments each. Batches 2 and 5 were interrupted (a session interruption and a workspace restart) and resumed from the last saved segment; segments are seeded per segment, so resumed runs are identical to uninterrupted ones.

| agent | reward/step | median episode return | episodes < 90% of median (of 1,000) | undefined-P windows (of 19,991) |
|---|---|---|---|---|
| ppo02 | 6.311 | 6365 | 10 | 26 |
| ppo04 | 7.011 | 7068 | 16 | 0 |
| ppo08 | 5.900 | 6090 | 57 | 40 |
| ppo11 | 6.606 | 6646 | 9 | 16 |
| ppo13 | 6.962 | 6994 | 7 | 7 |
| ppo17 | 5.396 | 5434 | 9 | 0 |
| ppo27 | 2.075 | 2078 | 0 | 0 |
| ppo30 | 2.079 | 2104 | 10 | 0 |
| ppo33 | 1.947 | 1976 | 18 | 57 |
| ppo36 | 4.129 | 4130 | 0 | 0 |
| sac01 | 12.193 | 12239 | 3 | 0 |
| sac03 | 15.892 | 15896 | 0 | 0 |
| sac05 | 13.206 | 13227 | 1 | 0 |
| sac07 | 13.360 | 13373 | 0 | 0 |
| sac09 | 13.416 | 13530 | 18 | 17 |
| sac10 | 13.277 | 13280 | 1 | 3 |
| sac11 | 7.715 | 7719 | 0 | 0 |
| sac13 | 11.631 | 11637 | 0 | 0 |
| sac14 | 11.977 | 11995 | 3 | 0 |
| sac18 | 15.836 | 15885 | 5 | 0 |
| sac19 | 11.561 | 11559 | 0 | 0 |

## Results at the long-run target (held out, 21 agents)

| method | frozen z (median; range) | held-out runs alarming within 100 episodes | median time to first false alarm (episodes) | runs without any false alarm in 985 episodes | archived fault detection (168 trials) | archived control alarms |
|---|---|---|---|---|---|---|
| three-of-four | 4.80 (4.40-4.80) | 6/21 | 395 | 7/21 | **46.4%** | 0/21 |
| P alone | 5.30 (4.15-5.30) | 6/21 | 641 | 9/21 | **43.5%** | 0/21 |
| reward | 6.10 (3.95-6.10) | 6/21 | 318 | 8/21 | **31.0%** | 2/21 |

Median times count runs without an alarm at 985 episodes.

Agent-level bootstrap (1,000 resamples, threshold selection repeated in each): three-of-four [14.3, 61.9]; P alone [14.9, 68.5]; reward [4.8, 45.8]; three-of-four − reward **[−3.6, +35.1]**; P alone − reward **[−7.1, +36.3]**.

### Time to first false alarm (share of the 21 held-out long runs that have alarmed, %)

| episodes after calibration | three-of-four | P alone | reward |
|---|---|---|---|
| 1 | 0.0 | 0.0 | 4.8 |
| 5 | 0.0 | 4.8 | 4.8 |
| 10 | 0.0 | 9.5 | 4.8 |
| 25 | 4.8 | 14.3 | 4.8 |
| 50 | 23.8 | 28.6 | 28.6 |
| 75 | 23.8 | 28.6 | 28.6 |
| 100 | 28.6 | 28.6 | 28.6 |
| 150 | 33.3 | 33.3 | 42.9 |
| 200 | 38.1 | 38.1 | 47.6 |
| 300 | 42.9 | 38.1 | 47.6 |
| 400 | 52.4 | 47.6 | 52.4 |
| 500 | 52.4 | 47.6 | 52.4 |
| 600 | 57.1 | 47.6 | 52.4 |
| 700 | 61.9 | 52.4 | 57.1 |
| 800 | 66.7 | 52.4 | 57.1 |
| 900 | 66.7 | 52.4 | 57.1 |
| 985 | 66.7 | 57.1 | 61.9 |

### Detection by fault condition (archived trials, 21 per condition, %)

| condition | three-of-four | P alone | reward |
|---|---|---|---|
| act_noise_01% | 9.5 | 28.6 | 23.8 |
| act_noise_03% | 57.1 | 47.6 | 19.0 |
| act_noise_04% | 81.0 | 52.4 | 33.3 |
| force_torso_10N | 14.3 | 23.8 | 23.8 |
| force_torso_5N | 14.3 | 23.8 | 14.3 |
| gravity_110% | 33.3 | 33.3 | 33.3 |
| obs_noise_01% | 71.4 | 52.4 | 33.3 |
| obs_noise_03% | 90.5 | 85.7 | 66.7 |

### Per agent (frozen z; time to first false alarm in the long run, episodes; * = no alarm in 985 episodes)

| agent | three-of-four | P alone | reward |
|---|---|---|---|
| ppo02 | z 4.40 · 43 | z 4.15 · 8 | z 3.95 · 43 |
| ppo04 | z 4.40 · 32 | z 4.15 · 32 | z 6.10 · 106 |
| ppo08 | z 4.80 · 562 | z 4.15 · 4 | z 3.95 · 1 |
| ppo11 | z 4.40 · 85 | z 4.15 · 26 | z 6.10 · 141 |
| ppo13 | z 4.40 · 31 | z 4.15 · 15 | z 3.95 · 31 |
| ppo17 | z 4.80 · 781 | z 5.30 · 985* | z 3.95 · 45 |
| ppo27 | z 4.80 · 985* | z 5.30 · 985* | z 6.10 · 985* |
| ppo30 | z 4.80 · 318 | z 5.30 · 318 | z 6.10 · 318 |
| ppo33 | z 4.40 · 28 | z 4.15 · 28 | z 3.95 · 28 |
| ppo36 | z 4.80 · 985* | z 5.30 · 985* | z 6.10 · 985* |
| sac01 | z 4.80 · 279 | z 5.30 · 971 | z 6.10 · 970 |
| sac03 | z 4.80 · 985* | z 5.30 · 985* | z 6.10 · 985* |
| sac05 | z 4.80 · 985* | z 5.30 · 985* | z 6.10 · 985* |
| sac07 | z 4.80 · 985* | z 5.30 · 985* | z 6.10 · 985* |
| sac09 | z 4.80 · 111 | z 5.30 · 111 | z 6.10 · 110 |
| sac10 | z 4.80 · 641 | z 5.30 · 641 | z 6.10 · 640 |
| sac11 | z 4.80 · 395 | z 5.30 · 395 | z 6.10 · 985* |
| sac13 | z 4.80 · 985* | z 5.30 · 985* | z 6.10 · 985* |
| sac14 | z 4.80 · 174 | z 5.30 · 174 | z 6.10 · 174 |
| sac18 | z 4.80 · 985* | z 5.30 · 985* | z 3.95 · 40 |
| sac19 | z 4.40 · 18 | z 5.30 · 985* | z 6.10 · 985* |

## Reading

1. **Calibrated to the episode scale, the telemetry still detects more archived faults than reward**: three-of-four 46.4% and P alone 43.5% against 31.0%, at the same held-out long-run false-alarm rate (6/21 runs alarming within 100 episodes for each method, target 5/20 on the inner runs). The ordering of the short-exposure analysis is preserved; the intervals for the differences now include zero ([−3.6, +35.1], [−7.1, +36.3]).
2. **The long-run target costs every method detection.** Thresholds rise from about 3.3–3.5 (6/21 controls over 34.5 episodes) to about 4.4–5.3 (telemetry) and 4–6 (reward); detection falls from 72.6% / 74.4% / 50.6% to 46.4% / 43.5% / 31.0%. The archived fault trials have only 34.5 evaluation episodes, so detection is measured over a short window at a threshold set for a 100-episode horizon.
3. **Where the telemetry keeps its margin** (three-of-four vs reward): action noise 3–4% (57–81% vs 19–33%) and observation noise (71–91% vs 33–67%). Force faults are hard for all three at this target (14–24%); gravity is equal (33%).
4. **False alarms accumulate with exposure for every method**: about a third of the runs alarm by 150 episodes, about half by 400–500 episodes, and 57–67% by the end of 985 episodes. No method is a validated long-horizon alarm at these settings; the telemetry and reward have comparable long-run false-alarm behaviour, which makes the detection comparison above a like-for-like one.
5. Archived controls (34.5 episodes) at the frozen thresholds: 0/21 alarms for both telemetry rules, 2/21 for reward.

## Restricted mean time to first false alarm (985-episode horizon, held out)

Runs without an alarm counted at 985 episodes (restricted mean; a lower bound on the unrestricted mean). Three-of-four 494.9 episodes (median 394.9; 7/21 runs without an alarm); P alone 551.8 (median 641.0; 9/21); reward 501.4 (median 318.2; 8/21). Target: at most 25% of runs alarming within 100 episodes; achieved held out 6/21 = 28.6% for each method (inner selection at 5/20 = 25%).
