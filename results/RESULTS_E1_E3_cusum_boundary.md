# E1–E3 — reward likelihood detector (CUSUM) and episode-boundary check (HalfCheetah, 21 agents)

8 Nov 2026 · code halfcheetah/E1_episode_readings.py (device), E1_long.py, E2_cusum_eval.py, E3_boundary.py · outputs halfcheetah/out/E1/ (per-trial npz), out/E1_long/, out/E2_*.csv, out/E3_*.csv

## E3 — episode boundary

In the archived evaluation loop (DummyVecEnv auto-reset, then an explicit reset), the logged S′ of an episode's last step is the auto-reset observation, not the successor state, and S of the next row comes from a second reset. Verified in the logs: last step t = 999 in all 50 episodes of every trial; s_next(999) ≠ s(1000). One triplet in 1,000 is therefore a reset triplet; 49.5% of the 500-step windows contain one. The manuscript sentence "Triplets never cross resets" (v34, l. 318) is inaccurate and is corrected. The H10b long runs replicate the archived loop, including this triplet.

Sensitivity: the 500-step windows recomputed with the boundary triplet removed (Chao–Shen), nested protocol as PRIMARY (ALPHA 0.25):

| method | archived windows (as published) | boundary triplet removed |
|---|---|---|
| three-of-four (fixed k, z selected) | 72.6% (6/21) | 73.2% (6/21) |
| P alone | 74.4% (6/21) | 74.4% (6/21) |
| k-of-four (k and z selected) | 76.8% (6/21) | 78.6% (5/21) |
| reward | 50.6% (6/21) | 50.6% (6/21) |

The left column reproduces the published primary numbers exactly (an independent recomputation of the pipeline). Per-trial score changes: P median −0.008 (IQR −0.039 to +0.009), Spearman 0.989 with the original; three-of-four third score median −0.001 (IQR −0.019 to +0.014), Spearman 0.998. **No material effect.**

## E1–E2 — CUSUM on per-episode readings

Design fixed before the run (script header): per-episode readings — reward (episode mean); P, H_f, H_b, ΔH (Chao–Shen) on the 999 within-episode triplets (the boundary triplet excluded); standardised with episodes 0–14; two-sided CUSUM from episode 15, latched; k ∈ {0, 0.25, 0.5, 0.75, 1, 1.5, 2}, h 0.25–300. Three-of-four: three of the four statistics have crossed h.
- Short exposure: (k, h) selected on the other 20 agents (max inner detection, inner control alarms ≤ 0.25), as for z.
- Long-run target: H10b nominal runs (episodes 15–999); for each k, the lowest h with ≤ 25% of the other agents' runs alarming within 100 episodes; k chosen by inner archived detection.

| analysis | method | detection (168 trials) | 95% interval | archived control alarms | selections | held-out runs alarming within 100 ep | runs without alarm (985 ep) | median first alarm (ep) |
|---|---|---|---|---|---|---|---|---|
| short | reward | **78.6%** | [66.1, 88.7] | 5/21 | {'k0.0/h16.25': 21} |  |  |  |
| short | P | **74.4%** | [63.7, 89.3] | 8/21 | {'k0.5/h5.0': 11, 'k0.75/h2.5': 6, 'k1.5/h1.0': 2} |  |  |  |
| short | three-of-four | **79.8%** | [66.1, 88.7] | 6/21 | {'k0.75/h2.5': 15, 'k0.5/h3.75': 6} |  |  |  |
| long | reward | **70.8%** | [39.3, 82.7] | 3/21 | {'k0.5/h8.5': 15, 'k0.5/h8.0': 6} | 6/21 | 4/21 | 174 |
| long | P | **63.1%** | [44.6, 78.6] | 1/21 | {'k0.25/h13.25': 20, 'k0.75/h5.0': 1} | 6/21 | 3/21 | 191 |
| long | three-of-four | **68.5%** | [53.0, 81.5] | 0/21 | {'k0.25/h12.0': 15, 'k0.25/h11.75': 6} | 6/21 | 3/21 | 174 |

Differences (agent-level bootstrap, 1,000 resamples, selection inside): short exposure three-of-four − reward [−13.7, +11.3], P − reward [−13.7, +11.9]; long-run target three-of-four − reward [−11.3, +19.6], P − reward [−15.5, +19.0].

Short-exposure P CUSUM exceeds its control budget held out (8/21); the others are within it.

### Comparison with the latched window detector (same 168 trials)

| detector | three-of-four | P alone | reward |
|---|---|---|---|
| latched threshold on windows, short exposure (PRIMARY/R1) | 72.6% | 74.4% | 50.6% |
| CUSUM per episode, short exposure | 79.8% | 74.4% | 78.6% |
| latched threshold on windows, long-run target (H10b) | 46.4% | 43.5% | 31.0% |
| CUSUM per episode, long-run target | 68.5% | 63.1% | 70.8% |

### By fault condition (%, 21 trials each)

| condition | short: three-of-four | short: P | short: reward | long: three-of-four | long: P | long: reward |
|---|---|---|---|---|---|---|
| act_noise_01% | 57.1 | 71.4 | 47.6 | 47.6 | 38.1 | 38.1 |
| act_noise_03% | 81.0 | 85.7 | 85.7 | 81.0 | 81.0 | 71.4 |
| act_noise_04% | 100.0 | 90.5 | 95.2 | 85.7 | 85.7 | 95.2 |
| force_torso_10N | 61.9 | 47.6 | 61.9 | 47.6 | 42.9 | 66.7 |
| force_torso_5N | 57.1 | 33.3 | 81.0 | 23.8 | 19.0 | 57.1 |
| gravity_110% | 100.0 | 90.5 | 100.0 | 85.7 | 76.2 | 95.2 |
| obs_noise_01% | 85.7 | 81.0 | 61.9 | 81.0 | 66.7 | 57.1 |
| obs_noise_03% | 95.2 | 95.2 | 95.2 | 95.2 | 95.2 | 85.7 |

### Long-run CUSUM: share of held-out nominal runs that have alarmed (%)

| episodes | three-of-four | P | reward |
|---|---|---|---|
| 1 | 0.0 | 0.0 | 0.0 |
| 5 | 0.0 | 0.0 | 0.0 |
| 10 | 0.0 | 0.0 | 0.0 |
| 25 | 0.0 | 4.8 | 4.8 |
| 50 | 14.3 | 19.0 | 23.8 |
| 100 | 28.6 | 28.6 | 28.6 |
| 150 | 47.6 | 38.1 | 47.6 |
| 200 | 61.9 | 52.4 | 52.4 |
| 300 | 66.7 | 52.4 | 57.1 |
| 500 | 76.2 | 66.7 | 61.9 |
| 700 | 76.2 | 71.4 | 71.4 |
| 985 | 85.7 | 85.7 | 81.0 |

## Reading

1. **The detector matters more than the reading.** Accumulating evidence per episode (CUSUM) raises every reading, reward most: reward 50.6% → 78.6% (short) and 31.0% → 70.8% (long-run target). With CUSUM, telemetry and reward detect the same share of faults; all differences include zero.
2. **The detection-superiority comparison of the latched window detector does not carry over to sequential detection.** The claim that the telemetry detects more faults than reward is therefore detector-specific and is not made in general.
3. **The two readings catch different faults, as the outcome/process distinction predicts.** Reward CUSUM is stronger on the faults that act on the body's motion — force 5 N (81% vs 57% short; 57% vs 24% long) and, at the long-run target, force 10 N (67% vs 48%) and gravity (95% vs 86%). The telemetry is stronger on faults that perturb the agent's sensing or actuation while the outcome is preserved — observation noise 1% (86% vs 62% short; 81% vs 57% long) and action noise 1% (57–71% vs 48% short).
4. **Without reward, the telemetry alone reaches the same level** (three-of-four 79.8% short, 68.5% at the long-run target) — the reward-free reading is comparably sensitive to a well-designed reward monitor (differences include zero), and complementary in which faults it catches.
5. Next (proposed): the combined sequential monitor (reward CUSUM OR telemetry CUSUM) under the same budgets, and the paired trial-level overlap (faults caught by one reading only).
