# E4 — combined sequential monitor and trial-level overlap (HalfCheetah, 21 agents)

8 Nov 2026 · code halfcheetah/E4_combined.py (uses E2_cusum_eval.py) · outputs out/E4_combined_summary.csv, E4_overlap_trials.csv, E4_overlap_summary.csv, E4_overlap_by_condition.csv, E4_boot.csv

Design (fixed in the script header): alarm when either component CUSUM (reward; telemetry = three-of-four or P) crosses its own level; components put on a common scale on the inner agents only (short exposure: 75th percentile of inner control scores; long run: the h at which the component alone meets the long-run target); one threshold on the combined score, selected exactly as for single readings; k shared and selected. Overlap: single-reading held-out operating points of E2.

## Combined monitor (held out)

| analysis | monitor | detection | 95% interval | archived control alarms | held-out runs alarming within 100 ep |
|---|---|---|---|---|---|
| short exposure | reward OR three-of-four | **82.7%** | [72.6, 92.3] | 10/21 |  |
| long-run target | reward OR three-of-four | **78.6%** | [58.3, 89.9] | 2/21 | 8/21 |
| short exposure | reward OR P | **82.7%** | [74.4, 92.9] | 6/21 |  |
| long-run target | reward OR P | **80.4%** | [53.6, 90.5] | 3/21 | 8/21 |

Single readings for reference (E2, CUSUM): short — three-of-four 79.8%, P 74.4%, reward 78.6%; long-run target — 68.5%, 63.1%, 70.8% (each 6/21 held-out runs alarming within 100 episodes).

Differences (1,000 agent-level resamples, selection inside each):

| analysis | combined − reward | combined − telemetry alone |
|---|---|---|
| short, reward OR three-of-four | [−1.8, +13.7] | [−2.4, +16.1] |
| short, reward OR P | [−1.2, +16.7] | [−1.8, +18.5] |
| long, reward OR three-of-four | **[+3.6, +26.8]** | **[+1.8, +17.9]** |
| long, reward OR P | **[+4.2, +26.2]** | **[+1.2, +25.0]** |

## Trial-level overlap at the single-reading operating points (168 fault trials)

| analysis | telemetry | both | reward only | telemetry only | neither |
|---|---|---|---|---|---|
| long | P | 87 | 32 | 19 | 30 |
| long | three-of-four | 97 | 22 | 18 | 31 |
| short | P | 107 | 25 | 18 | 18 |
| short | three-of-four | 117 | 15 | 17 | 19 |

### By condition (three-of-four vs reward)

| analysis | condition | both | reward only | telemetry only | neither |
|---|---|---|---|---|---|
| long | act_noise_01% | 6 | 2 | 4 | 9 |
| long | act_noise_03% | 14 | 1 | 3 | 3 |
| long | act_noise_04% | 17 | 3 | 1 | 0 |
| long | force_torso_10N | 9 | 5 | 1 | 6 |
| long | force_torso_5N | 5 | 7 | 0 | 9 |
| long | gravity_110% | 17 | 3 | 1 | 0 |
| long | obs_noise_01% | 11 | 1 | 6 | 3 |
| long | obs_noise_03% | 18 | 0 | 2 | 1 |
| short | act_noise_01% | 8 | 2 | 4 | 7 |
| short | act_noise_03% | 15 | 3 | 2 | 1 |
| short | act_noise_04% | 20 | 0 | 1 | 0 |
| short | force_torso_10N | 10 | 3 | 3 | 5 |
| short | force_torso_5N | 10 | 7 | 2 | 2 |
| short | gravity_110% | 21 | 0 | 0 | 0 |
| short | obs_noise_01% | 13 | 0 | 5 | 3 |
| short | obs_noise_03% | 20 | 0 | 0 | 1 |

## Reading

1. **The two readings are complementary.** At the single-reading operating points 17–19 fault trials are caught by the telemetry only and 15–32 by reward only. Telemetry-only catches concentrate on observation noise 1% and action noise 1%; reward-only on force 5 N (and force 10 N, gravity at the long-run target).
2. **Combining them gains where it matters most — at the long-run target**: 78.6–80.4% against 70.8% (reward) and 63.1–68.5% (telemetry), with intervals excluding zero against both components. The gain comes with more held-out false alarms (8/21 runs alarming within 100 episodes against 6/21 for the single readings; the inner target was 5/20), so part of it is bought with exposure.
3. In the short exposure the combination reaches 82.7%; its intervals include zero, and reward OR three-of-four exceeds its control budget held out (10/21); reward OR P stays within it (6/21).
4. This is the outcome/process distinction in operation: the outcome reading and the loop reading respond to different parts of the fault space, and together cover more of it than either.
