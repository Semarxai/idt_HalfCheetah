# H10a — time to first alarm on archived exposure (nested operating points)
8 Nov 2026 · script halfcheetah/H10a_time_to_alarm.py · outputs out/H10a_first_alarm.csv, out/H10a_output.txt,
out/H10a_paired.txt

For each held-out agent the operating point (statistic set, k, z) is selected on the other 20 agents
exactly as in H5 (ALPHA 0.25); the held-out agent's windowed |z| series give the first alarm window
under the latched rule. Evaluation runs from window 300 to 990: 691 windows = 34.5 episodes
(1 episode = 20 windows). Onset at step 15,000 (window 291–300).

## Controls — cumulative share with a false alarm, by evaluation episode

| method | ep 1 | ep 5 | ep 10 | ep 20 | ep 34.5 | alarmed | median first alarm |
|---|---|---|---|---|---|---|---|
| consensus Chao–Shen | 0.0% | 0.0% | 19.0% | 19.0% | 28.6% | 6/21 | 6.6 ep |
| P Chao–Shen | 0.0% | 4.8% | 23.8% | 23.8% | 28.6% | 6/21 | 6.7 ep |
| consensus plug-in | 0.0% | 14.3% | 19.0% | 28.6% | 38.1% | 8/21 | 8.3 ep |
| reward | 9.5% | 14.3% | 23.8% | 28.6% | 28.6% | 6/21 | 5.9 ep |
| H(S) | 0.0% | 4.8% | 14.3% | 19.0% | 28.6% | 6/21 | 14.3 ep |
| MMD | 0.0% | 4.8% | 4.8% | 19.0% | 28.6% | 6/21 | 14.8 ep |

False alarms accumulate with exposure for every method; none of the consensus Chao–Shen false alarms
occurs within the first 5 evaluation episodes.

## Perturbation trials — detection and delay from the start of evaluation

| method | detected | median delay | IQR | by ep 1 | by ep 5 |
|---|---|---|---|---|---|
| consensus Chao–Shen | 76.8% | 1.65 ep | [0.05, 8.40] | 32.7% | 48.8% |
| P Chao–Shen | 74.4% | 2.05 ep | [0.15, 7.95] | 30.4% | 45.2% |
| consensus plug-in | 72.0% | 5.60 ep | [1.35, 12.60] | 17.3% | 33.9% |
| reward | 50.6% | 5.15 ep | [1.60, 12.15] | 11.3% | 24.4% |
| H(S) | 50.0% | 5.70 ep | [0.74, 15.93] | 13.7% | 22.0% |
| MMD | 52.4% | 7.00 ep | [0.47, 17.57] | 14.3% | 23.8% |

Median delay by condition (episodes, detected trials only):

| condition | consensus CS | P CS | consensus plug-in | reward | H(S) | MMD |
|---|---|---|---|---|---|---|
| action noise 1% | 3.58 | 1.65 | 9.30 | 10.38 | 10.92 | 10.22 |
| action noise 3% | 3.10 | 5.05 | 3.15 | 4.70 | 2.60 | 3.90 |
| action noise 4% | 0.35 | 1.90 | 7.52 | 5.15 | 8.25 | 15.75 |
| force 10 N | 7.85 | 6.65 | 7.90 | 9.95 | 14.70 | 19.00 |
| force 5 N | 8.65 | 8.25 | 7.10 | 5.35 | 6.80 | 11.35 |
| gravity 110% | 1.00 | 1.10 | 4.50 | 0.60 | 4.50 | 0.00 |
| obs noise 1% | 0.82 | 3.40 | 7.65 | 19.80 | 16.18 | 14.85 |
| obs noise 3% | 0.00 | 0.15 | 2.38 | 1.82 | 0.00 | 0.00 |

## Paired lead on trials both methods detect (pooled trial-level, descriptive; trials cluster within agents)
**Superseded for inference by RESULTS_R6_lead_clustered.md (agent-clustered); the p-values below are withdrawn.**

| comparison | both detect | first earlier | later | tied | median lead | sign test p |
|---|---|---|---|---|---|---|
| consensus CS vs reward | 72 | 40 | 22 | 10 | 1.95 ep | 0.030 |
| consensus CS vs MMD | 72 | 39 | 20 | 13 | 2.12 ep | 0.018 |
| consensus CS vs H(S) | 75 | 40 | 19 | 16 | 0.10 ep | 0.009 |
| P CS vs reward | 72 | 41 | 26 | 5 | 1.88 ep | 0.086 |
| P CS vs MMD | 73 | 39 | 27 | 7 | 2.10 ep | 0.175 |
| consensus plug-in vs reward | 76 | 28 | 40 | 8 | −0.07 ep | 0.182 |

Reading: with the plug-in estimator the monitor has no lead over reward (as reported in the
submission). With the coverage-adjusted estimator it detects more trials and earlier: a third of
perturbation trials within one evaluation episode against 11% for reward, and a median lead of about
two episodes on the trials both detect.
