# R7 — normalisation ablation (submission Table 3) with coverage-adjusted estimates and held-out intervals
8 Nov 2026 · script halfcheetah/R7_ablation.py · outputs out/R7_trial_scores.csv, out/R7_point.csv/.txt,
out/R7_boot_*.csv, out/R7_bootsum.txt

Single statistics, latched, calibration 0–290, evaluation ≥300; zero-budget windows skipped for every
statistic (one convention throughout). In-sample: threshold midway between the 5th and 6th largest
control scores (as in the submission). Nested: H5 protocol, ALPHA 0.25, 1,000 bootstrap resamples with
selection inside each.

## ⚠ Correction to the submitted Table 3
The submission's script (v31_p04_matched_falsealarm.py) took the maximum over evaluation windows with
numpy, which returns NaN when any window is undefined. Zero-budget windows (the agent holding one state for
a whole window, after a fault) are NaN for I/min{H(S,A),H(S')} and I/H(S,A,S') — 17 and 15 perturbation
trials respectively — so those trials were scored as missed. In the stored metrics P is 0.0 rather than
undefined in the same windows, so P counted them as detected. Rerunning the submission script reproduces
its numbers exactly (41.1%, 52.4%); with one consistent convention the plug-in values are 50.6% and 61.3%.

## Results
| statistic | plug-in, in-sample (5/21) | plug-in, nested | **Chao–Shen, in-sample (5/21)** | **Chao–Shen, nested [95%]** | Chao–Shen: P − this, nested [95%] |
|---|---|---|---|---|---|
| I(S,A;S') unnormalised | 50.0 | 51.2 (6/21) | 50.0 | 50.6 (6/21) [38.7, 60.7] | **[+5.4, +39.3]** |
| I / min{H(S,A),H(S')} | 50.6 (submitted 41.1) | 55.4 (6/21) | 70.8 | 76.2 (6/21) [56.0, 86.9] | [−7.7, +10.7] |
| I / H(S,A,S') | 61.3 (submitted 52.4) | 61.3 (5/21) | 71.4 | 73.8 (6/21) [56.5, 83.9] | [−6.0, +10.1] |
| **P = I / C** | 58.9 | 59.5 (6/21) | 73.8 | **74.4 (6/21)** [59.5, 86.9] | — |
| H_f | 60.7 | 61.9 (6/21) | 74.4 | 75.6 (6/21) [61.9, 83.9] | [−14.3, +13.1] |
| H_b | 60.7 | 61.9 (6/21) | 74.4 | 74.4 (5/21) [65.5, 80.4] | [−14.9, +14.3] |
| ΔH | 50.6 | 53.6 (6/21) | 57.1 | 58.9 (6/21) [41.1, 76.8] | [−7.7, +34.5] |
| windowed reward | 44.0 | 50.6 (6/21) | 44.0 | 50.6 (6/21) [35.7, 66.7] | **[+3.0, +39.9]** |

In-sample thresholds (Chao–Shen): 3.25, 3.59, 3.29, 3.40, 3.74, 3.69, 3.89, reward 2.98.
Plug-in nested, P − others: all intervals include zero (raw MI [−11.9, +25.6]; min-entropy [−18.5, +19.1];
joint [−19.7, +13.7]; reward [−16.7, +33.3]).

## Reading
- **What the evidence supports:** normalising the realised mutual information is what matters — held-out,
  coverage-adjusted, P detects 74.4% against 50.6% for the unnormalised I(S,A;S') (interval [+5.4, +39.3])
  and against reward (interval [+3.0, +39.9]).
- **What it does not support:** a detection advantage of the budget denominator over the other
  normalisations or over H_f, H_b. With coverage-adjusted estimates they are equivalent within the
  intervals (74–76%). The submission's "9 points over raw MI and 18 over min-entropy" must be revised:
  the min-entropy gap came from plug-in saturation and the NaN handling above.
- The case for P as the reading rests on the framework rather than on this ablation: it is the share of
  the interaction budget, bounded by the theory, decomposable exactly (H8), and the best-conditioned
  plug-in quantity in the sparse regime (G1: ~2% bias where the components carry 50–80%).

## R7b — detection–false-alarm curves for every row (8 Nov)

Script R7b_curves.py, output out/R7b_curves.csv. Nested leave-one-agent-out (H5 protocol) at ALPHA 0.05–0.50; held-out detection (%) of 168 fault trials, held-out control alarms of 21 in the second table.

| statistic | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| I(S,A;S') unnormalised [cs] | 27.4 | 31.5 | 42.9 | 49.4 | 50.6 | 57.1 | 59.5 | 65.5 |
| I / min{H(S,A),H(S')} [cs] | 57.7 | 61.3 | 61.3 | 67.9 | 76.2 | 82.7 | 84.5 | 89.3 |
| I / H(S,A,S') [cs] | 47.6 | 58.3 | 64.9 | 69.0 | 73.8 | 79.8 | 80.4 | 84.5 |
| P = I / C [cs] | 63.1 | 64.3 | 66.1 | 71.4 | 74.4 | 79.8 | 82.1 | 88.7 |
| H_f [cs] | 62.5 | 65.5 | 72.0 | 72.6 | 75.6 | 75.6 | 81.5 | 83.9 |
| H_b [cs] | 66.7 | 70.8 | 72.6 | 73.2 | 74.4 | 74.4 | 76.2 | 79.8 |
| Delta H [cs] | 43.5 | 45.8 | 47.6 | 53.0 | 58.9 | 67.9 | 75.0 | 78.6 |
| I(S,A;S') unnormalised [pi] | 32.7 | 42.9 | 46.4 | 48.2 | 51.2 | 53.0 | 57.7 | 67.3 |
| I / min{H(S,A),H(S')} [pi] | 43.5 | 43.5 | 45.2 | 50.6 | 55.4 | 71.4 | 76.8 | 82.1 |
| I / H(S,A,S') [pi] | 53.0 | 57.7 | 60.7 | 60.7 | 61.3 | 61.9 | 64.9 | 72.0 |
| P = I / C [pi] | 38.1 | 38.7 | 45.8 | 58.9 | 59.5 | 64.3 | 74.4 | 82.7 |
| H_f [pi] | 41.7 | 48.8 | 51.8 | 58.3 | 61.9 | 70.2 | 78.0 | 82.1 |
| H_b [pi] | 46.4 | 53.0 | 54.8 | 59.5 | 61.9 | 63.7 | 73.8 | 82.1 |
| Delta H [pi] | 42.9 | 43.5 | 45.2 | 49.4 | 53.6 | 57.1 | 69.0 | 77.4 |
| windowed reward | 34.5 | 35.1 | 39.3 | 44.6 | 50.6 | 61.3 | 69.6 | 76.8 |

| statistic (control alarms of 21) | 0.05 | 0.1 | 0.15 | 0.2 | 0.25 | 0.3 | 0.4 | 0.5 |
|---|---|---|---|---|---|---|---|---|
| I(S,A;S') unnormalised [cs] | 1 | 3 | 4 | 5 | 6 | 7 | 9 | 10 |
| I / min{H(S,A),H(S')} [cs] | 2 | 2 | 4 | 5 | 6 | 7 | 8 | 10 |
| I / H(S,A,S') [cs] | 2 | 3 | 4 | 5 | 6 | 6 | 9 | 11 |
| P = I / C [cs] | 1 | 3 | 4 | 5 | 6 | 7 | 9 | 11 |
| H_f [cs] | 2 | 3 | 3 | 5 | 6 | 7 | 9 | 11 |
| H_b [cs] | 2 | 3 | 3 | 5 | 5 | 5 | 9 | 11 |
| Delta H [cs] | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 |
| I(S,A;S') unnormalised [pi] | 2 | 3 | 4 | 5 | 6 | 6 | 8 | 11 |
| I / min{H(S,A),H(S')} [pi] | 1 | 1 | 4 | 5 | 6 | 6 | 9 | 11 |
| I / H(S,A,S') [pi] | 2 | 3 | 3 | 5 | 5 | 7 | 8 | 11 |
| P = I / C [pi] | 1 | 3 | 4 | 5 | 6 | 7 | 9 | 9 |
| H_f [pi] | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 11 |
| H_b [pi] | 2 | 2 | 4 | 5 | 6 | 6 | 9 | 11 |
| Delta H [pi] | 2 | 2 | 4 | 5 | 6 | 7 | 9 | 10 |
| windowed reward | 2 | 3 | 4 | 5 | 6 | 7 | 9 | 11 |

Reading: with Chao–Shen estimates the normalised statistics are close across operating points from ALPHA 0.20 upward (P 71–89%, min-entropy 68–89%, joint 69–85%). At the strictest budgets (ALPHA 0.05–0.10, 1–3 control alarms) P, H_f and H_b keep 63–71% while the joint normalisation drops to 48–58%, the min-entropy normalisation to 58–61% and unnormalised I to 27–32%. Reward: 35–77% over the same range.
