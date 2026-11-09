# R1 — fixed three-of-four rule vs nested-selected k-of-four (review point 1)
8 Nov 2026 · script halfcheetah/R1_fixed_rule.py · outputs out/R1_point.txt, out/R1_boot_*.csv, out/R1_bootsum.txt

The 76.8% result of PRIMARY selects k as well as z; in all 21 folds it selects one-of-four (z 4.1 in 15 folds, 4.05 in 6).
It is therefore named **nested-selected k-of-four telemetry**. The fixed rules are reported separately.
Same trial scores (out/PRIMARY_trial_scores.csv) and nested protocol; 1,000 bootstrap resamples with
selection inside each.

## Fixed rules at ±3σ, no selection (in-sample, descriptive)
| rule | detection | control alarms |
|---|---|---|
| three-of-four, Chao–Shen | 86.9% | 11/21 |
| three-of-four, plug-in (submission) | 69.0% | 5/21 |
| four-of-four, Chao–Shen | 70.2% | 7/21 |
| P alone, Chao–Shen | 85.1% | 10/21 |
| reward | 44.0% | 5/21 |

With the corrected estimator the ±3σ threshold no longer holds 5/21 controls: the threshold has to be set
from healthy data, which the nested protocol does.

## Nested held-out, ALPHA 0.25 (z selected; k fixed unless stated)
| rule | detection [95%] | control alarms [95%] | minus reward, detection [95%] | minus reward, false alarms [95%] | selected z |
|---|---|---|---|---|---|
| **P alone, Chao–Shen** | **74.4%** [60.7, 86.9] | 6/21 [18.9, 33.3] | **[+4.8, +39.3]** | [−9.6, +9.5] | 3.4 (15), 3.25 (6) |
| three-of-four, Chao–Shen | 72.6% [62.5, 82.1] | 6/21 [14.3, 33.3] | [+5.4, +34.5] | [−14.3, +9.5] | 3.45 (15), 3.35 (6) |
| three-of-four, plug-in | 70.8% [55.4, 80.4] | 6/21 [14.3, 33.3] | [−3.0, +38.1] | [−14.3, +9.5] | 3.0 (15), 2.9 (6) |
| four-of-four, Chao–Shen | 64.9% [52.4, 77.4] | 5/21 [14.3, 33.3] | [−1.2, +28.0] | [−14.3, +9.5] | 3.15 (21) |
| k-of-four, Chao–Shen (k and z selected) | 76.8% [64.3, 86.9] | 6/21 [23.8, 47.6] | [+9.5, +39.3] | [−9.5, +23.8] | one-of-four 4.1 (15) / 4.05 (6) |
| reward | 50.6% [36.3, 66.7] | 6/21 [23.8, 33.3] | — | — | 2.9 (15), 2.5 (6) |

## Nested held-out, ALPHA 0.10
P alone CS 64.3% (3/21) · three-of-four CS 67.3% (3/21) · three-of-four plug-in 59.5% (2/21) ·
four-of-four CS 56.5% (3/21) · k-of-four CS 65.5% (5/21) · reward 35.1% (3/21).

## Reading
- The fixed three-of-four rule with a held-out threshold keeps the advantage: 72.6% vs reward 50.6%,
  interval [+5.4, +34.5], with false-alarm differences tight around zero ([−14.3, +9.5]).
- P alone gives the cleanest result: 74.4%, interval [+4.8, +39.3], false-alarm difference [−9.6, +9.5];
  no rule structure is selected, and it is the paper's own metric.
- Selecting k adds about 2–4 points at the cost of a wider false-alarm interval; it is not needed for the
  claim.
