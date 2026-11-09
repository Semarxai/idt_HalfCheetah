# G4d — benign in-envelope variation vs faults (grid world, HalfCheetah trial protocol)
8 Nov 2026 · code gridworld/G4d_benign_variation.py, G4d_eval.py · outputs gridworld/G4d_scores_7.csv,
G4d_scores_27.csv, G4d_summary.csv

Each trial is one 50,000-step rollout: healthy operation until step 15,000, then nothing (control), a benign
in-envelope change, or a fault. Benign changes move the start region; the shortest-path policy is valid from
every cell, so these are within the trained envelope and the loop's transition mechanism is unchanged. Fault:
10% of executed actions random. Windows W 500 (7×7) / 200 (27×27), stride 50; calibration on windows wholly
before step 15,000; evaluation from step 15,000; latched max |z|. Readings: P, H_f, H_b, ΔH (Chao–Shen;
three-of-four and P alone), H(S), MMD (unbiased, delta kernel on (s,a,s'); reference steps 0–7,499; calibration
windows from 7,500), reward (goal rate). Threshold per method midway between the 5th and 6th largest of the
21 control scores (5/21 controls, as in the submission). 21 independent seeds per condition.

## Alarm rate per condition (21 trials each), threshold matched to 5/21 controls

**Primary world 7×7, W 500**

| condition | three-of-four (CS) | P alone (CS) | H(S) | MMD | reward |
|---|---|---|---|---|---|
| control | 24% | 24% | 24% | 24% | 24% |
| benign: start top row of left room | 62% | 52% | 100% | 100% | 48% |
| benign: start anywhere in left room | 43% | 81% | 67% | 100% | 100% |
| benign: 50/50 left column + top row | 10% | 24% | 71% | 100% | 38% |
| benign: start bottom row of left room | 19% | 62% | 29% | 100% | 48% |
| **mean over benign** | **34%** | 55% | 67% | 100% | 59% |
| fault: 10% action noise | **100%** | **100%** | 95% | 76% | 100% |

**Stress world 27×27, W 200 (HalfCheetah-like coverage)**

| condition | three-of-four (CS) | P alone (CS) | H(S) | MMD | reward |
|---|---|---|---|---|---|
| control | 24% | 24% | 24% | 24% | 24% |
| benign: start top row of left room | 14% | 52% | 86% | 90% | 43% |
| benign: start anywhere in left room | 29% | 52% | 95% | 19% | 100% |
| benign: 50/50 left column + top row | 29% | 29% | 76% | 57% | 38% |
| benign: start bottom row of left room | 33% | 67% | 10% | 43% | 33% |
| **mean over benign** | **26%** | 50% | 67% | 52% | 54% |
| fault: 10% action noise | **100%** | **100%** | 48% | 38% | 10% |

Thresholds (7×7 / 27×27): three-of-four 3.50 / 3.65; P 3.66 / 3.87; H(S) 3.67 / 3.98; MMD 6.37 / 5.45;
reward 3.55 / 2.67. At ±3σ every method over-alarms on controls here (57–95% in 7×7, 71–95% in 27×27, reward
14–71%), so the matched threshold is the comparison.

## Reading
- **The four-reading rule is the most specific.** In the sparse world its alarm rate under benign start
  changes (26% on average) is at the control rate (24%), while it catches every fault trial. In the primary
  world it is the lowest of all methods (34%).
- **P alone is sensitive to benign changes of where the agent operates** (about 50%), though less than the
  occupancy reading H(S) (67%) and, in the primary world, MMD (100%). It still catches every fault.
- **MMD and H(S) mostly respond to where the agent goes**: high alarm rates under benign changes, and in the
  sparse world they miss most fault trials (38%, 48%).
- **Reward** fires whenever the benign change alters task outcome (start anywhere → 100%) and misses the
  fault in the sparse world (10%).
- This is the role of the three-of-four rule — P carries the detection, the agreement of
  the four readings adds specificity against benign operating changes.
