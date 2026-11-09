# G4c — gradual faults (grid world)
8 Nov 2026 · code gridworld/G4c_gradual.py, G4c_eval.py · outputs gridworld/G4c_running_7.npz, G4c_running_27.npz
(latched running max per window), G4c_alarm_by_magnitude.csv, G4c_first_alarm.csv

## Design
The G4b mechanisms, ramped linearly from zero at onset (step 15,000) to the G4b magnitude at the end of the trial
(step 50,000): λ(t) = (t − 15,000)/35,000. 21 seeds per condition; fresh controls (new seeds); Chao–Shen readings;
windows, calibration and latched alarms as G4b; thresholds matched to 5/21 controls at the end of the trial.

| condition | ramp |
|---|---|
| agent-side stochastic | executed action random with probability 0.1·λ |
| agent-side systematic | sensor offset applied with probability λ (one draw per step) |
| env-side stochastic | slip outside the zone 0.05 → 0.05 + 0.10·λ (kernel mixture; the kernel is linear in slip) |
| env-side systematic | wind south in the right room with probability 0.2·λ |

Note: ramping the sensor offset as a probability makes it intermittent, so this cell is a flickering observation
(stochastic in effect), not the fixed offset of G4b. Its results are not comparable to the G4b systematic cell.

## Alarm rate by fault magnitude (cumulative, latched)
Columns give the share of trials alarmed by the time the fault reached 25/50/75/100% of full magnitude.
Control rows give the false-alarm share over the same period (5/21 = 24% at the end by construction).

| world | condition | P alone | three-of-four | H(S) | MMD | reward |
|---|---|---|---|---|---|---|
| 7×7 | control | 0 / 5 / 24 / 24 | 5 / 10 / 24 / 24 | 10 / 19 / 24 / 24 | 10 / 19 / 19 / 24 | 10 / 14 / 19 / 24 |
| 7×7 | action noise | 33 / **100** / 100 / 100 | 19 / 76 / 100 / 100 | 10 / 10 / 24 / 67 | 10 / 19 / 57 / 76 | 5 / 33 / 67 / 86 |
| 7×7 | intermittent offset | 100 / 100 / 100 / 100 | 100 / 100 / 100 / 100 | 0 / 10 / 43 / 100 | 29 / 100 / 100 / 100 | 0 / 5 / 5 / **5** |
| 7×7 | slip | 43 / **100** / 100 / 100 | 29 / 95 / 100 / 100 | 14 / 14 / 29 / 76 | 38 / 57 / 90 / 100 | 14 / 38 / 86 / 100 |
| 7×7 | wind | 57 / **100** / 100 / 100 | 29 / 90 / 100 / 100 | 5 / 5 / 14 / 14 | 29 / 90 / 100 / 100 | 10 / 33 / 81 / 100 |
| 27×27 | control | 0 / 5 / 14 / 24 | 0 / 0 / 14 / 24 | 5 / 10 / 19 / 24 | 0 / 10 / 14 / 24 | 10 / 19 / 19 / 24 |
| 27×27 | action noise | 14 / 57 / 95 / **100** | 24 / 67 / 95 / 95 | 5 / 5 / 10 / 19 | 19 / 29 / 57 / 67 | 24 / 24 / 24 / **24** |
| 27×27 | intermittent offset | 100 / 100 / 100 / 100 | 86 / 100 / 100 / 100 | 0 / 0 / 10 / 19 | 10 / 10 / 19 / 19 | 24 / 29 / 29 / 33 |
| 27×27 | slip | 24 / 86 / 100 / **100** | 19 / 67 / 95 / 100 | 5 / 5 / 5 / 10 | 14 / 33 / 48 / 52 | 24 / 24 / 24 / **24** |
| 27×27 | wind | 38 / 76 / 100 / 100 | 24 / 86 / 95 / 100 | 0 / 0 / 0 / 0 | 14 / 38 / 57 / 90 | 19 / 48 / 90 / 100 |

Thresholds 7×7 / 27×27: P 3.86 / 4.28; three-of-four 3.53 / 3.67; H(S) 3.67 / 4.08; MMD 5.60 / 4.89; reward 3.94 / 2.66.

## Fault magnitude at first alarm (detected trials; median λ, IQR; median steps after onset)

| world | condition | P alone | three-of-four | H(S) | MMD | reward |
|---|---|---|---|---|---|---|
| 7×7 | action noise | 21/21 · 0.30 (0.22–0.36) · 10,600 | 21/21 · 0.33 · 11,600 | 14/21 · 0.78 | 16/21 · 0.60 | 18/21 · 0.56 |
| 7×7 | intermittent offset | 21/21 · 0.07 (0.05–0.09) · 2,600 | 21/21 · 0.11 · 3,900 | 21/21 · 0.78 | 21/21 · 0.30 | 1/21 |
| 7×7 | slip | 21/21 · 0.26 (0.20–0.31) · 9,050 | 21/21 · 0.31 · 10,750 | 16/21 · 0.81 | 21/21 · 0.43 | 21/21 · 0.59 |
| 7×7 | wind | 21/21 · 0.23 (0.16–0.32) · 8,200 | 21/21 · 0.31 · 10,850 | 3/21 | 21/21 · 0.34 | 21/21 · 0.56 |
| 27×27 | action noise | 21/21 · 0.48 (0.31–0.61) · 16,850 | 20/21 · 0.43 · 15,200 | 4/21 | 14/21 · 0.54 | 5/21 (chance) |
| 27×27 | intermittent offset | 21/21 · 0.12 (0.08–0.15) · 4,050 | 21/21 · 0.18 · 6,250 | 4/21 | 4/21 | 7/21 |
| 27×27 | slip | 21/21 · 0.36 (0.29–0.43) · 12,600 | 21/21 · 0.40 · 14,050 | 2/21 | 11/21 · 0.37 | 5/21 (chance) |
| 27×27 | wind | 21/21 · 0.36 (0.21–0.48) · 12,450 | 21/21 · 0.33 · 11,400 | 0/21 | 19/21 · 0.59 | 21/21 · 0.51 |

Controls: median λ at the (false) first alarm 0.58–0.72 for P and three-of-four, i.e. false alarms spread over the
evaluation period. Reward's 5/21 in the sparse-world stochastic cells equals the control rate, with the first alarm
at λ ≈ 0.01–0.03, i.e. before the fault had any size: these are chance alarms, not detections.

## Reading
1. **Gradual faults are caught early in their growth.** P alone detects every gradual fault in both worlds by the
   end, and in the 7×7 world all of them by half magnitude. Median magnitude at first alarm: about a quarter to a
   third of full size in the 7×7 world (action noise ≈ 3%, slip ≈ 0.08, wind ≈ 0.05), a third to a half in the sparse world.
2. **P alone is earlier than three-of-four** (lower median λ at alarm in 6 of 8 cells; three-of-four earlier for sparse-world action noise and wind); three-of-four's role remains
   specificity under benign change (G4d).
3. **Reward reads gradual faults late or not at all**: about twice the magnitude of P in the 7×7 world where it
   detects, misses the offset (5%), and is at chance for the sparse-world stochastic faults (24% = control rate).
4. **H(S) and MMD are late or weak under gradual change**, in particular in the sparse world (H(S) 0–19%,
   MMD 19–90%), while their abrupt-fault rates (G4b) were higher in the 7×7 world.
5. Gradual onset therefore does not hide the change from the accounting: the coupling share tracks the growing
   fault, not only an abrupt step.
