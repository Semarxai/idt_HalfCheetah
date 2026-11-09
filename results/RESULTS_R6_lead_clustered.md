# R6 — early detection with agent-clustered inference (review point 6)
8 Nov 2026 · scripts halfcheetah/R6_lead_clustered.py (uses out/H10a_first_alarm.csv; H10a rerun with the fixed
three-of-four rule added, out/H10a_output_v2.txt) · output out/R6_lead_clustered.csv, out/R6_output.txt

Nested operating points (ALPHA 0.25). Agent-level bootstrap, 10,000 resamples of the 21 agents. Replaces the
pooled trial-level sign tests of RESULTS_H10a (those p-values are withdrawn).
Columns: difference in share of perturbation trials detected within 1 / 5 evaluation episodes (percentage
points); on trials both detect, net share earlier = (earlier − later)/both; median lead in episodes;
agent-level sign test on each agent's median lead.

| comparison | detected by ep 1, diff [95%] | by ep 5, diff [95%] | net earlier share [95%] | median lead, ep [95%] | agents earlier v later |
|---|---|---|---|---|---|
| three-of-four CS vs reward | +14.3 [+5.4, +23.2] | +17.3 [+4.2, +29.2] | +0.12 [−0.09, +0.38] | +0.40 [0.00, +2.05] | 12 v 7, p 0.359 |
| P CS vs reward | **+19.0 [+7.1, +32.1]** | **+20.8 [+6.0, +35.7]** | +0.21 [−0.01, +0.47] | +1.88 [0.00, +4.95] | 14 v 5, p 0.064 |
| k-of-four CS vs reward | +21.4 [+10.7, +32.7] | +24.4 [+10.7, +38.1] | +0.25 [+0.01, +0.54] | +1.95 [0.00, +4.75] | 13 v 5, p 0.096 |
| three-of-four CS vs MMD | +11.3 [+6.0, +16.7] | +17.9 [+10.7, +25.0] | +0.20 [+0.06, +0.36] | +0.65 [0.00, +5.35] | 11 v 2, p 0.022 |
| P CS vs MMD | +16.1 [+7.7, +25.6] | +21.4 [+11.9, +32.1] | +0.16 [−0.04, +0.38] | +2.10 [0.00, +6.60] | 9 v 6, p 0.607 |
| k-of-four CS vs MMD | +18.5 [+11.3, +26.8] | +25.0 [+16.7, +33.9] | +0.26 [+0.11, +0.41] | +2.12 [0.00, +5.97] | 11 v 3, p 0.057 |
| P CS vs H(S) | +16.7 [+5.4, +29.2] | +23.2 [+10.7, +35.7] | +0.21 [−0.07, +0.48] | +0.25 [0.00, +3.75] | 12 v 6, p 0.238 |
| k-of-four plug-in vs reward | +6.0 [−3.0, +16.1] | +9.5 [−3.6, +23.2] | −0.16 [−0.44, +0.16] | −0.07 [−1.90, +0.80] | 11 v 8, p 0.648 |

Fixed three-of-four CS (H10a v2): detected 72.6%, median delay 3.33 ep, 25.6% within 1 episode, 41.7%
within 5; controls 0% false alarms by episode 1, 4.8% by 5, 28.6% by 34.5 (median first alarm 13.3 ep).

## Reading
- **Supported, agent-clustered:** the coverage-adjusted telemetry catches more faults early — within one
  evaluation episode P detects 19 points more of all fault trials than reward ([+7.1, +32.1]), within five
  episodes 21 points more ([+6.0, +35.7]); similar against MMD and H(S).
- **Suggestive, not established:** on the faults both methods catch, the telemetry tends to be earlier
  (median lead ≈ 2 episodes; most agents lean that way), but the agent-level intervals include or touch
  zero. The paper should say "detects more faults early", not "detects the same faults earlier".
- Plug-in: no early advantage (consistent with the submission).
