# H8 — decomposition of the coupling share, per condition (Chao–Shen primary, plug-in for comparison)
8 Nov 2026 · scripts halfcheetah/H8_decomposition.py, H8_eval.py · outputs out/H8/, out/H8_trial_changes.csv,
out/H8_eval_output.txt

Identity (T2): I(S,A;S') = I(S;S') + I(A;S'∣S), so with C = H(S)+H(A)+H(S'):
P = I(S;S')/C + I(A;S'∣S)/C = state term + action term.
Computed on one coherent set of entropies per window (H(S), H(A), H(S'), H(S,A), H(S,S'), H(S,A,S'));
the identity holds to 3.5e-16 in every window for both estimators. Retained cohort, 189 trials.
Change = mean over evaluation windows (≥300) − mean over calibration windows (0–290).

## Healthy levels (mean over trials)
| | Chao–Shen | plug-in |
|---|---|---|
| P | 0.3020 | 0.3337 |
| state term I(S;S')/C | 0.2831 | 0.3140 |
| action term I(A;S'∣S)/C | 0.0190 | 0.0198 |
| H(A∣S) bits | 0.850 | 0.531 |

The ratio forms I(S;S')/H(S) and I(A;S'∣S)/H(S'∣S) are undefined in some windows (zero or negative
corrected denominators in the sparse regime); the additive terms above are exact and used throughout.

## Chao–Shen: change by condition

| condition | ΔP ×10³ | Δ state term ×10³ | Δ action term ×10³ | ΔP / SD | agents with P falling (of 21) | state-term share of the change |
|---|---|---|---|---|---|---|
| action noise 1% | −5.23 | −6.67 | +1.44 | −0.64 | 16 | 0.69 |
| action noise 3% | −10.83 | −12.38 | +1.55 | −1.41 | 19 | 0.72 |
| action noise 4% | −13.53 | −15.46 | +1.92 | −1.69 | 21 | 0.83 |
| obs noise 1% | −9.92 | −11.75 | +1.83 | −1.21 | 20 | 0.77 |
| obs noise 3% | −17.05 | −19.24 | +2.19 | −2.20 | 21 | 0.81 |
| force 10 N | −2.71 | −4.04 | +1.33 | −0.32 | 16 | 0.70 |
| force 5 N | −0.57 | −0.95 | +0.38 | −0.09 | 11 | 0.68 |
| gravity 110% | −1.66 | −1.72 | +0.06 | −0.26 | 12 | 0.74 |
| controls | +0.55 | +0.39 | +0.16 | — | — | — |

Agents in which the state term falls: 15, 19, 21, 19, 21, 15, 10, 10 (same order). Agents in which the
action term rises: 12, 11, 12, 11, 11, 15, 12, 10.

## Plug-in: change by condition (for comparison with the submission)

| condition | ΔP ×10³ | Δ state term ×10³ | Δ action term ×10³ | ΔP / SD | agents with P falling |
|---|---|---|---|---|---|
| action noise 1% | −3.75 | −5.56 | +1.81 | −0.55 | 14 |
| action noise 3% | −2.39 | −5.40 | +3.01 | −0.39 | 9 |
| action noise 4% | +0.68 | −2.78 | +3.46 | +0.22 | 4 |
| obs noise 1% | +0.60 | −2.55 | +3.15 | +0.01 | 7 |
| obs noise 3% | +11.92 | +13.09 | −1.16 | +1.82 | 2 |
| force 10 N | −1.29 | −2.51 | +1.22 | −0.18 | 8 |
| force 5 N | −0.39 | −0.84 | +0.44 | −0.08 | 9 |
| gravity 110% | −2.74 | −2.98 | +0.24 | −0.58 | 14 |

## Findings
1. **With coverage-adjusted estimation the coupling share falls under every fault**, in the direction
   the interpretation predicts: on average in all 8 conditions, and in 19–21 of 21 agents for the four
   strongest faults (action noise 3–4%, observation noise 1–3%). Controls are flat (+0.55 × 10⁻³).
2. **The mixed signs reported for plug-in estimates are estimator-dependent: they disappear under coverage adjustment.** With plug-in estimates P rises under
   observation noise 3% (in 19 of 21 agents), 1% and action noise 4% — the conditions with the largest
   coverage loss (H1). Coverage-adjusted, all three fall, in 20–21 of 21 agents.
3. **The state term carries the change** (68–83% of |ΔP|): what the next state shares with the current
   one drops; the action term rises slightly, in about half the agents. Read with G1: in exact-symbol
   records with a deterministic policy the action term is zero, so its small positive value here is
   representation-dependent and is reported as a diagnostic, not as action impact (decision gate).
4. Weak faults (force 5 N, gravity) move P little and in only about half the agents; they are detected
   through the consensus of the four readings rather than through P alone.
