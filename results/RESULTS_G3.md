# G3 — MMD and the accounting on the same windows (grid world)
8 Nov 2026 · code: G3_accounting_vs_mmd.py · seed 20261110

Two MMD variants against a healthy reference sample (5,000 steps): unbiased MMD² with a delta kernel
on (s,a,s') symbols, and with an RBF kernel on (x,y of s, one-hot a, x,y of s') — median-heuristic
bandwidth. Same windows, same cases and constructions as G2. d = change / healthy SD.

| world | constr. | case | MMD delta | MMD RBF | P | H_f | H(S) |
|---|---|---|---|---|---|---|---|
| 7×7 | i rollout | visitation only | **33.6** | **16.5** | 0.47 | 0.25 | −4.60 |
| 7×7 | i rollout | action noise 10% | 2.5 | 0.4 | **−5.4** | **5.8** | 1.08 |
| 7×7 | ii d0 fixed | visitation only | 0.2 | 0.1 | 0.08 | −0.16 | −0.01 |
| 7×7 | ii d0 fixed | action noise 10% | 0.7 | 0.0 | **−6.0** | **6.6** | 0.03 |
| 27×27 | i rollout | visitation only | **1.7** | **3.1** | −0.49 | 0.57 | −1.74 |
| 27×27 | i rollout | action noise 10% | 0.4 | 0.5 | **−2.0** | **3.2** | −0.13 |
| 27×27 | ii d0 fixed | action noise 10% | 0.2 | 0.0 | **−1.4** | **2.2** | 0.23 |

## Finding
The two kinds of reading divide the work. MMD — with either kernel — responds most to **where the
agent goes**: a pure change of visitation, with the loop mechanism untouched, moves it 16–34 units;
a change in the loop mechanism moves it little, and with visitation held fixed hardly at all
(0.0–0.7). The accounting does the opposite: P and H_f move 5–7 units for the mechanism change, with
or without visitation held fixed, and little for the visitation change. MMD tests whether the window
distributions differ; the decomposition reports which information relationship in the loop changed.
