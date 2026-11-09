# Frozen configuration — 2026-09-09

**Frozen before any perturbation trial was opened.** Every choice below derives
from healthy `no_noise` runs or is carried over unchanged from `paper_v11`.
Nothing downstream may revise this file; a revision means a new arm, recorded
separately.

## Channels — 7, per joint plus torso

| channel | S dimensions | A |
|---|---|---|
| b_thigh | s_2 (angle), s_11 (ang. velocity) | a_0 |
| b_shin | s_3, s_12 | a_1 |
| b_foot | s_4, s_13 | a_2 |
| f_thigh | s_5, s_14 | a_3 |
| f_shin | s_6, s_15 | a_4 |
| f_foot | s_7, s_16 | a_5 |
| torso | s_0, s_1, s_9, s_10 | a_3, a_0 |

S' is the same dimensions one step later. The action pairing is **derived**: each
joint's argmax over bias-corrected MI(a_k;S'|S) on healthy data, which recovered
the anatomical map in all six joints (diagonal 0.29-0.45, off-diagonal 0.03-0.23).
The torso set is greedy forward selection, which stopped after {a_3, a_0}.

`s_8` (forward velocity) is dropped: 0.545 bits marginal entropy of a possible
1.585, 1.7% of drop-one MI, and it is the reward variable.

## Two arms

| arm | angle bins | velocity bins | action bins |
|---|---|---|---|
| **A — channels only** | 3 | 3 | 3 |
| **B — channels + action resolution** | 3 | 3 | **5** |

Arm A isolates the channel decomposition at the published binning. Arm B adds the
one binning change the sweep supported unanimously. Both beat the published
17-dimensional design on bias-corrected total MI and on the bias-corrected action
term; B by +2.3% and +9.4% respectively.

Edges: equal-width over the pooled healthy calibration range, per dimension.
The three edge rules tested (minmax, robust 1-99pct, quantile) agreed within 1%
on the joint channels.

## Carried over unchanged from paper_v11

Window 500 · stride 50 · Delta 1 · plug-in entropy, no bias correction in the
reported statistic · pre/post split at `window_id < injection_episode*1000/50` ·
seed exclusions ppo15, ppo18, ppo19, ppo21.

## Rejected, with reasons

- **A = all six actions per channel** (the swarm-consistent construction): raises
  bias from ~0.15 to ~0.75 bits and drives the corrected action share to zero
  (−0.9%, +0.7%). Does not survive this sample size.
- **2/2/5**: highest action term (+27%) but −15% total structure and a 4-cell
  state, which is barely a conditioning variable.
- **Omega = H(S')−H(S)**: void here, `s_next(t) == s(t+1)` exactly.
