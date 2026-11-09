# Results — H1 coverage, H2 bias correction, H4 occupancy-only baselines, H6 all seeds
8 Nov 2026 · script: analysis/H1H2_coverage_bias.py · per-trial outputs: halfcheetah/out/H1H2/

Gate: symbols recomputed from raw step logs with the published pipeline and shared baseline.json
reproduce the stored metrics to 1e-15 (P, H_f).

## H1 coverage (healthy calibration windows, W = 500, retained cohort)
occupied cells: S 210 · (S,A) 268 · (S,A,S') 343
singleton share: (S,A) 41% · (S,A,S') 59%
H(S,A,S') = 7.74 bits vs log2 W = 8.97
H_f plug-in 0.56 → Miller–Madow 0.67 bits; P plug-in 0.334 → 0.331

Change in occupied (S,A) cells after onset: obs 3% +109, act 4% +51, obs 1% +44, act 3% +33,
act 1% +11, force 10N +10, force 5N +4, gravity −5.5 (fewer cells, yet 71% detected).

## H2 bias correction (Miller–Madow), retained cohort, 168 trials / 21 controls
three-of-four ±3σ: plug-in 69.0% (5/21) → Miller–Madow 69.6% (5/21). Reward 44.0% (5/21).

## H4 single statistics at reward's 5/21 control alarms
P plug-in 58.9 · P MM 57.7 · H_f MM 58.9 · H_b MM 61.3
occupancy-only: H(S) 47.6 · occupied (S,A) cells 51.2 · occupied (S,A,S') cells 40.5 · reward 44.0

## H6 all seeds in the label map (24 agents incl. ppo15, ppo18, ppo19; 192 trials / 24 controls)
three-of-four: 67.2% (6/24) plug-in, 67.7% (7/24) MM · reward 42.7% (5/24)
Still to locate: ppo21 (raw logs present, not in label map) and two further screened PPO seeds.

## H1 completion — coverage (script halfcheetah/H1b_coverage.py; outputs out/H1b/)
Healthy calibration windows, retained cohort (189 trials), W = 500:

|                 | S    | A    | S'   | (S,A) | (S,A,S') |
|---|---|---|---|---|---|
| K/W             | 0.42 | 0.14 | 0.42 | 0.54  | 0.69 |
| singleton share | 0.29 | 0.05 | 0.29 | 0.41  | 0.59 |
| doubleton share | 0.06 | 0.02 | 0.06 | 0.06  | 0.05 |
| Good–Turing coverage | 0.71 | 0.95 | 0.71 | 0.59 | 0.41 |

Estimated unseen mass of the triplet ≈ 0.59. H(S,A,S')/log2 W is descriptive only.
Change in triplet coverage after onset: obs 3% −0.25, act 4% −0.14, obs 1% −0.12, act 3% −0.10,
act 1% −0.03, force 10N −0.02, force 5N −0.01, gravity +0.01 (coverage rises, yet 71% detected).
The two conditions with the largest coverage loss are the two where dP > 0.

## T7 — reward definition (gymnasium half_cheetah_v4.py)
reward = (x_after − x_before)/dt − 0.1·Σa², x position excluded from the observation; the logged
velocity is the instantaneous qvel at the end of the step. The control cost uses the action passed
to env.step, i.e. the executed action; action noise is added in step_async after the commanded
action is logged; observation noise is added to the returned observation, while reward is computed
from the true state. Reward is therefore a scalar task projection, not a function of the logged (S',A).

## H7 — perturbation definitions (app.py; eval-script wiring still to confirm)
obs/act noise: zero-mean Gaussian, std = noise_level "in normalized units"; act noise clipped to the
action box. Force: x-direction on the torso, 5 or 10 N, applied 10 of every 100 steps.
Gravity: z-gravity × 1.10.
