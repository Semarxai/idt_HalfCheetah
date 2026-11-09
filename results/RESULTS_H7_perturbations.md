# H7 — exact perturbation definitions and evaluation settings
8 Nov 2026 · sources: RL_sep_analysis/analysis/app.py (evaluation path used for the archived runs),
gymnasium half_cheetah_v4.py, and checks on the raw logs of all 252 logged trials.

## Environment stack (outermost last)
HalfCheetah-v4 → VecNormalize (loaded statistics, training = False, norm_reward = False) →
NoisyEnvWrapper → ForceInjectionWrapper → GravityPerturbationWrapper → IDTInterventionWrapper.
The intervention wrapper starts disabled; **intervention_regime is 0 in every step of all 252 logged
trials** (checked), so no correction acted on any reported run.
Policy: model.predict(obs, deterministic=True) — the mean action, no sampling.

## Episodes and onset
50 episodes × 1,000 steps = 50,000 steps per trial. Perturbation enabled from episode 15
(step 15,000) to the end: active share 0.70 of steps (checked: action noise, gravity). Calibration
windows 0–290 lie wholly before the onset; evaluation windows start at 300.

## Perturbations
| condition | definition | units |
|---|---|---|
| action noise 1 / 3 / 4% | zero-mean Gaussian, σ = 0.01 / 0.03 / 0.04, added to the commanded action in step_async, then clipped to the action box [−1, 1]; the log records the **commanded** action | action units; σ = 0.01 is 0.5% of the action range width |
| observation noise 1 / 3% | zero-mean Gaussian, σ = 0.01 / 0.03, added to the VecNormalize-normalised observation returned to the agent; the log records the noisy observation (what the policy saw) | normalised observation units (per-dimension running mean/SD) |
| external force 5 / 10 N | x-direction force on the torso, applied for 10 consecutive steps of every 100 (steps 0–9, 100–109, … of each episode; active share 0.10 after onset, checked) | newtons |
| gravity 110% | z-gravity scaled by 1.10 after onset | — |

The 1% conditions are σ = 0.01 as designed; the condition labels of the trial map are used throughout.

## Reward (for T7)
reward = (x_after − x_before)/dt − 0.1·Σa² with the executed action; observation noise does not
enter reward (computed from the true state); the logged velocity is instantaneous qvel.

## SAC training-time stochasticity
See RESULTS_Q6_SAC_policy_SD.md: training-time executed-action SD median 0.124, 10–14 × σ = 0.01.
