# 1% action noise vs SAC's training-time stochasticity
8 Nov 2026 · script halfcheetah/Q6_sac_policy_sd.py (run in the cloud workspace) · output halfcheetah/out/Q6_sac_policy_sd.csv

Method: for each of the 11 SAC agents, the actor is read from its saved checkpoint
(checkpoints/_all/SAC_Seeds/sac_claude_seed{N}.zip; weights parsed from policy.pth, network
17 → 256 → 256 → (μ, log σ), log σ clipped to [−20, 2]) and evaluated at every logged state of the
agent's unperturbed run. Training-time actions are tanh(μ + σ·ε); their SD is estimated per state and
action dimension from 200 draws at 2,000 sampled states.

Identity check: tanh(μ) reproduces the logged deterministic (commanded) action to ≤ 1e-4 for every
agent — the checkpoints are the evaluated policies and the logged observations are the policy's inputs.

| seed | Gaussian σ median [p10, p90] | executed-action SD median [p10, p90] | median / 0.01 |
|---|---|---|---|
| 1 | 0.408 [0.162, 0.533] | 0.123 [0.039, 0.296] | 12.3 |
| 3 | 0.420 [0.141, 0.504] | 0.140 [0.041, 0.354] | 14.0 |
| 5 | 0.437 [0.146, 0.550] | 0.128 [0.037, 0.329] | 12.8 |
| 7 | 0.441 [0.209, 0.545] | 0.124 [0.038, 0.329] | 12.4 |
| 9 | 0.351 [0.128, 0.467] | 0.099 [0.037, 0.232] | 9.9 |
| 10 | 0.433 [0.152, 0.564] | 0.119 [0.042, 0.343] | 11.9 |
| 11 | 0.429 [0.182, 0.710] | 0.135 [0.025, 0.498] | 13.5 |
| 13 | 0.431 [0.197, 0.573] | 0.118 [0.036, 0.366] | 11.7 |
| 14 | 0.447 [0.186, 0.575] | 0.132 [0.043, 0.319] | 13.2 |
| 18 | 0.449 [0.147, 0.565] | 0.134 [0.046, 0.329] | 13.4 |
| 19 | 0.398 [0.107, 0.538] | 0.118 [0.038, 0.341] | 11.8 |

Across agents: median training-time action SD 0.124 (agent medians 0.099–0.140), i.e. 10–14 × the
1% evaluation noise (σ = 0.01); even the 10th percentile (0.025–0.046) exceeds 0.01.

Reading: evaluation uses the deterministic mean action, so the healthy reference is the deterministic
loop. The 1% action-noise fault is an order of magnitude smaller than the stochasticity the policy
experienced in training, yet the telemetry detects it in 57.1% of held-out trials (consensus,
Chao–Shen; P alone 47.6%; reward 47.6%; MMD 47.6%; dynamics model 42.9%).
