# H3 — same-stream MMD on HalfCheetah (in-sample; nested evaluation follows in H5)
8 Nov 2026 · script halfcheetah/H3_mmd.py · per-trial outputs halfcheetah/out/H3/

Input: the monitor's own 40-d (S, A, S') stream, z-scored with the shared reference baseline, before
binning. Reference: 800 steps from steps 0–7,499. Unbiased MMD² with RBF kernel, median-heuristic
bandwidth on the reference. Each window (W 500, stride 50) sampled to 300 steps. Calibration
mean/SD on windows 150–290 (no overlap with the reference steps); evaluation on windows ≥ 300;
latched, one-sided. Retained cohort: 168 perturbation trials, 21 controls.

| detector | threshold | detection | control alarms |
|---|---|---|---|
| MMD | +3σ | 96.4% | 18/21 |
| MMD | matched (z = 8.06) | 51.8% | 5/21 |
| three-of-four consensus | ±3σ | 69.0% | 5/21 |
| reward | ±3σ | 44.0% | 5/21 |

At ±3σ MMD flags almost every healthy control: the healthy stream's distribution moves enough over
an episode sequence that a distribution distance needs a much wider threshold.

Overlap at matched control alarms: both detect 72 trials; consensus only 44; MMD only 15;
union 78.0%.

Per condition (consensus / MMD / reward):

| condition | consensus | MMD | reward |
|---|---|---|---|
| action noise 1% | 61.9 | 47.6 | 38.1 |
| action noise 3% | 57.1 | 28.6 | 28.6 |
| action noise 4% | 81.0 | 52.4 | 47.6 |
| obs noise 1% | 76.2 | 47.6 | 38.1 |
| obs noise 3% | 95.2 | 66.7 | 81.0 |
| force 5 N | 47.6 | 23.8 | 23.8 |
| force 10 N | 61.9 | 57.1 | 33.3 |
| gravity 110% | 71.4 | **90.5** | 61.9 |

Reading, consistent with grid world G3: MMD is strongest where the perturbation moves the state
distribution itself (gravity changes the posture the agent operates in); the accounting is strongest
where the loop's mechanism changes (noise, intermittent force). Complementary: together they reach
78.0%.

Sensitivity: consensus with calibration restricted to windows 150–290 (as MMD): 69.6% detection,
7/21 control alarms.
