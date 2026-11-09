# Interaction-loop telemetry for RL agents — code and results

Code, per-trial scores and result notes for the paper *Am I Still Coupled to My Environment?* (reward-independent
runtime telemetry for reinforcement-learning agents through the information accounting of the state–action–transition loop).

For a controlled record (S, A, S′) the interaction budget is C = H(S) + H(A) + H(S′) and the coupling share is
P = I(S,A; S′) / C ≤ ½, with the exact decomposition I(S,A; S′) = I(S; S′) + I(A; S′ | S). The monitor reads P and the
forward and backward uncertainties H_f = H(S′ | S,A), H_b = H(S,A | S′), ΔH = H_f − H_b in sliding windows, each
against the agent's own healthy calibration. Reward is not used.

## Layout

| folder | contents |
|---|---|
| `config/` | trial label map (perturbation of every archived trial), frozen protocol configuration, shared discretisation `baseline.json` |
| `pipeline/` | published windowing and discretisation pipeline (`analyze_trajectory.py`); evaluation, perturbation and training code that produced the archived runs (`app.py`, `idt_monitor.py`) |
| `halfcheetah/` | analyses of the 21-agent HalfCheetah study (scripts below); `out/` holds per-trial scores, summaries and bootstrap outputs |
| `halfcheetah/submission/` | scripts behind the numbers of the submitted version, kept for traceability |
| `gridworld/` | controlled grid world with exact population quantities (`gw_core.py`, `estimators.py`) and the G-series analyses with their outputs |
| `checkpoints/` | the 21 frozen agents (11 SAC, 10 PPO; Stable-Baselines3 zips) with their observation-normalisation statistics |
| `long_runs/` | long nominal runs of the frozen agents (1,000 episodes each): numpy policies, runner, replication gates, per-window readings of every run |
| `results/` | one note per analysis with all numbers (`RESULTS_*.md`) |

## Main results (held out, Chao–Shen estimates)

Nested leave-one-agent-out: for each held-out agent the operating point is chosen on the other 20 agents (maximise inner
detection subject to inner control false alarms ≤ 0.25); 168 perturbation trials, 21 controls, every method at 6/21 control alarms.

| method | detection |
|---|---|
| telemetry, three-of-four rule of the paper (threshold held out) | 72.6% |
| coupling share P alone | 74.4% |
| telemetry, k selected on the other agents, evaluated on the held-out agent | 76.8% |
| reward + telemetry (fixed OR) | 77.4% |
| one-step dynamics model (MLP, healthy data) | 61.3% |
| MMD (same windows) | 52.4% |
| reward | 50.6% |
| H(S), occupancy only | 50.0% |

Agent-level bootstrap with the selection repeated in each resample: P alone − reward [+4.8, +39.3] points;
three-of-four − reward [+5.4, +34.5]. Details: `results/RESULTS_PRIMARY_heldout.md`, `results/RESULTS_R1_fixed_rule.md`.

**Same-stream baselines and sequential detection.** A smoothed histogram KL on the same joint binned alphabet
ties the three-of-four rule (72.6% held out). With per-episode CUSUM detectors calibrated to the same long-run
target, telemetry and reward reach similar detection (68.5% and 70.8%) but catch different faults: the telemetry
caught 18 trials that reward missed, and reward 22 that the telemetry missed; combining them reaches 78.6–80.4% at
8/21 runs alarming within 100 episodes. See `results/RESULTS_E1_E3_cusum_boundary.md`, `RESULTS_E4_combined.md`,
`RESULTS_E5_histogram_KL.md`.

**Long-run false-alarm target.** Each of the 21 frozen agents was run for 1,000 nominal episodes. With thresholds set
on the other 20 agents so that at most 25% of nominal runs alarm within 100 episodes after calibration (achieved held out:
6/21 for each method), detection of the archived faults is 46.4% for the three-of-four telemetry rule, 43.5% for P alone
and 31.0% for reward (differences to reward [−3.6, +35.1] and [−7.1, +36.3]). False alarms accumulate with exposure for
all three methods. See `results/RESULTS_H10b_long_nominal.md`.

**Correction to the submitted Table 3.** The submitted ablation script (`halfcheetah/submission/v31_p04_matched_falsealarm.py`)
used `numpy.max`, so windows with a zero interaction budget (an agent holding one state for a whole window after a fault)
made the trial scores of the min-entropy and joint normalisations NaN, and those trials were counted as misses. With one
consistent convention the plug-in values are 50.6% (min-entropy; submitted 41.1%) and 61.3% (joint; submitted 52.4%).
With coverage-adjusted estimates the normalisations perform alike (74–76% held out). See `results/RESULTS_R7_ablation.md`.

## Reproducing

```bash
pip install -r requirements.txt
```

**From the shipped per-trial scores (no raw data needed):**

| result | command (in `halfcheetah/`) | note |
|---|---|---|
| primary held-out table, curves, intervals | `python PRIMARY_nested_table.py point` · `curve` · `boot <chunk> <n>` · `bootsum` | RESULTS_PRIMARY_heldout |
| nested protocol (plug-in, all methods) | `python H5_nested.py point` · `curve` · `boot` · `bootsum` | RESULTS_H5_nested |
| fixed three-of-four rule | `python R1_fixed_rule.py point` · `boot` · `bootsum` | RESULTS_R1_fixed_rule |
| MMD settings | `python R4_eval.py point` · `boot` · `bootsum` | RESULTS_R4_MMD_sensitivity |
| normalisation ablation | `python R7_ablation.py point` · `boot` · `bootsum` | RESULTS_R7_ablation |
| bins × windows | `python H9_eval.py point` · `boot <k> <W> <n>` · `bootsum` | RESULTS_H9_bins_windows |
| agent-clustered lead | `python R6_lead_clustered.py` | RESULTS_R6_lead_clustered |
| ablation curves | `python R7b_curves.py` | RESULTS_R7_ablation |
| seed-19 runs held out together | `python H6b_grouped.py` | RESULTS_H6_seeds |
| histogram KL, bootstrap | `python E5_boot.py` | RESULTS_E5_histogram_KL |
| reset-triplet sensitivity | `tar xzf out/E1_bundle.tgz && python E3_boundary.py` | RESULTS_E1_E3_cusum_boundary |
| CUSUM (short and long-run) | `tar xzf out/E1_bundle.tgz && tar xzf out/E1_long_bundle.tgz && python E2_cusum_eval.py 1000` | RESULTS_E1_E3_cusum_boundary |
| combined monitor, overlap | (after the line above) `python E4_combined.py 1000` | RESULTS_E4_combined |

The bundles `out/E1_bundle.tgz` (per-episode and boundary-corrected per-window readings of all 189 archived
trials) and `out/E1_long_bundle.tgz` (per-episode readings of the 21 long runs) are derived data; extract them in
`halfcheetah/` before running E2–E4.

**Scripts that start from the raw step logs** (408 evaluation trajectories, 50,000 steps each; not distributed, see Data):
`H1H2_coverage_bias.py`, `H1b_coverage.py`, `H2_chaoshen.py` → `H2_eval.py`, `H3_mmd.py`, `H6_extra_seeds.py` → `H6_eval.py`,
`H8_decomposition.py` → `H8_eval.py`, `H9_bins_windows.py`, `H10a_time_to_alarm.py`, `H13_dynamics.py`, `R4_mmd_sensitivity.py`,
`Q6_sac_policy_sd.py`, `E1_episode_readings.py`, `E5_kl.py` → `E5_eval.py`. Their outputs (per-trial scores and derived readings)
are included, so every table can be recomputed without them. Set `IDT_DATA_ROOT` to the folder that contains `RL_sep_analysis/` (raw runs, metrics, checkpoints).

**Grid world** (self-contained; in `gridworld/`):

| analysis | command | note |
|---|---|---|
| G1 truth vs estimate | `python G1_truth_vs_estimate.py` | RESULTS_G1 |
| G2 occupancy vs mechanism | `python G2_occupancy_vs_mechanism.py` | RESULTS_G2 |
| G3 accounting vs MMD | `python G3_accounting_vs_mmd.py` | RESULTS_G3 |
| replication over seeds | `python R2_replicate.py g1` · `g2` · `g3a` · `g3b` · `g3sum` | RESULTS_R2_gridworld_replication |
| G4a normalisation stability | `python G4a_normalisation_stability.py` | RESULTS_G4a_normalisation_stability |
| G4b entry point × mechanism | `python G4b_entry_mechanism.py 7` · `27`, then `python G4b_eval.py` | RESULTS_G4b_entry_mechanism |
| G4c gradual faults | `python G4c_gradual.py 7` · `27`, then `python G4c_eval.py` | RESULTS_G4c_gradual |
| G4d benign variation | `python G4d_benign_variation.py 7` · `27`, then `python G4d_eval.py` | RESULTS_G4d_benign_variation |

**Long nominal runs** (in `long_runs/`; the runner needs the checkpoints, and gate 1/2 the archived control logs):
`python gate1_actions.py` (checkpoints reproduce logged actions), `python runner.py gate 1 50000 <agents>` then
`python gate2_compare.py`, `./batch.sh <n> <agents> -- <agents>` (1,000,000 steps per agent), `python post_batch.py <agents>` (per-window readings into
`long_runs/readings/`, included for all 21 runs), `python E1_long.py <agents>` (per-episode readings), `python h10b_eval.py 1000` (held-out thresholds at the long-run target, applied to the archived
trials; outputs in `long_runs/out/`). Note: `results/RESULTS_H10b_long_nominal.md`.

## Data

Included: the 21 frozen checkpoints (`checkpoints/`), the trial label map and discretisation (`config/`), per-trial
scores for every analysis (`halfcheetah/out/`, shipped as `halfcheetah/out.tgz`; unpack with `tar -xzf halfcheetah/out.tgz` from the repository root), per-episode readings of the archived trials and the long runs
(`halfcheetah/out/*bundle.tgz`), per-window readings of all long runs (`long_runs/readings/`), and the grid-world
outputs. The raw step logs of the archived evaluations are not distributed; every reported table can be recomputed
from the included data, and the long runs can be regenerated from the checkpoints. Point `IDT_DATA_ROOT` at a folder
containing `RL_sep_analysis/` to run the scripts that need raw files.

## Environment

HalfCheetah-v4 (gymnasium); the long runs used gymnasium 1.4.0 and MuJoCo 3.4.0, with policies evaluated in numpy from
the saved weights (no torch needed). Python ≥ 3.10.

## License

MIT (see `LICENSE`).
