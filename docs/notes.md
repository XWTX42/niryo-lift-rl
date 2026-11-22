# Experiment notes

Running log of what was tried, why, and what was observed. Newest sections at the bottom.
All numbers come from runs in this repo unless stated otherwise.

---

## 2026-11-17 — Why did Pick/Lift get 0% in the course project?

### Starting point

- Course project result: PPO/SAC reach 100% on `NiryoReach-v1`, but 0% on Lift / PickAndPlace.
- Things tried back then: joint (`free`) and `mocap` control, a dense reward with a grasp bonus,
  fixed cube position as a curriculum, tuning friction and mass. Observation: the cube could be
  grasped but not lifted; higher friction lifted it slightly.
- Two hypotheses:
  - **H1 (reward):** the grasp bonus creates a local optimum ("hold still and collect reward").
  - **H2 (physics):** the contact physics does not allow lifting at all.

Approach: before any RL, test the environment itself — unit tests for suspected bugs, then a
hand-written (scripted) controller to check whether lifting is physically possible.

### Setup

| Package | Version | Note |
|---|---|---|
| Python | 3.12 | |
| mujoco | 3.3.7 | **must be < 3.12**, see bug B3 |
| gymnasium | 1.4.0 | |
| gymnasium-robotics | 1.4.2 | Fetch environments for the positive control |
| stable-baselines3 | 2.9.2 | |

Environment: fork of [m-zorn/niryo-gym](https://github.com/m-zorn/niryo-gym) (MIT), installed in
editable mode from `../niryo-gym`. Versions are locked in `uv.lock`.

### Bugs in upstream niryo-gym

Fixed in the fork, one branch per fix, each with a test that fails before and passes after.

| # | Bug | Evidence | Fix |
|---|---|---|---|
| B1 | **Mocap mode drops the gripper command.** `_set_mocap` overwrites `action` with a 7-dim array before reading `action[3]`, so it always reads 0. | `data.ctrl` stays `[0, 0]` for gripper commands +1 and −1 (`tests/test_gripper.py` in the fork). | Branch `fix/mocap-gripper-command`: keep `action` intact, build a separate `mocap_action`. |
| B2 | MJCF references `*.stl`, files are named `*.STL`. Works on macOS (case-insensitive), fails on Linux. | `tests/test_assets.py` compares names with `os.listdir`, not `os.path.exists`. | Branch `fix/mesh-filename-case`: match the XML to the real file names. |
| B3 | `get_joint_qpos` asserts `joint_type in (mjJNT_HINGE, mjJNT_SLIDE)`; this fails from MuJoCo 3.12 on. | Bisected: 3.11 OK, 3.12 fails. | Not fixed upstream yet; pinned mujoco 3.3.7 (current when the repo was last updated, June 2025). |

**Consequence of B1:** in mocap mode the gripper can never open or close, so 0% on any grasping
task is guaranteed regardless of algorithm or reward.

### Task-design problems (not bugs, fixed as parameters in `niryo_lift.envs.variants`)

**P1 — Most cubes are out of reach.**
The grip site cannot get further than ~0.56 m horizontally from the robot base (base at
x = −0.3). Cube spawn regions: Lift x ∈ [0.15, 0.45], PickAndPlace x ∈ [0.25, 0.55], both
|y| ≤ 0.15. Over 500 resets, the fraction of reachable cubes is **≈30% for Lift and ≈1% for
PickAndPlace** (rough estimate, assumes the same reach in every direction; only measured
straight ahead).
In x ∈ [−0.15, 0.15], |y| ≤ 0.15, the grip site reaches targets within 0.7 cm at cube height
(z = −0.235) and within 0.5 cm at goal height (z = 0.25). Success threshold is 5 cm.

**P2 — The cube spawns in mid-air and falls very slowly.**
Spawn at z = 0, resting height −0.235 (no table, floor at −0.25). The cube's free joint has
damping 0.1 on a 5 g body, so it falls at terminal velocity mg/c ≈ 0.49 m/s (measured: 0.98 cm
per step). It takes **29 steps** to settle, i.e. more than half of the 50-step episode.

| | damping | mass | damping / mass |
|---|---|---|---|
| Fetch | 0.01 | 2 kg | 0.005 |
| Niryo | 0.1 | 0.005 kg | 20 |

The same damping resists lifting: at the maximum mocap speed (2.5 m/s) the damping force is
≈0.25 N, about 5× the cube's weight (0.049 N).

**P3 — Timing.** The model timestep is 0.001 s with 20 substeps, so **one env step = 0.02 s**
(the upstream README says 0.002). 50 steps = 1 s. Fetch uses 0.04 s/step, so its 50-step
episode is 2 s.

**P4 — The gripper is very slow.** Finger joints have `armature = 10` (extra inertia of 10 kg
on a 67 g finger) and the actuator force is capped at 1 N. +1 opens, −1 closes. Initial finger
gap is 2.07 cm (narrower than the 3 cm cube); fully open is 4.07 cm.

| armature | open to > 3.1 cm | fully open | close to < 3.0 cm |
|---|---|---|---|
| 10 (original) | 17 steps | 23 | 17 |
| 1 | 6 | 8 | 6 |
| 0.1 | 3 | 3 | 3 |
| 0.01 | 2 | 2 | 2 |

Fetch opens in ~2 steps and closes in ~3, despite armature 100, because it uses a stiff
position actuator (kp = 30000). Speed depends on force / inertia.
Lowering armature changes speed but **not** the steady clamping force (still 1 N), so it does
not confound H2. At armature 0.1 the fingers overshoot their joint limit briefly (+10.2 mm,
gap −0.2 cm) because joint limits and contacts are soft.

**P5 — Lift target is high.** Goal z = 0.25, so the cube must rise ≈0.485 m. Not changed yet.

**P6 — PickAndPlace goals.** Goal x is sampled in [−0.15, 0.15] around the robot base while
the cube sits at x ≈ 0.4; 88% of goals have x < 0.1. Not addressed yet (Lift first).

### Gripper geometry (needed for the scripted controller)

- The gripper points horizontally along +x; the two fingers are separated along y.
- The slot between the fingers is open from above, so a top-down grasp works **if the gripper
  is opened first** (initial gap 2.07 cm < 3 cm cube).
- Along x, the slot spans [grip site − 2.4 cm, grip site + 1.2 cm]; behind that is the palm
  (`base_gripper`). The grip site should be ~5 mm ahead of the cube centre to centre the cube
  in the slot. A first attempt with 15 mm pushed the cube with the palm.

### Scripted controller

`niryo_lift/controllers/scripted.py`: a 4-phase state machine
(APPROACH → DESCEND → GRASP → LIFT) with proportional control capped at 0.5 (2.5 cm/step).
Problems found while writing it:

1. Full-speed motion overshoots because the arm lags the mocap target → added `max_speed`.
2. `grasp_x_offset = 0.015` pushed the cube along with the palm → measured slot, set 0.005.
3. Fingers sank up to **21.8 mm** into the cube and squeezed it out → see H2 below.

### H2: answered — single-point contact, not friction

With the default collision pipeline, the mesh fingers and the box get **one contact point per
finger**, located at the cube's corners. MuJoCo replaces meshes by their convex hulls for
collision, and general convex collision (MPR, GJK/EPA) returns a single contact point.
With one point per finger the cube rotates and slips out. The very light cube also makes the
soft contact very compliant.

Enabling MuJoCo's `multiccd` flag (`multi_contact=True` in our variant) returns several
contact points per pair (10 seeds, armature 1, 5 g cube):

| | contact points during grasp | max penetration | success |
|---|---|---|---|
| default | 2 | 21.8 mm | 0/10 |
| multiccd | 8 | 5.7 mm | 8/10 |

Supporting evidence: **raising finger force made things worse** (1 N → 5 N / 10 N dropped
every configuration to 0%, 30 seeds) — squeezing harder pushes the fingers deeper and ejects
the cube faster. This matches the course-project observation that more friction helped only a
little.

### Ablation with the scripted controller

Baseline for all rows: reachable, at-rest spawn (`cube_xy_center=(0, 0)`, `spawn_on_floor=True`).
Each row adds the listed changes; everything else is original. 50 seeds, up to 200 steps.

| Setting | success ≤ 50 steps | success ≤ 200 steps | steps to success (median / p90) |
|---|---|---|---|
| reachable spawn (baseline) | 0% | 0% | – |
| armature = 1 only | 0% | 0% | – |
| + multi_contact | 0% | 22% | 76 / 113 |
| + multi_contact + armature = 1 | 6% | 80% | 63 / 76 |
| **+ multi_contact + armature = 0.1** | 36% | **100%** | **55 / 66** |
| + multi_contact + armature = 1 + damping = 0.01 | 8% | 96% | 57 / 71 |
| + multi_contact + armature = 1 + mass = 0.05 | 4% | 68% | 52 / 65 |
| + multi_contact + armature = 0.1 + mass = 0.05 | 50% | 92% | 48 / 62 |

Reproduce: `uv run python scripts/eval_scripted.py`.

### How the failures happen

`scripts/failure_phases.py` classifies each episode by the highest point the cube reached
(lifted = rose more than 2 cm). 50 seeds, 200 steps.

| Setting | success | dropped (lifted, then lost) | never lifted |
|---|---|---|---|
| reachable spawn (baseline) | 0 | 4 (rose 3–17 cm) | 46 |
| + multi_contact | 11 | 37 (rose 3–43 cm) | 2 |
| + multi_contact + armature = 1 | 40 | 9 (rose 2–43 cm) | 1 |

- Almost all failures end in phase LIFT: the controller believes it holds the cube.
- `multi_contact` fixes "cannot pick up" (46 → 2 never lifted).
- A slower gripper causes more mid-air drops (37 vs 9). **Unverified hypothesis:** slow fingers
  meet the "fingers stopped" condition (`settle_tol` = 0.1 mm/step) before they clamp firmly,
  so the controller lifts too early. To check: stricter `settle_tol` at armature 10, or log
  finger motion at the GRASP → LIFT transition.

Results were identical on macOS (ARM) and Linux (x86).

### Final task: `NiryoLiftFixed-v0`

Chosen with the scripted controller only, before any RL training, so the task is not tuned
to favour one algorithm.

| Parameter | Value | Changed? |
|---|---|---|
| cube spawn | centre (0, 0), ±0.15, at rest on the floor | yes (P1, P2) |
| multi_contact | True | yes (H2) |
| finger armature | 0.1 | yes (P4) |
| cube mass / damping / finger force | 5 g / 0.1 / 1 N | no |
| goal height | z = 0.25 | no |
| episode length | 110 steps | yes: 2 × the scripted median (55) |

Scripted controller on `NiryoLiftFixed-v0`: 50/50 successes. Average steps per phase in
successful episodes: APPROACH 21.5, DESCEND 4.3, GRASP 10.2, LIFT 20.8. GRASP takes ~10 steps
although the fingers close in ~3, because the soft contact keeps the fingers creeping until
they move less than `settle_tol` per step.

### Status of the hypotheses

- **H2 (physics):** answered. Lifting was impossible mainly because of single-point mesh
  contact; in mocap mode additionally because of bug B1, and for most spawns because of P1.
- **H1 (reward local optimum):** still open — to be tested with RL (sparse vs dense reward).

### Open questions

- Is the reach really ~0.56 m in every direction? Only measured along +x.
- Does `settle_tol` explain the extra drops with a slow gripper?
- Which of the two multiccd pipelines is active here, and how much does it slow simulation?

### Next

Step 2: SAC + HER. First a positive control on `FetchPickAndPlace-v4` to validate the training
code, then the original Niryo Lift, then `NiryoLiftFixed-v0`.

---

## 2026-11-19 — First RL results: SAC + HER fails, SAC + HER + scripted demos solves Lift

### Scope decision

- The project stays on **Niryo Lift** (`NiryoLiftFixed-v0`). PickAndPlace only if time remains.
- Fetch is used for one thing only: a smoke test of the training code (FetchReach). The
  FetchPickAndPlace positive control planned on 11-17 was dropped: the zoo recipe (TQC,
  512×512×512, batch 2048, 1M steps) measured ~4 steps/s on a 2-core CPU machine.
- The "with vs without HER" ablation was dropped. Consequence: the README must not claim that
  HER is necessary.
- Experiments:

| ID | Experiment | Question |
|---|---|---|
| E1 | SAC + HER, sparse reward | Can it learn Lift at all? |
| E1b | E1 + scripted demonstrations in the replay buffer | Is exploration the bottleneck? |
| E2 | SAC + HER, dense reward | H1: does a grasp bonus cause a local optimum? |
| E3 | PPO, sparse reward | On-policy comparison (PPO cannot use HER) |

### Prediction written before running E1

HER relabels goals with positions the cube actually reached. Early in training the cube
barely moves, so relabelled goals are almost all on the floor, while every real goal is
~48.5 cm in the air. Prediction: SAC + HER on `NiryoLiftFixed-v0` may learn very slowly or
not at all.

### Smoke test: FetchReach (training pipeline check)

FetchReach = Fetch robot moves its gripper to a point; no object, gripper locked. It only
checks that our SAC + HER code and hyperparameters work.

- SB3 default hyperparameters: 0% after 5k steps, with and without HER.
- rl-baselines3-zoo hyperparameters (`gamma=0.95`, `learning_rate=1e-3`, `batch_size=256`,
  `net_arch=[64, 64]`): with HER, success jumps from ~0% to 100% around 10k steps
  (cloud: 11k; Mac: 10–12k). Without HER: ≤30% at single checkpoints, 0% at 20k (1 seed).
- On the Mac, one checkpoint dropped from 100% to 10% and recovered at the next evaluation:
  never judge a run by a single checkpoint.
- Results differ slightly between Mac and Linux for the same seed (PyTorch floating point);
  conclusions must come from several seeds, not bit-exact reproduction.
- Speed with small networks: Mac ≈ 340 steps/s, cloud 2-core VM ≈ 95 steps/s.

### Training pipeline

- `scripts/train.py <config> --seed N` → `runs/<experiment>/seed<N>/` with `config.yaml`,
  `run_info.json` (git commit + dirty flag of both repos, package versions), `eval.csv`,
  `model.zip`, `tb/`.
- Evaluation protocol: separate env, deterministic policy, 20 episodes with seeds
  10000–10019 (the same episodes at every checkpoint), success read at the **last** step,
  plus mean final cube–goal distance (shows progress while success is still 0).
- Raw simulation speed (random actions, cloud VM): `NiryoLiftFixed-v0` ≈ 1700 steps/s,
  `FetchPickAndPlace-v4` ≈ 790 steps/s. Training speed is limited by network updates.

### E1: SAC + HER, sparse reward — 0%

Config `configs/niryo_lift_fixed_sac_her.yaml`: FetchReach hyperparameters but
`net_arch=[256, 256]`, 300k steps, evaluation every 10k steps. Mac, seed 0, 1657 s.

- Success 0% at all 31 evaluations; final distance **exactly 48.5 cm** every time — in
  evaluation the policy never touches the cube.
- Exploration measurement (random actions, 200 episodes on `NiryoLiftFixed-v0`): the cube
  moved more than 1 cm in 8 episodes (4%), was lifted more than 2 cm in 1 episode.
- Likely mechanism (not verified directly): relabelled goals ≈ the cube's start position, for
  which "do not touch the cube" is rewarded; for the real goal everything gets −1. So the
  learned behaviour is to stay away from the cube.
- The prediction above was confirmed.

Mistake on the way: the first "300k" run actually used the old 20k config because the edit
went into `fetch_reach_sac_her.yaml` instead of the Niryo config. The identical output to the
previous run (same seed, same config) gave it away.

### Idea considered and rejected: Reach → Pick → Lift transfer

Training Reach first and loading the model for Lift does not work here: NiryoReach has a
10-dim observation and 3-dim action, Lift 25 and 4, so the networks do not fit; in Reach the
goal means "where the gripper goes", in Lift "where the cube goes"; and Reach never trains
grasping, which is the actual bottleneck.

### E1b: E1 + scripted demonstrations — 100%

- `scripts/generate_demos.py --episodes 100`: scripted controller on `NiryoLiftFixed-v0`,
  seeds 100000–100099, only successful episodes kept → **96 episodes, 10 560 transitions**
  in `data/demos_100.npz`.
- In the demos, 62% of transitions have the cube more than 2 cm above the floor. A sampled
  HER batch from a buffer holding only the demos: 52% of rewards are 0, 92% of goals are in
  the air.
- Demos are added to the replay buffer before training and do not count as env steps.
  Config `configs/niryo_lift_fixed_sac_her_demos.yaml` = E1 config + a `demos` block only.

Result (Mac, seed 0, 300k steps, 1799 s):

| Steps | E1 success / distance | E1b success / distance |
|---|---|---|
| 30k | 0% / 48.5 cm | 0% / 49.1 cm |
| 40k | 0% / 48.5 cm | 5% / 46.8 cm |
| 50k | 0% / 48.5 cm | 70% / 14.4 cm |
| 90k | 0% / 48.5 cm | 100% / 1.7 cm |
| 300k | 0% / 48.5 cm | 100% / 2.0 cm |

From 80k to 300k E1b stays at 95–100% (21 evaluations), final distance 1.5–5.0 cm.
A shorter cloud run (seed 0, 40k steps) reached 25% at 40k.

**Interpretation:** with the same algorithm and hyperparameters, adding experience in which
the cube leaves the floor turns 0% into 100%. On this task SAC + HER fails because of
exploration, not because the problem cannot be learned.

**Caveats:**
- n = 1 seed for both E1 and E1b.
- The demos come from a controller that is already 100% successful; how much of the policy
  is imitation versus improvement is open.

### Tools added

- `scripts/watch_policy.py <run_dir>`: live viewer (keys: S slower, F faster, Space pause,
  → single step) or `--gif <path> --speed 0.25`.
- `scripts/plot_scripted_trajectory.py`: three-panel plot of one scripted episode (height,
  horizontal alignment, finger position). Observation: during APPROACH the fingers oscillate
  between 0 and 7 mm before opening fully although the command is constantly "open";
  suspected inertial effect of the light fingers (armature 0.1) during fast arm motion,
  not verified. The `finger > 7 mm` condition before DESCEND prevents it from mattering.

### Next

1. E1 and E1b with 5 seeds each (Mac overnight, ~30 min per run).
2. README: diagnosis, ablation table, E1 vs E1b curves (mean and confidence interval), GIF.
3. E2 (dense reward, H1) and E3 (PPO).
