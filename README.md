# MuJoCo Pro7 + LinkerHand L20: reaching, grasp and pick & place

Train PPO agents to move a **Rokae xMate Pro7** arm so its end effector reaches
randomly placed targets, and drive the same cell through a vision-based grasp
and a full pick & place episode.  Physics run in MuJoCo, RL is
Stable-Baselines3 PPO.  Every model is built from the **real URDF STL meshes**
(`assets/meshes/xMatePro7/`), and four environment names are registered:

- `pro7_urdf` — the reach pipeline on the **real URDF STL meshes**.  The STL
  files ship inside the repository, so no external checkout is needed.
- `pro7_pick` — **real-URDF-mesh Pro7 + LinkerHand L20 dexterous hand +
  eye-in-hand RGB-D camera** on a two-bench cell: the source bench carries a red cube plus
  green/blue/yellow distractor blocks, the destination bench carries a drop-off
  pad.  The camera localises the red cube, so the vision pipeline has to pick
  the right block out of the clutter.  The same scene drives both the plain
  grasp task and **pick & place** (see `grasp/pick_place_demo.py`).
- `pro7_pick_place` — the **whole cell as an RL task**: approach, grasp, lift,
  carry to the destination pad and release.  The episode does not stop at the
  grasp, so a policy trained here is scored on *both* pick and place.  Because
  a friction grasp cannot survive a learned torque policy's noise (measured with
  the previous gripper), the quasi-static carry itself is executed by the
  environment's transport servo while the policy owns the approach, the grasp
  and the release.
- `pro7_pick_urdf` — kept as an **alias** of `pro7_pick` (the grasp scene has
  used the real meshes since the switch), so both names build the same model.

> 当前状态、结果与产物汇总见 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**；
> 架构、数据流与每个文件的详细职责见 **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**。

## Files

The root keeps the four entry points plus the three shared modules every script
imports; the two pipelines live in packages, so the checkout root stays short.
`tools/` regenerates what is checked in under `assets/`, `grasp/` is the whole
vision grasp / pick & place stack.

```
paths.py         project root / assets / results paths + checkpoint resolution
cli.py           shared --env / --model / --out argparse helpers, policy loading
live_viewer.py   throttled native MuJoCo window for live training / demos
assets/*.xml     MuJoCo models (real-URDF Pro7 reach + pick / pick & place)

# --- entry points (run from the checkout root) ---
train_ppo.py               PPO training entry point
eval_rollout.py            evaluate policy + render videos / learning curve
viewer_demo.py             live MuJoCo visualization window (trained policy)
make_montage.py            render an approach->reach montage

# --- environments (Gymnasium) ---
env/__init__.py            make_env() registry used by every script
env/base_reacher.py        shared reacher task: obs / action / reward / reset
env/rokae_reacher.py       Pro7 7-DOF reach on the real URDF meshes (obs 27, act 7)
env/rokae_pro7_pick.py     Pro7 vision-driven grasp (obs 30, act 8)
env/rokae_pro7_pick_place.py  Pro7 pick & place, whole episode (obs 38, act 8)
env/dual_arm_reacher.py    dual-arm tasks: independent reach + cooperative bar (obs 60, act 14)

# --- grasp / pick pipeline (camera + L20 hand + red cube) ---
grasp/common.py            grasp scene constants + distractors + expert servo
grasp/detect.py            red segmentation + depth -> cube 3D world coords
grasp/policy.py            the MLP grasp policy (train + replay consistently)
grasp/demo.py              detect -> 3D localise -> task-space servo -> grasp
grasp/pick_place_demo.py   grasp on the source bench -> carry -> place on pad
grasp/montage.py           approach -> grasp still frames
grasp/overlay.py           visualise red segmentation + 3D localisation
grasp/visualize.py         grasp rollout GIF (expert or learned policy)
grasp/view.py              live MuJoCo window: watch the grasp + detection marker
grasp/supervised.py        real-time DAgger / behaviour-cloning grasp training
grasp/train_live.py        real-time training-process GIF (pick & place by default)

# --- model / asset tooling (regenerates what is checked in under assets/) ---
tools/build_dual_arm_model.py  lkwy73_o1 description -> assets/dual_arm_reach.xml (+ meshes)
tools/convert_arm_urdf.py      MuJoCo Pro7 cell -> URDF (mesh + inertial, for rviz2)
tools/convert_hand_urdf.py     vendor L20 hand URDF -> MuJoCo fragments (+ grasp calibration)
tools/make_wrist_flange.py     Pro7 flange -> L20 adapter, as a revolved STL
tools/urdf_selfcheck.py        re-loads that URDF in MuJoCo and diffs it (--check)

results/                   saved model, logs, videos, plots
tests/                     pytest smoke + regression suite
ros2_ws/                   ROS 2 workspace: Pro7 pick & place node + client
```

Both styles work for every script below: `python3 grasp/demo.py` and
`python3 -m grasp.demo`; the same holds for `tools/*`.  `paths.py` stays at the
root on purpose - it is the single place that computes the checkout root, and
everything (including the ROS 2 node) finds the project through it.

Everything that needs the repository root, the MuJoCo assets or the results
directory goes through `paths.py`, and every script exposes the same
`--env` / `--model` / `--out` flags through `cli.py`.  That keeps the scripts
short and means the checkout can be moved (or the scripts run from any working
directory) without edits.

## Test

```bash
python3 -m pytest tests -q      # ~2 s: env contracts, obs layout, paths, grasp
```

The suite pins the environment observation layouts and the reacher/pick
contracts.  Observation sizes and order are *checkpoint ABI*: changing them
invalidates the trained `results/ppo_*.zip` files, and the tests are there to
catch that.

## ROS 2 node

`ros2_ws/` wraps the whole pick & place cell in one ROS 2 node
(`pro7_pick_place_ros`): it publishes the 7 arm + 21 hand joint states, the TF
chain, the eye-in-hand RGB-D frames and the detected cube, offers a `reset`
service and runs one full cycle through a `PickPlace` action.  Nothing is
re-implemented: the node drives `grasp.common.iter_pick_and_place` (the
step-wise form of the same scripted expert behind `grasp/pick_place_demo.py`) on a
plant thread that owns the MuJoCo renderers.

```bash
./start.sh          # one click: build (if needed) + every node (node + rviz2)
./start.sh --demo   # ...and run one pick & place by itself
./start.sh --no-rviz   # headless machine / server
./start.sh --stop      # stop it again

cd ros2_ws && ./run.sh test    # 13 pytest cases (scene, expert, action end-to-end)
```

`pro7_pick_place.desktop` is the same thing for a double-click
(`cp pro7_pick_place.desktop ~/.local/share/applications/`).  See
[ros2_ws/README.md](ros2_ws/README.md) for the topic/service/action reference,
the parameters and the design notes.

Nothing in rviz?  The launch loads `config/pick_place.rviz` for you (Fixed Frame
`world`, TF + markers + camera).  If you start `rviz2` by hand, pass the same
file with `-d $(ros2 pkg prefix pro7_pick_place_ros)/share/pro7_pick_place_ros/config/pick_place.rviz`
and make sure `ROS_DOMAIN_ID` matches the node's.

## Model

- The arm is the **real-URDF Pro7**: `assets/rokae_xmate_pro7_real.xml` hangs
  the official `xMatePro7_*.stl` meshes off the seven `hinge` joints
  (`z,y,z,y,z,y,z`), with the URDF's axes, offsets and limits.
- The target is a kinematic `mocap` sphere with contacts disabled, so it never
  collides with the arm.
- Action: torque on each of the seven joints, in `[-1, 1]`.
- Observation (27-d): `[cos q(7), sin q(7), dq(7), tool−target(3), target(3)]`.
- Reward: `-dist(tool, target) - 0.0005 * ||qvel||^2 + 0.5*exp(-dist/0.3)` plus
  a `+2` bonus on success (`dist < 0.15`). Episode ends on success or after
  `max_steps=250`.

## Dual-arm model (`dual_arm_reach.xml`)

The same conventions applied to the **lkwy73_o1** dual-arm description: two
7-DOF arms and two 11-DOF dexterous hands sharing one base, 11.1 kg.  The asset
is generated, not hand-edited - `tools/build_dual_arm_model.py` reads the cleaned
description, writes `assets/dual_arm_reach.xml` and copies the decimated meshes
into `assets/dual_arm/meshes/` (68.7 MB -> 8.4 MB, max 20k faces per part so
MuJoCo's 200k STL limit is respected).

What the generator adds:

- 14 normalised torque motors, 7 per arm (`ctrl` in `[-1, 1]`, gears
  6/6/5/4/3/2/2 Nm, sized for a 4 kg arm);
- the 22 finger joints **pinned to their open pose** by equality constraints, so
  a reach task does not have to model the hands (remove the `<equality>` block
  and add actuators to drive them instead);
- a `tip_left` / `tip_right` site at the centre of the four fingertips of each
  hand, plus one `mocap` target per arm;
- the four link pairs that overlap in the CAD (`L1`/`base`, `L5`/`L7`, mirrored)
  excluded, because those contacts would otherwise push the arms apart in every
  pose.

Two environments use it (obs 60, act 14):

| env | task |
|---|---|
| `dual_arm_reach` | each arm reaches its own random target - the multi-arm control baseline |
| `dual_arm_coop` | the two targets are the ends of one rigid bar, so the arms must also hold a matching relative pose |

The cooperative targets are built from one left-arm pose plus its mirror on the
right arm, then moved by a small rigid transform: the pair is therefore a
configuration the two arms really can hold. Reward per step is
`-mean(dist) + 0.5*exp(-max_dist/0.3) - 0.0005*||dq||^2`, minus
`2.0*bar_error` for the cooperative task, plus `+2` when both arms are inside
0.08 m. Resets are redrawn until both arms start at least 0.12 m from their
targets, so no episode opens with a free success.

```bash
python3 tools/build_dual_arm_model.py --check           # validate the asset
python3 train_ppo.py --env dual_arm_reach --steps 2000000 --n-envs 8
python3 train_ppo.py --env dual_arm_coop  --steps 1200000 --n-envs 8
python3 eval_rollout.py --env dual_arm_reach --episodes 200
python3 viewer_demo.py --env dual_arm_coop        # live window, T = new target
```

Both tasks are trained and checked in: `dual_arm_reach` ends **43.3 %** of
episodes with both arms arriving (64 % / 71 % per arm, deterministic, 150
episodes, 2.6M steps) and `dual_arm_coop` holds the bar in **21.3 %** (42.4 %
with the exploration noise seen during training - success there is an
instantaneous condition, see PROJECT_SUMMARY.md §11).

The source description and the derivation of the mirror signs, the CAD
self-contacts and the hand inertia asymmetry are documented in
`/home/wj/urdf/lkwy73_o1_dual_arm_clean/README.md`.

## Train

```bash
python3 train_ppo.py --steps 2000000             # Pro7 reach, fresh run
python3 train_ppo.py --steps 2000000 \
  --init-model results/ppo_pro7_urdf.zip         # continue from checkpoint
python3 train_ppo.py --n-envs 8 --device cuda    # tuning knobs
python3 train_ppo.py --steps 2000000 --viewer    # watch it train live
python3 train_ppo.py --env pro7_pick --steps 1000000   # learn to grasp (vision-driven)
```

`--viewer` opens the native MuJoCo window on one of the vectorised environments
and updates it while training runs, with step / fps / reward / episode-length
drawn in the window corners:

```bash
python3 train_ppo.py --viewer --viewer-env 2 --viewer-fps 30
python3 grasp/train_live.py --rounds 24 --live     # supervised grasp training, live
```

The window is throttled (`--viewer-fps`), so it never slows training down; close
it and training simply continues headless.

## Evaluate / visualize

```bash
python3 eval_rollout.py --env <env> --episodes 400    # metrics + curves + video
python3 make_montage.py --env <env>                  # approach->reach still frames
```

## Live MuJoCo viewer

Open the **native MuJoCo Simulate window** and watch the trained arm reach
random targets in real time:

```bash
python3 viewer_demo.py                         # Pro7 reach (default env)
# or from the eval script:
python3 eval_rollout.py --viewer

# ---- detect a red cube, localise it in 3D, and grasp it ----
python3 grasp/demo.py --episodes 10 --video   # 10/10 grasp success
python3 grasp/pick_place_demo.py --episodes 10 --video --montage   # 10/10 pick & place
python3 grasp/montage.py
python3 grasp/supervised.py --rounds 24       # real-time supervised (DAgger) training
python3 train_ppo.py --env pro7_urdf --steps 2000000   # reach with real URDF meshes
python3 train_ppo.py --env pro7_pick_urdf --steps 800000  # grasp with real URDF meshes
# --- visualize the grasp & training ---
python3 grasp/overlay.py                          # detection -> 3D overlay PNG
python3 grasp/visualize.py --mode expert           # grasp rollout GIF
python3 grasp/view.py --mode expert                 # live MuJoCo window (expert)
python3 grasp/train_live.py --rounds 24                  # pick & place: watch it train (GIF)
python3 grasp/train_live.py --rounds 24 --live           # + the real-time MuJoCo window
python3 grasp/train_live.py --rounds 24 --live --live-speed 0   # window, but train full speed
python3 grasp/train_live.py --task grasp --rounds 24 --live     # the grasp-only task
python3 grasp/view.py --task pick_place --mode policy      # replay what was learned
```

* Close the window (or press `ESC`) to stop.
* Press `T` while running to immediately jump to a new target.
* Use the mouse to orbit/zoom; the camera is fixed to the model's own view.

The same window is available *during* training (`--viewer`, see above) and for
the supervised grasp trainer (`grasp/train_live.py --live`); all three entry points
share `live_viewer.py`, which owns the camera, the on-screen status block and the
safe shutdown.

## URDF export (rviz2)

The ROS 2 side of the project draws the cell with **rviz2**, and rviz places a
robot's links with TF lookups only -- it never looks inside a MuJoCo model.  So
`tools/convert_arm_urdf.py` re-emits the simulated cell as URDF:

```bash
python3 tools/convert_arm_urdf.py --check        # -> assets/rokae_xmate_pro7_pick_real.urdf
python3 tools/convert_arm_urdf.py --with-cell    # + benches / drop pad as world props
python3 tools/convert_arm_urdf.py --hand articulated   # 22-link hand instead of baked
```

Everything comes from the *compiled* model (poses, axes, limits, meshes,
inertia), and `--check` proves it: MuJoCo re-loads the URDF and every geom has
to land within 1e-10 m of the MJCF, at the zero pose and at random joint poses,
with the same total mass.  Link and joint names match the ROS node's TF chain
(`base` → `link1`…`link7` → `gripper`) and its `joint_states`, so the display
is driven by the node itself: it reads `assets/<model>.urdf` and latches it on
`/robot_description` (override with the node's `urdf_path` parameter), and the
shipped `pick_place.rviz` already carries a RobotModel display for it.  See
`ros2_ws/README.md` §8 for the recipe.

`grasp/train_live.py` trains on **pick & place by default** (episode = approach,
pinch, lift, carry, release on the pad; the metrics panel shows pick % *and*
place %).  `--task grasp` runs the old grasp-only task instead.

`--live` plays the run back at real time (`--live-speed` scales it: `0.5` is
slow motion, `0` runs as fast as the CPU allows).  The pick & place carry is
quasi-static (~2 cm/s) and an episode is ~2500 steps, so at 1× one round is
~50 s - `--rounds 24 --live` is a ~20 minute run; use `--live-speed 4` or fewer
rounds if you want it quicker, or `--task grasp --live` for the short (≈1 s per
episode) grasp task, which defaults to 0.5× slow motion.  Evaluation rollouts
keep publishing frames, so the window keeps moving instead of freezing between
rounds.  Closing the window never kills the
run - the trainer says so and carries on headless.

On this machine the GL context is created through the software (llvmpipe)
renderer (`LIBGL_ALWAYS_SOFTWARE=1`), so the window opens reliably; on a normal
GPU desktop you can drop that env var.  The viewer deliberately
`close()`s and waits briefly for its UI thread before exiting to avoid a
GLFW-teardown segfault.

## Final results

| metric                        | pro7_urdf (Pro7, real meshes) |
|-------------------------------|-------------------------------|
| episode success rate          | **84.2 %**                    |
| final dist mean / median      | 0.147 / 0.147 m               |
| mean steps to reach           | ~61 (of 250)                  |
| targets                       | FK-derived (always reachable) |

Trained for 2.0 M timesteps (`results/ppo_pro7_urdf.zip`, ~8.4 min at
~4000 steps/s on an RTX 5080); curves, rollout video and montage live in
`results/pro7_urdf/`.

### End effector: LinkerHand L20

The wrist carries the **LinkerHand L20** (灵心巧手) instead of the old pinch
gripper.  The vendor ships it as a SolidWorks URDF; `tools/convert_hand_urdf.py`
re-emits it as MuJoCo fragments that the scenes `<include>`
(`assets/linkerhand_l20/`), because MuJoCo merges a URDF's root link into the
world and the hand then cannot be bolted onto the flange.  The converter also
calibrates the grasp: it closes the hand on a free cube of the scene's size,
mass and friction and keeps the grasp centre + closure scale that hold it.

The hand's 21 joints are driven as **one open/close synergy**, so the action
space is still "7 arm torques + 1 grip command" and existing checkpoints keep
their shape.  `grasp.common.hand_closure` reads the closure back and
`is_grasped` requires at least `GRASP_FINGERS` distinct fingers on the cube.

The hand is bolted 60 mm in front of the wrist, which is where the real adapter
sits, so the bare cell showed the arm and the hand floating apart.  That
adapter is now modelled: `tools/make_wrist_flange.py` revolves a plate → relieved
waist → plate profile sized from the two faces it joins (the wrist's 30 mm
annulus and the hand base's 20–32 mm ring), and
the scenes mount it as a **visual-only, massless** geom (`contype=0`,
`density=0`): the pick / pick & place cell (and therefore the URDF rviz draws),
and the real-mesh Pro7 **reach training** model (`pro7_urdf`).  Deliberate: the calibrated grasp,
the eye-in-hand camera image, the reach task and every rollout stay
bit-identical to the model without it — verified by re-running 400 control
steps with and without the part (max |Δqpos| = 0), and by diffing the eye-in-hand
frame (0 of 76800 camera pixels differ) — while MuJoCo and rviz both show a
continuous wrist.

Two vendor-model details the simulation needs: the URDF reuses its *visual*
meshes for collision (34 self-contacts, 10 mm deep, which rang the fingers at
90 rad/s), so the hand collides only with the graspable blocks; and the
phalanges carry almost no inertia, so the position servos are calibrated for the
scene's 20 ms timestep.

`pro7_pick` (grasp): the scene is the **real-URDF Pro7** and the bench carries
green/blue/yellow distractor blocks next to the red target (inside the hand
camera's view, outside the target's sampling band).  The eye-in-hand RGB-D camera
localises the red cube to **~5 mm** (max ~1.2 cm) and ignores the clutter
(100 % detection over 40 random positions), and a resolved-rate controller servos
the 7-DOF arm onto it and closes the hand
(`grasp/demo.py`, `results/pro7_pick/grasp_demo.gif`).

**Pick & place** (`grasp/pick_place_demo.py`) uses the same vision + expert to carry
the cube from the source bench to the destination bench's pad: **10/10 episodes**,
mean drop error 29 mm (`results/pro7_pick_place/`).  The carry is deliberately
quasi-static — the expert moves in 1 cm increments with soft gains; anything
faster lets the fingers be pushed open and the cube drops.  A PPO env is
registered for vision-driven grasping.  The L20 grips the cube reliably
(`tests/test_grasp.py` pins the grasp), but the grip is still marginal while the
arm accelerates, so the carry is the part that still needs tuning.
`pro7_pick_place` is the same cell as a *whole-episode* RL task: the expert
labels every step of the plan (`PlacePlanner`: settle → lift → carry → lower),
and `grasp/train_live.py` trains the policy on it by DAgger - the replay shows the arm
approaching, pinching, carrying and releasing, reaching **100 % pick / 70 %
place** in the best of 24 rounds (`results/pro7_pick_place/training_process.gif`;
the per-round numbers swing a lot, so the best checkpoint is the one that is
kept).
The one thing the learned policy cannot do is the *carry itself*: the transport
torque is ~1-2 % of the actuator range, i.e. far below the action noise an MLP
imitates, so a learned friction grasp always squeezes the cube out (measured:
behaviour cloning with 2e-4 action MSE places 0/6).  The environment therefore
keeps the jaws' grip centre on the plan's waypoint with the gentle transport
gains while a cube is held; the policy still owns the approach, the pinch, the
retries and the release, which is what pick % / place % score.
`grasp/supervised.py` adds **real-time supervised learning**: the servo acts as an
expert teacher and an online policy is trained by behaviour cloning / DAgger while
you watch train MSE and grasp success streaming in (MSE ~0.01, success up to ~40 %).
Visualisation: `grasp/overlay.py` (red cube → 3D point), `grasp/visualize.py`
(grasp rollout GIF) and `grasp/view.py` (live MuJoCo window with a detection marker).
`grasp/train_live.py` records the *training process itself*: per evaluation round it
composites the current policy's grasp storyboard with a live metric panel
(train MSE / grasp % / distance) into `training_process.gif`, so you watch the
policy improve in real time (optionally `--live` also opens the MuJoCo window).
Real-URDF geometry: `pro7_urdf` / `pro7_pick` (and its `pro7_pick_urdf` alias)
all use the official Pro7 STL meshes bundled in `assets/meshes/xMatePro7/`.
`pro7_urdf` is trained to **84.2 % success**, final dist median 0.147 m in
~61 steps of 250 (`results/ppo_pro7_urdf.zip`, `results/pro7_urdf/`).  Targets
are guaranteed reachable (they come from forward-kinematics of a random pose),
and the tool tip is off the last roll axis so all seven joints contribute.

Artifacts per environment live under `results/<env>/`: `rollout.mp4`,
`rollout.gif`, `montage.png`, `learning_curve.png`, plus the model
`results/ppo_<env>.zip`.

Model names default to `results/ppo_<env>`. If you pass a directory by mistake
(e.g. `--model results`), the scripts now print a clear error instead of the
cryptic `IsADirectoryError`.
