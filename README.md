# MuJoCo reachers with PPO (2-joint planar & 3-DOF spatial)

Train PPO agents to move robot arms so their end-effector reaches randomly
placed targets. Physics run in MuJoCo, RL is Stable-Baselines3 PPO. Two
environments are provided:

- `two_joint` — planar 2-DOF arm reaching a 2D target.
- `three_joint` — 3-DOF spatial arm (yaw + shoulder pitch + elbow pitch)
  reaching a 3D target.
- `six_joint` — **Rokae xMate ER3** 6-DOF industrial arm reaching a 3D target.
  Its joint axes / offsets / limits come from the real Rokae URDF.
- `pro7_joint` — **Rokae xMate Pro7** 7-DOF collaborative arm reaching a 3D
  target.  Joint axes (`z,y,z,y,z,y,z`), offsets and limits come from the real
  Rokae Pro7 URDF (from the `rokae_ros2` description package).
- `pro7_pick` — **real-URDF-mesh Pro7 + LinkerHand L20 dexterous hand +
  eye-in-hand RGB-D camera** on a two-bench cell: the source bench carries a red cube plus
  green/blue/yellow distractor blocks, the destination bench carries a drop-off
  pad.  The camera localises the red cube, so the vision pipeline has to pick
  the right block out of the clutter.  The same scene drives both the plain
  grasp task and **pick & place** (see `pick_place_demo.py`).
- `pro7_pick_place` — the **whole cell as an RL task**: approach, grasp, lift,
  carry to the destination pad and release.  The episode does not stop at the
  grasp, so a policy trained here is scored on *both* pick and place.  Because
  a friction grasp cannot survive a learned torque policy's noise (measured with
  the previous gripper), the quasi-static carry itself is executed by the
  environment's transport servo while the policy owns the approach, the grasp
  and the release.
- `pro7_urdf` — the reach pipeline on the **real URDF STL meshes**.  MuJoCo 3.x
  natively parses URDF; we fix the (bogus) inertia, add the actuators and use
  the official meshes.  The STL files ship inside the repository
  (`assets/meshes/xMatePro7/`), so no external checkout is needed.
  `pro7_pick_urdf` is kept as an **alias** of `pro7_pick` (the grasp scene has
  used the real meshes since the switch), so both names build the same model.

> 当前状态、结果与产物汇总见 **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**；
> 架构、数据流与每个文件的详细职责见 **[PROJECT_GUIDE.md](PROJECT_GUIDE.md)**。

## Files

```
paths.py         project root / assets / results paths + checkpoint resolution
cli.py           shared --env / --model / --out argparse helpers, policy loading
live_viewer.py   throttled native MuJoCo window for live training / demos
assets/*.xml     MuJoCo models (2-joint planar, 3-joint spatial, Rokae arms)

# --- environments (Gymnasium) ---
env/__init__.py            make_env() registry used by every script
env/base_reacher.py        shared reacher task: obs / action / reward / reset
env/two_joint_reacher.py   2-DOF planar reach (obs 10, act 2)
env/three_joint_reacher.py 3-DOF spatial reach (obs 15, act 3)
env/rokae_reacher.py       N-DOF Rokae reach: ER3, Pro7, Pro7-real-URDF
env/rokae_pro7_pick.py     Pro7 vision-driven grasp (obs 30, act 8)
env/rokae_pro7_pick_place.py  Pro7 pick & place, whole episode (obs 38, act 8)

# --- reach pipeline ---
train_ppo.py               PPO training entry point
eval_rollout.py            evaluate policy + render videos / learning curve
viewer_demo.py             live MuJoCo visualization window (trained policy)
make_montage.py            render an approach->reach montage
ik_probe.py  ik_probe3d.py analytic IK + PD feasibility baselines

# --- grasp / pick pipeline (camera + gripper + red cube) ---
grasp_common.py            grasp scene constants + distractors + expert servo
grasp_policy.py            the MLP grasp policy (train + replay consistently)
convert_hand_urdf.py       vendor L20 hand URDF -> MuJoCo fragments (+ grasp calibration)
detect_red_cube.py         red segmentation + depth -> cube 3D world coords
grasp_demo.py              detect -> 3D localise -> task-space servo -> grasp
pick_place_demo.py         grasp on the source bench -> carry -> place on pad
make_grasp_montage.py      approach -> grasp still frames
supervised_grasp.py        real-time DAgger / behaviour-cloning grasp training
detect_overlay.py          visualise red segmentation + 3D localisation
visualize_grasp.py         grasp rollout GIF (expert or learned policy)
view_pick.py               live MuJoCo window: watch the grasp + detection marker
train_live.py              real-time training-process GIF (pick & place by default)

results/                   saved model, logs, videos, plots
tests/                     pytest smoke + regression suite
ros2_ws/                   ROS 2 workspace: Pro7 pick & place node + client
```

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
re-implemented: the node drives `grasp_common.iter_pick_and_place` (the
step-wise form of the same scripted expert behind `pick_place_demo.py`) on a
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

- `link1` (red) and `link2` (blue) are capsules with revolute `hinge` joints
  about the z-axis; they stay in the xy-plane.
- The target is a kinematic `mocap` sphere with contacts disabled, so it never
  collides with the arm.
- The floor is placed below the arm so the shoulder/elbow balls never lock the
  base in a contact constraint (a subtle but critical detail).
- Action: torque on joint1 / joint2, each in `[-1, 1]`.
- Observation (10-d): `[cos q1, sin q1, cos q2, sin q2, dq1, dq2, tip−target, target]`.
- Reward: `-dist(tip, target) - 0.0005 * ||qvel||^2` plus `+2` bonus on success
  (`dist < 0.08`). Episode ends on success or after `max_steps=120`.

## Train

```bash
python3 train_ppo.py --steps 600000            # fresh run (~2.5 min on GPU)
python3 train_ppo.py --steps 600000 \
  --init-model results/ppo_two_joint.zip       # continue from checkpoint
python3 train_ppo.py --env three_joint --steps 600000   # train the 3-DOF arm
python3 train_ppo.py --env six_joint --steps 1000000    # train the Rokae 6-DOF arm
python3 train_ppo.py --env pro7_joint --steps 2000000   # train the Rokae Pro7 7-DOF arm
python3 train_ppo.py --env pro7_joint --n-envs 8 --device cuda   # tuning knobs
python3 train_ppo.py --env pro7_urdf --steps 2000000 --viewer    # watch it train live
```

`--viewer` opens the native MuJoCo window on one of the vectorised environments
and updates it while training runs, with step / fps / reward / episode-length
drawn in the window corners:

```bash
python3 train_ppo.py --env three_joint --steps 600000 --viewer
python3 train_ppo.py --env pro7_urdf --viewer --viewer-env 2 --viewer-fps 30
python3 train_live.py --rounds 24 --live     # supervised grasp training, live
```

The window is throttled (`--viewer-fps`), so it never slows training down; close
it and training simply continues headless.

## Evaluate / visualize

```bash
python3 eval_rollout.py --env <env> --episodes 400    # metrics + curves + video
python3 ik_probe.py / ik_probe3d.py                  # analytic baselines
python3 make_montage.py --env <env>                  # approach->reach still frames
```

## Live MuJoCo viewer

Open the **native MuJoCo Simulate window** and watch the trained arm reach
random targets in real time:

```bash
python3 viewer_demo.py                         # 2-joint arm (default)
# or from the eval script:
python3 eval_rollout.py --viewer
# 3-DOF arm:
python3 viewer_demo.py --env three_joint
python3 viewer_demo.py --env pro7_joint

# ---- detect a red cube, localise it in 3D, and grasp it ----
python3 grasp_demo.py --episodes 10 --video   # 10/10 grasp success
python3 pick_place_demo.py --episodes 10 --video --montage   # 10/10 pick & place
python3 make_grasp_montage.py
python3 supervised_grasp.py --rounds 24       # real-time supervised (DAgger) training
python3 train_ppo.py --env pro7_urdf --steps 2000000   # reach with real URDF meshes
python3 train_ppo.py --env pro7_pick_urdf --steps 800000  # grasp with real URDF meshes
# --- visualize the grasp & training ---
python3 detect_overlay.py                          # detection -> 3D overlay PNG
python3 visualize_grasp.py --mode expert           # grasp rollout GIF
python3 view_pick.py --mode expert                 # live MuJoCo window (expert)
python3 train_live.py --rounds 24                  # pick & place: watch it train (GIF)
python3 train_live.py --rounds 24 --live           # + the real-time MuJoCo window
python3 train_live.py --rounds 24 --live --live-speed 0   # window, but train full speed
python3 train_live.py --task grasp --rounds 24 --live     # the grasp-only task
python3 view_pick.py --task pick_place --mode policy      # replay what was learned
python3 train_ppo.py --env pro7_pick --steps 1000000   # learn to grasp (vision-driven)
```

* Close the window (or press `ESC`) to stop.
* Press `T` while running to immediately jump to a new target.
* Use the mouse to orbit/zoom; the camera is fixed to the `cam_xy` view.

The same window is available *during* training (`--viewer`, see above) and for
the supervised grasp trainer (`train_live.py --live`); all three entry points
share `live_viewer.py`, which owns the camera, the on-screen status block and the
safe shutdown.

`train_live.py` trains on **pick & place by default** (episode = approach,
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

| metric                        | two_joint (2D) | three_joint (3D) | six_joint (Rokae) |
|-------------------------------|----------------|------------------|-------------------|
| episode success rate          | **100 %**      | **98.5 %**       | **48 %**          |
| final dist mean / median      | 0.056 / 0.057  | 0.092 / 0.091    | 0.177 / 0.129     |
| mean steps to reach           | ~27 (of 120)   | ~31 (of 150)     | ~130 (of 200)     |
| analytic baseline             | 100 % (IK/PD)  | 100 % (IK/PD)    | FK-derived targets|

`pro7_joint` results (after ~2.2 M timesteps): **~89.5 % success**,
final dist median 0.147 m (success radius 0.15 m), ~61 steps (of 250) to reach.
The longer ~1.4 m Pro7 arm uses a slightly wider success circle and shorter
start→target distances than the ER3 to keep the 7-DOF reach learnable.

### End effector: LinkerHand L20

The wrist carries the **LinkerHand L20** (灵心巧手) instead of the old pinch
gripper.  The vendor ships it as a SolidWorks URDF; `convert_hand_urdf.py`
re-emits it as MuJoCo fragments that the scenes `<include>`
(`assets/linkerhand_l20/`), because MuJoCo merges a URDF's root link into the
world and the hand then cannot be bolted onto the flange.  The converter also
calibrates the grasp: it closes the hand on a free cube of the scene's size,
mass and friction and keeps the grasp centre + closure scale that hold it.

The hand's 21 joints are driven as **one open/close synergy**, so the action
space is still "7 arm torques + 1 grip command" and existing checkpoints keep
their shape.  `grasp_common.hand_closure` reads the closure back and
`is_grasped` requires at least `GRASP_FINGERS` distinct fingers on the cube.

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
(`grasp_demo.py`, `results/pro7_pick/grasp_demo.gif`).

**Pick & place** (`pick_place_demo.py`) uses the same vision + expert to carry
the cube from the source bench to the destination bench's pad: **10/10 episodes**,
mean drop error 29 mm (`results/pro7_pick_place/`).  The carry is deliberately
quasi-static — the expert moves in 1 cm increments with soft gains; anything
faster lets the fingers be pushed open and the cube drops.  A PPO env is
registered for vision-driven grasping.  The L20 grips the cube reliably
(`tests/test_grasp.py` pins the grasp), but the grip is still marginal while the
arm accelerates, so the carry is the part that still needs tuning.
`pro7_pick_place` is the same cell as a *whole-episode* RL task: the expert
labels every step of the plan (`PlacePlanner`: settle → lift → carry → lower),
and `train_live.py` trains the policy on it by DAgger - the replay shows the arm
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
`supervised_grasp.py` adds **real-time supervised learning**: the servo acts as an
expert teacher and an online policy is trained by behaviour cloning / DAgger while
you watch train MSE and grasp success streaming in (MSE ~0.01, success up to ~40 %).
Visualisation: `detect_overlay.py` (red cube → 3D point), `visualize_grasp.py`
(grasp rollout GIF) and `view_pick.py` (live MuJoCo window with a detection marker).
`train_live.py` records the *training process itself*: per evaluation round it
composites the current policy's grasp storyboard with a live metric panel
(train MSE / grasp % / distance) into `training_process.gif`, so you watch the
policy improve in real time (optionally `--live` also opens the MuJoCo window).
Real-URDF geometry: `pro7_urdf` / `pro7_pick_urdf` use the official Pro7 STL meshes
(bundled in `assets/meshes/xMatePro7/`), vs. the lower-cost capsule models.
`pro7_urdf` is trained to **84.2 % success**, final dist median 0.147 m in
~61 steps of 250 (`results/ppo_pro7_urdf.zip`, `results/pro7_urdf/`).

The Rokae 6-DOF task is much harder than the 1–3 DOF reachers: the policy
brings the tool tip from ~0.5 m to within ~13 cm of the target and reliably
reaches the 12 cm success radius ~half the time. Targets are guaranteed
reachable (they come from forward-kinematics of a random pose), and the tool
tip is off the last roll axis so all 6 joints contribute.

Artifacts per environment live under `results/<env>/`: `rollout.mp4`,
`rollout.gif`, `montage.png`, `learning_curve.png`, plus the model
`results/ppo_<env>.zip`.

Model names default to `results/ppo_<env>`. If you pass a directory by mistake
(e.g. `--model results`), the scripts now print a clear error instead of the
cryptic `IsADirectoryError`.
