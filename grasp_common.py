"""Single source of truth for the Pro7 "grasp the red cube" scene and expert.

The workbench height, cube size, start pose and grasp geometry are used by the
MuJoCo XML, the Gymnasium environment, the demo/montage scripts and the live
viewers.  The end effector is the LinkerHand L20 dexterous hand (see
``convert_hand_urdf.py``); its 21 joints are driven as a single open/close
synergy so the action space stays "7 arm torques + 1 grip command".  Keeping them here means tuning the scene is a one-line change instead
of a hunt for scattered magic numbers.

The scene is built on the **real URDF STL meshes** of the Pro7
(``rokae_xmate_pro7_pick_real.xml``) and carries green/blue/yellow distractor
blocks next to the red target cube, so the vision pipeline (and any learned
policy) has to pick the red one out of a cluttered bench.  A second bench
(``TABLE_2_POS``) with a flat drop-off pad (``PLACE_TARGET``) sits on the other
side of the base, so the same scene drives both "grasp" and "pick & place".

The resolved-rate *expert* lives here as well, so ``grasp_demo`` (scripted),
``supervised_grasp`` (DAgger teacher) and ``make_grasp_montage`` all drive the
arm with exactly the same control law.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import mujoco
import numpy as np

from detect_red_cube import DEFAULT_CUBE_SIDE, estimate_cube_world_rgbd, render_rgbd
from paths import asset_path

# --------------------------------------------------------------------------- #
# scene geometry
# --------------------------------------------------------------------------- #
#: The grasp scene is the real-URDF-mesh Pro7 (see README: real vs. capsule).
MODEL_PATH = asset_path("rokae_xmate_pro7_pick_real.xml")

TABLE_Z = 0.45  # workbench top (m)
CUBE_SIDE = DEFAULT_CUBE_SIDE  # cube edge length (m); owned by the detector
CUBE_Z = TABLE_Z + CUBE_SIDE / 2.0
CUBE_X = 0.78  # nominal cube x on the bench
CUBE_SPREAD = (0.12, 0.08)  # +-x / +-y sampling half-widths

#: Coloured blocks that clutter the bench.  ``(body name, (x, y))``; they sit in
#: the eye-in-hand camera's view but *outside* the red target's footprint
#: (x in CUBE_X +- CUBE_SPREAD[0], y in +- CUBE_SPREAD[1], plus half a cube), so
#: the detector sees them without ever confusing them for the target.  They
#: collide, so they are also physical obstacles the arm can bump into.
#: Positions are on the far side of the bench: at START_POSE that is where the
#: down-looking hand camera's (trapezoidal) footprint is widest.
DISTRACTORS = (
    ("distractor_green", (0.96, 0.15)),
    ("distractor_blue", (0.96, -0.15)),
    ("distractor_yellow", (1.00, 0.00)),
)

# Fixed "ready" pose: the gripper points down at the bench, so the eye-in-hand
# camera sees the cube from the first frame.
START_POSE = np.array([0.0, -0.3, 0.0, 1.65, 0.0, 0.9, 0.0])

N_ARM = 7  # actuated arm joints; the 8th actuator is the gripper

# --------------------------------------------------------------------------- #
# dexterous hand (LinkerHand L20, see convert_hand_urdf.py)
# --------------------------------------------------------------------------- #
#: ``open`` / ``close`` joint presets and the finger -> contact-geom grouping,
#: all generated from the vendor URDF alongside the model fragments.
with open(asset_path("linkerhand_l20/linkerhand_l20_poses.json"), encoding="utf-8") as _fh:
    HAND_POSES = json.load(_fh)
#: The hand's own frame is the tool frame: fingers run along +z, palm faces +x.
#: The 21 joints are driven as one open/close synergy, so the action space stays
#: "7 arm torques + 1 grip command" and existing checkpoints keep their shape.
HAND_JOINTS = tuple(sorted(HAND_POSES["open"]))
#: Effective jaw gap when the hand is wide open (m).  Only used to turn the
#: closure fraction into a length for the reward shaping below.
GRIP_OPEN = 0.09
#: A grasp needs this many distinct fingers on the cube, and this much closure.
GRASP_FINGERS = 2
GRASP_CLOSURE = 0.55
GRASP_GAP = GRIP_OPEN * (1.0 - GRASP_CLOSURE)

# --------------------------------------------------------------------------- #
# expert (resolved-rate) controller
# --------------------------------------------------------------------------- #
KP, KD = 45.0, 8.0  # gentle task-space gains: converge without punting the cube
FORCE_CLAMP = 40.0
CLOSE_EPS = 0.03  # start closing once the grasp centre is this close
#: If the hand is closing but the cube has been knocked further away than this,
#: the grasp clearly missed: drop the frozen target and re-approach.
CLOSE_RETRY_DIST = 0.05
GRASP_DIST = 0.06  # "centred on the cube" tolerance for a successful grasp

#: Normalised closure for the hand servos in a raw MuJoCo loop: 0 = the open
#: preset, 1 = the power-grasp preset (see ``hand_ctrl``).
GRIP_CTRL_OPEN = 0.0
GRIP_CTRL_CLOSED = 1.0


@dataclass(frozen=True)
class GraspIds:
    """Named ids / qpos addresses of the grasp-relevant scene elements."""

    grasp_site: int
    cube_body: int
    cube_geom: int
    cube_qpos: int
    marker_mocap: int
    n_arm: int
    #: The L20, in model joint order: joint / qpos / dof addresses, the matching
    #: actuator ids, and the open+closed targets that define the synergy.
    hand_joints: tuple[int, ...]
    hand_qpos: np.ndarray
    hand_dof: np.ndarray
    hand_acts: np.ndarray
    hand_open: np.ndarray
    hand_close: np.ndarray
    #: Geom id -> finger name, so contacts can be attributed to a finger.
    geom_finger: dict[int, str]
    #: ``(name, qpos address, (x, y))`` of every distractor block in the scene.
    distractors: tuple[tuple[str, int, tuple[float, float]], ...] = ()


# --------------------------------------------------------------------------- #
# pick & place
# --------------------------------------------------------------------------- #
#: Destination bench (the second table) and the drop-off pad on its top face.
TABLE_2_POS = (0.50, 0.62, 0.42)
TABLE_2_SIZE = (0.30, 0.26, 0.03)
#: Where the cube centre should end up: the near corner of the destination bench,
#: which shortest the carry (0.56 m from the target cube), stays inside the
#: arm's reach (radius 0.78 m) and is far outside the hand camera's view of the
#: source bench (the camera's footprint stops around y = 0.24).
PLACE_TARGET = np.array([0.58, 0.52, CUBE_Z])
#: Carry height above the bench tops while transporting (m).
LIFT_HEIGHT = 0.10
#: The cube counts as placed when its centre ends this close to PLACE_TARGET.
PLACE_TOLERANCE = 0.05

#: Transport is done *gently*: these gains were measured on the previous pinch
#: gripper, whose jaws were pushed open (gap 0.050 -> 0.067, contacts lost) as
#: soon as the arm moved faster than ~2 cm/s, whatever the cube's mass.  The
#: L20 grips harder, but the same centimetre waypoints with soft gains are what
#: keep the cube seated; the approach phase still uses the stiffer reaching
#: gains above.
TRANSPORT_KP, TRANSPORT_KD, TRANSPORT_FORCE = 20.0, 10.0, 6.0
TRANSPORT_WAYPOINT = 0.01  # m
TRANSPORT_TOL = 0.006  # m
#: Hold the (last detected) target for a few steps after closing, so the cube
#: seats itself in the hand before the arm starts accelerating.
SETTLE_STEPS = 15
#: Phase-2 waypoint walk (see :class:`PlacePlanner`).  ``PLACE_WAYPOINT`` and
#: the soft transport gains above set the carry speed at roughly
#: ``kp / kd * waypoint`` ~= 2 cm/s; anything faster lets a 3.75 kg cube creep
#: out from between the fingers.
PLACE_WAYPOINT = TRANSPORT_WAYPOINT
PLACE_TOL = TRANSPORT_TOL
#: The gripper is "over the pad" (and may start descending) within this xy
#: distance of :data:`PLACE_TARGET`.
PLACE_XY_TOL = 0.015


def _id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:
        raise ValueError(f"{name!r} not found in the grasp model")
    return int(i)


def scene_ids(model: mujoco.MjModel) -> GraspIds:
    """Resolve every grasp-relevant id once, instead of by magic index."""
    cube_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    marker_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "marker")
    distractors = tuple(
        (name, _free_joint_qpos(model, body), xy)
        for name, xy in DISTRACTORS
        if (body := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)) >= 0
    )
    hand_joints = tuple(
        _id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in HAND_JOINTS
    )
    geom_finger: dict[int, str] = {}
    for finger, geoms in HAND_POSES["fingers"].items():
        for geom in geoms:
            if (gid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom)) >= 0:
                geom_finger[gid] = finger
    return GraspIds(
        grasp_site=_id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_center"),
        cube_body=cube_body,
        cube_geom=_id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"),
        cube_qpos=_free_joint_qpos(model, cube_body),
        marker_mocap=int(model.body_mocapid[marker_body]) if marker_body >= 0 else -1,
        n_arm=N_ARM,
        hand_joints=hand_joints,
        hand_qpos=np.array([model.jnt_qposadr[j] for j in hand_joints]),
        hand_dof=np.array([model.jnt_dofadr[j] for j in hand_joints]),
        hand_acts=np.array(
            [
                _id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hand_{name}")
                for name in HAND_JOINTS
            ]
        ),
        hand_open=np.array([HAND_POSES["open"][name] for name in HAND_JOINTS]),
        hand_close=np.array([HAND_POSES["close"][name] for name in HAND_JOINTS]),
        geom_finger=geom_finger,
        distractors=distractors,
    )


def _free_joint_qpos(model: mujoco.MjModel, body_id: int) -> int:
    """Qpos address of a body's free joint (the free joint is its first joint)."""
    return int(model.jnt_qposadr[int(model.body_jntadr[body_id])])


# --------------------------------------------------------------------------- #
# scene construction
# --------------------------------------------------------------------------- #
def _place_free_body(data: mujoco.MjData, qpos_adr: int, xyz) -> None:
    """Teleport a free body to ``xyz`` (world) with an upright orientation."""
    data.qpos[qpos_adr:qpos_adr + 3] = xyz
    data.qpos[qpos_adr + 3:qpos_adr + 7] = np.array([1.0, 0.0, 0.0, 0.0])


def place_cube(data: mujoco.MjData, ids: GraspIds, xyz) -> None:
    """Put the target cube at ``xyz`` (world)."""
    _place_free_body(data, ids.cube_qpos, xyz)


def place_distractors(data: mujoco.MjData, ids: GraspIds) -> None:
    """Reset every distractor block to its home spot on the bench."""
    for _name, qpos_adr, (x, y) in ids.distractors:
        _place_free_body(data, qpos_adr, (x, y, CUBE_Z))


def make_scene(seed: int = 0, cube_xyz=None, model_path: Optional[str] = None):
    """Build the grasp scene; returns ``(model, data, cube_xyz)``."""
    model = mujoco.MjModel.from_xml_path(str(model_path or MODEL_PATH))
    data = mujoco.MjData(model)
    ids = scene_ids(model)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[: ids.n_arm] = START_POSE
    data.qpos[ids.hand_qpos] = ids.hand_open

    if cube_xyz is None:
        rng = np.random.default_rng(seed)
        cube_xyz = np.array(
            [
                CUBE_X + rng.uniform(-CUBE_SPREAD[0], CUBE_SPREAD[0]),
                rng.uniform(-CUBE_SPREAD[1], CUBE_SPREAD[1]),
                CUBE_Z,
            ]
        )
    place_cube(data, ids, cube_xyz)
    place_distractors(data, ids)
    mujoco.mj_forward(model, data)
    return model, data, np.asarray(cube_xyz, dtype=float)


# --------------------------------------------------------------------------- #
# vision helper
# --------------------------------------------------------------------------- #
def detect_cube(model, data, renderer, *, cam_name: str = "cam_hand"):
    """Localise the red cube with the eye-in-hand RGB-D camera.

    Returns the estimated world position, or ``None`` if no red blob is visible.
    """
    rgb, depth = render_rgbd(renderer, data, cam_name)
    return estimate_cube_world_rgbd(
        model,
        data,
        cam_name=cam_name,
        plane_z=TABLE_Z,
        cube_side=CUBE_SIDE,
        width=rgb.shape[1],
        height=rgb.shape[0],
        img=rgb,
        depth_img=depth,
    )


# --------------------------------------------------------------------------- #
# expert controller
# --------------------------------------------------------------------------- #
def gripper_gap(data: mujoco.MjData, ids: GraspIds) -> float:
    """Effective jaw gap (m) implied by how far the hand has closed.

    The L20 has no single slide joint to read, so the gap is the open-hand gap
    scaled by the *closure fraction* -- the same quantity the reward shaping and
    the ``is_grasped`` gate have always been written against.
    """
    return float(GRIP_OPEN * (1.0 - hand_closure(data, ids)))


def hand_closure(data: mujoco.MjData, ids: GraspIds) -> float:
    """How far the 21-joint hand has travelled towards its power-grasp preset.

    ``0`` is the open preset, ``1`` the closed one; the average is taken over
    the joints that actually move between the two (the finger side-swing joints
    share a target in both presets).
    """
    travel = ids.hand_close - ids.hand_open
    moving = np.abs(travel) > 1e-6
    if not np.any(moving):
        return 0.0
    reached = (data.qpos[ids.hand_qpos][moving] - ids.hand_open[moving]) / travel[moving]
    return float(np.clip(np.mean(reached), 0.0, 1.0))


def hand_ctrl(ids: GraspIds, close: float) -> np.ndarray:
    """Actuator targets for the L20 at a normalised closure in ``[0, 1]``."""
    close = float(np.clip(close, 0.0, 1.0))
    return ids.hand_open + close * (ids.hand_close - ids.hand_open)


def set_hand(data: mujoco.MjData, ids: GraspIds, close: float) -> None:
    """Drive the whole hand to a normalised closure in a raw MuJoCo loop."""
    data.ctrl[ids.hand_acts] = hand_ctrl(ids, close)


def cube_world(data: mujoco.MjData, ids: GraspIds) -> np.ndarray:
    return data.xpos[ids.cube_body][:3].copy()


def grasp_world(data: mujoco.MjData, ids: GraspIds) -> np.ndarray:
    return data.site_xpos[ids.grasp_site][:3].copy()


def cube_on_destination(
    data: mujoco.MjData, ids: GraspIds, target=None, tol: float = PLACE_TOLERANCE
) -> bool:
    """True when the cube sits on the drop-off pad of the destination bench."""
    target = PLACE_TARGET if target is None else np.asarray(target, dtype=float)
    return float(np.linalg.norm(cube_world(data, ids) - target)) < tol


def set_marker(data: mujoco.MjData, ids: GraspIds, xyz) -> None:
    """Move the non-colliding marker sphere to ``xyz`` (the detected point)."""
    if ids.marker_mocap < 0:
        return
    data.mocap_pos[ids.marker_mocap] = np.asarray(xyz, dtype=float)
    data.mocap_quat[ids.marker_mocap] = np.array([1.0, 0.0, 0.0, 0.0])


def servo_command(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target,
    ids: GraspIds,
    *,
    kp: float = KP,
    kd: float = KD,
    force_clamp: float = FORCE_CLAMP,
    close_eps: float = CLOSE_EPS,
) -> tuple[np.ndarray, bool, float]:
    """Resolved-rate command towards ``target``.

    Returns ``(arm_torques, close, dist)`` where ``arm_torques`` is the
    normalised ``[-1, 1]`` joint command for the first ``ids.n_arm`` joints and
    ``close`` says whether the gripper should be shutting.
    """
    err = np.asarray(target, dtype=float) - grasp_world(data, ids)
    dist = float(np.linalg.norm(err))

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ids.grasp_site)
    jac = jacp[:, : ids.n_arm]

    force = kp * err - kd * (jac @ data.qvel[: ids.n_arm])
    magnitude = float(np.linalg.norm(force))
    if magnitude > force_clamp:
        force *= force_clamp / magnitude
    torques = np.clip(jac.T @ force, -1.0, 1.0)
    return torques, dist < close_eps, dist


def is_grasped(
    data: mujoco.MjData,
    ids: GraspIds,
    dist: float,
    gap: float,
    *,
    max_dist: float = GRASP_DIST,
    max_gap: float = GRASP_GAP,
) -> bool:
    """True when enough fingers surround the cube while the hand is closed."""
    if dist >= max_dist or gap >= max_gap:
        return False
    touching = set()
    for contact in data.contact:
        pair = (contact.geom1, contact.geom2)
        if ids.cube_geom in pair:
            other = pair[0] if pair[1] == ids.cube_geom else pair[1]
            finger = ids.geom_finger.get(other)
            if finger is not None and finger != "palm":
                touching.add(finger)
    return len(touching) >= GRASP_FINGERS


def expert_target(env) -> np.ndarray:
    """The cube position the expert servo should drive to, given the env."""
    if getattr(env, "obs_target", "vision") == "true":
        return env.cube_world()
    if env.last_detected is not None:
        return env.last_detected.copy()
    return env.detected_world()[0].copy()


def teacher_action(env) -> np.ndarray:
    """The expert action for the current env state (gym action convention)."""
    torques, close, _ = servo_command(
        env.model, env.data, expert_target(env), env.ids
    )
    grip = -1.0 if close else 1.0
    return np.concatenate([torques, [grip]]).astype(np.float32)


# --------------------------------------------------------------------------- #
# pick & place: the carry plan and its teacher
# --------------------------------------------------------------------------- #
class PlacePlanner:
    """Sub-goal walk that carries a grasped cube to the drop-off pad.

    The plan is the one :func:`pick_and_place` proved out: hold still for
    :data:`SETTLE_STEPS` so the cube seats in the hand, rise straight up
    clear of the bench, cross to the pad at :data:`LIFT_HEIGHT`, then descend
    onto it.  Every leg is walked in :data:`PLACE_WAYPOINT` (1 cm) increments:
    the grip is soft, so a larger jump lets the fingers be pushed open and
    the cube slips out (measured: anything above ~4 cm/s loses contact).

    The environment owns the plan, and publishes ``goal - grasp`` in the
    observation, so the teacher and the learned policy always chase the same
    point.
    """

    #: Phase order; ``update`` walks it once each sub-goal is reached.
    PHASES = ("settle", "lift", "carry", "lower")

    def __init__(
        self,
        target=PLACE_TARGET,
        *,
        waypoint: float = PLACE_WAYPOINT,
        tol: float = PLACE_TOL,
        height: float = LIFT_HEIGHT,
        settle_steps: int = SETTLE_STEPS,
        xy_tol: float = PLACE_XY_TOL,
    ):
        self.target = np.asarray(target, dtype=float)
        self.waypoint = float(waypoint)
        self.tol = float(tol)
        self.height = float(height)
        self.settle_steps = int(settle_steps)
        self.xy_tol = float(xy_tol)
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self, grasp=None) -> None:
        """Start a new episode; ``grasp`` seeds the settle sub-goal."""
        self.phase = "settle"
        self.settle_left = self.settle_steps
        self.done = False
        self.goal = None if grasp is None else np.asarray(grasp, dtype=float).copy()

    def update(self, grasp) -> np.ndarray:
        """Advance the plan for the measured gripper position; return the goal."""
        grasp = np.asarray(grasp, dtype=float)
        if self.settle_left > 0:
            self.settle_left -= 1
            if self.goal is None:
                self.goal = grasp.copy()
            return self.goal.copy()
        if self.done:
            return self.target.copy()
        if self.phase == "settle":
            self._enter("lift", grasp)
        while not self.done and np.linalg.norm(self.goal - grasp) < self.tol:
            leg_end = self._end_point(self.phase, grasp)
            if np.linalg.norm(leg_end - grasp) > self.tol:
                # same leg, next centimetre: the cube only survives a gentle walk
                self._enter(self.phase, grasp)
                continue
            nxt = self.PHASES[self.PHASES.index(self.phase) + 1 :]
            if not nxt:  # the gripper reached the pad: the walk is over
                self.done = True
                self.goal = self.target.copy()
                break
            self._enter(nxt[0], grasp)
        return self.goal.copy()

    # ------------------------------------------------------------------ #
    def _enter(self, phase: str, grasp: np.ndarray) -> None:
        """Aim at the next sub-goal, at most :data:`PLACE_WAYPOINT` away."""
        self.phase = phase
        end = self._end_point(phase, grasp)
        delta = end - grasp
        dist = float(np.linalg.norm(delta))
        step = min(self.waypoint, dist)
        self.goal = grasp + (delta * (step / dist) if dist > 1e-12 else 0.0)

    def _end_point(self, phase: str, grasp: np.ndarray) -> np.ndarray:
        carry_z = self.target[2] + self.height
        if phase == "lift":  # straight up, clear of the source bench
            return np.array([grasp[0], grasp[1], carry_z])
        if phase == "carry":  # across to the pad, still at carry height
            return np.array([self.target[0], self.target[1], carry_z])
        if phase == "lower":  # down onto the pad
            return self.target.copy()
        raise ValueError(f"unknown place phase {phase!r}")


def teacher_action_place(env) -> np.ndarray:
    """Expert action for ``RokaePro7PickPlace`` (gym action convention).

    Mirrors the environment's plan: servo the stiff reach gains onto the
    detected cube and close the hand, then walk the carry waypoints with the
    soft transport gains, and finally open once the plan reaches the pad.
    """
    goal = np.asarray(env.goal, dtype=float)
    if not env.holding:
        torques, close, _dist = servo_command(env.model, env.data, goal, env.ids)
        grip = -1.0 if close else 1.0
    else:
        torques, _close, _dist = servo_command(
            env.model, env.data, goal, env.ids,
            kp=TRANSPORT_KP, kd=TRANSPORT_KD, force_clamp=TRANSPORT_FORCE,
        )
        grip = 1.0 if env.plan.done else -1.0  # open only once parked on the pad
    return np.concatenate([torques, [grip]]).astype(np.float32)


# --------------------------------------------------------------------------- #
# scripted pick & place (source bench -> destination bench)
# --------------------------------------------------------------------------- #
@dataclass
class PickPlaceResult:
    """Outcome of one scripted pick-and-place episode."""

    picked: bool
    placed: bool
    steps: int
    reason: str = ""

    @property
    def success(self) -> bool:
        return self.picked and self.placed


def _apply_action(model, data, ids: GraspIds, torques, closed: bool) -> None:
    data.ctrl[: ids.n_arm] = torques
    set_hand(data, ids, GRIP_CTRL_CLOSED if closed else GRIP_CTRL_OPEN)
    mujoco.mj_step(model, data)


def _servo_to(
    model,
    data,
    ids: GraspIds,
    target,
    *,
    closed: bool = True,
    tol: float = TRANSPORT_TOL,
    max_steps: int = 120,
    on_step=None,
    gentle: bool = True,
) -> int:
    """Drive the grasp centre onto ``target``; returns the steps used.

    ``gentle=True`` walks the target in :data:`TRANSPORT_WAYPOINT` (1 cm)
    increments with the soft transport gains, which is what keeps a grasped cube
    from being squeezed out of the hand.  ``gentle=False`` is the single, stiffer
    reaching move used to close in on the cube.
    """
    start = grasp_world(data, ids)
    delta = np.asarray(target, dtype=float) - start
    waypoints = (
        max(1, int(np.ceil(np.linalg.norm(delta) / TRANSPORT_WAYPOINT)))
        if gentle
        else 1
    )
    gains = (
        dict(kp=TRANSPORT_KP, kd=TRANSPORT_KD, force_clamp=TRANSPORT_FORCE)
        if gentle
        else {}
    )

    steps = 0
    for i in range(1, waypoints + 1):
        point = start + delta * (i / waypoints)
        for _ in range(max_steps):
            torques, _close, dist = servo_command(model, data, point, ids, **gains)
            _apply_action(model, data, ids, torques, closed)
            steps += 1
            if on_step is not None:
                on_step()
            if dist < tol:
                break
    return steps


def pick_and_place(
    model,
    data,
    ids: GraspIds,
    renderer=None,
    *,
    cube_hint=None,
    settle_steps: int = SETTLE_STEPS,
    on_step=None,
    on_phase=None,
) -> PickPlaceResult:
    """Scripted expert for the two-bench task.

    Phases: approach + grasp on the source bench, lift, carry across to the
    destination pad, lower, release.  ``on_step`` (if given) is called after
    every physics step and ``on_phase(name)`` at the start of each phase, which
    is how the demo scripts record the motion.
    """
    steps = 0

    def tick() -> None:
        if on_step is not None:
            on_step()

    def phase(name: str) -> None:
        if on_phase is not None:
            on_phase(name)

    def holding() -> bool:
        dist = float(np.linalg.norm(grasp_world(data, ids) - cube_world(data, ids)))
        return is_grasped(data, ids, dist, gripper_gap(data, ids))

    # --- 1) approach the detected cube and close the hand on it ----------
    phase("approach")
    target = None if cube_hint is None else np.asarray(cube_hint, dtype=float)
    picked = False
    closing = False
    for _ in range(200):
        # Keep re-detecting while reaching, but *freeze* the target once the
        # hand starts to close.  Re-detecting through the close makes the servo
        # chase a target that moves as the hand occludes the cube, which wedges
        # the cube out past the fingertips (measured: cube centre at z=78 mm
        # while the fingers only reach 75 mm) and it then slips when carried.
        if not closing:
            estimate = (
                detect_cube(model, data, renderer) if renderer is not None else None
            )
            if estimate is not None:
                target = estimate
        if target is None:
            target = cube_world(data, ids)
        torques, close, _dist = servo_command(model, data, target, ids)
        _apply_action(model, data, ids, torques, close)
        steps += 1
        tick()
        closing = closing or close
        if holding():
            picked = True
            break
    if not picked:
        return PickPlaceResult(False, False, steps, "grasp failed")

    # --- 1b) let the cube seat in the hand -------------------------------
    if settle_steps:
        phase("settle")
        for _ in range(settle_steps):
            torques, _close, _dist = servo_command(model, data, target, ids)
            _apply_action(model, data, ids, torques, closed=True)
            steps += 1
            tick()
        if not holding():
            return PickPlaceResult(True, cube_on_destination(data, ids), steps, "lost grip")

    # --- 2) lift clear of the source bench ------------------------------
    phase("lift")
    lift = grasp_world(data, ids) + np.array([0.0, 0.0, LIFT_HEIGHT])
    steps += _servo_to(model, data, ids, lift, on_step=tick)
    if not holding():
        return PickPlaceResult(True, cube_on_destination(data, ids), steps, "dropped on lift")

    # --- 3) carry the cube over to the destination bench -----------------
    phase("transport")
    over_place = PLACE_TARGET + np.array([0.0, 0.0, LIFT_HEIGHT])
    steps += _servo_to(model, data, ids, over_place, on_step=tick)
    if not holding():
        return PickPlaceResult(
            True, cube_on_destination(data, ids), steps, "dropped in transit"
        )

    # --- 4) lower onto the drop-off pad ---------------------------------
    phase("lower")
    steps += _servo_to(model, data, ids, PLACE_TARGET, on_step=tick)

    # --- 5) release and settle -------------------------------------------
    phase("release")
    for _ in range(25):
        torques, _close, _dist = servo_command(model, data, PLACE_TARGET, ids)
        _apply_action(model, data, ids, torques, closed=False)
        steps += 1
        tick()

    placed = cube_on_destination(data, ids)
    return PickPlaceResult(
        True, placed, steps, "placed" if placed else "cube missed the pad"
    )
