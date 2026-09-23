"""Dual-arm box lifting for the take_box cell (14 arm joints + 2 grips).

Task, stage by stage - the same four beats as the ROS YAML task
``prepare_action -> take_box_step1 -> wait -> take_box_step2``:

1. **approach**: both palms move onto their box face (reward = distance).
2. **grasp**: both palms on the face and both grip channels closed -> the box
   latches onto the hands (``+grasp_bonus``).
3. **lift**: the box follows the rigid frame between the two palms, so the
   policy has to raise *both* arms and keep the box level (reward = height
   error + tilt).
4. **hold**: staying inside the height/orientation tolerance for
   ``hold_steps`` consecutive steps is the success condition.

Everything is kinematic: arm meshes have no contacts and the box is a mocap
body moved by the hands, matching the quasi-static carry convention used in the
sibling project (``mujoco_arm_ppo``).  Forces and finger-object friction are not
modelled; that is a deliberate simplification so the policy learns the
coordination first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .rewards import GraspStep, RewardConfig, TakeBoxReward

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
MODEL_PATH = ASSETS / "take_box_cell.xml"
READY_PATH = ASSETS / "take_box_ready.json"
CONFIG_PATH = ROOT / "cell_config.json"

SIDES = ("left", "right")
FINGERS = ("index", "middle", "ring", "pinky")


def palm_frame_local(model, data, wrist_body: str, finger_prefix: str,
                     distal_suffix: str = "_distal",
                     thumb_body: str = "thumb_distal") -> tuple:
    """Palm centre and palm axes ``(finger, spread, normal)`` in the wrist frame.

    ``left_grasp`` / ``right_grasp`` are **position-only** sites, so the frame
    they expose is the wrist link, not the palm.  Everything that needs the
    palm frame (the wrist-roll check of the box, the Cartesian targets of
    ``take_box_env/box_api.py``, the pre-grasp IK in ``build_cell_model.py``) therefore
    measures it from the finger bodies instead of trusting ``site_xmat``.
    """
    mujoco.mj_forward(model, data)
    wrist = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body)
    if wrist < 0:
        raise KeyError(f"wrist body {wrist_body!r} is missing from the model")
    origin = data.xpos[wrist].copy()
    rotation = data.xmat[wrist].reshape(3, 3)

    def local(name):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if index < 0:
            raise KeyError(f"hand body {name!r} is missing from the model")
        return rotation.T @ (data.xpos[index] - origin)

    tips = np.mean([local(f"{finger_prefix}{f}{distal_suffix}") for f in FINGERS],
                   axis=0)
    thumb = local(f"{finger_prefix}{thumb_body}")
    centre = (tips + thumb) / 2
    finger = tips - centre
    finger /= np.linalg.norm(finger)
    spread = (local(f"{finger_prefix}pinky{distal_suffix}")
              - local(f"{finger_prefix}index{distal_suffix}"))
    spread -= finger * float(np.dot(spread, finger))
    spread /= np.linalg.norm(spread)
    normal = np.cross(finger, spread)
    return centre, np.column_stack([finger, spread, normal])


class TakeBoxEnv(gym.Env):
    """Both arms lift one box together."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        stage: str = "lift",
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        grasp_tol: Optional[float] = None,
        lift_height: Optional[float] = None,
        model_path: Optional[str] = None,
        reset_noise: Optional[float] = None,
        allow_release: bool = False,
        seed: Optional[int] = None,
    ):
        super().__init__()
        assert stage in ("reach", "lift"), "stage must be 'reach' or 'lift'"
        self.stage = stage
        self.render_mode = render_mode

        self.config = json.loads(CONFIG_PATH.read_text())
        task = self.config["task"]
        self.grasp_tol = task["grasp_tol"] if grasp_tol is None else grasp_tol
        self.lift_height = task["lift_height"] if lift_height is None else lift_height
        self.tilt_tol = np.radians(task["tilt_tol_deg"])
        self.height_tol = task.get("height_tol", 0.02)
        self.hold_steps = task["hold_steps"]
        self.max_steps = (
            task["max_steps"] if max_steps is None else max_steps
        )
        #: 奖励函数在 take_box_env/rewards.py 里（势函数塑形 + 事件奖励），系数可在
        #: cell_config.json 的 task.reward 覆盖
        self.reward = TakeBoxReward(RewardConfig.from_config(self.config))
        #: With ``allow_release`` the hand can let go: a grip channel below
        #: ``reward.release_grip`` drops the grasp and the box stops riding the
        #: palms.  It is off by default because the lift task never needs it and
        #: it costs a lot of samples: nothing rewards "keep the grip closed"
        #: directly (the potential does not depend on the grip), the payoff only
        #: shows up ~40 steps later as the ability to lift.  400k steps measured
        #: here: 0% success with the release on, 77% with it off.  Turn it on for
        #: the "place + let go" stage (and budget more training).
        self.allow_release = allow_release
        #: std of the joint noise added to the pre-grasp pose at reset.  The
        #: lift task starts close to the box (the real cell does a "prepare"
        #: move first); the reach-only curriculum starts much further away,
        #: otherwise the episode would end on the first step.
        if reset_noise is None:
            reset_noise = 0.02 if stage == "lift" else 0.15
        self.reset_noise = reset_noise

        self.model_path = str(model_path or MODEL_PATH)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self._renderer = None

        ready = json.loads(READY_PATH.read_text())
        self.ready_qpos = {}
        for side in SIDES:
            self.ready_qpos[side] = np.asarray(ready["joints"][side], dtype=float)
        self.grasp_targets = {
            side: np.asarray(ready["targets"][side]["site_pos"], dtype=float)
            for side in SIDES
        }
        #: desired palm position, expressed in the box frame.  It is not the box
        #: face itself: the grasp site is a virtual point inside the hand, and
        #: build_cell_model.py measures how far it has to stay outside the face
        #: for the *palm surface* to rest on the box.
        self.palm_offsets = {}
        for side in SIDES:
            offset = ready["targets"][side].get("box_offset")
            if offset is None:  # asset from an older build: fall back to the face
                offset = self.grasp_targets[side] - np.asarray(
                    self.config["cell"]["box"]["pos"], dtype=float)
            self.palm_offsets[side] = np.asarray(offset, dtype=float)

        self.box_size = np.asarray(self.config["cell"]["box"]["size"], dtype=float)
        self.box_home = np.asarray(self.config["cell"]["box"]["pos"], dtype=float)
        self.table_top = (
            self.config["cell"]["table"]["pos"][2]
            + self.config["cell"]["table"]["size"][2] / 2
        )
        self.box_target_z = self.table_top + self.lift_height

        self._index_model()
        self.n_arm_dof = 2 * self.arm_dof
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_arm_dof + 2,), dtype=np.float32
        )
        # per arm: [cos q(7), sin q(7), dq(7), palm-face error(3)] = 24
        # global : box offset(3), tilt(1), grasped(1), grip(2) = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(2 * 24 + 7,), dtype=np.float32,
        )

        self._step_count = 0
        self._hold_count = 0
        self.grasped = False
        #: the latch bonus is an event reward: pay it once per episode, otherwise
        #: a policy can farm it by opening and closing the hand on the spot
        self._grasp_rewarded = False
        #: where the box sits inside the palm frame, captured when the grasp
        #: latches (see _capture_box_offset)
        self._box_offset_rot = np.eye(3)
        self._box_offset_pos = np.zeros(3)
        self._reset_box()
        self._potential = 0.0

    # ------------------------------------------------------------------ #
    def _index_model(self) -> None:
        m = self.model
        robot = self.config["robot"]
        self.robot_source = robot.get("source", "single_arm_urdf")
        # joint names: the dual-arm description uses L1_Joint..R7_Joint, the
        # single-arm builds use left_joint1..right_joint7
        if self.robot_source == "dual_arm_mjcf":
            self.joint_names = {
                side: list(robot["arms"][side]["joints"]) for side in SIDES
            }
            self.finger_joints = {
                side: list(robot["arms"][side]["finger_joints"]) for side in SIDES
            }
        else:
            self.joint_names = {
                side: [f"{side}_joint{i}" for i in range(1, 8)] for side in SIDES
            }
            channels = ("thumb_flex", "thumb_swing", "index", "middle", "ring", "pinky")
            self.finger_joints = {
                side: [f"{side}_{c}_joint" for c in channels] for side in SIDES
            }
        self.arm_dof = len(self.joint_names["left"])

        self.qpos_idx = {}
        self.dof_idx = {}
        self.act_idx = {}
        self.hand_act = {}
        self.grasp_site = {}
        self.limits = {}
        #: palm "up" direction (fingers' spread axis) expressed in the wrist
        #: link frame.  The dual-arm hands hang off the flange at an arbitrary
        #: assembly angle, so the wrist frame is ~15-20 deg away from the palm
        #: frame and cannot be used to judge whether the box is level.
        self.palm_up_local = {}
        for side in SIDES:
            qpos, dof, acts, limits = [], [], [], []
            names = self.joint_names[side]
            for index, name in enumerate(names, start=1):
                joint = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                qpos.append(int(m.jnt_qposadr[joint]))
                dof.append(int(m.jnt_dofadr[joint]))
                limits.append(m.jnt_range[joint])
                actuator_name = (
                    f"{side}_{name}" if self.robot_source == "dual_arm_mjcf"
                    else f"{side}_a{index}"
                )
                acts.append(mujoco.mj_name2id(
                    m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
            self.qpos_idx[side] = np.asarray(qpos)
            self.dof_idx[side] = np.asarray(dof)
            self.act_idx[side] = np.asarray(acts)
            self.limits[side] = np.asarray(limits)
            self.hand_act[side] = np.asarray([
                mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{joint}_pos")
                for joint in self.finger_joints[side]
            ])
            self.grasp_site[side] = mujoco.mj_name2id(
                m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp")
        self._measure_palm_up_axes()

    def _measure_palm_up_axes(self) -> None:
        """Palm "up" axis (the fingers' spread axis) in each wrist link frame.

        The grasp sites are position-only, so they expose the wrist frame, not
        the palm frame; this axis is measured from the finger bodies instead.
        The box frame used by the carry is ``[finger, left-right, up]``, so both
        hands have to agree on "up": the dual-arm hands are CAD mirrors of each
        other and therefore point their index->pinky axes in opposite world
        directions in the pre-grasp pose, and the second one is flipped.

        This axis is *not* the world vertical - the pre-grasp roll is whatever
        the wrists can reach (``task.grasp_roll_deg``), and whether the box is
        level is judged from the box body itself.
        """
        saved = self.data.qpos.copy()
        for side in SIDES:
            self.data.qpos[self.qpos_idx[side]] = self.ready_qpos[side]
        local, world = {}, {}
        for side in SIDES:
            if self.robot_source != "dual_arm_mjcf":
                # the parametric hand of the single-arm build is bolted to the
                # flange without a rotation, so its palm frame *is* the wrist
                # frame and the plain y axis is already the spread direction
                self.palm_up_local[side] = np.array([0.0, 1.0, 0.0])
                continue
            arm = self.config["robot"]["arms"][side]
            _centre, source = palm_frame_local(
                self.model, self.data, arm["wrist_body"], arm["finger_prefix"])
            wrist = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, arm["wrist_body"])
            local[side] = source[:, 1].copy()
            world[side] = self.data.xmat[wrist].reshape(3, 3) @ local[side]
        if local:
            agreement = float(np.dot(world["left"], world["right"]))
            if abs(agreement) < 0.5:
                raise RuntimeError(
                    "the two hands disagree about their spread axis in the " 
                    f"pre-grasp pose ({np.round(world['left'], 3)} vs "
                    f"{np.round(world['right'], 3)}): re-solve it with "
                    "build_cell_model.py")
            self.palm_up_local["left"] = local["left"]
            self.palm_up_local["right"] = local["right"] * (-1.0 if agreement < 0 else 1.0)
        self.data.qpos[:] = saved
        mujoco.mj_forward(self.model, self.data)

    def _hand_ctrlrange(self, side: str) -> tuple:
        """Per-actuator ``(low, high)`` of one hand's finger servos.

        Each finger channel has its own travel (the thumb pitch range is 0.58 rad,
        the finger curl 1.6 rad), so the single grip scalar is mapped per joint:
        ``grip = 1`` means "every channel at its own limit", the same convention
        as the real 6-channel O6 hand.  Handing one shared number to all eleven
        joints used to over-drive the thumb into its limit stop and leave the
        fingers at 75% of their range.
        """
        ctrlrange = self.model.actuator_ctrlrange[self.hand_act[side]]
        return ctrlrange[:, 0].copy(), ctrlrange[:, 1].copy()

    # ------------------------------------------------------------------ #
    def _reset_box(self) -> None:
        self.data.mocap_pos[0] = self.box_home
        self.data.mocap_quat[0] = np.array([1.0, 0.0, 0.0, 0.0])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        mujoco.mj_resetData(self.model, self.data)
        # start from the solved pre-grasp pose, with a small perturbation so the
        # episodes are not identical
        for side in SIDES:
            noise = self.np_random.normal(0.0, self.reset_noise, size=self.arm_dof)
            q = np.clip(
                self.ready_qpos[side] + noise,
                self.limits[side][:, 0], self.limits[side][:, 1],
            )
            self.data.qpos[self.qpos_idx[side]] = q
            self.data.qvel[self.dof_idx[side]] = 0.0
        for side in SIDES:
            low, _high = self._hand_ctrlrange(side)
            self.data.ctrl[self.hand_act[side]] = low
        self._reset_box()
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._hold_count = 0
        self.grasped = False
        self._grasp_rewarded = False
        self._box_offset_rot = np.eye(3)
        self._box_offset_pos = np.zeros(3)
        self._potential = self.reward.potential(self._reward_step(np.zeros(2)))
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        arm_action, grip = action[:self.n_arm_dof], action[self.n_arm_dof:]
        for index, side in enumerate(SIDES):
            self.data.ctrl[self.act_idx[side]] = arm_action[index * self.arm_dof:(index + 1) * self.arm_dof]
            low, high = self._hand_ctrlrange(side)
            self.data.ctrl[self.hand_act[side]] = low + np.clip(grip[index], 0.0, 1.0) * (high - low)

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        if self.grasped:
            self._carry_box()

        reward, terminated = self._compute_reward(grip)
        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode is None:
            raise RuntimeError("render() without render_mode='rgb_array'")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="cam_iso")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------ #
    # geometry helpers
    # ------------------------------------------------------------------ #
    def palm_positions(self) -> np.ndarray:
        return np.asarray([
            self.data.site_xpos[self.grasp_site[side]].copy() for side in SIDES
        ])

    def face_points(self) -> np.ndarray:
        """Where each palm must sit: the grasp pose on the (current) box.

        The offsets are measured at build time with the palms resting on the box
        faces (``palm_clearance`` apart), so this is where the wrist has to take
        the grasp site - not the bare face plane, which would bury the hand.
        """
        centre = self.data.mocap_pos[0].copy()
        # the offsets belong to the box's own frame: once the grasp has latched
        # the box rides the palms, so a carried box can be yawed/rolled.
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.mocap_quat[0])
        rotation = rotation.reshape(3, 3)
        return np.asarray([centre + rotation @ self.palm_offsets[side]
                           for side in SIDES])

    def hand_errors(self) -> np.ndarray:
        """Distance from each palm centre to its box face."""
        return np.linalg.norm(self.palm_positions() - self.face_points(), axis=1)

    def grasp_ready(self) -> bool:
        """Both palms sit on their box face within ``grasp_tol``."""
        return bool(self.hand_errors().max() < self.grasp_tol)

    def box_tilt(self) -> float:
        """Angle between the box's own up axis and world +z.

        Taken from the box body rather than from the palms, so it is defined in
        both grasp modes: a released box keeps whatever orientation it had, and
        the potential below must not jump when the grasp latches or drops.
        """
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.mocap_quat[0])
        return float(np.arccos(np.clip(rotation.reshape(3, 3)[2, 2], -1.0, 1.0)))

    def _box_frame(self):
        left, right = self.palm_positions()
        y_axis = left - right
        norm = np.linalg.norm(y_axis)
        y_axis = y_axis / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0])
        # average the two palm "up" directions (finger spread axis), then
        # orthogonalise.  The up axis must come from the measured palm frame:
        # the wrist frame is tilted by the hand's assembly angle, and using it
        # made a level box read as a 15-20 deg tilt right after grasping.
        ups = []
        for side in SIDES:
            rot = self.data.site_xmat[self.grasp_site[side]].reshape(3, 3)
            ups.append(rot @ self.palm_up_local[side])
        z_axis = np.mean(ups, axis=0)
        z_axis = z_axis - y_axis * float(np.dot(z_axis, y_axis))
        norm = np.linalg.norm(z_axis)
        z_axis = z_axis / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        x_axis = np.cross(y_axis, z_axis)
        return np.column_stack([x_axis, y_axis, z_axis]), z_axis

    def _carry_box(self) -> None:
        """Transport servo: the box rides the rigid frame between the palms.

        The box keeps the pose offset it had *when the grasp latched* instead of
        being snapped onto the palm frame.  Snapping teleported the box by the
        remaining grasp error and, worse, rotated it onto the palm frame, whose
        "up" axis is ~9 deg off vertical at the pre-grasp pose - that is what
        made the box visibly take off the moment the hands touched it.
        """
        rotation, _ = self._box_frame()
        centre = self.palm_positions().mean(axis=0)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (rotation @ self._box_offset_rot).reshape(-1))
        self.data.mocap_quat[0] = quat
        self.data.mocap_pos[0] = centre + rotation @ self._box_offset_pos

    def _capture_box_offset(self) -> None:
        """Freeze the box pose inside the palm frame, for the carry above."""
        rotation, _ = self._box_frame()
        centre = self.palm_positions().mean(axis=0)
        box_rotation = np.zeros(9)
        mujoco.mju_quat2Mat(box_rotation, self.data.mocap_quat[0])
        self._box_offset_rot = rotation.T @ box_rotation.reshape(3, 3)
        self._box_offset_pos = rotation.T @ (self.data.mocap_pos[0] - centre)

    # ------------------------------------------------------------------ #
    def _reward_step(self, grip) -> GraspStep:
        """把当前状态打包给奖励函数（take_box_env/rewards.py）。"""
        return GraspStep(
            stage=self.stage,
            hand_errors=self.hand_errors(),
            box_z=float(self.data.mocap_pos[0][2]),
            box_target_z=float(self.box_target_z),
            box_tilt=self.box_tilt(),
            grip=np.asarray(grip, dtype=float),
            ctrl=self.data.ctrl,
            grasped=self.grasped,
            grasp_ready=self.grasp_ready(),
            grasp_rewarded=self._grasp_rewarded,
            allow_release=self.allow_release,
            hold_count=self._hold_count,
            potential=self._potential,
            grasp_tol=self.grasp_tol,
            height_tol=self.height_tol,
            tilt_tol=self.tilt_tol,
            hold_steps=self.hold_steps,
        )

    def _compute_reward(self, grip):
        """奖励全部在 take_box_env/rewards.py 里算，这里只负责喂状态、写回结果。"""
        result = self.reward.step(self._reward_step(grip))
        if result.grasped and not self.grasped:
            # 刚刚锁存：记下箱子在掌心坐标系里的相对位姿（见 _capture_box_offset）
            self._capture_box_offset()
        self.grasped = result.grasped
        self._grasp_rewarded = result.grasp_rewarded
        self._hold_count = result.hold_count
        self._potential = result.potential
        return result.reward, result.terminated

    # ------------------------------------------------------------------ #
    def _get_obs(self) -> np.ndarray:
        blocks = []
        palms = self.palm_positions()
        faces = self.face_points()
        for index, side in enumerate(SIDES):
            q = self.data.qpos[self.qpos_idx[side]]
            dq = self.data.qvel[self.dof_idx[side]]
            blocks.append(np.concatenate([
                np.cos(q), np.sin(q), dq, palms[index] - faces[index],
            ]))
        grips = []
        for side in SIDES:
            low, high = self._hand_ctrlrange(side)
            closed = (self.data.ctrl[self.hand_act[side]] - low) / (high - low)
            grips.append(float(np.mean(closed)))
        blocks.append(np.concatenate([
            self.data.mocap_pos[0] - self.box_home,
            [self.box_tilt(), float(self.grasped)],
            grips,
        ]))
        return np.concatenate(blocks).astype(np.float32)

    def _get_info(self) -> dict:
        errors = self.hand_errors()
        box_z = float(self.data.mocap_pos[0][2])
        if self.stage == "reach":
            success = bool(errors.max() < self.grasp_tol)
        else:
            success = bool(self.grasped and self._hold_count >= self.hold_steps)
        return {
            "dist_to_target": float(errors.mean()),
            "dist_left": float(errors[0]),
            "dist_right": float(errors[1]),
            "grasped": bool(self.grasped),
            "box_z": box_z,
            "lift": box_z - self.box_home[2],
            "target_lift": self.lift_height,
            "tilt_deg": float(np.degrees(self.box_tilt())),
            "hold_steps": self._hold_count,
            "success": success,
            "steps": self._step_count,
        }
