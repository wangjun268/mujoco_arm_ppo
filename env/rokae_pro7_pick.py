"""Gymnasium environment: Rokae xMate Pro7 7-DOF arm grasps a red cube.

The scene (real URDF STL meshes, workbench, red cube, coloured distractors,
eye-in-hand RGB-D camera, LinkerHand L20 dexterous hand) is the one defined by
:mod:`grasp_common`, which also supplies the scene constants and the "is it
grasped" test.  ``detect_red_cube`` turns the camera image into a 3D cube
position, so the observation is *vision-driven*: 7 arm torques + 1 grip command
are the actions -- the 21 hand joints close as one synergy -- and the reward
pushes the hand's grasp centre onto the cube and awards a large bonus once
enough fingers surround it.

Green/blue/yellow blocks clutter the bench next to the target: the red
segmenter must ignore them, and the arm can bump into them on its way in.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

import grasp_common as gc


class RokaePro7Pick(gym.Env):
    """Vision-driven pick task for the Pro7 arm (observation size 30)."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}
    MODEL_PATH = gc.MODEL_PATH

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 200,
        vel_cost: float = 0.0005,
        success_reward: float = 2.0,
        obs_target: str = "vision",  # "vision" (RGB-D) or "true" (ground truth)
        cube_spread: tuple = gc.CUBE_SPREAD,
        model_path: Optional[str] = None,
        action_scale: float = 1.0,
    ):
        super().__init__()
        self.model_path = str(model_path or self.MODEL_PATH)
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.vel_cost = vel_cost
        self.success_reward = success_reward
        self.obs_target = obs_target
        self.cube_spread = cube_spread
        #: Multiplier on the arm commands (the grip command is never scaled).
        #: ``1.0`` is the raw torque command; smaller values slow the arm down,
        #: but the grasp needs the stiff approach to seat the cube in the hand:
        #: with the previous gripper, measured success dropped from 5/5 at 1.0
        #: to 1/5 at 0.5.  Use this for PPO experiments, not for the scripted
        #: expert pull-outs.
        self.action_scale = float(action_scale)
        self._renderer = None

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self._det_renderer = mujoco.Renderer(self.model, height=240, width=320)

        self.ids = gc.scene_ids(self.model)
        self.n_arm = self.ids.n_arm
        # 7 arm torques + 1 grip command; the L20's 21 joints are driven as one
        # open/close synergy, so the action space is unchanged from the previous
        # pinch gripper (and so is every trained checkpoint's shape).
        self.n_action = self.n_arm + 1

        self._step_count = 0
        self._last_detected: Optional[np.ndarray] = None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_action,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3 * self.n_arm + 9,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # public accessors (used by the expert controller and the viewers)
    # ------------------------------------------------------------------ #
    @property
    def last_detected(self) -> Optional[np.ndarray]:
        """Last successfully localised cube position, or ``None``."""
        return self._last_detected

    def cube_world(self) -> np.ndarray:
        """Ground-truth cube position in world coordinates."""
        return gc.cube_world(self.data, self.ids)

    def grasp_world(self) -> np.ndarray:
        """World position of the gripper's grasp centre."""
        return gc.grasp_world(self.data, self.ids)

    def gripper_gap(self) -> float:
        """Current inner finger gap (m)."""
        return gc.gripper_gap(self.data, self.ids)

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[: self.n_arm] = gc.START_POSE
        self.data.qpos[self.ids.hand_qpos] = self.ids.hand_open

        # Random cube on the workbench, within reach and in the camera's view.
        cx, cy = self.cube_spread
        gc.place_cube(
            self.data,
            self.ids,
            [
                gc.CUBE_X + self.np_random.uniform(-cx, cx),
                self.np_random.uniform(-cy, cy),
                gc.CUBE_Z,
            ],
        )
        # Put the coloured distractors back on their bench spots (a previous
        # episode may have knocked them around).
        gc.place_distractors(self.data, self.ids)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._last_detected = None
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        # Arm motors are torque (ctrl [-1, 1]); the hand is position-servoed, so
        # map its action to a closure in [0, 1] (action -1 = fully closed).
        self.data.ctrl[: self.n_arm] = self.action_scale * action[: self.n_arm]
        gc.set_hand(self.data, self.ids, 0.5 * (1.0 - action[self.n_arm]))
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        reward, terminated = self._compute_reward()
        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode != "rgb_array":
            raise RuntimeError("render() called without render_mode='rgb_array'")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="cam_iso")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._det_renderer.close()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def detected_world(self) -> tuple[np.ndarray, float]:
        """Vision estimate of the cube (world) plus a ``seen`` flag."""
        if self.obs_target == "true":
            return self.cube_world(), 1.0
        try:
            estimate = gc.detect_cube(self.model, self.data, self._det_renderer)
        except Exception:
            estimate = None
        if estimate is not None:
            self._last_detected = estimate[:3]
            return estimate[:3], 1.0
        if self._last_detected is not None:
            return self._last_detected.copy(), 0.0
        # Never seen: report the gripper as a neutral (uninformative) target.
        return self.grasp_world(), 0.0

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[: self.n_arm]
        dq = self.data.qvel[: self.n_arm]
        target, seen = self.detected_world()
        rel = target - self.grasp_world()
        return np.concatenate(
            [np.cos(q), np.sin(q), dq, [self.gripper_gap()], [self.gripper_vel()],
             target, rel, [seen]]
        ).astype(np.float32)

    def gripper_vel(self) -> float:
        """Mean finger-joint speed (the hand has no single gripper joint)."""
        return float(np.mean(self.data.qvel[self.ids.hand_dof]))

    def _compute_reward(self):
        dist = float(np.linalg.norm(self.grasp_world() - self.cube_world()))
        vel = float(
            np.sum(self.data.qvel[: self.n_arm] ** 2) + self.gripper_vel() ** 2
        )
        gap = self.gripper_gap()

        reward = -dist - self.vel_cost * vel + 0.5 * np.exp(-dist / 0.3)
        # Encourage closing only once near the cube.
        if dist < 0.10:
            reward += 0.05 * max(0.0, (gc.GRIP_OPEN - gap) / gc.GRIP_OPEN)

        terminated = gc.is_grasped(self.data, self.ids, dist, gap)
        if terminated:
            reward += self.success_reward
        return reward, terminated

    def _get_info(self):
        dist = float(np.linalg.norm(self.grasp_world() - self.cube_world()))
        gap = self.gripper_gap()
        target, seen = self.detected_world()
        return {
            "dist_to_cube": dist,
            "gripper_gap": gap,
            "grasped": gc.is_grasped(self.data, self.ids, dist, gap),
            "detected": target,
            "seen": seen,
            "true_cube": self.cube_world(),
            "steps": self._step_count,
        }


class RokaePro7PickReal(RokaePro7Pick):
    """Backwards-compatible alias: the pick scene *is* the real-URDF one now.

    ``pro7_pick`` and ``pro7_pick_urdf`` therefore build byte-identical models;
    the alias is kept so existing commands and checkpoints keep working.
    """

    MODEL_PATH = gc.asset_path("rokae_xmate_pro7_pick_real.xml")
