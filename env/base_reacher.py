"""Shared MuJoCo / Gymnasium plumbing for every reacher environment.

A "reacher" in this project is always the same task:

* a MuJoCo model with ``n`` actuated hinge joints and exactly one ``mocap``
  body used as the target (massless, non-colliding),
* a tip / tool site that must be driven onto that target,
* observation ``[cos q, sin q, dq, tip - target, target]``,
* action = normalised torque in ``[-1, 1]`` for each joint,
* reward  = ``-dist - ctrl_cost * ||ctrl||^2 - vel_cost * ||qvel||^2
  + shaping(dist) + success_reward`` on a hit.

Implementing that once is what keeps the mesh models from drifting apart: a
subclass only picks the model, the tip site and how the start pose and target
are sampled.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


class BaseReacher(gym.Env):
    """Common reacher environment; subclasses supply model + sampling."""

    #: Absolute path of the MuJoCo XML describing the arm (set by subclasses).
    MODEL_PATH: str = ""
    #: Name of the site whose world position is the end effector.
    TIP_SITE: str = "tip"
    #: Camera used by :meth:`render`.
    CAMERA: str = "cam_iso"
    #: Target / end-effector dimensionality (every current model is spatial).
    POS_DIM: int = 3

    DEFAULT_MAX_STEPS: int = 120
    DEFAULT_TARGET_RADIUS: float = 0.08

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        target_radius: Optional[float] = None,
        ctrl_cost: float = 0.0,
        vel_cost: float = 0.0005,
        success_reward: float = 2.0,
        model_path: Optional[str] = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = self.DEFAULT_MAX_STEPS if max_steps is None else max_steps
        self.target_radius = (
            self.DEFAULT_TARGET_RADIUS if target_radius is None else target_radius
        )
        self.ctrl_cost = ctrl_cost
        self.vel_cost = vel_cost
        self.success_reward = success_reward

        self.model_path = str(model_path or self.MODEL_PATH)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self._renderer = None

        self.n_dof = int(self.model.nu)  # actuators == actuated hinge joints
        if self.model.nmocap < 1:
            raise ValueError(f"{self.model_path} has no mocap body for the target")
        self._tip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, self.TIP_SITE
        )
        if self._tip_id < 0:
            raise ValueError(f"site {self.TIP_SITE!r} missing from {self.model_path}")
        # Mocap bodies are not addressable via mjtObj; every reacher model has
        # exactly one (the target), so it is index 0.
        self._target_id = 0

        self._step_count = 0
        self._target = np.zeros(self.POS_DIM)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_dof,), dtype=np.float32
        )
        # [cos q, sin q, dq, tip - target, target]
        obs_dim = 3 * self.n_dof + 2 * self.POS_DIM
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # public accessors
    # ------------------------------------------------------------------ #
    @property
    def target(self) -> np.ndarray:
        """Current target position in world coordinates (``POS_DIM`` dims)."""
        return self._target.copy()

    @property
    def tip(self) -> np.ndarray:
        """Current end-effector position in world coordinates."""
        return self._get_tip()

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._drop_renderer()

        self._target = np.asarray(self._reset_episode(), dtype=np.float64)
        self._set_target(self._target)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        reward, terminated = self._compute_reward()
        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode is None:
            raise RuntimeError("render() called without render_mode='rgb_array'")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera=self.CAMERA)
        return self._renderer.render()

    def close(self):
        self._drop_renderer()

    # ------------------------------------------------------------------ #
    # hooks for subclasses
    # ------------------------------------------------------------------ #
    def _reset_episode(self) -> np.ndarray:
        """Set ``qpos``/``qvel`` for a fresh episode and return the target.

        Must draw every random number from ``self.np_random`` so that a fixed
        ``seed`` reproduces an episode exactly.
        """
        raise NotImplementedError

    def _shaping(self, dist: float) -> float:
        """Extra smooth reward term; 0 for the small arms."""
        return 0.0

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _drop_renderer(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _set_target(self, target: np.ndarray):
        pos = np.zeros(3)
        pos[: self.POS_DIM] = target
        self.data.mocap_pos[self._target_id] = pos
        self.data.mocap_quat[self._target_id] = np.array([1.0, 0.0, 0.0, 0.0])

    def _get_tip(self) -> np.ndarray:
        return self.data.site_xpos[self._tip_id][: self.POS_DIM].copy()

    def _get_obs(self) -> np.ndarray:
        q = self.data.qpos[: self.n_dof]
        dq = self.data.qvel[: self.n_dof]
        diff = self._get_tip() - self._target
        return np.concatenate(
            [self._angle_obs(q), dq, diff, self._target]
        ).astype(np.float32)

    def _angle_obs(self, q: np.ndarray) -> np.ndarray:
        """``cos``/``sin`` joint features, block layout ``[cos q..., sin q...]``."""
        return np.concatenate([np.cos(q), np.sin(q)])

    def _compute_reward(self):
        dist = float(np.linalg.norm(self._get_tip() - self._target))
        ctrl = float(np.sum(self.data.ctrl ** 2))
        vel = float(np.sum(self.data.qvel ** 2))
        reward = (
            -dist
            - self.ctrl_cost * ctrl
            - self.vel_cost * vel
            + self._shaping(dist)
        )
        terminated = dist < self.target_radius
        if terminated:
            reward += self.success_reward
        return reward, terminated

    def _get_info(self):
        tip = self._get_tip()
        dist = float(np.linalg.norm(tip - self._target))
        return {
            "dist_to_target": dist,
            "tip": tip,
            "target": self._target.copy(),
            "success": bool(dist < self.target_radius),
            "steps": self._step_count,
        }
