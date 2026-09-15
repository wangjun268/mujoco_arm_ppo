"""Gymnasium environments: Rokae xMate arms reach a 3D target.

The target is sampled from the reachable workspace: a random joint pose is
forward-kinematic-ed to get its tool-tip position (plus a little noise), so any
target is guaranteed reachable.  The tool tip is deliberately off the last roll
axis so that *every* joint affects the tip position.

The DOF, joint ranges and tool site are read straight out of the MuJoCo model,
so the same class drives both the 6-DOF ER3 and the 7-DOF Pro7 arms.
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np

from paths import asset_path

from .base_reacher import BaseReacher


class RokaeReacher(BaseReacher):
    """Generic Rokae arm reacher; override ``MODEL_PATH`` for another model."""

    MODEL_PATH = asset_path("rokae_xmate_er3.xml")
    TIP_SITE = "tool"
    CAMERA = "cam_iso"
    POS_DIM = 3
    DEFAULT_MAX_STEPS = 200
    DEFAULT_TARGET_RADIUS = 0.12

    #: Spread of the start pose / target pose around a random "anchor" pose.
    START_STD = 0.15
    TARGET_STD = 0.35
    #: Anchor poses are drawn from this fraction of each joint's range.
    CENTER_SCALE = 0.45
    #: Uniform noise added to the FK-derived target tip (m).
    TARGET_JITTER = 0.03
    #: Smooth closeness shaping: ``SHAPING_SCALE * exp(-dist / SHAPING_LENGTH)``.
    #: Keeps the gradient alive far from the target, which is what makes the
    #: 6-/7-DOF reaches learnable at all.
    SHAPING_SCALE = 0.5
    SHAPING_LENGTH = 0.3

    def __init__(
        self,
        *args,
        start_std: Optional[float] = None,
        target_std: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._start_std = self.START_STD if start_std is None else start_std
        self._target_std = self.TARGET_STD if target_std is None else target_std
        # Only the actuated joints are sampled; any extra DoF stay untouched.
        self._limits = self.model.jnt_range[: self.n_dof].copy()

    # ------------------------------------------------------------------ #
    def _reset_episode(self) -> np.ndarray:
        # Sample a random "anchor" pose, then put the start and (FK-derived)
        # target near it -> moderately reachable distances, easier to learn.
        center = self._sample_center()
        q_start = self._sample_around(center, self._start_std)
        q_target = self._sample_around(center, self._target_std)

        # FK the target pose to get a reachable tip location.
        self.data.qpos[: self.n_dof] = q_target
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        tip = self.data.site_xpos[self._tip_id][:3]
        target = tip + self.np_random.uniform(
            -self.TARGET_JITTER, self.TARGET_JITTER, size=3
        )

        # Start from a different random pose so the arm must actually move.
        self.data.qpos[: self.n_dof] = q_start
        self.data.qvel[:] = 0.0
        return target

    def _shaping(self, dist: float) -> float:
        return self.SHAPING_SCALE * float(np.exp(-dist / self.SHAPING_LENGTH))

    # ------------------------------------------------------------------ #
    def _sample_center(self) -> np.ndarray:
        lo, hi = self._limits[:, 0] * self.CENTER_SCALE, self._limits[:, 1] * self.CENTER_SCALE
        return self.np_random.uniform(lo, hi, size=self.n_dof)

    def _sample_around(self, center: np.ndarray, std: float) -> np.ndarray:
        q = center + self.np_random.normal(0.0, std, size=self.n_dof)
        lo, hi = self._limits[:, 0], self._limits[:, 1]
        return np.clip(q, lo, hi)


class RokaePro7Reacher(RokaeReacher):
    """Rokae xMate Pro7 7-DOF arm reacher."""

    MODEL_PATH = asset_path("rokae_xmate_pro7.xml")
    # The Pro7 reach is ~1.4 m (much longer than the ER3), so give the agent a
    # shorter start-target distance and a slightly looser success radius.
    START_STD = 0.10
    TARGET_STD = 0.24
    DEFAULT_MAX_STEPS = 250
    DEFAULT_TARGET_RADIUS = 0.15


class RokaePro7RealReacher(RokaeReacher):
    """Rokae xMate Pro7 7-DOF reacher using the *real* URDF STL meshes."""

    MODEL_PATH = asset_path("rokae_xmate_pro7_real.xml")
    START_STD = 0.10
    TARGET_STD = 0.24
    DEFAULT_MAX_STEPS = 250
    DEFAULT_TARGET_RADIUS = 0.15
