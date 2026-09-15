"""Gymnasium environment for a planar two-joint (2-DOF) robotic arm.

The end effector must reach a randomly placed target in the ``xy`` plane.
Physics come from MuJoCo; observation / action / reward logic comes from
:class:`~env.base_reacher.BaseReacher`.
"""

from __future__ import annotations

import numpy as np

from paths import asset_path

from .base_reacher import BaseReacher


class TwoJointReacher(BaseReacher):
    """Reach a random 2D target with a planar two-link arm."""

    MODEL_PATH = asset_path("two_joint_arm.xml")
    TIP_SITE = "tip"
    CAMERA = "cam_xy"
    POS_DIM = 2
    INTERLEAVED_ANGLES = True
    DEFAULT_MAX_STEPS = 120
    DEFAULT_TARGET_RADIUS = 0.08

    #: Initial joint jitter (rad).
    START_STD = 0.2
    #: Target sampling annulus, ``(r_min, r_max)``; the arm is 1.05 m long so
    #: everything in this range is reachable.
    TARGET_RADIUS_RANGE = (0.25, 1.0)

    def _reset_episode(self) -> np.ndarray:
        self.data.qpos[: self.n_dof] = self.np_random.uniform(
            -self.START_STD, self.START_STD, size=self.n_dof
        )
        self.data.qvel[:] = 0.0

        r = self.np_random.uniform(*self.TARGET_RADIUS_RANGE)
        ang = self.np_random.uniform(-np.pi, np.pi)
        return np.array([r * np.cos(ang), r * np.sin(ang)])
