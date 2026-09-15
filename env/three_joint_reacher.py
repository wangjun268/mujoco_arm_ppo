"""Gymnasium environment for a 3-DOF spatial arm reaching a 3D target.

Joints: yaw (base, about world-z), shoulder pitch (about local-y), elbow pitch
(about local-y).  The tip is a point on a 3-link chain that must reach a random
point inside a reachable spherical cone.
"""

from __future__ import annotations

import numpy as np

from paths import asset_path

from .base_reacher import BaseReacher


class ThreeJointReacher(BaseReacher):
    """Reach a random 3D target inside a reachable spherical cone."""

    MODEL_PATH = asset_path("three_joint_arm.xml")
    TIP_SITE = "tip"
    CAMERA = "cam_iso"
    POS_DIM = 3
    DEFAULT_MAX_STEPS = 150
    DEFAULT_TARGET_RADIUS = 0.10

    #: Initial joint jitter (rad).
    START_STD = 0.1
    #: Target sampling sphere: radius and polar angle (from +z) ranges.
    TARGET_RADIUS_RANGE = (0.25, 0.80)
    TARGET_POLAR_RANGE = (0.05, 1.0)

    def _reset_episode(self) -> np.ndarray:
        self.data.qpos[: self.n_dof] = self.np_random.uniform(
            -self.START_STD, self.START_STD, size=self.n_dof
        )
        self.data.qvel[:] = 0.0

        r = self.np_random.uniform(*self.TARGET_RADIUS_RANGE)
        phi = self.np_random.uniform(*self.TARGET_POLAR_RANGE)  # polar from +z
        theta = self.np_random.uniform(-np.pi, np.pi)
        return np.array(
            [
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ]
        )
