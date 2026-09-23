"""Multi-arm reacher environments for the ``lkwy73_o1`` dual-arm model.

Two 7-DoF arms, one tip site and one mocap target each.  Everything is written
for ``A`` arms rather than two: the arm list is read from the model, so adding a
third arm means editing ``tools/build_dual_arm_model.py``, not this file.

Two tasks share the plumbing:

``dual_arm_reach``
    Each arm gets its own independent target.  This is the "multi-arm control"
    baseline: two 7-DoF controllers in one action vector, no coupling.

``dual_arm_coop``
    The two targets are the two ends of one rigid "bar", so a successful
    episode needs the arms to hold a specific *relative* pose, not just reach
    two unrelated points.  The bar is sampled from a pair of reachable poses
    and only moved by a small rigid transform, so it stays reachable.

Observation (all arms concatenated, then the global blocks)::

    [cos q_a, sin q_a, dq_a, tip_a - target_a] for each arm
    + [tip_0 .. tip_{A-1}]
    + [target_0 .. target_{A-1}]

Action: normalised torque in ``[-1, 1]`` for every arm joint (14 for two arms),
so ``ctrl`` maps straight onto the model's ``<motor>`` actuators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from paths import asset_path


@dataclass(frozen=True)
class Arm:
    """One arm's model-side handles."""

    label: str
    tip_site: str
    actuators: tuple
    joints: tuple


def arms_for(prefixes: Sequence[tuple]) -> tuple:
    """Build the :class:`Arm` list from ``(label, joint prefix)`` pairs."""
    arms = []
    for label, prefix in prefixes:
        arms.append(Arm(
            label=label,
            tip_site=f"tip_{label}",
            actuators=tuple(f"a{prefix}{i}_Joint" for i in range(1, 8)),
            joints=tuple(f"{prefix}{i}_Joint" for i in range(1, 8)),
        ))
    return tuple(arms)


def mirror_safe_limits(left_limits, right_limits, signs) -> np.ndarray:
    """Left-arm joint ranges whose mirrored pose also fits the right arm.

    ``right_limits`` must hold ``q_right = signs * q_left``; both arms are
    symmetric in the CAD but their URDF ranges are not (R2 is [0, 3.14] while
    L2 is [-3.14, 0.15]), so a mirrored pose has to be sampled from the
    intersection.  Returns ``(low, high)`` arrays for the left arm.
    """
    signs = np.asarray(signs)
    left_limits = np.asarray(left_limits)
    right_limits = np.asarray(right_limits)
    low = np.maximum(
        left_limits[:, 0],
        np.where(signs > 0, right_limits[:, 0], -right_limits[:, 1]),
    )
    high = np.minimum(
        left_limits[:, 1],
        np.where(signs > 0, right_limits[:, 1], -right_limits[:, 0]),
    )
    return low, high


class MultiArmReacher(gym.Env):
    """Shared implementation for the dual-arm reach tasks."""

    MODEL_PATH: str = ""
    ARMS: tuple = ()
    CAMERA: str = "cam_iso"

    DEFAULT_MAX_STEPS: int = 200
    DEFAULT_TARGET_RADIUS: float = 0.08

    #: Spread of the start / target pose around a random anchor pose (rad).
    START_STD: float = 0.12
    TARGET_STD: float = 0.30
    #: Anchors are drawn from this fraction of each joint's range.
    CENTER_SCALE: float = 0.45
    #: Uniform noise added to the FK-derived target tip (m).
    TARGET_JITTER: float = 0.02
    #: A reset must start this far from every target, otherwise the episode
    #: would open with a free success (sampling 0.5% of resets hit the radius).
    MIN_START_DIST: float = 0.12
    #: How many times reset() may redraw the poses before accepting a close one.
    RESET_ATTEMPTS: int = 25
    #: Latched arrival: an arm counts as done once it has touched its target,
    #: so the episode ends when every arm has arrived at least once.  The
    #: cooperative task turns this off, because holding the bar *together* is
    #: the point there.
    LATCH_ARRIVAL: bool = True
    #: Smooth closeness shaping: ``SHAPING_SCALE * exp(-dist / SHAPING_LENGTH)``.
    SHAPING_SCALE: float = 0.5
    SHAPING_LENGTH: float = 0.3

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: Optional[int] = None,
        target_radius: Optional[float] = None,
        ctrl_cost: float = 0.0,
        vel_cost: float = 0.0005,
        success_reward: float = 2.0,
        arm_success_reward: float = 1.0,
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
        # One-off credit for a single arm arriving, so the policy is not blind
        # until *both* arms succeed (it is paid once per arm per episode, so it
        # cannot be farmed by parking inside the radius).
        self.arm_success_reward = arm_success_reward

        self.model_path = str(model_path or self.MODEL_PATH)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self._renderer = None

        self.arms = self.ARMS
        if not self.arms:
            raise ValueError("ARMS is empty; subclass before instantiating")
        if self.model.nmocap != len(self.arms):
            raise ValueError(
                f"{self.model_path} has {self.model.nmocap} mocap bodies but "
                f"{len(self.arms)} arms are declared"
            )
        self._resolve_handles()

        self.n_dof = int(self.model.nu)
        if self.n_dof != sum(len(arm.actuators) for arm in self.arms):
            raise ValueError(
                f"{self.model_path} has {self.n_dof} actuators, expected one per "
                "arm joint"
            )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_dof,), dtype=np.float32
        )
        # [cos q, sin q, dq, tip - target] per arm + all tips + all targets
        obs_dim = len(self.arms) * (3 * self.arm_dof + 3) + 6 * len(self.arms)
        high = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self._step_count = 0
        self._targets = np.zeros((len(self.arms), 3))
        self._arm_done = np.zeros(len(self.arms), dtype=bool)

    # ------------------------------------------------------------------ #
    # model handles
    # ------------------------------------------------------------------ #
    def _resolve_handles(self) -> None:
        self.arm_dof = len(self.arms[0].actuators)
        self._actuator_ids = []
        self._qpos_idx = []
        self._dof_idx = []
        self._tip_ids = []
        self._limits = []

        for arm in self.arms:
            actuator_ids = []
            qpos_idx = []
            dof_idx = []
            limits = []
            for name in arm.actuators:
                actuator = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                if actuator < 0:
                    raise ValueError(f"actuator {name!r} missing from {self.model_path}")
                actuator_ids.append(actuator)

            for name in arm.joints:
                joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint < 0:
                    raise ValueError(f"joint {name!r} missing from {self.model_path}")
                qpos_idx.append(int(self.model.jnt_qposadr[joint]))
                dof_idx.append(int(self.model.jnt_dofadr[joint]))
                limits.append(self.model.jnt_range[joint])

            tip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, arm.tip_site)
            if tip < 0:
                raise ValueError(f"site {arm.tip_site!r} missing from {self.model_path}")

            self._actuator_ids.append(np.asarray(actuator_ids))
            self._qpos_idx.append(np.asarray(qpos_idx))
            self._dof_idx.append(np.asarray(dof_idx))
            self._tip_ids.append(tip)
            self._limits.append(np.asarray(limits))

    # ------------------------------------------------------------------ #
    # public accessors (shared with the single-arm tooling)
    # ------------------------------------------------------------------ #
    @property
    def targets(self) -> np.ndarray:
        return self._targets.copy()

    @property
    def target(self) -> np.ndarray:
        """Mean target, so scripts written for one arm still work."""
        return self._targets.mean(axis=0)

    @property
    def tips(self) -> np.ndarray:
        return np.asarray([self.data.site_xpos[i].copy() for i in self._tip_ids])

    @property
    def tip(self) -> np.ndarray:
        return self.tips.mean(axis=0)

    def arm_qpos(self, index: int) -> np.ndarray:
        return self.data.qpos[self._qpos_idx[index]]

    def arm_dq(self, index: int) -> np.ndarray:
        return self.data.qvel[self._dof_idx[index]]

    def arm_dist(self, index: int) -> float:
        return float(np.linalg.norm(self.tips[index] - self._targets[index]))

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._drop_renderer()

        for _ in range(self.RESET_ATTEMPTS):
            starts, anchors, reference = self._sample_poses()
            targets = self._sample_targets(starts, anchors, reference)
            if self._clearance(starts, targets) >= self.MIN_START_DIST:
                break
        self._targets = np.asarray(targets, dtype=np.float64)

        for index, q in enumerate(starts):
            self.data.qpos[self._qpos_idx[index]] = q
            self.data.qvel[self._dof_idx[index]] = 0.0
        self.data.qvel[:] = 0.0
        self._set_targets(self._targets)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        self._arm_done = np.zeros(len(self.arms), dtype=bool)
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        reward, terminated = self._compute_reward(action)
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
    # hooks
    # ------------------------------------------------------------------ #
    def _sample_targets(self, starts, anchors, reference) -> np.ndarray:
        """Targets for every arm; the default is one independent target each."""
        targets = []
        for index, q in enumerate(reference):
            targets.append(self._tip_of(index, q))
        return np.asarray(targets)

    def _sample_poses(self):
        """(starts, anchors, reference poses) for every arm."""
        starts, anchors, reference = [], [], []
        for index in range(len(self.arms)):
            center = self._sample_center(index)
            anchors.append(center)
            starts.append(self._sample_around(index, center, self.START_STD))
            reference.append(self._sample_around(index, center, self.TARGET_STD))
        return starts, anchors, reference

    def _clearance(self, starts, targets) -> float:
        """Smallest tip-to-target distance over the arms at the start pose."""
        distances = []
        for index, q in enumerate(starts):
            tip = self._tip_of(index, q, jitter=False)
            distances.append(float(np.linalg.norm(tip - targets[index])))
        return min(distances) if distances else float("inf")

    def _success(self, dists: np.ndarray) -> bool:
        if self.LATCH_ARRIVAL:
            return bool(np.all(self._arm_done))
        return bool(np.all(dists < self.target_radius))

    def _shaping(self, dist: float) -> float:
        return self.SHAPING_SCALE * float(np.exp(-dist / self.SHAPING_LENGTH))

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _drop_renderer(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _tip_of(self, index: int, q: np.ndarray, jitter: bool = True) -> np.ndarray:
        """Forward kinematics of one arm at ``q``, in world frame."""
        saved_qpos = self.data.qpos[self._qpos_idx[index]].copy()
        self.data.qpos[self._qpos_idx[index]] = q
        mujoco.mj_kinematics(self.model, self.data)
        tip = self.data.site_xpos[self._tip_ids[index]].copy()
        self.data.qpos[self._qpos_idx[index]] = saved_qpos
        if not jitter:
            return tip
        return tip + self.np_random.uniform(
            -self.TARGET_JITTER, self.TARGET_JITTER, size=3
        )

    def _set_targets(self, targets: np.ndarray) -> None:
        for index, position in enumerate(targets):
            self.data.mocap_pos[index] = position
            self.data.mocap_quat[index] = np.array([1.0, 0.0, 0.0, 0.0])

    def _sample_center(self, index: int) -> np.ndarray:
        limits = self._limits[index]
        low = limits[:, 0] * self.CENTER_SCALE
        high = limits[:, 1] * self.CENTER_SCALE
        return self.np_random.uniform(low, high, size=self.arm_dof)

    def _sample_around(self, index: int, center: np.ndarray, std: float) -> np.ndarray:
        limits = self._limits[index]
        q = center + self.np_random.normal(0.0, std, size=self.arm_dof)
        return np.clip(q, limits[:, 0], limits[:, 1])

    def _get_obs(self) -> np.ndarray:
        blocks = []
        for index in range(len(self.arms)):
            q = self.arm_qpos(index)
            blocks.append(np.concatenate([
                np.cos(q), np.sin(q), self.arm_dq(index),
                self.tips[index] - self._targets[index],
            ]))
        blocks.append(self.tips.reshape(-1))
        blocks.append(self._targets.reshape(-1))
        return np.concatenate(blocks).astype(np.float32)

    def _compute_reward(self, action: np.ndarray):
        dists = np.asarray([self.arm_dist(i) for i in range(len(self.arms))])
        ctrl = float(np.sum(action ** 2))
        vel = float(np.sum(np.concatenate(
            [self.arm_dq(i) for i in range(len(self.arms))]
        ) ** 2))
        reward = (
            -float(dists.mean())
            - self.ctrl_cost * ctrl
            - self.vel_cost * vel
            + self._shaping(float(dists.max()))
            + self._coordination_reward()
        )
        arrived = dists < self.target_radius
        newly_done = arrived & ~self._arm_done
        reward += float(newly_done.sum()) * self.arm_success_reward
        self._arm_done = self._arm_done | arrived
        terminated = self._success(dists)
        if terminated:
            reward += self.success_reward
        return reward, terminated

    def _coordination_reward(self) -> float:
        """Extra term for tasks that constrain the arms relative to each other."""
        return 0.0

    def _get_info(self) -> dict:
        dists = np.asarray([self.arm_dist(i) for i in range(len(self.arms))])
        info = {
            "dist_to_target": float(dists.mean()),
            "max_dist": float(dists.max()),
            "tips": self.tips.copy(),
            "targets": self._targets.copy(),
            "tip": self.tip,
            "target": self.target,
            "success": self._success(dists),
            "steps": self._step_count,
        }
        for index, arm in enumerate(self.arms):
            info[f"dist_{arm.label}"] = float(dists[index])
            info[f"tip_{arm.label}"] = self.tips[index].copy()
            info[f"target_{arm.label}"] = self._targets[index].copy()
        return info


class DualArmReach(MultiArmReacher):
    """Both arms reach independent targets (no coordination requirement)."""

    MODEL_PATH = asset_path("dual_arm_reach.xml")
    ARMS = arms_for((("left", "L"), ("right", "R")))

    DEFAULT_MAX_STEPS = 200
    #: 8 cm: the two hands are 0.47 m apart, so this is a genuine "hit".
    DEFAULT_TARGET_RADIUS = 0.08


class DualArmCoopReach(DualArmReach):
    """Both arms must hold one rigid bar: the targets keep a fixed offset.

    The bar is built from *one* reference pose for the left arm plus the mirrored
    pose for the right arm, so the pair is a configuration the two arms can
    actually hold at the same time.  It is then moved by a small rigid transform
    (yaw + translation about the pair's midpoint) to make every episode
    different while staying inside both workspaces.
    """

    # Mirrored joint signs from the left arm to the right arm.  The arms are
    # separated along y, so the symmetry plane is y = 0: reflecting a rotation
    # about axis a by q gives a rotation about (ax, -ay, az) by -q.  Verified
    # against the meshes - a correctly mirrored pose matches to 3 mm while a
    # wrong sign shows up as 75 mm+ of mismatch (see the asset README).
    MIRROR_SIGNS: tuple = (-1, -1, -1, 1, -1, 1, -1)
    #: Tighter anchor range than the independent-target task: it keeps the
    #: symmetric pair inside a plausible bar length (0.4 m - 1.45 m).
    CENTER_SCALE: float = 0.35
    #: The bar has to be held: both tips must be on target *at the same time*,
    #: with a matching relative pose (see ``_success``).
    LATCH_ARRIVAL: bool = False
    #: Maximum rotation / translation applied to the sampled pair.
    PAIR_YAW_RANGE: float = 0.22  # rad
    PAIR_SHIFT: float = 0.04  # m
    #: Accepted distance between the two hands, i.e. the length of the bar.  A
    #: symmetric pair of poses naturally puts the hands 0.4 m to 2.0 m apart
    #: (the arms swing outward together), so the bar is a wide rod rather than
    #: a small object.  Rejection sampling only trims the tails; solvability
    #: never depends on it, because the pair is always reachable by
    #: construction.
    BAR_MIN: float = 0.40
    BAR_MAX: float = 1.45
    #: How far the achieved tip-to-tip vector may deviate from the bar.
    PAIR_TOLERANCE: float = 0.08  # m
    #: Weight of the (negative) bar-length error in the reward.
    #: Kept below the distance weight: with the initial bar error at ~0.7 m a
    #: large coefficient swamps the reach signal and the policy stops learning.
    PAIR_COST: float = 0.5
    #: Holding a shared bar needs a slightly looser tip tolerance than the
    #: independent task (both hands must be right at the same time).
    DEFAULT_TARGET_RADIUS: float = 0.10

    def _sample_targets(self, starts, anchors, reference) -> np.ndarray:
        # Draw poses until the mirrored pair is a plausible two-hand grip: the
        # tips must sit between BAR_MIN and BAR_MAX apart.
        signs = np.asarray(self.MIRROR_SIGNS)
        for _ in range(50):
            left = self._sample_around(0, anchors[0], self.TARGET_STD)
            right = np.clip(signs * left, self._limits[1][:, 0], self._limits[1][:, 1])
            pair = np.asarray([
                self._tip_of(0, left, jitter=False),
                self._tip_of(1, right, jitter=False),
            ])
            if self.BAR_MIN <= float(np.linalg.norm(pair[1] - pair[0])) <= self.BAR_MAX:
                break
        # Kept for diagnostics and tests: setting the arms to these joint
        # values puts the tips on ``achievable_pair``, so the episode is
        # solvable by construction.
        self.reference_pose = np.vstack([left, right])
        self.achievable_pair = pair + self.np_random.uniform(
            -self.TARGET_JITTER, self.TARGET_JITTER, size=(2, 3)
        )
        targets = self.achievable_pair
        centre = targets.mean(axis=0)
        yaw = self.np_random.uniform(-self.PAIR_YAW_RANGE, self.PAIR_YAW_RANGE)
        shift = self.np_random.uniform(-self.PAIR_SHIFT, self.PAIR_SHIFT, size=3)
        shift[2] *= 0.5  # keep the bar near the arms' working height

        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        return (targets - centre) @ rotation.T + centre + shift

    def _bar_vector(self) -> np.ndarray:
        return self.tips[1] - self.tips[0]

    def _target_bar_vector(self) -> np.ndarray:
        return self._targets[1] - self._targets[0]

    def _coordination_reward(self) -> float:
        error = float(np.linalg.norm(self._bar_vector() - self._target_bar_vector()))
        return -self.PAIR_COST * error

    def _success(self, dists: np.ndarray) -> bool:
        bar_error = float(np.linalg.norm(self._bar_vector() - self._target_bar_vector()))
        return bool(np.all(dists < self.target_radius) and bar_error < self.PAIR_TOLERANCE)

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["bar_error"] = float(
            np.linalg.norm(self._bar_vector() - self._target_bar_vector())
        )
        info["bar_length"] = float(np.linalg.norm(self._bar_vector()))
        return info
