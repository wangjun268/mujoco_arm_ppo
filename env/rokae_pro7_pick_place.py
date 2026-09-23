"""Gymnasium environment: Pro7 grasps the red cube **and places it on the pad**.

``RokaePro7Pick`` stops the episode the moment the jaws pinch the cube, so a
policy trained on it never learns (or even sees) the second half of the task.
This environment runs the *whole* cell: approach -> pinch -> lift -> carry ->
lower -> release, and only terminates when the cube sits on the destination
pad (or the step budget runs out).

Two things are worth knowing about the plant:

* The pinch is weak (~0.3 N, see :mod:`grasp.common`), so the carry has to be
  walked in centimetre waypoints with soft gains or the cube is squeezed out of
  the jaws.  That sub-goal walk lives in the environment
  (:class:`grasp.common.PlacePlanner`) and is published in the observation
  (``goal_rel``), which is exactly what the expert teacher servos to: the
  policy learns the closed-loop controller for the approach and the pinch.
* Carrying is done by the plant (**transport mode**): the torque that holds the
  quasi-static carry together is only ~1-2 % of the actuator range, i.e. far
  below the ~1.4 % action noise an imitation policy has, so a *learned*
  friction grasp squeezes the cube out the moment the arm moves (measured: a
  behaviour-cloned policy with 2e-4 action MSE places 0/6, the scripted
  transport 10/10).  While the jaws hold the cube the environment therefore
  tracks the plan's sub-goal with the gentle transport gains itself; the policy
  still owns the approach, the pinch and the *release* (it decides when to let
  go), which is what the pick / place metrics are scored on.
* ``action_scale`` scales the arm commands, which slows the whole motion down.
  It is applied by the plant, so teacher data and policy rollouts see the same
  dynamics (see ``RokaePro7Pick``: below ~0.7 the pinch itself starts to miss).
"""

from __future__ import annotations

from typing import Optional

import mujoco
import numpy as np
from gymnasium import spaces

import grasp.common as gc

from .rokae_pro7_pick import RokaePro7Pick


#: How the plant reads the gripper command: the position servo only squeezes
#: with ``kp * (travel - measured)`` = 0.3 N at full close, so a policy that
#: outputs -0.95 instead of -1.0 already loses a quarter of the pinch force and
#: drops the cube.  Any negative command therefore means "close hard" (2x
#: over-travel saturates at the stop) and only a positive command opens.
GRIPPER_CLOSE_GAIN = 2.0

#: Half-extents of the source bench top (see ``assets/..._pick_real.xml``), used
#: to decide when a cube has been knocked out of reach.
BENCH_CENTRE = np.array([gc.CUBE_X, 0.0])
BENCH_HALF = np.array([0.44, 0.34])


class RokaePro7PickPlace(RokaePro7Pick):
    """Vision-driven pick & place task (observation size ``3 * 7 + 17 = 38``)."""

    #: ``3 * n_arm`` joint terms + 9 grasp terms (see ``RokaePro7Pick``) plus the
    #: pick & place block: holding, sub-goal error (3), cube-to-pad error (3) and
    #: the plan's "parked on the pad" flag (the signal to let go of the cube).
    OBS_DIM = 3 * gc.N_ARM + 9 + 8

    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 3000,
        vel_cost: float = 0.0005,
        success_reward: float = 2.0,
        place_reward: float = 4.0,
        obs_target: str = "vision",
        cube_spread: tuple = gc.CUBE_SPREAD,
        model_path: Optional[str] = None,
        action_scale: float = 1.0,
        transport_assist: bool = True,
        plan: Optional[gc.PlacePlanner] = None,
    ):
        super().__init__(
            render_mode=render_mode,
            max_steps=max_steps,
            vel_cost=vel_cost,
            success_reward=success_reward,
            obs_target=obs_target,
            cube_spread=cube_spread,
            model_path=model_path,
            action_scale=action_scale,
        )
        self.place_reward = place_reward
        #: Hand the *carry* to the plant's gentle transport servo instead of the
        #: policy's arm command (see the module docstring: the transport torque
        #: is far below what an imitation policy can reproduce).
        self.transport_assist = bool(transport_assist)
        self.plan = plan if plan is not None else gc.PlacePlanner()
        #: Latched "the jaws are around the cube" flag: the raw contact test
        #: flickers while the pinch settles, but the phase must not.
        self.holding = False
        #: Latched "the cube was pinched at some point in this episode".
        self.picked = False
        #: Current sub-goal (world): the plan's next waypoint, or the cube.
        self.goal = np.zeros(3)
        #: Latched "the jaws have started closing" flag.  Re-detecting the cube
        #: through the pinch makes the servo chase a target that moves as the
        #: hand occludes (and pushes) it, which wedges the cube out past the
        #: fingertips - the same trap ``grasp.common.pick_and_place`` freezes its
        #: target for.
        self.closing = False
        self._pinch_paid = False
        #: Set when the episode ended because the cube left the source bench.
        self.lost = False
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32
        )

    # ------------------------------------------------------------------ #
    # public accessors (teacher, trainers, viewers)
    # ------------------------------------------------------------------ #
    @property
    def phase(self) -> str:
        """``reach`` before the pinch, then the plan's own phase name."""
        return "reach" if not self.holding else self.plan.phase

    def pad_world(self) -> np.ndarray:
        """Where the cube has to end up (world)."""
        return np.asarray(self.plan.target, dtype=float)

    def released(self) -> bool:
        """True once the cube rests on the pad and the jaws are open."""
        return gc.cube_on_destination(self.data, self.ids) and not self._pinched()

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, _info = super().reset(seed=seed, options=options)
        self.plan.reset(self.grasp_world())
        self.holding = False
        self.picked = False
        self.closing = False
        self._pinch_paid = False
        self.lost = False
        self.goal = self._reach_goal()
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        if self.holding and self.transport_assist:
            arm, _close, _dist = gc.servo_command(
                self.model, self.data, self.goal, self.ids,
                kp=gc.TRANSPORT_KP, kd=gc.TRANSPORT_KD,
                force_clamp=gc.TRANSPORT_FORCE,
            )
        else:
            arm = action[: self.n_arm]
        self.data.ctrl[: self.n_arm] = self.action_scale * arm
        close = np.clip(
            0.5 * (1.0 - action[self.n_arm]) * GRIPPER_CLOSE_GAIN, 0.0, 1.0
        )
        gc.set_hand(self.data, self.ids, close)
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        self._advance_plan()
        reward, terminated = self._compute_reward()
        if not terminated and self._cube_off_bench():
            # A cube that has been shoved off the bench cannot be picked any
            # more.  Ending the episode keeps the DAgger buffer on-task: those
            # unreachable states are exactly what made the online loop diverge.
            self.lost = True
            reward -= 1.0
            terminated = True
        truncated = self._step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _pinched(self) -> bool:
        dist = float(np.linalg.norm(self.grasp_world() - self.cube_world()))
        return gc.is_grasped(self.data, self.ids, dist, self.gripper_gap())

    def _reach_goal(self) -> np.ndarray:
        """While the cube is not held the sub-goal is (a frozen) cube position."""
        if self.closing:
            return self.goal.copy()
        return np.asarray(gc.expert_target(self), dtype=float)

    def _cube_off_bench(self) -> bool:
        """True once an un-held cube has left the source bench (unreachable)."""
        if self.holding:
            return False
        cube = self.cube_world()
        if cube[2] < gc.TABLE_Z - 0.02:
            return True
        return bool(np.any(np.abs(cube[:2] - BENCH_CENTRE) > BENCH_HALF))

    def _advance_plan(self) -> None:
        """Latch the phase and refresh the sub-goal after one physics step."""
        dist = float(np.linalg.norm(self.grasp_world() - self.cube_world()))
        if gc.is_grasped(self.data, self.ids, dist, self.gripper_gap()):
            self.holding = True
            self.picked = True
        if not self.holding:
            if dist < gc.CLOSE_EPS:
                self.closing = True
            elif self.closing and dist > gc.CLOSE_RETRY_DIST:
                # The pinch missed and the cube was knocked clear: open the jaws
                # and re-detect, instead of shoving the frozen target around.
                self.closing = False
            self.plan.reset(self.grasp_world())
            self.goal = self._reach_goal()
            return
        self.goal = self.plan.update(self.grasp_world())

    def _get_obs(self) -> np.ndarray:
        base = super()._get_obs()
        pad_rel = self.pad_world() - self.cube_world()
        goal_rel = self.goal - self.grasp_world()
        return np.concatenate(
            [base, [float(self.holding)], goal_rel, pad_rel, [float(self.plan.done)]]
        ).astype(np.float32)

    def _compute_reward(self):
        cube = self.cube_world()
        grasp = self.grasp_world()
        vel = float(
            np.sum(self.data.qvel[: self.n_arm] ** 2) + self.gripper_vel() ** 2
        )
        gap = self.gripper_gap()

        if not self.holding:
            dist = float(np.linalg.norm(grasp - cube))
            reward = -dist - self.vel_cost * vel + 0.5 * np.exp(-dist / 0.3)
            if dist < 0.10:
                reward += 0.05 * max(0.0, (gc.GRIP_OPEN - gap) / gc.GRIP_OPEN)
            if self._pinched() and not self._pinch_paid:
                self._pinch_paid = True
                reward += self.success_reward  # one-off "picked it up" bonus
            return reward, False

        dist = float(np.linalg.norm(cube - self.pad_world()))
        reward = -dist - self.vel_cost * vel
        placed = self.released()
        if placed:
            reward += self.place_reward
        return reward, placed

    def _get_info(self):
        info = super()._get_info()
        info.update(
            {
                "picked": self.picked,
                "placed": self.released(),
                "place_dist": float(
                    np.linalg.norm(self.cube_world() - self.pad_world())
                ),
                "holding": self.holding,
                "phase": self.phase,
                "lost": self.lost,
                "success": self.released(),
            }
        )
        return info
