"""Regression tests for the scripted carry interface (``take_box_env/box_api.py``)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from take_box_env.box_api import (  # noqa: E402
    BoxCarryInterface, ConstraintViolation, Waypoint,
)
from take_box_env.take_box import SIDES, TakeBoxEnv  # noqa: E402


@pytest.fixture
def robot():
    env = TakeBoxEnv(reset_noise=0.0)
    env.reset(seed=0)
    interface = BoxCarryInterface(env, strict=True)
    #: same relaxed tilt budget as demo_plan.py: the lateral grasp sits at the
    #: edge of the right wrist's roll range (see README "现状与限制")
    interface.limits.tilt_max_deg = 25.0
    yield interface
    env.close()


@pytest.fixture
def grasped(robot):
    robot.home(steps=60)
    robot.close_hands(steps=40)
    assert robot.env.grasped
    return robot


def test_home_puts_both_palms_on_the_box_faces(robot):
    state = robot.home(steps=60)
    assert state.hand_errors.max() < 0.005
    assert state.tilt_deg < 1.0
    assert not robot.violation_log


def test_close_hands_latches_the_grasp(grasped):
    state = grasped.state()
    assert state.grasped
    assert state.hand_errors.max() < 0.005
    assert not grasped.violation_log


def test_goto_box_pose_lifts_and_keeps_the_box_level(grasped):
    base = grasped.ready_box_pos
    lifted = grasped.goto_box_pose(base + np.array([0.0, 0.0, 0.10]), steps=80)
    assert lifted.box_pos[2] - base[2] > 0.07
    assert lifted.tilt_deg < 15.0
    assert lifted.hand_errors.max() < 0.02

    offset = np.array([0.09, 0.06, 0.10])
    carried = grasped.goto_box_pose(base + offset, yaw=0.10, steps=90)
    assert np.linalg.norm(carried.box_pos - (base + offset)) < 0.03
    assert abs(carried.yaw_deg - np.degrees(0.10)) < 10.0
    assert carried.tilt_deg < 15.0
    assert not grasped.violation_log


def test_plan_returns_one_state_per_step(grasped):
    base = grasped.ready_box_pos
    trace = grasped.plan([
        Waypoint(box=base + np.array([0.0, 0.0, 0.02]), steps=20, label="place"),
    ])
    assert len(trace) == 20
    assert trace[-1].box_pos[2] - base[2] > 0.0
    assert not grasped.violation_log


def test_open_hands_releases_the_box(grasped):
    state = grasped.open_hands(steps=30)
    assert not grasped.env.grasped
    assert state.grip.max() < 0.5


def test_check_command_rejects_unsafe_waypoints(grasped):
    base = grasped.ready_box_pos
    assert grasped.check_command(Waypoint(box=base)) == []
    below_the_table = Waypoint(box=base + np.array([0.0, 0.0, -0.5]))
    assert grasped.check_command(below_the_table)
    with pytest.raises(ConstraintViolation):
        grasped.plan([below_the_table])
    with pytest.raises(ConstraintViolation):
        grasped.plan([Waypoint(box=base, yaw=3.0)])  # degrees/radians mix-up


def test_strict_mode_raises_on_a_violating_step(grasped):
    grasped.limits.height_min = 0.90  # above the box: every step now violates
    with pytest.raises(ConstraintViolation):
        grasped.servo_step({side: grasped.env.ready_qpos[side] for side in SIDES})
    assert grasped.violation_log, "the violation must be logged as well"


def test_non_strict_mode_only_logs():
    env = TakeBoxEnv(reset_noise=0.0)
    env.reset(seed=0)
    try:
        robot = BoxCarryInterface(env, strict=False)
        robot.limits.height_min = 0.90
        robot.servo_step({side: env.ready_qpos[side] for side in SIDES})
        assert robot.violation_log
    finally:
        env.close()
