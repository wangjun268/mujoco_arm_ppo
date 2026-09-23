"""每个搬运流程单独跑，以及完整流程串起来跑。"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from take_box_flows import carry, grasp, lift, place, prepare, release  # noqa: E402
from take_box_flows.common import make_robot  # noqa: E402


@pytest.fixture
def robot():
    env, interface = make_robot()
    yield interface
    env.close()


def test_prepare_puts_both_palms_on_the_box_faces(robot):
    state = prepare(robot, steps=60)
    assert state.hand_errors.max() < 0.005
    assert not robot.violation_log


def test_grasp_latches_after_prepare(robot):
    prepare(robot, steps=60)
    state = grasp(robot, steps=40)
    assert state.grasped
    assert not robot.violation_log


def test_lift_carry_place_release_chain(robot):
    prepare(robot, steps=60)
    grasp(robot, steps=40)
    lifted = lift(robot, height=0.10, steps=70)
    assert lifted.lift > 0.06, "the box has to leave the table"
    assert lifted.tilt_deg < 5.0, "the box has to stay level while lifted"

    carried = carry(robot, forward=0.08, steps=70)
    assert carried.box_pos[0] - lifted.box_pos[0] > 0.04, "the arm moves forward"
    assert carried.tilt_deg < 5.0

    placed = place(robot, steps=70)
    assert placed.box_pos[2] < lifted.box_pos[2] - 0.03, "the box goes back down"
    assert placed.box_pos[2] > robot.env.box_home[2] - 0.01, "but not into the table"

    released = release(robot, steps=25)
    assert not robot.env.grasped
    assert not robot.violation_log
