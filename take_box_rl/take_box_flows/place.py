"""流程 5 / place：把箱子放回台面（沿 -z 下降回到台面高度）。

单独运行会自动先 prepare + grasp + lift：

    python3 -m take_box_flows.place
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import add_common_args, make_robot, say, violations
from .grasp import DEFAULT_STEPS as GRASP_STEPS, grasp
from .lift import DEFAULT_HEIGHT, DEFAULT_STEPS as LIFT_STEPS, lift
from .prepare import DEFAULT_STEPS as PREPARE_STEPS, prepare

STAGE = "place"
DEFAULT_STEPS = 100
#: 放下时停在台面上方一点点，避免和台面几何体穿插
DEFAULT_CLEARANCE = 0.004


def place(robot, clearance: float = DEFAULT_CLEARANCE, steps: int = DEFAULT_STEPS):
    """保持当前 x/y，把箱子降到台面（``robot.ready_box_pos`` 的高度）。"""
    target = robot.box_pos.copy()
    target[2] = robot.ready_box_pos[2] + clearance
    return robot.goto_box_pose(target, yaw=0.0, steps=steps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args(argv)

    env, robot = make_robot(strict=args.strict, tilt_limit_deg=args.tilt_limit)
    say(STAGE, prepare(robot, PREPARE_STEPS), "（prepare）")
    say(STAGE, grasp(robot, GRASP_STEPS), "（grasp）")
    say(STAGE, lift(robot, DEFAULT_HEIGHT, LIFT_STEPS), "（lift）")
    state = place(robot, steps=args.steps or DEFAULT_STEPS)
    say(STAGE, state, f"箱高 {state.box_pos[2]:.3f} m  {violations(robot)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
