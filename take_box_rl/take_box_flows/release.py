"""流程 6 / release：松手（张开手指，箱子留在原地）。

单独运行会自动先 prepare + grasp + lift + place：

    python3 -m take_box_flows.release
"""

from __future__ import annotations

import argparse

from .common import add_common_args, make_robot, say, violations
from .grasp import DEFAULT_STEPS as GRASP_STEPS, grasp
from .lift import DEFAULT_HEIGHT, DEFAULT_STEPS as LIFT_STEPS, lift
from .place import DEFAULT_STEPS as PLACE_STEPS, place
from .prepare import DEFAULT_STEPS as PREPARE_STEPS, prepare

STAGE = "release"
DEFAULT_STEPS = 30


def release(robot, steps: int = DEFAULT_STEPS):
    """张开双手；箱子不再跟着走（``open_hands`` 会解除锁存）。"""
    return robot.open_hands(steps=steps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args(argv)

    env, robot = make_robot(strict=args.strict, tilt_limit_deg=args.tilt_limit)
    say(STAGE, prepare(robot, PREPARE_STEPS), "（prepare）")
    say(STAGE, grasp(robot, GRASP_STEPS), "（grasp）")
    say(STAGE, lift(robot, DEFAULT_HEIGHT, LIFT_STEPS), "（lift）")
    say(STAGE, place(robot, steps=PLACE_STEPS), "（place）")
    state = release(robot, args.steps or DEFAULT_STEPS)
    say(STAGE, state, f"grasped={robot.env.grasped}  {violations(robot)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
