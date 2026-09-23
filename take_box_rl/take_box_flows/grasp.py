"""流程 2 / grasp：闭手抓住箱子（掌心到位即锁存，不需要接触力）。

单独运行会自动先做 prepare：

    python3 -m take_box_flows.grasp
"""

from __future__ import annotations

import argparse

from .common import add_common_args, make_robot, say, violations
from .prepare import DEFAULT_STEPS as PREPARE_STEPS, prepare

STAGE = "grasp"
DEFAULT_STEPS = 40


def grasp(robot, steps: int = DEFAULT_STEPS):
    """两手掌心贴在箱面上、抓取通道闭到 1.0 → 锁存抓取。"""
    return robot.close_hands(steps=steps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args(argv)

    env, robot = make_robot(strict=args.strict, tilt_limit_deg=args.tilt_limit)
    say(STAGE, prepare(robot, PREPARE_STEPS), "（prepare）")
    state = grasp(robot, args.steps or DEFAULT_STEPS)
    say(STAGE, state, f"grasped={robot.env.grasped}  {violations(robot)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
