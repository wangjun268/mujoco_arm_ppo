"""流程 1 / prepare：回到 IK 解出来的预抓取位姿。

    python3 -m take_box_flows.prepare
"""

from __future__ import annotations

import argparse

from .common import add_common_args, make_robot, say, violations

STAGE = "prepare"
DEFAULT_STEPS = 80


def prepare(robot, steps: int = DEFAULT_STEPS):
    """两臂回到箱子两侧的抓取位（掌心贴着箱面，~2 mm 间隙）。"""
    return robot.home(steps=steps)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    args = parser.parse_args(argv)

    env, robot = make_robot(strict=args.strict, tilt_limit_deg=args.tilt_limit)
    state = prepare(robot, args.steps or DEFAULT_STEPS)
    say(STAGE, state, violations(robot))
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
