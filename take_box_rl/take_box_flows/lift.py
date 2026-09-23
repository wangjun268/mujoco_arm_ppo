"""流程 3 / lift：把箱子抬起来（默认 0.09 m；本工位端平上限 ~0.10 m）。

单独运行会自动先 prepare + grasp：

    python3 -m take_box_flows.lift --height 0.09

请求超过 ~0.10 m 时腕关节 R6/R7 先顶到限位，箱子会被拖歪——脚本会打印告警，
打印的抬升量是**实际**值（通常是指令的 ~75%）。
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import add_common_args, make_robot, say, violations
from .grasp import DEFAULT_STEPS as GRASP_STEPS, grasp
from .prepare import DEFAULT_STEPS as PREPARE_STEPS, prepare

STAGE = "lift"
DEFAULT_STEPS = 80
#: 这个工位能"端平"抬起的上限（实测）：再高的话 R6/R7 腕关节先顶到限位，
#: 箱子会明显倾斜。想要更高就得放宽腕关节限位或改箱位，见 README。
MAX_LEVEL_LIFT = 0.10
DEFAULT_HEIGHT = 0.09


def lift(robot, height: float = DEFAULT_HEIGHT, steps: int = DEFAULT_STEPS):
    """竖直抬起：掌心带着箱子沿 +z 走 ``height`` 米。"""
    target = robot.ready_box_pos + np.array([0.0, 0.0, height])
    state = robot.goto_box_pose(target, yaw=0.0, steps=steps)
    if state.tilt_deg > robot.limits.tilt_max_deg * 0.5:
        print(f"  ! 抬起 {height * 1000:.0f} mm 后箱体倾角 {state.tilt_deg:.1f}°："
              f"超过本工位能端平抬起的 ~{MAX_LEVEL_LIFT * 1000:.0f} mm，"
              f"腕关节 R6/R7 已经跟不上（实际抬升 {state.lift * 1000:+.0f} mm）")
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT,
                        help="抬起高度（m）")
    args = parser.parse_args(argv)

    env, robot = make_robot(strict=args.strict, tilt_limit_deg=args.tilt_limit)
    say(STAGE, prepare(robot, PREPARE_STEPS), "（prepare）")
    say(STAGE, grasp(robot, GRASP_STEPS), "（grasp）")
    state = lift(robot, args.height, args.steps or DEFAULT_STEPS)
    say(STAGE, state, f"抬起 {state.lift * 1000:+.0f} mm  {violations(robot)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
