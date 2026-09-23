"""流程 4 / carry：托着箱子向前搬运（机械臂沿 +x 移动，可选横向偏移与偏航）。

这个工位里机械臂在 x=0，箱子在 x=+0.42，所以"向前"= 让箱子继续沿 +x 走。

单独运行会自动先 prepare + grasp + lift：

    python3 -m take_box_flows.carry --forward 0.10
    python3 -m take_box_flows.carry --forward 0.10 --side 0.06 --yaw 0.10
"""

from __future__ import annotations

import argparse

import numpy as np

from .common import add_common_args, make_robot, say, violations
from .grasp import DEFAULT_STEPS as GRASP_STEPS, grasp
from .lift import DEFAULT_HEIGHT, DEFAULT_STEPS as LIFT_STEPS, lift
from .prepare import DEFAULT_STEPS as PREPARE_STEPS, prepare

STAGE = "carry"
DEFAULT_STEPS = 90
#: 向前搬多远。箱子在 x≈0.42，再往前(到 ~0.6)腕关节就很难保持箱体水平了
DEFAULT_FORWARD = 0.10


def carry(robot, forward: float = DEFAULT_FORWARD, side: float = 0.0,
          yaw: float = 0.0, steps: int = DEFAULT_STEPS):
    """把箱子从当前位姿沿 +x 搬 ``forward`` 米（``side`` 为横向偏移，``yaw`` 为偏航）。"""
    start = robot.box_pos
    target = start + np.array([forward, side, 0.0])
    state = robot.goto_box_pose(target, yaw=yaw, steps=steps)
    moved = float(state.box_pos[0] - start[0])
    if state.tilt_deg > robot.limits.tilt_max_deg * 0.5:
        print(f"  ! 前移 {moved * 1000:+.0f} mm 后箱体倾角 {state.tilt_deg:.1f}°："
              f"这一段腕关节已经跟不上，箱子在被拖歪（试小一点的前移量）")
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--forward", type=float, default=DEFAULT_FORWARD, help="向前距离（m）")
    parser.add_argument("--side", type=float, default=0.0, help="横向偏移（m）")
    parser.add_argument("--yaw", type=float, default=0.0, help="偏航（rad）")
    args = parser.parse_args(argv)

    env, robot = make_robot(strict=args.strict, tilt_limit_deg=args.tilt_limit)
    say(STAGE, prepare(robot, PREPARE_STEPS), "（prepare）")
    say(STAGE, grasp(robot, GRASP_STEPS), "（grasp）")
    say(STAGE, lift(robot, DEFAULT_HEIGHT, LIFT_STEPS), f"（lift {DEFAULT_HEIGHT:.2f} m）")
    state = carry(robot, args.forward, args.side, args.yaw, args.steps or DEFAULT_STEPS)
    say(STAGE, state, f"前移 {args.forward * 1000:+.0f} mm  {violations(robot)}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
