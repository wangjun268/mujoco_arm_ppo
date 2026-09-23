"""完整搬运流程：prepare → grasp → lift → carry（向前）→ place → release。

    python3 -m take_box_flows.full_task                      # 打印每步状态
    python3 -m take_box_flows.full_task --video              # 同时存 runs/take_box/full_task.gif
    python3 -m take_box_flows.full_task --forward 0.10 --side 0.06 --yaw 0.10

每一段都是 ``flows/`` 下的独立文件，这里只负责按顺序串起来。
"""

from __future__ import annotations

import argparse
import os

from .carry import DEFAULT_FORWARD, DEFAULT_STEPS as CARRY_STEPS, carry
from .common import Recorder, add_common_args, make_robot, say, violations
from .grasp import DEFAULT_STEPS as GRASP_STEPS, grasp
from .lift import DEFAULT_HEIGHT, DEFAULT_STEPS as LIFT_STEPS, lift
from .place import DEFAULT_STEPS as PLACE_STEPS, place
from .prepare import DEFAULT_STEPS as PREPARE_STEPS, prepare
from .release import DEFAULT_STEPS as RELEASE_STEPS, release


def run(strict: bool = False, tilt_limit: float = 25.0, height: float = DEFAULT_HEIGHT,
        forward: float = DEFAULT_FORWARD, side: float = 0.0, yaw: float = 0.0,
        video: bool = False, steps: int | None = None):
    """跑完整流程，返回 (robot, states)。"""
    lift_steps = steps or LIFT_STEPS
    carry_steps = steps or CARRY_STEPS
    place_steps = steps or PLACE_STEPS
    env, robot = make_robot(strict=strict, tilt_limit_deg=tilt_limit)
    recorder = Recorder(env, enabled=video)
    env.render_mode = "rgb_array" if video else env.render_mode
    recorder.attach(robot)

    states = []
    say("prepare", prepare(robot, PREPARE_STEPS))
    recorder.add()
    states.append(grasp(robot, GRASP_STEPS))
    say("grasp", states[-1], f"grasped={env.grasped}")
    recorder.add()
    states.append(lift(robot, height, lift_steps))
    say("lift", states[-1], f"抬起 {states[-1].lift * 1000:+.0f} mm")
    recorder.add()
    states.append(carry(robot, forward, side, yaw, carry_steps))
    say("carry", states[-1], f"前移 {forward * 1000:+.0f} mm")
    recorder.add()
    states.append(place(robot, steps=place_steps))
    say("place", states[-1], f"箱高 {states[-1].box_pos[2]:.3f} m")
    recorder.add()
    states.append(release(robot, RELEASE_STEPS))
    say("release", states[-1], f"grasped={env.grasped}")
    recorder.add()
    return env, robot, states, recorder


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--height", type=float, default=DEFAULT_HEIGHT)
    parser.add_argument("--forward", type=float, default=DEFAULT_FORWARD)
    parser.add_argument("--side", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args(argv)

    env, robot, states, recorder = run(
        strict=args.strict, tilt_limit=args.tilt_limit, height=args.height,
        forward=args.forward, side=args.side, yaw=args.yaw, video=args.video,
        steps=args.steps)

    print(f"\n流程结束：{robot.steps} 步，{violations(robot)}")
    for step, bad in robot.violation_log[:5]:
        print(f"  step {step}: {'; '.join(bad)}")
    start, end = robot.ready_box_pos, states[-1].box_pos
    print(f"箱心：起点 ({start[0]:.3f}, {start[1]:.3f}, {start[2]:.3f}) → "
          f"终点 ({end[0]:.3f}, {end[1]:.3f}, {end[2]:.3f})")
    print(f"抬起 {states[1].lift * 1000:+.1f} mm，前移 {(states[2].box_pos[0] - start[0]) * 1000:+.1f} mm")
    path = recorder.save("full_task.gif")
    if path:
        print(f"视频 -> {path}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
