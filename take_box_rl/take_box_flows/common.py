"""流程共用的小工具：建环境/接口、打印一步状态、可选存视频。"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")

DEFAULT_ENV = "take_box"
DEFAULT_TILT_LIMIT_DEG = 25.0
#: 掌心/箱面姿态的 IK 步进；0.15 比默认更"贴种子"，规划轨迹更稳
DEFAULT_IK_REG = 0.15


def make_robot(env: str = DEFAULT_ENV, *, seed: int = 0, strict: bool = False,
               reset_noise: float = 0.0, tilt_limit_deg: float = DEFAULT_TILT_LIMIT_DEG,
               ik_reg: float = DEFAULT_IK_REG):
    """建一个环境 + BoxCarryInterface（流程都跑在确定性复位上）。"""
    from take_box_env.box_api import BoxCarryInterface
    from take_box_env.take_box import TakeBoxEnv

    environment = TakeBoxEnv(reset_noise=reset_noise)
    environment.reset(seed=seed)
    robot = BoxCarryInterface(environment, strict=strict, ik_reg=ik_reg)
    robot.limits.tilt_max_deg = tilt_limit_deg
    return environment, robot


def say(stage: str, state, extra: str = "") -> None:
    """打印一步：流程名 + 一行状态。"""
    print(f"[{stage:8s}] {state.summary()}{('  ' + extra) if extra else ''}")


def violations(robot) -> str:
    """约束记录的一行摘要。"""
    if not robot.violation_log:
        return "约束记录 0 条"
    first = "; ".join(robot.violation_log[0][1])
    return f"约束记录 {len(robot.violation_log)} 条（首条：{first}）"


class Recorder:
    """把流程的每一步渲染成帧（可选），最后存 gif/mp4。

    ``attach(robot)`` 会把 ``servo_step`` 包一层，所以每走一步就抓一帧，
    不需要每个流程自己去关心录像。
    """

    def __init__(self, env, enabled: bool = False, fps: int = 25, every: int = 4,
                 downscale: int = 2, name: str = DEFAULT_ENV):
        self.env = env
        self.enabled = enabled
        self.fps = fps
        self.every = max(1, every)
        self.downscale = max(1, downscale)
        self.name = name
        self.frames = []
        self.steps = 0

    def _frame(self):
        frame = self.env.render()
        if self.downscale > 1:  # 原始 480x640 存 gif 太占地方
            frame = frame[::self.downscale, ::self.downscale]
        return frame

    def attach(self, robot) -> None:
        """每 ``every`` 步存一帧。"""
        if not self.enabled:
            return
        original = robot.servo_step

        def recording_step(*args, **kwargs):
            state = original(*args, **kwargs)
            self.steps += 1
            if self.steps % self.every == 0:
                self.frames.append(self._frame())
            return state

        robot.servo_step = recording_step

    def add(self) -> None:
        if self.enabled:
            self.frames.append(self._frame())

    def save(self, filename: str = "full_task.gif") -> Optional[str]:
        if not self.enabled or not self.frames:
            return None
        import imageio.v2 as imageio

        out_dir = os.path.join(RUNS, self.name)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, filename)
        if path.endswith(".gif"):
            imageio.mimsave(path, self.frames, fps=self.fps)
        else:
            imageio.mimsave(path, self.frames, fps=self.fps)
        return path


def add_common_args(parser) -> None:
    """每个流程文件都有的命令行参数。"""
    parser.add_argument("--env", default=DEFAULT_ENV, choices=["take_box", "take_box_reach"])
    parser.add_argument("--steps", type=int, default=None, help="本流程的插值步数")
    parser.add_argument("--strict", action="store_true",
                        help="约束越界抛 ConstraintViolation（默认只记录）")
    parser.add_argument("--tilt-limit", type=float, default=DEFAULT_TILT_LIMIT_DEG,
                        help="倾角约束（deg）")
