"""take_box 的奖励函数（从环境里拆出来单独成文件）。

奖励是这套环境里最容易被"刷"的部分（README 的"踩过的坑"第 4、9 条都是奖励漏洞），
所以把它和环境解耦，用纯数据进出的方式写，方便单独测（``tests/test_rewards.py``）。

结构：

* :class:`RewardConfig` —— 全部可调系数。默认值就是现在定稿的那一套，可以在
  ``cell_config.json`` 的 ``task.reward`` 里按需覆盖。
* :class:`GraspStep`    —— 一步里奖励函数读得到的量，由环境填好。
* :class:`RewardResult` —— 奖励函数算出来的量，由环境写回自己的状态。
* :class:`TakeBoxReward`—— 势函数 + 事件奖励。``potential()`` **只依赖状态**
  （掌心误差、箱高误差、箱体倾角），``step()`` 负责抓取锁存/脱手、保持计数与终止判定。

两条不能破的规矩（都有回归测试盯着）：

1. 势函数不能随"是否抓住"切换：抓取标志一变就换一套 Φ，松手等于把负项甩掉、白拿奖励。
2. 抓取奖励每回合只发一次：否则策略学会原地开合手反复吃 bonus，一步都不抬。
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np


@dataclass
class RewardConfig:
    """奖励系数（默认 = 定稿值）。"""

    #: 抓取通道大于它就认为"手是闭的"（锁存条件之一）
    grasp_close: float = 0.5
    #: 抓取通道小于它就脱手（仅在环境允许脱手时生效）
    release_grip: float = 0.2
    #: 首次抓住的奖励（每回合只发一次）
    grasp_bonus: float = 2.0
    #: 达标（保持住 hold_steps 步）的奖励并结束回合
    success_reward: float = 5.0
    #: 每保持住一步的奖励
    hold_reward: float = 0.5
    #: 势函数里箱体倾角的权重
    tilt_cost: float = 2.0
    #: 势函数里箱高误差的权重（掌心误差权重固定为 1）
    height_weight: float = 2.0
    #: 动作代价（Σctrl² 的系数）
    ctrl_cost: float = 0.0005
    #: 势函数塑形折扣
    gamma: float = 0.99

    @classmethod
    def from_config(cls, config: dict) -> "RewardConfig":
        """从 cell_config.json 读 ``task.reward``（缺项用默认值）。"""
        overrides = dict(config.get("task", {}).get("reward", {}))
        known = {field.name for field in fields(cls)}
        unknown = set(overrides) - known
        if unknown:
            raise KeyError(f"unknown reward option(s): {sorted(unknown)}")
        return cls(**overrides)

    def as_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass
class GraspStep:
    """一步的输入：奖励函数需要的全部量（由环境填）。"""

    stage: str                 # "reach"（只练靠近）或 "lift"（完整任务）
    hand_errors: np.ndarray    # 每只掌心到抓取目标的距离
    box_z: float               # 箱心高度
    box_target_z: float        # 抬升目标高度
    box_tilt: float            # 箱体倾角（rad，相对水平面）
    grip: np.ndarray           # 双手抓取通道（-1..1 的动作值）
    ctrl: np.ndarray           # 本步写进执行器的控制量
    grasped: bool
    grasp_ready: bool          # 掌心是否都贴在抓取目标上
    grasp_rewarded: bool       # 本回合是否已经发过抓取奖励
    allow_release: bool
    hold_count: int
    potential: float           # 上一时刻的势函数值
    # 任务规格（判定用，不是训练出来的）
    grasp_tol: float
    height_tol: float
    tilt_tol: float
    hold_steps: int


@dataclass
class RewardResult:
    """一步的输出：环境把 state 字段写回自己的状态。"""

    reward: float
    terminated: bool
    grasped: bool
    grasp_rewarded: bool
    hold_count: int
    potential: float


class TakeBoxReward:
    """势函数塑形 + 事件奖励。"""

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    # ------------------------------------------------------------------ #
    def potential(self, step: GraspStep) -> float:
        """Φ(s)：越靠近抓取点、箱体越高、越平，值越大（负数，趋近 0）。

        只依赖状态，不依赖抓取标志——否则松手就能把"高度+倾角"这两项负值甩掉。
        """
        height_error = abs(step.box_z - step.box_target_z)
        return -(float(np.mean(step.hand_errors))
                 + self.config.height_weight * height_error
                 + self.config.tilt_cost * step.box_tilt)

    # ------------------------------------------------------------------ #
    def step(self, step: GraspStep) -> RewardResult:
        """算这一步的奖励，并给出新的抓取/保持状态。"""
        config = self.config
        grasped = step.grasped
        rewarded = step.grasp_rewarded
        hold_count = step.hold_count

        bonus = 0.0
        if (step.allow_release and grasped
                and float(step.grip.max()) < config.release_grip):
            # 张开手就脱手：箱子不再跟着掌心坐标系走
            grasped, hold_count = False, 0
        elif (not grasped and float(step.grip.min()) > config.grasp_close
              and step.grasp_ready):
            grasped = True
            if not rewarded:      # 每回合只发一次，防止原地开合手刷分
                rewarded, bonus = True, config.grasp_bonus

        potential = self.potential(step)
        reward = config.gamma * potential - step.potential + bonus
        reward -= config.ctrl_cost * float(np.sum(step.ctrl ** 2))

        terminated = False
        if step.stage == "reach":
            terminated = bool(step.hand_errors.max() < step.grasp_tol)
            if terminated:
                reward += config.success_reward
            return RewardResult(reward, terminated, grasped, rewarded,
                                hold_count, potential)

        if grasped:
            height_error = abs(step.box_z - step.box_target_z)
            if height_error < step.height_tol and step.box_tilt < step.tilt_tol:
                hold_count += 1
                reward += config.hold_reward
                if hold_count >= step.hold_steps:
                    terminated = True
                    reward += config.success_reward
            else:
                hold_count = 0
        return RewardResult(reward, terminated, grasped, rewarded,
                            hold_count, potential)
