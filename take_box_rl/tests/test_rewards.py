"""奖励函数单独跑（take_box_env/rewards.py），不依赖 MuJoCo 环境。"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from take_box_env.rewards import GraspStep, RewardConfig, TakeBoxReward  # noqa: E402


def make_step(**overrides) -> GraspStep:
    """一个"抓住、箱子已经抬到目标高度、端平"的基准状态。"""
    base = GraspStep(
        stage="lift",
        hand_errors=np.array([0.002, 0.002]),
        box_z=0.65,
        box_target_z=0.65,
        box_tilt=0.0,
        grip=np.array([1.0, 1.0]),
        ctrl=np.zeros(16),
        grasped=True,
        grasp_ready=True,
        grasp_rewarded=True,
        allow_release=False,
        hold_count=0,
        potential=-0.002,
        grasp_tol=0.045,
        height_tol=0.03,
        tilt_tol=np.radians(15.0),
        hold_steps=10,
    )
    return replace(base, **overrides)


@pytest.fixture
def reward():
    return TakeBoxReward(RewardConfig())


def test_potential_depends_on_state_only(reward):
    """势函数不能随"是否抓住"切换，否则松手就是白拿奖励。"""
    held = make_step(grasped=True)
    let_go = make_step(grasped=False)
    assert reward.potential(held) == pytest.approx(reward.potential(let_go))


def test_potential_rewards_lifting_and_levelling(reward):
    on_table = make_step(box_z=0.55)
    lifted = make_step(box_z=0.65)
    tilted = make_step(box_tilt=np.radians(10.0))
    assert reward.potential(lifted) > reward.potential(on_table)
    assert reward.potential(lifted) > reward.potential(tilted)


def test_latch_bonus_is_paid_once(reward):
    first = reward.step(make_step(grasped=False, grasp_rewarded=False,
                                  potential=reward.potential(make_step(grasped=False))))
    assert first.grasped and first.grasp_rewarded
    assert first.reward > 1.0, "first latch pays the bonus"

    released = reward.step(make_step(grasped=False, grasp_rewarded=True,
                                     potential=first.potential))
    again = reward.step(make_step(grasped=False, grasp_rewarded=True,
                                  potential=released.potential))
    assert again.grasped and again.reward < 1.0, "re-latching must not pay again"


def test_release_does_not_pay(reward):
    """松手不能比保持更划算（势函数连续 + 每回合只发一次 bonus）。"""
    step = make_step(box_z=0.60)
    hold = reward.step(step)
    drop = reward.step(replace(step, potential=hold.potential,
                               allow_release=True, grip=np.array([0.0, 0.0])))
    assert drop.reward < 0.1, f"dropping paid {drop.reward:+.3f}"
    assert not drop.grasped


def test_hold_count_and_success(reward):
    step = make_step(potential=reward.potential(make_step()))
    result = None
    for expected in range(1, 11):
        result = reward.step(replace(step, hold_count=expected - 1,
                                     potential=reward.potential(step)))
        assert result.hold_count == expected
        assert result.reward >= reward.config.hold_reward
    assert result.terminated and result.reward > reward.config.success_reward


def test_hold_resets_when_the_box_leaves_the_window(reward):
    step = make_step(box_z=0.60, hold_count=5)   # 高度误差 0.05 > height_tol
    result = reward.step(replace(step, potential=reward.potential(step)))
    assert result.hold_count == 0
    assert not result.terminated


def test_reach_stage_terminates_on_the_grasp_target(reward):
    step = make_step(stage="reach", hand_errors=np.array([0.01, 0.01]))
    result = reward.step(replace(step, potential=reward.potential(step)))
    assert result.terminated and result.reward > 0


def test_control_cost_is_charged(reward):
    quiet = reward.step(make_step(ctrl=np.zeros(16),
                                  potential=reward.potential(make_step())))
    loud = reward.step(make_step(ctrl=np.ones(16),
                                 potential=reward.potential(make_step())))
    assert quiet.reward > loud.reward
    assert quiet.reward - loud.reward == pytest.approx(16 * reward.config.ctrl_cost)


def test_config_from_cell_config_and_unknown_keys():
    config = {"task": {"reward": {"hold_reward": 0.75}}}
    parsed = RewardConfig.from_config(config)
    assert parsed.hold_reward == 0.75
    assert parsed.grasp_bonus == RewardConfig().grasp_bonus   # 其余保持默认
    assert RewardConfig.from_config({}) == RewardConfig()
    with pytest.raises(KeyError):
        RewardConfig.from_config({"task": {"reward": {"no_such_knob": 1.0}}})
