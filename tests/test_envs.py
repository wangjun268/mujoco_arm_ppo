"""Smoke + regression tests for the MuJoCo reacher / pick environments."""

import os

import numpy as np
import pytest

from env import env_names, make_env


EXPECTED_ENVS = {
    "two_joint",
    "three_joint",
    "six_joint",
    "pro7_joint",
    "pro7_urdf",
    "pro7_pick",
    "pro7_pick_urdf",
    "pro7_pick_place",
}
EXPECTED_OBS = {
    "two_joint": 10,
    "three_joint": 15,
    "six_joint": 24,
    "pro7_joint": 27,
    "pro7_urdf": 27,
    "pro7_pick": 30,
    "pro7_pick_urdf": 30,
    "pro7_pick_place": 38,
}


def test_registry_is_complete():
    assert set(env_names()) == EXPECTED_ENVS


@pytest.mark.parametrize("name", sorted(EXPECTED_ENVS))
def test_observation_shape_is_stable(name):
    """Observation sizes are baked into the saved checkpoints - never change them."""
    env = make_env(name)
    assert env.observation_space.shape == (EXPECTED_OBS[name],)
    obs, info = env.reset(seed=0)
    assert obs.shape == (EXPECTED_OBS[name],)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert "steps" in info
    env.close()


@pytest.mark.parametrize("name", sorted(EXPECTED_ENVS))
def test_step_returns_valid_transition(name):
    env = make_env(name)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert obs.shape == (EXPECTED_OBS[name],)
    assert np.isfinite(reward)
    assert isinstance(terminated, (bool, np.bool_))
    assert isinstance(truncated, (bool, np.bool_))
    assert info["steps"] == 1
    env.close()


@pytest.mark.parametrize("name", ["two_joint", "three_joint", "six_joint", "pro7_joint"])
def test_reset_is_seed_reproducible(name):
    env = make_env(name)
    first, _ = env.reset(seed=42)
    target = env.target.copy()
    second, _ = env.reset(seed=42)
    assert np.allclose(first, second)
    assert np.allclose(target, env.target)
    env.close()


def test_two_joint_observation_layout():
    """The planar arm interleaves cos/sin; its checkpoints depend on that order."""
    env = make_env("two_joint")
    obs, _ = env.reset(seed=3)
    q, dq = env.data.qpos[:2], env.data.qvel[:2]
    expected = np.concatenate(
        [[np.cos(q[0]), np.sin(q[0]), np.cos(q[1]), np.sin(q[1])],
         dq, env.tip - env.target, env.target]
    )
    assert np.allclose(obs, expected)
    env.close()


def test_episode_terminates_on_success():
    """A tip sitting on the target must terminate with the success bonus."""
    env = make_env("three_joint")
    env.reset(seed=0)
    env.data.qvel[:] = 0.0
    env._target = env.tip.copy()  # move the target onto the current tip
    reward, terminated = env._compute_reward()
    assert terminated is True
    assert reward >= env.success_reward
    assert env._get_info()["success"] is True
    env.close()


def test_rendering_produces_frames(tmp_path):
    """``render_mode='rgb_array'`` must return a HxWx3 uint8 image."""
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    env = make_env("two_joint", render_mode="rgb_array")
    env.reset(seed=0)
    frame = env.render()
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    env.close()
