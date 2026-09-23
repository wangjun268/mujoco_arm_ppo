"""Model + environment regression tests for the take_box MuJoCo stack."""

import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from take_box_env import env_names, make_env  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(ROOT / "assets" / "take_box_cell.xml"))


def _config():
    return json.loads((ROOT / "cell_config.json").read_text())


def test_cell_has_two_arms_and_two_hands(model):
    robot = _config()["robot"]
    assert model.nu == 36, "14 arm motors + 22 finger servos (lkwy73 hands)"
    assert model.nq == 36
    assert model.nmocap == 1, "the box"
    for side in ("left", "right"):
        assert mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp") >= 0
        for name in robot["arms"][side]["joints"]:
            assert mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0, "the cell is contact free by design"


def test_ready_pose_puts_the_palms_on_the_box_faces(model):
    ready = json.loads((ROOT / "assets" / "take_box_ready.json").read_text())
    robot = _config()["robot"]
    data = mujoco.MjData(model)
    for side, q in ready["joints"].items():
        for name, value in zip(robot["arms"][side]["joints"], q):
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    # what matters is the palm landing on the face; the palm *orientation* of
    # these hands can stay ~20 deg off without changing the (kinematic) grasp
    for side in ("left", "right"):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp")
        target = np.asarray(ready["targets"][side]["site_pos"])
        assert np.linalg.norm(data.site_xpos[site] - target) < 5e-3


def test_registry():
    assert set(env_names()) == {"take_box", "take_box_reach"}


@pytest.mark.parametrize("name", ["take_box", "take_box_reach"])
def test_spaces_reset_and_step(name):
    env = make_env(name)
    try:
        assert env.action_space.shape == (16,)
        assert env.observation_space.shape == (55,)
        obs, info = env.reset(seed=0)
        assert obs.shape == (55,) and obs.dtype == np.float32
        assert np.all(np.isfinite(obs))
        assert info["steps"] == 0
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert np.isfinite(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert info["steps"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("name", ["take_box", "take_box_reach"])
def test_reset_is_seed_reproducible(name):
    env = make_env(name)
    try:
        first, _ = env.reset(seed=11)
        second, _ = env.reset(seed=11)
        assert np.allclose(first, second)
    finally:
        env.close()


def test_lift_starts_ready_to_grasp():
    env = make_env("take_box")
    try:
        inside = 0
        for seed in range(20):
            _, info = env.reset(seed=seed)
            if max(info["dist_left"], info["dist_right"]) < env.grasp_tol:
                inside += 1
        assert inside >= 15, "the lift task must start near the box (prepare move)"
    finally:
        env.close()


def test_reach_starts_far_from_the_box():
    env = make_env("take_box_reach")
    try:
        for seed in range(20):
            _, info = env.reset(seed=seed)
            assert max(info["dist_left"], info["dist_right"]) > env.grasp_tol
    finally:
        env.close()


def test_grasp_latches_without_teleporting_the_box():
    """Touching the box must not make it jump or flip.

    Regression test: the carry used to snap the box onto the palm frame, so at
    the latch the box rotated onto a frame whose "up" axis is ~9 deg off
    vertical (and moved by whatever grasp error was left) - in the viewer the
    box visibly took off the moment the hands reached it.
    """
    env = make_env("take_box", reset_noise=0.0)
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape)
        action[14:] = 1.0  # close both grips
        box_before = env.data.mocap_pos[0].copy()
        quat_before = env.data.mocap_quat[0].copy()
        _, _, _, _, info = env.step(action)
        assert info["grasped"] is True
        assert np.linalg.norm(env.data.mocap_pos[0] - box_before) < 1e-3
        assert np.allclose(env.data.mocap_quat[0], quat_before, atol=1e-3)
        assert info["tilt_deg"] < 1.0, "the box starts level on the table"

        # the box then rides the hands rigidly: the offset frozen at the latch
        # is preserved while the arms move it around
        offset = env.data.mocap_pos[0] - env.palm_positions().mean(axis=0)
        box_before = env.data.mocap_pos[0].copy()
        palms_before = env.palm_positions().mean(axis=0).copy()
        for index in range(14):  # a small torque on every arm joint
            action[index] = 0.15
        for _ in range(30):
            _, _, _, _, info = env.step(action)
        moved_box = env.data.mocap_pos[0] - box_before
        moved_palms = env.palm_positions().mean(axis=0) - palms_before
        assert np.linalg.norm(moved_box) > 5e-3, "the box must follow the hands"
        assert np.linalg.norm(moved_box - moved_palms) < 2e-3
        assert np.linalg.norm(
            env.data.mocap_pos[0] - env.palm_positions().mean(axis=0) - offset) < 2e-3
    finally:
        env.close()


def test_rendering():
    env = make_env("take_box", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        frame = env.render()
        assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8
    finally:
        env.close()


def _tip_advance(env, side, grip):
    """Move the fingertips/thumb 0..1 of their travel, report how far each tip
    moved towards the box (fingers always curl towards the palm side)."""
    for joint, _actuator in zip(env.finger_joints[side], env.hand_act[side]):
        joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        env.data.qpos[env.model.jnt_qposadr[joint_id]] = (
            grip * env.model.jnt_range[joint_id][1])
    mujoco.mj_forward(env.model, env.data)
    prefix = env.config["robot"]["arms"][side]["finger_prefix"]
    palm = env.data.site_xpos[env.grasp_site[side]].copy()
    to_box = env.data.mocap_pos[0] - palm
    to_box /= np.linalg.norm(to_box)

    def tip(name):
        body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}{name}")
        return env.data.xpos[body] - palm

    fingers = np.mean([np.dot(tip(f"{f}_distal"), to_box)
                       for f in ("index", "middle", "ring", "pinky")])
    thumb = np.dot(tip("thumb_distal"), to_box)
    return float(fingers), float(thumb)


def test_pre_grasp_pose_grasps_with_the_palms():
    """The pre-grasp pose must present the *palm* of both hands to the box.

    Regression test for two bugs found in a row: the grasp sites are
    position-only (so the palm frame has to be measured from the fingers), and
    the two hands are CAD mirrors, so the same target rotation serves both -
    getting that wrong rolls a hand 180 deg about its fingers and puts the back
    of the hand on the box.  Fingers always curl towards the palm side, so
    "closing the hand moves the tips towards the box" is the anatomical check.
    """
    env = make_env("take_box", reset_noise=0.0)
    try:
        _, info = env.reset(seed=0)
        assert info["tilt_deg"] < 1.0, "the box starts level on the table"
        for side in ("left", "right"):
            open_f, open_t = _tip_advance(env, side, 0.0)
            shut_f, shut_t = _tip_advance(env, side, 1.0)
            assert shut_f - open_f > 0.015, (
                f"{side} fingers curl away from the box by "
                f"{(shut_f - open_f) * 1000:.1f} mm - the hand is reversed")
            assert shut_t - open_t > 0.005, (
                f"{side} thumb opposes away from the box by "
                f"{(shut_t - open_t) * 1000:.1f} mm - the hand is reversed")
    finally:
        env.close()


def test_box_rests_on_the_table():
    """Geometry guard: the box must sit on the table, not inside it."""
    cell = _config()["cell"]
    table_top = cell["table"]["pos"][2] + cell["table"]["size"][2] / 2
    box_bottom = cell["box"]["pos"][2] - cell["box"]["size"][2] / 2
    assert abs(table_top - box_bottom) < 0.02, (
        f"box bottom {box_bottom:.3f} vs table top {table_top:.3f}")


def test_lift_target_is_reachable():
    """The reward target must stay inside the arms' workspace."""
    env = make_env("take_box")
    try:
        # the pre-grasp pose is reachable by construction; ask the analytic
        # check as well: a target the arms cannot lift makes the task unsolvable
        assert env.box_target_z - env.box_home[2] <= 0.15, (
            f"lift target asks for {env.box_target_z - env.box_home[2]:.3f} m "
            "of travel")
    finally:
        env.close()


def test_grip_channel_scales_every_finger_to_its_own_travel():
    """``grip = 1`` must close each channel, not drive them all to 1.2 rad.

    The hand has eleven channels with different travels (thumb pitch 0.58 rad,
    finger curl 1.6 rad).  A single shared ctrlrange over-drove the thumb into
    its limit stop - the position servo kept pushing past the joint range - and
    left the fingers at 75% of theirs.
    """
    env = make_env("take_box", reset_noise=0.0)
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape)
        action[14:] = 1.0
        for _ in range(150):
            env.step(action)
        for side in ("left", "right"):
            for name, actuator in zip(env.finger_joints[side], env.hand_act[side]):
                joint = mujoco.mj_name2id(
                    env.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                travel = env.model.jnt_range[joint][1]
                ctrl = env.data.ctrl[actuator]
                angle = env.data.qpos[env.model.jnt_qposadr[joint]]
                assert ctrl <= travel + 1e-9, (
                    f"{name} commanded {ctrl:.2f} rad, travel is {travel:.2f}")
                assert abs(angle - travel) < 0.05, (
                    f"{name} only closed to {angle:.2f} of {travel:.2f} rad")
    finally:
        env.close()


def _latch(env, steps: int = 20):
    action = np.zeros(env.action_space.shape)
    action[14:] = 1.0
    for _ in range(steps):
        env.step(action)
    assert env.grasped
    return action


def test_open_hand_keeps_the_latched_grasp_by_default():
    """The default carry convention is quasi-static: only the palms matter.

    Kept as the default because "hold the grip closed for the whole carry" is a
    ~40-step credit assignment problem that PPO does not solve in this budget.
    """
    env = make_env("take_box", reset_noise=0.0)
    try:
        env.reset(seed=0)
        _latch(env)
        for _ in range(5):
            env.step(np.zeros(env.action_space.shape))
        assert env.grasped, "allow_release=False must keep the box on the palms"
    finally:
        env.close()


def test_allow_release_drops_the_box():
    env = make_env("take_box", reset_noise=0.0, allow_release=True)
    try:
        env.reset(seed=0)
        _latch(env)

        open_action = np.zeros(env.action_space.shape)  # grips commanded open
        for _ in range(3):
            env.step(open_action)
        assert not env.grasped, "an open hand must let go of the box"
        box = env.data.mocap_pos[0].copy()

        away = np.zeros(env.action_space.shape)
        away[1] = away[8] = 0.8  # swing both arms away from the box
        for _ in range(30):
            env.step(away)
        assert np.linalg.norm(env.data.mocap_pos[0] - box) < 1e-6, (
            "a released box must stay where it was")
    finally:
        env.close()


def test_grasp_bonus_is_paid_once_per_episode():
    """Opening and closing the hand on the spot must not farm the latch bonus.

    Regression test: as soon as the grasp could be released, a policy learned to
    alternate ``grip`` every step - +2 per re-latch beat both the hold reward and
    the success reward, and it never lifted the box.
    """
    env = make_env("take_box", reset_noise=0.0, allow_release=True)
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape)
        action[14:] = 1.0
        _, reward, *_ = env.step(action)  # latch
        assert env.grasped and reward > 1.0, "the latch must pay the bonus"

        action[14:] = 0.0
        env.step(action)  # open the hand again
        assert not env.grasped

        action[14:] = 1.0
        _, reward, *_ = env.step(action)  # re-latch on the same spot
        assert env.grasped
        assert reward < 1.0, "the latch bonus must only be paid once per episode"
    finally:
        env.close()


def test_grasp_targets_follow_the_box_frame():
    env = make_env("take_box", reset_noise=0.0)
    try:
        env.reset(seed=0)
        env.grasped = True
        separation = env.palm_offsets["left"] - env.palm_offsets["right"]
        assert np.allclose(env.face_points()[0] - env.face_points()[1], separation)
        # yaw the box 90 deg: the grasp targets have to rotate with it
        env.data.mocap_quat[0] = [np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)]
        yawed = env.face_points()
        yaw_90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert np.allclose(yawed[0] - yawed[1], yaw_90 @ separation, atol=1e-6)
    finally:
        env.close()
