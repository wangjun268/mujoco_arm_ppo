"""The MuJoCo wrapper: state, frames, vision and the place target."""

import numpy as np
import pytest

pytest.importorskip("mujoco")  # the node drives the MuJoCo plant

from pro7_pick_place_ros import project
from pro7_pick_place_ros.scene import (
    CAMERA_FRAME,
    CAMERA_OPTICAL_FRAME,
    TOOL_FRAME,
    WORLD_FRAME,
    GraspScene,
)


@pytest.fixture(scope="module")
def gc():
    grasp_common, _paths = project.import_project(project.find_project_root())
    return grasp_common


@pytest.fixture()
def scene(gc):
    scene = GraspScene(gc, seed=0)
    yield scene
    scene.close()


def test_state_matches_the_project_scene(scene, gc):
    state = scene.state()
    assert state.scene_id == 1
    assert len(state.arm_qpos) == gc.N_ARM == 7
    assert len(state.joint_names) == 7 + len(gc.HAND_JOINTS)
    assert np.allclose(state.cube_position, scene.cube_start, atol=1e-9)
    assert np.allclose(state.place_target, gc.PLACE_TARGET)
    assert state.place_error == pytest.approx(
        np.linalg.norm(scene.cube_start - gc.PLACE_TARGET)
    )
    assert not state.holding  # nothing is grasped at the start
    assert state.hand_closure == pytest.approx(0.0, abs=1e-6)


def test_tf_chain_covers_arm_tool_and_camera(scene):
    frames = {frame.child: frame for frame in scene.state().frames}
    assert frames["base"].parent == WORLD_FRAME
    assert frames["link1"].parent == "base"
    assert frames["link7"].parent == "link6"
    assert frames["gripper"].parent == "link7"
    assert frames[TOOL_FRAME].parent == "gripper"
    assert frames[CAMERA_FRAME].parent == "gripper"
    assert frames[CAMERA_OPTICAL_FRAME].parent == CAMERA_FRAME
    for frame in frames.values():
        assert np.linalg.norm(frame.quaternion) == pytest.approx(1.0, abs=1e-9)


def test_vision_localises_the_cube(scene):
    estimate = scene.detect()
    assert estimate is not None
    assert np.linalg.norm(estimate - scene.cube_start) < 0.03
    # The cached estimate is what the status topic reports.
    assert np.allclose(scene.state().detected, estimate)


def test_place_target_override_moves_the_pad(scene):
    import mujoco

    target = np.array([0.62, 0.44, scene.gc.CUBE_Z])
    scene.set_place_target(target)
    pad = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "place_pad")
    assert pad >= 0
    assert np.allclose(scene.model.geom_pos[pad][:2], target[:2])
    assert np.allclose(scene.state().place_target, target)


def test_reset_is_reproducible_per_seed(gc):
    scene = GraspScene(gc, seed=3)
    try:
        first = scene.state().cube_position
        assert scene.reset() == 2
        assert np.allclose(scene.state().cube_position, first)
        scene.reset(randomize_cube=True)  # next seed -> a different spot
        assert scene.seed == 4
        assert not np.allclose(scene.state().cube_position, first)
    finally:
        scene.close()
