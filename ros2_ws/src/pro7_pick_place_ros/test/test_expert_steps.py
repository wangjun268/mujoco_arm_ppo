"""The step-wise expert the node drives is the blocking expert, step for step."""

import numpy as np
import pytest

pytest.importorskip("mujoco")

from pro7_pick_place_ros import project
from pro7_pick_place_ros.scene import GraspScene


@pytest.fixture(scope="module")
def gc():
    grasp_common, _paths = project.import_project(project.find_project_root())
    return grasp_common


def _scene(gc, seed):
    return GraspScene(gc, seed=seed)


def _run_stepwise(gc, scene):
    """Drive ``iter_pick_and_place`` by hand, like the plant thread does."""
    episode = gc.iter_pick_and_place(
        scene.model, scene.data, scene.ids, scene.detect_renderer
    )
    phases, steps = [], 0
    while True:
        try:
            step = next(episode)
        except StopIteration as done:
            return done.value, phases, steps
        phases.append(step.phase)
        assert step.step == steps + 1
        steps += 1


def test_stepwise_expert_matches_the_blocking_call(gc):
    blocking = _scene(gc, seed=0)
    stepwise = _scene(gc, seed=0)
    try:
        expected = gc.pick_and_place(
            blocking.model, blocking.data, blocking.ids, blocking.detect_renderer
        )
        result, phases, steps = _run_stepwise(gc, stepwise)
    finally:
        blocking.close()
        stepwise.close()

    assert (result.picked, result.placed, result.steps, result.reason) == (
        expected.picked, expected.placed, expected.steps, expected.reason
    )
    assert result.steps == steps
    assert phases[0] == "approach"
    order = ["approach", "settle", "lift", "transport", "lower", "release"]
    indices = [order.index(phase) for phase in phases]
    assert indices == sorted(indices)  # phases never go backwards
    assert phases[-1] in order


def test_stepwise_expert_agrees_on_the_cube_position(gc):
    blocking = _scene(gc, seed=0)
    stepwise = _scene(gc, seed=0)
    try:
        gc.pick_and_place(blocking.model, blocking.data, blocking.ids,
                          blocking.detect_renderer)
        _run_stepwise(gc, stepwise)
        assert np.allclose(
            gc.cube_world(blocking.data, blocking.ids),
            gc.cube_world(stepwise.data, stepwise.ids),
            atol=1e-9,
        )
    finally:
        blocking.close()
        stepwise.close()
