"""``PickPlaceJob``: resumable expert, cancellation, thread-free replay."""

import pytest

pytest.importorskip("mujoco")

from pro7_pick_place_ros import project
from pro7_pick_place_ros.scene import GraspScene
from pro7_pick_place_ros.simulator import PickPlaceJob, PickPlaceRequest


@pytest.fixture(scope="module")
def gc():
    common, _paths = project.import_project(project.find_project_root())
    return common


def test_job_walks_the_expert_to_completion(gc):
    scene = GraspScene(gc, seed=1)
    try:
        job = PickPlaceJob(scene, PickPlaceRequest(randomize_cube=False))
        phases = []
        while job.advance():
            if not phases or phases[-1] != job.phase:
                phases.append(job.phase)
            assert not job.finished
        assert job.finished
        assert job.result is not None
        assert job.result.picked and job.result.placed
        assert job.result.steps == job.snapshot().steps
        assert phases == ["approach", "settle", "lift", "transport", "lower", "release"]
        assert job.snapshot().place_error < gc.PLACE_TOLERANCE
    finally:
        scene.close()


def test_pick_only_stops_after_the_pinch(gc):
    scene = GraspScene(gc, seed=1)
    try:
        job = PickPlaceJob(scene, PickPlaceRequest(pick_only=True))
        while job.advance():
            pass
        assert job.result.picked and not job.result.placed
        assert job.result.reason == "grasped"
        assert job.holding and scene.holding()
    finally:
        scene.close()


def test_cancel_keeps_the_plant_but_ends_the_job(gc):
    scene = GraspScene(gc, seed=1)
    try:
        job = PickPlaceJob(scene, PickPlaceRequest())
        for _ in range(20):
            assert job.advance()
        assert job.cancel()
        snapshot = job.snapshot()
        assert job.finished and not snapshot.running
        assert snapshot.result.reason == "cancelled"
        assert job.advance() is False  # advancing a finished job is a no-op
        assert not job.cancel()
    finally:
        scene.close()
