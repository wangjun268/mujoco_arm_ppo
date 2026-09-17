"""End-to-end: a real PickPlace action through a running node."""

import gc
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("mujoco")

from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Parameter  # noqa: E402

from pro7_pick_place_interfaces.action import PickPlace  # noqa: E402
from pro7_pick_place_interfaces.srv import ResetScene  # noqa: E402
from pro7_pick_place_ros.node import PickPlaceNode  # noqa: E402
from pro7_pick_place_ros.simulator import PickPlaceRequest  # noqa: E402

TIMEOUT = 120.0  # the expert walks its waypoints in real time


def _wait(future, timeout, what):
    deadline = time.time() + timeout
    while not future.done():
        if time.time() > deadline:
            raise AssertionError(f"timeout waiting for {what}")
        time.sleep(0.05)
    return future.result()


@pytest.fixture()
def node():
    rclpy.init()
    # sim_hz 0 -> run the plant unthrottled; seed 1 is a known placable cube.
    overrides = [
        Parameter("seed", value=1),
        Parameter("sim_hz", value=0.0),
        Parameter("vision_hz", value=0.0),
        Parameter("publish_images", value=False),
    ]
    node = PickPlaceNode(parameter_overrides=overrides)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield node
    finally:
        executor.shutdown()
        thread.join(timeout=5.0)
        node.destroy_node()
        # Collect the action/service futures while the context is still alive:
        # letting them die after rclpy.shutdown() prints an rclpy warning.
        gc.collect()
        rclpy.shutdown()


def test_reset_service_rebuilds_the_cell(node):
    client = node.create_client(ResetScene, "/pro7_pick_place/reset")
    try:
        assert client.wait_for_service(timeout_sec=20.0)
        request = ResetScene.Request()
        request.randomize_cube = True
        response = _wait(client.call_async(request), TIMEOUT, "reset")
        assert response.success
        assert response.scene_id >= 2
        assert response.cube_position.z > 0.0
    finally:
        node.destroy_client(client)


def test_pick_place_goal_places_the_cube(node):
    client = ActionClient(node, PickPlace, "/pro7_pick_place/pick_place")
    assert client.wait_for_server(timeout_sec=20.0)

    goal = PickPlace.Goal()
    goal.seed = float("nan")
    goal.reset_scene = False
    goal.pick_only = False
    seen = []
    try:
        handle = _wait(
            client.send_goal_async(
                goal, feedback_callback=lambda msg: seen.append(msg.feedback)
            ),
            TIMEOUT, "goal",
        )
        assert handle.accepted
        wrapped = _wait(handle.get_result_async(), TIMEOUT, "result")
        result = wrapped.result
        assert result.success, result.message
        assert result.picked and result.placed
        assert result.message == "placed"
        assert result.place_error < 0.05
        assert seen, "no feedback was published"
        assert seen[-1].steps > 0
        # The node's own plant agrees with the result it reported.
        assert node.scene.state().place_error < 0.05
    finally:
        client.destroy()


def test_second_goal_is_rejected_while_one_runs(node):
    node.sim.submit(PickPlaceRequest(randomize_cube=False))
    client = ActionClient(node, PickPlace, "/pro7_pick_place/pick_place")
    try:
        assert client.wait_for_server(timeout_sec=20.0)
        goal = PickPlace.Goal()
        goal.seed = float("nan")
        handle = _wait(client.send_goal_async(goal), TIMEOUT, "rejected goal")
        assert not handle.accepted
    finally:
        client.destroy()
        node.sim.cancel()
