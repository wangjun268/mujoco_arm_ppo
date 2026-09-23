"""The URDF the node publishes has to describe the cell it actually simulates.

The node latches ``/robot_description`` so rviz's RobotModel display can draw
the arm (see ``pro7_pick_place_ros.robot_description``).  These tests check the
contract that makes the display work: the description arrives, its links are the
frames the node publishes on ``/tf``, its joints are the names it puts on
``joint_states``, and every mesh it names is on disk.
"""

import os
import gc
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("mujoco")

from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Parameter  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

from pro7_pick_place_ros import robot_description  # noqa: E402
from pro7_pick_place_ros.node import PickPlaceNode  # noqa: E402

TIMEOUT = 30.0
LATCHED = QoSProfile(
    depth=1,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    reliability=ReliabilityPolicy.RELIABLE,
)


def _collect(messages, timeout=TIMEOUT):
    """Wait until at least one message arrived, or fail."""
    deadline = time.time() + timeout
    while not messages:
        if time.time() > deadline:
            raise AssertionError("timeout waiting for a message")
        time.sleep(0.05)
    return messages


@pytest.fixture(scope="module")
def node():
    rclpy.init()
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
        # Let the node (and the subscriptions the tests left on it) die while
        # the context is still alive, or rclpy warns on the way out.
        del node
        gc.collect()
        rclpy.shutdown()


def _urdf(node, qos=LATCHED) -> str:
    """The description as a subscriber sees it (latched by default)."""
    messages = []
    # The subscriptions are left to the fixture's node: destroying one from a
    # test while the executor is still spinning trips rclpy's use-after-free
    # guard at shutdown.
    node.create_subscription(String, "/robot_description", messages.append, qos)
    _collect(messages)
    return messages[0].data


def test_description_is_latched_for_a_late_subscriber(node):
    """rviz usually starts after the node, so the publish has to be latched."""
    assert node.robot_desc_pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL
    text = _urdf(node)  # subscribed long after the node published it
    root = ET.fromstring(text.encode("utf-8"))
    assert root.get("name")


def test_description_also_reaches_a_volatile_subscriber(node):
    """rviz's own default durability is volatile; the timer has to cover it."""
    messages = []
    node.create_subscription(String, "/robot_description", messages.append, 10)
    _collect(messages)
    assert ET.fromstring(messages[0].data.encode("utf-8")).get("name")


def test_links_are_the_frames_the_node_publishes(node):
    links = {link.get("name") for link in ET.fromstring(_urdf(node)).findall("link")}
    frames = []
    node.create_subscription(TFMessage, "/tf", frames.append, 100)
    _collect(frames)
    published = {t.child_frame_id for message in frames for t in message.transforms}
    arm = {f"link{i}" for i in range(1, 8)}
    assert arm <= links, "the arm links of the URDF are not the ones the node draws"
    assert arm | {"base", "gripper"} <= published
    # Nothing in the URDF may be a frame the node never publishes: rviz places
    # links by TF, so such a link would collapse onto its parent.
    robot_frames = published - {"tool0", "cam_hand", "cam_hand_optical"}
    assert links <= robot_frames


def test_joint_names_match_joint_states(node):
    text = _urdf(node)
    joints = {
        joint.get("name"): joint.get("type")
        for joint in ET.fromstring(text.encode("utf-8")).findall("joint")
    }
    states = []
    node.create_subscription(JointState, "/pro7_pick_place/joint_states", states.append, 10)
    _collect(states)
    movable = {name for name, kind in joints.items() if kind != "fixed"}
    assert movable == {f"joint{i}" for i in range(1, 8)}
    assert movable <= set(states[0].name)


def test_every_mesh_the_description_names_exists(node):
    missing = [
        uri
        for uri in robot_description._MESH_URI.findall(_urdf(node))
        if not os.path.isfile(uri.replace("file://", ""))
    ]
    assert not missing, f"rviz would draw nothing for {missing[:3]}"


def test_shipped_rviz_layout_shows_the_robot_model():
    """The default layout has to ask for the latched description."""
    yaml = pytest.importorskip("yaml")
    config = Path(__file__).resolve().parents[1] / "config" / "pick_place.rviz"
    doc = yaml.safe_load(config.read_text())
    manager = doc["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "world"
    displays = {display["Name"]: display for display in manager["Displays"]}
    robot = displays["RobotModel"]
    assert robot["Class"] == "rviz_default_plugins/RobotModel"
    assert robot["Description Source"] == "Topic"
    assert robot["Description Topic"]["Value"] == "/robot_description"
    # rviz starts after the node, so it has to ask for the latched description.
    assert robot["Description Topic"]["Durability Policy"] == "Transient Local"
    assert robot["Enabled"] is True


def test_mesh_paths_are_repaired_or_reported(tmp_path):
    """A URDF generated for another checkout still has to be diagnosed."""
    assets = tmp_path / "assets"
    (assets / "meshes").mkdir(parents=True)
    (assets / "meshes" / "link1.stl").write_bytes(b"solid x\nendsolid x\n")
    urdf = tmp_path / "robot.urdf"
    urdf.write_text(
        '<?xml version="1.0"?>\n<robot name="r">\n'
        '  <link name="base"><visual><geometry>'
        '<mesh filename="file:///gone/assets/meshes/link1.stl"/>'
        "</geometry></visual></link>\n"
        '  <link name="gone"><visual><geometry>'
        '<mesh filename="file:///gone/assets/meshes/missing.stl"/>'
        "</geometry></visual></link>\n"
        "</robot>\n"
    )
    text, notes = robot_description.read(str(urdf), str(assets))
    assert f"file://{assets}/meshes/link1.stl" in text
    assert any("repointed 1 mesh path" in note for note in notes)
    assert any("1 mesh file(s)" in note for note in notes)
    assert robot_description.counts(text) == (2, 0)
