"""ROS 2 node for the MuJoCo Pro7 (7-DOF) pick & place cell.

Topics (private names, i.e. ``/pro7_pick_place/...``)
    ``status``         pro7_pick_place_interfaces/PickPlaceStatus -- plant + job
    ``joint_states``   sensor_msgs/JointState -- 7 arm joints + 21 hand joints
    ``cube_pose``      geometry_msgs/PoseStamped -- ground-truth cube
    ``detected_cube``  geometry_msgs/PoseStamped -- RGB-D estimate (eye in hand)
    ``camera/color/image_raw``   sensor_msgs/Image -- ``rgb8``
    ``camera/depth/image_raw``   sensor_msgs/Image -- ``32FC1`` in metres
    ``camera/color/camera_info`` / ``camera/depth/camera_info``
    ``markers``        visualization_msgs/MarkerArray -- pad, cube, sub-goal
    ``/tf``            tf2_msgs/TFMessage -- world -> base -> link1..7 -> tool0
    ``/robot_description``  std_msgs/String -- URDF of the arm + hand, latched so
                       an rviz RobotModel display can draw the real geometry
                       (see :mod:`pro7_pick_place_ros.robot_description`)

Services
    ``reset``          pro7_pick_place_interfaces/ResetScene -- rebuild the cell

Actions
    ``pick_place``     pro7_pick_place_interfaces/PickPlace -- one full cycle

Everything the node does is a thin driver on top of the project's own scene,
vision pipeline and scripted expert (see :mod:`pro7_pick_place_ros.scene` and
:mod:`pro7_pick_place_ros.simulator`).
"""

from __future__ import annotations

import math
import os
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, TransformStamped, Vector3
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import ColorRGBA, Header, String
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker, MarkerArray

from pro7_pick_place_interfaces.action import PickPlace
from pro7_pick_place_interfaces.msg import PickPlaceStatus
from pro7_pick_place_interfaces.srv import ResetScene

from . import project
from . import robot_description
from .scene import CAMERA_OPTICAL_FRAME, WORLD_FRAME, GraspScene
from .simulator import PickPlaceRequest, PickPlaceSimulator


def _resolve_point(point, default=None) -> Optional[np.ndarray]:
    """Read a ``geometry_msgs/Point`` into world coordinates.

    ``None`` means "not specified": the point is all-NaN or all-zero, which is
    what an empty message carries.  A *partly* specified point (the common
    ``--place-x/--place-y`` case) takes its missing components from ``default``
    -- the node's current place target -- and is ignored when there is none.
    """
    if point is None:
        return None
    values = np.array([point.x, point.y, point.z], dtype=float)
    if not np.any(values) or not np.any(np.isfinite(values)):
        return None
    if default is None:
        return values if np.all(np.isfinite(values)) else None
    base = np.asarray(default, dtype=float)
    return np.where(np.isfinite(values), values, base)


def _point(xyz) -> Point:
    xyz = np.asarray(xyz, dtype=float)
    return Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))


def _pose(position, quaternion=(1.0, 0.0, 0.0, 0.0)) -> Pose:
    pose = Pose()
    pose.position = _point(position)
    pose.orientation = Quaternion(
        w=float(quaternion[0]), x=float(quaternion[1]),
        y=float(quaternion[2]), z=float(quaternion[3]),
    )
    return pose


class PickPlaceNode(Node):
    """Exposes the MuJoCo pick & place cell over ROS 2."""

    def __init__(self, **kwargs):
        super().__init__("pro7_pick_place", **kwargs)

        # ---- locate the project and its modules -------------------------
        self.declare_parameter("project_root", "")
        self.project_root = project.find_project_root(
            self.get_parameter("project_root").value or None
        )
        self.gc, self.paths = project.import_project(self.project_root)

        # ---- plant ------------------------------------------------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("seed", 0)
        self.declare_parameter("randomize_cube", False)
        self.declare_parameter("place_target", [0.0, 0.0, 0.0])
        self.declare_parameter("image_width", 320)
        self.declare_parameter("image_height", 240)
        place_target = self.get_parameter("place_target").value
        if not np.any(np.asarray(place_target, dtype=float)):
            place_target = None  # "use the project default pad (gc.PLACE_TARGET)"
        self.scene = GraspScene(
            self.gc,
            model_path=self.get_parameter("model_path").value or None,
            seed=self.get_parameter("seed").value,
            randomize_cube=self.get_parameter("randomize_cube").value,
            place_target=place_target,
            image_width=self.get_parameter("image_width").value,
            image_height=self.get_parameter("image_height").value,
            defer_renderers=True,  # the plant thread owns the GL contexts
        )

        # ---- timing -----------------------------------------------------
        self.declare_parameter("sim_hz", 240.0)
        self.declare_parameter("idle_mode", "freeze")
        self.declare_parameter("vision_hz", 5.0)
        self.declare_parameter("status_hz", 20.0)
        self.declare_parameter("image_hz", 5.0)
        self.declare_parameter("publish_images", True)
        self.declare_parameter("feedback_hz", 5.0)
        self.declare_parameter("tf_prefix", "")
        self.tf_prefix = str(self.get_parameter("tf_prefix").value)
        self.sim = PickPlaceSimulator(
            self.scene,
            sim_hz=self.get_parameter("sim_hz").value,
            idle_mode=self.get_parameter("idle_mode").value,
            vision_hz=self.get_parameter("vision_hz").value,
            image_hz=self.get_parameter("image_hz").value,
        )
        self._feedback_period = 1.0 / max(
            1e-3, float(self.get_parameter("feedback_hz").value)
        )
        self._publish_images = bool(self.get_parameter("publish_images").value)

        self._cb_group = ReentrantCallbackGroup()

        # ---- publishers -------------------------------------------------
        self.status_pub = self.create_publisher(PickPlaceStatus, "~/status", 10)
        self.joint_pub = self.create_publisher(JointState, "~/joint_states", 10)
        self.cube_pub = self.create_publisher(PoseStamped, "~/cube_pose", 10)
        self.detected_pub = self.create_publisher(PoseStamped, "~/detected_cube", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "~/markers", 10)
        self.rgb_pub = self.create_publisher(
            Image, "~/camera/color/image_raw", qos_profile_sensor_data
        )
        self.depth_pub = self.create_publisher(
            Image, "~/camera/depth/image_raw", qos_profile_sensor_data
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, "~/camera/color/camera_info", qos_profile_sensor_data
        )
        self.depth_info_pub = self.create_publisher(
            CameraInfo, "~/camera/depth/camera_info", qos_profile_sensor_data
        )
        self.tf_pub = self.create_publisher(TFMessage, "/tf", 100)

        # ---- robot description (what rviz's RobotModel draws) -----------
        # A URDF never changes, so it is published *transient local* -- the ROS 2
        # "latched": an rviz that starts after the node still gets it.  The
        # shipped layout asks for exactly that durability; volatile subscribers
        # (rviz's default) are served by the low-rate republish below.
        self.declare_parameter("urdf_path", "")
        self.robot_desc_pub = self.create_publisher(
            String,
            "/robot_description",
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self._robot_description = self._load_robot_description()
        self._served_subscribers = 0
        if self._robot_description is not None:
            self.robot_desc_pub.publish(String(data=self._robot_description))

        # ---- services / action ------------------------------------------
        self.reset_srv = self.create_service(
            ResetScene, "~/reset", self._on_reset, callback_group=self._cb_group
        )
        self.action_server = ActionServer(
            self,
            PickPlace,
            "~/pick_place",
            execute_callback=self._execute_pick_place,
            goal_callback=self._on_goal,
            cancel_callback=self._on_cancel,
            callback_group=self._cb_group,
        )

        # ---- timers -----------------------------------------------------
        status_hz = max(0.1, float(self.get_parameter("status_hz").value))
        image_hz = max(0.1, float(self.get_parameter("image_hz").value))
        self._status_timer = self.create_timer(1.0 / status_hz, self._publish_state)
        self._vision_timer = self.create_timer(1.0 / image_hz, self._publish_vision)
        self._robot_desc_timer = self.create_timer(2.0, self._serve_robot_description)

        self._shut_down = False
        if not self.sim.start():
            raise RuntimeError(f"the plant thread failed to start: {self.sim.startup_error}")
        self.get_logger().info(
            f"Pro7 pick & place up: project={self.project_root}, "
            f"model={self.scene.model_path_used}, seed={self.scene.seed}, "
            f"pad={np.round(self.scene.place_target, 3).tolist()}, "
            f"sim_hz={self.sim.sim_hz:g}, "
            f"images={'on' if self._publish_images else 'off'}"
        )

    # ------------------------------------------------------------------ #
    # publishing
    # ------------------------------------------------------------------ #
    def _load_robot_description(self) -> Optional[str]:
        """URDF text for rviz, or ``None`` (with a logged reason) if unreadable."""
        path = robot_description.path_for(
            self.scene.model_path_used,
            str(self.get_parameter("urdf_path").value or ""),
        )
        try:
            text, notes = robot_description.read(
                path, os.path.join(self.project_root, "assets")
            )
        except OSError as exc:
            self.get_logger().warn(
                f"no robot description at {path} ({exc.strerror}); rviz's "
                f"RobotModel will stay empty -- generate it with: python3 "
                f"{os.path.join(self.project_root, 'tools', 'convert_arm_urdf.py')}"
            )
            return None
        for note in notes:
            self.get_logger().warn(f"robot description: {note}")
        links, joints = robot_description.counts(text)
        self.get_logger().info(
            f"robot description: {path} ({links} links, {joints} joints) "
            "-> /robot_description"
        )
        return text

    def _serve_robot_description(self) -> None:
        """Re-send the URDF when a new subscriber shows up.

        The transient-local publish covers late joiners that ask for it (the
        shipped rviz layout does); this covers a subscriber on volatile QoS,
        which is rviz's default.  It deliberately fires only when the subscriber
        count *grows*: rviz rebuilds its whole RobotModel on every message, so
        republishing on a timer would re-parse the meshes twice a second.
        """
        if self._robot_description is None:
            return
        served = self.robot_desc_pub.get_subscription_count()
        if served <= self._served_subscribers:
            return
        self._served_subscribers = served
        self.robot_desc_pub.publish(String(data=self._robot_description))

    def _frame_id(self, name: str) -> str:
        """Frame id with the optional ``tf_prefix``.

        Keeps the simulated tree (``world``, ``base``, ``link1``...) out of the
        way of a real robot publishing to the same global ``/tf``.
        """
        return f"{self.tf_prefix}{name}" if self.tf_prefix else name

    def _header(self, stamp=None, frame_id: Optional[str] = None) -> Header:
        header = Header()
        header.stamp = self.get_clock().now().to_msg() if stamp is None else stamp
        header.frame_id = self._frame_id(WORLD_FRAME) if frame_id is None else frame_id
        return header

    def _publish_state(self) -> None:
        state = self.scene.state()
        header = self._header()
        phase, running, steps, picked, placed = self._job_summary()

        status = PickPlaceStatus()
        status.header = header
        status.phase = phase
        status.running = running
        status.holding = bool(state.holding)
        status.picked = bool(picked)
        status.placed = bool(placed)
        status.steps = int(steps)
        status.detected = state.detected is not None
        status.detected_cube = _point(
            state.detected if state.detected is not None else state.cube_position
        )
        status.cube_position = _point(state.cube_position)
        status.place_target = _point(state.place_target)
        status.place_error = float(state.place_error)
        status.gripper_closure = float(state.hand_closure)
        self.status_pub.publish(status)

        joints = JointState()
        joints.header = header
        joints.name = list(state.joint_names)
        joints.position = [float(value) for value in state.joint_positions]
        joints.velocity = [float(value) for value in state.joint_velocities]
        self.joint_pub.publish(joints)

        cube = PoseStamped()
        cube.header = header
        cube.pose = _pose(state.cube_position)
        self.cube_pub.publish(cube)

        message = TFMessage()
        for frame in state.frames:
            transform = TransformStamped()
            transform.header.stamp = header.stamp
            transform.header.frame_id = self._frame_id(frame.parent)
            transform.child_frame_id = self._frame_id(frame.child)
            transform.transform.translation.x = float(frame.position[0])
            transform.transform.translation.y = float(frame.position[1])
            transform.transform.translation.z = float(frame.position[2])
            transform.transform.rotation.w = float(frame.quaternion[0])
            transform.transform.rotation.x = float(frame.quaternion[1])
            transform.transform.rotation.y = float(frame.quaternion[2])
            transform.transform.rotation.z = float(frame.quaternion[3])
            message.transforms.append(transform)
        self.tf_pub.publish(message)

    def _job_summary(self) -> Tuple[str, bool, int, bool, bool]:
        """``(phase, running, steps, picked, placed)`` of the active/last job."""
        task = self.sim.task()
        if task is not None:
            return task.phase, True, task.steps, task.ever_holding, False
        last = self.sim.last_result()
        if last is not None:
            return "idle", False, int(last.steps), bool(last.picked), bool(last.placed)
        return "idle", False, 0, False, False

    def _publish_vision(self) -> None:
        state = self.scene.state()
        stamp = self.get_clock().now().to_msg()

        if state.detected is not None:
            detected = PoseStamped()
            detected.header = self._header(stamp)
            detected.pose = _pose(state.detected)
            self.detected_pub.publish(detected)
        self.marker_pub.publish(self._markers(state, stamp))

        # The plant thread renders the frames; here we only publish the latest.
        self.sim.image_enabled = self._publish_images and (
            self.rgb_pub.get_subscription_count() > 0
            or self.depth_pub.get_subscription_count() > 0
            or self.camera_info_pub.get_subscription_count() > 0
            or self.depth_info_pub.get_subscription_count() > 0
        )
        images = self.sim.images()
        if not self.sim.image_enabled or images is None:
            return
        rgb, depth = images
        header = self._header(stamp, self._frame_id(CAMERA_OPTICAL_FRAME))
        self.rgb_pub.publish(self._image_msg(rgb, "rgb8", header))
        self.depth_pub.publish(self._image_msg(depth.astype(np.float32), "32FC1", header))
        camera = self._camera_info_msg(header)
        self.camera_info_pub.publish(camera)
        self.depth_info_pub.publish(camera)

    def _image_msg(self, image: np.ndarray, encoding: str, header: Header) -> Image:
        image = np.ascontiguousarray(image)
        message = Image()
        message.header = header
        message.height = int(image.shape[0])
        message.width = int(image.shape[1])
        message.encoding = encoding
        message.is_bigendian = False
        message.step = int(image.strides[0])
        message.data = image.tobytes()
        return message

    def _camera_info_msg(self, header: Header) -> CameraInfo:
        _fovy, width, height, fx, fy, cx, cy = self.scene.camera_info()
        info = CameraInfo()
        info.header = header
        info.width = int(width)
        info.height = int(height)
        info.distortion_model = "plumb_bob"
        info.d = [0.0] * 5
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return info

    def _markers(self, state, stamp) -> MarkerArray:
        array = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        def marker(name: str, ident: int, kind: int, pose, scale, colour) -> Marker:
            item = Marker()
            item.header = self._header(stamp)
            item.ns = name
            item.id = ident
            item.type = kind
            item.action = Marker.ADD
            item.pose = pose
            item.scale = Vector3(x=scale[0], y=scale[1], z=scale[2])
            item.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=colour[3])
            return item

        pad_z = self.gc.CUBE_Z - self.gc.CUBE_SIDE / 2.0
        array.markers.append(
            marker("pad", 0, Marker.CYLINDER,
                   _pose([state.place_target[0], state.place_target[1], pad_z + 0.001]),
                   (0.11, 0.11, 0.004), (0.12, 0.55, 0.60, 0.55))
        )
        array.markers.append(
            marker("cube", 1, Marker.CUBE, _pose(state.cube_position),
                   (self.gc.CUBE_SIDE,) * 3, (1.0, 0.05, 0.05, 0.85))
        )
        if state.detected is not None:
            array.markers.append(
                marker("detected", 2, Marker.SPHERE, _pose(state.detected),
                       (0.02, 0.02, 0.02), (0.1, 0.9, 0.2, 0.8))
            )
        task = self.sim.task()
        if task is not None and task.running:
            array.markers.append(
                marker("goal", 3, Marker.SPHERE, _pose(task.target),
                       (0.02, 0.02, 0.02), (1.0, 0.9, 0.1, 0.8))
            )
        return array

    # ------------------------------------------------------------------ #
    # reset service
    # ------------------------------------------------------------------ #
    def _on_reset(self, request, response):
        if self.sim.busy:
            response.success = False
            response.message = "busy: a pick & place goal is running"
            response.scene_id = int(self.scene.state().scene_id)
            return response

        cube = _resolve_point(request.cube_position)
        place_target = _resolve_point(request.place_target, self.scene.place_target)
        randomize = bool(request.randomize_cube)
        seed = int(request.seed) if randomize else None
        scene_id = self.sim.reset(
            seed=seed,
            randomize_cube=randomize,
            cube_xyz=cube,
            place_target=place_target,
        )
        state = self.scene.state()
        response.success = scene_id is not None
        response.scene_id = int(scene_id if scene_id is not None else state.scene_id)
        response.cube_position = _point(state.cube_position)
        response.place_target = _point(state.place_target)
        response.message = (
            f"scene {scene_id} ready (seed {self.scene.seed}, "
            f"cube {np.round(state.cube_position, 3).tolist()})"
            if scene_id is not None else "reset failed"
        )
        self.get_logger().info(response.message)
        return response

    # ------------------------------------------------------------------ #
    # pick & place action
    # ------------------------------------------------------------------ #
    def _on_goal(self, goal_request) -> GoalResponse:
        if self.sim.busy:
            self.get_logger().warn("rejecting goal: a pick & place goal is already running")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _on_cancel(self, _goal_handle) -> CancelResponse:
        return CancelResponse.ACCEPT

    def _execute_pick_place(self, goal_handle):
        goal = goal_handle.request
        seed = None if math.isnan(float(goal.seed)) else int(goal.seed)
        request = PickPlaceRequest(
            place_target=_resolve_point(goal.place_target, self.scene.place_target),
            cube_hint=_resolve_point(goal.cube_hint),
            seed=seed,
            randomize_cube=bool(goal.reset_scene),
            pick_only=bool(goal.pick_only),
        )
        job = self.sim.submit(request)
        if job is None:
            self.get_logger().warn("goal aborted: the cell is busy")
            goal_handle.abort()
            return self._result_message(False, False, False, 0, 0.0, "busy")

        self.get_logger().info(
            f"pick & place started (pad {np.round(self.scene.place_target, 3).tolist()}, "
            f"seed {self.scene.seed}, pick_only={bool(goal.pick_only)})"
        )
        # Always answer with one feedback frame, even if the goal finishes before
        # the first loop iteration (sim_hz:=0 runs the expert as fast as it can).
        goal_handle.publish_feedback(self._feedback_message(job.snapshot()))
        cancelled = False
        while rclpy.ok():
            snapshot = job.snapshot()
            if snapshot.finished:
                break
            if goal_handle.is_cancel_requested and not cancelled:
                cancelled = self.sim.cancel()
                self.get_logger().info("cancel requested: stopping the expert")
            goal_handle.publish_feedback(self._feedback_message(snapshot))
            time.sleep(self._feedback_period)

        snapshot = job.snapshot()
        outcome = snapshot.result
        state = self.scene.state()
        picked = bool(outcome.picked) if outcome is not None else snapshot.ever_holding
        placed = bool(outcome.placed) if outcome is not None else False
        result = self._result_message(
            # A pick-only goal is done as soon as the cube sits in the jaws.
            picked and (placed or bool(goal.pick_only)),
            picked,
            placed,
            int(outcome.steps) if outcome is not None else snapshot.steps,
            float(state.place_error),
            getattr(outcome, "reason", "") if outcome is not None else "no result",
        )
        if goal_handle.is_cancel_requested and cancelled:
            goal_handle.canceled()
        elif result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        self.get_logger().info(
            f"pick & place finished: picked={result.picked} placed={result.placed} "
            f"steps={result.steps} pad error={result.place_error * 1000:.1f} mm "
            f"({result.message})"
        )
        return result

    @staticmethod
    def _feedback_message(snapshot) -> "PickPlace.Feedback":
        feedback = PickPlace.Feedback()
        feedback.phase = snapshot.phase
        feedback.steps = int(snapshot.steps)
        feedback.holding = bool(snapshot.holding)
        feedback.distance_to_goal = float(snapshot.distance)
        feedback.place_error = float(snapshot.place_error)
        return feedback

    @staticmethod
    def _result_message(success, picked, placed, steps, place_error, message):
        result = PickPlace.Result()
        result.success = bool(success)
        result.picked = bool(picked)
        result.placed = bool(placed)
        result.steps = int(steps)
        result.place_error = float(place_error)
        result.message = str(message)
        return result

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Stop the plant thread and release the MuJoCo renderers."""
        if self._shut_down:
            return
        self._shut_down = True
        self.sim.shutdown()
        self.scene.close()

    def destroy_node(self):
        self.shutdown()
        return super().destroy_node()


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = PickPlaceNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        # ``rclpy`` handles SIGTERM by shutting the context down, which makes a
        # spurious spin raise here; stop quietly instead of dumping a traceback.
        if rclpy.ok():
            node.get_logger().warn(f"executor stopped: {exc}")
    finally:
        executor.shutdown()
        node.shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    import sys

    from .runtime import ensure_runtime

    ensure_runtime(("rclpy", "mujoco"), command="node", argv=sys.argv[1:])
    main(sys.argv[1:])
