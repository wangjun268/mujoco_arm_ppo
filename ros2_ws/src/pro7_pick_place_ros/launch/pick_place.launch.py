"""Bring up every node of the Pro7 pick & place cell in one command.

Started by default:

* ``pick_place_node`` -- the node itself (topics, ``reset`` service, ``pick_place``
  action),
* ``rviz2`` with ``config/pick_place.rviz`` -- TF tree, markers and the
  eye-in-hand image (turn it off with ``rviz:=false`` on a headless machine).

Optionally:

* ``demo:=true`` -- also runs ``pick_place_client`` once, so the whole cycle
  (approach -> pinch -> carry -> release) plays by itself 8 s after boot.

Examples::

    ros2 launch pro7_pick_place_ros pick_place.launch.py
    ros2 launch pro7_pick_place_ros pick_place.launch.py rviz:=false
    ros2 launch pro7_pick_place_ros pick_place.launch.py demo:=true demo_seed:=1 demo_reset:=true
    ros2 launch pro7_pick_place_ros pick_place.launch.py seed:=3 sim_hz:=500.0
    ros2 launch pro7_pick_place_ros pick_place.launch.py params_file:=/tmp/my.yaml

Every ``key:=value`` argument is optional: an empty value means "keep whatever
the parameter file (or the node's own default) says", so the shipped
``config/pick_place.yaml`` stays in charge unless you override a single knob.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

PACKAGE = "pro7_pick_place_ros"
NODE_NAME = "pro7_pick_place"

#: Launch argument -> (parameter name, converter).  Empty value = keep default.
_OVERRIDES = {
    "project_root": str,
    "model_path": str,
    "seed": int,
    "idle_mode": str,
    "sim_hz": float,
    "vision_hz": float,
    "status_hz": float,
    "image_hz": float,
    "feedback_hz": float,
}
_BOOL_OVERRIDES = ("randomize_cube", "publish_images")
_DEMO_DELAY = 8.0  # s: give the node time to build the scene before the client


def _as_bool(text: str) -> bool:
    return text.strip().lower() in ("1", "true", "yes", "on")


def _as_floats(text: str) -> list:
    return [float(part) for part in text.strip().strip("[]").split(",") if part.strip()]


def _launch_setup(context, *_args, **_kwargs):
    """Build the node (plus the optional demo client) from the launch context."""
    parameters = [LaunchConfiguration("params_file").perform(context)]
    overrides = {}
    for name, convert in _OVERRIDES.items():
        raw = LaunchConfiguration(name).perform(context).strip()
        if raw:
            overrides[name] = convert(raw)
    for name in _BOOL_OVERRIDES:
        raw = LaunchConfiguration(name).perform(context).strip()
        if raw:
            overrides[name] = _as_bool(raw)
    raw_target = LaunchConfiguration("place_target").perform(context).strip()
    if raw_target:
        overrides["place_target"] = _as_floats(raw_target)
    if overrides:
        parameters.append(overrides)

    actions = [
        Node(
            package=PACKAGE,
            executable="pick_place_node",
            name=NODE_NAME,
            namespace=LaunchConfiguration("namespace"),
            output="screen",
            parameters=parameters,
            condition=IfCondition(LaunchConfiguration("simulate")),
        )
    ]

    client_args = ["--timeout", "600"]
    seed = LaunchConfiguration("demo_seed").perform(context).strip()
    if seed:
        client_args += ["--seed", seed]
    if _as_bool(LaunchConfiguration("demo_reset").perform(context) or "false"):
        client_args.append("--reset")
    if _as_bool(LaunchConfiguration("demo_pick_only").perform(context) or "false"):
        client_args.append("--pick-only")
    actions.append(
        TimerAction(
            period=_DEMO_DELAY,
            actions=[
                Node(
                    package=PACKAGE,
                    executable="pick_place_client",
                    name="pick_place_client",
                    namespace=LaunchConfiguration("namespace"),
                    output="screen",
                    arguments=client_args,
                )
            ],
            condition=IfCondition(LaunchConfiguration("demo")),
        )
    )
    return actions


def generate_launch_description() -> LaunchDescription:
    share = FindPackageShare(PACKAGE)
    default_params = PathJoinSubstitution([share, "config", "pick_place.yaml"])

    arguments = [
        DeclareLaunchArgument("simulate", default_value="true",
                              description="start the pick & place node"),
        DeclareLaunchArgument("rviz", default_value="true",
                              description="start rviz2 with the shipped layout"),
        DeclareLaunchArgument("demo", default_value="false",
                              description="also run one pick_place_client goal"),
        DeclareLaunchArgument("demo_seed", default_value="",
                              description="seed of the demo goal (empty -> node default)"),
        DeclareLaunchArgument("demo_reset", default_value="false",
                              description="demo goal reshuffles the cube first"),
        DeclareLaunchArgument("demo_pick_only", default_value="false",
                              description="demo goal stops after the pinch"),
        DeclareLaunchArgument("namespace", default_value="",
                              description="node namespace (default: none)"),
        DeclareLaunchArgument("params_file", default_value=default_params,
                              description="parameter file of the node"),
        DeclareLaunchArgument("project_root", default_value="",
                              description="mujoco_arm_ppo checkout (empty -> auto-detect)"),
        DeclareLaunchArgument("model_path", default_value="",
                              description="MuJoCo model (empty -> project default)"),
        DeclareLaunchArgument("seed", default_value=""),
        DeclareLaunchArgument("randomize_cube", default_value=""),
        DeclareLaunchArgument("place_target", default_value="",
                              description='drop point, e.g. "[0.58, 0.52, 0.475]"'),
        DeclareLaunchArgument("sim_hz", default_value="",
                              description="plant rate; 0 runs unthrottled"),
        DeclareLaunchArgument("idle_mode", default_value="",
                              description="freeze | step"),
        DeclareLaunchArgument("vision_hz", default_value=""),
        DeclareLaunchArgument("status_hz", default_value=""),
        DeclareLaunchArgument("image_hz", default_value=""),
        DeclareLaunchArgument("feedback_hz", default_value=""),
        DeclareLaunchArgument("publish_images", default_value=""),
    ]

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        namespace=LaunchConfiguration("namespace"),
        output="screen",
        arguments=["-d", PathJoinSubstitution([share, "config", "pick_place.rviz"])],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(arguments + [OpaqueFunction(function=_launch_setup), rviz])
