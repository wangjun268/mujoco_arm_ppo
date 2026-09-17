"""Command-line client for the Pro7 pick & place node.

Send one goal and follow the expert's feedback::

    ros2 run pro7_pick_place_ros pick_place_client                 # default pad
    ros2 run pro7_pick_place_ros pick_place_client --seed 1 --reset
    ros2 run pro7_pick_place_ros pick_place_client --place-x 0.6 --place-y 0.45
    ros2 run pro7_pick_place_ros pick_place_client --pick-only     # grasp only
    ros2 run pro7_pick_place_ros pick_place_client --reset-scene   # just reset

Exit status is 0 when the goal succeeded (and for a successful reset).
"""

from __future__ import annotations

import argparse
import math
import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from geometry_msgs.msg import Point
from pro7_pick_place_interfaces.action import PickPlace
from pro7_pick_place_interfaces.srv import ResetScene

DEFAULT_ACTION = "/pro7_pick_place/pick_place"
DEFAULT_RESET_SERVICE = "/pro7_pick_place/reset"


def _optional(value: str) -> float:
    """``"nan"`` (or an empty string) means "let the node decide"."""
    if value is None or value == "" or value.lower() == "nan":
        return math.nan
    return float(value)


class PickPlaceClient(Node):
    """Thin rclpy wrapper around the action + reset service."""

    def __init__(self, action: str = DEFAULT_ACTION, reset_service: str = DEFAULT_RESET_SERVICE):
        super().__init__("pro7_pick_place_client")
        self.action_name = action
        self.reset_name = reset_service
        self.action_client = ActionClient(self, PickPlace, action)
        self.reset_client = self.create_client(ResetScene, reset_service)

    # ------------------------------------------------------------------ #
    def reset_scene(self, seed: int = 0, randomize: bool = False, cube=None,
                    place=None, timeout: float = 10.0):
        if not self.reset_client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {self.reset_name} not available")
        request = ResetScene.Request()
        request.seed = int(seed)
        request.randomize_cube = bool(randomize)
        request.cube_position = self._point(cube)
        request.place_target = self._point(place)
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        return future.result()

    def pick_and_place(self, place=None, cube=None, seed=None, reset_scene: bool = False,
                       pick_only: bool = False, timeout: float = 300.0):
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            raise RuntimeError(f"action {self.action_name} not available")
        goal = PickPlace.Goal()
        goal.place_target = self._point(place)
        goal.cube_hint = self._point(cube)
        goal.seed = math.nan if seed is None else float(seed)
        goal.reset_scene = bool(reset_scene)
        goal.pick_only = bool(pick_only)

        self._last_phase = None
        send = self.action_client.send_goal_async(goal, feedback_callback=self._feedback)
        rclpy.spin_until_future_complete(self, send, timeout_sec=10.0)
        handle = send.result()
        if handle is None or not handle.accepted:
            raise RuntimeError("goal rejected by the node")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        return result_future.result()

    # ------------------------------------------------------------------ #
    @staticmethod
    def _point(xyz) -> Point:
        point = Point()
        if xyz is None:
            point.x = point.y = point.z = math.nan
        else:
            point.x, point.y, point.z = (float(value) for value in xyz)
        return point

    def _feedback(self, message) -> None:
        feedback = message.feedback
        if feedback.phase != self._last_phase or feedback.steps % 50 == 0:
            self._last_phase = feedback.phase
            print(
                f"  [{feedback.phase:>9}] step {feedback.steps:5d}"
                f"  holding={int(feedback.holding)}"
                f"  |goal|={feedback.distance_to_goal * 1000:6.1f} mm"
                f"  pad error={feedback.place_error * 1000:6.1f} mm",
                flush=True,
            )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--place-x", default="nan")
    parser.add_argument("--place-y", default="nan")
    parser.add_argument("--place-z", default="nan")
    parser.add_argument("--cube-x", default="nan")
    parser.add_argument("--cube-y", default="nan")
    parser.add_argument("--cube-z", default="nan")
    parser.add_argument("--seed", type=int, default=None,
                        help="scene seed (NaN/omitted keeps the current scene)")
    parser.add_argument("--reset", action="store_true",
                        help="reshuffle the cube before starting")
    parser.add_argument("--pick-only", action="store_true",
                        help="stop after the pinch (no carry, no release)")
    parser.add_argument("--reset-scene", action="store_true",
                        help="only rebuild the cell, then exit")
    parser.add_argument("--action", default=DEFAULT_ACTION)
    parser.add_argument("--reset-service", default=DEFAULT_RESET_SERVICE)
    parser.add_argument("--timeout", type=float, default=300.0)
    # ``ros2 launch`` / ``ros2 run`` append ``--ros-args -r ...``: hand those to
    # rclpy (they remap this node's name) and only feed the rest to argparse.
    raw = list(sys.argv if argv is None else [sys.argv[0], *argv])
    args = parser.parse_args(remove_ros_args(raw)[1:])

    def point(x, y, z):
        values = (_optional(x), _optional(y), _optional(z))
        return None if all(math.isnan(v) for v in values) else values

    place = point(args.place_x, args.place_y, args.place_z)
    cube = point(args.cube_x, args.cube_y, args.cube_z)

    rclpy.init(args=raw)
    client = PickPlaceClient(args.action, args.reset_service)
    status = 1
    try:
        if args.reset_scene:
            response = client.reset_scene(seed=args.seed or 0, randomize=args.reset)
            if response is None:
                print("reset timed out", file=sys.stderr)
            else:
                print(f"reset: success={response.success} scene={response.scene_id} "
                      f"cube=({response.cube_position.x:.3f}, {response.cube_position.y:.3f}, "
                      f"{response.cube_position.z:.3f}) -- {response.message}")
                status = 0 if response.success else 1
            return status

        print(f"goal: place={place} cube={cube} seed={args.seed} "
              f"reset={args.reset} pick_only={args.pick_only}")
        wrapped = client.pick_and_place(
            place=place, cube=cube, seed=args.seed,
            reset_scene=args.reset, pick_only=args.pick_only, timeout=args.timeout,
        )
        if wrapped is None:
            print("goal timed out", file=sys.stderr)
            return 1
        result = wrapped.result
        print(
            f"result: success={result.success} picked={result.picked} "
            f"placed={result.placed} steps={result.steps} "
            f"pad error={result.place_error * 1000:.1f} mm ({result.message})"
        )
        status = 0 if result.success else 1
    except (RuntimeError, KeyboardInterrupt) as exc:
        print(f"client error: {exc}", file=sys.stderr)
    finally:
        client.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return status


if __name__ == "__main__":
    sys.exit(main())
