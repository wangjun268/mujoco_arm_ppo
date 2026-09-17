#!/usr/bin/env bash
#
# Convenience wrapper around the ros2_ws workspace.
#
#   ./run.sh                       # build if needed, then start every node
#                                  # (pick_place_node + rviz2)
#   ./run.sh build                 # colcon build --symlink-install
#   ./run.sh test                  # pytest suite (needs a python with mujoco+rclpy)
#   ./run.sh ros2 topic list       # any command inside the prepared environment
#
# What it prepares:
#   * /opt/ros/$ROS_DISTRO/setup.bash and this workspace's install/setup.bash
#   * MUJOCO_ARM_PPO_ROOT -> the checkout that holds grasp_common.py
#   * LD_PRELOAD of the system libstdc++, which the Conda python needs before it
#     can import rclpy's extension modules (see pro7_pick_place_ros/runtime.py;
#     unset PRO7_SKIP_PRELOAD to opt out)
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$here")"

# ROS's setup files are not written for `set -u`.
set +u
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u
export MUJOCO_ARM_PPO_ROOT="${MUJOCO_ARM_PPO_ROOT:-$project_root}"

if [[ -z "${PRO7_SKIP_PRELOAD:-}" && -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi

build() {
  cd "$here" && colcon build --symlink-install "$@"
}

if [[ ! -f "$here/install/setup.bash" ]]; then
  build
fi
set +u
source "$here/install/setup.bash"
set -u

case "${1:-}" in
  build) shift; build "$@" ;;
  test)  shift; cd "$here" && exec "${PRO7_PYTHON:-python3}" -m pytest \
           src/pro7_pick_place_ros/test "$@" ;;
  "")    exec ros2 launch pro7_pick_place_ros pick_place.launch.py ;;
  *)     exec "$@" ;;
esac
