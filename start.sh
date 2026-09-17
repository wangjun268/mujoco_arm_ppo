#!/usr/bin/env bash
#
# 一键启动：MuJoCo 七轴抓取放置（ROS 2）
#
#   ./start.sh                      构建(如需) → 起全部节点：pick_place_node + rviz2
#   ./start.sh --no-rviz            无显示器 / 服务器上启动
#   ./start.sh --demo               起来后自动跑一次完整取放
#   ./start.sh --demo --demo-seed 1 --pick-only
#   ./start.sh --with-robot         额外拉起真机栈（先填 ros2_ws/src/pro7_pick_place_ros/config/robot_stack.sh）
#   ./start.sh --build              强制重新 colcon build
#   ./start.sh --force              已有实例在跑时也强制再起一个
#   ./start.sh --stop               停掉正在跑的实例（含 rviz / 真机栈）
#
# 其它：--domain ID（换 ROS_DOMAIN_ID）、--namespace NS、--tf-prefix P、
#       --sim-hz HZ、--seed N、--place X,Y,Z
#
# 退出：Ctrl-C 会停掉本次启动的全部进程（含 --with-robot 拉起的真机栈）。
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$HERE/ros2_ws"
PACKAGE="pro7_pick_place_ros"
LAUNCH_FILE="pick_place.launch.py"

RVIZ=true
DEMO=false
DEMO_SEED=""
PICK_ONLY=false
WITH_ROBOT=false
FORCE=false
DO_BUILD=false
STOP_ONLY=false
DOMAIN=""
NAMESPACE=""
TF_PREFIX=""
SIM_HZ=""
SEED=""
PLACE=""
EXTRA_PIDS=()

usage() {
  sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
}

_need_value() {  # $1 = 选项, $2 = 值
  if [[ -z "${2:-}" ]]; then
    echo "错误：$1 需要一个值" >&2
    usage >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-rviz)     RVIZ=false ;;
    --rviz)        RVIZ=true ;;
    --demo)        DEMO=true ;;
    --demo-seed)   _need_value "$1" "${2:-}"; DEMO_SEED="$2"; shift ;;
    --pick-only)   PICK_ONLY=true ;;
    --with-robot)  WITH_ROBOT=true ;;
    --stop)        STOP_ONLY=true ;;
    --force)       FORCE=true ;;
    --build)       DO_BUILD=true ;;
    --domain)      _need_value "$1" "${2:-}"; DOMAIN="$2"; shift ;;
    --namespace)   _need_value "$1" "${2:-}"; NAMESPACE="$2"; shift ;;
    --tf-prefix)   _need_value "$1" "${2:-}"; TF_PREFIX="$2"; shift ;;
    --sim-hz)      _need_value "$1" "${2:-}"; SIM_HZ="$2"; shift ;;
    --seed)        _need_value "$1" "${2:-}"; SEED="$2"; shift ;;
    --place)       _need_value "$1" "${2:-}"; PLACE="$2"; shift ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "错误：未知参数 $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# --------------------------------------------------------------------------- #
# 环境准备
# --------------------------------------------------------------------------- #
if [[ -n "$DOMAIN" ]]; then export ROS_DOMAIN_ID="$DOMAIN"; fi

if [[ "$STOP_ONLY" == true ]]; then
  # 先按 Ctrl-C 的方式优雅停（launch 会把 SIGINT 转发给子进程，rclpy 会走正常退出路径），
  # 几秒后还没停就兜底 SIGTERM/KILL。模式都带字符类，避免匹配到自己这行命令。
  pkill -INT -f "[r]os2 launch pro7_pick_place_ros" 2>/dev/null || true
  pkill -INT -f "[p]ro7_pick_place_ros.entry" 2>/dev/null || true
  pkill -INT -f "[r]viz2 .*pro7_pick_place_ros" 2>/dev/null || true
  if [[ "$WITH_ROBOT" == true ]]; then
    pkill -INT  -f "[r]obot_control_launch.py" 2>/dev/null || true
    pkill -INT  -f "[r]os2 launch hand_control_ui" 2>/dev/null || true
    pkill -INT  -f "[c]apture_node" 2>/dev/null || true
  fi
  sleep 3
  pkill -TERM -f "[r]os2 launch pro7_pick_place_ros" 2>/dev/null || true
  pkill -TERM -f "[p]ro7_pick_place_ros.entry" 2>/dev/null || true
  pkill -TERM -f "[r]viz2 .*pro7_pick_place_ros" 2>/dev/null || true
  if [[ "$WITH_ROBOT" == true ]]; then
    pkill -TERM -f "[r]obot_control_launch.py" 2>/dev/null || true
    pkill -TERM -f "[r]os2 launch hand_control_ui" 2>/dev/null || true
    pkill -TERM -f "[c]apture_node" 2>/dev/null || true
  fi
  sleep 2
  if [[ "$WITH_ROBOT" == true ]]; then
    echo "已发送停止信号（pro7_pick_place 实例 + 真机栈）"
  else
    echo "已发送停止信号（pro7_pick_place 实例）"
  fi
  exit 0
fi

set +u                       # ROS 的 setup.bash 不能在 set -u 下 source
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
set -u

if [[ ! -d "$WS" ]]; then
  echo "错误：找不到工作空间 $WS" >&2
  exit 1
fi

if [[ "$DO_BUILD" == true || ! -f "$WS/install/setup.bash" ]]; then
  echo "==> colcon build --symlink-install"
  ( cd "$WS" && colcon build --symlink-install )
fi

set +u
source "$WS/install/setup.bash"
set -u

NODE_NAME="${NAMESPACE:+$NAMESPACE/}pro7_pick_place"
NODE_FQN="/${NODE_NAME}"

# 已有实例先拦一下：两个同名节点会抢同一个 action，客户端会收到串掉的结果。
if [[ "$FORCE" == false ]] && ros2 node list 2>/dev/null | grep -qx "$NODE_FQN"; then
  echo "已有实例在跑：$NODE_FQN"
  echo "  · 想停掉它：在那台终端按 Ctrl-C"
  echo "  · 想再起一个：加 --force（或用 --namespace / --domain 区分开）"
  exit 1
fi

LAUNCH_ARGS=("rviz:=$RVIZ")
[[ -n "$NAMESPACE" ]] && LAUNCH_ARGS+=("namespace:=$NAMESPACE")
[[ -n "$TF_PREFIX" ]] && LAUNCH_ARGS+=("tf_prefix:=$TF_PREFIX")
[[ -n "$SIM_HZ"    ]] && LAUNCH_ARGS+=("sim_hz:=$SIM_HZ")
[[ -n "$SEED"      ]] && LAUNCH_ARGS+=("seed:=$SEED")
[[ -n "$PLACE"     ]] && LAUNCH_ARGS+=("place_target:=[$PLACE]")
if [[ "$DEMO" == true ]]; then
  LAUNCH_ARGS+=("demo:=true")
  [[ -n "$DEMO_SEED" ]] && LAUNCH_ARGS+=("demo_seed:=$DEMO_SEED")
  [[ "$PICK_ONLY" == true ]] && LAUNCH_ARGS+=("demo_pick_only:=true")
fi

# --------------------------------------------------------------------------- #
# 可选的“真机栈”
# --------------------------------------------------------------------------- #
cleanup() {
  local pid
  for pid in ${EXTRA_PIDS[@]+"${EXTRA_PIDS[@]}"}; do
    kill "$pid" 2>/dev/null || true
  done
  sleep 1
  for pid in ${EXTRA_PIDS[@]+"${EXTRA_PIDS[@]}"}; do
    kill -9 "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

start_robot_stack() {
  local preset="$WS/src/pro7_pick_place_ros/config/robot_stack.sh"
  ROBOT_STACK_ENABLED=0
  ROBOT_STACK_WS=""
  ROBOT_STACK_CMDS=()
  if [[ -f "$preset" ]]; then
    # shellcheck disable=SC1090
    source "$preset"
  fi
  if [[ "${ROBOT_STACK_ENABLED:-0}" != "1" ]]; then
    echo "[真机栈] 未启用，跳过（要一起起：编辑 $preset，把 ROBOT_STACK_ENABLED 改成 1）"
    return 0
  fi
  if [[ -z "${ROBOT_STACK_WS:-}" || ! -f "$ROBOT_STACK_WS/install/setup.bash" ]]; then
    echo "[真机栈] 找不到工作空间：${ROBOT_STACK_WS:-未设置}" >&2
    return 1
  fi
  local log_dir="$WS/log"
  mkdir -p "$log_dir"
  local cmd name log
  for cmd in "${ROBOT_STACK_CMDS[@]}"; do
    name="$(echo "$cmd" | awk '{print $2"-"$3}' | tr -c 'A-Za-z0-9._-' '_')"
    log="$log_dir/robot_stack_${name}.log"
    echo "[真机栈] 启动：$cmd"
    echo "          日志：$log"
    (
      set +u
      source "$ROBOT_STACK_WS/install/setup.bash"
      set -u
      exec $cmd
    ) > "$log" 2>&1 &
    EXTRA_PIDS+=("$!")
  done
}

if [[ "$WITH_ROBOT" == true ]]; then
  start_robot_stack
fi

# --------------------------------------------------------------------------- #
# 摘要 + 前台启动（Ctrl-C 停全部）
# --------------------------------------------------------------------------- #
if [[ "$RVIZ" == true ]]; then
  NODES_LINE="+  /rviz2"
else
  NODES_LINE="(无 rviz)"
fi

cat <<EOF

== MuJoCo 七轴抓取放置 · 一键启动 ==
  节点    $NODE_FQN  $NODES_LINE
  话题    /${NODE_NAME}/status  /${NODE_NAME}/joint_states  /${NODE_NAME}/cube_pose
          /${NODE_NAME}/detected_cube  /${NODE_NAME}/camera/{color,depth}/image_raw
          /${NODE_NAME}/markers  /tf
  服务    /${NODE_NAME}/reset
  Action  /${NODE_NAME}/pick_place
  跑一次  $WS/run.sh ros2 run ${PACKAGE} pick_place_client --seed 1 --reset
  停止    Ctrl-C

EOF

cd "$WS"
# 后台起 + wait：这样被 kill / timeout 时 trap 能立刻回收子进程；
# Ctrl-C（终端会把 SIGINT 发给整个前台进程组）也照常优雅退出。
ros2 launch "$PACKAGE" "$LAUNCH_FILE" "${LAUNCH_ARGS[@]}" &
LAUNCH_PID=$!
EXTRA_PIDS+=("$LAUNCH_PID")
wait "$LAUNCH_PID" || true
