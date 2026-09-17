# 真机栈启动清单（`./start.sh --with-robot` 会读这个文件）。
#
# 默认关闭：机械臂/手部会真的动，启用前请确认工作台、急停与现场安全。
# 启用方法：把 ROBOT_STACK_ENABLED 改成 1，并按需取消注释 ROBOT_STACK_CMDS 里的行。
#
# 下面的 robot_control 一行是从当前正在跑的进程里抄出来的（robot_new/humanrobot_new，
# DDS domain 164 + CycloneDDS + three_five 工作台）；改工作台/网卡配置时同步改这里。

ROBOT_STACK_ENABLED=0

# 真机工作空间：start.sh 会先 source 它的 install/setup.bash，再执行下面的命令
ROBOT_STACK_WS="/home/wj/robot_new/humanrobot_new"

ROBOT_STACK_CMDS=(
  "ros2 launch robot_control robot_control_launch.py workspace:=three_five use_batch_motion:=1 domain_id:=164 rmw_implementation:=rmw_cyclonedds_cpp cyclonedds_config:=/home/wj/robot_new/humanrobot_new/config/cyclonedds.xml"
  # "ros2 launch robot_control hand_joint_receiver.launch.py"
  # "ros2 launch hand_control_ui hand_ui.launch.py"
  # "ros2 run s6d_camera_capture capture_node"
)
