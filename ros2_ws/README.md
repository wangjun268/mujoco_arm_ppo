# Pro7 七轴抓取放置 · ROS 2 节点

把本仓库已有的「源台抓取 → 提起 → 搬运 → 落到目标台放置垫 → 松开」能力
封装成一个 ROS 2 节点：

- **话题**输出机械臂/灵巧手关节状态、TF、关节相机 RGB-D、检测到的红块位置；
- **服务**重建场景（重新采样方块、改放置点）；
- **Action** 跑一次完整取放（带阶段反馈、可取消）。

节点本身**不重新实现任何控制律**：物理、视觉和解析专家全部来自仓库根部的
`grasp_common.py` / `detect_red_cube.py`，改一次参数两边同时生效
（`pick_place_demo.py` 跑什么，节点就跑什么）。

---

## 1. 目录结构

```text
ros2_ws/
├── run.sh                                  # 环境准备 + build/test/run 便捷入口
└── src/
    ├── pro7_pick_place_interfaces/          # ament_cmake：msg / srv / action
    │   ├── msg/PickPlaceStatus.msg
    │   ├── srv/ResetScene.srv
    │   └── action/PickPlace.action
    └── pro7_pick_place_ros/                 # ament_python：节点
        ├── pro7_pick_place_ros/
        │   ├── project.py    # 定位 mujoco_arm_ppo 仓库并导入其模块
        │   ├── scene.py      # 线程安全的 MuJoCo 场景封装（状态 / TF / 视觉 / 复位）
        │   ├── simulator.py  # 植物线程 + 唯一的取放作业（可取消）
        │   ├── node.py       # ROS 2 节点：话题、服务、Action
        │   ├── client.py     # 命令行客户端 pick_place_client
        │   ├── runtime.py    # 解释器选择 + libstdc++ 预加载
        │   └── entry.py      # 控制台入口（先修解释器，再导入 rclpy/mujoco）
        ├── launch/pick_place.launch.py
        ├── config/pick_place.yaml, pick_place.rviz
        └── test/             # 13 个用例：场景 / 专家一致性 / 作业 / 端到端 Action
```

依赖：ROS 2 Jazzy（`rclpy`、`std_msgs`、`geometry_msgs`、`sensor_msgs`、
`visualization_msgs`、`tf2_msgs`、`rosidl_default_generators`）+ 项目本身的
`mujoco` / `numpy`（见仓库根 `requirements.txt`）。

## 2. 构建与运行

一键启动（会自动构建、起全部节点、Ctrl-C 全停）：

```bash
cd /home/wj/mujoco_arm_ppo
./start.sh                       # = pick_place_node + rviz2
./start.sh --demo                # 再自动跑一次完整取放
./start.sh --demo --demo-seed 1 --pick-only
./start.sh --no-rviz             # 无显示器 / 服务器
./start.sh --stop                # 停掉正在跑的实例（含 rviz）
./start.sh --with-robot          # 连真机栈一起起（默认关闭，见 §5.2）
./start.sh --help                # 全部选项
```

桌面上双击也行：把 `pro7_pick_place.desktop` 拷到 `~/.local/share/applications/`
（或直接双击该文件 → “允许启动”）。

`start.sh` 选项：

| 选项 | 作用 |
|---|---|
| `--demo` / `--demo-seed N` / `--pick-only` | 起完节点后自动发一个 goal（可指定 seed、只抓不放） |
| `--no-rviz` | 只起节点（无显示器时用） |
| `--stop` | 停掉正在跑的实例（含 rviz；`--with-robot` 时连真机栈一起停） |
| `--build` | 强制 `colcon build`（默认只在 `install/` 不存在时构建） |
| `--force` | 已有同名实例在跑时也强制再起一个（默认会拦下来并提示） |
| `--domain ID` / `--namespace NS` / `--tf-prefix P` | 换 `ROS_DOMAIN_ID` / 命名空间 / TF 前缀 |
| `--sim-hz HZ` / `--seed N` / `--place X,Y,Z` | 传给节点的仿真速率 / 场景 seed / 放置点 |

工作空间内部的手工操作：

```bash
cd ros2_ws
./run.sh                 # 首次自动 colcon build，然后【起来所有节点】
                         #   = pick_place_node + rviz2（带本包 rviz 配置）
./run.sh ros2 launch pro7_pick_place_ros pick_place.launch.py rviz:=false   # 无显示器
./run.sh build           # colcon build --symlink-install
./run.sh test            # pytest（13 个用例，约 20 s）
./run.sh ros2 topic list # 任意命令都在“已 source 好”的环境里执行
```

等价的手工流程：

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install && source install/setup.bash

# 一条命令把所有节点起来：节点 + rviz2
ros2 launch pro7_pick_place_ros pick_place.launch.py
ros2 launch pro7_pick_place_ros pick_place.launch.py rviz:=false            # 无显示器
ros2 launch pro7_pick_place_ros pick_place.launch.py demo:=true demo_seed:=1 demo_reset:=true

# 只想单起一个节点（不带 rviz）
ros2 run pro7_pick_place_ros pick_place_node
ros2 run pro7_pick_place_ros pick_place_client --seed 1 --reset
```

本包只有两个可执行文件（`ros2 pkg executables pro7_pick_place_ros`）：
`pick_place_node`（常驻节点）和 `pick_place_client`（跑一次就退出的客户端）；
再加上外部的 `rviz2`，就是 launch 默认拉起的全部节点。

### 2.1 launch 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `simulate` | `true` | 起不起 `pick_place_node` |
| `rviz` | `true` | 起不起 `rviz2`（带 `config/pick_place.rviz`：TF + markers + 相机图） |
| `demo` | `false` | 额外跑一次 `pick_place_client`（节点起来 8 s 后发一个 goal，跑完整取放） |
| `demo_seed` / `demo_reset` / `demo_pick_only` | `""` / `false` / `false` | 那个 demo goal 的 seed / 是否重采样 / 是否只抓不放 |
| `namespace` | `""` | 节点命名空间（同时作用于节点、rviz、demo 客户端） |
| `params_file` | `config/pick_place.yaml` | 节点参数文件 |
| 其余 | 空 | `seed` / `place_target` / `sim_hz` / `rviz` … 见 §4；留空 = 用参数文件或节点默认值 |

## 3. 接口

### 3.1 话题（都在节点私有命名空间下）

| 话题 | 类型 | 说明 |
|---|---|---|
| `/pro7_pick_place/status` | `PickPlaceStatus` | 阶段 / 是否夹住 / 步数 / 检测结果 / 落点误差（默认 20 Hz） |
| `/pro7_pick_place/joint_states` | `sensor_msgs/JointState` | 7 个臂关节 + 21 个 L20 手关节，名字与模型一致 |
| `/pro7_pick_place/cube_pose` | `PoseStamped` | 红块真值位姿（`world` 坐标系） |
| `/pro7_pick_place/detected_cube` | `PoseStamped` | 眼在手 RGB-D 的估计值（没看到红块时不发） |
| `/pro7_pick_place/camera/color/image_raw` | `sensor_msgs/Image` | `rgb8`，默认 320×240，5 Hz |
| `/pro7_pick_place/camera/depth/image_raw` | `sensor_msgs/Image` | `32FC1`，单位 m |
| `/pro7_pick_place/camera/{color,depth}/camera_info` | `CameraInfo` | 针孔模型（`plumb_bob`，零畸变），frame 为 `cam_hand_optical` |
| `/pro7_pick_place/markers` | `visualization_msgs/MarkerArray` | 放置垫、方块、检测点、当前子目标（给 rviz） |
| `/tf` | `tf2_msgs/TFMessage` | `world → base → link1…link7 → gripper → tool0 / cam_hand → cam_hand_optical` |

> 相机话题只在**有订阅者**时才渲染：没人看就不做额外的 OpenGL 渲染。
> 图像/深度按 **ROS 光学坐标系**（x 右、y 下、z 沿视线）发布，深度是沿视线的
> z 距离，因此可以直接喂给 `image_geometry` / `cv_bridge`。

### 3.2 服务

`/pro7_pick_place/reset`（`pro7_pick_place_interfaces/srv/ResetScene`）：重建场景。
`randomize_cube=true` 时换一个种子重新采样；也可以给定 `cube_position` 精确摆放，
`place_target` 改落点（全 0 / NaN 表示保持当前值）。

```bash
ros2 service call /pro7_pick_place/reset pro7_pick_place_interfaces/srv/ResetScene \
  "{randomize_cube: true, seed: 7}"
```

### 3.3 Action

`/pro7_pick_place/pick_place`（`pro7_pick_place_interfaces/action/PickPlace`）

| 字段 | 方向 | 说明 |
|---|---|---|
| `place_target` | goal | 落点（世界坐标，m）；缺省/NaN 分量沿用当前落点 |
| `cube_hint` | goal | 方块初值；全 NaN 表示用相机检测 |
| `seed` / `reset_scene` | goal | 是否在开始前重采样方块（`seed` 给定则落点可复现） |
| `pick_only` | goal | 只抓不放（等价于 `grasp_demo` 的任务） |
| `phase` / `steps` / `holding` / `distance_to_goal` / `place_error` | feedback | 约 5 Hz |
| `success` / `picked` / `placed` / `steps` / `place_error` / `message` | result | `message` 是专家给出的原因（`placed`、`dropped on lift`、`grasped`…） |

## 4. 参数

默认值见 `src/pro7_pick_place_ros/config/pick_place.yaml`，也可以用 `-p name:=value` 覆盖。

| 参数 | 默认 | 说明 |
|---|---|---|
| `project_root` | `""` | `mujoco_arm_ppo` 仓库路径；空 = 自动查找（也认 `MUJOCO_ARM_PPO_ROOT`） |
| `model_path` | `""` | MuJoCo 模型；空 = `assets/rokae_xmate_pro7_pick_real.xml` |
| `seed` / `randomize_cube` | `0` / `false` | 方块落点；`true` 时每个 goal 换一个采样 |
| `place_target` | `[0.58, 0.52, 0.475]` | 放置点（同时把可视放置垫挪到该点）；`[0,0,0]` = 用项目默认 |
| `sim_hz` | `240.0` | 植物线程速率；**`0` = 不限速**（测试/批处理用） |
| `idle_mode` | `freeze` | 空闲时保持当前位姿；`step` = 继续用最后的控制量步进物理 |
| `vision_hz` | `5.0` | 空闲时的检测频率（`0` = 关） |
| `status_hz` / `image_hz` | `20.0` / `5.0` | 状态与相机/标记的发布频率 |
| `publish_images` | `true` | 关掉则完全不渲染相机图 |
| `feedback_hz` | `5.0` | Action 反馈频率 |
| `tf_prefix` | `""` | 例如 `sim/`，把仿真 TF 树与真机 `/tf` 区分开 |

## 5. 用法示例

```bash
# 1) 默认落点，跑一次完整取放
ros2 run pro7_pick_place_ros pick_place_client --seed 1 --reset

# 2) 只抓不放
ros2 run pro7_pick_place_ros pick_place_client --seed 1 --reset --pick-only

# 3) 换落点（只给 x/y，z 沿用当前值）
ros2 run pro7_pick_place_ros pick_place_client --place-x 0.62 --place-y 0.46

# 4) 只重建场景
ros2 run pro7_pick_place_ros pick_place_client --reset-scene --reset --seed 7

# 5) 看状态 / 看相机 / 看 TF
ros2 topic echo --once /pro7_pick_place/status
ros2 topic hz /pro7_pick_place/camera/color/image_raw
ros2 run rviz2 rviz2 -d src/pro7_pick_place_ros/config/pick_place.rviz

# 6) 一条命令：节点 + rviz 全起来，并且自动跑一次取放
ros2 launch pro7_pick_place_ros pick_place.launch.py demo:=true demo_seed:=1 demo_reset:=true
```

客户端退出码：`0` = 任务成功（`pick_only` 时夹住即成功），`1` = 失败/被拒/超时。

### 5.1 要和别的节点一起起

本包不假设自己独占系统：话题都在私有命名空间下，`/tf` 可以用 `tf_prefix` 隔离。

```bash
# 和真机/其它 ROS 2 系统并存：换个 domain，或加命名空间 + tf 前缀
ROS_DOMAIN_ID=77 ros2 launch pro7_pick_place_ros pick_place.launch.py
ros2 launch pro7_pick_place_ros pick_place.launch.py \
  namespace:=sim tf_prefix:=sim/ sim_hz:=240.0
```

想把自己仓库里的其它节点和它一起拉起来，在自己的 launch 里 include 本文件即可：

```python
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare

IncludeLaunchDescription(
    PythonLaunchDescriptionSource([
        FindPackageShare("pro7_pick_place_ros"), "/launch/pick_place.launch.py"
    ]),
    launch_arguments={"rviz": "false", "namespace": "sim"}.items(),
)
```

### 5.2 连真机栈一起起（`--with-robot`）

`src/pro7_pick_place_ros/config/robot_stack.sh` 是“真机栈启动清单”，**默认关闭**：

```bash
ROBOT_STACK_ENABLED=0     # 改成 1 才会被 --with-robot 执行
ROBOT_STACK_WS="/home/wj/robot_new/humanrobot_new"
ROBOT_STACK_CMDS=(
  "ros2 launch robot_control robot_control_launch.py workspace:=three_five ..."
  # "ros2 launch robot_control hand_joint_receiver.launch.py"
  # "ros2 launch hand_control_ui hand_ui.launch.py"
  # "ros2 run s6d_camera_capture capture_node"
)
```

`start.sh --with-robot` 会先 source 该工作空间的 `install/setup.bash`，再逐条后台启动，
日志写到 `ros2_ws/log/robot_stack_*.log`，Ctrl-C（或 `start.sh --with-robot --stop`）
会一起停掉。真机命令是照你当前在跑的那条抄的，改工作台/网卡/domain 时同步改这个文件。

## 6. 实现要点

1. **只有一根植物线程**（`simulator.py`）。MuJoCo 的 `Renderer` 是 OpenGL 对象，
   在不同线程里创建/使用/销毁会**段错误**（实测：一边渲染一边重建场景必崩）。
   因此节点用 `defer_renderers=True` 建场景，由植物线程负责建渲染器、渲染相机图、
   执行复位；ROS 定时器只读取快照。`scene.py` 里的 `_assert_renderer_thread()`
   会把这类误用变成明确报错而不是崩溃。
2. **作业在植物线程上一步一步跑**。`grasp_common.iter_pick_and_place()` 是解析专家的
   逐控制步生成器（`pick_and_place()` 是它的阻塞包装，行为逐位一致，有测试钉住）。
   节点每个 tick 推进一步，所以取放是「实时可见」的，期间状态/相机照常发布；
   `sim_hz` 决定播放速度，`sim_hz:=0` 直接不限速跑完。
3. **同一时刻只有一个作业**。`ActionServer` 的 goal 回调在忙时直接 REJECT，
   取消请求会让专家停在原地（结果 `message="cancelled"`）。
4. **解释器/libstdc++ 兼容**（`runtime.py`）。本机 `rclpy` 属于系统 Python 而
   MuJoCo 装在 Conda 里，且 Conda 自带的 `libstdc++` 太旧。控制台入口会先找一个
   「两个都能 import」的解释器，必要时预加载系统 `libstdc++` 后 `execve` 过去；
   可用 `PRO7_PYTHON` 指定解释器、`PRO7_SKIP_PRELOAD=1` 关闭预加载。
5. 话题全部用**私有名**（`~/status`），不会占用别处的 `/joint_states` 之类；
   `/tf` 是全局的，用得上时请配 `tf_prefix`。

## 7. 测试

```bash
./run.sh test          # = pytest src/pro7_pick_place_ros/test
```

| 用例 | 覆盖 |
|---|---|
| `test_scene.py` | 关节数/落点/夹持状态、TF 链、检测精度、放置点搬移、seed 可复现 |
| `test_expert_steps.py` | 逐控制步专家 vs 阻塞专家：结果、步数、方块位置逐位一致 |
| `test_job.py` | 作业走完六个阶段、`pick_only`、取消语义 |
| `test_node_end_to_end.py` | 真的起节点 + ActionClient 发目标，`success/placed/place_error` 校验；忙时拒单 |

> `colcon test` 也能跑，但它使用系统 Python（没有 `mujoco`），用例会 SKIP；
> 想跑完整用例请用 `./run.sh test`（或把 `PRO7_PYTHON` 指到项目的解释器）。

## 8. 已知问题

### rviz 里看不到东西？

按这四条查（前两条最常见）：

1. **配置没加载**：rviz 必须用本包的布局文件启动，Fixed Frame 才是 `world`、显示项才齐全：

   ```bash
   ros2 run rviz2 rviz2 -d \
     $(ros2 pkg prefix pro7_pick_place_ros)/share/pro7_pick_place_ros/config/pick_place.rviz
   ```

   直接用 `ros2 run rviz2 rviz2` 打开的是空工程：左侧 Displays 里得自己加
   TF / MarkerArray / Image，并把 Global Options → Fixed Frame 改成 `world`。
   （`pick_place.rviz` 之前漏装进 `share/`，`ros2 launch` 里 `-d` 指向不存在的文件就会静默空跑，已修。）
2. **不在同一个 domain / RMW**：`echo $ROS_DOMAIN_ID` 要和节点一致，
   `ros2 node list` 里应能看到 `/pro7_pick_place` 和 `/rviz2`。
   你机器上的真机栈是 `domain 164 + CycloneDDS`，本节点默认 domain 0；
   两边不一致时 rviz 看不到任何话题。
3. **Fixed Frame 没有 TF**：节点按 20 Hz 发 `/tf`（`world → base → link1…link7 → gripper → tool0`）。
   `ros2 topic hz /tf` 应该在动；如果用了 `tf_prefix:=sim/`，
   rviz 里的 Fixed Frame 要相应改成 `sim/world`。
4. **相机图是“有订阅才渲染”**：Image 显示项加载后（Best Effort QoS）图像才会出现，
   `ros2 topic hz /pro7_pick_place/camera/color/image_raw` 应从 0 变成 ~5 Hz。

一条命令自查：

```bash
ros2 node list && ros2 topic hz /tf &   # Ctrl-C 退出
ros2 topic info /pro7_pick_place/markers --verbose | grep -i node
```

- 该项目本身「夹持偏临界」：`grasp_common.pick_and_place` 目前对具体落点敏感，
  实测 seed 0~4 只有 seed 1 能完整放置（`tests/test_grasp.py` 里用 xfail 记录了这一点）。
  节点只是忠实地把专家的结果（含失败原因）报出来，不做额外补偿；
  同一 seed 的结果是可复现的。
- 关节力矩是**直接控制**（`data.ctrl` 归一化力矩），没有真实控制器的位置/速度环；
  这条链路只复现 MuJoCo 里的专家，不是给真机下发的接口。
