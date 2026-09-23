# 项目架构与逐文件说明

> 用 **MuJoCo** 做物理仿真、**Gymnasium** 定义强化学习环境、**Stable-Baselines3 PPO** 训练，让 **珞石 xMate Pro7 七自由度臂**的末端达到随机目标点，并在同一套真实网格场景上完成视觉抓取与整段取放。当前注册 4 个环境名（`pro7_urdf` 到达、`pro7_pick` / `pro7_pick_urdf` 抓取、`pro7_pick_place` 取放），**全部使用官方 URDF STL 网格，没有任何胶囊体基元模型**。

> **末端执行器更新**：抓取场景的平行两指夹爪已替换为**灵心巧手 LinkerHand L20**（厂商 URDF +
> 22 个 STL 见 `assets/linkerhand_l20/`，由 `tools/convert_hand_urdf.py` 生成 MuJoCo 片段并标定
> 抓取姿态）。L20 的 21 个关节按开合协同驱动，`grasp.common.scene_ids()` 提供 `hand_*` 标识，
> `is_grasped()` 改为"至少 N 根手指接触方块"。下文中"夹爪 / 两指 / 钳口"的描述属于更换前的
> 历史实现。

> **腕部转接件**：手装在腕部法兰前方 60 mm（真实转接件的位置），中间原来是空的，臂和手在
> MuJoCo / rviz 里看着是断开的。现在由 `tools/make_wrist_flange.py` 生成回转体转接件
> `assets/meshes/pro7_l20_flange.stl`，抓取/取放场景（`rokae_xmate_pro7_pick_real.xml`）与
> **真实网格版到达训练模型**（`rokae_xmate_pro7_real.xml`，`pro7_urdf`）都以**纯视觉、零质量**
> 的 geom 挂上（`contype=0` / `density=0`）。400 步随机控制下 qpos 与加之前
> **逐位相同**，手眼相机画面 76800 像素零差异：标定、训练好的策略、rollout 都不受影响。

---

## 1. 项目目标

这是一个“从物理建模到强化学习训练再到可视化演示”的完整闭环：

1. 用 MuJoCo XML 描述机械臂（关节、连杆、目标、驱动、传感器）。
2. 把模型包成标准的 `gymnasium.Env`（定义观测/动作/奖励/重置逻辑）。
3. 用 PPO 学一个策略，让末端在尽可能少的步数内到达随机目标。
4. 提供离线评估（指标 + 学习曲线 + 视频）和在线可视化（MuJoCo 实况窗口）。

全部环境共享同一套训练/评估/可视化流程，通过 `--env` 参数切换。

---

## 2. 总体架构（数据流）

```text
 ① 物理模型（XML，全部为真实 STL 网格）
  assets/rokae_xmate_pro7_real.xml        assets/rokae_xmate_pro7_pick_real.xml
         │                                         │
         │ mujoco.MjModel.from_xml_path()          │
         ▼                                         ▼
② 任务封装（Gymnasium.Env）
 env/rokae_reacher.py                    env/rokae_pro7_pick.py
                                         env/rokae_pro7_pick_place.py
         │       ② 工厂/注册表
         │       env/__init__.py  →  make_env(name)
         ▼
 ③ 训练（Stable-Baselines3 PPO）
  train_ppo.py  ──►  results/ppo_<env>.zip（策略权重）
                            │
                            ├──► eval_rollout.py  指标 + learning_curve.png + rollout.mp4/gif
                            ├──► viewer_demo.py   实时 MuJoCo 窗口
                            └──► make_montage.py  逼近→命中 静态拼图
```

等价地，分层如下：

| 层次 | 文件 | 职责 |
|---|---|---|
| **物理层** | `assets/*.xml` | 定义刚体、关节、几何、驱动、传感器、相机、灯光、目标 |
| **环境基类** | `env/base_reacher.py` | 所有「到达」任务共享的观测/动作/奖励/渲染实现（只写一遍） |
| **环境层** | `env/rokae_reacher.py`、`env/rokae_pro7_pick.py`、`env/rokae_pro7_pick_place.py` | 各自只定义模型、末端 site、起点/目标采样与抓取逻辑 |
| **工厂层** | `env/__init__.py` | 环境注册表与 `make_env()`，供所有脚本按名字取环境 |
| **RL 层** | `train_ppo.py` | PPO 策略网络、向量化环境、回调统计、断点续训 |
| **评估/可视化层** | `eval_rollout.py`、`viewer_demo.py`、`make_montage.py` | 加载策略做指标、视频、窗口、拼图 |
| **抓取层** | `grasp/`（`common.py`、`detect.py`、`policy.py` + 演示/训练/可视化脚本） | 抓取场景常量、专家伺服、视觉定位、策略网络（训练与回放共用） |
| **共享模块层** | `paths.py`、`cli.py`、`live_viewer.py` | 路径/默认值集中管理、统一的 `--env/--model/--out` 命令行与策略加载、节流实时窗口 |
| **模型工具层** | `tools/`（`build_dual_arm_model.py`、`convert_*_urdf.py`、`make_wrist_flange.py`、`urdf_selfcheck.py`） | 从厂商 URDF / 清洗描述重新生成 `assets/` 下的 MJCF、URDF、STL |
| **测试层** | `tests/` | pytest 冒烟 + 回归测试（观测维度/顺序、路径、抓取契约） |
| **文档/产物层** | `README.md`、`PROJECT_GUIDE.md`、`results/` | 快速上手、本说明、训练产物 |

---

## 3. 目录结构

根目录只留 **3 个共享模块 + 4 个入口脚本**，其余按职责进包：

- `paths.py` / `cli.py` / `live_viewer.py`：所有脚本都要用的共享模块（`paths.py` 是仓库根的
  **唯一锚点**，路径全部由它算出来，ROS 节点也靠它找仓库）。
- `train_ppo.py` / `eval_rollout.py` / `viewer_demo.py` / `make_montage.py`：到达（reach）主流程的
  四个入口，最常用，留在根目录保持 `python3 train_ppo.py ...` 的手感。
- `grasp/`：抓取层整条链路（场景、视觉、策略、演示、训练、可视化）。
- `tools/`：模型工具层，把 `assets/` 里的资产从厂商/清洗描述重新生成出来。
- `env/`、`tests/`、`ros2_ws/`、`results/`：环境包、测试、ROS 2 封装、训练产物。

每个脚本**两种跑法都支持**：`python3 grasp/demo.py`（脚本）和 `python3 -m grasp.demo`（模块）；
`tools/*` 同理。包内文件用 `from grasp.common import ...` 这种绝对导入，入口脚本自己在开头把
仓库根放上 `sys.path`（`if __package__ in (None, ""): ...`），所以从任意工作目录执行都不会找不到模块。

```text
mujoco_arm_ppo/
├── README.md                 # 快速上手：安装/训练/评估/查看器
├── PROJECT_GUIDE.md          # 本文档：架构 + 逐文件说明
├── PROJECT_SUMMARY.md        # 当前状态/结果/产物汇总
├── requirements.txt          # 依赖清单（mujoco / sb3 / torch / ...）
├── setup.cfg                 # flake8 + pytest 配置
├── start.sh                  # ROS 2 一键启动（构建 + 起节点 + rviz2）
├── pro7_pick_place.desktop   # 同上，双击版
│
│   —— 共享模块（全项目都从这里取路径 / 命令行 / 窗口）——
├── paths.py                  # 项目路径、workspace 默认值、模型路径校验（仓库根锚点）
├── cli.py                    # 统一 --env/--model/--out 命令行与策略加载
├── live_viewer.py            # 节流的原生 MuJoCo 实时窗口（训练/演示共用）
│
│   —— 到达（reach）入口脚本 ——
├── train_ppo.py              # PPO 训练入口
├── eval_rollout.py           # 评估 + 学习曲线 + 演示视频
├── viewer_demo.py            # 实时 MuJoCo 可视化窗口
├── make_montage.py           # 逼近→命中 静态拼图
│
├── env/                      # 环境包（Gymnasium）
│   ├── __init__.py           # make_env 注册表
│   ├── base_reacher.py       # 到达任务基类（观测/动作/奖励/渲染唯一实现）
│   ├── rokae_reacher.py      # Pro7 真实网格到达环境（子类）
│   ├── rokae_pro7_pick.py    # Pro7 视觉抓取环境（obs 30）
│   ├── rokae_pro7_pick_place.py # Pro7 取放整段环境（obs 38）
│   └── dual_arm_reacher.py   # 双臂到达 + 双臂协作（bar）任务（obs 60, act 14）
├── grasp/                    # 抓取层：相机 + L20 手 + 红块整条链路
│   ├── __init__.py           # 包说明（故意不 import 子模块，保持 `import grasp` 轻量）
│   ├── common.py             # 抓取场景常量 + 干扰块 + 专家伺服（唯一来源）
│   ├── detect.py             # 红色分割 + 深度反投影 → 红块 3D 坐标
│   ├── policy.py             # 抓取策略网络（训练/回放共用）
│   ├── demo.py               # 检测 → 定位 → 伺服 → 抓取
│   ├── pick_place_demo.py    # 源台抓取 → 提起 → 搬运 → 放到目标台放置垫
│   ├── montage.py            # 逼近→抓住 拼图
│   ├── overlay.py            # 检测可视化（分割 + 3D 对比）
│   ├── visualize.py          # 抓取回放 GIF
│   ├── view.py               # 抓取/取放实时窗口（--task grasp|pick_place）
│   ├── supervised.py         # 在线 DAgger / 行为克隆训练
│   └── train_live.py         # 取放训练过程可视化 GIF（默认 --task pick_place）
├── tools/                    # 模型工具层：重新生成 assets/ 下的资产
│   ├── __init__.py           # 包说明
│   ├── build_dual_arm_model.py   # lkwy73_o1 双臂描述 → assets/dual_arm_reach.xml（+ 网格拷贝）
│   ├── convert_hand_urdf.py      # 厂商 L20 URDF → MuJoCo 片段（含抓取标定）
│   ├── convert_arm_urdf.py       # MuJoCo 抓取场景 → URDF（网格 + 惯量，给 rviz2 用）
│   ├── urdf_selfcheck.py         # 把导出的 URDF 重新用 MuJoCo 加载，逐 geom / 质量对比（--check）
│   └── make_wrist_flange.py      # Pro7 法兰 → L20 转接件（回转体 STL，补上 60 mm 间隙）
├── assets/
│   ├── rokae_xmate_pro7_real.xml      # Pro7 七自由度到达模型（真实 STL 网格）
│   ├── rokae_xmate_pro7_pick_real.xml # Pro7 + L20 抓取/取放场景（真实 STL 网格）
│   ├── dual_arm_reach.xml    # lkwy73_o1 双臂任务模型（14 电机 + 双侧 mocap 目标）
│   ├── dual_arm/meshes/*.STL # 双臂降面网格（8.4 MB，单件 ≤ 20k 面）
│   ├── meshes/xMatePro7/*.stl # 官方 Pro7 STL 网格（随仓库提供，相对路径引用）
│   └── linkerhand_l20/       # 厂商 L20 URDF/STL + 生成的 MuJoCo 片段
├── tests/                    # pytest 冒烟 + 回归测试
├── ros2_ws/                  # ROS 2 封装（取放节点 + 客户端 + 接口包）
│   └── src/
│       ├── pro7_pick_place_interfaces/  # msg/srv/action 接口
│       └── pro7_pick_place_ros/         # 节点：话题/服务/Action + 植物线程
└── results/                  # 训练产物（模型/日志/图片/视频/TensorBoard）
    ├── ppo_<env>.zip         # 各环境策略权重
    ├── tb_<env>/             # TensorBoard 事件（训练曲线）
    ├── <env>/                # 该环境 视频/曲线/拼图
    └── logs/<env>_<n>.txt    # 训练 stdout 日志（按环境/轮次命名）
```

重组时文件只挪位置、不改逻辑，旧路径与新路径一一对应：

| 旧路径（根目录扁平） | 新路径 |
|---|---|
| `grasp_common.py` | `grasp/common.py` |
| `detect_red_cube.py` | `grasp/detect.py` |
| `grasp_policy.py` | `grasp/policy.py` |
| `grasp_demo.py` | `grasp/demo.py` |
| `pick_place_demo.py` | `grasp/pick_place_demo.py` |
| `make_grasp_montage.py` | `grasp/montage.py` |
| `detect_overlay.py` | `grasp/overlay.py` |
| `visualize_grasp.py` | `grasp/visualize.py` |
| `view_pick.py` | `grasp/view.py` |
| `supervised_grasp.py` | `grasp/supervised.py` |
| `train_live.py` | `grasp/train_live.py` |
| `build_dual_arm_model.py` | `tools/build_dual_arm_model.py` |
| `convert_arm_urdf.py` | `tools/convert_arm_urdf.py` |
| `convert_hand_urdf.py` | `tools/convert_hand_urdf.py` |
| `make_wrist_flange.py` | `tools/make_wrist_flange.py` |
| `urdf_selfcheck.py` | `tools/urdf_selfcheck.py` |

只有两处内容跟着路径改了一行：`tools/convert_arm_urdf.py` 写进
`assets/rokae_xmate_pro7_pick_real.urdf` 的生成器注释、`tools/convert_hand_urdf.py` 写进
`assets/linkerhand_l20/linkerhand_l20_right.xml` 的同一句话（都是"由谁生成"的自述）。

---

## 4. 逐文件说明

### 4.1 物理模型层 `assets/`

> **Pro7 真实网格**：`rokae_xmate_pro7_real.xml` / `rokae_xmate_pro7_pick_real.xml`
> 引用 `assets/meshes/xMatePro7/*.stl`（8 个官方 STL，约 7 MB，随仓库提供），
> 路径写法相对 XML 所在目录 → 不再依赖项目外的 `/home/wj/rokae_ros2`，
> 换机器/删掉外部仓库都能加载。改网格只需替换 `assets/meshes/` 下的文件。

#### `assets/rokae_xmate_pro7_real.xml` —— Pro7 七自由度到达模型（真实 STL 网格）

`pro7_urdf` 的模型：官方 Pro7 STL 网格 + 官方 URDF 的关节链、偏置与限位。

| 关节 | 轴 | 相对父连杆原点 z (m) | 限位 (rad) |
|---|---|---|---|
| 1 | (0,0,1) | 0 | ±3.0527 |
| 2 | (0,1,0) | 0.404 | ±2.0933 |
| 3 | (0,0,1) | 0.23743878 | ±3.0527 |
| 4 | (0,1,0) | 0.15549959 | ±2.0933 |
| 5 | (0,0,1) | 0.22044633 | ±3.0527 |
| 6 | (0,1,0) | 0.14512568 | ±2.0933 |
| 7 | (0,0,1) | 0.25090877 | ±6.2832 |

- **几何**：8 个官方 STL（`assets/meshes/xMatePro7/*.stl`），`scale=0.001`（URDF 以 mm 为单位）；所有连杆 geom 都是 `contype=0 / conaffinity=0`，到达任务里臂不与任何物体接触。
- **驱动**：7 个 `motor`，`gear = [10, 10, 8, 6, 5, 4, 3]`，`ctrlrange=[-1,1]`。
- **工具尖点**：`tool` site 在 `link7` 上位于局部 `(0, 0.06, 0)`，偏离腕滚转轴，因此 7 个关节都会移动末端（关节 1/3/5/7 约 3 mm、关节 4 约 31 mm、关节 6 约 13 mm、关节 2 约 50 mm @ 0.05 rad）。
- **目标**：`target` 是 `mocap` 无碰撞球体，复位时放到由随机位姿正运动学(FK) 得到的可达点上。
- **零重力**：`gravity="0 0 0"`，成为纯惯性控制问题，保证 PPO 可收敛。
- **相机**：`cam_iso` 等距视角，供渲染/窗口/拼图使用。
- **腕部转接件**：`wrist_flange` 引用 `assets/meshes/pro7_l20_flange.stl`，`density=0` + 无接触，纯视觉（见文首说明）。

#### `assets/rokae_xmate_pro7_pick_real.xml` —— 抓取 / 取放场景（真实 STL 网格）

`pro7_pick`（与别名 `pro7_pick_urdf`）和 `pro7_pick_place` 的模型，在同一个臂上装了整条操作链路：

- **臂 + 手**：同一套 Pro7 网格；腕法兰前方 60 mm 处 `<include>` 灵心巧手 L20 的 MuJoCo 片段（`assets/linkerhand_l20/`），手指沿 +z 伸出、沿 +x 夹紧。
- **工位**：源台 `table`（x>0）上放 5 cm 红块与绿/蓝/黄干扰块，目标台 `table2` 在 +y 侧、台面嵌 `place_pad` 放置垫标出落点。
- **相机**：腕部 `cam_hand` 为 RGB-D 眼在手相机，用于红块检测；`cam_iso` 供整场景渲染。
- **接触分组**：臂/手网格 `contype=0`，红块与干扰块用 `contype=4 / conaffinity=6`，只与 L20 手指和彼此碰撞，避免 34 处手指自接触（详见 `tools/convert_hand_urdf.py`）。

#### `assets/dual_arm_reach.xml` —— lkwy73_o1 双臂任务模型（生成物）

`dual_arm_reach` / `dual_arm_coop` 的模型。**不要手改**：由
`tools/build_dual_arm_model.py` 从清洗过的双臂描述
（`/home/wj/urdf/lkwy73_o1_dual_arm_clean`，原厂 URDF 的 41 links / 40 joints）
生成，并把 29 个降面网格拷进 `assets/dual_arm/meshes/`（68.7 MB → 8.4 MB，
单件最多 20000 面；原模型 `base_link` 有 274744 面，超过 MuJoCo 的 20 万面上限，
直接用原 URDF 会加载失败）。

| 项 | 值 |
|---|---|
| 自由度 | 36 个 hinge（双臂各 7 + 双手各 11），其中 14 个被驱动 |
| 驱动 | 每臂 7 个 `motor`，`gear = [6,6,5,4,3,2,2] Nm`（单臂 4 kg，按 Pro7 比例缩小） |
| 手部 | 22 个手指关节用 `<equality>` 钉在张开位（`polycoef="0 1 0 0 0"`），要控手就删掉该块并加执行器 |
| 末端 | `tip_left` / `tip_right` 站点放在各自四指指尖中心（生成时用 FK 计算，偏差 0 mm） |
| 目标 | 每臂一个 `mocap` 无碰撞球体，mocap 序号 = 臂序号 |
| 自碰撞 | CAD 里 `L1↔base`(−7.1 mm)、`L5↔L7`(−11.2 mm) 及镜像对在任何位姿都重叠，已用 `<contact><exclude>` 排除；零位接触数 0 |
| 其他 | `gravity="0 0 0"`、`timestep=0.02`、`integrator=Euler`，与 Pro7 到达模型同约定；总质量 11.11 kg 与原 URDF 一致 |

镜像关系（左臂 → 右臂）符号为 `(-1,-1,-1,+1,-1,+1,-1)`：双臂沿 y 分列两侧，
对称面是 y=0，绕轴 a 转 q 的镜像 = 绕 `(ax,-ay,az)` 转 −q。用网格点云核对过，
正确符号残差约 3–5 mm（CAD 两件本身的不对称量），符号错误会跳到 75 mm 以上。

---

### 4.2 环境层 `env/`

#### `env/__init__.py` —— 环境注册表与工厂

```python
_REGISTRY = {
    "pro7_urdf": RokaePro7RealReacher,       # Pro7 7-DOF（真实 URDF 网格）
    "pro7_pick": RokaePro7Pick,              # Pro7 + L20 + 相机 + 红块
    "pro7_pick_urdf": RokaePro7PickReal,     # 兼容别名，同一场景
    "pro7_pick_place": RokaePro7PickPlace,   # 取放整段（抓取 → 搬运 → 放置）
    "dual_arm_reach": DualArmReach,          # 双臂各自到达（lkwy73_o1，14 关节）
    "dual_arm_coop": DualArmCoopReach,       # 双臂共持一根刚性杆
}
def make_env(name, **kwargs): ...
def env_names(): ...
```

- 所有脚本通过 `make_env(env_name, ...)` 创建环境，从而用 `--env` 一键切换任务，而不必改代码。
- `env_names()` 返回可选环境列表，用于 argparse 的 `choices`。

#### `env/base_reacher.py` —— `BaseReacher(gym.Env)`（所有到达任务的基类）

这个项目里「到达」任务永远是同一件事，所以**观测/动作/奖励/渲染/步进只实现一次**：

- **模型**：`n` 个力矩驱动的 hinge 关节 + 恰好一个 `mocap` 目标体（无质量、不参与碰撞），
  以及一个 `tip/tool` site 作为末端。
- **观测**：`[cos q, sin q, dq, tip−target, target]`，维度由模型自动推出
  （`3·n_dof + 2·pos_dim`，当前模型 `pos_dim = 3`）。
- **动作**：每关节 `[-1, 1]` 归一化力矩；`step` 里 `np.clip`。
- **奖励**：`-dist - ctrl_cost·Σctrl² - vel_cost·Σqvel² + shaping(dist) + 命中奖励`。
- **渲染**：惰性创建 `mujoco.Renderer`，相机取类属性 `CAMERA`。
- **公开访问器**：`env.target` / `env.tip`，脚本不再需要碰私有字段。

子类只需要提供三样东西：`MODEL_PATH`、`TIP_SITE`/`CAMERA`/`POS_DIM`，
以及 `_reset_episode()`（设置 `qpos/qvel` 并返回目标点）。需要“接近度塑形”
的 Pro7 再额外覆盖 `_shaping()`。

> ⚠️ **观测布局就是 checkpoint 的 ABI**：维度与顺序一旦改变，已训练的
> `results/ppo_*.zip` 就会失效。当前所有到达环境都用**分块**顺序
> `[cos q1.., sin q1.., …]`，`tests/test_envs.py` 会把它钉死。

#### `env/rokae_reacher.py` —— `RokaePro7RealReacher(BaseReacher)`

把 `rokae_xmate_pro7_real.xml` 包装成七自由度到达任务。

- **观测（27 维）**：`[cos q1..7, sin q1..7, dq1..7, tool−target(3), target(3)]`。
- **动作（7 维）**：七关节力矩。
- **目标采样（关键设计）**：以随机“锚点姿态”为中心，起始与目标姿态都在其附近
  采样；目标由目标姿态的**正运动学 (FK)** 得到 → **天然可达**，无需解析逆解。
  初始距离适中、可学（`START_STD=0.10`、`TARGET_STD=0.24`）。
- **奖励**：`-dist − 0.0005·Σqvel² + 0.5·exp(−dist/0.3) + 2(命中)`。指数“接近度”
  塑形让远距离也有连续梯度，是 7 轴能学会的关键。
- **命中阈值**：`target_radius=0.15`，时间上限 `max_steps=250`。

#### `env/dual_arm_reacher.py` —— `MultiArmReacher` 及其两个双臂任务

双臂版本没有沿用 `BaseReacher`：那里的实现假设"一个 tip site + 一个 mocap 目标 +
`nq == nu`"，而双臂模型是 36 个关节里只驱动 14 个。这里改成按 **模型里的臂列表**
驱动，`ARMS` 里加一项就能扩到三臂。

```text
MultiArmReacher(gym.Env)               # 观测/动作/奖励/复位/render 的唯一实现
├── DualArmReach(MultiArmReacher)      # dual_arm_reach：两臂各自独立目标
└── DualArmCoopReach(DualArmReach)     # dual_arm_coop：两目标是一根刚性杆的两端
```

- **观测（60 维）**：每臂 `[cos q(7), sin q(7), dq(7), tip−target(3)]` 共 24 维 →
  两臂 48 维，再接全局 `tips(6)` 与 `targets(6)`，让每条臂都能看到对方的进度。
- **动作（14 维）**：两臂 14 个关节的归一化力矩 `[-1, 1]`，直接写入 `data.ctrl`。
- **目标采样**：与 Pro7 同思路——随机锚点姿态附近采样"起始姿态"和"目标姿态"，
  目标由目标姿态 **FK** 得到，因此天然可达（`START_STD=0.12`、`TARGET_STD=0.30`）。
  复位时最多重抽 25 次，保证两臂起始距离都 ≥ `MIN_START_DIST=0.12 m`
  （否则约 0.5% 的回合会"一开局就成功"）。
- **奖励**：`−mean(dist) − 0.0005·Σdq² + 0.5·exp(−max_dist/0.3)`
  `+ 1.0×新建模的达标臂数 + 2.0(两臂都命中)`。单臂一次性加分是多臂任务的关键：
  否则策略在"两条臂都到位"之前拿不到任何离散信号。
- **终止**：两臂都进入 `target_radius=0.08 m`；`max_steps=200`。
- **协作变体**：目标由**左臂参考姿态 + 其镜像姿态** FK 得到（因此是两臂真的能同时
  摆出的构型），再整体做一次小刚性变换（绕中点的 yaw ≤0.22 rad、平移 ≤0.04 m）。
  奖励额外减去 `2.0×|实际杆向量−目标杆向量|`，成功还要求杆长误差 <
  `PAIR_TOLERANCE=0.05 m`。`env.reference_pose` / `env.achievable_pair` 保留下来
  供测试核对"这一对目标确实可同时达到"（实测目标离可达构型中位 0.06 m、95% 0.14 m）。

---

### 4.3 RL 训练层

#### `train_ppo.py` —— PPO 训练入口

- **CLI**：`--env`（默认 `pro7_urdf`）、`--steps`、`--seed`、`--lr`、`--model`、`--init-model`、
  `--n-envs`、`--device`，以及**实时可视化**开关 `--viewer` / `--viewer-env` / `--viewer-fps`。
- **向量化环境**：`DummyVecEnv` 包 4 个 `Monitor` 环境。
- **策略**：`PPO("MlpPolicy")`，超参 `n_steps=1024`、`batch_size=256`、`n_epochs=10`、`gamma=0.99`、`gae_lambda=0.95`、`clip_range=0.2`、`vf_coef=0.5`，在 `cuda` 上训练。
- **TensorBoard**：写到 `results/tb_<env>/`。
- **EpisodeStats 回调**：逐 episode 记录 `ep_rew_mean` 与逐帧 `success`，训练结束打印。
- **断点续训**：`--init-model <zip>` 会 `PPO.load(...)` 继续学；否则从头建模型。
- **保存**：`model.save(<model>)` → 生成 `results/ppo_<env>.zip`。
- **实时训练可视化**（`--viewer`）：`attach_live_viewer()` 打开原生 MuJoCo 窗口并绑定到
  **被训练的那个子环境**（`venv.envs[i].unwrapped.model/data`，`--viewer-env` 选择第几个），
  `LiveViewerCallback` 每步调用 `viewer.sync()`（内部按 `--viewer-fps` 节流，不拖慢训练），
  并把 `step / fps / ep_rew / ep_len` 画在窗口四角。关闭窗口只是停止刷新，训练继续跑完。
  `model.learn` 包在 `try/finally` 里，异常时也会关窗（否则 UI 线程会吊住进程）。

#### `live_viewer.py` —— 实时窗口（训练/演示共用）

- `LiveViewer(model, data, camera=..., fps=..., key_callback=...)`：
  - `sync()` 发布当前 `data` 状态，**按 fps 节流**，所以"每步都 sync"也不会压垮 UI；
  - `set_camera()` 依次尝试给定相机名（环境自带 `CAMERA` → `cam_iso`）；
  - `set_status([...])` 用 `Handle.set_texts()` 在窗口四角显示训练指标（最多 4 行）；
  - `close()` 关窗后 `sleep(0.6)`，等 UI 线程自己收完 GLFW，避免解释器退出时段错误；
  - `launch=` 可注入（测试用假句柄，**不需要显示器**）。
- `Pacer(speed, sim_dt)`：把循环限制到可观看的速率（`speed=1` 即 1× 实时，
  物理步长 0.02 s → 50 步/秒；`speed<=0` 表示不限速）。训练循环每步 `tick()`
  即可，落后时会自动重新对齐而不追赶。
- 被 `train_ppo.py`、`grasp/train_live.py`、`viewer_demo.py`、`grasp/view.py` 共用，
  启动/相机/关闭这三段样板代码只写一次。

---

### 4.4 评估/可视化层

#### `eval_rollout.py` —— 评估 + 学习曲线 + 演示视频

- **CLI**：`--env`、`--model`、`--episodes`、`--video-episodes`、`--out`、`--viewer`、`--fps`。
- **量化评估**：用确定性策略跑 N 集，统计整集成功率、最终距离均值/中位数、`dist≤0.10` 比例、平均到达步数、平均奖励。
- **学习曲线**：读取 `results/tb_<env>/` 的 TensorBoard 事件，把多次续训的 `ep_rew_mean` 按累计时间步拼接画成 `learning_curve.png`。
- **演示视频**：以 `rgb_array` 渲染多集，用 `imageio` 存 `rollout.gif` 与 `rollout.mp4`。
- **窗口模式**：`--viewer` 时调用 `viewer_demo.run_live_viewer(...)` 打开实况窗口。

#### `viewer_demo.py` —— 实时 MuJoCo 可视化窗口

- 强制软件 OpenGL（`LIBGL_ALWAYS_SOFTWARE=1`、`GALLIUM_DRIVER=llvmpipe`），保证在无 GPU 上下文/软渲染环境下也能开窗。
- `launch_passive(env.model, env.data, key_callback)` 打开原生 MuJoCo Simulate 窗口（非阻塞 UI 线程）。
- **主循环**：`predict → env.step → viewer.sync()`，按 `fps` 限帧；到达/超时/按 `T` 就换新目标。
- 把相机固定为模型自带的 `CAMERA`（Pro7 场景是 `cam_iso`）。
- **收尾**：`finally` 里 `viewer.close()` 后 `sleep(0.6)`，等 daemon UI 线程自行清理 GLFW，避免主线程与它并发 `glfw.terminate()` 导致的段错误。
- **CLI**：`--env/--model/--seed/--fps/--episodes`（`--episodes` 用于自动关闭，默认运行到关窗）。

#### `make_montage.py` —— “逼近→命中”静态拼图

- 加载策略，固定 seed 跑一个 episode，在若干指定时间步截取 `env.render()` 帧，排成网格保存为 `results/<env>/montage.png`，直观展示手臂从初始到命中目标的轨迹。

---

### 4.5 共享模块层（`paths.py` / `cli.py`）

#### `paths.py` —— 项目路径 / 默认值 / 模型校验

- `PROJECT_ROOT / ASSETS_DIR / RESULTS_DIR`、`asset_path()`、`results_path()`：
  所有绝对路径都从这里来，**脚本里不再出现 `/home/wj/...`**，换机器/改目录名不用改代码，
  而且从任意工作目录运行都可以。
  这个文件**必须留在仓库根**：`PROJECT_ROOT` 就是"装着 `paths.py` 的那个目录"，
  `ros2_ws/src/pro7_pick_place_ros/project.py` 也用它（连同 `grasp/` 下的文件）来认仓库。
- `default_model_path(env)` / `default_tb_dir(env)` / `default_out_dir(env)`：
  `results/ppo_<env>.zip`、`results/tb_<env>/`、`results/<env>/` 的唯一出处。
- `ensure_dir(path)`：需要时创建输出目录。
- `resolve_model(model_path)`：若传的是**目录**或**不存在的 `.zip`**，打印清晰提示并退出
  （避免 `IsADirectoryError`/`FileNotFoundError` 这类晦涩报错），否则返回可加载的 `.zip` 路径。

#### `cli.py` —— 统一命令行与策略加载

- `add_env_arg / add_model_arg / add_out_arg / apply_defaults`：
  训练、评估、窗口、拼图脚本共用同一套 `--env`（带 `choices`）、`--model`、`--out` 及默认值，
  新增脚本只需追加自己的参数。
- `load_policy(model_path, device="cpu")`：统一加载 PPO 检查点。MLP 策略在 CPU 上评估更合适，
  也顺带消掉了 SB3 “PPO on GPU without a CNN” 的告警。

---

### 4.6 模型工具层 `tools/`

`assets/` 下的模型都是**生成物**：这几个脚本把厂商 URDF / 清洗过的机器人描述变成本项目要用的
MJCF、URDF 与 STL。改一处参数就跑一次脚本，别手改生成物——测试会重新生成并与仓库里的文件比对。

#### `tools/convert_hand_urdf.py` —— 厂商 L20 URDF → MuJoCo 片段

- 读 `assets/linkerhand_l20/right/` 的 SolidWorks 版 URDF，重新输出
  `linkerhand_l20_right_{assets,body,actuators}.xml`（22 个 body 的 `hand_*` 树、22 个网格声明、
  21 个位置伺服）、`linkerhand_l20_poses.json`（开手 / 抓握预设）与单文件预览模型。
- 顺手标定抓取姿态（`grip` 的一行实测输出），并把结果写进 `poses.json`。
- 直接 `python3 tools/convert_hand_urdf.py` 重新生成；没有参数。

#### `tools/convert_arm_urdf.py` —— MuJoCo 抓取场景 → URDF（给 rviz2）

- 读**编译后的** `assets/rokae_xmate_pro7_pick_real.xml`，把机器人部分重新写成 URDF：
  位姿、轴、限位、geom、惯量全部来自 `mjModel`，所以导出不会和仿真漂移。
- `--hand merged`（默认）把 L20 按开手姿态烘进 `gripper` 连杆；`--hand articulated` 出 22 连杆的
  完整手；`--with-cell` 连台面/放置垫一起出；`--collision` 每个 geom 再加一个 `<collision>`；
  `--mesh-uri package:...` 换成 ROS 包内路径。
- `--check` 调 `tools/urdf_selfcheck.py` 复核；生成物的第三行会写"由哪个脚本生成"，改了脚本名要
  重跑一次（`tests/test_arm_urdf.py::test_committed_urdf_is_regenerable` 会比对）。

#### `tools/urdf_selfcheck.py` —— 导出的 URDF 与 MuJoCo 逐 geom 对比

- 把导出重新用 MuJoCo 加载（把 `<visual>` 改写成 `<collision>`，因为 MuJoCo 读 URDF 只认碰撞体），
  在零位与若干随机关节位对比**每个 geom 的世界位姿与包围半径**，质量按总和比。
- 被 `tools/convert_arm_urdf.py --check` 与 `tests/test_arm_urdf.py` 复用。

#### `tools/make_wrist_flange.py` —— Pro7 法兰 → L20 转接件

- `PROFILE` 是一组 `(半径, z)` 回转母线，转成 `assets/meshes/pro7_l20_flange.stl`，
  补上腕法兰与手底座之间 60 mm 的空档。
- 场景里以**纯视觉、零质量**的 geom 挂载（`contype=0` / `density=0`），质点、惯性、相机画面
  一律不变（`tests/test_wrist_flange.py` 钉住这一点）。

#### `tools/build_dual_arm_model.py` —— lkwy73_o1 双臂描述 → 任务模型

- 读清洗过的双臂描述，生成 `assets/dual_arm_reach.xml`：两个 7 轴臂的归一化力矩电机、
  22 个手指关节用 equality 钉在开手位、每只手一个 `tip` site、两个 mocap 目标、四对 CAD
  自接触 `exclude`；同时把降面网格拷到 `assets/dual_arm/meshes/`。
- `--check` 只校验不写盘。

---

### 4.7 抓取层（相机 + L20 手 + 红块）

#### `grasp/common.py` —— 抓取场景与专家的唯一来源

- **场景常量**：工作台高度 `TABLE_Z`、红块边长 `CUBE_SIDE`、`CUBE_X/CUBE_SPREAD`、
  初始位姿 `START_POSE`、夹爪行程 `GRIPPER_TRAVEL` / 张开间隙 `GRIP_OPEN`。
  `MODEL_PATH = assets/rokae_xmate_pro7_pick_real.xml`（**真实 URDF 网格**）与所有脚本都以这里为准
  （红块边长复用 `grasp.detect.DEFAULT_CUBE_SIDE`，视觉与仿真不可能各说各话）。
- **干扰块**：`DISTRACTORS = ((绿, 0.96/0.15), (蓝, 0.96/-0.15), (黄, 1.00/0.00))`，
  与目标块同尺寸、同质量。位置经过实测挑选：既落在眼在手相机的可见梯形内
  （能看到 → 视觉必须区分），又完全避开红块的采样足迹（不会被误当成目标）。
  `place_distractors()` 在每次 `reset()` 与 `make_scene()` 时把它们摆回原位
  （上一集被手臂撞飞也会复位）。
- **双工位与取放**：`TABLE_2_POS/TABLE_2_SIZE` 是目标台，`PLACE_TARGET` 是台面放置垫上的落点
  （半径 0.80 m，专家约 35 步可达；远离手眼相机视野，不影响红块检测）。
  `pick_and_place()` 是解析专家：接近+夹紧 → 提起 → 搬运 → 下降 → 张开，
  分阶段回调 `on_phase(name)` / 逐步回调 `on_step()` 供演示脚本录像与出图；
  `cube_on_destination()` 判定落点。搬运用**单独的柔增益**（`TRANSPORT_*`，1 cm 分段）：
  夹持力约 0.3 N，快了就会把活动指顶开，与方块质量无关。
- `scene_ids(model)`：按**名字**解析 site/geom/joint/body 及 qpos/dof 地址，
  替代原先散落的 `qpos[7]`、`qpos[8:11]` 这类魔数索引。
- `make_scene(seed, cube_xyz)` / `place_cube()`：构造抓取场景。
- `detect_cube(model, data, renderer)`：眼在手 RGB-D → 红块世界坐标（内部调用 `grasp.detect`）。
- `servo_command(...)`：**专家解析伺服**（分辨率控制 + 力限幅），返回
  `(各臂关节力矩, 是否该闭合夹爪, 当前距离)`；`teacher_action(env)` 把它包成策略用的 8 维动作。
  `grasp.demo`、`grasp.montage`、`grasp.supervised`（DAgger 老师）共用同一套控制律。
- `is_grasped(...)` / `gripper_gap(...)`：抓取成功判据（双指接触 + 间隙 + 中心距），
  环境与脚本共用，不会出现“演示说抓住了、环境说没抓住”。

#### `grasp/policy.py` —— 抓取策略网络

- `Policy`（128-128-Tanh MLP）、`OBS_DIM=30 / ACT_DIM=8`、`action()`、`load_policy()`、`rollout()`。
- 训练脚本与回放脚本都从这里取网络定义，检查点不会因为架构分散而加载不上。

#### `env/rokae_pro7_pick_place.py` —— `RokaePro7PickPlace(gym.Env)`

- `RokaePro7Pick` 的子类：episode **不在夹住时结束**，而是走完整段
  接近 → 夹紧 → 提起 → 搬运 → 下降到放置垫 → 松开，只有方块落到垫上（或步数用尽）才结束。
  抓取失败、方块被撞出台面时提前以 `info["lost"]=True` 结束（否则这种不可恢复状态会
  灌满 DAgger 缓冲区，把在线训练带跑偏——实测过）。
- **观测（38 维）** = 抓取环境的 30 维 + `holding` + 子目标误差 `goal-grasp`(3) +
  方块到垫子的误差 `pad-cube`(3) + `plan.done`（到位该松手的标志）。
- **搬运计划** `grasp.common.PlacePlanner`：`settle(15 步) → lift → carry → lower`，
  每一段以 1 cm 为一步的虚拟路点推进（`PLACE_WAYPOINT`），步进速度 ≈ `kp/kd·waypoint` ≈ 2 cm/s。
  **必须**这么慢：夹爪只有 ~0.3 N 夹紧力，实测超过 ~4 cm/s 方块就被顶出钳口。
- **搬运由环境执行**（`transport_assist=True`）：抓住方块后，环境用
  `TRANSPORT_KP/KD/FORCE` 直接把夹爪中心压在计划路点上。原因：搬运力矩只有
  执行器量程的 1~2%，远低于模仿网络 ~1.4% 的动作噪声——纯模仿的搬运一定掉块
  （实测：2e-4 动作 MSE 的行为克隆放 0/6，脚本搬运 10/10）。策略仍负责**接近、夹紧、
  失败后重试、以及松手时机**，pick % / place % 正是按这些打分。

#### `env/rokae_pro7_pick.py` —— `RokaePro7Pick(gym.Env)`

- **场景**：`assets/rokae_xmate_pro7_pick_real.xml`（真实 URDF 网格 + 夹爪 +
  `cam_hand` 相机 + 红块 + 绿/蓝/黄干扰块 + 源台/目标台两张工作台）。`RokaePro7PickReal` 只是兼容别名，
  两个注册名（`pro7_pick` / `pro7_pick_urdf`）指向同一套几何与动力学。
  > ⚠️ `grasp_center` 站点**必须**落在两指内侧面之间（当前 z=0.048，指面中心 0.040）。
  > 它一度被放在指端（0.075），导致伺服把方块顶到两指之外、只剩指尖角接触：
  > 静态能"夹住"，一动就掉（抓取 16/20、搬运 0/10）。改成钳口中心后是 20/20 与 10/10。
- **观测（30 维）**：`[cos q(7), sin q(7), dq(7), 夹爪间隙, 夹爪速度, 检测目标(3), 相对位置(3), seen]`。
- **动作（8 维）**：7 个臂关节力矩 + 1 个夹爪开关（映射到 `[-0.045, 0]` 的位置伺服目标）。
- **奖励**：`-dist - 0.0005·Σqvel² + 0.5·exp(-dist/0.3)`，靠近后鼓励闭合，夹住给 `+2`。
- **复位**：随机摆放红块后调用 `place_distractors()`，把三块干扰物摆回固定位置
  （即使上一集被手臂撞飞也会复位），保证每个 episode 的初始状态可复现。
- 公开访问器 `cube_world() / grasp_world() / detected_world() / gripper_gap() / last_detected`，
  专家控制器与可视化脚本据此工作，不再直接读写私有字段。

#### 抓取脚本

- `grasp/detect.py`：红色分割（`red_mask`，阈值 `RED_MIN/RED_DOMINANCE` 只命中
  高饱和红色，绿/蓝/黄干扰块天然被排除）+ 针孔反投影 + 深度反投影（`estimate_cube_world_rgbd`），
  并提供 `render_rgbd()`（颜色/深度渲染顺序只写一次，避免把彩色图渲染坏）与
  `project_world()`（世界点 → 像素，`_ray_world` 的严格逆变换）。
  > `project_world()` 修了一个真实 bug：图像行向下增长、相机局部 `y` 向上，
  > 竖直像素应为 `cy - fy·y`；`grasp/overlay.py` 原先用的是 `+`，导致绿色
  > “3D 投影”标记上下镜像、偏 60 多像素。现由 `test_projection_matches_the_red_blob` 钉住。
- `grasp/demo.py`：检测 → 定位 → 伺服 → 抓取，输出成功率与定位误差。
- `grasp/pick_place_demo.py`：源台抓取 → 提起 → 搬运 → 放到目标台放置垫，
  输出成功率与落点误差，可选 `--video`（GIF）与 `--montage`（阶段拼图）。
- `grasp/montage.py` / `grasp/overlay.py` / `grasp/visualize.py` / `grasp/view.py`：
  静态拼图 / 检测可视化 / 回放 GIF / 实时窗口。
- `grasp/supervised.py` / `grasp/train_live.py`：在线 DAgger 监督训练（后者把训练过程录成 GIF）。
  `grasp/train_live.py` 默认 `--task pick_place`：采集整段取放、评估同时打印 **pick % / place %**，
  故事板按**阶段切换**取帧（接近/夹紧/提起/搬运/下降/松开），所以 GIF 里能看到完整的放置过程；
  `--task grasp` 回到原来的抓取任务。逐任务默认值（`TASK_DEFAULTS`）：取放 1 条/轮、
  400 次更新/轮、`--beta-min 0.5`、每轮至少采 2500 步（失败会提前结束 episode，
  靠这个下限保证数据量）；抓取 3 条/轮、50 次更新/轮、`--beta-min 0.2`。
  每轮评估后保留**最好**的检查点（这套物理很"刀尖"，轮间波动很大）。
- `grasp/train_live.py --live` 打开实时窗口，默认按 **1× 实时**播放（`--live-speed N` 调速：
  `0.5` 是慢放、`0` 为全速），采集阶段逐步 `sync()`、评估阶段通过
  `rollout_info(on_step=...)` 持续刷帧，所以窗口不会在轮次之间卡住；关窗只停止刷新，训练继续。
  仿真时长的物理下限是硬的（搬运被 0.3 N 夹紧力限制在 ~2 cm/s），所以"动作看起来太快"
  只能靠 `--live-speed` 放慢播放，而不是压低力矩（实测：`action_scale<1` 或速度上限
  会让方块夹不住，抓取率从 5/5 掉到 1/5）。取放一轮 ≈ 50 s，24 轮 ≈ 20 min；
  抓取任务默认 0.5× 慢放。

---

### 4.8 测试与依赖

- `tests/`：pytest 冒烟 + 回归测试，约 2 秒跑完：
  - `test_envs.py`：注册表完整性、**观测维度与顺序**（checkpoint ABI）、seed 可复现、渲染输出。
  - `test_paths_and_cli.py`：路径助手、默认值、`resolve_model` 的报错分支、`--env` 校验。
  - `test_grasp.py`：场景构建、红块检测精度、专家伺服确实能抓住、策略检查点往返。
  ```bash
  python3 -m pytest tests -q
  ```
- `requirements.txt`：`mujoco / gymnasium / stable-baselines3 / torch / numpy / scipy /
  matplotlib / imageio(-ffmpeg) / tensorboard / pytest` 依赖清单。
- `.gitignore`：忽略 `__pycache__`、`results/` 等生成物。

---

### 4.9 文档与产物

- `README.md`：快速上手（安装、训练、评估、查看器、结果表）。
- `PROJECT_GUIDE.md`：本文档。
- `results/`：
  - `ppo_<env>.zip`：训练好的策略权重。
  - `tb_<env>/`：TensorBoard 事件（训练曲线）。
  - `<env>/rollout.(mp4|gif)`：策略演示视频。
  - `<env>/learning_curve.png`：训练奖励曲线。
  - `<env>/montage.png`：逼近→命中拼图。
  - `logs/<env>_<n>.txt`：训练 stdout 日志（按环境/轮次命名）。

### 4.10 ROS 2 封装层 `ros2_ws/`

把整段七轴取放（`grasp.common` 的解析专家 + `grasp.detect` 的眼在手视觉）
封装成 ROS 2 节点，物理/视觉/控制律一行都不复制。

- `src/pro7_pick_place_interfaces/`：`PickPlaceStatus.msg`、`ResetScene.srv`、
  `PickPlace.action`（goal 里可给落点、方块初值、seed；feedback 给阶段与落点误差）。
- `src/pro7_pick_place_ros/project.py`：定位 `mujoco_arm_ppo` 仓库并把根目录放上 `sys.path`。
- `src/pro7_pick_place_ros/scene.py`：线程安全的 MuJoCo 场景（状态快照、TF 链、
  RGB-D 检测/渲染、放置点搬移）。渲染器是 OpenGL 对象，只在**创建它的线程**里用，
  误用会被 `_assert_renderer_thread()` 拦成异常而不是段错误。
- `src/pro7_pick_place_ros/simulator.py`：植物线程 + 唯一作业。逐控制步推进
  `grasp.common.iter_pick_and_place()`，负责相机渲染与场景复位；`sim_hz` 决定播放速度
  （`0` = 不限速）。
- `src/pro7_pick_place_ros/node.py`：话题（`status`/`joint_states`/`cube_pose`/
  `detected_cube`/相机/`markers`/`/tf`）、服务（`reset`）、Action（`pick_place`）。
- `src/pro7_pick_place_ros/client.py`：命令行客户端；`runtime.py`/`entry.py`：先选对解释器
  （ROS 的 rclpy 在系统 Python、MuJoCo 在 Conda）再导入 rclpy/mujoco。
- 详细接口表、参数表与设计说明见 `ros2_ws/README.md`。

```bash
cd /home/wj/mujoco_arm_ppo
./start.sh                          # 一键：构建（如需）+ 起全部节点（pick_place_node + rviz2）
./start.sh --demo --demo-seed 1     # 连 demo 客户端一起起：自动跑一次完整取放
./start.sh --stop                   # 停掉（含 rviz / 真机栈）

cd ros2_ws && ./run.sh              # 等价的工程内入口
./run.sh ros2 launch pro7_pick_place_ros pick_place.launch.py demo:=true demo_seed:=1
                                    # 连 demo 客户端一起起：自动跑一次完整取放
./run.sh test                       # 13 个用例：场景 / 专家一致性 / 作业 / Action 端到端
./run.sh ros2 run pro7_pick_place_ros pick_place_client --seed 1 --reset
```

---

## 5. 关键设计决策（为什么这么做）

| 决策 | 原因 |
|---|---|
| 奖励主项用 `-距离`（稠密） | 提供连续梯度，比“命中有奖励才学得快”；配合命中 `+2` 提供稀疏事件信号。 |
| 关节角用 `cos/sin` 进观测 | hinge 角无界，直接用角度会带来周期歧义。 |
| 目标用 `mocap` 无碰撞体 | 运动学控制目标、不参与动力学，避免“靶子被撞飞”。 |
| 到达任务里臂 geom 全部关闭接触 | 目标/台面之外没有任何接触力，纯惯性控制，PPO 更容易收敛。 |
| Pro7 到达设为 0-g | 有俯仰关节时重力增大难度；0-g 保持“纯惯性控制”，更易收敛。 |
| 所有入口用 `--env` 切换任务 | 复用同一套训练/评估/可视化代码，避免复制粘贴。 |
| 路径/默认值集中在 `paths.py` | 脚本里不出现绝对路径，移动仓库或换工作目录都不用改代码。 |
| 命令行集中在 `cli.py` | `--env/--model/--out` 语义与默认值永远一致，新脚本只写自己的参数。 |
| 到达任务收敛到 `env/base_reacher.py` | 观测/动作/奖励/渲染只实现一次，7 轴模型不会各自漂移。 |
| 抓取场景与专家收敛到 `grasp/common.py` | 台面/红块/夹爪尺寸与伺服增益只有一个来源；演示与环境的判据一致。 |
| 取放的"搬运"由环境执行，而不是交给策略 | 搬运力矩仅占量程 1~2%，比模仿网络的动作噪声还小，学出来的搬运必掉块；环境负责准静态搬运，策略负责接近/夹紧/重试/松手。 |
| 取放专家提供"逐控制步"生成器（`iter_pick_and_place`） | 阻塞版一次跑完几千步，外部驱动（ROS 节点、实时窗口）无法中途发布状态或取消；生成器让 `pick_and_place()`、节点、测试共用同一份控制律。 |
| ROS 节点把 MuJoCo 渲染器收在一根植物线程里 | `mujoco.Renderer` 是 OpenGL 对象，跨线程创建/销毁会直接段错误（实测：边渲染边重建场景必崩），所以复位与渲染都排队给植物线程执行。 |
| 观测维度与顺序被测试钉死 | 它们是 checkpoint 的 ABI，改坏了等于让已训练模型全部失效。 |

### 5.1 改动指引（维护者速查）

| 想改什么 | 改哪里 |
|---|---|
| 新增一个机械臂/任务 | 写一个 XML 放进 `assets/`，在 `env/` 加一个 `BaseReacher` 子类（只需 `MODEL_PATH` + `_reset_episode`），注册进 `env/__init__.py` |
| 调整到达任务的成功半径/步数 | 子类的 `DEFAULT_TARGET_RADIUS` / `DEFAULT_MAX_STEPS`（或构造函数参数） |
| 调整观测/奖励结构 | `env/base_reacher.py`；**必须**同步更新 `tests/test_envs.py` 与重新训练 |
| 调整工作台高度/红块大小/夹爪尺寸 | `grasp/common.py`（红块边长源自 `grasp.detect.DEFAULT_CUBE_SIDE`）+ 对应 XML |
| 调整专家伺服的快慢/力矩 | `grasp/common.py` 的 `KP/KD/FORCE_CLAMP/CLOSE_EPS`（演示、拼图、DAgger 老师同时生效） |
| 调整取放落点 / 目标台 | `grasp/common.py` 的 `PLACE_TARGET` / `TABLE_2_POS`（+ `assets/rokae_xmate_pro7_pick_real.xml` 里的 `table2` / `place_pad` 几何） |
| 调整搬运快慢 | `grasp/common.py` 的 `TRANSPORT_KP/KD/FORCE/WAYPOINT`（默认准静态；加快会让方块滑脱） |
| 取放任务的搬运计划 / 松手时机 | `grasp/common.py` 的 `PlacePlanner`（`PLACE_WAYPOINT`/`PLACE_TOL`/`PLACE_XY_TOL`） |
| 取放奖励 / 步数 / 观测 | `env/rokae_pro7_pick_place.py`（观测结构改动要同步 `grasp.policy.OBS_DIM_PLACE` 与 `tests`） |
| 取放训练速度 / 播放快慢 | `grasp/train_live.py` 的 `TASK_DEFAULTS`（每轮条数、更新次数、`live_speed`）|
| 改夹爪几何 / 抓取参考点 | `assets/rokae_xmate_pro7_pick_real.xml` 的 `finger_*_g` 与 `grasp_center`（参考点必须在两指内侧面之间） |
| 调整 PPO 超参 | `train_ppo.py` |
| 换机器 / 换目录 / 改默认产物位置 | `paths.py` |
| 新增命令行入口 | 复用 `cli.py` 的 `add_env_arg/add_model_arg/add_out_arg/apply_defaults` |

---

## 6. 常用命令速查

```bash
# 训练 Pro7 到达（默认环境 pro7_urdf）
python3 train_ppo.py --steps 2000000

# 训练时实时看 MuJoCo 窗口（关窗后训练继续）
python3 train_ppo.py --steps 2000000 --viewer
python3 grasp/train_live.py --rounds 24 --live          # 取放整段训练 + 实时窗口（默认任务）
python3 grasp/train_live.py --task grasp --rounds 24 --live   # 只训练抓取（0.5× 慢放）

# 断点续训
python3 train_ppo.py --steps 2000000 \
  --init-model results/ppo_pro7_urdf.zip

# 评估 + 曲线 + 视频
python3 eval_rollout.py --env pro7_urdf --episodes 400

# 实时 MuJoCo 窗口（关窗停止，T 换目标）
python3 viewer_demo.py --env pro7_urdf

# 逼近→命中拼图
python3 make_montage.py --env pro7_urdf

# 视觉抓取（训练 / 回放）
python3 train_ppo.py --env pro7_pick --steps 800000
```

---

## 7. 参考结果

| 指标 | `pro7_urdf`(Pro7 真实网格到达) |
|---|---|
| 整集成功率 | 84.2% |
| 终距 均值/中位 | 0.147 / 0.147 m |
| 平均到达步数 | ~61 (of 250) |
| 目标 | FK 采样（天然可达） |

> `pro7_urdf` 由 2.0M 步训练得到（`results/ppo_pro7_urdf.zip`，约 8.4 分钟 @ ~4000 步/秒）。
> 命中圈 0.15 m、单集上限 250 步；终距中位 0.147 m 说明多数回合是"刚好压线"命中，
> 想更稳可以续训或把命中半径略放宽。
