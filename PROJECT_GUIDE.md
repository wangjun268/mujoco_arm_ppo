# 项目架构与逐文件说明

> 用 **MuJoCo** 做物理仿真、**Gymnasium** 定义强化学习环境、**Stable-Baselines3 PPO** 训练，让机械臂末端达到随机目标点。当前共 7 个环境：平面二连杆（2D）、三自由度空间臂（3D）、**珞石 xMate ER3 六自由度臂**、**xMate Pro7 七自由度臂**（胶囊体版 / 真实 STL 网格版）、以及 Pro7 的**视觉抓取**场景（胶囊体版 / 真实网格版）。

> **末端执行器更新**：抓取场景的平行两指夹爪已替换为**灵心巧手 LinkerHand L20**（厂商 URDF +
> 22 个 STL 见 `assets/linkerhand_l20/`，由根目录 `convert_hand_urdf.py` 生成 MuJoCo 片段并标定
> 抓取姿态）。L20 的 21 个关节按开合协同驱动，`grasp_common.scene_ids()` 提供 `hand_*` 标识，
> `is_grasped()` 改为"至少 N 根手指接触方块"。下文中"夹爪 / 两指 / 钳口"的描述属于更换前的
> 历史实现。

---

## 1. 项目目标

这是一个“从物理建模到强化学习训练再到可视化演示”的完整闭环：

1. 用 MuJoCo XML 描述机械臂（关节、连杆、目标、驱动、传感器）。
2. 把模型包成标准的 `gymnasium.Env`（定义观测/动作/奖励/重置逻辑）。
3. 用 PPO 学一个策略，让末端在尽可能少的步数内到达随机目标。
4. 提供离线评估（指标 + 学习曲线 + 视频）和在线可视化（MuJoCo 实况窗口）。

三个环境共享同一套训练/评估/可视化流程，通过 `--env` 参数切换。

---

## 2. 总体架构（数据流）

```text
 ① 物理模型（XML）                    ① 物理模型（XML）
  assets/two_joint_arm.xml            assets/three_joint_arm.xml
  assets/rokae_xmate_er3.xml
         │                                      │
         │ mujoco.MjModel.from_xml_path()       │
         ▼                                      ▼
② 任务封装（Gymnasium.Env）
 env/two_joint_reacher.py            env/three_joint_reacher.py
 env/rokae_reacher.py
         │       ② 工厂/注册表
         │       env/__init__.py  →  make_env(name)
         ▼
 ③ 训练（Stable-Baselines3 PPO）
  train_ppo.py  ──►  results/ppo_<env>.zip（策略权重）
                            │
                            ├──► eval_rollout.py  指标 + learning_curve.png + rollout.mp4/gif
                            ├──► viewer_demo.py   实时 MuJoCo 窗口
                            ├──► make_montage.py  逼近→命中 静态拼图
                            └──► ik_probe*.py     解析 IK + PD 可行性基线（对比参照）
```

等价地，分层如下：

| 层次 | 文件 | 职责 |
|---|---|---|
| **物理层** | `assets/*.xml` | 定义刚体、关节、几何、驱动、传感器、相机、灯光、目标 |
| **环境基类** | `env/base_reacher.py` | 所有「到达」任务共享的观测/动作/奖励/渲染实现（只写一遍） |
| **环境层** | `env/*_reacher.py`、`env/rokae_pro7_pick.py` | 各自只定义模型、末端 site、起点/目标采样与抓取逻辑 |
| **工厂层** | `env/__init__.py` | 环境注册表与 `make_env()`，供所有脚本按名字取环境 |
| **RL 层** | `train_ppo.py` | PPO 策略网络、向量化环境、回调统计、断点续训 |
| **评估/可视化层** | `eval_rollout.py`、`viewer_demo.py`、`make_montage.py` | 加载策略做指标、视频、窗口、拼图 |
| **基准层** | `ik_probe*.py` | 手写解析逆运动学 + PD 力控，当作“人类最优”参照 |
| **抓取层** | `grasp_common.py`、`grasp_policy.py`、`detect_red_cube.py` | 抓取场景常量、专家伺服、视觉定位、策略网络（训练与回放共用） |
| **工具层** | `paths.py`、`cli.py` | 路径/默认值集中管理、统一的 `--env/--model/--out` 命令行与策略加载 |
| **测试层** | `tests/` | pytest 冒烟 + 回归测试（观测维度/顺序、路径、抓取契约） |
| **文档/产物层** | `README.md`、`PROJECT_GUIDE.md`、`results/` | 快速上手、本说明、训练产物 |

---

## 3. 目录结构

```text
mujoco_arm_ppo/
├── README.md                 # 快速上手：安装/训练/评估/查看器
├── PROJECT_GUIDE.md          # 本文档：架构 + 逐文件说明
├── PROJECT_SUMMARY.md        # 当前状态/结果/产物汇总
├── requirements.txt          # 依赖清单（mujoco / sb3 / torch / ...）
├── paths.py                  # 项目路径、workspace 默认值、模型路径校验
├── cli.py                    # 统一 --env/--model/--out 命令行与策略加载
├── live_viewer.py            # 节流的原生 MuJoCo 实时窗口（训练/演示共用）
├── train_ppo.py              # PPO 训练入口
├── eval_rollout.py           # 评估 + 学习曲线 + 演示视频
├── viewer_demo.py            # 实时 MuJoCo 可视化窗口
├── make_montage.py           # 逼近→命中 静态拼图
├── ik_probe.py               # 二连杆解析 IK + PD 基线
├── ik_probe3d.py             # 三自由度(3R)解析 IK + PD 基线
├── grasp_common.py           # 抓取场景常量 + 专家伺服（唯一来源）
├── grasp_policy.py           # 抓取策略网络（训练/回放共用）
├── detect_red_cube.py        # 红色分割 + 深度反投影 → 红块 3D 坐标
├── grasp_demo.py             # 检测 → 定位 → 伺服 → 抓取
├── pick_place_demo.py        # 源台抓取 → 提起 → 搬运 → 放到目标台放置垫
├── supervised_grasp.py       # 在线 DAgger / 行为克隆训练
├── train_live.py             # 取放训练过程可视化 GIF（默认 --task pick_place）
├── detect_overlay.py         # 检测可视化（分割 + 3D 对比）
├── visualize_grasp.py        # 抓取回放 GIF
├── view_pick.py              # 抓取/取放实时窗口（--task grasp|pick_place）
├── make_grasp_montage.py     # 逼近→抓住 拼图
├── convert_hand_urdf.py      # 厂商 L20 URDF → MuJoCo 片段（含抓取标定）
├── assets/
│   ├── two_joint_arm.xml     # 平面二连杆模型
│   ├── three_joint_arm.xml   # 3 自由度空间机械臂模型
│   ├── rokae_xmate_er3.xml   # 珞石 xMate ER3 六自由度臂（真实几何）
│   ├── rokae_xmate_pro7*.xml # Pro7 七自由度臂 / 抓取场景（含真实网格版）
│   ├── meshes/xMatePro7/*.stl # 官方 Pro7 STL 网格（随仓库提供，相对路径引用）
│   └── linkerhand_l20/       # 厂商 L20 URDF/STL + 生成的 MuJoCo 片段
├── env/                      # Python 包（仓库根目录在 sys.path 上即可导入）
│   ├── __init__.py           # make_env 注册表
│   ├── base_reacher.py       # 到达任务基类（观测/动作/奖励/渲染唯一实现）
│   ├── two_joint_reacher.py  # 二维到达任务环境（子类）
│   ├── three_joint_reacher.py# 三维到达任务环境（子类）
│   ├── rokae_reacher.py      # 珞石 ER3 / Pro7 到达环境（子类）
│   ├── rokae_pro7_pick.py    # Pro7 视觉抓取环境（obs 30）
│   └── rokae_pro7_pick_place.py # Pro7 取放整段环境（obs 38）
├── tests/                    # pytest 冒烟 + 回归测试
└── results/                  # 训练产物（模型/日志/图片/视频/TensorBoard）
    ├── ppo_<env>.zip         # 各环境策略权重
    ├── tb_<env>/             # TensorBoard 事件（训练曲线）
    ├── <env>/                # 该环境 视频/曲线/拼图
    └── logs/<env>_<n>.txt    # 训练 stdout 日志（按环境/轮次命名）
```

---

## 4. 逐文件说明

### 4.1 物理模型层 `assets/`

> **Pro7 真实网格**：`rokae_xmate_pro7_real.xml` / `rokae_xmate_pro7_pick_real.xml`
> 引用 `assets/meshes/xMatePro7/*.stl`（8 个官方 STL，约 7 MB，随仓库提供），
> 路径写法相对 XML 所在目录 → 不再依赖项目外的 `/home/wj/rokae_ros2`，
> 换机器/删掉外部仓库都能加载。改网格只需替换 `assets/meshes/` 下的文件。

#### `assets/two_joint_arm.xml` —— 平面二连杆机械臂

在 `xy` 平面内运动的两连杆臂，适合作为入门/验证任务。

- **关节**：`joint1`（肩，绕 `z` 轴 `hinge`）、`joint2`（肘，绕 `z` 轴 `hinge`）。两个关节轴都平行于世界 `z`，所以运动始终落在 `xy` 平面。
- **几何**：两节 `capsule`（红色上臂、蓝色前臂），肩/肘处用 `sphere` 做关节外观。
- **目标**：`target` 是 `mocap`（运动学控制）球体，`contype=0 / conaffinity=0` 表示**无碰撞、无质量**，从头到尾不会和手臂相撞。
- **驱动**：两个 `motor`（力矩），`gear=6.0 / 4.0`，`ctrlrange=-1..1`（仿真里力矩 = 控制量 × gear）。
- **传感器**：`jointpos / jointvel` 供观测使用。
- **相机**：`cam_xy`（俯视 xy 平面），供渲染/窗口/拼图使用。
- **地面**：`plane` 放在 `z=-0.12`，离肩/肘球体足够远。

> ⚠️ 这里的“地面必须在臂下方且留间隙”是关键：肩/肘球体如果嵌入地面会形成把基座锁死的接触约束，导致肩关节完全动不了（早期踩过的坑）。

#### `assets/three_joint_arm.xml` —— 3 自由度空间机械臂

三杆链式臂：**基座偏航 + 肩俯仰 + 肘俯仰**，末端在 3D 空间内运动。

- **关节**：`j1`（偏航，绕世界 `z`）、`j2`（肩俯仰，绕局部 `y`）、`j3`（肘俯仰，绕局部 `y`）。三关节组合出一个类球形的可达工作空间。
- **重力**：`gravity="0 0 0"`（**零重力再到达**）。因为存在俯仰关节，有重力会显著增加难度；设为 0-g 让任务与二连杆一样是“纯惯性控制”，更易学习（观测/动作空间维度却高很多）。
- **目标**：同样是 `mocap` 无碰撞球体。
- **驱动**：3 个 `motor`，`gear=6.0 / 6.0 / 4.0`。
- **相机**：`cam_iso`（等距视角）。它的 `xyaxes` 是“正视原点”的单位正交朝向——如果朝向偏了，渲染会是全黑画面。
- **地面**：`plane` 但 `contype=0`，纯视觉，永远不与臂碰撞。

> ⚠️ 两个关键坑：① 本模型 hinge 绕局部 `+y` 的转向与标准平面 IK 相反，解析 IK 需要把第二、三关节取反（`[q1,-q2,-q3]`）；② `cam_iso` 若未归一化/朝上偏会上调视锥导致黑屏。

#### `assets/rokae_xmate_er3.xml` —— 珞石 xMate ER3 六自由度工业臂

接近“真机”的模型：**关节轴、原点偏置、限位全部取自官方
`RokaeRobot/rokae_ros2` 仓库 `rokae_description/urdf/xMateER3.urdf.xacro`
的真实 URDF 参数**（几何原本在闭源 `libxMateModel.a` 里，本文件是它的开源复刻）。

| 关节 | 轴 | 相对父连杆原点 z | 限位 (rad) |
|---|---|---|---|
| 1 偏航 | (0,0,1) | 0 | ±2.967 |
| 2 肩俯仰 | (0,1,0) | 0.404 | ±2.094 |
| 3 肘俯仰 | (0,1,0) | 0.4375 | ±2.094 |
| 4 腕偏航 | (0,0,1) | 0.4125 | ±2.967 |
| 5 腕俯仰 | (0,1,0) | 0 | ±2.094 |
| 6 腕滚转 | (0,0,1) | 0.2755 | ±6.283 |

- **六关节全驱动**：`gear = [8, 8, 6, 3, 3, 2]`，`ctrlrange=[-1,1]`。
- **工具尖点偏离腕滚转轴**（局部 `(0, 0.10, 0.12)`）：这样 6 个关节都会影响
  末端位置。否则第 4、6 轴只改变朝向、不移动位置，在纯位置到达任务里会变成
  “无效自由度”、白给 PPO 增加噪声。
- **零重力**：真实 xMate 重载且多为水平段，重力会让 6 轴难以用 PPO 学会；
  设为 0-g 使其成为“0-g 到达”问题（与 3-DOF 一致），保证收敛。

---

### 4.2 环境层 `env/`

#### `env/__init__.py` —— 环境注册表与工厂

```python
_REGISTRY = {
    "two_joint": TwoJointReacher,
    "three_joint": ThreeJointReacher,
    "six_joint": RokaeReacher,
    "pro7_joint": RokaePro7Reacher,          # Pro7 7-DOF（胶囊体版）
    "pro7_urdf": RokaePro7RealReacher,       # Pro7 7-DOF（真实 URDF 网格）
    "pro7_pick": RokaePro7Pick,              # Pro7 + 夹爪 + 相机 + 红块
    "pro7_pick_urdf": RokaePro7PickReal,     # 同上，真实网格
    "pro7_pick_place": RokaePro7PickPlace,   # 取放整段（抓取 → 搬运 → 放置）
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
  （`3·n_dof + 2·pos_dim`，`pos_dim` 平面臂为 2、空间臂为 3）。
- **动作**：每关节 `[-1, 1]` 归一化力矩；`step` 里 `np.clip`。
- **奖励**：`-dist - ctrl_cost·Σctrl² - vel_cost·Σqvel² + shaping(dist) + 命中奖励`。
- **渲染**：惰性创建 `mujoco.Renderer`，相机取类属性 `CAMERA`。
- **公开访问器**：`env.target` / `env.tip`，脚本不再需要碰私有字段。

子类只需要提供三样东西：`MODEL_PATH`、`TIP_SITE`/`CAMERA`/`POS_DIM`，
以及 `_reset_episode()`（设置 `qpos/qvel` 并返回目标点）。需要“接近度塑形”
的六/七轴臂再额外覆盖 `_shaping()`。

> ⚠️ **观测布局就是 checkpoint 的 ABI**：维度与顺序一旦改变，已训练的
> `results/ppo_*.zip` 就会失效。平面二连杆历史上用的是**交错**顺序
> `[cos q1, sin q1, cos q2, sin q2, …]`，其余环境用的是**分块**顺序
> `[cos q1.., sin q1.., …]`；基类用 `INTERLEAVED_ANGLES` 明确标记这一差异，
> `tests/test_envs.py` 会把它钉死。

#### `env/two_joint_reacher.py` —— `TwoJointReacher(BaseReacher)`

把 `two_joint_arm.xml` 包装成标准 Gymnasium 环境。

- **观测（10 维，交错 cos/sin）**：`[cos q1, sin q1, cos q2, sin q2, dq1, dq2, tip−target(2), target(2)]`。
- **动作（2 维）**：`joint1/joint2` 的力矩。
- **重置**：随机初始关节角（±0.2），目标均匀采样在半径 `0.25~1.0` 的环带内（臂总长 1.05，保证可达）。
- **任务参数**：命中阈值 `target_radius=0.08`，时间上限 `max_steps=120`，相机 `cam_xy`。
- **info**：`dist_to_target / tip / target / success / steps`，供回调与评估使用。

#### `env/three_joint_reacher.py` —— `ThreeJointReacher(BaseReacher)`

与二维版本同构，差异在于：

- **观测（15 维，分块 cos/sin）**：`[cos q1..3, sin q1..3, dq1..3, tip−target(3), target(3)]`。
- **动作（3 维）**：三关节力矩。
- **目标采样**：球坐标——半径 `0.25~0.8`、极角 `0.05~1.0`、方位角 `[-π,π]`，保证落在可达圆锥内（总臂长 0.95）。
- **奖励/终止**：命中阈值 `0.10`，`max_steps=150`；其余逻辑一致。
- **渲染**：`cam_iso` 等距相机。

#### `env/rokae_reacher.py` —— `RokaeReacher(BaseReacher)`（ER3 / Pro7）

把 `rokae_xmate_er3.xml` 包装成六自由度到达任务；`RokaePro7Reacher` /
`RokaePro7RealReacher` 只改 `MODEL_PATH` 与采样/命中参数，就得到七自由度版。

- **观测（24 维）**：`[cos q1..6, sin q1..6, dq1..6, tip−target(3), target(3)]`。
- **动作（6 维）**：六关节力矩。
- **目标采样（关键设计）**：以随机“锚点姿态”为中心，起始与目标姿态都在其附近
  采样；目标由目标姿态的**正运动学 (FK)** 得到 → **天然可达**，无需解析逆解。
  初始距离均值约 0.5 m，适中、可学。
- **奖励**：`-dist − 0.0005·Σqvel² + 0.5·exp(−dist/0.3) + 2(命中)`。指数“接近度”
  塑形让远距离也有连续梯度，是六自由度能学会的关键。
- **命中阈值**：`target_radius=0.12`，时间上限 `max_steps=200`。

---

### 4.3 RL 训练层

#### `train_ppo.py` —— PPO 训练入口

- **CLI**：`--env`（默认 `two_joint`）、`--steps`、`--seed`、`--lr`、`--model`、`--init-model`、
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
  - `set_camera()` 依次尝试给定相机名（环境自带 `CAMERA` → `cam_iso` → `cam_xy`）；
  - `set_status([...])` 用 `Handle.set_texts()` 在窗口四角显示训练指标（最多 4 行）；
  - `close()` 关窗后 `sleep(0.6)`，等 UI 线程自己收完 GLFW，避免解释器退出时段错误；
  - `launch=` 可注入（测试用假句柄，**不需要显示器**）。
- `Pacer(speed, sim_dt)`：把循环限制到可观看的速率（`speed=1` 即 1× 实时，
  物理步长 0.02 s → 50 步/秒；`speed<=0` 表示不限速）。训练循环每步 `tick()`
  即可，落后时会自动重新对齐而不追赶。
- 被 `train_ppo.py`、`train_live.py`、`viewer_demo.py`、`view_pick.py` 共用，
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
- 把相机固定为模型自带的 `cam_xy` 或 `cam_iso`。
- **收尾**：`finally` 里 `viewer.close()` 后 `sleep(0.6)`，等 daemon UI 线程自行清理 GLFW，避免主线程与它并发 `glfw.terminate()` 导致的段错误。
- **CLI**：`--env/--model/--seed/--fps/--episodes`（`--episodes` 用于自动关闭，默认运行到关窗）。

#### `make_montage.py` —— “逼近→命中”静态拼图

- 加载策略，固定 seed 跑一个 episode，在若干指定时间步截取 `env.render()` 帧，排成网格保存为 `results/<env>/montage.png`，直观展示手臂从初始到命中目标的轨迹。

---

### 4.5 可行性基准层

#### `ik_probe.py` —— 二连杆解析 IK + PD 力控

- 二连杆标准逆运动学（`L1=0.55, L2=0.50`），解出肩/肘目标角。
- 用饱和 PD（`a = 6·(q*−q) − 1.2·q̇`）驱动，统计 200 集的初始/最终距离与成功率。
- 用途：作为“这个任务到底能不能做到”的**上界参考**。若 IK/PD 都做不好，通常是动力学/接触/奖励设计有问题。

#### `ik_probe3d.py` —— 3R 解析 IK + PD 力控

- 偏航 `q1=atan2(y,x)` + 垂直平面二连杆 IK，求肩/肘目标角；注意把 `q2/q3` 取反以匹配 MuJoCo 的 hinge 转向。
- 同样用 PD 力控跑 200 集，给出 3D 任务的可达性基线。

---

### 4.6 工具层

#### `paths.py` —— 项目路径 / 默认值 / 模型校验

- `PROJECT_ROOT / ASSETS_DIR / RESULTS_DIR`、`asset_path()`、`results_path()`：
  所有绝对路径都从这里来，**脚本里不再出现 `/home/wj/...`**，换机器/改目录名不用改代码，
  而且从任意工作目录运行都可以。
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

### 4.7 抓取层（相机 + 夹爪 + 红块）

#### `grasp_common.py` —— 抓取场景与专家的唯一来源

- **场景常量**：工作台高度 `TABLE_Z`、红块边长 `CUBE_SIDE`、`CUBE_X/CUBE_SPREAD`、
  初始位姿 `START_POSE`、夹爪行程 `GRIPPER_TRAVEL` / 张开间隙 `GRIP_OPEN`。
  `MODEL_PATH = assets/rokae_xmate_pro7_pick_real.xml`（**真实 URDF 网格**）与所有脚本都以这里为准
  （红块边长复用 `detect_red_cube.DEFAULT_CUBE_SIDE`，视觉与仿真不可能各说各话）。
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
- `detect_cube(model, data, renderer)`：眼在手 RGB-D → 红块世界坐标（内部调用 `detect_red_cube`）。
- `servo_command(...)`：**专家解析伺服**（分辨率控制 + 力限幅），返回
  `(各臂关节力矩, 是否该闭合夹爪, 当前距离)`；`teacher_action(env)` 把它包成策略用的 8 维动作。
  `grasp_demo`、`make_grasp_montage`、`supervised_grasp`（DAgger 老师）共用同一套控制律。
- `is_grasped(...)` / `gripper_gap(...)`：抓取成功判据（双指接触 + 间隙 + 中心距），
  环境与脚本共用，不会出现“演示说抓住了、环境说没抓住”。

#### `grasp_policy.py` —— 抓取策略网络

- `Policy`（128-128-Tanh MLP）、`OBS_DIM=30 / ACT_DIM=8`、`action()`、`load_policy()`、`rollout()`。
- 训练脚本与回放脚本都从这里取网络定义，检查点不会因为架构分散而加载不上。

#### `env/rokae_pro7_pick_place.py` —— `RokaePro7PickPlace(gym.Env)`

- `RokaePro7Pick` 的子类：episode **不在夹住时结束**，而是走完整段
  接近 → 夹紧 → 提起 → 搬运 → 下降到放置垫 → 松开，只有方块落到垫上（或步数用尽）才结束。
  抓取失败、方块被撞出台面时提前以 `info["lost"]=True` 结束（否则这种不可恢复状态会
  灌满 DAgger 缓冲区，把在线训练带跑偏——实测过）。
- **观测（38 维）** = 抓取环境的 30 维 + `holding` + 子目标误差 `goal-grasp`(3) +
  方块到垫子的误差 `pad-cube`(3) + `plan.done`（到位该松手的标志）。
- **搬运计划** `grasp_common.PlacePlanner`：`settle(15 步) → lift → carry → lower`，
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

- `detect_red_cube.py`：红色分割（`red_mask`，阈值 `RED_MIN/RED_DOMINANCE` 只命中
  高饱和红色，绿/蓝/黄干扰块天然被排除）+ 针孔反投影 + 深度反投影（`estimate_cube_world_rgbd`），
  并提供 `render_rgbd()`（颜色/深度渲染顺序只写一次，避免把彩色图渲染坏）与
  `project_world()`（世界点 → 像素，`_ray_world` 的严格逆变换）。
  > `project_world()` 修了一个真实 bug：图像行向下增长、相机局部 `y` 向上，
  > 竖直像素应为 `cy - fy·y`；`detect_overlay.py` 原先用的是 `+`，导致绿色
  > “3D 投影”标记上下镜像、偏 60 多像素。现由 `test_projection_matches_the_red_blob` 钉住。
- `grasp_demo.py`：检测 → 定位 → 伺服 → 抓取，输出成功率与定位误差。
- `pick_place_demo.py`：源台抓取 → 提起 → 搬运 → 放到目标台放置垫，
  输出成功率与落点误差，可选 `--video`（GIF）与 `--montage`（阶段拼图）。
- `make_grasp_montage.py` / `detect_overlay.py` / `visualize_grasp.py` / `view_pick.py`：
  静态拼图 / 检测可视化 / 回放 GIF / 实时窗口。
- `supervised_grasp.py` / `train_live.py`：在线 DAgger 监督训练（后者把训练过程录成 GIF）。
  `train_live.py` 默认 `--task pick_place`：采集整段取放、评估同时打印 **pick % / place %**，
  故事板按**阶段切换**取帧（接近/夹紧/提起/搬运/下降/松开），所以 GIF 里能看到完整的放置过程；
  `--task grasp` 回到原来的抓取任务。逐任务默认值（`TASK_DEFAULTS`）：取放 1 条/轮、
  400 次更新/轮、`--beta-min 0.5`、每轮至少采 2500 步（失败会提前结束 episode，
  靠这个下限保证数据量）；抓取 3 条/轮、50 次更新/轮、`--beta-min 0.2`。
  每轮评估后保留**最好**的检查点（这套物理很"刀尖"，轮间波动很大）。
- `train_live.py --live` 打开实时窗口，默认按 **1× 实时**播放（`--live-speed N` 调速：
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

---

## 5. 关键设计决策（为什么这么做）

| 决策 | 原因 |
|---|---|
| 奖励主项用 `-距离`（稠密） | 提供连续梯度，比“命中有奖励才学得快”；配合命中 `+2` 提供稀疏事件信号。 |
| 关节角用 `cos/sin` 进观测 | hinge 角无界，直接用角度会带来周期歧义。 |
| 目标用 `mocap` 无碰撞体 | 运动学控制目标、不参与动力学，避免“靶子被撞飞”。 |
| 物理上保证臂不与任何物体接触 | 曾因肩/肘球体嵌入地面把基座锁死，导致肩关节不动、PPO/IK 都学不动。 |
| 3D 臂设为 0-g | 有俯仰关节时重力增大难度；0-g 保持“纯惯性控制”，与 2D 任务性质一致、更易收敛。 |
| 提供解析 IK/PD 基线 | 用来区分“算法没学好”还是“任务本身不可行/环境有病”。 |
| 所有入口用 `--env` 切换任务 | 复用同一套训练/评估/可视化代码，避免复制粘贴。 |
| 路径/默认值集中在 `paths.py` | 脚本里不出现绝对路径，移动仓库或换工作目录都不用改代码。 |
| 命令行集中在 `cli.py` | `--env/--model/--out` 语义与默认值永远一致，新脚本只写自己的参数。 |
| 到达任务收敛到 `env/base_reacher.py` | 观测/动作/奖励/渲染只实现一次，2/3/6/7 轴不会各自漂移。 |
| 抓取场景与专家收敛到 `grasp_common.py` | 台面/红块/夹爪尺寸与伺服增益只有一个来源；演示与环境的判据一致。 |
| 取放的"搬运"由环境执行，而不是交给策略 | 搬运力矩仅占量程 1~2%，比模仿网络的动作噪声还小，学出来的搬运必掉块；环境负责准静态搬运，策略负责接近/夹紧/重试/松手。 |
| 观测维度与顺序被测试钉死 | 它们是 checkpoint 的 ABI，改坏了等于让已训练模型全部失效。 |

### 5.1 改动指引（维护者速查）

| 想改什么 | 改哪里 |
|---|---|
| 新增一个机械臂/任务 | 写一个 XML 放进 `assets/`，在 `env/` 加一个 `BaseReacher` 子类（只需 `MODEL_PATH` + `_reset_episode`），注册进 `env/__init__.py` |
| 调整到达任务的成功半径/步数 | 子类的 `DEFAULT_TARGET_RADIUS` / `DEFAULT_MAX_STEPS`（或构造函数参数） |
| 调整观测/奖励结构 | `env/base_reacher.py`；**必须**同步更新 `tests/test_envs.py` 与重新训练 |
| 调整工作台高度/红块大小/夹爪尺寸 | `grasp_common.py`（红块边长源自 `detect_red_cube.DEFAULT_CUBE_SIDE`）+ 对应 XML |
| 调整专家伺服的快慢/力矩 | `grasp_common.py` 的 `KP/KD/FORCE_CLAMP/CLOSE_EPS`（演示、拼图、DAgger 老师同时生效） |
| 调整取放落点 / 目标台 | `grasp_common.py` 的 `PLACE_TARGET` / `TABLE_2_POS`（+ `assets/rokae_xmate_pro7_pick_real.xml` 里的 `table2` / `place_pad` 几何） |
| 调整搬运快慢 | `grasp_common.py` 的 `TRANSPORT_KP/KD/FORCE/WAYPOINT`（默认准静态；加快会让方块滑脱） |
| 取放任务的搬运计划 / 松手时机 | `grasp_common.py` 的 `PlacePlanner`（`PLACE_WAYPOINT`/`PLACE_TOL`/`PLACE_XY_TOL`） |
| 取放奖励 / 步数 / 观测 | `env/rokae_pro7_pick_place.py`（观测结构改动要同步 `grasp_policy.OBS_DIM_PLACE` 与 `tests`） |
| 取放训练速度 / 播放快慢 | `train_live.py` 的 `TASK_DEFAULTS`（每轮条数、更新次数、`live_speed`）|
| 改夹爪几何 / 抓取参考点 | `assets/rokae_xmate_pro7_pick_real.xml` 的 `finger_*_g` 与 `grasp_center`（参考点必须在两指内侧面之间） |
| 调整 PPO 超参 | `train_ppo.py` |
| 换机器 / 换目录 / 改默认产物位置 | `paths.py` |
| 新增命令行入口 | 复用 `cli.py` 的 `add_env_arg/add_model_arg/add_out_arg/apply_defaults` |

---

## 6. 常用命令速查

```bash
# 训练二连杆 / 三自由度臂
python3 train_ppo.py                              # toy 二连杆，200k 步
python3 train_ppo.py --env three_joint --steps 600000

# 训练时实时看 MuJoCo 窗口（关窗后训练继续）
python3 train_ppo.py --env three_joint --steps 600000 --viewer
python3 train_live.py --rounds 24 --live          # 取放整段训练 + 实时窗口（默认任务）
python3 train_live.py --task grasp --rounds 24 --live   # 只训练抓取（0.5× 慢放）

# 断点续训
python3 train_ppo.py --env three_joint --steps 600000 \
  --init-model results/ppo_three_joint.zip

# 评估 + 曲线 + 视频
python3 eval_rollout.py --env three_joint --episodes 400

# 实时 MuJoCo 窗口（关窗停止，T 换目标）
python3 viewer_demo.py --env three_joint

# 逼近→命中拼图
python3 make_montage.py --env three_joint

# 可行性基线
python3 ik_probe.py
python3 ik_probe3d.py

# 珞石六自由度臂
python3 train_ppo.py --env six_joint --steps 1000000
python3 eval_rollout.py --env six_joint --episodes 300
python3 viewer_demo.py --env six_joint
```

---

## 7. 参考结果

| 指标 | `two_joint`(2D) | `three_joint`(3D) | `six_joint`(ER3) | `pro7_joint`(Pro7) | `pro7_urdf`(Pro7 真实网格) |
|---|---|---|---|---|---|
| 整集成功率 | 100% | 98.5% | 48% | ~89.5% | 84.2% |
| 终距 均值/中位 | 0.056 / 0.057 | 0.092 / 0.091 | 0.177 / 0.129 | 0.163 / 0.147 | 0.147 / 0.147 |
| 平均到达步数 | ~27 (of 120) | ~31 (of 150) | ~130 (of 200) | ~61 (of 250) | ~61 (of 250) |
| 基线 | 100% (IK/PD) | 100% (IK/PD) | FK 采样（天然可达） | FK 采样 | FK 采样 |

> `pro7_urdf` 由 2.0M 步训练得到（`results/ppo_pro7_urdf.zip`，约 8.4 分钟 @ ~4000 步/秒）。
> 胶囊版策略可零样本迁移到真实网格模型（关节链一致），实测 90% 命中率。

> 六自由度明显更难：策略能把末端从约 0.5 m 压到约 0.13 m，并稳定进入 12 cm
> 命中圈约一半时间。这是 PPO 从零学习六轴协调的真实水平；把命中半径放宽到
> 0.15 m，成功率会显著上升。
