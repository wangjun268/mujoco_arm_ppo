# take_box · MuJoCo 双臂训练栈

`take_box` 工作台的 **MuJoCo 训练环境 + PPO 训练/评估/可视化脚本**，独立成一个自包含的包：
放在 `/home/wj/mujoco_arm_ppo/take_box_rl/`（和 Pro7 + L20 那套 `mujoco_arm_ppo` 同一个仓库，
但**互不依赖**：本包自带模型生成、环境、流程和测试，只依赖机器人描述文件）。
任务语义对齐 `src/robot_control/config/workspaces/take_box/task_engine/` 里的 YAML：

```
prepare_action（两臂到箱子两侧） → take_box_step1（抬起） → wait（保持） → take_box_step2（放下）
```

RL 版本把它压成一句话：**两只手一起把一个箱子从台面上端起来，并在空中保持水平**。

```
take_box_rl/
├── SUMMARY.md                # 维护总结：做了什么 / 还有什么问题 / 后面怎么改进
├── cell_config.json          # 工位几何 + 训练参数（改这里就能对齐真实工位）
├── build_cell_model.py       # 生成 MuJoCo 模型 + IK 求解 pre-grasp 姿态
├── take_box_env/
│   ├── __init__.py           # 注册表：take_box / take_box_reach
│   ├── take_box.py           # 环境实现（观测/动作/阶段/几何）
│   ├── rewards.py            # 奖励函数（势函数塑形 + 事件奖励），系数在 cell_config.json
│   └── box_api.py            # 不训练的运动规划接口（IK + PD + 约束）
├── train_ppo.py              # PPO 训练入口（SB3）
├── eval_rollout.py           # 成功率/抬升高度/倾斜角 + 可选视频
├── viewer_demo.py            # 实时窗口跑策略
├── take_box_flows/           # 搬运流程，一个阶段一个文件（见下）
│   ├── prepare.py            #   1. 回到预抓取位
│   ├── grasp.py              #   2. 闭手抓住
│   ├── lift.py               #   3. 抬起
│   ├── carry.py              #   4. 向前搬运
│   ├── place.py              #   5. 放回台面
│   ├── release.py            #   6. 松手
│   ├── full_task.py          #   把 1-6 串成完整流程
│   └── common.py             #   建环境/接口、打印、录像等公共小工具
├── tests/                    # 回归测试：环境/几何 17 + box_api 7 + flows 5 + 奖励 9 项
├── assets/                   # 生成物：take_box_cell.xml / take_box_ready.json / meshes/
├── runs/                     # 训练产物：ppo_*.zip 权重、TensorBoard、录像
├── requirements.txt          # 依赖
└── pyproject.toml            # 包元数据（可选 pip install -e .）
```

## 快速开始

```bash
cd /home/wj/mujoco_arm_ppo/take_box_rl
python3 build_cell_model.py --check          # 校验已有的模型资产
python3 build_cell_model.py                  # 重新生成（改过 cell_config.json 后跑）

python3 -m pytest tests/ -q                  # 30 项回归测试

# 课程 1：两臂把掌心送到箱子两侧抓取面（初始偏差 ~0.2 m）
python3 train_ppo.py --env take_box_reach --steps 400000 --n-envs 4
# 课程 2：抓住并端平抬起（初始就在抓取位，对应真实流程的 prepare 之后）
python3 train_ppo.py --env take_box --steps 1200000 --n-envs 8

python3 eval_rollout.py --env take_box --episodes 100 --video
python3 viewer_demo.py --env take_box --episodes 5

# 完整搬运流程（不用训练）：prepare → grasp → lift → 向前搬运 → place → release
python3 -m take_box_flows.full_task                   # 各阶段独立文件也能单独跑，如 python3 -m take_box_flows.lift
python3 -m take_box_flows.full_task --video           # 同时存 runs/take_box/full_task.gif
```

依赖：`mujoco`、`gymnasium`、`stable-baselines3`、`torch`、`imageio`（与 `mujoco_arm_ppo` 同一套）。

## 模型是怎么来的

**当前模型：lkwy73_o1 双臂本体**（`/home/wj/urdf/lkwy73_o1_dual_arm_clean`，2×7 自由度臂 +
2 只 11 关节手，11.11 kg）。`cell_config.json → robot.source = "dual_arm_mjcf"` 就是这条路径；
改成 `"single_arm_urdf"` 会回到"单臂描述实例化两次"的 ROKAE 方案（配置在 `legacy_arm` 里）。

`build_cell_model.py` 不手写 XML，而是：

1. **读入机器人描述**：`dual_arm_mjcf` 模式直接复用清洗好的 lkwy73 双臂 MJCF（含网格），
   整台本体按 `robot.pos` 挂到工位上（底座法兰离地 0.95 m，双臂自然下垂后手在 z≈0.2 m）。
   另一条路径（`single_arm_urdf`）读取 `legacy_arm.urdf`。本机 `rokae_ros2/rokae_description`
   里只有 **Pro3 / Pro7 是 7 轴**（`xMateER7` 那份实际只有 6 轴），所以默认用 **xMate Pro7**
   的真实网格与惯量；拿到 ER Pro 7 轴的官方 URDF 后改 `arm.urdf` 即可，脚本对本机 xacro
   模板做了内联展开（只需处理 `xacro:property` / `xacro:include` / `$(find ...)`）。
2. **（单臂路径）实例化两次**：把 MuJoCo 从 URDF 导出的单臂树复制成 `left_*` / `right_*`，按
   `cell.arms` 的 `pos/euler_deg` 摆到工位上（默认相对而立、间距 0.92 m）。
3. **手**：lkwy73 自带两只 11 关节手（`lh_*` / `rh_*`），环境用**每手 1 个抓取标量**驱动
   这 11 个关节的协同（与真机 O6 的 6 通道语义一致：拇弯/拇摆/食/中/无名/小）。
   标量按**每个关节自己的行程**展开：`grip = 1` 是"每个通道到自己限位"，不是"每个通道都
   收到 1.2 rad"（拇指俯仰只有 0.58 rad，会被顶在限位上持续出力）。单臂路径下才会用参数化
   O6 代理手。
4. **加台面、箱子和相机**；箱子是 `mocap` 刚体，由环境驱动。
5. **IK 求解 pre-grasp 姿态**：让两只**掌面**贴在箱子两侧面上（grasp site 是手掌内部的一个
   虚拟点，脚本会一边解 IK 一边量"这只手到箱子的最近距离"，把目标沿掌法线挪到留 2 mm 空隙；
   直接拿 site 对箱面会让整只手插进箱子 3 cm），结果写进
   `assets/take_box_ready.json`，环境复位时从这里出发。目标姿态 = 掌心法线朝箱面、再绕掌法线
   滚转 `task.grasp_roll_deg`（默认 30°，手指朝前下方）；**两只手用同一个正常旋转目标帧**
   （镜像的 CAD + 会反号的叉乘，正好抵消），双脚的 IK 残差都是 0.014（≈1°）。滚转角不是随心
   选的：这个工位用"手指朝前"的滚转时，两只手都只能做到差 21°，而且唯一够得到的姿态是
   手背贴箱面——详见下面"踩过的坑"第 6 条。

生成物规模：36 个自由度（2×7 臂 + 2×11 手）、36 个执行器（14 个臂力矩电机 + 22 个手指
位置伺服）、29 个网格、整机 12.11 kg（机器人 11.11 + 箱子 1.0）、零位无接触。
双臂的 4 对 CAD 固有自重叠（`L1/base`、`L5/L7` 及镜像）已用 `<contact><exclude>` 排除。

## 环境

| 项 | 值 |
|---|---|
| 动作（16） | 14 个臂关节归一化力矩 `[-1,1]` + 左右各 1 个抓取通道（按各手指自己的行程展开到 11 个手关节） |
| 观测（55） | 每臂 `[cos q(7), sin q(7), dq(7), 掌心到箱面误差(3)]` = 24 → 48，再加箱体偏移(3)、箱体倾角(1)、是否抓住(1)、双手抓取通道(2) |
| 阶段 | `approach → grasp → lift → hold` |
| 抓取 | 两手掌心都在箱面 `grasp_tol` 内、且两个抓取通道都 > 0.5 → 锁存抓取，并**记住此刻箱子在掌心坐标系里的相对位姿**；之后箱子就按这个固定偏移跟着双手做刚体运动（不瞬移、不翻转）。`allow_release=True` 时抓取通道 < 0.2 会**脱手**（默认关闭，见下面"现状与限制"） |
| 奖励 | 在 **`take_box_env/rewards.py`** 里单独实现（`TakeBoxReward`）：**势函数塑形** `0.99·Φ(s') - Φ(s)`，`Φ = -(平均掌心误差 + height_weight·箱高误差 + tilt_cost·箱体倾角)`（只依赖状态，不随抓取标志切换）；首次抓住 `+grasp_bonus`（每回合只发一次）；保持住每步 `+hold_reward`（最多 `hold_steps` 步），达标 `+success_reward` 并结束；另有 `-ctrl_cost·Σctrl²`。系数在 `cell_config.json → task.reward` |
| 复位 | lift 任务从 pre-grasp 姿态附近（噪声 0.02 rad）出发；reach 课程从更远处出发（0.15 rad），否则第一步就满足阈值 |
| 时间上限 | 260 步 × 0.02 s = 5.2 s |

两个注册环境：

| 环境 | 用途 |
|---|---|
| `take_box_reach` | 课程 1：只练"两臂把掌心贴到箱面"，作为热启动 |
| `take_box` | 完整任务：抓住、抬起、端平并保持 |

## 用程序规划搬箱动作（`take_box_env/box_api.py`，不训练）

训练之外还有一条路：直接用**关节 PD 伺服 + 笛卡尔 IK**写运动程序，自己定约束。
接口和 Gym 环境共用同一个模型，所以规划出来的动作可以直接对照 YAML 任务执行。
**具体流程按阶段拆在 `flows/` 下，一个文件一段**（`prepare / grasp / lift / carry / place /
release`，`flows/full_task.py` 把它们串起来），下面这段是同一件事的"一页速查":

```python
from take_box_env.take_box import TakeBoxEnv
from take_box_env.box_api import BoxCarryInterface, Waypoint

env = TakeBoxEnv()
robot = BoxCarryInterface(env, strict=True)      # strict：越界抛 ConstraintViolation
robot.limits.tilt_max_deg = 15.0                 # 改约束

robot.home()                                     # prepare：回到 IK 解好的抓取位
robot.close_hands()                              # 闭手 → 掌心到位即锁存"抓住"
base = robot.ready_box_pos
robot.goto_box_pose(base + (0, 0, 0.12))                     # 抬起
robot.goto_box_pose(base + (0.09, 0.06, 0.12), yaw=0.10)     # 平移 + 小幅偏航
robot.goto_box_pose(base + (0.09, 0.06, 0.0))                # 放下
robot.open_hands()                                           # 松手
```

也可以一次性交一串路点让它规划并记录：

```python
trace = robot.plan([
    Waypoint(box=base + (0, 0, 0.12), steps=80, label="lift"),
    Waypoint(box=base + (0.09, 0.06, 0.12), yaw=0.10, steps=90, label="carry"),
])
```

接口清单：

| 成员 | 作用 |
|---|---|
| `home(steps)` | 回到 pre-grasp（IK 解好的 `take_box_ready.json`） |
| `close_hands(steps)` / `open_hands(steps)` | 闭手（自动锁存抓取）/ 松手 |
| `goto_box_pose(pos, yaw, steps, grip, settle_steps)` | 笛卡尔搬箱：按箱位姿反解两侧掌心位姿 → IK → PD 跟踪 + 收敛段 |
| `servo_step(q_target, grip)` / `step` | 单步原语，自己写规划器时用 |
| `state()` → `BoxState` | 箱位姿/偏航/倾角/抬升/掌心误差/抓取状态/关节速度（`.summary()` 一行打印） |
| `plan(waypoints, on_step)` | 路点序列规划 + 执行，返回逐步 trace |
| `violations()` / `check_command()` / `violation_log` | 约束检查：倾角、高度、掌心贴合、关节速度、箱速；`strict=True` 直接抛异常 |
| `ik(side, pos, rot, seed)` | 单臂 6 自由度 IK（先位置后姿态 + 关节正则），可单独拿来做自己的规划 |

工程上踩过的三个坑都写在 `box_api.py` 的注释里：

1. **IK 必须还原仿真状态**：`ik()` 在 `data.qpos` 上试算，若不复原，第一次规划就会把机器人瞬移到求解器最后一步的姿态（表现为 6 m/s 的速度尖峰）。
2. **IK 要加关节正则**：否则改一点点 yaw，腕部就可能跳到翻转解，箱子朝向瞬间差 180°。
3. **抓取瞬间要跳过那一步的箱速检查**——箱子在锁存那一步被记进掌心坐标系，检查器会把它读成一次速度尖峰。

精度现状（用 `python3 -m take_box_flows.full_task` 实测）：抬起 +91 mm（指令 120 mm）、掌心贴合误差
≤1.9 mm、**抬升/搬运段倾角 ≤0.2°**（放置段 3.9°）、约束记录 **0** 条，箱心
(0.420, 0, 0.550) → (0.505, 0, 0.578)。抬起高度只到指令的 ~75% 是腕关节限位造成的
位置/姿态不可兼得（见下），流程里打印的是**实际**抬升量。此前记录的"误差 35 mm／倾斜漂移
17–22°"是老问题叠加的结果：掌心坐标系接错、箱子被吸附到掌心坐标系、抓取目标姿态让手背贴箱面，
以及抓取 site 是个悬在手掌内部的虚拟点。

## 与 ROS 侧的对接

| ROS YAML 里的东西 | 这里的对应物 |
|---|---|
| `move_cart` + `pose: part_A/part_B`（当前工作台的 `target_arm_pose.json` 还是空的 `{}`） | pre-grasp 姿态由 IK 从箱子位姿反解（`take_box_ready.json`），箱子位姿在 `cell_config.json` 里；真实工位测出来后改这两处即可 |
| `target_hand_pose.json` 的 16 通道（O6 只用前 6） | 环境把每只手的 1 个标量展开成 6 个通道，顺序与 O6 一致 |
| `default_speed` / `velocity_ratio` | 训练里由力矩上限 + 关节阻尼隐式决定；部署时建议把策略输出转成关节速度或笛卡尔增量，再交给现有的 `LuoshiArmController` |
| 抬起高度（YAML 里没有，靠 `pose` 差值） | `cell_config.json → task.lift_height`（默认 0.18 m） |

策略是**关节力矩**层面的，部署到真机有两条路：
1. 直接用 SDK 的力矩/阻抗接口（`LuoshiArmController` 里有 RT 与力控路径）；
2. 更稳妥的是把策略蒸馏成"关节角增量 + 关节速度"，走现有的 `move_joint`/`move_cart` 接口。

## 现状与限制（重要）

### 训练结果（2026-09-22 复核，确定性 100 回合）

| 环境 / 权重 | 训练步数 | 成功率 | 抓取率 | 抬起高度（中位） | 箱体倾角（中位） | 回合步数 |
|---|---|---|---|---|---|---|
| `take_box`（**2026-09-22 重训并替换进 `runs/`**） | 0.4M | **100%** | 100% | 0.103 m | 6.2° | 16 |
| `take_box`（2026-09-18 旧权重，备份 `runs/ppo_take_box.zip.bak-20260922`） | 1.5M | 43% | 99% | 0.081 m | 14.3° | 163 |
| `take_box_reach`（**2026-09-22 重训并替换进 `runs/`**） | 1.2M | **100%** | — | — | — | 22 |

训练速度约 2500 步/秒（8 个并行环境，CPU）。TensorBoard 事件在 `runs/tb_<env>/`。

> **两个环境的权重都已在 2026-09-22 重训替换**，旧文件备份成 `runs/*.zip.bak-20260922`
> （旧权重在现在的环境里只有 43% / 3% 成功率，而它在掌心坐标系、手指行程上有偏置，
> `viewer_demo.py` 里就会看到箱子一接触就被甩起来）。复现命令：
> `python3 train_ppo.py --env take_box --steps 400000 --n-envs 8`（约 3 分钟）、
> `python3 train_ppo.py --env take_box_reach --steps 1200000 --n-envs 8`（约 7 分钟）。

> `height_tol` 取 0.03 m 是**任务规格**，不是训练出来的指标。旧策略在 0.02 m 下只能连续满足
> 7–8 步（需要 10 步），所以当初放宽到 0.03；复核时的重训策略在 0.02 m 下仍是 100%
> （0.01 m 掉到 54%）。想让误差更小，就得加长训练或让保持判定更细（例如按"误差随时间的
> 积分"给分）。

* **臂型假设**：默认用 xMate Pro7 的真实网格代替 ER Pro 7 轴。两者同为珞石 7 轴、关节序相同
  （z/y/z/y/z/y/z），但连杆长度与惯量有差别（Pro7 单臂约 19 kg）。拿到 ER Pro 7 轴的描述后
  重跑 `build_cell_model.py` 即可，训练超参不用改。
* **O6 没有 CAD**：手是参数化代理件，几何、质量、摩擦都不是真值；抓取判定用的是"掌心到箱面距离"，
  不是接触力。
* **无力学接触**：臂与台面、箱子全部 `contype=0`，箱子由"双手刚性frame"运载（和
  `mujoco_arm_ppo` 抓取项目里的 quasi-static carry 约定一致）。所以这个模型能学**协同与轨迹**，
  不能用来评估夹持力、滑动和碰撞。
* **脱手默认关闭**：一旦抓住，箱子就一直跟着双手，抓取通道再压到多低都不会掉（`box_api`
  的 `open_hands()` 是显式置位，不受影响）。原因是"保持闭手"没有任何直接回报——势函数不依赖
  抓取通道，收益要 40 步之后才以"还能不能抬"的形式体现——实测 40 万步：带脱手 0%、不带脱手
  77%。做放下/松手阶段时用 `TakeBoxEnv(allow_release=True)`，并预留更多训练步数。
* **只做到"抬起并保持"**：YAML 的 `take_box_step2`（放下）还没做，加一段"降回台面 + 松手"的
  阶段即可（`_compute_reward` 里再切一个 stage）。
* **手腕滚转范围决定了抓取姿态**：两根腕关节（`R6`/`R7`、`L6`/`L7`，±1.22 rad）只有 ±70°，
  所以"掌心法线朝箱面"能实现，但绕掌法线的滚转只能在 φ≈+30°…+150° 里选（`task.grasp_roll_deg`
  默认取最靠前的 +30°）。想换成"手指正朝前"的抓取姿态，得换箱位/臂位或放宽腕关节限位
  （`cell_config.json` 里的关节范围来自 lkwy73 描述，不是本项目加的）。
* **扭矩上限是缩放值**：真机是 217/217/102/66/66/66/66 Nm，训练里按 20 ms 步长的数值稳定性缩到
  18/18/12/12/8/5/5（同系列 Pro7 到达模型用 10 Nm）。数值稳定性和物理真实性之间的取舍写在这里，改大之前先看 `cell_config.json` 的注释。

## 踩过的坑

1. `mj_jacSite` 必须在 `mj_fwdPosition`（或 `mj_forward`）之后调用；只调 `mj_kinematics`
   时雅可比全是 0，IK 会"一动不动"（`build_cell_model.py` 里有注释）。
2. 步长和扭矩必须一起定：最初用 0.002 s + 60 Nm，260 步只有 0.5 s，机械臂来不及动；
   改成 0.02 s 后 60 Nm 又会数值发散（`QACC` 爆炸、训练里随机动作把臂甩飞），最终取
   **0.02 s + 18 Nm + damping 2.0 + armature 0.05**，压力测试 3×500 步无发散、无不稳定警告。
3. 复位噪声要按阶段区分：lift 任务如果一开始就在容差外，策略学到的是"先靠近"，而真实流程
   是先 `prepare_action` 再抬；reach 课程则必须从容差外开始，否则第一步就"成功"。
4. **奖励塑形绝不能用"贴住就给正分"**：第一版用 `0.5·exp(-误差/0.15)` 做靠近奖励，训练出的
   确定性策略把掌心停在离箱面 3 mm 处、却**故意不闭手**——因为不抓就能一直吃 +0.48/步 × 260 步，
   而抓起来之后只有负的高度/倾角项。改成势函数塑形后，"静止"回报为 0，"抓住但不抬"为 −0.17，
   "抓住并抬起保持"才有正收益，激励才和任务目标一致。
5. **掌心坐标系不能拿腕部坐标系凑**：`left_grasp`/`right_grasp` 是只有位置的 site，`site_xmat`
   给的是腕部连杆的姿态，而 lkwy73 的手是斜着装到法兰上的，两者差 15–20°。用腕部 y 轴当
   "掌心朝上轴"，箱子在预抓取位姿就被读成倾斜 15°，渲染出来的箱子也是歪的。现在由
   `take_box_env/take_box.py::palm_frame_local()` 从手指刚体贴片量出掌心坐标系，`build_cell_model.py`
   和环境共用同一份实现。
6. **镜像双手 + 手掌坐标系 = 一连串符号陷阱**（这一步踩了三次，最后靠"指尖卷曲方向"才定案）：
   掌心坐标系由 `(手指, 食→小指, 叉乘)` 定义，而叉乘在镜像下反号，所以
   ① 直接用"右手 = 左手的镜像"去写目标帧，得到的矩阵 det = −1（反射），`mju_mat2Quat` 对非旋转
   矩阵给的是垃圾，腕部滚转悄悄跑 90°；
   ② 把张开/法线一起翻过来"修"成正常旋转，等于把手绕手指轴转 180°，**手背贴到箱面上**；
   ③ 两只手都够不到"手指朝前"的那个滚转角——用穷举扫过绕掌法线的滚转 φ，φ=0 时残差 0.11
   （21°，够得到的解是手背朝箱），φ=+30° 时才降到 0.011（1°）。
   结论：**两只手用同一个正常旋转目标帧**（掌心法线朝箱面），再绕掌法线滚转
   `task.grasp_roll_deg = 30°`（手指朝前下方）。验收判据写在
   `tests/test_take_box.py::test_pre_grasp_pose_grasps_with_the_palms`：闭手时指尖和拇指
   都必须朝箱子移动（手指只会往掌心侧卷，这是与坐标系无关的解剖学事实）。
   现在两只手的 IK 残差都是 0.01445/0.01449，关节角几乎完全镜像。
7. **配置注释会和几何一起腐烂**：`comment_cell` 还写着"台面顶面 0.37、箱心 0.45"时，
   `cell.table.pos[2]` 已经被改成 0.55——箱子有一半埋在台面里，抬升目标也从 0.65 涨到 0.75
   （手臂够不到，任务直接不可解）。`tests/` 现在有 `test_box_rests_on_the_table()` 和
   `test_lift_target_is_reachable()` 守住这两条。
8. **一个抓取标量不能把同一个数字塞给 11 个关节**：手指行程从 0.58 rad（拇指俯仰）到 1.6 rad
   （手指屈曲）不等，统一发 1.2 rad 的结果是拇指一直顶在限位上（位置伺服持续出力，真机上
   就是堵转）、四指只闭到行程的 75%、拇指 IP 102%。现在每个通道按自己的行程缩放，
   `grip = 1` 就是"每个通道到自己限位"。
9. **"松手脱手"一次打开了两个奖励漏洞**（都已堵上）：(a) 势函数按抓取标志切换
   （未抓 `Φ = -掌心误差`、抓住 `Φ = -(高度+倾角)`）时，松手等于把负项甩掉、白拿 +0.44；
   (b) 抓取奖励只要重新闭手就能再领一次，于是策略学会**每隔一步开合手、原地反复吃 +2**，
   一步都不抬（40 万步训练后 `ep_success_rate = 0`）。现在势函数只依赖状态（掌心误差、
   箱高误差、箱体倾角），抓取奖励每回合只发一次。
10. **抓到的一瞬间不要把箱子"吸附"到掌心坐标系**：原来的 carry 直接令 `箱姿态 = 掌心坐标系`，
    于是接触那一刻箱子会被瞬移剩下的抓取误差（最多 4.5 cm）、并旋转到掌心坐标系上——而后者
    在预抓取位姿就歪 9°，策略手腕一动就放大到 40°+，`viewer_demo.py` 里就是"一碰到箱子就跑"。
    现在锁存时记录**箱子在掌心坐标系里的固定相对位姿**，之后箱子跟着手做刚体运动：接触瞬间
    0 mm / 0.000°，抬升过程中倾角从 0° 起算（重训后中位 4.3°）。
11. **grasp site 是手掌内部的虚拟点，不是掌面**：它取的是"四指尖中点与拇指尖的中点"，
    把它对到箱面上，整只手会**插进箱子 27–32 mm**（`mj_geomDistance` 量出来的负距离）；
    反过来如果只对箱面不做修正，手背又会在外面留出几厘米空隙——用户看到的"手心离箱子还有
    很大间隙"就是这个。现在 `solve_ready()` 边解 IK 边量"这只手到箱子的最近距离"，把目标沿
    掌法线挪到留 `task.palm_clearance`（默认 2 mm）空隙，并把这个箱体坐标系下的偏移写进
    `take_box_ready.json`，环境（`face_points()`）和 `box_api.palm_targets()` 都按它算目标，
    所以"掌心贴箱面"在 RL 和脚本两条路里是同一个定义。

## 维护记录（2026-09-21）

一次代码维护的清单。改动都跑过 `python3 -m pytest tests/ -q`（22 项）、`python3 build_cell_model.py
--check`、`python3 demo_plan.py`（0 条约束记录）、`eval_rollout.py`，以及一次 40 万步的对照训练。

| 位置 | 改动 |
|---|---|
| `take_box_env/take_box.py` | 掌心竖直轴改为实测掌心坐标系（`palm_frame_local()`，含右手镜像符号 + 预抓取位姿校验），箱子朝向与 `box_tilt()` 因此才真实；`step()` 里把 `_hand_ctrlrange()` 提到循环外 |
| `build_cell_model.py` | 右手抓取目标帧改成正常旋转（`det = +1`）并加断言；掌心坐标系测量委托给 `env.take_box.palm_frame_local()`，与运行时共用一份实现；删掉未使用参数 |
| `take_box_env/box_api.py` | 同样复用掌心坐标系测量，右手掌心目标帧也改成正常旋转（同一处 det = −1 的反射 bug）；`ik()` 局部收敛即退出（右手腕够不到目标姿态时原来每步白跑 400 次迭代，`plan()` 因此快 8 倍）；删死代码（`offset = centre[2] * 0.0`）与未使用参数；`check_command()` 的偏航上限提成 `BoxLimits.yaw_max_deg`；`plan()` 的位置/偏航插值统一；模块文档里的 `robot.execute()` 改成 `robot.plan()` |
| `train_ppo.py` | `--viewer` 改用 `viewer.is_running()`：原来访问的 `viewer.running` 不存在，一开窗口就 AttributeError |
| `cell_config.json` | `cell.table.pos[2]` 0.55 → 0.45（改前箱子有一半埋在台面里，抬升目标因此是够不到的 0.75）；`comment_cell` 与真实几何对齐 |
| `assets/` | `build_cell_model.py` 重新生成；右臂预抓取位姿变化，箱子端平度从 21° 降到 9° |
| `tests/` | 新增 `tests/test_box_api.py`（7 项，这个模块此前 0 覆盖）+ 4 项掌心坐标系/几何回归测试 |
| `requirements.txt` | 新增，按本机实测版本给范围约束（`mujoco 3.12 / gymnasium 1.3 / sb3 2.9 / torch 2.13 / numpy 1.26`） |

**没有动**：`runs/` 里的旧权重、`task` 的奖励与容差参数、PPO 超参。旧权重与新工位不匹配这件事
记在上面"训练结果"里——重训 40 万步（约 3 分钟）就能替换掉。

## 维护记录（2026-09-22）：抓取动作

针对"抓取动作"这条路径的复查，`pytest tests/ -q` 27 项全过；每次改动都在临时目录里重跑
40 万步对照训练验证过。

| 位置 | 改动 |
|---|---|
| `build_cell_model.py` | 手指执行器 `ctrlrange` 改成**每个关节自己的行程**（原来所有通道都写 `0 1.2`：拇指俯仰顶限位持续出力、四指只闭 75%） |
| `take_box_env/take_box.py` | 抓取标量按每个通道行程展开（`_hand_ctrlrange(side)` 返回数组）；`face_points()` 改用箱子自身坐标系（原来固定世界 ±y，箱子偏航 10° 就凭空报出 20 mm 掌心误差）；新增 `hand_errors()` / `grasp_ready()` 供环境和 `box_api` 共用；势函数去掉抓取标志分支、抓取奖励每回合只发一次；新增 `allow_release`（默认关闭） |
| `take_box_env/box_api.py` | 复用 `env.grasp_ready()`；`servo_step()` 的抓取映射跟着改成按通道行程 |
| `tests/` | 新增 5 项：手指行程映射、箱子坐标系、默认不脱手、`allow_release` 脱手、抓取奖励不重复发 |

修完后 `demo_plan.py` 的搬运段掌心误差从 13.0 mm（其中大部分是"世界 ±y"的假误差）降到
**0.9 mm**，抬起 0.117 m、倾角最大 12.3°、约束记录 0 条；重训 40 万步的策略确定性
100 回合 **100%** 成功、抬起 0.092 m、倾角 4.8°、25 步收尾。

## 维护记录（2026-09-22 续）：viewer 里"一碰到箱子就跑"

`viewer_demo.py` 加载的是 `runs/ppo_take_box.zip`，两个原因叠在一起让它看起来像"关节坏了、
碰到箱子就飞"：

1. **箱子在接触瞬间被吸附到掌心坐标系**——见上面第 10 条。修法是锁存时记录固定相对位姿。
2. **权重是 2026-09-18 的旧权重**，输入分布已经对不上（43% 成功率），策略输出长期贴着 ±1
   饱和，箱子跟着手腕一起甩。已重训并替换，旧文件备份为 `runs/ppo_take_box.zip.bak-20260922`。

验证：接触瞬间箱子位移 **0.00 mm**、姿态变化 **0.000°**；重训策略确定性 100 回合
**100% 成功**、抬起 0.104 m、倾角中位 4.3°、26 步收尾（`viewer_demo.py` 现在跑的就是这份）。

### 追加：右手抓取姿态是"反"的（2026-09-22 晚）

用户看图后指出右手是反的——**确认属实，而且比"反"更糟：两只手都在用手背贴箱面**。
判据是解剖学事实"手指只会往掌心侧卷"：闭手时指尖/拇指相对箱面的位移

| 目标姿态 | 左手四指 / 拇指 | 右手四指 / 拇指 |
|---|---|---|
| 旧代码（右手中标称镜像） | −23.7 / −17.0 mm（背离箱子） | −23.8 / −16.6 mm（背离箱子） |
| 修正后（掌心朝箱 + 绕法线滚转 30°） | **+23.6 / +14.7 mm** | **+23.6 / +14.2 mm** |

根因和修法写在"踩过的坑"第 6 条。修正后连带的好处：两只手 IK 残差从 0.0137/0.111 变成
0.01445/0.01449（完全镜像）、`demo_plan.py` 抬升段倾角从 12.3° 降到 **0.5°**、掌心误差从
2.0 mm 降到 0.1 mm，`tests/` 新增 `test_pre_grasp_pose_grasps_with_the_palms` 守住解剖学判据。
两个环境的权重都按新几何重训替换（旧文件备份成 `runs/*.zip.bak-20260922b`）：
`take_box` 100%、抬起 0.122 m、倾角 5.4°、51 步；`take_box_reach` 100%、7 步。

### 追加：掌心贴箱面 + 流程拆文件（2026-09-22 深夜）

* **"双手离箱子有很大间隙"**：grasp site 是"四指尖中点与拇指尖的中点"，落在手掌**内部**，
  把它对到箱面上整只手会插进箱子 27–32 mm（`mj_geomDistance` 实测）；反过来不做补偿时手又
  停在外面。现在 `solve_ready()` 边解 IK 边量手到箱子的最近距离，把目标沿掌法线挪到留
  `task.palm_clearance = 2 mm`，并把箱体坐标系下的偏移写进 `take_box_ready.json`——
  环境（`face_points()`）和 `box_api.palm_targets()` 用同一份偏移，实测掌面离箱面
  **1.7–2.6 mm**（闭手后最近 5 mm，不穿插）。
* **搬运流程拆成独立文件**：`flows/prepare.py → grasp.py → lift.py → carry.py → place.py →
  release.py`，`flows/full_task.py` 只负责按顺序串 (对应 YAML 的
  `prepare_action → take_box_step1 → wait → step2`)。每个文件都能单独跑
  （`python3 -m take_box_flows.lift` 会自动先做前面几步），也可以带参数
  （`--height / --forward / --side / --yaw`）。`demo_plan.py` 变成 `flows.full_task` 的薄包装，
  老命令照旧可用。录像改成在 `servo_step` 上挂钩子逐帧抓（`--video`）。
* **完整流程实测**（`python3 -m take_box_flows.full_task`）：抬起 +91 mm、前移 +90 mm、放回台面后松手，
  箱心 (0.420, 0, 0.550) → (0.505, 0, 0.578)，抬升/搬运段倾角 ≤0.2°、掌心误差 ≤1.9 mm、
  约束记录 **0** 条。抬起只到指令 120 mm 的 75% 是腕部限位下"位置/姿态不可兼得"，
  流程打印的是实际值（想要 120 mm 得放宽 `R6/R7` 范围或改箱位）。
* 权重按新几何又重训了一轮：`take_box` 100%（抬起 0.103 m、倾角中位 6.2°、16 步）、
  `take_box_reach` 100%（22 步），旧文件备份成 `runs/*.zip.bak-20260922c`。
  `tests/` 新增 `tests/test_flows.py`（5 项，含整条 lift→carry→place→release 链路）。

### 追加：奖励函数单独成文件（2026-09-22）

* 奖励从 `take_box_env/take_box.py` 挪到 **`take_box_env/rewards.py`**：`RewardConfig`（系数，可在
  `cell_config.json → task.reward` 覆盖，未知键直接报错）、`GraspStep` / `RewardResult`
  （纯数据进出）、`TakeBoxReward.potential()` / `.step()`。环境只负责喂状态、写回结果
  （`_reward_step()` / `_compute_reward()`），所以奖励可以脱离 MuJoCo 单测。
* 新增 `tests/test_rewards.py`（9 项）：势函数只依赖状态（不随抓取标志跳变）、抬升/端平
  单调变好、抓取奖励每回合只发一次、松手不再白拿、保持计数与达标、控制代价、配置解析。
* 顺手删掉 `task.reach_steps` / `task.target_radius` 两个没人读的配置键。
* 行为与拆分前逐项对齐：确定性 100 回合仍是 **100%**，锁存那一步的奖励仍是 `+1.98`。

### 追加：抬升/搬运上限报警（2026-09-22）

`flows/lift.py`、`flows/carry.py` 默认值改回可达范围（lift 0.09 m / carry 0.10 m），
并且在**超过腕关节能力**时直接打印告警而不是把箱子甩歪：

```
$ python3 -m take_box_flows.full_task --height 0.30
  ! 抬起 300 mm 后箱体倾角 42.7°：超过本工位能端平抬起的 ~100 mm，腕关节 R6/R7 已经跟不上
    （实际抬升 +245 mm）
```

端平抬升的上限（实测 ~0.10 m，抬到 0.10 m 时倾角 0.3°）来自 `R6/R7` 只有 ±1.22 rad；
要更高就得放宽这两个关节的限位或把箱位往前挪。默认流程（lift 0.09 / carry 0.10）实测：
抬起 +68 mm、前移 +100 mm、倾角 ≤0.3°、约束记录 0 条。
