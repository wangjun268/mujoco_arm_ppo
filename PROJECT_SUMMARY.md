# 当前内容总结

> 一句话：用 **MuJoCo** 建模 + **Gymnasium** 封装 + **Stable-Baselines3 PPO** 训练，让珞石 xMate Pro7 的末端到达随机目标，并驱动同一套场景完成视觉抓取与整段取放。当前注册 **4 个环境名，全部使用真实 URDF STL 网格**（官方 Pro7 + L20 手部网格），并提供评估、曲线、演示视频与实时查看器。

> **胶囊体模型已全部移除**：原 `two_joint`、`three_joint`、`six_joint`（ER3）、
> `pro7_joint`（Pro7 胶囊体版）以及 `rokae_xmate_pro7_pick.xml`（旧胶囊体抓取场景）
> 连同对应环境、解析 IK 基线脚本与文档章节一起删除，只保留有实物网格的模型。

> **末端执行器已更换**：Pro7 抓取场景的平行两指夹爪已被**灵心巧手 LinkerHand L20 灵巧手**替换。
> 厂商 URDF 与 22 个 STL 放在 `assets/linkerhand_l20/`，`tools/convert_hand_urdf.py` 把 URDF 转成 MuJoCo
> 片段并**用仿真标定抓取**（抓取中心 + 闭合量）；21 个关节按"开合协同"驱动，动作空间仍是
> 「7 臂力矩 + 1 抓握指令」，因此既有 checkpoint 形状不变。下文凡提到"夹爪 / 两指 / 钳口"的
> 段落，都是更换之前的历史实测结论，保留作调参依据。

---

## 1. 现状总览

| 环境 | 自由度 / 运动 | 模型 | 整集命中率 | 终距 中位 | 平均到达步数 |
|---|---|---|---|---|---|
| `pro7_urdf` | 珞石 xMate Pro7 七自由度到达（**真实 STL 网格**） | `results/ppo_pro7_urdf.zip` | **84.2%** | 0.147 | ~61 (of 250) |
| `pro7_pick` / `pro7_pick_urdf` | Pro7 + L20 视觉抓取（同一场景的两个名字） | 专家伺服 / `grasp_policy_online.pt` | 20/20 抓取 | — | — |
| `pro7_pick_place` | 整段取放（接近→夹紧→搬运→放置） | 专家伺服 / `grasp_policy_online.pt` | 最好一轮 pick 100% / place 70% | — | — |
| `dual_arm_reach` | lkwy73_o1 双臂各自到达（真实 STL 网格，14 关节） | `results/ppo_dual_arm_reach.zip` | 见 §11 | — | — |
| `dual_arm_coop` | 双臂共持一根刚性杆（两目标保持固定相对位姿） | `results/ppo_dual_arm_coop.zip` | 见 §11 | — | — |

上述环境共用同一套训练/评估/可视化代码，用 `--env <name>` 切换；`pro7_urdf` 在 RTX 5080 + CUDA 上训练，速度约 3500–4000 步/秒（2.0M 步约 8.4 分钟）。

---

## 2. 分层架构

```text
物理模型 XML (assets/rokae_xmate_pro7_real.xml / rokae_xmate_pro7_pick_real.xml)
   │  mujoco.MjModel.from_xml_path()
   ▼
Gymnasium 环境 (env/rokae_reacher.py + 两个抓取环境，由 env/__init__.py 的 make_env 工厂按名创建)
   │
   ▼
PPO 训练 (train_ppo.py) ──► results/ppo_<env>.zip
   │
   ├──► eval_rollout.py   指标 + 学习曲线 + 演示视频（--viewer 开窗口）
   ├──► viewer_demo.py    实时 MuJoCo 窗口
   └──► make_montage.py   逼近→命中 静态拼图

取放专家 (grasp.common.iter_pick_and_place，逐控制步)
   ├──► grasp/pick_place_demo.py   批处理演示 / GIF / 拼图
   └──► ros2_ws/             ROS 2 节点：话题 + 服务 + PickPlace Action

模型工具层 (tools/*.py，离线跑一次) ──► 重新生成 assets/ 下的 MJCF / URDF / STL
```

---

## 3. 文件清单（角色）

根目录只留 **3 个共享模块 + 4 个到达入口**，抓取链路整体进 `grasp/` 包、模型生成脚本整体进
`tools/` 包（旧路径 → 新路径的对照表见 [PROJECT_GUIDE.md](PROJECT_GUIDE.md) §3）。每个脚本
`python3 grasp/demo.py` 与 `python3 -m grasp.demo` 两种跑法都支持。

### 3.1 根目录：共享模块 + 到达入口

| 文件 | 作用 |
|---|---|
| `paths.py` | 项目路径/默认产物位置集中管理 + 模型路径校验与友好报错（**仓库根锚点**，ROS 节点也靠它认仓库） |
| `cli.py` | 统一 `--env/--model/--out` 命令行助手与 `load_policy()` |
| `live_viewer.py` | **实时可视化窗口**（节流 + 相机 + 窗口内状态文字 + 安全关窗），训练与演示共用 |
| `train_ppo.py` | PPO 训练入口（向量化环境、回调统计、续训） |
| `eval_rollout.py` | 量化解算 + 学习曲线 + 演示视频 |
| `viewer_demo.py` | 实时 MuJoCo 可视化窗口（关窗停止，`T` 换目标） |
| `make_montage.py` | 逼近→命中 静态拼图 |
| `start.sh` / `pro7_pick_place.desktop` | ROS 2 一键启动（构建 + 起节点 + rviz2），带桌面图标入口 |

### 3.2 环境层 `env/`

| 文件 | 作用 |
|---|---|
| `env/__init__.py` | `make_env()` 注册表，`--env` 一键切换任务 |
| `env/base_reacher.py` | **所有到达任务的基类**：观测/动作/奖励/渲染/步进只实现一次（子类只给模型与采样） |
| `env/rokae_reacher.py` | `RokaePro7RealReacher`：真实 STL 网格版 Pro7 到达环境（obs 27，动作 7）——`BaseReacher` 子类 |
| `env/rokae_pro7_pick.py` | Pro7 抓取环境（obs 30，动作 8）：真实网格场景 + 干扰块，观察用 RGB-D 检测到的红块 3D 坐标驱动 |
| `env/rokae_pro7_pick_place.py` | Pro7 **整段取放**环境（obs 38，动作 8）：夹住不算结束，方落到放置垫才结束 |
| `env/dual_arm_reacher.py` | `MultiArmReacher` 基类 + `DualArmReach` / `DualArmCoopReach`：双臂模型上按"臂列表"驱动的多臂任务（obs 60，动作 14） |

### 3.3 抓取层 `grasp/`（相机 + L20 手 + 红块）

| 文件 | 作用 |
|---|---|
| `grasp/common.py` | **抓取场景与专家的唯一来源**：台面/红块/**干扰块**/夹爪常量、场景构建、专家伺服、抓取判据、取放计划 |
| `grasp/detect.py` | 红色像素分割 + RGB-D 深度反投影 → 红块世界 3D 坐标（平面吸附，误差 ~6 mm） |
| `grasp/policy.py` | 抓取 MLP 策略（`Policy`/`OBS_DIM`/`ACT_DIM`/`load_policy`/`rollout`），训练与回放共用 |
| `grasp/demo.py` | 端到端演示：检测红块 → 3D 坐标 → 任务空间伺服 → 七轴抓取 |
| `grasp/pick_place_demo.py` | **取放演示**：源台抓取 → 提起 → 搬运 → 放到目标台的放置垫（含 GIF / 拼图） |
| `grasp/montage.py` | 生成“逼近→抓住”静态拼图 |
| `grasp/overlay.py` | 相机检测可视化：红块分割/质心/定位 3D 点回投影 + 检测点 vs 真值 3D 对比 |
| `grasp/visualize.py` | 抓取回放 GIF（专家/已学策略驱动，带绿色检测标记点） |
| `grasp/view.py` | 实时 MuJoCo 窗口：实时看七轴臂抓取（专家/策略），绿色标记实时显示检测点 |
| `grasp/supervised.py` | 实时监督（DAgger 行为克隆）训练：专家老师实时标注 (观测, 动作)，在线训练策略并实时监测损失/抓取成功率 |
| `grasp/train_live.py` | **实时训练过程可视化**：默认训练**取放整段**，每个评估轮渲染“当前策略 接近→夹紧→提起→搬运→放置 故事板 + 实时指标面板（pick % / place %）”，逐帧合成 `training_process.gif`；`--task grasp` 回到抓取任务 |

### 3.4 模型工具层 `tools/`

| 文件 | 作用 |
|---|---|
| `tools/build_dual_arm_model.py` | 由双臂描述生成 `assets/dual_arm_reach.xml` 并拷贝网格（`--check` 只校验） |
| `tools/convert_hand_urdf.py` | 厂商 L20 URDF → MuJoCo 片段（含抓取标定，无参数，直接重跑） |
| `tools/convert_arm_urdf.py` | 编译后的 MuJoCo 抓取场景 → URDF（网格 + 惯量，给 rviz2 用，`--check` 复核） |
| `tools/urdf_selfcheck.py` | 导出的 URDF 重新用 MuJoCo 加载，逐 geom 位姿 + 总质量对比（被 `--check` 与测试复用） |
| `tools/make_wrist_flange.py` | 生成腕部转接件 STL（`assets/meshes/pro7_l20_flange.stl`） |

### 3.5 物理模型、测试与文档产物

| 文件 | 作用 |
|---|---|
| `assets/rokae_xmate_pro7_real.xml` | **真实 Pro7 URDF STL 网格**（官方几何）的 7 轴到达模型（带电机、mocap 目标） |
| `assets/rokae_xmate_pro7_pick_real.xml` | **抓取/取放场景（当前默认）**：真实 Pro7 网格 + L20 手 + 相机 + 红块 + 绿/蓝/黄干扰块 + **两张工作台**（源台 + 目标台与放置垫） |
| `assets/meshes/xMatePro7/*.stl` | 官方 Pro7 STL 网格（随仓库提供，相对路径引用） |
| `assets/meshes/pro7_l20_flange.stl` | 腕部转接件（`tools/make_wrist_flange.py` 生成的回转体），补上腕法兰与 L20 底座之间的 60 mm 间隙；纯视觉零质量，抓取场景与 `pro7_urdf` 到达训练模型都挂 |
| `assets/dual_arm_reach.xml` | lkwy73_o1 双臂任务模型（14 个力矩电机 + 双侧 mocap 目标 + 指尖 tip site） |
| `assets/dual_arm/meshes/*.STL` | 双臂降面网格（29 件，8.4 MB，单件 ≤ 20k 面） |
| `tests/` | pytest 冒烟 + 回归测试：观测维度/顺序（checkpoint ABI）、路径、抓取契约、生成的资产可复现 |
| `ros2_ws/` | ROS 2 封装：取放节点 + 客户端 + msg/srv/action 接口 |
| `requirements.txt` | 依赖清单（mujoco / gymnasium / sb3 / torch / …） |
| `README.md` / `PROJECT_GUIDE.md` / `PROJECT_SUMMARY.md` | 快速上手 / 架构与逐文件说明 / 本文档 |
| `results/` | 训练产物（模型/日志/曲线/视频/TensorBoard） |

---

## 4. 关键设计决策

- 奖励主项 = `-距离`（稠密），配合命中 `+2`；Pro7 到达任务另加 `0.5·exp(-dist/0.3)` 的“接近度”塑形，让远距离也有连续梯度。
- 关节角用 `cos/sin` 进观测，消除周期歧义。
- 目标用 `mocap` 无碰撞体；目标点由随机位姿的正运动学(FK) 得到，天然保证可达。
- 到达任务设零重力（纯惯性控制），保证 PPO 可收敛。
- 工具尖点偏离腕滚转轴，使 7 个关节都影响末端位置。
- 可维护性：路径/默认值集中在 `paths.py`，命令行集中在 `cli.py`，到达任务收敛到
  `env/base_reacher.py`，抓取场景与专家收敛到 `grasp/common.py`，
  并用 `tests/` 把观测维度与顺序（checkpoint ABI）钉死。
- 实时可视化：`live_viewer.py` 统一封装原生 MuJoCo 窗口（按 fps 节流、窗口四角显示
  step/fps/奖励/回合长度、关闭后安全退出），`train_ppo.py --viewer` 与
  `grasp/train_live.py --live` 都能"边训边看"；关掉窗口训练继续。

> 细节与踩坑见 [PROJECT_GUIDE.md](PROJECT_GUIDE.md) 的 §5。

---

## 5. 训练产物清单

训练产物集中在 `results/` 下，按类型分层：

| 产物 | 说明 |
|---|---|
| `results/ppo_<env>.zip` | 训练好的 PPO 策略权重 |
| `results/<env>/rollout.mp4` / `rollout.gif` | 策略演示视频 |
| `results/<env>/montage.png` | 逼近→命中 静态拼图 |
| `results/<env>/learning_curve.png` | 训练奖励曲线（累计时间步） |
| `results/tb_<env>/` | TensorBoard 事件（`tensorboard --logdir results/tb_<env>`） |
| `results/logs/<env>_<n>.txt` | 训练 stdout 日志（按环境/轮次命名） |

当前 `pro7_urdf/` 的到达产物与 `pro7_pick/`、`pro7_pick_place/` 的抓取产物均已产出；
到达权重为 `results/ppo_pro7_urdf.zip`，训练 stdout 见 `results/logs/pro7_urdf_1.txt`。
（`results/` 里被移除环境的旧产物是历史数据，未随本次清理删除。）

---

## 6. 快速使用

```bash
# 回归测试（约 2 秒）
python3 -m pytest tests -q

# 训练 / 续训
python3 train_ppo.py --env pro7_urdf --steps 2000000                           # 训练 Pro7 到达（默认环境）
python3 train_ppo.py --env pro7_urdf --steps 2000000 --viewer                  # 实时看 MuJoCo 窗口训练
python3 train_ppo.py --env pro7_urdf --init-model results/ppo_pro7_urdf        # 续训 Pro7
python3 train_ppo.py --env pro7_pick --steps 800000                            # 训练视觉抓取
python3 grasp/train_live.py --rounds 24 --live                                       # 取放整段训练 + 实时窗口（默认任务）
python3 grasp/train_live.py --task grasp --rounds 24 --live                          # 只训练抓取（0.5× 慢放）

# 评估 + 曲线 + 视频
python3 eval_rollout.py --env pro7_urdf --episodes 400

# 实时 MuJoCo 窗口（关窗停止，T 换目标）
python3 viewer_demo.py --env pro7_urdf

# ---- 抓取管道 ----
python3 grasp/demo.py --episodes 10 --video         # 检测红块→3D坐标→七轴抓取
python3 grasp/pick_place_demo.py --episodes 10 --video --montage   # 源台抓取→搬到目标台放置垫
python3 grasp/montage.py
python3 grasp/supervised.py --rounds 24            # 实时监督训练（在线 DAgger）

# ---- ROS 2 封装（七轴取放节点）----
./start.sh                                         # 一键：构建（如需）+ 起全部节点（节点 + rviz2）
./start.sh --demo --demo-seed 1                    # 连 demo 客户端一起起，自动跑一次取放
./start.sh --stop                                  # 停掉（含 rviz / 真机栈）
cd ros2_ws && ./run.sh                             # 等价的工程内入口
./run.sh ros2 launch pro7_pick_place_ros pick_place.launch.py demo:=true demo_seed:=1
./run.sh test                                      # 13 个用例：场景/专家/作业/Action
ros2 run pro7_pick_place_ros pick_place_client --seed 1 --reset
# `pro7_pick_urdf` 是 `pro7_pick` 的兼容别名：两者构建同一个真实网格场景
python3 train_ppo.py --env pro7_pick_urdf --steps 800000

# ---- 可视化 ----
python3 grasp/overlay.py                          # 检测→3D 可视化 PNG
python3 grasp/visualize.py --mode expert           # 抓取回放 GIF（专家）
python3 grasp/view.py --mode expert                 # 实时 MuJoCo 窗口（专家）
python3 grasp/view.py --mode policy --policy-path results/pro7_pick/grasp_policy_online.pt
python3 grasp/supervised.py --rounds 24 --visualize   # 训练后用学到的策略开实时窗口
python3 grasp/train_live.py --rounds 24                   # 取放训练过程 GIF（边训边看）
python3 grasp/train_live.py --rounds 24 --live            # 过程 GIF + 实时 MuJoCo 窗口（取放一轮 ≈ 50 s）
python3 grasp/train_live.py --rounds 24 --live --live-speed 4   # 同上，4× 播放，几分钟跑完
python3 grasp/view.py --task pick_place --mode policy      # 回放学到的取放策略

# 逼近→命中拼图
python3 make_montage.py --env pro7_urdf
```

---

## 7. 当前局限与说明

- Pro7 到达：0-g 任务，臂展 ~1.4 m、7 轴，起点/目标采样围绕随机锚点收窄（命中圈 0.15 m、单集 250 步）以利于收敛；真实网格版训练 2.0M 步后命中率 84.2%、终距中位 0.147 m。

## 8. 抓取扩展（夹爪 + 相机 + 红块检测 → 七轴抓取）

- **场景（双工位）**：**真实 Pro7 网格**（`rokae_xmate_pro7_pick_real.xml`）+ 平行两指夹爪（手指沿 +z 伸出、沿 +x 夹紧）+ 腕部 RGB-D 相机（`cam_hand`）。**源台**（`table`）放一个自发光红色方块（边长 5 cm）与三块**同尺寸干扰块**（绿 / 蓝 / 黄，位置经实测挑选：在相机可见范围内、但完全避开红块的采样足迹，且参与碰撞）；**目标台**（`table2`）在 +y 侧，台面嵌一块**放置垫**（`place_pad`，青绿色、不碰撞、顶面与台面齐平），标出落点 `PLACE_TARGET`。`pro7_pick` 与 `pro7_pick_urdf` 指向同一个真实网格场景（后者为兼容别名）。
- **检测**：`grasp/detect.py` 对手部相机做红色分割，取其像素深度反投影 3D，再吸附到已知桌面平面，得到红块世界坐标（均值 ~6 mm、最大 ~1.2 cm）；支持纯深度版与平面吸附版。
- **检测鲁棒性**：加入干扰块后，40 个随机红块位置仍 **100% 检出**、定位误差均值 6.3 mm（最大 11.7 mm），与无干扰时一致；红色分割只命中红块（连通域只有一个）。
- **抓取**：`grasp/demo.py` 用**任务空间分辨率伺服**（7 自由度雅可比，温和力限幅）把夹爪中心移到检测点，到位后闭合夹爪；当两指都与红块建立约束接触且夹紧时判定成功。修复夹持参考点后（见下）**20/20 成功**。
- **取放（新增）**：`grasp/pick_place_demo.py` 在同一场景上做「源台抓取 → 提起 → 搬运 → 落到目标台放置垫 → 张开」的解析专家流程（`grasp.common.pick_and_place`），实测 **10/10 成功**，落点平均误差 **29 mm**（判定阈值 50 mm），产物在 `results/pro7_pick_place/`（`pick_place_demo.gif` + `pick_place_montage.png`）。
  搬运是**准静态**的：夹持力只有约 0.3 N，专家以 1 cm 分段 + 柔增益移动（`TRANSPORT_*`）；一旦加快，方块会把活动指顶开（间隙 0.050 → 0.097，接触归零）而掉落 —— 与方块质量无关（3.75 kg 与 0.0004 kg 同样会掉）。
- **修复的根因**：`grasp_center` 站点原本在**指端**（抓手系 z=0.075），而手指内侧面只延伸到 z=0.075、其中心在 z=0.04。于是伺服把方块**顶到两指之外**（实测方块中心在 z=0.078），只剩指尖角接触——静态能"夹住"，一动就滑脱。把站点移到真正的钳口中心（0.048）后，方块落在两指之间，抓取率 16/20 → **20/20**，搬运也从 0/10 → **10/10**。`tests/test_grasp.py::test_cube_is_pinched_between_the_fingers` 钉住这个回归。
- **RL 环境**：`pro7_pick` 将检测到的红块 3D 坐标作为观测，动作 = 7 臂扭矩 + 1 夹爪开关；奖励 = 接近度 + 抓取大奖。由于 8 自由度 + 视觉观测，PPO 收敛较慢，可作为“学习式抓取”的进一步训练路径。
- **实时监督训练**：`grasp/supervised.py` 让解析伺服从 `grasp.demo` 作为专家老师，实时给出每个 (观测, 动作) 标签；策略在线用 MSE 监督学习，β 混合 DAgger（前期跟随专家、后期跟随策略但仍由专家标注）以覆盖策略自身状态分布、消除纯行为克隆的误差累积。实测 train MSE 从 0.14 降到 ~0.01，策略回放抓取成功率从 0% 升到 ~40%（默认 `--obs-target true` 使用真值目标便于学；`--obs-target vision` 换成 RGB-D 估计）。
- **取放 RL 任务（新增）**：`env/rokae_pro7_pick_place.py`（`pro7_pick_place`，obs 38 维）把**整段取放**变成一个 episode：抓取成功不再结束，只有方块落到目标台放置垫上才结束（抓丢/方块被撞出台面以 `lost=True` 提前结束，避免污染 DAgger 缓冲区）。观测量 = 抓取环境 30 维 + `holding` + 子目标误差 + 方块到垫子误差 + "到位该松手" 标志；搬运计划由 `grasp.common.PlacePlanner` 给出（settle → lift → carry → lower，1 cm 路点，≈2 cm/s 准静态）。`grasp/train_live.py --rounds 24` 用带计划的专家做 DAgger，实测 24 轮内最好一轮 **pick 100% / place 70%**（轮间波动大，脚本自动保留最好检查点），产物 `results/pro7_pick_place/training_process.gif` + `grasp_policy_online.pt`。
  > 只有**搬运**由环境执行（transport mode）：搬运力矩只占执行器量程 1~2%，远小于模仿网络 ~1.4% 的动作噪声，纯模仿的搬运必掉块（实测 2e-4 动作 MSE 的行为克隆放 0/6，脚本搬运 10/10）。策略负责接近、夹紧、失败重试与松手时机——pick % / place % 正是按这些打分。

## 9. 使用真实 URDF 几何（Pro7）

MuJoCo 3.x **原生支持读 URDF**（`MjModel.from_xml_path("*.urdf")`），但因 Pro7 原 URDF 惯性“不正确”且无电机/夹爪，直接加载需修正惯性（非对角线清零 + 三角不等式平衡）再补执行器。

项目提供两条“真实网格”路径：
- **`pro7_urdf`**：`assets/rokae_xmate_pro7_real.xml`，用官方 STL 作连杆网格，保留 URDF 关节链/限位 + 7 电机 + mocap 目标 → 到达环境真实几何版。已训练 **2.0M 步**：`results/ppo_pro7_urdf.zip`，评估 400 集命中率 **84.2%**、终距中位 0.147 m、平均 ~61 步（上限 250）；配套 `learning_curve.png` / `rollout.(mp4|gif)` / `montage.png` 在 `results/pro7_urdf/`。
- **`pro7_pick_urdf`**：`assets/rokae_xmate_pro7_pick_real.xml`，真实网格臂 + 夹爪 + 腕部相机 + 红块；RGB-D 检测→伺服抓取同样可用（实测 pin 定位误差 2–10 mm，专家 3 次抓取 2/3 成功）。

**网格已随仓库提供**：8 个官方 STL 放在 `assets/meshes/xMatePro7/`，XML 里用相对路径引用，
因此不再依赖 `/home/wj/rokae_ros2`（即使外部目录不存在也能加载，几何与原先逐项一致）。

- 到达任务采用零重力以保收敛；真实有重力时需更多训练或更强调控。
- 到达仍是位置任务；腕部各轴因工具点偏离被间接利用，但未要求姿态对齐（更贴近真实抓取需“到达+姿态”任务）。

---

## 10. 可选下一步

1. 到达 + 姿态（让第 4、6 轴有独立作用，贴近抓取/装配）。
2. 加入重力 / 真实动力学 + 域随机化（Sim-to-Real 准备）。
3. 更强算法对比：SAC / TD3 / 复现学习（behavior cloning）。
4. 控制约束任务：避障、沿轨迹跟踪、推箱/插销等操作。
5. 调参加长训练，把 Pro7 真实网格版的到达命中率推到更高。
6. 双臂任务：给"保持成功 K 步"作为成功判据（协作任务更需要），以及把双手从"钉住"
   改成可驱动的抓取-搬运任务。

---

## 11. 双臂扩展（lkwy73_o1，两臂 14 关节）

模型 `assets/dual_arm_reach.xml` 由 `tools/build_dual_arm_model.py` 生成：官方双臂描述
（2×7 自由度臂 + 2×11 自由度手，11.11 kg）→ 14 个归一化力矩电机 + 双侧 mocap 目标
+ 指尖 `tip_left/tip_right` 站点；22 个手指关节用等式约束钉在张开位；CAD 里固有的
4 对自重叠（`L1↔base` −7.1 mm、`L5↔L7` −11.2 mm 及镜像）已用 `<contact><exclude>`
排除，零位接触数为 0。网格降面后 68.7 MB → 8.4 MB，单件 ≤ 20000 面（原 `base_link`
274744 面超过 MuJoCo 的 20 万面上限，直接用原 URDF 会加载失败）。

| 环境 | 观测/动作 | 任务语义 | 训练步数 | 整集成功率（确定性 150 回合） | 单臂至少命中一次 | 平均步数 |
|---|---|---|---|---|---|---|
| `dual_arm_reach` | 60 / 14 | 两臂各自独立目标，**到达即锁存**（命中一次即算完成，不需要保持） | 2.6M | **43.3%** | 左 64.0% / 右 71.0% | 139 / 200 |
| `dual_arm_coop` | 60 / 14 | 两目标是一根刚性杆的两端，必须**同时**到位且相对位姿匹配 | 1.2M | **21.3%**（训练中随机策略 42.4%） | — | 165 / 200 |

训练与评估命令：

```bash
python3 tools/build_dual_arm_model.py --check                       # 校验资产（nu=14, nmocap=2, 接触 0）
python3 train_ppo.py --env dual_arm_reach --steps 2000000 --n-envs 8
python3 train_ppo.py --env dual_arm_coop  --steps 1200000 --n-envs 8
python3 eval_rollout.py --env dual_arm_reach --episodes 150  # 指标 + 曲线 + 视频
python3 viewer_demo.py  --env dual_arm_coop                   # 实时窗口
```

实现要点与踩坑：

1. **多臂而不是"两臂写死"**：`MultiArmReacher` 按模型里的臂列表构建观测/动作/目标，
   `ARMS` 加一项就能扩到三臂；`nq≠nu`（36 个关节只驱动 14 个），所以没有复用假设
   `nq==nu` 的 `BaseReacher`。
2. **单臂一次性加分**：`+1.0`（每臂每回合一次）+ `+2.0`（两臂都到）。没有它时策略在
   "两条臂都到位"之前拿不到任何离散信号，成功率明显更低（实测同预算 21% vs 26%）。
3. **到达是否锁存是关键设计**：早期版本要求两臂同时停在阈值内，结果策略"到了又飘走"
   （左臂 58% 曾命中，但最终距离中位 0.122 m）。单臂任务改为锁存后确定性命中率从
   18.3% → 43.3%；协作任务保留"必须同时保持"，因为共同持杆本来就是保持问题。
4. **协作任务的奖励权重**：杆长误差系数最初取 2.0，初始杆长误差约 0.7 m，奖励被它
   完全支配（回合回报 −213，成功率 1.7%）。改成 0.5 + 容差 0.08 m + 半径 0.10 m 后
   成功率 42.4%（训练中）。
5. **镜像关系**：两臂沿 y 分列，对称面 y=0，镜像符号 `(-1,-1,-1,+1,-1,+1,-1)`；用
   网格点云核对，正确符号残差 3–5 mm（CAD 两件自身的不对称量），符号错会跳到 75 mm+。
   协作任务的"可同时达到"正是靠"左臂姿态 + 镜像姿态"构造，再施加小刚性变换
   （yaw ≤0.22 rad、平移 ≤0.04 m），实测目标离可达构型中位 0.06 m、95% 0.14 m。
6. **已知局限**：协作任务的"成功"是瞬时判据，探索噪声比收敛后的确定性策略更容易
   抓到那一瞬间（42.4% vs 21.3%），把它改成"连续 K 步保持"是下一步；右臂在 1.1% 的
   步数里会碰到中央底座（`base_link ↔ rh_index_*`），因为本项目到达任务默认保留
   机器人自碰撞（Pro7 到达模型则是彻底关闭接触）。

---

## 12. 目录重组（脚本分组）

根目录原来平铺了 23 个 `.py`，现在按职责收进包，**只挪位置、不改逻辑**：根目录留
`paths.py` / `cli.py` / `live_viewer.py` 三个共享模块 + `train_ppo.py` / `eval_rollout.py` /
`viewer_demo.py` / `make_montage.py` 四个到达入口；抓取链路 11 个文件进 `grasp/` 包，
模型生成 5 个文件进 `tools/` 包（完整对照表见 `PROJECT_GUIDE.md` §3）。

- **包内导入**改成 `from grasp.common import ...` 这类绝对导入；每个入口脚本开头加了三行
  `if __package__ in (None, ""): sys.path.insert(...)`，因此 `python3 grasp/demo.py` 与
  `python3 -m grasp.demo` 都能跑（`tools/*` 同理），从任意工作目录执行也不会找不到模块。
- **文档与注释同步**：三份 Markdown、`ros2_ws/README.md`、`assets/*.xml` 里的注释、
  `tests/*.py` 的导入全部跟着改；`README.md` 的 Files 段和 `PROJECT_GUIDE.md` §3
  现在是新的目录树。
- **ROS 2 侧**：`pro7_pick_place_ros/project.py` 的仓库识别标记改成
  `("grasp/common.py", "grasp/detect.py", "paths.py")`，导入名改成 `grasp.common` /
  `grasp.detect`；节点日志里的 URDF 生成命令提示也指向 `tools/convert_arm_urdf.py`。
- **生成物**：`tools/convert_arm_urdf.py` 与 `tools/convert_hand_urdf.py` 写进资产里的
  "由谁生成"注释跟着更新，两个资产已按新脚本名重新生成（逐行 diff 只有那一行注释），
  `test_committed_urdf_is_regenerable` 因此仍然通过。
- **验证**：`python3 -m pytest tests -q` 全绿（73 项，1 项 xfail）；
  `python3 grasp/demo.py --episodes 1` 抓到 1/1；
  `python3 tools/convert_arm_urdf.py --check` 的 31/31 geom 与质量仍然逐项一致；
  12 个搬动过的脚本 `--help` 与 `python3 -m` 方式均正常。

---

*本文件为项目当前状态总结；实现细节见 `PROJECT_GUIDE.md`，快速上手见 `README.md`。*
