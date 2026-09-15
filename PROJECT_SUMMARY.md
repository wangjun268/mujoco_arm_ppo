# 当前内容总结

> 一句话：用 **MuJoCo** 建模 + **Gymnasium** 封装 + **Stable-Baselines3 PPO** 训练，让不同自由度的机械臂末端到达随机目标；目前已跑通 **7 个环境（含真实网格版与抓取版）、5 个已训模型**，并提供评估、曲线、演示视频与实时查看器。

> **末端执行器已更换**：Pro7 抓取场景的平行两指夹爪已被**灵心巧手 LinkerHand L20 灵巧手**替换。
> 厂商 URDF 与 22 个 STL 放在 `assets/linkerhand_l20/`，`convert_hand_urdf.py` 把 URDF 转成 MuJoCo
> 片段并**用仿真标定抓取**（抓取中心 + 闭合量）；21 个关节按"开合协同"驱动，动作空间仍是
> 「7 臂力矩 + 1 抓握指令」，因此既有 checkpoint 形状不变。下文凡提到"夹爪 / 两指 / 钳口"的
> 段落，都是更换之前的历史实测结论，保留作调参依据。

---

## 1. 现状总览

| 环境 | 自由度 / 运动 | 模型 | 整集命中率 | 终距 中位 | 平均到达步数 |
|---|---|---|---|---|---|
| `two_joint` | 2D 平面两连杆（肩+肘） | `results/ppo_two_joint.zip` | **100%** | 0.057 | ~27 (of 120) |
| `three_joint` | 3D 空间三自由度（偏航+肩+肘） | `results/ppo_three_joint.zip` | **98.5%** | 0.091 | ~31 (of 150) |
| `six_joint` | 珞石 xMate ER3 六自由度（真实几何） | `results/ppo_six_joint.zip` | **~48%** | 0.129 | ~130 (of 200) |
| `pro7_joint` | 珞石 xMate Pro7 七自由度（真实几何） | `results/ppo_pro7_joint.zip` | **~89.5%** | 0.147 | ~61 (of 250) |
| `pro7_urdf` | 珞石 xMate Pro7 七自由度（**真实 STL 网格**） | `results/ppo_pro7_urdf.zip` | **84.2%** | 0.147 | ~61 (of 250) |

上述环境共用同一套训练/评估/可视化代码，用 `--env <name>` 切换；全部在 RTX 5080 + CUDA 上训练，速度约 3500–4000 步/秒（`pro7_urdf` 2.0M 步约 8.4 分钟）。

---

## 2. 分层架构

```text
物理模型 XML (assets/*.xml)
   │  mujoco.MjModel.from_xml_path()
   ▼
Gymnasium 环境 (env/*_reacher.py，由 env/__init__.py 的 make_env 工厂按名创建)
   │
   ▼
PPO 训练 (train_ppo.py) ──► results/ppo_<env>.zip
   │
   ├──► eval_rollout.py   指标 + 学习曲线 + 演示视频（--viewer 开窗口）
   ├──► viewer_demo.py    实时 MuJoCo 窗口
   ├──► make_montage.py   逼近→命中 静态拼图
   └──► ik_probe*.py      解析 IK + PD 可行性基准（对比参照）
```

---

## 3. 文件清单（角色）

| 文件 | 作用 |
|---|---|
| `assets/two_joint_arm.xml` | 平面二连杆模型（绕 z 的肩/肘） |
| `assets/three_joint_arm.xml` | 3D 三自由度空间臂（偏航+肩+肘） |
| `assets/rokae_xmate_er3.xml` | 珞石 xMate ER3 六自由度臂（官方 URDF 真实几何） |
| `assets/rokae_xmate_pro7.xml` | 珞石 xMate Pro7 七自由度臂（官方 URDF 关节轴/限位/offset） |
| `assets/rokae_xmate_pro7_real.xml` | **真实 Pro7 URDF STL 网格**（官方几何）的 7 轴到达模型（带电机、mocap 目标） |
| `assets/rokae_xmate_pro7_pick.xml` | 抓取场景的**旧胶囊体版**（已不再注册环境，保留作低配对照） |
| `assets/rokae_xmate_pro7_pick_real.xml` | **抓取/取放场景（当前默认）**：真实 Pro7 网格 + 夹爪 + 相机 + 红块 + 绿/蓝/黄干扰块 + **两张工作台**（源台 + 目标台与放置垫） |
| `assets/meshes/xMatePro7/*.stl` | 官方 Pro7 STL 网格（随仓库提供，相对路径引用） |
| `env/__init__.py` | `make_env()` 注册表，`--env` 一键切换任务 |
| `env/base_reacher.py` | **所有到达任务的基类**：观测/动作/奖励/渲染/步进只实现一次（子类只给模型与采样） |
| `env/two_joint_reacher.py` | 二维到达环境（obs 10，动作 2）——`BaseReacher` 子类 |
| `env/three_joint_reacher.py` | 三维到达环境（obs 15，动作 3）——`BaseReacher` 子类 |
| `env/rokae_reacher.py` | 通用 Rokae 到达环境（按模型自动取自由度）：驱动 ER3（obs 24，动作 6）与 Pro7 子类（obs 27，动作 7） |
| `env/rokae_pro7_pick.py` | Pro7 抓取环境（obs 30，动作 8）：真实网格场景 + 干扰块，观察用 RGB-D 检测到的红块 3D 坐标驱动 |
| `grasp_common.py` | **抓取场景与专家的唯一来源**：台面/红块/**干扰块**/夹爪常量、场景构建、专家伺服、抓取判据 |
| `grasp_policy.py` | 抓取 MLP 策略（`Policy`/`OBS_DIM`/`ACT_DIM`/`load_policy`/`rollout`），训练与回放共用 |
| `cli.py` | 统一 `--env/--model/--out` 命令行助手与 `load_policy()` |
| `detect_red_cube.py` | 红色像素分割 + RGB-D 深度反投影 → 红块世界 3D 坐标（平面吸附，误差 ~6 mm） |
| `grasp_demo.py` | 端到端演示：检测红块 → 3D 坐标 → 任务空间伺服 → 七轴夹爪抓取 |
| `pick_place_demo.py` | **取放演示**：源台抓取 → 提起 → 搬运 → 放到目标台的放置垫（含 GIF / 拼图） |
| `make_grasp_montage.py` | 生成“逼近→抓住”静态拼图 |
| `supervised_grasp.py` | 实时监督（DAgger 行为克隆）训练：专家老师实时标注 (观测, 动作)，在线训练策略并实时监测损失/抓取成功率 |
| `detect_overlay.py` | 相机检测可视化：红块分割/质心/定位 3D 点回投影 + 检测点 vs 真值 3D 对比 |
| `visualize_grasp.py` | 抓取回放 GIF（专家/已学策略驱动，带绿色检测标记点） |
| `view_pick.py` | 实时 MuJoCo 窗口：实时看七轴臂抓取（专家/策略），绿色标记实时显示检测点 |
| `train_live.py` | **实时训练过程可视化**：默认训练**取放整段**，每个评估轮渲染“当前策略 接近→夹紧→提起→搬运→放置 故事板 + 实时指标面板（pick % / place %）”，逐帧合成 `training_process.gif`；`--task grasp` 回到抓取任务 |
| `train_ppo.py` | PPO 训练入口（向量化环境、回调统计、续训） |
| `eval_rollout.py` | 量化解算 + 学习曲线 + 演示视频 |
| `viewer_demo.py` | 实时 MuJoCo 可视化窗口（关窗停止，`T` 换目标） |
| `make_montage.py` | 逼近→命中 静态拼图 |
| `ik_probe.py` / `ik_probe3d.py` | 二连杆 / 3R 解析 IK + PD 可行性基线 |
| `paths.py` | 项目路径/默认产物位置集中管理 + 模型路径校验与友好报错 |
| `live_viewer.py` | **实时可视化窗口**（节流 + 相机 + 窗口内状态文字 + 安全关窗），训练与演示共用 |
| `tests/` | pytest 冒烟 + 回归测试：观测维度/顺序（checkpoint ABI）、路径、抓取契约 |
| `requirements.txt` | 依赖清单（mujoco / gymnasium / sb3 / torch / …） |
| `README.md` | 快速上手 |
| `PROJECT_GUIDE.md` | 架构 + 逐文件深度说明 |
| `PROJECT_SUMMARY.md` | 本文档：当前状态总结 |
| `results/` | 训练产物（模型/日志/曲线/视频/TensorBoard） |

---

## 4. 关键设计决策

- 奖励主项 = `-距离`（稠密），配合命中 `+2`；六自由度另加 `0.5·exp(-dist/0.3)` 的“接近度”塑形，让远距离也有连续梯度。
- 关节角用 `cos/sin` 进观测，消除周期歧义。
- 目标用 `mocap` 无碰撞体；六自由度目标由随机位姿的正运动学(FK) 得到，天然保证可达。
- 物理上保证臂不与任何物体接触（曾因肩/肘球体嵌入地面锁死基座）。
- 3D / 6D 均设零重力（纯惯性控制），保证 PPO 可收敛。
- 六自由度工具尖点偏离腕滚转轴，使 6 个关节都影响末端位置。
- 提供解析 IK/PD 基线，用来区分“算法没学会”还是“任务不可行”。
- 可维护性：路径/默认值集中在 `paths.py`，命令行集中在 `cli.py`，到达任务收敛到
  `env/base_reacher.py`，抓取场景与专家收敛到 `grasp_common.py`，
  并用 `tests/` 把观测维度与顺序（checkpoint ABI）钉死。
- 实时可视化：`live_viewer.py` 统一封装原生 MuJoCo 窗口（按 fps 节流、窗口四角显示
  step/fps/奖励/回合长度、关闭后安全退出），`train_ppo.py --viewer` 与
  `train_live.py --live` 都能"边训边看"；关掉窗口训练继续。

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

当前 `two_joint/`、`three_joint/`、`six_joint/`、`pro7_joint/`、`pro7_urdf/` 均已产出，
对应权重 `results/ppo_<env>.zip`；`pro7_urdf` 的训练 stdout 见 `results/logs/pro7_urdf_1.txt`。

---

## 6. 快速使用

```bash
# 回归测试（约 2 秒）
python3 -m pytest tests -q

# 训练 / 续训
python3 train_ppo.py --env three_joint --steps 600000
python3 train_ppo.py --env six_joint --steps 1000000 --init-model results/ppo_six_joint
python3 train_ppo.py --env pro7_joint --steps 2000000                          # 训练 xMate Pro7 7-DOF
python3 train_ppo.py --env pro7_urdf --steps 2000000 --viewer                  # 实时看 MuJoCo 窗口训练
python3 train_live.py --rounds 24 --live                                       # 取放整段训练 + 实时窗口（默认任务）
python3 train_live.py --task grasp --rounds 24 --live                          # 只训练抓取（0.5× 慢放）
python3 train_ppo.py --env pro7_joint --init-model results/ppo_pro7_joint      # 续训 Pro7
python3 train_ppo.py --env pro7_urdf --steps 2000000                           # 真实 URDF 网格版 Pro7 到达

# 评估 + 曲线 + 视频
python3 eval_rollout.py --env six_joint --episodes 300
python3 eval_rollout.py --env pro7_joint --episodes 400

# 实时 MuJoCo 窗口（关窗停止，T 换目标）
python3 viewer_demo.py --env six_joint
python3 viewer_demo.py --env pro7_joint

# ---- 抓取管道 ----
python3 grasp_demo.py --episodes 10 --video         # 检测红块→3D坐标→七轴抓取
python3 pick_place_demo.py --episodes 10 --video --montage   # 源台抓取→搬到目标台放置垫
python3 make_grasp_montage.py
python3 supervised_grasp.py --rounds 24            # 实时监督训练（在线 DAgger）
# 用“真实 URDF 网格”版：训练或标记驱动
python3 train_ppo.py --env pro7_pick_urdf --steps 800000

# ---- 可视化 ----
python3 detect_overlay.py                          # 检测→3D 可视化 PNG
python3 visualize_grasp.py --mode expert           # 抓取回放 GIF（专家）
python3 view_pick.py --mode expert                 # 实时 MuJoCo 窗口（专家）
python3 view_pick.py --mode policy --policy-path results/pro7_pick/grasp_policy_online.pt
python3 supervised_grasp.py --rounds 24 --visualize   # 训练后用学到的策略开实时窗口
python3 train_live.py --rounds 24                   # 取放训练过程 GIF（边训边看）
python3 train_live.py --rounds 24 --live            # 过程 GIF + 实时 MuJoCo 窗口（取放一轮 ≈ 50 s）
python3 train_live.py --rounds 24 --live --live-speed 4   # 同上，4× 播放，几分钟跑完
python3 view_pick.py --task pick_place --mode policy      # 回放学到的取放策略

# 逼近→命中拼图
python3 make_montage.py --env six_joint

# 可行性基线（2D / 3D）
python3 ik_probe.py
python3 ik_probe3d.py
```

---

## 7. 当前局限与说明

- 六自由度命中率约 48%、中位终距约 0.13 m（0-g、工具点偏离腕轴设定下）。它是三者中最难的任务：策略能把末端从 ~0.5 m 压到 ~0.13 m 并稳定进入 12 cm 圈。
- 七自由度 Pro7：同样 0-g 到达任务，但臂展 ~1.4 m、7 轴，默认把起点/目标采样与离异端距离收窄（命中圈 0.15 m、单集 250 步）以利于收敛，训练 ~2.2M 步后命中率 ~89.5%。

## 8. 抓取扩展（夹爪 + 相机 + 红块检测 → 七轴抓取）

- **场景（双工位）**：**真实 Pro7 网格**（`rokae_xmate_pro7_pick_real.xml`）+ 平行两指夹爪（手指沿 +z 伸出、沿 +x 夹紧）+ 腕部 RGB-D 相机（`cam_hand`）。**源台**（`table`）放一个自发光红色方块（边长 5 cm）与三块**同尺寸干扰块**（绿 / 蓝 / 黄，位置经实测挑选：在相机可见范围内、但完全避开红块的采样足迹，且参与碰撞）；**目标台**（`table2`）在 +y 侧，台面嵌一块**放置垫**（`place_pad`，青绿色、不碰撞、顶面与台面齐平），标出落点 `PLACE_TARGET`。`pro7_pick` 与 `pro7_pick_urdf` 指向同一个真实网格场景（后者为兼容别名）。
- **检测**：`detect_red_cube.py` 对手部相机做红色分割，取其像素深度反投影 3D，再吸附到已知桌面平面，得到红块世界坐标（均值 ~6 mm、最大 ~1.2 cm）；支持纯深度版与平面吸附版。
- **检测鲁棒性**：加入干扰块后，40 个随机红块位置仍 **100% 检出**、定位误差均值 6.3 mm（最大 11.7 mm），与无干扰时一致；红色分割只命中红块（连通域只有一个）。
- **抓取**：`grasp_demo.py` 用**任务空间分辨率伺服**（7 自由度雅可比，温和力限幅）把夹爪中心移到检测点，到位后闭合夹爪；当两指都与红块建立约束接触且夹紧时判定成功。修复夹持参考点后（见下）**20/20 成功**。
- **取放（新增）**：`pick_place_demo.py` 在同一场景上做「源台抓取 → 提起 → 搬运 → 落到目标台放置垫 → 张开」的解析专家流程（`grasp_common.pick_and_place`），实测 **10/10 成功**，落点平均误差 **29 mm**（判定阈值 50 mm），产物在 `results/pro7_pick_place/`（`pick_place_demo.gif` + `pick_place_montage.png`）。
  搬运是**准静态**的：夹持力只有约 0.3 N，专家以 1 cm 分段 + 柔增益移动（`TRANSPORT_*`）；一旦加快，方块会把活动指顶开（间隙 0.050 → 0.097，接触归零）而掉落 —— 与方块质量无关（3.75 kg 与 0.0004 kg 同样会掉）。
- **修复的根因**：`grasp_center` 站点原本在**指端**（抓手系 z=0.075），而手指内侧面只延伸到 z=0.075、其中心在 z=0.04。于是伺服把方块**顶到两指之外**（实测方块中心在 z=0.078），只剩指尖角接触——静态能"夹住"，一动就滑脱。把站点移到真正的钳口中心（0.048）后，方块落在两指之间，抓取率 16/20 → **20/20**，搬运也从 0/10 → **10/10**。`tests/test_grasp.py::test_cube_is_pinched_between_the_fingers` 钉住这个回归。
- **RL 环境**：`pro7_pick` 将检测到的红块 3D 坐标作为观测，动作 = 7 臂扭矩 + 1 夹爪开关；奖励 = 接近度 + 抓取大奖。由于 8 自由度 + 视觉观测，PPO 收敛较慢，可作为“学习式抓取”的进一步训练路径。
- **实时监督训练**：`supervised_grasp.py` 让解析伺服从 `grasp_demo` 作为专家老师，实时给出每个 (观测, 动作) 标签；策略在线用 MSE 监督学习，β 混合 DAgger（前期跟随专家、后期跟随策略但仍由专家标注）以覆盖策略自身状态分布、消除纯行为克隆的误差累积。实测 train MSE 从 0.14 降到 ~0.01，策略回放抓取成功率从 0% 升到 ~40%（默认 `--obs-target true` 使用真值目标便于学；`--obs-target vision` 换成 RGB-D 估计）。
- **取放 RL 任务（新增）**：`env/rokae_pro7_pick_place.py`（`pro7_pick_place`，obs 38 维）把**整段取放**变成一个 episode：抓取成功不再结束，只有方块落到目标台放置垫上才结束（抓丢/方块被撞出台面以 `lost=True` 提前结束，避免污染 DAgger 缓冲区）。观测量 = 抓取环境 30 维 + `holding` + 子目标误差 + 方块到垫子误差 + "到位该松手" 标志；搬运计划由 `grasp_common.PlacePlanner` 给出（settle → lift → carry → lower，1 cm 路点，≈2 cm/s 准静态）。`train_live.py --rounds 24` 用带计划的专家做 DAgger，实测 24 轮内最好一轮 **pick 100% / place 70%**（轮间波动大，脚本自动保留最好检查点），产物 `results/pro7_pick_place/training_process.gif` + `grasp_policy_online.pt`。
  > 只有**搬运**由环境执行（transport mode）：搬运力矩只占执行器量程 1~2%，远小于模仿网络 ~1.4% 的动作噪声，纯模仿的搬运必掉块（实测 2e-4 动作 MSE 的行为克隆放 0/6，脚本搬运 10/10）。策略负责接近、夹紧、失败重试与松手时机——pick % / place % 正是按这些打分。

## 9. 使用真实 URDF 几何（Pro7）

MuJoCo 3.x **原生支持读 URDF**（`MjModel.from_xml_path("*.urdf")`），但因 Pro7 原 URDF 惯性“不正确”且无电机/夹爪，直接加载需修正惯性（非对角线清零 + 三角不等式平衡）再补执行器。

项目提供两条“真实网格”路径：
- **`pro7_urdf`**：`assets/rokae_xmate_pro7_real.xml`，用官方 STL 作连杆网格，保留 URDF 关节链/限位 + 7 电机 + mocap 目标 → 到达环境真实几何版。已训练 **2.0M 步**：`results/ppo_pro7_urdf.zip`，评估 400 集命中率 **84.2%**、终距中位 0.147 m、平均 ~61 步（上限 250）；配套 `learning_curve.png` / `rollout.(mp4|gif)` / `montage.png` 在 `results/pro7_urdf/`。
- **`pro7_pick_urdf`**：`assets/rokae_xmate_pro7_pick_real.xml`，真实网格臂 + 夹爪 + 腕部相机 + 红块；RGB-D 检测→伺服抓取同样可用（实测 pin 定位误差 2–10 mm，专家 3 次抓取 2/3 成功）。

**网格已随仓库提供**：8 个官方 STL 放在 `assets/meshes/xMatePro7/`，XML 里用相对路径引用，
因此不再依赖 `/home/wj/rokae_ros2`（即使外部目录不存在也能加载，几何与原先逐项一致）。

胶囊体版（`pro7_joint`/`pro7_pick`）与已训模型 `ppo_pro7_joint.zip` 保持兼容；真实网格版外观更接近实物，物理几何来自官方 URDF。
实测胶囊版策略**零样本**迁移到真实网格模型也有 90% 命中率（两者关节链一致，仅末端 tool 偏置与 joint7 限位末位小数不同），可作为跨模型基线。
- 3D / 6D 采用零重力以保收敛；真实有重力时需更多训练或更强调控。
- 六自由度仍是位置到达任务；第 4、6 轴因工具点偏离被间接利用，但未要求姿态对齐（更贴近真实抓取需“到达+姿态”任务）。
- 几何参数来自官方 Rokae xMate ER3 URDF；惯性为按几何简化估算，非线下一一匹配。

---

## 10. 可选下一步

1. 到达 + 姿态（让第 4、6 轴有独立作用，贴近抓取/装配）。
2. 加入重力 / 真实动力学 + 域随机化（Sim-to-Real 准备）。
3. 更强算法对比：SAC / TD3 / 复现学习（behavior cloning）。
4. 控制约束任务：避障、沿轨迹跟踪、推箱/插销等操作。
5. 调参加长训练，把六自由度命中率推到更高。

---

*本文件为项目当前状态总结；实现细节见 `PROJECT_GUIDE.md`，快速上手见 `README.md`。*
