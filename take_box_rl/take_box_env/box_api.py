"""可编程的搬箱接口：不用 RL，直接写规划与约束程序。

和 Gym 环境（``take_box_env/take_box.py``）共用同一个 MuJoCo 模型，但驱动方式换成
**关节 PD 伺服 + 笛卡尔 IK**，所以可以像写运动脚本一样控制箱子：

```python
from take_box_env.take_box import TakeBoxEnv
from take_box_env.box_api import BoxCarryInterface, Waypoint

env = TakeBoxEnv()
robot = BoxCarryInterface(env)          # 约束与增益都可配
robot.home()                            # 回到 pre-grasp（IK 解好的位姿）
robot.close_hands()                     # 闭手 → 抓到箱子（掌心到位即锁存）
print(robot.state())                    # 箱位姿 / 倾角 / 手心误差 / 关节速度
robot.plan([                             # 规划一段"抬起 → 平移 → 放下"
    Waypoint(box=robot.box_pos + (0, 0, 0.12), steps=120, label="lift"),
    Waypoint(box=robot.box_pos + (0.10, 0.05, 0), yaw=0.3, steps=150, label="carry"),
    Waypoint(box=robot.box_pos + (0, 0, -0.12), steps=120, label="place"),
])
robot.open_hands()
```

约束（`BoxLimits`）在每一步之后检查，越界会抛 ``ConstraintViolation``（``strict=False``
时只记录），可以据此拒绝不安全的指令：

* 箱体倾角上限（默认 15°，对应 YAML 里双臂必须同时端平）、
* 箱体高度范围（不穿台面、不超高）、
* 掌心到箱面误差（抓着不能松脱）、
* 关节速度上限、箱体线速度上限。

单步原语 (`servo_step` / `step`) 也开放出来，方便自己写更复杂的规划器（比如带避障或力控的）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Optional, Sequence

import mujoco
import numpy as np

from .take_box import SIDES, TakeBoxEnv, palm_frame_local


class ConstraintViolation(RuntimeError):
    """指令会把箱子带出允许范围时抛出。"""

    def __init__(self, violations: Sequence[str]):
        super().__init__("; ".join(violations))
        self.violations = list(violations)


@dataclass
class BoxLimits:
    """搬箱过程的安全约束（按需覆盖）。"""

    tilt_max_deg: float = 15.0
    height_min: float = 0.30
    height_max: float = 1.05
    hand_error_max: float = 0.05
    joint_vel_max: float = 4.0
    box_speed_max: float = 0.60
    #: 指令里的偏航角上限（度）。写错单位（把度当弧度）时能提前拦住
    yaw_max_deg: float = 90.0
    #: 抓着时是否要求掌心始终贴着箱面
    keep_contact: bool = True

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class BoxState:
    """一步之后可读的完整状态。"""

    step: int
    box_pos: np.ndarray
    box_quat: np.ndarray
    yaw_deg: float
    tilt_deg: float
    lift: float
    grasped: bool
    palms: np.ndarray
    hand_errors: np.ndarray
    grip: np.ndarray
    joint_vel_max: float
    box_speed: float

    def summary(self) -> str:
        return (
            f"step {self.step:4d} | box ({self.box_pos[0]:+.3f}, {self.box_pos[1]:+.3f}, "
            f"{self.box_pos[2]:+.3f}) | yaw {self.yaw_deg:+6.1f}° tilt {self.tilt_deg:5.1f}° | "
            f"lift {self.lift:+.3f} m | hand err {self.hand_errors.max() * 1000:5.1f} mm | "
            f"grasp {'Y' if self.grasped else 'N'} | |dq|max {self.joint_vel_max:.2f}"
        )


@dataclass
class Waypoint:
    """一个目标点：箱心位置 + 偏航角（+ 可选的抓取通道与阶段名）。"""

    box: Sequence[float]
    yaw: float = 0.0
    steps: int = 100
    grip: Optional[float] = None
    label: str = ""


class BoxCarryInterface:
    """关节/笛卡尔伺服 + 规划 + 约束检查。"""

    def __init__(
        self,
        env: TakeBoxEnv,
        limits: Optional[BoxLimits] = None,
        kp: float = 4.0,
        kd: float = 1.2,
        strict: bool = True,
        joint_rate: float = 1.5,
        ik_reg: float = 0.05,
    ):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.limits = limits or BoxLimits()
        self.kp = kp
        self.kd = kd
        self.strict = strict
        #: 每个控制步允许的关节目标变化量（rad/步），把 IK 的大跳变磨平
        self.joint_rate = joint_rate
        #: IK 关节正则强度（越大越贴着种子解，越小越"自由"）
        self.ik_reg = ik_reg
        self.violation_log: list = []
        self._skip_speed_once = False

        self._frames = {side: self._hand_frame(side) for side in SIDES}
        self._grip = {side: 0.0 for side in SIDES}
        self._box_speed = 0.0
        self.steps = 0

    @property
    def box_pos(self) -> np.ndarray:
        """当前箱心位置（规划里直接用它做起点）。"""
        return self.data.mocap_pos[0].copy()

    @property
    def ready_box_pos(self) -> np.ndarray:
        """抓取位姿对应的箱心位置。"""
        return np.asarray(self.env.box_home, dtype=float).copy()

    # ------------------------------------------------------------------ #
    # 低层：手部坐标系 / IK / PD
    # ------------------------------------------------------------------ #
    def _hand_frame(self, side: str) -> tuple:
        """掌心在腕部坐标系下的位置，以及"手指/张开/掌法线"三轴。"""
        env = self.env
        arm = env.config["robot"]["arms"][side]
        centre, source = palm_frame_local(
            self.model, self.data, arm["wrist_body"], arm["finger_prefix"])
        # 与 build_cell_model.dual_arm_grasp_targets() 同一套约定：掌心法线朝箱面，
        # 再绕法线滚转 task.grasp_roll_deg（这个工位够不到"手指朝前"的滚转角，
        # 见该函数的注释）。两只手用同一个目标旋转矩阵。
        roll = np.radians(env.config["task"].get("grasp_roll_deg", 30.0))
        c, s = np.cos(roll), np.sin(roll)
        roll_about_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        target = roll_about_y @ np.column_stack(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        rel = target @ source.T
        assert np.linalg.det(rel) > 0, "掌心目标姿态必须是旋转矩阵"
        return centre, source, rel

    def palm_targets(self, box_pos: np.ndarray, yaw: float = 0.0):
        """箱位姿 → 两侧掌心应有的世界位姿。"""
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        targets = {}
        for side in SIDES:
            _centre, _source, rel = self._frames[side]
            # 用建模型时量出来的掌心偏移（掌面贴着箱面），而不是箱面本身
            position = np.asarray(box_pos) + rotation @ self.env.palm_offsets[side]
            targets[side] = (position, rotation @ rel)
        return targets

    def ik(self, side: str, position, rotation, seed=None, iters: int = 400):
        """单臂 6 自由度 IK（先位置后姿态），返回关节角。

        关节正则 ``ik_reg`` 把解拉回种子附近：没有它，稍微改一点 yaw 就可能让
        求解器跳到腕部翻转的那一支，箱子朝向会瞬间差 180°。
        """
        env = self.env
        site = env.grasp_site[side]
        qpos_idx = env.qpos_idx[side]
        dof_idx = env.dof_idx[side]
        limits = env.limits[side]
        # IK 是"试算"：结束后必须把仿真状态原样还原，否则第一次规划就会
        # 把机器人瞬移到求解器最后一步的姿态（表现为速度/倾角尖峰）。
        saved_qpos = self.data.qpos.copy()
        q = np.asarray(seed if seed is not None else self.data.qpos[qpos_idx],
                       dtype=float).copy()
        seed = q.copy()
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        # 姿态权重不能太小：这个任务的核心约束是"箱子保持水平"，位置差几毫米
        # 无所谓，掌心姿态差 20° 就会让箱子倾斜。
        rot_weight = 0.8

        def position_pass(q, gain=0.5):
            for _ in range(iters):
                self.data.qpos[qpos_idx] = q
                mujoco.mj_fwdPosition(self.model, self.data)
                error = np.asarray(position) - self.data.site_xpos[site]
                if np.linalg.norm(error) < 1e-3:
                    break
                mujoco.mj_jacSite(self.model, self.data, jacp, None, site)
                jac = jacp[:, dof_idx]
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(3), error)
                dq = dq + self.ik_reg * (seed - q)
                q = np.clip(q + gain * dq, limits[:, 0], limits[:, 1])
            return q

        q = position_pass(q)
        # 综合分 = |位置误差| + rot_weight*|姿态误差|：本工位腕部滚转顶在限位上，
        # 位置和姿态不可能同时满足，实测"姿态优先"才是对的那个（箱子端平 0.1°，
        # 代价是抬起高度只能到指令的 ~75%）——想要位置更准就得放宽腕关节限位。
        best = (np.inf, q.copy())
        stalled = 0
        for _ in range(iters):
            self.data.qpos[qpos_idx] = q
            mujoco.mj_fwdPosition(self.model, self.data)
            e_pos = np.asarray(position) - self.data.site_xpos[site]
            current = self.data.site_xmat[site].reshape(3, 3)
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, (np.asarray(rotation) @ current.T).reshape(-1))
            e_rot = np.zeros(3)
            mujoco.mju_quat2Vel(e_rot, quat, 1.0)
            score = np.linalg.norm(e_pos) + rot_weight * np.linalg.norm(e_rot)
            if score < best[0]:
                best = (score, q.copy())
                stalled = 0
            else:
                # 局部收敛就收手：右手腕滚转顶在限位上时够不到目标姿态，
                # 否则每一步都要白跑满 400 次迭代
                stalled += 1
                if stalled >= 40:
                    break
            if np.linalg.norm(e_pos) < 1e-3 and np.linalg.norm(e_rot) < 0.05:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site)
            jac = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
            weighted = np.concatenate([e_pos, rot_weight * e_rot])
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), weighted)
            dq = dq + self.ik_reg * (seed - q)
            q = np.clip(q + 0.35 * dq, limits[:, 0], limits[:, 1])
            if np.linalg.norm(np.asarray(position) - self.data.site_xpos[site]) > 0.02:
                q = position_pass(q, gain=0.6)
        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return best[1]

    def _write_torque(self, side: str, q_target: np.ndarray) -> None:
        """关节 PD → 归一化力矩（模型用 motor，gear 已含在模型里）。"""
        env = self.env
        q = self.data.qpos[env.qpos_idx[side]]
        dq = self.data.qvel[env.dof_idx[side]]
        # 限速：目标离当前位置太远时先走一小步，避免速度尖峰和姿态甩飞
        limited = q + np.clip(np.asarray(q_target) - q, -self.joint_rate, self.joint_rate)
        gear = self.model.actuator_gear[env.act_idx[side], 0]
        torque = self.kp * (limited - q) - self.kd * dq
        self.data.ctrl[env.act_idx[side]] = np.clip(torque / gear, -1.0, 1.0)

    def servo_step(self, q_target: dict, grip: Optional[dict] = None) -> BoxState:
        """推进一步：关节 PD +（可选）抓取通道。自己写规划器时用这个。"""
        for side in SIDES:
            self._write_torque(side, np.asarray(q_target[side]))
            if grip is not None and grip.get(side) is not None:
                self._grip[side] = float(np.clip(grip[side], 0.0, 1.0))
            low, high = self.env._hand_ctrlrange(side)
            self.data.ctrl[self.env.hand_act[side]] = low + self._grip[side] * (high - low)

        previous = self.data.mocap_pos[0].copy()
        mujoco.mj_step(self.model, self.data)
        self.steps += 1
        if self.env.grasped:
            self.env._carry_box()
            self._box_speed = float(
                np.linalg.norm(self.data.mocap_pos[0] - previous) / self.model.opt.timestep
            )
        else:
            self._box_speed = 0.0
        self._check()
        return self.state()

    # ------------------------------------------------------------------ #
    # 状态与约束
    # ------------------------------------------------------------------ #
    def state(self) -> BoxState:
        env = self.env
        palms = env.palm_positions()
        errors = env.hand_errors()
        # 箱体姿态直接读箱体自己：抓取时它带着"抓取瞬间的相对位姿"跟着手走，
        # 不再等于掌心坐标系，倾角因此就是箱子相对水平面的真实倾角。
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, self.data.mocap_quat[0])
        rotation = rotation.reshape(3, 3)
        yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        return BoxState(
            step=self.steps,
            box_pos=self.data.mocap_pos[0].copy(),
            box_quat=self.data.mocap_quat[0].copy(),
            yaw_deg=yaw,
            tilt_deg=float(tilt),
            lift=float(self.data.mocap_pos[0][2] - env.box_home[2]),
            grasped=bool(env.grasped),
            palms=palms,
            hand_errors=errors,
            grip=np.array([self._grip[side] for side in SIDES]),
            # 只看臂关节：手指位置伺服闭合时速度天然很大，不代表臂在甩
            joint_vel_max=float(np.abs(np.concatenate(
                [self.data.qvel[self.env.dof_idx[side]] for side in SIDES]
            )).max()),
            box_speed=self._box_speed,
        )

    def violations(self, state: Optional[BoxState] = None) -> list:
        """返回当前状态违反的约束（空列表 = 安全）。"""
        s = state or self.state()
        limits = self.limits
        bad = []
        if s.tilt_deg > limits.tilt_max_deg:
            bad.append(f"tilt {s.tilt_deg:.1f}° > {limits.tilt_max_deg:.1f}°")
        if not (limits.height_min <= s.box_pos[2] <= limits.height_max):
            bad.append(f"height {s.box_pos[2]:.3f} outside "
                       f"[{limits.height_min:.2f}, {limits.height_max:.2f}]")
        if s.grasped and limits.keep_contact and s.hand_errors.max() > limits.hand_error_max:
            bad.append(f"hand error {s.hand_errors.max() * 1000:.1f} mm > "
                       f"{limits.hand_error_max * 1000:.0f} mm")
        if s.joint_vel_max > limits.joint_vel_max:
            bad.append(f"|dq| {s.joint_vel_max:.2f} > {limits.joint_vel_max:.2f} rad/s")
        if s.box_speed > limits.box_speed_max:
            bad.append(f"box speed {s.box_speed:.2f} > {limits.box_speed_max:.2f} m/s")
        return bad

    def _check(self) -> None:
        if self._skip_speed_once:
            # 抓取锁存那一步箱子是"瞬移"到掌心坐标系的，不是真实速度
            self._box_speed = 0.0
            self._skip_speed_once = False
        bad = self.violations()
        if not bad:
            return
        self.violation_log.append((self.steps, list(bad)))
        if self.strict:
            raise ConstraintViolation(bad)

    def check_command(self, waypoint: "Waypoint") -> list:
        """只校验**指令**本身是否越界（不推进物理），用于规划阶段提前拦截。"""
        bad = []
        target = np.asarray(waypoint.box)
        if not (self.limits.height_min <= target[2] <= self.limits.height_max):
            bad.append(f"target height {target[2]:.3f} outside limits")
        if abs(np.degrees(waypoint.yaw)) > self.limits.yaw_max_deg:
            bad.append(f"target yaw {np.degrees(waypoint.yaw):.1f}° > "
                       f"{self.limits.yaw_max_deg:.0f}°")
        return bad

    # ------------------------------------------------------------------ #
    # 高层原语
    # ------------------------------------------------------------------ #
    def home(self, steps: int = 60, grip: float = 0.0) -> BoxState:
        """回到 IK 解好的 pre-grasp 位姿。"""
        q = {side: self.env.ready_qpos[side] for side in SIDES}
        return self._track(q, steps=steps, grip={side: grip for side in SIDES})

    def close_hands(self, steps: int = 40) -> BoxState:
        """闭手；掌心已贴到箱面就锁存为"抓住"。"""
        state = self.home(steps=1)
        for _ in range(steps):
            state = self.servo_step(
                {side: self.env.ready_qpos[side] for side in SIDES},
                grip={side: 1.0 for side in SIDES},
            )
            if not self.env.grasped:
                if self.env.grasp_ready():
                    self.env.grasped = True
                    # 记下"抓到这一刻"箱子在掌心坐标系里的相对位姿，之后箱子
                    # 跟着手做刚体运动，不会在接触瞬间瞬移/翻转
                    self.env._capture_box_offset()
                    self._box_speed = 0.0
                    self._skip_speed_once = True
        return state

    def open_hands(self, steps: int = 30, hold: bool = False) -> BoxState:
        state = self.state()
        if self.env.grasped and not hold:
            self.env.grasped = False
        return self._track(
            {side: self.data.qpos[self.env.qpos_idx[side]] for side in SIDES},
            steps=steps, grip={side: 0.0 for side in SIDES},
        )

    def goto_box_pose(self, box, yaw: float = 0.0, steps: int = 100,
                      grip: Optional[float] = None,
                      settle_steps: int = 30) -> BoxState:
        """把箱子（也就是两只手）移动到指定位姿：IK + PD 逐段插值。"""
        target = np.asarray(box, dtype=float)
        q_start = {side: self.data.qpos[self.env.qpos_idx[side]] for side in SIDES}
        q_goal = {}
        for side, (position, rotation) in self.palm_targets(target, yaw).items():
            q_goal[side] = self.ik(side, position, rotation,
                                   seed=q_start[side])
        state = self._track_between(q_start, q_goal, steps, grip)
        # 收敛段：目标不动，等 PD 把残差吃掉（否则"命令到位"≠"实际到位"）
        for _ in range(settle_steps):
            state = self.servo_step(q_goal, grip=None if grip is None
                                    else {side: grip for side in SIDES})
        return state

    def _track(self, q_goal: dict, steps: int, grip: Optional[dict] = None) -> BoxState:
        start = {side: self.data.qpos[self.env.qpos_idx[side]] for side in SIDES}
        state = self.state()
        for step in range(steps):
            alpha = (step + 1) / steps
            target = {
                side: (1 - alpha) * start[side] + alpha * np.asarray(q_goal[side])
                for side in SIDES
            }
            state = self.servo_step(target, grip=grip)
        return state

    def _track_between(self, q_start, q_goal, steps, grip):
        state = self.state()
        for step in range(steps):
            alpha = (step + 1) / steps
            q_target = {
                side: (1 - alpha) * np.asarray(q_start[side]) + alpha * np.asarray(q_goal[side])
                for side in SIDES
            }
            state = self.servo_step(
                q_target,
                grip=None if grip is None else {side: grip for side in SIDES},
            )
        return state

    # ------------------------------------------------------------------ #
    # 轨迹规划
    # ------------------------------------------------------------------ #
    def plan(self, waypoints: Iterable[Waypoint],
             on_step: Optional[Callable[[BoxState], None]] = None) -> list:
        """校验并执行一串路点，返回每步状态（可直接画图/存日志）。"""
        trace = []
        for waypoint in waypoints:
            bad = self.check_command(waypoint)
            if bad:
                raise ConstraintViolation(bad)
            state = self.state()
            print(f"[plan] {waypoint.label or 'waypoint':>10s} -> "
                  f"({waypoint.box[0]:+.3f}, {waypoint.box[1]:+.3f}, {waypoint.box[2]:+.3f}) "
                  f"yaw {np.degrees(waypoint.yaw):+.0f}° in {waypoint.steps} steps")
            start_pos = np.asarray(state.box_pos, dtype=float)
            start_yaw = state.yaw_deg
            for step in range(waypoint.steps):
                alpha = (step + 1) / waypoint.steps
                # 位置和偏航用同一条插值：都从**这一段开始**时的状态出发
                target = start_pos * (1 - alpha) + np.asarray(waypoint.box) * alpha
                yaw = start_yaw * (1 - alpha) + np.degrees(waypoint.yaw) * alpha
                q_start = {side: self.data.qpos[self.env.qpos_idx[side]] for side in SIDES}
                q_goal = {}
                for side, (position, rotation) in self.palm_targets(
                    target, np.radians(yaw)
                ).items():
                    q_goal[side] = self.ik(side, position, rotation, seed=q_start[side])
                step_state = self.servo_step(
                    q_goal,
                    grip=None if waypoint.grip is None
                    else {side: waypoint.grip for side in SIDES},
                )
                trace.append(step_state)
                if on_step is not None:
                    on_step(step_state)
        print(f"[plan] done, {len(trace)} steps, "
              f"violations {len(self.violation_log)}")
        return trace
