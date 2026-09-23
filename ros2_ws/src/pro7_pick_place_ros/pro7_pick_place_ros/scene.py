"""The MuJoCo plant the node drives: model, data, renderers and snapshots.

One :class:`GraspScene` owns everything MuJoCo-related (the Pro7 + LinkerHand
L20 model, the physics state, the eye-in-hand RGB-D renderer).  It is a thin,
thread-safe wrapper around :mod:`grasp.common`: the scene layout, the vision
helper and the expert all stay in the project, so the node can never drift from
the demo scripts.

Every public method takes the scene's re-entrant lock, which is also what lets
the simulator thread hold the plant still while it runs the expert one control
step at a time (see :mod:`pro7_pick_place_ros.simulator`).

The RGB-D renderers are OpenGL objects: they are created, used *and* destroyed
on a single thread.  The node therefore builds the scene with
``defer_renderers=True`` and lets the plant thread call :meth:`build_renderers`,
:meth:`detect`, :meth:`rgbd` and :meth:`reset` -- mixing those across threads
segfaults the GL context.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

#: Fixed frame the whole scene is expressed in (the MuJoCo world frame).
WORLD_FRAME = "world"
#: Frame of the closed fingertips (the point the expert servos onto the cube).
TOOL_FRAME = "tool0"
#: Eye-in-hand camera in MuJoCo's own convention (x right, y up, -z = view).
CAMERA_FRAME = "cam_hand"
#: The same camera in the ROS optical convention (x right, y down, z = view).
CAMERA_OPTICAL_FRAME = "cam_hand_optical"
#: Camera name inside the MuJoCo model.
CAMERA_NAME = "cam_hand"
#: The seven arm joints, in model order.
ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
#: Bodies published as the TF chain ``world -> base -> link1..7 -> gripper``.
TF_BODIES = ("base", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "gripper")
#: Rigid rotation from MuJoCo camera axes to the ROS optical convention.
_CAMERA_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class FramePose:
    """A child frame expressed in its parent frame (translation + wxyz quat)."""

    child: str
    parent: str
    position: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class SceneState:
    """Immutable snapshot of the plant, safe to hand to another thread."""

    scene_id: int
    sim_time: float
    joint_names: Tuple[str, ...]
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    arm_qpos: np.ndarray
    arm_qvel: np.ndarray
    hand_closure: float
    gripper_gap: float
    cube_position: np.ndarray
    grasp_position: np.ndarray
    place_target: np.ndarray
    place_error: float
    holding: bool
    detected: Optional[np.ndarray]
    frames: Tuple[FramePose, ...]


def _quat_from_mat(mat: np.ndarray) -> np.ndarray:
    """Rotation matrix (3x3, row-major) -> unit quaternion ``(w, x, y, z)``."""
    m = np.asarray(mat, dtype=float).reshape(3, 3)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s,
             (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s,
             (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s,
             0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s,
             (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    quat = np.asarray(q, dtype=float)
    return quat / np.linalg.norm(quat)


def _relative_pose(parent, child) -> Tuple[np.ndarray, np.ndarray]:
    """Express a world pose ``(position, rotation 3x3)`` in another world pose."""
    pos_p, rot_p = parent
    pos_c, rot_c = child
    return rot_p.T @ (pos_c - pos_p), _quat_from_mat(rot_p.T @ rot_c)


class GraspScene:
    """MuJoCo model + data + renderers of the two-bench pick & place cell."""

    def __init__(
        self,
        gc,
        *,
        model_path: Optional[str] = None,
        seed: int = 0,
        randomize_cube: bool = False,
        place_target=None,
        image_width: int = 320,
        image_height: int = 240,
        defer_renderers: bool = False,
    ):
        self.gc = gc
        self.model_path = model_path or gc.MODEL_PATH
        self.image_size = (int(image_width), int(image_height))
        self.lock = threading.RLock()
        self._seed = int(seed)
        self._randomize_cube = bool(randomize_cube)
        self._place_target = np.asarray(
            gc.PLACE_TARGET if place_target is None else place_target, dtype=float
        )
        self._scene_id = 0
        self._detected: Optional[np.ndarray] = None
        self._detect_renderer: Optional[mujoco.Renderer] = None
        self._image_renderer: Optional[mujoco.Renderer] = None
        self._renderer_thread: Optional[int] = None
        self._with_renderers = not defer_renderers
        self._build()

    # ------------------------------------------------------------------ #
    # construction / reset
    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        """(Re)build the MuJoCo state and the renderers."""
        self.close_renderers()
        self.model, self.data, self.cube_start = self.gc.make_scene(
            seed=self._seed, model_path=self.model_path
        )
        self.ids = self.gc.scene_ids(self.model)
        self._resolve_joints()
        self._detected = None
        self._apply_place_target()
        if self._with_renderers:
            self._open_renderers()
        self._scene_id += 1

    def _open_renderers(self) -> None:
        width, height = self.image_size
        if self._detect_renderer is None:
            self._detect_renderer = mujoco.Renderer(self.model, height=height, width=width)
        if self._image_renderer is None:
            self._image_renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer_thread = threading.get_ident()

    def _assert_renderer_thread(self) -> None:
        """Turn "GL context touched from the wrong thread" into a clear error.

        MuJoCo's renderer wraps an OpenGL context: creating, using or destroying
        it from a thread other than its owner crashes the process instead of
        raising, so the node funnels every render through the plant thread
        (:class:`pro7_pick_place_ros.simulator.PickPlaceSimulator`) and this
        guard catches the mistake early.
        """
        if self._renderer_thread is not None and self._renderer_thread != threading.get_ident():
            raise RuntimeError(
                "MuJoCo renderers belong to the thread that created them: drive "
                "the scene through PickPlaceSimulator instead of calling "
                "reset()/detect()/rgbd() from another thread"
            )

    def build_renderers(self) -> None:
        """Create the RGB-D renderers (once) on the calling thread.

        Call this from the thread that will keep rendering: the GL contexts
        belong to it.
        """
        with self.lock:
            self._with_renderers = True
            if self._detect_renderer is None:
                self._open_renderers()

    @property
    def renderers_ready(self) -> bool:
        with self.lock:
            return self._detect_renderer is not None and self._image_renderer is not None

    def _resolve_joints(self) -> None:
        """Cache joint names and their qpos/qvel addresses (free joints excluded)."""
        names, qpos, qvel = [], [], []
        for joint in range(self.model.njnt):
            if self.model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            names.append(mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint))
            qpos.append(self.model.jnt_qposadr[joint])
            qvel.append(self.model.jnt_dofadr[joint])
        self.joint_names = tuple(names)
        self.joint_qpos = np.asarray(qpos, dtype=int)
        self.joint_dof = np.asarray(qvel, dtype=int)
        arm_qpos, arm_dof = [], []
        for name in ARM_JOINT_NAMES:
            joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise RuntimeError(f"joint {name!r} missing from {self.model_path}")
            arm_qpos.append(self.model.jnt_qposadr[joint])
            arm_dof.append(self.model.jnt_dofadr[joint])
        self.arm_qpos_addr = np.asarray(arm_qpos, dtype=int)
        self.arm_dof_addr = np.asarray(arm_dof, dtype=int)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
        )
        if self.camera_id < 0:
            raise RuntimeError(f"camera {CAMERA_NAME!r} missing from {self.model_path}")

    def _apply_place_target(self) -> None:
        """Park the (decorative) drop-off pad under the node's place target.

        The pad is a non-colliding geom, so moving it only keeps the rendered /
        marked drop point honest when ``place_target`` is overridden.
        """
        pad = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "place_pad")
        if pad < 0:
            return
        self.model.geom_pos[pad][:2] = self._place_target[:2]
        mujoco.mj_forward(self.model, self.data)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        randomize_cube: Optional[bool] = None,
        cube_xyz=None,
        place_target=None,
    ) -> int:
        """Re-build the cell; returns the new ``scene_id``.

        The cube always starts on the spot sampled from the scene's seed, so a
        reset with the same seed is reproducible.  ``randomize_cube=True``
        without an explicit ``seed`` advances the seed by one, which is how a
        goal asks for "a different cube, please"; ``cube_xyz`` pins it exactly.
        """
        with self.lock:
            if self._detect_renderer is not None:
                self._assert_renderer_thread()
            if randomize_cube is None:
                randomize_cube = self._randomize_cube
            if seed is not None:
                self._seed = int(seed)
            elif randomize_cube:
                self._seed += 1
            self._randomize_cube = bool(randomize_cube)
            if place_target is not None:
                self._place_target = np.asarray(place_target, dtype=float)
            cube = None
            if cube_xyz is not None:
                cube = np.asarray(cube_xyz, dtype=float)
            self._build()
            if cube is not None:
                self.gc.place_cube(self.data, self.ids, cube)
                self.cube_start = cube
                mujoco.mj_forward(self.model, self.data)
            return self._scene_id

    # ------------------------------------------------------------------ #
    # plant access
    # ------------------------------------------------------------------ #
    @property
    def detect_renderer(self) -> mujoco.Renderer:
        """The RGB-D renderer the expert uses (must be held under the lock)."""
        return self._detect_renderer

    @property
    def model_path_used(self) -> str:
        return self.model_path

    @property
    def seed(self) -> int:
        with self.lock:
            return self._seed

    @property
    def place_target(self) -> np.ndarray:
        with self.lock:
            return self._place_target.copy()

    def set_place_target(self, xyz) -> np.ndarray:
        """Move the drop-off point (and the visual pad) under the lock."""
        with self.lock:
            self._place_target = np.asarray(xyz, dtype=float)
            self._apply_place_target()
            return self._place_target.copy()

    def step(self) -> None:
        """Advance the physics one model timestep with the current controls."""
        with self.lock:
            mujoco.mj_step(self.model, self.data)

    def state(self) -> SceneState:
        """Snapshot of the plant (copies, safe to use after the lock is released)."""
        with self.lock:
            gc = self.gc
            cube = gc.cube_world(self.data, self.ids)
            grasp = gc.grasp_world(self.data, self.ids)
            return SceneState(
                scene_id=self._scene_id,
                sim_time=float(self.data.time),
                joint_names=self.joint_names,
                joint_positions=np.array(self.data.qpos[self.joint_qpos]),
                joint_velocities=np.array(self.data.qvel[self.joint_dof]),
                arm_qpos=np.array(self.data.qpos[self.arm_qpos_addr]),
                arm_qvel=np.array(self.data.qvel[self.arm_dof_addr]),
                hand_closure=gc.hand_closure(self.data, self.ids),
                gripper_gap=gc.gripper_gap(self.data, self.ids),
                cube_position=np.array(cube),
                grasp_position=np.array(grasp),
                place_target=self._place_target.copy(),
                place_error=float(np.linalg.norm(cube - self._place_target)),
                holding=gc.holding_cube(self.data, self.ids),
                detected=None if self._detected is None else np.array(self._detected),
                frames=self._frames(),
            )

    def holding(self) -> bool:
        with self.lock:
            return self.gc.holding_cube(self.data, self.ids)

    # ------------------------------------------------------------------ #
    # vision
    # ------------------------------------------------------------------ #
    def detect(self) -> Optional[np.ndarray]:
        """Refresh (and return) the RGB-D estimate of the red cube."""
        with self.lock:
            if self._detect_renderer is None:
                raise RuntimeError("renderers are not built yet (see build_renderers)")
            self._assert_renderer_thread()
            estimate = self.gc.detect_cube(self.model, self.data, self._detect_renderer)
            if estimate is not None:
                self._detected = np.asarray(estimate, dtype=float)
            return None if self._detected is None else self._detected.copy()

    def rgbd(self):
        """Render ``(colour, depth)`` of the eye-in-hand camera for publishing."""
        from grasp.detect import render_rgbd

        with self.lock:
            if self._image_renderer is None:
                raise RuntimeError("renderers are not built yet (see build_renderers)")
            self._assert_renderer_thread()
            return render_rgbd(self._image_renderer, self.data, CAMERA_NAME)

    def camera_info(self):
        """``(fovy_deg, width, height, fx, fy, cx, cy)`` of the hand camera."""
        from grasp.detect import intrinsics

        with self.lock:
            height, width = self.image_size[1], self.image_size[0]
            fovy = float(self.model.cam_fovy[self.camera_id])
            fx, fy, cx, cy = intrinsics(fovy, height, width)
            return fovy, width, height, fx, fy, cx, cy

    # ------------------------------------------------------------------ #
    # frames
    # ------------------------------------------------------------------ #
    def _frames(self) -> Tuple[FramePose, ...]:
        """World-anchored TF chain of the arm plus the tool / camera frames."""
        with self.lock:
            model, data = self.model, self.data
            world = {}
            for name in TF_BODIES:
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body < 0:
                    continue
                world[name] = (
                    np.array(data.xpos[body]),
                    np.array(data.xmat[body]).reshape(3, 3),
                )

            frames = []
            for name in TF_BODIES:
                if name not in world:
                    continue
                body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                parent = mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body])
                )
                if parent in world:
                    parent_pose = world[parent]
                else:
                    parent = WORLD_FRAME
                    parent_pose = (np.zeros(3), np.eye(3))
                position, quaternion = _relative_pose(parent_pose, world[name])
                frames.append(FramePose(name, parent, position, quaternion))

            # Tool frame: the grasp site, i.e. what the expert servos.
            if "gripper" in world:
                tool = (
                    np.array(data.site_xpos[self.ids.grasp_site]),
                    np.array(data.site_xmat[self.ids.grasp_site]).reshape(3, 3),
                )
                position, quaternion = _relative_pose(world["gripper"], tool)
                frames.append(FramePose(TOOL_FRAME, "gripper", position, quaternion))

                # Camera frame (MuJoCo convention) and its ROS optical twin.
                camera = (
                    np.array(data.cam_xpos[self.camera_id]),
                    np.array(data.cam_xmat[self.camera_id]).reshape(3, 3),
                )
                position, quaternion = _relative_pose(world["gripper"], camera)
                frames.append(FramePose(CAMERA_FRAME, "gripper", position, quaternion))
                optical = (camera[0], camera[1] @ _CAMERA_TO_OPTICAL)
                position, quaternion = _relative_pose(camera, optical)
                frames.append(
                    FramePose(CAMERA_OPTICAL_FRAME, CAMERA_FRAME, position, quaternion)
                )
            return tuple(frames)

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def close_renderers(self) -> None:
        for attr in ("_detect_renderer", "_image_renderer"):
            renderer = getattr(self, attr, None)
            if renderer is not None:
                renderer.close()
                setattr(self, attr, None)
        self._renderer_thread = None

    def close(self) -> None:
        with self.lock:
            self.close_renderers()
