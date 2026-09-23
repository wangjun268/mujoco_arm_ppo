"""Export the MuJoCo Pro7 + LinkerHand L20 cell as URDF, for rviz2.

The mirror image of :mod:`tools.convert_hand_urdf`: it reads the *compiled* model
(``assets/rokae_xmate_pro7_pick_real.xml``, what every env and the ROS node
build their scene from) and re-emits the robot as URDF.  Everything comes from
``mjModel`` -- poses, axes, limits, geoms, inertia -- so the URDF cannot drift
from the simulation; only the mesh *files* are read from the XML, because
``mjModel`` keeps no paths.  Names are the ones the ROS node already uses
(links ``base``/``link1..7``/``gripper`` on ``/tf``, joints ``joint1..7`` on
``joint_states``), so the display is driven by the node itself.

``--hand merged`` (the default) bakes the L20 into the ``gripper`` link at its
open pose.  rviz places *every* RobotModel link with a TF lookup and the node
only publishes TF down to ``gripper``, so an articulated hand would collapse
onto the wrist.  Use ``--hand articulated`` for the full 22-link tree (that is
what a robot_state_publisher / MoveIt setup wants).

Usage::

    python3 tools/convert_arm_urdf.py                      # assets/rokae_xmate_pro7_pick_real.urdf
    python3 tools/convert_arm_urdf.py --check              # + re-load it in MuJoCo and diff
    python3 tools/convert_arm_urdf.py --hand articulated --out assets/pro7_l20_hand.urdf
    python3 tools/convert_arm_urdf.py --with-cell          # + benches, drop pad, floor
    python3 tools/convert_arm_urdf.py --collision          # + a <collision> per geom
    python3 tools/convert_arm_urdf.py --mesh-uri package:pro7_pick_place_ros

``--check`` lives in :mod:`tools.urdf_selfcheck`: it re-loads the export in
MuJoCo and diffs geometry and mass against the model, which is what keeps the
file honest.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import mujoco

# Allow `python3 tools/convert_arm_urdf.py` as well as `python3 -m tools.convert_arm_urdf`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import ASSETS_DIR, PROJECT_ROOT, asset_path

#: The model the whole project (and the ROS node) builds its scene from.
DEFAULT_MODEL = "rokae_xmate_pro7_pick_real.xml"
#: ``<robot name=...>`` of the export.
ROBOT_NAME = "rokae_xmate_pro7_l20"
#: Body the robot part of the URDF is rooted at.
ROOT_BODY = "base"
#: Body that carries the hand; the merged export bakes the fingers into it.
GRIPPER_BODY = "gripper"

#: MJCF has no velocity limits; URDF ``<limit>`` wants one anyway.
DEFAULT_VELOCITY = 3.141592653589793
#: ``<limit effort>`` when a joint has no actuator to read it from.
DEFAULT_EFFORT = 10.0

_MESH_URI_MODES = ("file", "relative", "package")

_COMMENT = re.compile(r"<!--.*?-->", re.S)


def _read_xml(path: str) -> ET.Element:
    """Parse an MJCF file with its comments taken out.

    MuJoCo tolerates ``--`` inside a comment (the shipped model writes prose
    with it), while expat -- and so ``ElementTree`` -- rejects the file.  The
    comments carry no ``<mesh>`` or ``<include>``, which is all we read.
    """
    with open(path, "r") as fh:
        return ET.fromstring(_COMMENT.sub("", fh.read()).encode("utf-8"))


# --------------------------------------------------------------------------- #
# small maths helpers
# --------------------------------------------------------------------------- #
def _fmt(values, digits: int = 10) -> str:
    """Format a float sequence as XML attribute text (``-0`` prints as ``0``)."""
    if isinstance(values, str):
        values = values.split()
    numbers = [0.0 if float(v) == 0.0 else float(v) for v in values]
    return " ".join(f"{v:.{digits}g}" for v in numbers)


def _quat_matrix(quat) -> np.ndarray:
    """Rotation matrix of a MuJoCo ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in quat)
    norm = w * w + x * x + y * y + z * z
    if norm <= 0.0:
        return np.eye(3)
    s = 2.0 / norm
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
            [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
            [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
        ]
    )


def _quat_to_rpy(quat) -> np.ndarray:
    """URDF ``rpy`` (fixed-axis XYZ) of a MuJoCo ``(w, x, y, z)`` quaternion."""
    r = _quat_matrix(quat)
    sin_pitch = float(np.clip(-r[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    if abs(sin_pitch) < 1.0 - 1e-9:
        roll = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock: roll and yaw are not separable, fold everything in
        roll = math.atan2(-r[1, 2], r[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def _quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _pose(position, quaternion) -> Tuple[np.ndarray, np.ndarray]:
    """A ``(position, quaternion)`` pair as plain arrays."""
    return np.array(position, dtype=float), np.array(quaternion, dtype=float)


def _compose(a, b):
    """``a . b`` for ``(position, quaternion)`` poses."""
    pa, qa = a
    pb, qb = b
    return pa + _quat_matrix(qa) @ pb, _quat_mul(qa, qb)


def _invert(a):
    pa, qa = a
    qi = np.array([qa[0], -qa[1], -qa[2], -qa[3]])
    return -_quat_matrix(qi) @ pa, qi


# --------------------------------------------------------------------------- #
# MJCF -> mesh files
# --------------------------------------------------------------------------- #
def mesh_table(model_path: str) -> Dict[str, Tuple[str, np.ndarray]]:
    """``mesh name -> (absolute file, scale)`` for every mesh the model uses.

    ``mjModel`` knows the mesh *names* (``geom_dataid``) but not their files, so
    this is the one thing that has to come from the XML.  ``<include>`` is
    followed the way MuJoCo follows it, and ``file`` is resolved against the
    top-level model's ``<compiler meshdir>`` (its own directory by default),
    which is what the shipped fragments rely on.
    """
    root_dir = os.path.dirname(os.path.abspath(model_path))
    root = _read_xml(model_path)
    compiler = root.find("compiler")
    meshdir = root_dir
    if compiler is not None and compiler.get("meshdir"):
        meshdir = os.path.normpath(os.path.join(root_dir, compiler.get("meshdir")))

    table: Dict[str, Tuple[str, np.ndarray]] = {}

    def visit(path: str, seen: set) -> None:
        path = os.path.abspath(path)
        if path in seen:  # a diamond of includes is legal, re-reading it is not
            return
        seen.add(path)
        here = _read_xml(path)
        for mesh in here.iter("mesh"):
            name, file = mesh.get("name"), mesh.get("file")
            if not name or not file:
                continue
            scale = np.array([float(v) for v in (mesh.get("scale") or "1 1 1").split()])
            table[name] = (os.path.normpath(os.path.join(meshdir, file)), scale)
        for include in here.iter("include"):
            target = include.get("file")
            if target:
                visit(os.path.join(os.path.dirname(path), target), seen)

    visit(model_path, set())
    return table


# --------------------------------------------------------------------------- #
# the compiled cell, indexed the way the emitters need it
# --------------------------------------------------------------------------- #
class CellModel:
    """The compiled MuJoCo cell plus the naming helpers the emitters need."""

    def __init__(self, model_path: str):
        self.path = os.path.abspath(model_path)
        self.model = mujoco.MjModel.from_xml_path(self.path)
        self.meshes = mesh_table(self.path)
        #: What had to be skipped or guessed while emitting; printed by ``main``.
        self.warnings: List[str] = []

    # -- names ---------------------------------------------------------- #
    def body_name(self, body: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)

    def joint_name(self, joint: int) -> Optional[str]:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint)

    def geom_name(self, geom: int) -> str:
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom)

    def mesh_of(self, geom: int) -> str:
        return mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_MESH, int(self.model.geom_dataid[geom])
        )

    def body(self, name: str) -> int:
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body < 0:
            raise ValueError(f"body {name!r} is not in {self.path}")
        return body

    # -- structure ------------------------------------------------------ #
    def children(self, body: int) -> List[int]:
        return [b for b in range(self.model.nbody) if self.model.body_parentid[b] == body]

    def walk(self, roots: List[int]) -> List[int]:
        """Depth-first through the body tree, parents before children."""
        order: List[int] = []
        pending = list(roots)
        while pending:
            body = pending.pop(0)
            order.append(body)
            pending[0:0] = self.children(body)
        return order

    def joint_of(self, body: int) -> Optional[int]:
        """The joint that moves ``body`` (``None`` = the body is welded on)."""
        adr, num = int(self.model.body_jntadr[body]), int(self.model.body_jntnum[body])
        if num <= 0:
            return None
        if num > 1:
            raise ValueError(f"body {self.body_name(body)!r} has {num} joints")
        return adr

    def robot_bodies(self, root: str = ROOT_BODY) -> List[int]:
        """Body ids of the robot, parents before children.

        Starting at ``base`` already skips the cube, the distractors and the
        mocap marker: they hang off the world body.  A free or ball joint below
        the root would have no URDF equivalent, so it is an error, not a guess.
        """
        bodies = self.walk([self.body(root)])
        for body in bodies:
            joint = self.joint_of(body)
            if joint is not None and self.model.jnt_type[joint] in (
                mujoco.mjtJoint.mjJNT_FREE,
                mujoco.mjtJoint.mjJNT_BALL,
            ):
                raise ValueError(
                    f"body {self.body_name(body)!r} moves on a "
                    f"{mujoco.mjtJoint(self.model.jnt_type[joint]).name} joint, "
                    "which URDF cannot express"
                )
        return bodies

    def hand_bodies(self, gripper: str = GRIPPER_BODY) -> List[int]:
        """Bodies below ``gripper``, i.e. the 22 L20 links."""
        return self.walk(self.children(self.body(gripper)))

    def geoms_of(self, body: int) -> List[int]:
        return [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body]

    def world_geoms(self) -> List[int]:
        """Geoms that belong to no body at all: the floor and the benches."""
        return self.geoms_of(0)

    def mesh_frame(self, geom: int) -> Tuple[np.ndarray, np.ndarray]:
        """``(position, quaternion)`` of a mesh geom's *file* frame.

        MuJoCo keeps vertex data in the mesh's own frame -- ``mesh_pos`` /
        ``mesh_quat``, the mesh's centre of mass and principal axes -- while
        URDF drops a mesh in at its ``<origin>`` in file coordinates.  The two
        differ by exactly this frame, so every emitted mesh origin is the geom
        frame with it divided out.
        """
        mesh = int(self.model.geom_dataid[geom])
        return _pose(self.model.mesh_pos[mesh], self.model.mesh_quat[mesh])

    # -- values --------------------------------------------------------- #
    def material(self, geom: int) -> Tuple[str, np.ndarray]:
        """URDF material name + rgba of a geom, as MuJoCo resolved it."""
        rgba = np.asarray(self.model.geom_rgba[geom], dtype=float)
        matid = int(self.model.geom_matid[geom])
        if matid >= 0:
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, matid)
            if name:
                return name, rgba
        kind = mujoco.mjtGeom(self.model.geom_type[geom]).name.removeprefix("mjGEOM_")
        return f"{kind.lower()}_{self.geom_name(geom) or geom}", rgba

    def effort(self, joint: int, default: float = DEFAULT_EFFORT) -> float:
        """``<limit effort>``: the joint's actuator range, in N (N m for hinges).

        A ``motor`` pulls with ``gear * ctrl``, so its effort is the gear times
        the force range; every other actuator type (the hand's position servos)
        already is a force.
        """
        model = self.model
        for act in range(model.nu):
            if model.actuator_trntype[act] != mujoco.mjtTrn.mjTRN_JOINT:
                continue
            if int(model.actuator_trnid[act, 0]) != joint:
                continue
            force = float(np.max(np.abs(model.actuator_forcerange[act])))
            if model.actuator_biastype[act] == mujoco.mjtBias.mjBIAS_NONE:
                force *= abs(float(model.actuator_gear[act, 0]))
            return force if force > 0 else default
        return default


# --------------------------------------------------------------------------- #
# URDF emitter
# --------------------------------------------------------------------------- #
def _mesh_uri(mode: str, path: str, urdf_dir: str) -> str:
    """How a mesh file is written into the URDF.

    ``file``        -- ``file://`` absolute; the assets stay in the checkout.
    ``relative``    -- relative to the URDF, which is what MuJoCo itself needs
                       to re-load the file (see ``--check``) and what a URDF
                       living in ``assets/`` wants.
    ``package:PKG`` -- ``package://PKG/<path under assets/>``, for a package
                       that installs ``assets/``'s contents at its share root.
    """
    if mode.startswith("package:"):
        return f"package://{mode.split(':', 1)[1]}/{os.path.relpath(path, ASSETS_DIR)}"
    if mode == "relative":
        return os.path.relpath(os.path.abspath(path), urdf_dir).replace(os.sep, "/")
    return "file://" + os.path.abspath(path)


class UrdfWriter:
    """Emits ``<link>``/``<joint>`` XML for one compiled :class:`CellModel`."""

    def __init__(
        self,
        cell: CellModel,
        *,
        mesh_uri: str = "file",
        urdf_dir: str = ASSETS_DIR,
        hand: str = "merged",
        collision: bool = False,
        with_cell: bool = False,
        velocity: float = DEFAULT_VELOCITY,
        root: str = ROOT_BODY,
    ):
        mode = mesh_uri.split(":", 1)[0]
        if mode not in _MESH_URI_MODES:
            raise ValueError(f"unknown --mesh-uri {mesh_uri!r}")
        if hand not in ("merged", "articulated", "skip"):
            raise ValueError(f"unknown --hand {hand!r}")
        self.cell = cell
        self.model = cell.model
        self.mesh_uri = mesh_uri
        self.urdf_dir = os.path.abspath(urdf_dir)
        self.hand = hand
        self.collision = collision
        self.with_cell = with_cell
        self.velocity = float(velocity)
        self.root = root
        self.materials: Dict[str, np.ndarray] = {}

    # -- pieces --------------------------------------------------------- #
    def _geometry_xml(self, geom: int) -> Optional[str]:
        """The ``<geometry>`` child of one geom, or ``None`` if URDF has none."""
        model = self.model
        kind = model.geom_type[geom]
        size = model.geom_size[geom]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            name = self.cell.mesh_of(geom)
            if name not in self.cell.meshes:
                self.cell.warnings.append(f"mesh {name!r} has no file, geom skipped")
                return None
            path, scale = self.cell.meshes[name]
            if not os.path.isfile(path):
                self.cell.warnings.append(f"mesh file missing: {path}")
            attr = "" if np.allclose(scale, 1.0) else f' scale="{_fmt(scale)}"'
            uri = _mesh_uri(self.mesh_uri, path, self.urdf_dir)
            return f'<mesh filename="{uri}"{attr}/>'
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return f'<sphere radius="{_fmt([size[0]])}"/>'
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return f'<box size="{_fmt(2.0 * np.asarray(size[:3]))}"/>'
        if kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                self.cell.warnings.append(
                    f"geom {self.cell.geom_name(geom)!r}: URDF has no capsule, "
                    "written as a cylinder"
                )
            return (
                f'<cylinder radius="{_fmt([size[0]])}" '
                f'length="{_fmt([2.0 * float(size[1])])}"/>'
            )
        self.cell.warnings.append(
            f"geom {self.cell.geom_name(geom)!r} ({mujoco.mjtGeom(kind).name}) "
            "has no URDF equivalent, skipped"
        )
        return None

    def _visual_xml(
        self, geom: int, *, tag: str, pose=None, name: Optional[str] = None
    ) -> List[str]:
        """One ``<visual>``/``<collision>`` block; ``pose`` overrides the geom's."""
        position, quaternion = (
            (self.model.geom_pos[geom], self.model.geom_quat[geom])
            if pose is None
            else pose
        )
        if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            position, quaternion = _compose(
                (position, quaternion), _invert(self.cell.mesh_frame(geom))
            )
        geometry = self._geometry_xml(geom)
        if geometry is None:
            return []
        material, rgba = self.cell.material(geom)
        self.materials.setdefault(material, rgba)
        label = name or self.cell.geom_name(geom) or f"geom{geom}"
        lines = [f'    <{tag} name="{label}">']
        origin = self._origin_xml(position, quaternion, "      ")
        if origin:
            lines.append(origin)
        lines += ["      <geometry>", f"        {geometry}", "      </geometry>"]
        if tag == "visual":
            lines.append(f'      <material name="{material}"/>')
        lines.append(f"    </{tag}>")
        return lines

    def _inertial_xml(self, body: int, override=None) -> List[str]:
        model = self.model
        name = self.cell.body_name(body)
        if override is not None:
            mass, com, tensor = override
            # No principal-axis rotation needed: URDF takes the full tensor,
            # products of inertia included.
            return [
                "    <inertial>",
                f'      <origin xyz="{_fmt(com)}" rpy="0 0 0"/>',
                f'      <mass value="{_fmt([mass])}"/>',
                f'      <inertia ixx="{_fmt([tensor[0, 0]])}" ixy="{_fmt([tensor[0, 1]])}" '
                f'ixz="{_fmt([tensor[0, 2]])}" iyy="{_fmt([tensor[1, 1]])}" '
                f'iyz="{_fmt([tensor[1, 2]])}" izz="{_fmt([tensor[2, 2]])}"/>',
                "    </inertial>",
            ]
        mass = float(model.body_mass[body])
        if mass <= 0.0:
            # Say it explicitly.  A URDF consumer that finds no <inertial>
            # guesses one from the geometry -- MuJoCo reads a 5 cm cube as 1 kg
            # at its default density, which would turn the massless adapter on
            # the gripper into 0.14 kg of imaginary aluminium.
            return [
                "    <inertial>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                '      <mass value="0"/>',
                '      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>',
                "    </inertial>",
            ]
        # MuJoCo reports the body inertia in its principal frame, so the tensor
        # is diagonal there -- which is exactly what URDF wants.
        inertia = model.body_inertia[body]
        return [
            "    <inertial>",
            f'      <origin xyz="{_fmt(model.body_ipos[body])}" '
            f'rpy="{_fmt(_quat_to_rpy(model.body_iquat[body]))}"/>',
            f'      <mass value="{_fmt([mass])}"/>',
            f'      <inertia ixx="{_fmt([inertia[0]])}" ixy="0" ixz="0" '
            f'iyy="{_fmt([inertia[1]])}" iyz="0" izz="{_fmt([inertia[2]])}"/>',
            "    </inertial>",
        ]

    def _link_xml(
        self,
        body: int,
        extra_visuals: Optional[List[Tuple[int, tuple]]] = None,
        inertial=None,
    ) -> List[str]:
        lines = [f'  <link name="{self.cell.body_name(body)}">']
        lines.extend(self._inertial_xml(body, inertial))
        for geom in self.cell.geoms_of(body):
            lines.extend(self._visual_xml(geom, tag="visual"))
            if self.collision:
                lines.extend(self._visual_xml(geom, tag="collision"))
        for geom, pose in extra_visuals or []:
            lines.extend(
                self._visual_xml(
                    geom, tag="visual", pose=pose,
                    name=f"{self.cell.geom_name(geom)}_hand",
                )
            )
            if self.collision:
                lines.extend(
                    self._visual_xml(
                        geom, tag="collision", pose=pose,
                        name=f"{self.cell.geom_name(geom)}_hand_col",
                    )
                )
        lines.append("  </link>")
        return lines

    @staticmethod
    def _origin_xml(position, quaternion, indent: str = "    ") -> Optional[str]:
        """An ``<origin>`` line, or ``None`` when the pose is the identity."""
        if np.allclose(position, 0.0) and np.allclose(quaternion, [1, 0, 0, 0]):
            return None
        return (
            f'{indent}<origin xyz="{_fmt(position)}" '
            f'rpy="{_fmt(_quat_to_rpy(quaternion))}"/>'
        )

    def _joint_xml(self, body: int) -> List[str]:
        """The joint that hangs ``body`` off its parent (fixed ones included)."""
        model = self.model
        name = self.cell.body_name(body)
        parent = self.cell.body_name(int(model.body_parentid[body]))
        joint = self.cell.joint_of(body)
        if joint is not None and not np.allclose(model.jnt_pos[joint], 0.0):
            raise ValueError(
                f"joint {self.cell.joint_name(joint)!r} is anchored away from "
                f"its body origin ({model.jnt_pos[joint]}); URDF cannot say that"
            )
        if joint is None:
            kind, label = "fixed", f"{name}_fixed"
        elif model.jnt_limited[joint]:
            kind, label = "revolute", self.cell.joint_name(joint)
        else:
            kind, label = "continuous", self.cell.joint_name(joint)
        lines = [
            f'  <joint name="{label}" type="{kind}">',
            f'    <parent link="{parent}"/>',
            f'    <child link="{name}"/>',
        ]
        origin = self._origin_xml(model.body_pos[body], model.body_quat[body])
        if origin:
            lines.append(origin)
        if joint is None:
            return lines + ["  </joint>"]
        lines.append(f'    <axis xyz="{_fmt(model.jnt_axis[joint])}"/>')
        if kind == "revolute":
            lo, hi = (float(v) for v in model.jnt_range[joint])
            lines.append(
                f'    <limit lower="{_fmt([lo])}" upper="{_fmt([hi])}" '
                f'effort="{_fmt([self.cell.effort(joint)])}" '
                f'velocity="{_fmt([self.velocity])}"/>'
            )
        lines.append("  </joint>")
        return lines

    def _prop_xml(self, name: str, geom: int) -> Tuple[List[str], List[str]]:
        """A static world geom: a massless link, pinned where MuJoCo puts it."""
        pose = _pose(self.model.geom_pos[geom], self.model.geom_quat[geom])
        link = [f'  <link name="{name}">']
        link.extend(self._visual_xml(geom, tag="visual", pose=pose))
        link.append("  </link>")
        joint = [
            f'  <joint name="{name}_fixed" type="fixed">',
            '    <parent link="world"/>',
            f'    <child link="{name}"/>',
            f'    <origin xyz="{_fmt(pose[0])}" rpy="{_fmt(_quat_to_rpy(pose[1]))}"/>',
            "  </joint>",
        ]
        return link, joint

    def _hand_in_gripper(self) -> List[Tuple[int, tuple]]:
        """``(body, pose in the gripper frame)`` for the whole hand.

        ``qpos`` 0 is the open hand (see ``linkerhand_l20_poses.json``), which is
        the pose the merged export freezes.
        """
        model = self.model
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        gripper = self.cell.body(GRIPPER_BODY)
        to_gripper = _invert(_pose(data.xpos[gripper], data.xquat[gripper]))
        return [
            (body, _compose(to_gripper, _pose(data.xpos[body], data.xquat[body])))
            for body in self.cell.hand_bodies()
        ]

    def _merged_hand(self) -> List[Tuple[int, tuple]]:
        """Hand geoms expressed in the ``gripper`` frame, at the open pose."""
        model = self.model
        return [
            (geom, _compose(pose, _pose(model.geom_pos[geom], model.geom_quat[geom])))
            for body, pose in self._hand_in_gripper()
            for geom in self.cell.geoms_of(body)
        ]

    def _merged_hand_inertia(self):
        """``(mass, com, tensor)`` of the whole open hand, in the ``gripper`` frame.

        Baking 22 links into one has to carry their inertia with it: a massless
        ``gripper`` would make MuJoCo (and anything else that reads the file)
        fall back to a density guess for those meshes -- 1.7 kg instead of the
        hand's 0.085 kg.
        """
        model = self.model
        identity = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        parts = []
        for body, pose in [(self.cell.body(GRIPPER_BODY), identity), *self._hand_in_gripper()]:
            mass = float(model.body_mass[body])
            if mass <= 0.0:
                continue
            com = pose[0] + _quat_matrix(pose[1]) @ model.body_ipos[body]
            # MuJoCo gives each body its inertia in the principal frame named by
            # body_iquat, located at body_ipos.
            rot = _quat_matrix(pose[1]) @ _quat_matrix(model.body_iquat[body])
            parts.append((mass, com, rot @ np.diag(model.body_inertia[body]) @ rot.T))
        mass = sum(part[0] for part in parts)
        com = sum(part[0] * part[1] for part in parts) / mass
        tensor = np.zeros((3, 3))
        for part_mass, part_com, part_tensor in parts:
            offset = part_com - com
            tensor += part_tensor
            tensor += part_mass * (
                float(offset @ offset) * np.eye(3) - np.outer(offset, offset)
            )
        return mass, com, tensor

    def _props(self) -> List[Tuple[str, int]]:
        """``(link, geom)`` of every world geom a URDF can draw.

        A MuJoCo plane has no URDF counterpart; rviz's Grid covers the floor.
        """
        props: List[Tuple[str, int]] = []
        for geom in self.cell.world_geoms():
            if self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_PLANE:
                self.cell.warnings.append(
                    f"floor {self.cell.geom_name(geom)!r} is a MuJoCo plane "
                    "(rviz's Grid replaces it)"
                )
                continue
            props.append((f"prop_{self.cell.geom_name(geom)}", geom))
        return props

    def xml(self) -> str:
        cell = self.cell
        bodies = cell.robot_bodies(self.root)
        hand = set(cell.hand_bodies())
        if self.hand != "articulated":
            # The hand is either baked into 'gripper' or left out entirely.
            bodies = [b for b in bodies if b not in hand]
        merged = self._merged_hand() if self.hand == "merged" else []
        merged_inertia = self._merged_hand_inertia() if self.hand == "merged" else None
        props = self._props() if self.with_cell else []

        links: List[str] = []
        joints: List[str] = []
        if props:
            # A URDF has exactly one root, so the robot hangs off 'world' here.
            links.append('  <link name="world"/>')
            joints.append(
                '  <joint name="world_fixed" type="fixed">\n'
                '    <parent link="world"/>\n'
                f'    <child link="{cell.body_name(bodies[0])}"/>\n'
                "  </joint>"
            )
            for name, geom in props:
                prop_link, prop_joint = self._prop_xml(name, geom)
                links.extend(prop_link)
                joints.extend(prop_joint)

        for body in bodies:
            is_gripper = cell.body_name(body) == GRIPPER_BODY
            links.extend(
                self._link_xml(
                    body,
                    merged if is_gripper else None,
                    merged_inertia if is_gripper else None,
                )
            )
            if body != bodies[0]:
                joints.extend(self._joint_xml(body))

        header = [
            '<?xml version="1.0"?>',
            "<!--",
            f"  {ROBOT_NAME}: generated by tools/convert_arm_urdf.py from",
            f"  {os.path.relpath(cell.path, PROJECT_ROOT)}",
            "  Do not edit by hand; re-run the converter instead.",
            "-->",
            f'<robot name="{ROBOT_NAME}">',
        ]
        materials = [
            f'  <material name="{name}">\n'
            f'    <color rgba="{_fmt(rgba)}"/>\n'
            "  </material>"
            for name, rgba in self.materials.items()
        ]
        return "\n".join(header + materials + links + joints + ["</robot>", ""])


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", default=asset_path(DEFAULT_MODEL),
                    help="MuJoCo model to export")
    ap.add_argument("--out", default=None,
                    help="output file (default: <model>.urdf, next to the model)")
    ap.add_argument("--mesh-uri", default="file",
                    help="mesh URIs: " + " | ".join(_MESH_URI_MODES) + "[:PKG]")
    ap.add_argument("--hand", default="merged",
                    choices=("merged", "articulated", "skip"),
                    help="merged: bake the open hand into 'gripper' (what the "
                         "node's TF can drive, default); articulated: 22 links + "
                         "21 joints; skip: arm only")
    ap.add_argument("--collision", action="store_true",
                    help="also emit a <collision> per geom")
    ap.add_argument("--with-cell", action="store_true",
                    help="export the benches / drop pad as static world props")
    ap.add_argument("--velocity", type=float, default=DEFAULT_VELOCITY,
                    help="<limit velocity> to write (MJCF has no velocity limit)")
    ap.add_argument("--check", action="store_true",
                    help="re-load the result in MuJoCo and diff it against the MJCF")
    args = ap.parse_args(argv)

    out = os.path.abspath(args.out or os.path.splitext(args.model)[0] + ".urdf")
    kwargs = dict(
        mesh_uri=args.mesh_uri,
        urdf_dir=os.path.dirname(out),
        hand=args.hand,
        collision=args.collision,
        with_cell=args.with_cell,
        velocity=args.velocity,
    )
    cell = CellModel(args.model)
    text = UrdfWriter(cell, **kwargs).xml()
    # ``--`` is legal in MJCF comments but not in XML ones, and a URDF that
    # strict parsers (ElementTree, most ROS tooling) reject is worse than no
    # URDF at all.  Cheap, and it has already caught one header.
    ET.fromstring(text.encode("utf-8"))
    with open(out, "w") as fh:
        fh.write(text)

    print(f"wrote {out}")
    print(f"  {len(cell.robot_bodies())} MuJoCo bodies -> {text.count('<link ')} links, "
          f"{text.count('<joint ')} joints (hand={args.hand}, mesh-uri={args.mesh_uri})")
    for warning in cell.warnings:
        print(f"  note: {warning}")

    if not args.check:
        return 0
    # Imported here so the exporter itself stays free of the verification code
    # (see tools/urdf_selfcheck.py, and tests/test_arm_urdf.py which calls it too).
    from tools.urdf_selfcheck import check

    print("checking against the MuJoCo model:")
    return 0 if check(cell, kwargs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
