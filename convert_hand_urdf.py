"""Turn the official LinkerHand L20 URDF into MuJoCo model fragments.

The vendor (灵心巧手 / LinkerHand) ships the L20 as a SolidWorks-exported URDF
in ``assets/linkerhand_l20/right/``.  MuJoCo can load that URDF directly, but it
merges the root ``base_link`` into the *world* body, which makes the hand
impossible to bolt onto the Pro7 wrist.  This script therefore re-emits the same
kinematics as MJCF fragments that the arm scenes ``<include>``:

* ``linkerhand_l20_right_assets.xml``    -- ``<mesh>`` declarations
* ``linkerhand_l20_right_body.xml``      -- the 22-body hand, rooted at ``hand_base``
* ``linkerhand_l20_right_actuators.xml`` -- one position servo per joint

plus ``linkerhand_l20_poses.json`` (the open / power-grasp finger presets) and
``linkerhand_l20_right.xml`` (a standalone model for eyeballing the hand).

Everything is generated, so the vendor URDF stays the single source of truth:
drop in a new URDF and re-run ``python3 convert_hand_urdf.py``.
"""

from __future__ import annotations

import json
import os
import struct
import xml.etree.ElementTree as ET

import numpy as np

import mujoco

from paths import asset_path

#: The hand we vendor, as ``<side>`` under ``assets/linkerhand_l20/``.
SIDE = "right"

#: Root folder of the vendored vendor files.
HAND_DIR = os.path.join("linkerhand_l20", SIDE)
#: Where the fragments land, relative to ``assets/``.
FRAGMENT_DIR = "linkerhand_l20"

#: ``hand_`` prefix keeps every hand body clear of the arm's ``base``/``linkN``.
BODY_PREFIX = "hand_"
#: Mesh names inside MJCF (the arm model already owns ``base``, ``link1``, ...).
MESH_PREFIX = "l20_"

#: Power-grasp preset, as a fraction of each joint's travel.  ``finger`` covers
#: index/middle/ring/pinky; the thumb gets its own column so it can oppose them.
#: The finger side-swing (``mcp_roll``) sits at 0.5 -- the middle of its travel,
#: i.e. no splay -- because a fraction of 0.0 would drive the fingers onto their
#: negative limit and open the palm behind the cube.
CLOSE_FRACTION = {
    ("finger", "mcp_roll"): 0.50,
    ("finger", "mcp_pitch"): 0.72,
    ("finger", "pip"): 0.78,
    ("finger", "dip"): 0.62,
    ("thumb", "cmc_yaw"): 0.60,
    ("thumb", "cmc_roll"): 0.50,
    ("thumb", "cmc_pitch"): 0.60,
    ("thumb", "mcp"): 0.65,
    ("thumb", "dip"): 0.55,
}

#: The L20 is pre-shaped to wrap the grasp scene's 5 cm cube; the power-grasp
#: preset and the grasp centre are calibrated against it by
#: :class:`GraspCalibration`.
GRASP_CUBE = 0.05
#: The graspable blocks of the scene (``assets/rokae_xmate_pro7_pick_real.xml``);
#: the calibration simulates exactly these contacts.  The sticky, six-dimensional
#: contact is what stops the cube sliding out of the fingers while it is carried.
CUBE_DENSITY = 30000
CUBE_FRICTION = "3 0.5 0.1"
CUBE_CONDIM = 6
CUBE_CONTYPE, CUBE_CONAFFINITY = 4, 6

#: Position-servo gains for the hand joints.  The phalanges carry almost no
#: inertia of their own, so at the scene's 20 ms timestep anything stiff rings
#: badly (measured: kp=8, kv=0.6 drove the joint velocities past 60 rad/s and
#: walked the joints out of their limits).  These gains give w*dt ~ 0.2 with the
#: rotor inertia below -- critically damped and ~0.2 s to close a finger, which
#: is still ~14 N of grip force per phalanx on a 5 cm cube.
KP, KV, FORCE = 20.0, 1.0, 8.0

#: Joint-space damping / rotor inertia; the vendor URDF has neither, and a bare
#: hinge rattles under the position servos.
JOINT_DAMPING, JOINT_ARMATURE = 0.10, 0.05
#: Contact friction of the phalanges (the vendor model has grippy pads).
GEOM_FRICTION = "1.5 0.3 0.1"
#: Collision bits.  The vendor URDF reuses the *visual* meshes for collision,
#: and a real hand assembly has them overlapping at every joint -- left as-is
#: that is 34 self-contacts and 10 mm of interpenetration, which rings the
#: fingers at 90 rad/s and punts whatever they touch.  The hand therefore
#: collides only with the graspable objects (bit 4), never with itself.
GEOM_CONTYPE = 2
GEOM_CONAFFINITY = 4


def _fmt(values) -> str:
    """Format a float sequence as a compact MJCF attribute."""
    if isinstance(values, str):
        values = values.split()
    return " ".join(f"{float(v):.10g}" for v in values)


def _rpy_to_quat(rpy) -> str:
    """URDF ``rpy`` (fixed-axis XYZ) to MJCF ``quat`` (w x y z)."""
    if isinstance(rpy, str):
        rpy = rpy.split()
    r, p, y = (float(v) for v in rpy)
    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
    return _fmt(q)


def _stl_points(path: str) -> np.ndarray:
    """Vertices of a binary (or ASCII) STL, as an ``(n, 3)`` array."""
    with open(path, "rb") as fh:
        blob = fh.read()
    if len(blob) >= 84:
        n = struct.unpack("<I", blob[80:84])[0]
        if len(blob) == 84 + n * 50:  # binary STL
            rec = np.frombuffer(blob[84:], dtype=np.uint8).reshape(n, 50)
            # 50-byte record: 12 bytes normal, 36 bytes (3 vertices), 2 bytes attr.
            verts = rec[:, 12:48].copy().view("<f4").reshape(n, 3, 3)
            return verts.reshape(-1, 3).astype(np.float64)
    pts = []
    for line in blob.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            pts.append([float(v) for v in parts[1:]])
    return np.asarray(pts, dtype=np.float64)


def _tip_local(stl_path: str) -> np.ndarray:
    """A fingertip site, in the distal link frame.

    The phalanges are modelled with the joint at the link origin and the shell
    growing outwards, so the tip is the far end of the mesh's longest axis,
    centred on the other two.
    """
    pts = _stl_points(stl_path)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    mid = (lo + hi) / 2.0
    axis = int(np.argmax(hi - lo))
    tip = mid.copy()
    tip[axis] = hi[axis] if abs(hi[axis]) >= abs(lo[axis]) else lo[axis]
    return tip


def _finger_of(link_name: str) -> str:
    """``index_distal`` -> ``index``; ``base_link`` -> ``palm``."""
    head = link_name.split("_", 1)[0]
    return "palm" if head == "base" else head


def _closed_angle(name: str, lo: float, hi: float) -> float:
    """Power-grasp target for one joint at unit scale, from its travel."""
    finger, _, rest = name.partition("_")
    kind = "thumb" if finger == "thumb" else "finger"
    frac = CLOSE_FRACTION[(kind, rest)]
    return lo + frac * (hi - lo)


def closed_angle(name: str, lo: float, hi: float, scale: float = 1.0) -> float:
    """Power-grasp target: ``scale`` 1.0 is the nominal preset from the table."""
    # Every joint's *open* target is zero, so scaling the closed angle scales the
    # whole excursion -- including the centred side-swing, which stays at zero.
    return scale * _closed_angle(name, lo, hi)


class HandModel:
    """The vendor URDF, parsed into the pieces the emitters need."""

    def __init__(self, urdf_path: str):
        self.urdf_path = urdf_path
        self.dir = os.path.dirname(urdf_path)
        root = ET.parse(urdf_path).getroot()
        self.name = root.get("name")
        self.links = {l.get("name"): l for l in root.findall("link")}
        self.joints = root.findall("joint")

        self.children: dict[str, list[ET.Element]] = {n: [] for n in self.links}
        child_links = set()
        for joint in self.joints:
            parent = joint.find("parent").get("link")
            child = joint.find("child").get("link")
            self.children[parent].append(joint)
            child_links.add(child)
        roots = [n for n in self.links if n not in child_links]
        if len(roots) != 1:
            raise ValueError(f"expected one root link, got {roots}")
        self.root = roots[0]

    # ------------------------------------------------------------------ #
    def mesh_file(self, link: str) -> str:
        """Mesh path as written in the URDF (relative to the URDF)."""
        return self.links[link].find("visual/geometry/mesh").get("filename")

    @staticmethod
    def _xyz(element, default="0 0 0"):
        origin = element.find("origin") if element is not None else None
        return (origin.get("xyz") or default) if origin is not None else default

    @staticmethod
    def _rpy(element, default="0 0 0"):
        origin = element.find("origin") if element is not None else None
        return (origin.get("rpy") or default) if origin is not None else default

    def link_visual_rgba(self, link: str) -> tuple:
        """RGBA of the link's material, so the hand does not render flat grey."""
        material = self.links[link].find("visual/material")
        named = (material.get("name") if material is not None else None) or "silver"
        palette = {"silver": (0.52, 0.53, 0.56, 1.0), "BlackAccent": (0.13, 0.13, 0.14, 1.0)}
        return palette.get(named, palette["silver"])

    # ------------------------------------------------------------------ #
    def fragments(self, grasp_site: str | None = None) -> dict[str, str]:
        """The three ``<mujocoinclude>`` fragments, keyed by output filename."""
        return {
            f"linkerhand_l20_{SIDE}_assets.xml": self._assets_xml(),
            f"linkerhand_l20_{SIDE}_body.xml": self._body_xml(grasp_site),
            f"linkerhand_l20_{SIDE}_actuators.xml": self._actuator_xml(),
        }

    def _assets_xml(self) -> str:
        lines = ["<mujocoinclude>"]
        for link in self.links:
            mesh = f"{MESH_PREFIX}{link}"
            file = f"{HAND_DIR}/" + self.mesh_file(link)
            lines.append(f'    <mesh name="{mesh}" file="{file}"/>')
        lines.append("</mujocoinclude>")
        return "\n".join(lines) + "\n"

    def _actuator_xml(self) -> str:
        lines = ["<mujocoinclude>"]
        for joint in self.joints:
            name = joint.get("name")
            lines.append(
                f'    <position name="hand_{name}" joint="{name}"'
                f' kp="{KP}" kv="{KV}" forcerange="{-FORCE} {FORCE}"/>'
            )
        lines.append("</mujocoinclude>")
        return "\n".join(lines) + "\n"

    def _body_xml(self, grasp_site: str | None = None) -> str:
        lines = ["<mujocoinclude>"]
        self._emit_body(self.root, 1, lines, parent_is_palm=True, extra_site=grasp_site)
        lines.append("</mujocoinclude>")
        return "\n".join(lines) + "\n"

    def _emit_body(
        self,
        link: str,
        depth: int,
        lines: list,
        *,
        parent_is_palm=False,
        joint=None,
        extra_site: str | None = None,
    ) -> None:
        """Emit one ``<body>`` plus its finger, then recurse into its children."""
        pad = "  " * depth
        name = BODY_PREFIX + ("base" if parent_is_palm else link)
        body = self.links[link]
        node = body.find("inertial")
        iso = node.find("inertia") if node is not None else None
        mass = node.find("mass") if node is not None else None
        inertial = ""
        if iso is not None and mass is not None:
            rpy = self._rpy(node)
            quat = f' quat="{_rpy_to_quat(rpy)}"' if any(float(v) for v in rpy.split()) else ""
            inertial = (
                f' pos="{self._xyz(node)}"{quat}'
                f' mass="{mass.get("value")}"'
                f' fullinertia="{iso.get("ixx")} {iso.get("iyy")} {iso.get("izz")}'
                f' {iso.get("ixy")} {iso.get("ixz")} {iso.get("iyz")}"/>'
            )

        rgba = _fmt(self.link_visual_rgba(link))
        # ``pos``/``quat`` come from the joint that attaches this link, i.e. the
        # URDF joint origin -- the MuJoCo body frame *is* the joint frame.
        if joint is None:
            attach = ' pos="0 0 0"'
        else:
            rpy = self._rpy(joint)
            attach = f' pos="{self._xyz(joint)}"'
            if any(float(v) for v in rpy.split()):
                attach += f' quat="{_rpy_to_quat(rpy)}"'
        lines.append(f'{pad}<body name="{name}"{attach}>')
        if joint is not None:
            axis = _fmt(joint.find("axis").get("xyz"))
            limit = joint.find("limit")
            lo, hi = float(limit.get("lower")), float(limit.get("upper"))
            lines.append(
                f'{pad}  <joint name="{joint.get("name")}" type="hinge"'
                f' pos="0 0 0" axis="{axis}" limited="true" range="{lo} {hi}"'
                f' damping="{JOINT_DAMPING}" armature="{JOINT_ARMATURE}"/>'
            )
        if inertial:
            lines.append(f"{pad}  <inertial{inertial}")
        lines.append(
            f'{pad}  <geom name="{BODY_PREFIX}{link}_g" type="mesh"'
            f' mesh="{MESH_PREFIX}{link}" rgba="{rgba}"'
            f' friction="{GEOM_FRICTION}"'
            f' contype="{GEOM_CONTYPE}" conaffinity="{GEOM_CONAFFINITY}"/>'
        )
        if link.split("_")[-1] == "distal":
            stl = os.path.join(self.dir, self.mesh_file(link))
            tip = _fmt(_tip_local(stl))
            lines.append(f'{pad}  <site name="{BODY_PREFIX}{link}_tip" pos="{tip}"/>')
        if extra_site:
            lines.append(f"{pad}  {extra_site}")

        for joint in self.children[link]:
            child = joint.find("child").get("link")
            self._emit_body(child, depth + 1, lines, joint=joint)
        lines.append(f"{pad}</body>")


def poses(hand: HandModel, scale: float = 1.0) -> dict:
    """Open / power-grasp joint targets and the finger -> geom grouping."""
    open_pose, close_pose, fingers = {}, {}, {}
    for joint in hand.joints:
        name = joint.get("name")
        limit = joint.find("limit")
        lo, hi = float(limit.get("lower")), float(limit.get("upper"))
        open_pose[name] = 0.0
        close_pose[name] = closed_angle(name, lo, hi, scale)
        child = joint.find("child").get("link")
        fingers.setdefault(_finger_of(child), []).append(f"{BODY_PREFIX}{child}_g")
    # The palm is the root link, so it never shows up as a joint's child.
    fingers.setdefault("palm", []).append(f"{BODY_PREFIX}{hand.root}_g")
    return {
        "side": SIDE,
        "source": f"{HAND_DIR}/linkerhand_l20_{SIDE}.urdf",
        "open": open_pose,
        "close": close_pose,
        "fingers": fingers,
    }


def standalone_model(meshdir: str = "../") -> str:
    """A one-file MJCF that includes the same fragments, for previews."""
    assets = f"linkerhand_l20_{SIDE}_assets.xml"
    body = f"linkerhand_l20_{SIDE}_body.xml"
    actuators = f"linkerhand_l20_{SIDE}_actuators.xml"
    return f"""<mujoco model="linkerhand_l20_{SIDE}">
  <!-- Standalone preview of the vendored L20.  Generated by
       convert_hand_urdf.py; do not edit by hand. -->
  <compiler angle="radian" meshdir="{meshdir}" autolimits="true" balanceinertia="true"/>
  <option timestep="0.02" integrator="Euler"/>
  <asset>
    <include file="{assets}"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 0.6" dir="-0.4 0.4 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="preview" pos="0.28 -0.30 0.30" xyaxes="0.72 0.68 0 -0.30 0.32 0.90" fovy="45"/>
    <geom name="floor" type="plane" size="1 1 0.01" pos="0 0 -0.02" rgba="0.8 0.8 0.8 1"/>
    <include file="{body}"/>
  </worldbody>
  <actuator>
    <include file="{actuators}"/>
  </actuator>
</mujoco>
"""


class GraspCalibration:
    """Calibrate the grasp by *simulating* it, not just measuring the meshes.

    Two numbers decide whether the closing hand traps the cube or punts it:

    * the **scale** of the power-grasp preset (how hard the pads drive into the
      cube), and
    * the **grasp centre** -- the point the scene's resolved-rate expert servos
      onto the cube, expressed in the hand frame.

    Reading the vendor meshes is not enough to pick them.  The first, purely
    geometric fit put the pads 4 mm inside the cube's volume; closing there
    flicked the cube out of the palm at 1.4 m/s.  So this class drops a *free*
    cube of the scene's size, mass and friction into a copy of the hand, closes
    the hand on it, shoves it sideways, and keeps the parameters that hold it.

    Only the cheap, hand-centred part of the scene is simulated: the arm's own
    approach is handled by the expert in ``grasp_common``.
    """

    #: Coarse sweep; the best candidate is then refined by a local grid.
    GRID_X = (0.020, 0.035, 0.050, 0.065, 0.080)
    GRID_Z = (0.135, 0.150, 0.165, 0.180, 0.195)
    GRID_SCALE = (0.70, 0.85, 1.00, 1.15, 1.30)
    #: Close over this many steps, then hold; shove with this force (N) to see
    #: whether the grip survives being carried.
    RAMP, STEPS, SHOVE = 60, 220, 4.0
    #: A grip is only interesting if this many hand geoms touch the cube.
    MIN_CONTACTS = 6

    def __init__(self, hand: HandModel, out_dir: str):
        self.hand = hand
        self.nominal = {
            j.get("name"): closed_angle(
                j.get("name"),
                float(j.find("limit").get("lower")),
                float(j.find("limit").get("upper")),
            )
            for j in hand.joints
        }

        self.path = os.path.join(out_dir, "_calibrate.xml")
        half = GRASP_CUBE / 2.0
        cube = (
            '<body name="cube" pos="0 0 0"><freejoint/>'
            f'<geom name="cube_geom" type="box" size="{half} {half} {half}"'
            f' density="{CUBE_DENSITY}" friction="{CUBE_FRICTION}"'
            f' condim="{CUBE_CONDIM}" contype="{CUBE_CONTYPE}"'
            f' conaffinity="{CUBE_CONAFFINITY}"/></body>'
        )
        xml = standalone_model().replace(
            f'<include file="linkerhand_l20_{SIDE}_body.xml"/>',
            f'<include file="linkerhand_l20_{SIDE}_body.xml"/>\n    {cube}',
        )
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write(xml)
        self.model = mujoco.MjModel.from_xml_path(self.path)
        self.model.opt.gravity[:] = 0.0  # the grasp scene is weightless
        self.joint_adr = {
            name: self.model.jnt_qposadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in self.nominal
        }
        self.acts = [
            (
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"hand_{name}"),
                name,
            )
            for name in self.nominal
        ]
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_body = body
        self.cube_qpos = int(self.model.jnt_qposadr[int(self.model.body_jntadr[body])])
        self.cube_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")

        # The cube sits between the finger roots, centred across them.
        roots = [
            j
            for j in hand.children[hand.root]
            if j.find("child").get("link").split("_")[0] in ("index", "middle", "ring", "pinky")
        ]
        self.y0 = float(
            np.mean([[float(v) for v in hand._xyz(j).split()][1] for j in roots])
        )

    # ------------------------------------------------------------------ #
    def trial(self, centre, scale: float, shove=None) -> tuple[float, int]:
        """Close on a cube at ``centre``; return how far it moved and contacts."""
        data = mujoco.MjData(self.model)
        data.qpos[self.cube_qpos : self.cube_qpos + 3] = centre
        for step in range(self.STEPS):
            ramp = min(1.0, max(0, step - 10) / self.RAMP)
            for act, name in self.acts:
                data.ctrl[act] = ramp * scale * self.nominal[name]
            if shove is not None and step >= self.STEPS - 60:
                data.xfrc_applied[self.cube_body, :3] = shove
            mujoco.mj_step(self.model, data)
        moved = float(
            np.linalg.norm(
                data.qpos[self.cube_qpos : self.cube_qpos + 3] - np.asarray(centre)
            )
        )
        touches = sum(
            1
            for c in data.contact
            if self.cube_geom in (c.geom1, c.geom2)
        )
        return moved, touches

    def solve(self) -> tuple[np.ndarray, float]:
        best = None
        for x in self.GRID_X:
            for z in self.GRID_Z:
                for scale in self.GRID_SCALE:
                    centre = np.array([x, self.y0, z])
                    settled, n_settled = self.trial(centre, scale)
                    shoved, n_shoved = self.trial(centre, scale, shove=[0, self.SHOVE, 0])
                    # Prefer a grip that neither drifts when it closes nor when
                    # it is carried; anything that barely touches is no grip.
                    weak = 0.0 if min(n_settled, n_shoved) >= self.MIN_CONTACTS else 0.05
                    score = settled + shoved + weak
                    if best is None or score < best[0]:
                        best = (score, centre, scale, settled, shoved, n_settled, n_shoved)
        _, centre, scale, settled, shoved, n1, n2 = best
        print(
            f"   grip: centre={_fmt(centre)} scale={scale:.2f} "
            f"drift={settled*1000:.1f} mm +{shoved*1000:.1f} mm on a {self.SHOVE:g} N shove, "
            f"{n1}/{n2} contacts"
        )
        return centre, scale


def main() -> None:
    urdf_path = asset_path(os.path.join(HAND_DIR, f"linkerhand_l20_{SIDE}.urdf"))
    if not os.path.isfile(urdf_path):
        raise SystemExit(f"vendor URDF not found: {urdf_path}")
    hand = HandModel(urdf_path)
    out_dir = asset_path(FRAGMENT_DIR)
    os.makedirs(out_dir, exist_ok=True)

    preset = poses(hand)
    preview = os.path.join(out_dir, f"linkerhand_l20_{SIDE}.xml")
    fragments = hand.fragments()
    for name, xml in fragments.items():
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            fh.write(xml)
        print(f"wrote {name}")

    with open(preview, "w", encoding="utf-8") as fh:
        fh.write(standalone_model())

    # Second pass: the grasp preset and centre can only be calibrated once the
    # hand builds, so re-emit the body fragment with the ``grasp_center`` site.
    calibration = GraspCalibration(hand, out_dir)
    centre, scale = calibration.solve()
    preset = poses(hand, scale)
    site = f'<site name="grasp_center" pos="{_fmt(centre)}"/>'
    body = f"linkerhand_l20_{SIDE}_body.xml"
    with open(os.path.join(out_dir, body), "w", encoding="utf-8") as fh:
        fh.write(hand.fragments(grasp_site=site)[body])

    preset["grasp_center"] = [float(v) for v in centre]
    with open(os.path.join(out_dir, "linkerhand_l20_poses.json"), "w", encoding="utf-8") as fh:
        json.dump(preset, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("wrote linkerhand_l20_poses.json")

    model = mujoco.MjModel.from_xml_path(preview)
    os.remove(os.path.join(out_dir, "_calibrate.xml"))
    print(
        f"check: nbody={model.nbody} njnt={model.njnt} ngeom={model.ngeom} "
        f"nmesh={model.nmesh} nu={model.nu} mass={model.body_mass.sum():.4f} kg"
    )


if __name__ == "__main__":
    main()
