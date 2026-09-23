"""Turn the dual-arm description into the MuJoCo task model of this project.

Source: the cleaned dual-arm description (real URDF geometry, 41 links /
36 DoF / 11 kg) that lives outside this repository.  Everything the RL
environments need is added here, so the asset can be regenerated from the
description instead of being hand-edited:

* two actuated 7-DoF arms as normalised torque motors (``ctrl`` in [-1, 1]),
  the same convention as ``rokae_xmate_pro7_real.xml``;
* the 22 finger joints pinned to their open pose with equality constraints, so
  a reach task does not have to model hand control (they stay available for a
  later grasp task: drop the ``<equality>`` block and add actuators);
* one ``tip`` site per hand, placed at the centre of the four fingertips;
* two mocap targets (left arm first, then right) plus a camera and floor;
* the four link pairs that interpenetrate in the CAD (L1/base, L5/L7 and the
  mirrored pair) excluded, because those contacts would otherwise push the
  arms apart in every pose.

Run::

    python3 tools/build_dual_arm_model.py            # regenerate assets/dual_arm_reach.xml
    python3 tools/build_dual_arm_model.py --check    # verify without writing
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

# Allow `python3 tools/build_dual_arm_model.py` as well as `python3 -m tools.build_dual_arm_model`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import ASSETS_DIR

# Cleaned dual-arm description produced from the vendor URDF package.
DEFAULT_SOURCE = Path(
    os.environ.get(
        "DUAL_ARM_SOURCE",
        "/home/wj/urdf/lkwy73_o1_dual_arm_clean/lkwy73_o1_dual_arm.xml",
    )
)

# Task model written into assets/, meshes copied next to it.
MODEL_OUT = Path(ASSETS_DIR) / "dual_arm_reach.xml"
MESH_OUT = Path(ASSETS_DIR) / "dual_arm" / "meshes"

# Normalised-torque gears (Nm at ctrl = 1), shoulder to wrist.  Each arm weighs
# 4 kg, about a third of the Pro7, so the gains are scaled down to keep the
# 20 ms control step stable.
GEARS = [6.0, 6.0, 5.0, 4.0, 3.0, 2.0, 2.0]

# (label, joint prefix, wrist body, hand prefix) - the order defines the arm
# ordering of the observation, the action vector and the mocap targets.
ARMS = (
    ("left", "L", "L7_Link", "lh"),
    ("right", "R", "R7_Link", "rh"),
)

FINGER_PREFIXES = ("lh_", "rh_")
FINGER_DISTAL = ("index", "middle", "ring", "pinky")
EXCLUDES = (
    ("base_link", "L1_Link"),
    ("base_link", "R1_Link"),
    ("L5_Link", "L7_Link"),
    ("R5_Link", "R7_Link"),
)

FLOOR_GEOM = (
    '    <geom name="floor" type="plane" size="4 4 0.1" pos="0 0 -0.90" '
    'rgba="0.85 0.85 0.85 1" material="floor" density="0" '
    'contype="0" conaffinity="0"/>'
)
#: Camera framing of the working volume: (lookat, distance, azimuth, elevation).
CAMERA_LOOKAT = (0.0, 0.0, -0.35)
CAMERA_DISTANCE = 1.9
CAMERA_AZIMUTH = 215.0
CAMERA_ELEVATION = -18.0
LIGHT = (
    '    <light name="light" pos="1.5 -1.5 2.5" dir="-0.5 0.4 -1" '
    'diffuse="0.9 0.9 0.9" specular="0.1 0.1 0.1"/>'
)
DEFAULTS = (
    "<default>\n"
    '    <joint damping="0.5" armature="0.02"/>\n'
    '    <geom friction="0.7 0.15 0.15"/>\n'
    '    <motor ctrllimited="true" ctrlrange="-1 1" forcerange="-1 1"/>\n'
    "  </default>"
)


def arm_joint_names(prefix: str) -> list:
    return [f"{prefix}{i}_Joint" for i in range(1, 8)]


def camera_xml() -> str:
    """MuJoCo camera looking at the working volume from a three-quarter view.

    MuJoCo only takes ``pos``/``xyaxes`` in XML, so the usual free-camera
    controls (lookat + distance + azimuth + elevation) are converted here.
    """
    lookat = np.asarray(CAMERA_LOOKAT, dtype=float)
    azimuth = np.radians(CAMERA_AZIMUTH)
    elevation = np.radians(CAMERA_ELEVATION)
    z_axis = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        -np.sin(elevation),
    ])
    position = lookat + CAMERA_DISTANCE * z_axis
    x_axis = np.cross([0.0, 0.0, 1.0], z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    numbers = " ".join(f"{value:.5f}" for value in position)
    axes = " ".join(f"{value:.5f}" for value in np.concatenate([x_axis, y_axis]))
    return (
        f'    <camera name="cam_iso" pos="{numbers}" xyaxes="{axes}" fovy="50"/>'
    )


def fingertip_centroid(model, data, hand: str, body: str) -> np.ndarray:
    """Centre of the four fingertips, expressed in the wrist body frame."""
    mujoco.mj_forward(model, data)
    points = []
    for finger in FINGER_DISTAL:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{hand}_{finger}_distal"
        )
        if body_id < 0:
            raise ValueError(f"body {hand}_{finger}_distal missing from the model")
        points.append(data.xpos[body_id].copy())
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    rotation = data.xmat[body_id].reshape(3, 3)
    world = np.mean(points, axis=0)
    return rotation.T @ (world - data.xpos[body_id])


def build(source: Path) -> ET.ElementTree:
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)

    export = Path(ASSETS_DIR) / ".dual_arm_export.tmp.xml"
    export.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(export), model)
    tree = ET.parse(export)
    export.unlink()
    root = tree.getroot()
    root.set("model", "dual_arm_reach")

    # compiler, option and defaults
    compiler = root.find("compiler")
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")
    compiler.set("balanceinertia", "true")
    for tag in ("option", "default"):
        for element in root.findall(tag):
            root.remove(element)
    index = list(root).index(compiler) + 1
    root.insert(index, ET.fromstring(
        '<option timestep="0.02" gravity="0 0 0" integrator="Euler"/>'
    ))
    root.insert(index + 1, ET.fromstring(DEFAULTS))

    # mesh files now point at the copy that ships with the project
    for mesh in root.iter("mesh"):
        name = Path(mesh.get("file", "")).name
        mesh.set("file", f"dual_arm/meshes/{name}")
        mesh.attrib.pop("scale", None)

    # floor, light, camera and the checker material
    asset = root.find("asset")
    asset.extend(ET.fromstring(
        "<asset>\n"
        '    <texture name="tex" type="2d" builtin="checker" rgb1="0.16 0.16 0.16" '
        'rgb2="0.26 0.26 0.26" width="256" height="256"/>\n'
        '    <material name="floor" texture="tex" texrepeat="6 6" reflectance="0"/>\n'
        "  </asset>"
    ))
    worldbody = root.find("worldbody")
    worldbody.insert(0, ET.fromstring(FLOOR_GEOM))
    worldbody.insert(0, ET.fromstring(camera_xml()))
    worldbody.insert(0, ET.fromstring(LIGHT))

    # tip site per hand, at the centre of the four fingertips
    for label, _prefix, body_name, hand in ARMS:
        body = None
        for candidate in root.iter("body"):
            if candidate.get("name") == body_name:
                body = candidate
                break
        if body is None:
            raise ValueError(f"body {body_name} missing from the exported model")
        local = fingertip_centroid(model, data, hand, body_name)
        site = ET.Element("site", {
            "name": f"tip_{label}",
            "pos": " ".join(f"{value:.6f}" for value in local),
            "size": "0.01",
            "rgba": "0.1 0.75 0.25 1",
        })
        children = list(body)
        insert_at = next(
            (i for i, child in enumerate(children) if child.tag == "geom"),
            len(children),
        )
        body.insert(insert_at, site)

    # one mocap target per arm (index order == the order of ARMS)
    for label, _prefix, _body, _hand in ARMS:
        colour = "0.10 0.75 0.25 1" if label == "left" else "0.85 0.35 0.10 1"
        worldbody.append(ET.fromstring(
            f'    <body name="target_{label}" mocap="true">\n'
            f'      <geom name="target_{label}_geom" type="sphere" size="0.035" '
            f'rgba="{colour}" density="0" contype="0" conaffinity="0"/>\n'
            f'      <site name="target_{label}_site" pos="0 0 0"/>\n'
            f"    </body>"
        ))

    # 14 normalised torque motors, in the same order as ARMS
    for actuator in root.findall("actuator"):
        root.remove(actuator)
    actuator = ET.SubElement(root, "actuator")
    for _label, prefix, _body, _hand in ARMS:
        for joint, gear in zip(arm_joint_names(prefix), GEARS):
            ET.SubElement(actuator, "motor", {
                "name": f"a{joint}",
                "joint": joint,
                "gear": f"{gear:.1f}",
            })

    # Pin the fingers to the open pose.  A MuJoCo joint equality couples
    # ``joint2`` to a polynomial of ``joint1``; with only ``joint1`` given the
    # constraint is "polynomial == 0", so polycoef "0 1 0 0 0" means q == 0.
    equality = ET.SubElement(root, "equality")
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if name.startswith(FINGER_PREFIXES):
            ET.SubElement(equality, "joint", {
                "joint1": name,
                "polycoef": "0 1 0 0 0",
                "solref": "0.005 1",
            })

    # CAD self-contact pairs that overlap in every configuration
    contact = ET.SubElement(root, "contact")
    for first, second in EXCLUDES:
        ET.SubElement(contact, "exclude", {"body1": first, "body2": second})

    ET.indent(tree, space="  ")
    return tree


def copy_meshes(source: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    meshes = sorted((source.parent / "meshes").glob("*.STL"))
    for mesh in meshes:
        shutil.copy2(mesh, target / mesh.name)
    return len(meshes)


def verify(path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    def id_name(obj, index):
        return mujoco.mj_id2name(model, obj, index)

    actuators = [id_name(mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    sites = [id_name(mujoco.mjtObj.mjOBJ_SITE, i) for i in range(model.nsite)]
    return {
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "nmocap": model.nmocap,
        "nmesh": model.nmesh,
        "ngeom": model.ngeom,
        "mass": round(float(model.body_mass.sum()), 4),
        "max_faces": int(model.mesh_facenum.max()),
        "tip_sites": sum(1 for name in sites if name.startswith("tip")),
        "contacts_at_zero": int(data.ncon),
        "first_actuator": actuators[0] if actuators else "-",
        "last_actuator": actuators[-1] if actuators else "-",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="dual-arm description (MJCF) to read")
    parser.add_argument("--check", action="store_true",
                        help="validate the current asset instead of rebuilding it")
    args = parser.parse_args()

    if args.check:
        if not MODEL_OUT.exists():
            print(f"missing {MODEL_OUT}", file=sys.stderr)
            return 1
        for key, value in verify(MODEL_OUT).items():
            print(f"  {key:16s} {value}")
        return 0

    if not args.source.exists():
        print(f"source model not found: {args.source}", file=sys.stderr)
        return 1

    tree = build(args.source)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    tree.write(MODEL_OUT, encoding="utf-8", xml_declaration=False)
    copied = copy_meshes(args.source, MESH_OUT)

    print(f"wrote {MODEL_OUT}")
    print(f"copied {copied} meshes -> {MESH_OUT}")
    for key, value in verify(MODEL_OUT).items():
        print(f"  {key:16s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
