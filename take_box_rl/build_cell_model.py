#!/usr/bin/env python3
"""Build the take_box dual-arm cell as a MuJoCo model.

Input  : the ROKAE arm description named in ``cell_config.json`` (URDF or the
         plain xacro variants shipped with rokae_ros2) plus the geometry of the
         cell (arm mounting, table, box, hand).
Output : ``assets/take_box_cell.xml`` (+ copied arm meshes) and
         ``assets/take_box_ready.json`` - the pre-grasp arm pose, solved by IK
         so both palms sit on the box faces.

The MuJoCo URDF importer does the hard part (axes, offsets, inertias, meshes);
this script then:

* instantiates the exported single-arm tree **twice**, renamed ``L_*`` / ``R_*``
  and placed by the mounting transform from the config;
* attaches a parametric 6-channel hand (LinkerHand O6 has no CAD on this
  machine - the O6 channels are thumb-flex/thumb-swing/index/middle/ring/pinky)
  with one position servo per channel;
* adds the table, the box (a mocap body, moved by the env) and a camera;
* disables contacts, matching the reach/pick models of this project family:
  the carry is kinematic, forces are not modelled.

Run::

    python3 build_cell_model.py            # build + solve the ready pose
    python3 build_cell_model.py --check    # re-verify the existing asset
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from take_box_env.take_box import palm_frame_local

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "cell_config.json"
ASSETS = ROOT / "assets"
MODEL_OUT = ASSETS / "take_box_cell.xml"
READY_OUT = ASSETS / "take_box_ready.json"
MESH_DIR = ASSETS / "meshes"

HAND_CHANNELS = ("thumb_flex", "thumb_swing", "index", "middle", "ring", "pinky")
#: rotation from the world frame to each arm's mount (deg around z)


# --------------------------------------------------------------------------- #
# arm description -> plain URDF
# --------------------------------------------------------------------------- #
def expand_arm_description(path: Path) -> str:
    """Return a plain URDF string for a URDF or the rokae xacro templates.

    The rokae descriptions only use ``xacro:property`` values and one
    ``xacro:include``; expanding them here keeps the build dependency-free
    (no ROS / no python-xacro on this machine).
    """
    text = path.read_text()
    if path.suffix != ".xacro":
        return text

    properties = dict(re.findall(
        r'<xacro:property\s+name="([^"]+)"\s+value\s*=\s*"([^"]*)"', text
    ))
    text = re.sub(r"<xacro:include[^>]*?/>", "", text)
    text = re.sub(r"<xacro:property[^>]*?/>", "", text)
    for name, value in properties.items():
        text = text.replace("${%s}" % name, value)

    # package:// -> the checkout on this machine
    package_root = path.parents[1]
    text = text.replace("package://rokae_description/", f"{package_root}/")
    text = re.sub(r"<\?xml-model[^>]*\?>", "", text)
    return text


def export_arm_mjcf(urdf_text: str, workdir: Path) -> ET.ElementTree:
    urdf_path = workdir / "arm.urdf"
    urdf_path.write_text(urdf_text)
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    out = workdir / "arm.xml"
    mujoco.mj_saveLastXML(str(out), model)
    return ET.parse(out)


# --------------------------------------------------------------------------- #
# arm instantiation
# --------------------------------------------------------------------------- #
def rename_subtree(element: ET.Element, prefix: str) -> None:
    """Prefix every body/joint/geom/site name so two arms can coexist."""
    for child in element.iter():
        if child.tag in ("body", "joint", "geom", "site", "camera", "light"):
            name = child.get("name")
            if name:
                child.set("name", f"{prefix}{name}")
        if child.tag == "geom":
            # contact-free arms: the carry is kinematic in this project family
            child.set("contype", "0")
            child.set("conaffinity", "0")


def instantiate_arm(export: ET.ElementTree, side: str, mount: dict) -> ET.Element:
    """Deep-copy the exported arm tree into a placement body for one side."""
    worldbody = export.getroot().find("worldbody")
    prefix = f"{side}_"
    placement = ET.Element("body", {
        "name": f"{prefix}mount",
        "pos": " ".join(f"{v:g}" for v in mount["pos"]),
        "euler": " ".join(f"{np.radians(v):.6f}" for v in mount["euler_deg"]),
    })
    for child in list(worldbody):
        node = copy.deepcopy(child)
        rename_subtree(node, prefix)
        placement.append(node)
    # remember the last link so the hand can be attached to it
    return placement


def last_link_body(placement: ET.Element) -> ET.Element:
    """Deepest body of the arm chain (the flange link)."""
    body = None
    for candidate in placement.iter("body"):
        if candidate is not placement:
            body = candidate
    if body is None:
        raise ValueError("no arm bodies found in the exported model")
    return body


# --------------------------------------------------------------------------- #
# hand
# --------------------------------------------------------------------------- #
def build_hand(side: str, hand_cfg: dict) -> ET.Element:
    """Parametric 6-channel hand: palm + 4 fingers + 2-joint thumb.

    Axis convention: the palm normal (approach direction) is +z, fingers extend
    along +z and curl about x, the thumb opposes them from +y.  ``grasp`` marks
    the point that must touch the box face and is also the RL end effector.
    """
    prefix = f"{side}_"
    palm = hand_cfg["palm"]
    finger = hand_cfg["finger"]
    thumb = hand_cfg["thumb"]
    hx, hy, hz = palm["half_size"]

    hand = ET.Element("body", {
        "name": f"{prefix}hand",
        "pos": " ".join(f"{v:g}" for v in hand_cfg.get("flange_offset", [0, 0, 0])),
    })
    ET.SubElement(hand, "geom", {
        "name": f"{prefix}palm_geom",
        "type": "box",
        "size": f"{hx:g} {hy:g} {hz:g}",
        "pos": f"0 0 {hz:g}",
        "mass": f"{palm['mass']:g}",
        "rgba": "0.85 0.55 0.20 1",
        "contype": "0",
        "conaffinity": "0",
    })
    ET.SubElement(hand, "site", {
        "name": f"{prefix}grasp",
        "pos": f"0 0 {2 * hz + 0.01:g}",
        "size": "0.006",
        "rgba": "0.1 0.75 0.25 1",
    })

    # four fingers, one hinge each, spread along x
    spread = np.linspace(-hx * 0.72, hx * 0.72, 4)
    for name, offset in zip(("index", "middle", "ring", "pinky"), spread):
        finger_body = ET.SubElement(hand, "body", {
            "name": f"{prefix}{name}",
            "pos": f"{offset:g} {-hy * 0.85:g} {2 * hz + 0.006:g}",
        })
        ET.SubElement(finger_body, "joint", {
            "name": f"{prefix}{name}_joint",
            "type": "hinge",
            "axis": "1 0 0",
            "range": f"0 {hand_cfg['grip_closed']:g}",
            "damping": "0.02",
            "armature": "0.002",
        })
        ET.SubElement(finger_body, "geom", {
            "name": f"{prefix}{name}_geom",
            "type": "capsule",
            "fromto": f"0 0 0 0 0 {finger['length']:g}",
            "size": f"{finger['radius']:g}",
            "mass": f"{finger['mass']:g}",
            "rgba": "0.9 0.9 0.92 1",
            "contype": "0",
            "conaffinity": "0",
        })

    # thumb: flex (about x) then a swinging distal segment (about y)
    thumb_base = ET.SubElement(hand, "body", {
        "name": f"{prefix}thumb_base",
        "pos": f"{-hx * 1.05:g} {hy * 0.55:g} {2 * hz + 0.012:g}",
    })
    ET.SubElement(thumb_base, "joint", {
        "name": f"{prefix}thumb_flex_joint",
        "type": "hinge",
        "axis": "1 0 0",
        "range": f"0 {hand_cfg['grip_closed']:g}",
        "damping": "0.02",
        "armature": "0.002",
    })
    ET.SubElement(thumb_base, "geom", {
        "name": f"{prefix}thumb_base_geom",
        "type": "capsule",
        "fromto": f"0 0 0 0 0 {thumb['length'] * 0.6:g}",
        "size": f"{thumb['radius']:g}",
        "mass": f"{thumb['mass'] * 0.6:g}",
        "rgba": "0.9 0.9 0.92 1",
        "contype": "0",
        "conaffinity": "0",
    })
    thumb_tip = ET.SubElement(thumb_base, "body", {
        "name": f"{prefix}thumb_tip",
        "pos": f"0 0 {thumb['length'] * 0.6:g}",
    })
    ET.SubElement(thumb_tip, "joint", {
        "name": f"{prefix}thumb_swing_joint",
        "type": "hinge",
        "axis": "0 1 0",
        "range": f"0 {hand_cfg['grip_closed']:g}",
        "damping": "0.02",
        "armature": "0.002",
    })
    ET.SubElement(thumb_tip, "geom", {
        "name": f"{prefix}thumb_tip_geom",
        "type": "capsule",
        "fromto": f"0 0 0 0 0 {thumb['length'] * 0.7:g}",
        "size": f"{thumb['radius'] * 0.9:g}",
        "mass": f"{thumb['mass'] * 0.4:g}",
        "rgba": "0.9 0.9 0.92 1",
        "contype": "0",
        "conaffinity": "0",
    })
    return hand


# --------------------------------------------------------------------------- #
# cell assembly
# --------------------------------------------------------------------------- #
def build_cell(config: dict) -> ET.ElementTree:
    if config["robot"].get("source", "single_arm_urdf") == "dual_arm_mjcf":
        return build_dual_arm_cell(config)
    return build_single_arm_cell(config)


def build_single_arm_cell(config: dict) -> ET.ElementTree:
    config = dict(config, arm=config["legacy_arm"])
    workdir = ROOT / ".build"
    workdir.mkdir(exist_ok=True)
    arm_path = Path(config["arm"]["urdf"])
    export = export_arm_mjcf(expand_arm_description(arm_path), workdir)

    root = ET.Element("mujoco", {"model": "take_box_cell"})
    ET.SubElement(root, "compiler", {
        "angle": "radian", "autolimits": "true", "balanceinertia": "true",
    })
    ET.SubElement(root, "option", {
        # 0.02 s matches the reach/pick models of this project family: an
        # episode of 260 steps is then 5.2 s of arm motion instead of 0.5 s.
        "timestep": "0.02", "gravity": "0 0 0", "integrator": "implicitfast",
    })
    ET.SubElement(root, "default").extend([
        ET.fromstring(
            f'<joint damping="{config["arm"].get("joint_damping", 0.5):g}" '
            f'armature="{config["arm"].get("joint_armature", 0.05):g}"/>'
        ),
        ET.fromstring('<geom friction="0.7 0.15 0.15"/>'),
        ET.fromstring('<motor ctrllimited="true" ctrlrange="-1 1" forcerange="-1 1"/>'),
        ET.fromstring('<position ctrllimited="true" kp="1"/>'),
    ])

    # assets: one copy of the arm meshes, shared by both arms
    asset = ET.SubElement(root, "asset")
    for mesh in export.getroot().findall("./asset/mesh"):
        node = copy.deepcopy(mesh)
        node.set("file", f"meshes/{Path(node.get('file')).name}")
        node.set("scale", " ".join(f"{v:g}" for v in config["arm"]["mesh_scale"]))
        asset.append(node)
    ET.SubElement(asset, "texture", {
        "name": "tex", "type": "2d", "builtin": "checker",
        "rgb1": "0.18 0.18 0.20", "rgb2": "0.28 0.28 0.30",
        "width": "256", "height": "256",
    })
    ET.SubElement(asset, "material", {
        "name": "floor", "texture": "tex", "texrepeat": "8 8", "reflectance": "0",
    })
    ET.SubElement(asset, "material", {
        "name": "box_mat", "rgba": "0.55 0.34 0.16 1",
    })
    ET.SubElement(asset, "material", {
        "name": "table_mat", "rgba": "0.35 0.36 0.40 1",
    })

    worldbody = ET.SubElement(root, "worldbody")
    cell = config["cell"]
    ET.SubElement(worldbody, "light", {
        "name": "light", "pos": "1.6 -1.4 2.4", "dir": "-0.5 0.4 -1",
        "diffuse": "0.9 0.9 0.9", "specular": "0.1 0.1 0.1",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "cam_iso", "pos": "1.35 -1.65 1.15",
        "xyaxes": "0.76 0.65 0 -0.37 0.44 0.82", "fovy": "50",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "3 3 0.1",
        "pos": f"0 0 {cell['floor_z']:g}", "material": "floor",
        "density": "0", "contype": "0", "conaffinity": "0",
    })

    table = cell["table"]
    ET.SubElement(worldbody, "geom", {
        "name": "table", "type": "box",
        "size": " ".join(f"{v / 2:g}" for v in table["size"]),
        "pos": " ".join(f"{v:g}" for v in table["pos"]),
        "material": "table_mat", "density": "0",
        "contype": "0", "conaffinity": "0",
    })

    box = cell["box"]
    box_body = ET.SubElement(worldbody, "body", {
        "name": "box", "mocap": "true",
        "pos": " ".join(f"{v:g}" for v in box["pos"]),
    })
    ET.SubElement(box_body, "geom", {
        "name": "box_geom", "type": "box",
        "size": " ".join(f"{v / 2:g}" for v in box["size"]),
        "material": "box_mat", "mass": f"{box['mass']:g}",
        "contype": "0", "conaffinity": "0",
    })
    ET.SubElement(box_body, "site", {"name": "box_site", "pos": "0 0 0", "size": "0.005"})

    # two arms with hands
    actuators = ET.Element("actuator")
    for side in ("left", "right"):
        mount = cell["arms"][side]
        placement = instantiate_arm(export, side, mount)
        flange = last_link_body(placement)
        flange.append(build_hand(side, config["hand"]))
        worldbody.append(placement)

        gears = config["arm"]["joint_gears"]
        for index, gear in enumerate(gears, start=1):
            ET.SubElement(actuators, "motor", {
                "name": f"{side}_a{index}",
                "joint": f"{side}_joint{index}",
                "gear": f"{gear:g}",
            })
        for channel in HAND_CHANNELS:
            ET.SubElement(actuators, "position", {
                "name": f"{side}_{channel}_pos",
                "joint": f"{side}_{channel}_joint",
                "kp": f"{config['hand']['kp']:g}",
                "ctrlrange": f"0 {config['hand']['grip_closed']:g}",
            })
    root.append(actuators)

    ET.indent(ET.ElementTree(root), space="  ")
    return ET.ElementTree(root)


# --------------------------------------------------------------------------- #
# ready pose via IK
# --------------------------------------------------------------------------- #
def hand_frame(model, data, arm_cfg: dict) -> tuple:
    """Palm centre and palm frame in the wrist body's local coordinates.

    The lkwy73 hand points along its local -z, spreads along local y and the
    palm faces local +x/-x; measuring it here keeps the IK target independent of
    that guessing game.
    """
    return palm_frame_local(
        model, data, arm_cfg["wrist_body"], arm_cfg["finger_prefix"])


def dual_arm_grasp_targets(config: dict, model, data) -> dict:
    """Box faces plus the palm orientation each arm needs to hold the box."""
    box = config["cell"]["box"]
    centre = np.asarray(box["pos"], dtype=float)
    half = np.asarray(box["size"], dtype=float) / 2
    targets = {}
    for side, arm_cfg in config["robot"]["arms"].items():
        sign = 1.0 if side == "left" else -1.0
        palm_local, source = hand_frame(model, data, arm_cfg)
        # Which way each palm has to face: the *measured* palm frame (finger,
        # spread, normal = finger x spread) with its normal towards the box, and
        # rolled by GRASP_ROLL_DEG about that normal.  The roll is not free
        # choice but a reachability constraint of this cell: with the fingers
        # straight forward (roll = 0) the best the two wrists can do is 21 deg
        # off - and the configuration they *can* reach is the one with the BACK
        # of the hand against the box.  Rolling 30 deg towards the fingers-down
        # grasp brings both hands inside the workspace (residual ~1 deg) with the
        # palms on the faces and the fingers curling towards the box.
        roll = np.radians(config["task"].get("grasp_roll_deg", 30.0))
        c, s = np.cos(roll), np.sin(roll)
        roll_about_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        target_frame = roll_about_y @ np.column_stack([
            [1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0],
        ])
        rotation = target_frame @ source.T
        assert np.linalg.det(rotation) > 0, "grasp target must be a rotation"
        targets[side] = {
            "site_local": palm_local.tolist(),
            "site_pos": (centre + np.array([0.0, sign * half[1], 0.0])).tolist(),
            "rotation": rotation.tolist(),
        }
    return targets


def build_dual_arm_cell(config: dict) -> ET.ElementTree:
    """Use the lkwy73 dual-arm description as-is, mounted in the cell."""
    robot = config["robot"]
    source = ET.parse(robot["path"])
    model = mujoco.MjModel.from_xml_path(robot["path"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    root = ET.Element("mujoco", {"model": "take_box_cell"})
    ET.SubElement(root, "compiler", {
        "angle": "radian", "autolimits": "true", "balanceinertia": "true",
    })
    ET.SubElement(root, "option", {
        "timestep": "0.02", "gravity": "0 0 0", "integrator": "implicitfast",
    })
    ET.SubElement(root, "default").extend([
        ET.fromstring(
            f'<joint damping="{robot["joint_damping"]:g}" '
            f'armature="{robot["joint_armature"]:g}"/>'
        ),
        ET.fromstring('<geom friction="0.7 0.15 0.15"/>'),
        ET.fromstring('<motor ctrllimited="true" ctrlrange="-1 1" forcerange="-1 1"/>'),
        ET.fromstring(f'<position ctrllimited="true" kp="{robot["finger_kp"]:g}"/>'),
    ])

    asset = ET.SubElement(root, "asset")
    for mesh in source.getroot().findall("./asset/mesh"):
        node = copy.deepcopy(mesh)
        node.set("file", f"meshes/{Path(node.get('file')).name}")
        node.attrib.pop("scale", None)
        asset.append(node)
    ET.SubElement(asset, "texture", {
        "name": "tex", "type": "2d", "builtin": "checker",
        "rgb1": "0.18 0.18 0.20", "rgb2": "0.28 0.28 0.30",
        "width": "256", "height": "256",
    })
    ET.SubElement(asset, "material", {
        "name": "floor", "texture": "tex", "texrepeat": "8 8", "reflectance": "0",
    })
    ET.SubElement(asset, "material", {"name": "box_mat", "rgba": "0.55 0.34 0.16 1"})
    ET.SubElement(asset, "material", {"name": "table_mat", "rgba": "0.35 0.36 0.40 1"})

    worldbody = ET.SubElement(root, "worldbody")
    cell = config["cell"]
    ET.SubElement(worldbody, "light", {
        "name": "light", "pos": "1.6 -1.4 2.6", "dir": "-0.4 0.4 -1",
        "diffuse": "0.9 0.9 0.9", "specular": "0.1 0.1 0.1",
    })
    ET.SubElement(worldbody, "camera", {
        "name": "cam_iso", "pos": "1.55 -1.45 1.45",
        "xyaxes": "0.68 0.73 0 -0.42 0.39 0.82", "fovy": "50",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "size": "3 3 0.1",
        "pos": f"0 0 {cell['floor_z']:g}", "material": "floor",
        "density": "0", "contype": "0", "conaffinity": "0",
    })
    table = cell["table"]
    ET.SubElement(worldbody, "geom", {
        "name": "table", "type": "box",
        "size": " ".join(f"{v / 2:g}" for v in table["size"]),
        "pos": " ".join(f"{v:g}" for v in table["pos"]),
        "material": "table_mat", "density": "0",
        "contype": "0", "conaffinity": "0",
    })
    box = cell["box"]
    box_body = ET.SubElement(worldbody, "body", {
        "name": "box", "mocap": "true",
        "pos": " ".join(f"{v:g}" for v in box["pos"]),
    })
    ET.SubElement(box_body, "geom", {
        "name": "box_geom", "type": "box",
        "size": " ".join(f"{v / 2:g}" for v in box["size"]),
        "material": "box_mat", "mass": f"{box['mass']:g}",
        "contype": "0", "conaffinity": "0",
    })
    ET.SubElement(box_body, "site", {"name": "box_site", "pos": "0 0 0", "size": "0.005"})

    # the robot itself, placed as one unit
    placement = ET.Element("body", {
        "name": "robot",
        "pos": " ".join(f"{v:g}" for v in robot["pos"]),
        "euler": " ".join(f"{np.radians(v):.6f}" for v in robot["euler_deg"]),
    })
    for child in list(source.getroot().find("worldbody")):
        node = copy.deepcopy(child)
        for geom in node.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
        placement.append(node)
    worldbody.append(placement)

    # grasp sites: local palm centre, measured on the source model
    targets = dual_arm_grasp_targets(config, model, data)
    for side, arm_cfg in robot["arms"].items():
        wrist = None
        for candidate in placement.iter("body"):
            if candidate.get("name") == arm_cfg["wrist_body"]:
                wrist = candidate
                break
        if wrist is None:
            raise ValueError(f"wrist body {arm_cfg['wrist_body']} missing")
        ET.SubElement(wrist, "site", {
            "name": f"{side}_grasp",
            "pos": " ".join(f"{v:.6f}" for v in targets[side]["site_local"]),
            "size": "0.008",
            "rgba": "0.1 0.75 0.25 1",
        })

    actuators = ET.SubElement(root, "actuator")
    for side, arm_cfg in robot["arms"].items():
        for joint, gear in zip(arm_cfg["joints"], robot["joint_gears"]):
            ET.SubElement(actuators, "motor", {
                "name": f"{side}_{joint}", "joint": joint, "gear": f"{gear:g}",
            })
        for joint in arm_cfg["finger_joints"]:
            # Every finger channel gets its own travel: ``grip = 1`` has to mean
            # "this joint fully closed" (the convention of the real 6-channel O6
            # hand).  A single shared value over-drives the thumb (whose pitch
            # range is 0.58 rad) and leaves the fingers at 75% of their range.
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
            if joint_id < 0:
                raise ValueError(f"finger joint {joint!r} missing from {robot['path']}")
            travel = float(model.jnt_range[joint_id][1])
            ET.SubElement(actuators, "position", {
                "name": f"{joint}_pos", "joint": joint,
                "ctrlrange": f"0 {travel:g}",
            })

    contact = ET.SubElement(root, "contact")
    for first, second in robot.get("excludes", []):
        ET.SubElement(contact, "exclude", {"body1": first, "body2": second})

    ET.indent(ET.ElementTree(root), space="  ")
    return ET.ElementTree(root)


def grasp_targets(config: dict) -> dict:
    """Where each palm must sit to hold the box, and which way it must face."""
    box = config["cell"]["box"]
    center = np.asarray(box["pos"], dtype=float)
    half = np.asarray(box["size"], dtype=float) / 2
    targets = {}
    for side, sign in (("left", 1.0), ("right", -1.0)):
        # palm face = box face; the palm's +z (approach axis) points at the box
        z_axis = np.array([0.0, -sign, 0.0])
        x_axis = np.array([sign, 0.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        targets[side] = {
            "site_pos": (center + np.array([0.0, sign * half[1], 0.0])).tolist(),
            "rotation": np.column_stack([x_axis, y_axis, z_axis]).tolist(),
        }
    return targets


def solve_ready_pose(model, data, site: int, target_pos, target_rot,
                     qpos_idx, dof_idx, limits, seed, iters=400):
    """Two-phase damped-least-squares IK: position first, then orientation.

    Solving both at once stalls in a local minimum (the palm orientation is a
    large rotation away from the seed), so the position is solved on its own and
    the orientation is only refined afterwards, never at the cost of the palm
    landing on the box face.
    """
    q = np.clip(seed.copy(), limits[:, 0], limits[:, 1])
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    rot_weight = 0.3

    def position_step(q, error, gain=0.4):
        data.qpos[qpos_idx] = q
        # mj_jacSite needs the composite quantities (cdof) that only
        # mj_fwdPosition / mj_forward produce - with mj_kinematics alone the
        # Jacobian comes back as all zeros and the solver never moves.
        mujoco.mj_fwdPosition(model, data)
        mujoco.mj_jacSite(model, data, jacp, None, site)
        jac = jacp[:, dof_idx]
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(3), error)
        return np.clip(q + gain * dq, limits[:, 0], limits[:, 1])

    # phase 1: position only, multi-start friendly
    for _ in range(iters):
        data.qpos[qpos_idx] = q
        mujoco.mj_fwdPosition(model, data)
        e_pos = target_pos - data.site_xpos[site]
        if np.linalg.norm(e_pos) < 2e-3:
            break
        q = position_step(q, e_pos)

    # phase 2: orientation, with the position kept in check
    best = (np.inf, np.inf, q.copy())
    for _ in range(iters):
        data.qpos[qpos_idx] = q
        mujoco.mj_fwdPosition(model, data)
        pos = data.site_xpos[site].copy()
        rot = data.site_xmat[site].reshape(3, 3)
        e_pos = target_pos - pos
        # rotation vector of R_target * R_current^T
        quat_err = np.zeros(4)
        mujoco.mju_mat2Quat(quat_err, (target_rot @ rot.T).reshape(-1))
        e_rot = np.zeros(3)
        mujoco.mju_quat2Vel(e_rot, quat_err, 1.0)
        score = np.linalg.norm(e_pos) + rot_weight * np.linalg.norm(e_rot)
        if score < best[0]:
            best = (score, np.linalg.norm(e_pos), q.copy())
        if np.linalg.norm(e_pos) < 2e-3 and np.linalg.norm(e_rot) < 0.05:
            break
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        jac = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        weighted = np.concatenate([e_pos, rot_weight * e_rot])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-3 * np.eye(6), weighted)
        q = np.clip(q + 0.35 * dq, limits[:, 0], limits[:, 1])
        if np.linalg.norm(target_pos - data.site_xpos[site]) > 0.02:
            q = position_step(q, target_pos - data.site_xpos[site], gain=0.6)
    return best[2], best[0], best[1]


def solve_ready(config: dict, path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    box_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box_geom")
    box_centre = np.asarray(config["cell"]["box"]["pos"], dtype=float)
    #: 掌面与箱面之间留的空隙。site 是"四指尖与拇指尖的中点"，那是手掌体积
    #: 内部的一个虚拟点，直接把它放到箱面上会让整只手插进箱子 3 cm，所以这里
    #: 边解 IK 边量"这只手到箱子的最近距离"，把目标沿掌法线往外挪到刚好贴上。
    clearance = float(config["task"].get("palm_clearance", 0.002))
    if config["robot"].get("source", "single_arm_urdf") == "dual_arm_mjcf":
        # the source model is needed for the palm-frame measurement
        source_model = mujoco.MjModel.from_xml_path(config["robot"]["path"])
        source_data = mujoco.MjData(source_model)
        targets = dual_arm_grasp_targets(config, source_model, source_data)
    else:
        targets = grasp_targets(config)
    ready = {"targets": targets, "joints": {}}

    for side in ("left", "right"):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{side}_grasp")
        if site < 0:
            raise ValueError("hand grasp site missing from the built model")
        qpos_idx, dof_idx, limits = [], [], []
        if config["robot"].get("source") == "dual_arm_mjcf":
            joint_names = config["robot"]["arms"][side]["joints"]
        else:
            joint_names = [f"{side}_joint{i}" for i in range(1, 8)]
        for name in joint_names:
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint < 0:
                raise ValueError(f"joint {name} missing from the built model")
            qpos_idx.append(model.jnt_qposadr[joint])
            dof_idx.append(model.jnt_dofadr[joint])
            limits.append(model.jnt_range[joint])
        qpos_idx = np.asarray(qpos_idx)
        dof_idx = np.asarray(dof_idx)
        limits = np.asarray(limits)

        target_pos = np.asarray(targets[side]["site_pos"])
        target_rot = np.asarray(targets[side]["rotation"])
        geoms = arm_geoms(model, config, side)
        # multi-start: a single seed lands in local minima for the side grasp
        rng = np.random.default_rng(0 if side == "left" else 1)
        starts = [np.zeros(len(limits)), 0.5 * limits.mean(axis=1)]
        starts += [rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(10)]
        q, error, gap = None, np.inf, np.inf
        for _ in range(6):
            best_q, best_score, best_pos = None, np.inf, np.inf
            for seed in starts:
                candidate, score, pos_error = solve_ready_pose(
                    model, data, site, target_pos, target_rot,
                    qpos_idx, dof_idx, limits, np.asarray(seed),
                )
                if score < best_score:
                    best_q, best_score, best_pos = candidate, score, pos_error
                if best_pos < 2e-3:
                    break
            q, error = best_q, best_score
            data.qpos[qpos_idx] = q
            mujoco.mj_forward(model, data)
            gap = hand_box_gap(model, data, geoms, box_geom)
            if abs(gap - clearance) < 1e-3:
                break
            outward = data.site_xpos[site] - box_centre
            outward /= np.linalg.norm(outward)
            target_pos = target_pos + outward * (clearance - gap)
            starts = [q]  # warm start the refinement from the previous solution
        targets[side]["site_pos"] = np.asarray(target_pos).tolist()
        targets[side]["box_offset"] = (np.asarray(target_pos) - box_centre).tolist()
        ready["joints"][side] = q.tolist()
        ready[f"ik_error_{side}"] = float(error)
        ready[f"palm_gap_{side}"] = float(gap)
        print(f"  {side}: IK 残差 {error:.5f}  掌面离箱面 {gap * 1000:+.1f} mm  "
              f"关节 {' '.join(f'{v:+.3f}' for v in q)}")
    return ready


def arm_geoms(model, config: dict, side: str) -> list:
    """Every geom of one arm + hand (they must not end up inside the box)."""
    if config["robot"].get("source", "single_arm_urdf") == "dual_arm_mjcf":
        prefixes = (side[0].upper(), config["robot"]["arms"][side]["finger_prefix"])
    else:
        prefixes = (f"{side}_",)
    geoms = []
    for geom in range(model.ngeom):
        body = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[geom]) or ""
        if body.startswith(prefixes):
            geoms.append(geom)
    return geoms


def hand_box_gap(model, data, geoms, box_geom, distmax: float = 0.4) -> float:
    """Signed distance from the closest geom of a hand to the box (negative = inside)."""
    fromto = np.zeros(6)
    return float(min(
        mujoco.mj_geomDistance(model, data, geom, box_geom, distmax, fromto)
        for geom in geoms))


# --------------------------------------------------------------------------- #
def copy_meshes(arm_path: Path) -> int:
    config = json.loads(CONFIG_PATH.read_text())
    if config["robot"].get("source") == "dual_arm_mjcf":
        source = Path(config["robot"]["mesh_dir"])
        pattern = "*.STL"
    else:
        package_root = arm_path.parents[1]
        model_name = re.search(r"xMate\w+", arm_path.name).group(0)
        source = package_root / "meshes" / model_name / "visual"
        pattern = "*.stl"
    MESH_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob(pattern))
    for mesh in files:
        shutil.copy2(mesh, MESH_DIR / mesh.name)
    return len(files)


def verify(path: Path) -> dict:
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return {
        "nq": model.nq, "nv": model.nv, "nu": model.nu,
        "nbody": model.nbody, "ngeom": model.ngeom, "nmesh": model.nmesh,
        "nmocap": model.nmocap,
        "sites": [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
                  for i in range(model.nsite)],
        "contacts_at_rest": int(data.ncon),
        "mass": round(float(model.body_mass.sum()), 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())

    if args.check:
        if not MODEL_OUT.exists():
            print(f"missing {MODEL_OUT}", file=sys.stderr)
            return 1
        for key, value in verify(MODEL_OUT).items():
            print(f"  {key:16s} {value}")
        return 0

    tree = build_cell(config)
    ASSETS.mkdir(parents=True, exist_ok=True)
    tree.write(MODEL_OUT, encoding="utf-8", xml_declaration=False)
    if config["robot"].get("source") == "dual_arm_mjcf":
        copied = copy_meshes(Path(config["robot"]["mesh_dir"]))
    else:
        copied = copy_meshes(Path(config["legacy_arm"]["urdf"]))
    print(f"wrote {MODEL_OUT}\ncopied {copied} arm meshes -> {MESH_DIR}")

    ready = solve_ready(config, MODEL_OUT)
    READY_OUT.write_text(json.dumps(ready, indent=2))
    print(f"wrote {READY_OUT}")

    for key, value in verify(MODEL_OUT).items():
        print(f"  {key:16s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
