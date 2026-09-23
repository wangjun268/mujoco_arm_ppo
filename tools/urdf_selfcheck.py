"""Diff an exported URDF against the MuJoCo model it came from.

``tools/convert_arm_urdf.py --check`` runs this: it writes the export out with
``<collision>`` instead of ``<visual>`` -- MuJoCo ignores visuals when it reads
a URDF, it simulates the collision geometry -- re-loads that with MuJoCo, and
compares the two models.  Both tags come out of the same ``UrdfWriter`` code, so
the origins and geometries under test are exactly the ones the rviz file
carries.

MuJoCo fuses welded links into their parent when it imports a URDF, so bodies
cannot be compared one to one.  Geoms can: each MJCF geom has to appear in the
URDF model with the same bounding radius and the same world pose, at the zero
pose and at random joint poses.  Masses are compared as a total, for the same
reason.
"""

from __future__ import annotations

import math
import os
import tempfile
from typing import Dict, List, Sequence, Tuple

import numpy as np

import mujoco

from tools.convert_arm_urdf import ASSETS_DIR, ROOT_BODY, CellModel, UrdfWriter


def _mat_angle(a: np.ndarray, b: np.ndarray) -> float:
    """Angle (rad) between two rotation matrices."""
    cos = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return float(math.acos(max(-1.0, min(1.0, cos))))


def _qpos_index(model) -> Dict[str, int]:
    """``joint name -> qpos address``, so one joint vector drives both models."""
    return {
        name: int(model.jnt_qposadr[joint])
        for joint in range(model.njnt)
        if (name := mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint))
    }


def _sample_joints(model, names: Sequence[str], rng) -> Dict[str, float]:
    """One random value per joint, inside its limits."""
    values = {}
    for name in names:
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        lo, hi = (float(v) for v in model.jnt_range[joint])
        if not model.jnt_limited[joint]:
            lo, hi = -math.pi, math.pi
        values[name] = float(rng.uniform(lo, hi))
    return values


def _match_geoms(mjcf, mjcf_data, urdf, urdf_data, geoms) -> Tuple[int, float, float, List[str]]:
    """Greedy-match every MJCF geom to a URDF one by bounding radius.

    The bounding radius is a property of the geometry in its own frame, so it
    identifies the same piece in both models whatever pose they are in.
    Returns ``(matched, worst position error, worst angle error, missing)``.
    """
    available = [
        (
            geom,
            urdf_data.geom_xpos[geom].copy(),
            urdf_data.geom_xmat[geom].reshape(3, 3).copy(),
            float(urdf.geom_rbound[geom]),
        )
        for geom in range(urdf.ngeom)
    ]
    used: set = set()
    worst_pos = worst_ang = 0.0
    missing: List[str] = []
    for geom in geoms:
        radius = float(mjcf.geom_rbound[geom])
        best, gap = None, np.inf
        for candidate in available:
            if candidate[0] in used:
                continue
            delta = abs(candidate[3] - radius)
            if delta < gap:
                best, gap = candidate, delta
        if best is None or gap > 1e-6:
            missing.append(
                mujoco.mj_id2name(mjcf, mujoco.mjtObj.mjOBJ_GEOM, geom) or f"geom{geom}"
            )
            continue
        used.add(best[0])
        worst_pos = max(
            worst_pos, float(np.linalg.norm(best[1] - mjcf_data.geom_xpos[geom]))
        )
        worst_ang = max(
            worst_ang,
            _mat_angle(mjcf_data.geom_xmat[geom].reshape(3, 3), best[2]),
        )
    return len(geoms) - len(missing), worst_pos, worst_ang, missing


def check(cell: CellModel, writer_kwargs: dict, poses: int = 4) -> bool:
    """Report whether the URDF these kwargs describe matches ``cell``'s model."""
    ok = True

    def report(good: bool, text: str) -> None:
        nonlocal ok
        ok = ok and bool(good)
        print(f"  {'PASS' if good else 'FAIL'}  {text}")

    kwargs = dict(writer_kwargs)
    kwargs.update(mesh_uri="relative", urdf_dir=ASSETS_DIR, collision=True)
    handle, probe = tempfile.mkstemp(prefix=".urdf_check_", suffix=".urdf", dir=ASSETS_DIR)
    try:
        with os.fdopen(handle, "w") as fh:
            fh.write(UrdfWriter(cell, **kwargs).xml())
        try:
            urdf = mujoco.MjModel.from_xml_path(probe)
        except Exception as exc:  # pragma: no cover - only on a broken export
            report(False, f"MuJoCo cannot load the exported URDF: {exc}")
            return False

        mjcf = cell.model
        hand = set(cell.hand_bodies())
        # --hand skip drops the fingers on purpose, so leave them out of the
        # expectation instead of reporting them as lost.
        skip_hand = kwargs["hand"] == "skip"
        robot = set(cell.robot_bodies()) - (hand if skip_hand else set())
        geoms = [g for body in robot for g in cell.geoms_of(body)]

        mjcf_qpos, urdf_qpos = _qpos_index(mjcf), _qpos_index(urdf)
        shared = [name for name in mjcf_qpos if name in urdf_qpos]
        report(
            all(f"joint{i}" in shared for i in range(1, 8)),
            f"{len(shared)} joints shared with the URDF; all 7 arm joints present",
        )
        report(
            all(name in mjcf_qpos for name in urdf_qpos),
            "every URDF joint exists in the MJCF too",
        )

        mjcf_data, urdf_data = mujoco.MjData(mjcf), mujoco.MjData(urdf)
        rng = np.random.default_rng(0)
        for step in range(poses + 1):
            values = (
                {name: 0.0 for name in shared}
                if not step
                else _sample_joints(mjcf, shared, rng)
            )
            # mj_resetData, not qpos[:] = 0: the MJCF's free joints need a unit
            # quaternion, and every joint starts at its own qpos0.
            mujoco.mj_resetData(mjcf, mjcf_data)
            mujoco.mj_resetData(urdf, urdf_data)
            for name, value in values.items():
                mjcf_data.qpos[mjcf_qpos[name]] = value
                urdf_data.qpos[urdf_qpos[name]] = value
            mujoco.mj_forward(mjcf, mjcf_data)
            mujoco.mj_forward(urdf, urdf_data)

            matched, worst_pos, worst_ang, missing = _match_geoms(
                mjcf, mjcf_data, urdf, urdf_data, geoms
            )
            label = "zero pose" if not step else f"random pose {step}"
            report(
                not missing and worst_pos < 1e-6 and worst_ang < 1e-6,
                f"{label}: {matched}/{len(geoms)} geoms matched, "
                f"max |xyz| {worst_pos:.2e} m, max angle {worst_ang:.2e} rad"
                + (f", missing {missing[:4]}" if missing else ""),
            )

        # Masses are compared as a total: the fusing above moves mass between
        # bodies, but the moving part of the robot has to weigh the same either
        # way -- whether the hand is 22 links or one baked one.
        mjcf_mass = sum(
            float(mjcf.body_mass[body])
            for body in cell.robot_bodies()
            if cell.body_name(body) != ROOT_BODY and not (skip_hand and body in hand)
        )
        urdf_mass = sum(float(urdf.body_mass[body]) for body in range(1, urdf.nbody))
        report(
            abs(mjcf_mass - urdf_mass) < 1e-9,
            f"mass of the moving robot: MJCF {mjcf_mass:.6f} kg, URDF {urdf_mass:.6f} kg",
        )
        return ok
    finally:
        os.remove(probe)
