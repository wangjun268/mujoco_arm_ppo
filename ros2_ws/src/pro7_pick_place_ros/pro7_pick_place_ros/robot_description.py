"""Find the cell's URDF for rviz, and keep its mesh paths honest.

rviz's RobotModel display draws geometry from a URDF, so
``tools/convert_arm_urdf.py`` exports one from the compiled MJCF and the node
republishes it on ``/robot_description`` (see
:mod:`pro7_pick_place_ros.node`).  This module is the file side of that: read
the URDF, and repoint ``file://`` mesh URIs that refer to a checkout that has
moved since it was generated.  A URDF whose meshes cannot be found draws
nothing at all, so a mesh that cannot be resolved either way is reported rather
than silently left out.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import List, Tuple

#: ``filename="file:///..."`` of every mesh the URDF references.
_MESH_URI = re.compile(r'filename="(file://[^"]+)"')
#: The mesh URIs point inside the checkout's ``assets/``; this is what marks the
#: part of the path that survives a move.
_ASSETS_MARKER = "/assets/"


def path_for(model_path: str, explicit: str = "") -> str:
    """The URDF to read: ``explicit`` if given, else ``<MuJoCo model>.urdf``.

    That is where ``tools/convert_arm_urdf.py`` writes it, next to the model.
    """
    explicit = explicit.strip()
    if explicit:
        return os.path.abspath(os.path.expanduser(explicit))
    return os.path.splitext(os.path.abspath(model_path))[0] + ".urdf"


def counts(text: str) -> Tuple[int, int]:
    """``(links, joints)`` of an URDF, for the node's startup log."""
    root = ET.fromstring(text.encode("utf-8"))
    return len(root.findall("link")), len(root.findall("joint"))


def read(path: str, assets_dir: str) -> Tuple[str, List[str]]:
    """URDF text at ``path`` plus notes about the mesh paths inside it.

    Raises :class:`OSError` when the file cannot be read.  A mesh that is
    missing *and* cannot be found under ``assets_dir`` is reported, not fixed:
    guessing a path would hide a URDF that no longer matches the model.
    """
    with open(path) as fh:
        text = fh.read()

    repaired: List[str] = []
    missing: List[str] = []

    def replace(match: "re.Match") -> str:
        uri = match.group(1)
        original = uri.removeprefix("file://")
        if os.path.isfile(original):
            return match.group(0)
        # The URDF was generated elsewhere (or the checkout moved): keep the
        # part below ``assets/`` and look it up under *this* checkout.
        tail = original.split(_ASSETS_MARKER, 1)[-1]
        candidate = os.path.join(assets_dir, tail)
        if os.path.isfile(candidate):
            repaired.append(candidate)
            return f'filename="file://{candidate}"'
        missing.append(original)
        return match.group(0)

    text = _MESH_URI.sub(replace, text)
    notes: List[str] = []
    if repaired:
        notes.append(
            f"repointed {len(repaired)} mesh path(s) at {assets_dir} "
            "(the URDF was generated for another checkout)"
        )
    if missing:
        notes.append(
            f"{len(missing)} mesh file(s) of the robot description are missing, "
            f"rviz will draw nothing for them (first: {missing[0]})"
        )
    return text, notes
