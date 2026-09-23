"""Generate the Pro7 wrist -> LinkerHand L20 adapter as an STL.

The cell mounts the hand 60 mm in front of the Pro7's last flange face
(``<body name="gripper" pos="0 0 0.06">``), which is where the real adapter
sits, so the space between them was empty and the arm and hand looked
separated -- in MuJoCo and in rviz alike.

``PROFILE`` below is a surface of revolution sized from the two faces it joins,
measured in the ``link7`` frame: the wrist face at ``z = 0`` (an annulus, outer
radius 30.0 mm) and the hand base's mounting face at ``z = 0.06`` (a ring,
radius 20.0..32.0 mm).  Every scene mounts the result as a **visual-only,
massless** geom (``contype="0"``, ``density="0"``), so the calibrated grasp, the
reach task and the eye-in-hand view are untouched.

Run::

    python3 tools/make_wrist_flange.py            # -> assets/meshes/pro7_l20_flange.stl
    python3 tools/make_wrist_flange.py --segments 128 --out /tmp/flange.stl
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from typing import List, Sequence, Tuple

import numpy as np

# Allow `python3 tools/make_wrist_flange.py` as well as `python3 -m tools.make_wrist_flange`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import asset_path

#: ``(radius, z)`` from the wrist face up to the hand mount, in metres.  Plate,
#: relieved waist, plate -- with chamfers, so it reads as a machined part.
PROFILE: Tuple[Tuple[float, float], ...] = (
    (0.0300, 0.0000),  # seats on the wrist flange face
    (0.0300, 0.0070),  # 7 mm mounting plate
    (0.0280, 0.0085),  # chamfer off the plate
    (0.0260, 0.0100),
    (0.0260, 0.0250),  # 26 mm waist
    (0.0235, 0.0265),  # shallow relief groove
    (0.0235, 0.0295),
    (0.0260, 0.0310),
    (0.0260, 0.0450),
    (0.0295, 0.0470),  # chamfer up to the hand plate
    (0.0295, 0.0585),
    (0.0280, 0.0600),  # top chamfer, face for the hand base
)

#: Name written into the STL header, and the file it lands in.
PART_NAME = "pro7_l20_wrist_flange"
DEFAULT_OUT = "meshes/pro7_l20_flange.stl"
#: Facets around the axis; 96 keeps the 30 mm rim smooth at rviz zoom.
SEGMENTS = 96


def revolve(profile: Sequence[Tuple[float, float]], segments: int = SEGMENTS) -> np.ndarray:
    """Turn a ``(r, z)`` outline into a closed triangle soup.

    ``profile`` runs bottom to top with ``r > 0``; both ends are capped by a fan
    to the axis, so the solid is watertight and every triangle is wound so its
    normal points outwards.
    """
    angle = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    unit = np.stack([np.cos(angle), np.sin(angle)], axis=1)
    rings = [np.column_stack([r * unit, np.full(segments, z)]) for r, z in profile]

    triangles: List[np.ndarray] = []
    for lower, upper in zip(rings, rings[1:]):
        for k in range(segments):
            n = (k + 1) % segments
            triangles.append(np.stack([lower[k], upper[n], upper[k]]))
            triangles.append(np.stack([lower[k], lower[n], upper[n]]))

    bottom = np.array([0.0, 0.0, profile[0][1]])
    top = np.array([0.0, 0.0, profile[-1][1]])
    for k in range(segments):
        n = (k + 1) % segments
        triangles.append(np.stack([bottom, rings[0][n], rings[0][k]]))  # faces -z
        triangles.append(np.stack([top, rings[-1][k], rings[-1][n]]))  # faces +z
    return np.asarray(triangles)


def write_stl(triangles: np.ndarray, path: str, name: str = PART_NAME) -> None:
    """Binary STL, the format the arm's vendor meshes use."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    normals = np.divide(
        normals, lengths[:, None], out=np.zeros_like(normals), where=lengths[:, None] > 0
    )
    header = f"{name} (surface of revolution, {len(triangles)} facets)".encode()
    with open(path, "wb") as fh:
        fh.write(header[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(triangles)))
        for normal, corners in zip(normals, triangles):
            fh.write(struct.pack("<12fH", *normal, *corners[0], *corners[1], *corners[2], 0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default=asset_path(DEFAULT_OUT),
                    help="output STL (default: assets/%s)" % DEFAULT_OUT)
    ap.add_argument("--segments", type=int, default=SEGMENTS,
                    help="facets around the axis")
    args = ap.parse_args(argv)

    triangles = revolve(PROFILE, args.segments)
    write_stl(triangles, args.out)

    vertices = triangles.reshape(-1, 3)
    # Divergence theorem: a turned part is easy to get subtly wrong, and a
    # negative volume means the winding is inverted (it would render inside-out).
    volume = float(
        np.einsum("ij,ij->i", triangles[:, 0], np.cross(triangles[:, 1], triangles[:, 2])).sum()
        / 6.0
    )
    print(
        f"wrote {args.out}\n"
        f"  profile  z {PROFILE[0][1] * 1e3:.1f} -> {PROFILE[-1][1] * 1e3:.1f} mm, "
        f"r {min(r for r, _ in PROFILE) * 1e3:.1f} .. {max(r for r, _ in PROFILE) * 1e3:.1f} mm\n"
        f"  facets   {len(triangles)} ({args.segments} around the axis)\n"
        f"  bounds   {np.round(vertices.min(0), 4)} .. {np.round(vertices.max(0), 4)}\n"
        f"  volume   {volume * 1e6:.1f} cm^3 "
        f"(~{volume * 2700 * 1e3:.0f} g if it were solid aluminium)\n"
        "  mates    link7 flange face r 15.7..30.0 mm @ z 0\n"
        "           hand base ring    r 20.0..32.0 mm @ z 60 mm"
    )
    if volume <= 0:
        raise SystemExit(f"{args.out} is inside-out: check the winding")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
